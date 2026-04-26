---
name: FA4 hd256 backward — kernel optimization opportunities (ncu)
description: Bottleneck analysis from ncu profiling of dq_kernel and dkdv_kernel on B200. Identifies tensor-pipe-bound + L1TEX-stall pattern with two specific source-line hotspots for "uncoalesced global access".
type: project
originSessionId: 5ce87721-bf30-40aa-8632-d0b628cc1cc5
---
ncu profiling of FA4 hd256 backward (`pdl-hd256-bwd` branch @ 4819cf2, B200 base clocks) on 2026-04-25 surfaced a concrete optimization opportunity that the dual-stream experiment did not address.

**Why:** Identify what's leaving perf on the table beyond the dual-stream and PDL changes. Result: kernels are tensor-pipe (UMMA + TMEM) bound, NOT DRAM-bound; the lever is reducing L1TEX-dependency stalls via better epilogue store layout and a strided-load fix.

**How to apply:** When optimizing FA4 hd256 backward kernels (or analogous SM100 attention backwards), prioritize the following two source-level hotspots over occupancy/DRAM/dual-stream concerns. Local artifacts at `/workspace/flash-attention/agent_space/ncu_hd256/` (REPORT.md, repro.py, profile.sh, per_line.py — Python script that aggregates ncu metric to source line via `ncu_report` API). Raw reports at `/tmp/ncu/{A,B}_{dq,dkdv}{,_lineinfo}.ncu-rep`.

## Bottleneck classification (configs: hd=256, B=8, S=16384, MHA H=32, both causal modes)

- All four kernels: **tensor-pipe bound on Blackwell TMEM** (`UTCMMA/UTCCP/LDT(M)/STT(M)`) at 47–87%. DRAM <4% — kernels stream K/V/dO through L1/L2, not HBM.
- Top stall on 3 of 4 kernels: **L1TEX scoreboard dependency** — 57–60% of warp cycles for dq, 31% for A_dkdv. ncu est. speedup if removed: 13–28%.
- `B_dkdv` (causal dkdv) is already healthy: IPC 1.10, no dominant stall — leave alone.
- Achieved occupancy 18.7% pinned at theoretical max (`Block Limit Registers=1`, `Block Limit Shared Mem=1`). Increasing occupancy is not the lever.

## Source-line hotspots for "uncoalesced global access"

ncu's headline `Est. Speedup: 41–89%` is theoretical (assumes excess sectors = DRAM cost, but L1 hit rate is 97%). Realistic ceiling is the L1TEX-stall fraction → 5–15% wallclock if both fixes land.

**(1) Epilogue store layout — 71–96% of all excess.** TMEM→register→GMEM store path. MMA accumulator in TMEM has each thread holding strided slices of many output rows; with hd=256 (16 sectors per output row), each warp's 32 stores hit 32 different rows → 32× sector traffic vs coalesced.
- `flash_attn/cute/sm100_hd256_2cta_fmha_backward_dkdvkernel.py:2694` — `gmem_i.store(regs_i.load().to(gmem.element_type))` in `BlackwellFusedMultiHeadAttentionBackwardDKDVKernel.store()`
- `flash_attn/cute/sm100_hd256_2cta_fmha_backward_dqkernel.py:2148` — `cute.autovec_copy(tSMrdQ, tTMEM_LOADgdQ_i)` in dq epilogue
- Standard fix: TMEM → SMEM (via UTCSHIFT/UTCCP) → GMEM (via TMA bulk store). Bigger refactor but it's the bigger win.

**(2) Strided per-element scalar loads of LSE / dpsum — 8–14% each.** Loop indexing `LSE_idx = tile_Q*iter + thread_idx*N` with inner loop `for i in range(N): copy at LSE_idx + i` makes a single inner iteration access stride-N across the warp. For `N>1` (typical N=4) this gives `(N-1)/N` = 75% sector excess on these loads.
- `flash_attn/cute/sm100_hd256_2cta_fmha_backward_dkdvkernel.py:1922` — LSE per-K-iter load
- `flash_attn/cute/sm100_hd256_2cta_fmha_backward_dkdvkernel.py:1953` — sum_OdO per-K-iter load
- Smaller earlier-iteration variants at lines 1827, 1866 (0.1% each)
- Fix: swap loop nesting to `for i: idx = tile_Q*iter + i*32 + thread_idx` (warp axis innermost). One-file, ≤20-line change. Smaller share of the win but easy to validate.

## Notes on the methodology

- Need `CUTE_DSL_LINEINFO=1` + `CUTE_DSL_KEEP_PTX=1` set when launching ncu so that the cute compile emits `.file`/`.loc` directives in PTX, then `ncu --import-source on`. With those, ncu's report has source lines correlated to SASS.
- ncu `--print-source` only shows SASS-level metrics in CLI text; for source-line aggregation use the `ncu_report` Python API at `/opt/nvidia/nsight-compute/2025.4.1/extras/python/ncu_report.py`. The local `agent_space/ncu_hd256/per_line.py` does this aggregation.
- ncu uses base SM clocks (~1.07 GHz on B200) for stable measurement; absolute durations don't compare to wallclock bench, but ratios and percentages do.

## Design sketch for the epilogue-store fix (TMEM→SMEM→TMA)

Detailed plan at `/workspace/flash-attention/agent_space/ncu_hd256/EPILOGUE_REFACTOR_DESIGN.md`. Key load-bearing facts for picking it up:

### SMEM audit completed 2026-04-25 (results captured in design doc)

- B200 SM dynamic-smem ceiling = 233.47 KB; current dkdv+dq each use 230.8 KB → **only ~1.67 KB headroom**, cannot add new regions, MUST alias.
- **dkdv**: `sP` (16 KB) + `sdST` (16 KB) confirmed dead at epilogue → 32 KB available for staging. Need only ≤2 KB per chunk → fits with huge slack. Decision: stage one full output tile (64×256 bf16 = 32 KB) per TMA store, exactly fits sP+sdST combined.
- **dq**: No sP/sdST equivalents (P/dS in TMEM only). All 5 load buffers (sQ 64 KB, sK 32 KB, sV 32 KB, sdO 64 KB, sKT 32 KB) dead at epilogue → alias staging onto sQ or sdO. Mirror `flash_fwd_sm100.py:660-664` `overlap_sO_sQ` pattern (sO_size=0, sQ resized via `cutlass.max(cosize_sQ, cosize_sO * dtype_ratio)`).
- Audit method: temporary `print` of `cute.cosize(layout) * dtype.width // 8` for each smem layout, before the SharedStorage @cute.struct definition. Removed after audit.

### Implementation progress on dkdv (working tree, uncommitted, +95 lines, all 8/8 correctness, lint clean)

**Done (5a + 5b plumbing):**
- TMA store atoms `tma_atom_dK`, `tma_atom_dV` built in `__call__` via `cpasync.make_tiled_tma_atom` (mirrors `flash_fwd_sm100.py:603-605`).
- `epi_tile_dKV = (cta_tiler[1]=64, cta_tiler[2]=256)` for full-tile staging; layouts `sdK_epi_layout` / `sdV_epi_layout` built via `sm100_utils.make_smem_layout_epi` with `LayoutEnum.from_tensor(dK/dV)`.
- All threaded through full call chain: `__call__` → `dkdv_bwd` → `compute` (both call sites) → `epilogue`. Args added: `sdK_epi_layout`, `sdV_epi_layout`, `tma_atom_dK`, `tma_atom_dV`, `dK_tma`, `dV_tma`. Plus `sP` for SMEM staging address.
- Staging tensor views `s_epi_dK` / `s_epi_dV` built inside `epilogue` via `cute.make_tensor(cute.recast_ptr(sP.iterator, sdK_epi_layout.inner, dK.element_type), sdK_epi_layout.outer)` — mirrors `flash_fwd_sm100.py:1002`. The 32 KB view spans sP (16 KB) + sdST (16 KB), contiguous in SharedStorage.
- All currently UNUSED — store path is still the broken per-thread `self.store(...)`. Behavior identical to HEAD by construction.

**Gotcha learned:** Layout objects cannot be captured via `self.attr` from inside a `@cute.jit` region — fails IR verification with "using value defined outside the region". Must pass through the kernel call signature as args (mirror `flash_fwd_sm100.py:749/796` where `sO_layout` is a kernel arg).

**Remaining for variant 3a (the actual perf win — step 5d):**
- Replace `self.store(tTR_gdV/dK, tTR_rdV/dK, tTR_cdV/dK, (K, D))` at lines 2861/2873 (post-shift; was 2834/2847) with TMEM→reg→SMEM→TMA path.
- Open API question: `flash_fwd_sm100.py:2740` references `copy_utils.partition_D_position_independent(thr_tmem_load, tOsO_i[(None, None), None])` for the SMEM partition setup, but that function does NOT exist in the local `flash_attn/cute/copy_utils.py`. Either fwd is broken (unlikely — we've verified the module works) OR the function is imported from elsewhere / renamed. Need to investigate before writing the store path.
- Other helpers needed: `sm100_utils.get_smem_store_op(layout_enum, dtype, acc_dtype, tiled_tmem_load)` for the smem copy atom, `cute.make_tiled_copy_D` to wrap it.
- Validation gates: correctness 8/8 + ncu re-profile (excessive sectors on the new line should drop to <5%) + rep=50 bench targeting +5–15% on dkdv.

**Recommended next chunk:** Resolve the `partition_D_position_independent` mystery first.

**Investigation 2026-04-26:** Function is NOT in `flash_attn/cute/copy_utils.py` (verified via `python -c "from flash_attn.cute import copy_utils; print(hasattr(copy_utils, 'partition_D_position_independent'))"` → False). NOT in `/workspace/cutlass`, NOT in `/usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl`. Yet `flash_fwd_sm100.py:2740` calls it inside `correction_epilogue`, which IS reachable from the standard fwd path (line 2505). And the fwd kernel DOES compile and run (verified via repro.py multiple times this session).

Hypothesis: `correction_epilogue` may not actually be entered at runtime for our config — `@cute.jit` lazily compiles only the branches that fire, so a missing reference in dead code can survive. Worth verifying: add a `cute.printf` at the top of `correction_epilogue` and run trigger.py to see if it prints. If not, the reference is dead and we need to write our own SMEM partition idiom (likely `thr_tmem_load.partition_D(s_epi_tile)` + manual layout fixup, since `partition_D` is the standard cute API).

Alternative path forward (skip the mystery): use `thr_tmem_load.partition_D(s_epi_dV)` directly and see if it produces a valid SMEM store partition. cute may support this natively without needing `_position_independent`. The "position independent" wrinkle is for cases where threads at different positions need symmetric partitions — for a one-shot epilogue store this may not matter.

**Where to pick up:** Working tree has +95 lines of plumbing in `dkdvkernel.py`, all 8/8 correctness, lint clean, no behavior change. Atoms + tensors + epi-layouts all threaded through `__call__` → `dkdv_bwd` → `compute` → `epilogue`. Staging tensor views `s_epi_dK`/`s_epi_dV` built inside epilogue ready to receive writes. The `self.store(...)` calls at the original lines 2834/2847 (now shifted due to additions) are still the per-thread broken stores. The next session can either (a) write a `partition_D` based store path and test, or (b) instrument fwd to confirm the mystery line is dead before proceeding.

- The forward kernel already does this pattern. Mirror `flash_fwd_sm100.py` `correction_epilogue` (TMEM→reg→SMEM via `cvt_copy(tiled_smem_store, ...)` with `sm100_utils_basic.get_smem_store_op`) + `epilogue_s2g` (SMEM→GMEM via `cpasync.CopyBulkTensorTileS2GOp`, `tma_atom_O`). Reference epi tile: `epi_tile = (m_block_size, corr_tile_size)` where `corr_tile_size = 8*32 // dtype.width` (≈16 for bf16).
- **Recommended first cut is variant 3a** — keep the existing compute warps, insert SMEM staging inline (no new pipeline, no warp split). Variant 3b (separate correction/epilogue warps with a producer/consumer pipeline) is only if 3a leaves headroom.
- Concrete edit points: `dqkernel.py:2148` (replace `cute.autovec_copy(tSMrdQ, tTMEM_LOADgdQ_i)`) and `dkdvkernel.py:2694` (replace `gmem_i.store(...)` in the per-helper `store()` method). New TMA atoms `tma_atom_dQ/dK/dV` need to be built in the kernel setup mirroring fwd's `tma_atom_O`.
- **Hard risk: SMEM budget.** ncu reports `Block Limit Shared Mem = 1` — kernel is at the ceiling. Three escape hatches in preference order: (1) reuse `sP` / `sdST` regions (dead by epilogue time, mirrors fwd's `sQ`/`sK`↔`sO` aliasing); (2) drop a K/V/dO pipeline stage (~64 KB freed but may regress the very L1TEX-stall we're trying to fix); (3) halve `corr_tile_size`.
- Implementation order: audit SMEM budget first (compile-time print of `cute.size_in_bytes(SharedStorage)`), pick a SMEM-recovery option, build TMA atoms, implement in dkdv first (smaller change, easier A/B), validate via `pytest tests/cute/test_flash_attn.py -k "headdim_256 and not fwd"` + ncu re-profile (excessive sectors on lines 2694/2148 should drop to <5%) + rep=50 bench targeting +5–15% wallclock on the affected kernels.

## Companion small fix (validated 2026-04-25, deferred and stashed)

Loop reorder for the strided LSE/dpsum loads at `dkdvkernel.py:1922/:1953` (and early-iter `:1827/:1866`). Pattern `LSE_idx = tile_Q*iter + thread_idx*N; for i in range(N): copy at LSE_idx + i` was a stride-N warp access; fixed to `for i: idx = tile_Q*iter + i*W + thread_idx; smem_idx = thread_idx + i*W` where `W = self.threads_per_warp = 32`.

**Validation results:**
- Correctness 8/8 vs PyTorch reference (MHA + GQA, causal + non-causal, seqlens 256–4096)
- ncu: excessive sectors 2.82 B → 2.01 B (−28.6%); the lines vanish from the hotspot list
- ncu SOL/stall: Compute throughput 72.09 → 72.14% flat, L1TEX-stall 5.0 → 5.1 cycles flat
- Wallclock A/B (rep=50, B200, clock-locked): median Δ ≈ −0.4%, all rows in this branch's known run-to-run noise band (±50 TFLOPS at MHA-causal-32K)

**Lives at `stash@{0}` on `pdl-hd256-bwd` branch.** Not discarded because (a) it's the correct access pattern, (b) saves L1 bandwidth, (c) becomes a hard prerequisite if anyone migrates these loads to TMA bulk (which requires coalesced access). Pop and apply alongside the TMEM→SMEM→TMA epilogue refactor — that fix reduces L1TEX-stall fraction enough that the sector savings start translating to wallclock.
