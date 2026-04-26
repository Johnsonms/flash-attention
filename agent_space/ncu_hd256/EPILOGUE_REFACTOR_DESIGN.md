# Design sketch: TMEM→SMEM→TMA epilogue for hd256 backward dq/dkdv

**Status:** design draft, no code changes yet. Audience: returning author.
**Targets:**
- `flash_attn/cute/sm100_hd256_2cta_fmha_backward_dqkernel.py:2148` —
  `cute.autovec_copy(tSMrdQ, tTMEM_LOADgdQ_i)` in the dQ epilogue
- `flash_attn/cute/sm100_hd256_2cta_fmha_backward_dkdvkernel.py:2694` —
  `gmem_i.store(regs_i.load().to(gmem.element_type))` in
  `BlackwellFusedMultiHeadAttentionBackwardDKDVKernel.store()`

These two lines together carry **71–96% of the "excessive global sectors"**
ncu flagged for the hd256 bwd kernels (see `REPORT.md`). The realistic
ceiling — bounded by the L1TEX-stall fraction, since L1 hit rate is 97% —
is **5–15% wallclock improvement** on the affected kernels.

---

## 1. Why the current path is wasteful

Both kernels compute their output in TMEM via `tcgen05.umma`, then run an
epilogue that does:

```
TMEM ─tcgen05.copy.Ld32x32bOp(Repetition(32))─► register fragment (fp32)
register fragment ─.to(bf16)──────────────────► register fragment (bf16)
register fragment ─direct gmem.store / autovec_copy──► GMEM (per-thread)
```

The `Ld32x32b` MMA accumulator layout has each thread holding strided slices
of many output rows. With hd=256 (16 sectors/row) the warp's 32 stores hit
32 different rows — i.e., a per-warp store generates **32× the sector
traffic** of a coalesced 128-byte write. Mathematically: ncu's
`l2_theoretical_sectors_global_excessive / l2_theoretical_sectors_global`
≈ 31/32 = 97%, matching the 95.5% (dq) / 71–83% (dkdv) we measured.

The fix is to add a SMEM staging step where the layout is reorganized so
the GMEM-side store can be a coalesced (or, better, TMA bulk) write.

## 2. The pattern already exists in the forward kernel

`flash_fwd_sm100.py` does this for the O output:

```
correction_epilogue (corr warps):
    TMEM ──tcgen05.copy.Ld32x32bOp(Repetition(corr_tile_size))──► reg frag
    reg frag ──cvt_copy(tiled_smem_store, ...)─────────────────► SMEM (sO)
                                ▲
                       smem_copy_atom built via
                       sm100_utils_basic.get_smem_store_op
                       — picks a swizzle that matches Ld32x32b
    fence_view_async_shared

epilogue_s2g (epi warps):
    consumer_wait on pipeline_o_epi
    SMEM (sO) ──cpasync.CopyBulkTensorTileS2GOp(tma_atom_O)──► GMEM
    cp_async_bulk_commit_group / wait_group
    consumer_release on pipeline_o_epi
```

Two warp groups; producer/consumer pipeline (`pipeline_o_epi`) coordinates
TMEM→SMEM (correction warps) and SMEM→GMEM (epilogue warps). The SMEM
staging tile is small (`epi_tile = (m_block_size, corr_tile_size)` where
`corr_tile_size = 8*32//bf16.width = 16`) — only ~2 KB per stage — so
SMEM cost is modest.

## 3. Target flow for the bwd kernels

Mirror the fwd pattern. Two design variants:

### 3a. Minimal variant — same warps, in-place SMEM staging

Keep the existing compute warps; insert a SMEM staging step before the
GMEM write. No new pipeline.

```
existing TMEM load ──────────────────────► reg frag (fp32)
quantize fp32 → bf16 ───────────────────► reg frag (bf16)
NEW: cvt_copy(tiled_smem_store, ...) ───► SMEM (sdQ / sdK / sdV)
NEW: cute.arch.fence_view_async_shared
NEW: cute.arch.barrier(across the warpgroup that did the load)
NEW: tma_store via cpasync.CopyBulkTensorTileS2GOp ─► GMEM
NEW: cp_async_bulk_commit_group / wait_group
```

Pros: small change, no pipeline/warp restructuring, no register-pressure
shift. Cons: no overlap between TMEM→SMEM and SMEM→GMEM phases — both
compete for the same compute warps' time. For the dq epilogue this is
fine because dq runs only at the *end* of the per-block loop; for dkdv
it's also fine since the epilogue only fires once per K-block.

This is the **recommended first cut**.

### 3b. Pipelined variant — split warps (mirrors fwd exactly)

Designate `(num_epilogue_warps,)` from the existing compute warps as
"epilogue warps" plus a `pipeline_o_epi` between TMEM→SMEM and
SMEM→GMEM. Better latency hiding when the epilogue is on the critical
path.

Skip for v1 unless 3a turns out to leave significant headroom.

## 4. Concrete changes per file

### `sm100_hd256_2cta_fmha_backward_dqkernel.py`

**Setup phase (constructor / `_compute_setup`):**
- Build TMA store atom for `dQ` analogous to fwd's `tma_atom_O`:
  ```python
  tma_atom_dQ, tma_tensor_dQ = cpasync.make_tiled_tma_atom_with_copy_register_tensor(
      cpasync.CopyBulkTensorTileS2GOp(),
      dQ,                                  # the dQ tensor
      cute.select(sdQ_layout, mode=[0, 1]),
      epi_tile_dQ,
  )
  ```
- Define `epi_tile_dQ = (self.tile_shape_Q, corr_tile_K)` where
  `corr_tile_K = 8 * 32 // self.q_dtype.width` (≈16 for bf16; matches
  fwd convention so the existing `get_smem_store_op` swizzle works).
- Add `sdQ` to the `SharedStorage` dataclass:
  ```python
  sdQ: cute.struct.Align[
      cute.struct.MemRange[Q.element_type, cute.cosize(sdQ_layout_staged)],
      self.buffer_align_bytes,
  ]
  ```
  with `sdQ_layout_staged` built from `sm100_utils.make_smem_layout_…`
  using the same atom kind as the existing Q/K smem layouts. Stages = 1
  (no pipeline in 3a) or 2 (3b).

**Epilogue body (lines 2120–2148, `dq_epi` or wherever):**
- Replace the inner `cute.autovec_copy(tSMrdQ, tTMEM_LOADgdQ_i)` with
  `cvt_copy(tiled_smem_store, tSMrdQ, tSMEM_STOREsdQ_i)` using a
  `tiled_smem_store` built from `get_smem_store_op` (matching the
  Ld32x32b layout) and the new `sdQ` smem buffer.
- After the per-tile inner loop completes, `fence_view_async_shared` +
  `cute.arch.barrier(epilogue_sync_bar_id, …)` to synchronize the warp
  group.
- Issue TMA store of `sdQ → gdQ` via the new `tma_atom_dQ` and
  `cp_async_bulk_commit_group` / `cp_async_bulk_wait_group`.

### `sm100_hd256_2cta_fmha_backward_dkdvkernel.py`

Same pattern as dq, twice (one for dK, one for dV). The existing
`store(self, gmem, regs, coord, tensor_shape)` helper at line 2682 is
where the per-thread `gmem_i.store(...)` lives — it gets replaced by a
SMEM-staging version, e.g. a new helper:

```python
@cute.jit
def store_via_smem(self, gmem, regs, coord, tensor_shape, sBuf, tma_atom):
    # 1) reg → SMEM via cvt_copy(tiled_smem_store, regs, sBuf_partition)
    # 2) fence_view_async_shared + named barrier
    # 3) cp_async_bulk SMEM → GMEM via tma_atom
    # 4) cp_async_bulk_commit_group + wait_group
```

Two new SMEM regions: `sdK` and `sdV`. Sized ≤
`tile_shape_K × corr_tile_size × 2 bytes` per stage (for bf16) — for
`tile_shape_K=128`, `corr_tile_size=16`: 4 KB each.

The two existing call sites (line 2834: `self.store(tTR_gdV, ...)` and
line 2847: `self.store(tTR_gdK, ...)`) become calls to the new helper.

## 5. SMEM budget — the hard constraint

ncu reports `Block Limit Shared Mem = 1` on every bwd kernel. SMEM is at
the design ceiling. Adding sdQ / sdK / sdV unconditionally will likely
**push the kernel over the per-SM SMEM budget**.

Three ways out, in preference order:

**(a) Reuse already-dead SMEM regions.** By the time the epilogue runs:
- In dkdv: the sP / sdST scratch regions (lines 613, 617) are no
  longer needed. They are sized for the MMA matrix (~`P_smem_layout_staged`
  and `dST_smem_layout_staged`) and likely large enough to host a small
  `(tile_shape_K × 16)` bf16 staging tile.
- In dq: similarly inspect after the dSQ MMA chain finishes.
- This needs a `cute.recast_tensor` or carefully aliased SharedStorage
  layout. Mirror how the fwd kernel reuses `sQ`/`sK` regions for `sO`
  staging (it does this — see `flash_fwd_sm100.py` SharedStorage.)

**(b) Reduce a pipeline stage on K/V/dO loads.** The bwd kernels run
multiple in-flight stages of K/V/dO TMA loads. Dropping one stage frees
≈ tile-shape × hd × bf16 bytes (e.g. 128×256×2 = 64 KB for K). That's
plenty of headroom but reduces compute/memory overlap → could regress
the L1TEX-stall fraction we're trying to fix. Use only if (a) doesn't
work.

**(c) Smaller `epi_tile`.** Drop `corr_tile_size` from 16 → 8 (fp16
half-byte) or even 4. SMEM cost halves but TMA stores get smaller and
more numerous — tradeoff.

**Validation gate before writing code:** sum the SharedStorage byte
budget today vs proposed, confirm it stays ≤ the SM100 dynamic-smem
limit (228 KB on B200). Print
`cute.size_in_bytes(SharedStorage)` at compile time and check.

### Audit results (run 2026-04-25 via temporary print in setup)

B200 SM dynamic-smem ceiling = **233.47 KB**.

**dkdv kernel** (`cta_tiler=(128, 64, 256)`, all stages = 1):

| Region | Bytes | Role | Dead at epilogue? |
|--------|-------|------|---|
| sK     | 32,768 | TMA-loaded K | No (also reused for K^T view via sQT) |
| sV     | 32,768 | TMA-loaded V | No (last MMA fires in epilogue path) |
| sQ     | 32,768 | TMA-loaded Q | yes |
| sQT    | 32,768 | TMA-loaded Q^T | yes |
| sdO    | 32,768 | TMA-loaded dO | yes |
| sdOT   | 32,768 | TMA-loaded dO^T | yes |
| **sP** | **16,384** | P (=softmax(QK^T)) intermediate | **YES — last consumer is the dKdV MMA** |
| **sdST**| **16,384** | dS^T intermediate for dV side | **YES — last consumer is the dKdV MMA** |
| sLSE   | 512 | scalar LSE per Q row | yes |
| sSum_OdO | 512 | scalar dpsum per Q row | yes |
| **Sum** | **225,408** | | |

ncu reports total dynamic smem = 230.78 KB; the ~5.4 KB delta is mbarriers (13 × 16 bytes = 208 B) + alignment padding (1024-byte align on the large TMA buffers). Net headroom: **1.67 KB** — too tight to allocate any new region.

**dq kernel** (`cta_tiler=(128, 128, 256)`, all stages = 1):

| Region | Bytes | Role | Dead at epilogue? |
|--------|-------|------|---|
| sQ     | 65,536 | TMA-loaded Q | YES |
| sK     | 32,768 | TMA-loaded K | YES |
| sV     | 32,768 | TMA-loaded V | YES |
| sdO    | 65,536 | TMA-loaded dO | YES |
| sKT    | 32,768 | TMA-loaded K^T | YES |
| sLSE   | 512 | per-row | yes |
| sSum_OdO | 512 | per-row | yes |
| **Sum** | **230,400** | | |

dq has no sP / sdST equivalents (P/dS live entirely in TMEM). Headroom 1.79 KB. **All 5 load buffers are dead at epilogue** — alias staging onto any of them.

### Strategy refinement based on the audit

The design's option (1) — "reuse sP/sdST" — applies cleanly to **dkdv**, with 32 KB combined available exactly when the staging buffer is needed.

For **dq**, since there is no sP/sdST, alias staging onto **sQ or sdO** (both 64 KB, dead at epilogue). The fwd kernel already does this pattern (`overlap_sO_sQ` flag, `sO_size = 0` when overlapping, `sQ_size = max(...)` resizing) — see `flash_fwd_sm100.py:660-664`. Mirror that.

For **chunk size**, the fwd pattern of `epi_tile = (tile_M, corr_tile_size = 8*32 // dtype.width)` gives 16 elements for bf16. Per-chunk staging:

| Kernel | Output | Chunk shape | Bytes | Chunks per output |
|--------|--------|-------------|-------|---:|
| dkdv | dK / dV | (64, 16) bf16 | 2,048 | 16 |
| dq   | dQ      | (128, 16) bf16 | 4,096 | 16 |

Both fit inside the dead regions with substantial slack — no need to drop pipeline stages or shrink corr_tile.

## 6. Synchronization & ordering

**Variant 3a (recommended):**

```
[compute warps already finished MMA]
TMEM → reg frag                            (already exists)
reg frag → SMEM (NEW)
fence_view_async_shared                    (NEW)
named-barrier across the warps that wrote SMEM  (NEW)
arrive on tma_store_mbar                   (NEW, or use barrier)
cp_async_bulk SMEM → GMEM                  (NEW; only one warp issues)
cp_async_bulk_commit_group                 (NEW)
cp_async_bulk_wait_group(0, read=True)     (NEW; before SMEM is reused / kernel exits)
```

The `cp_async_bulk_wait_group(..., read=True)` is the gate before SMEM
can be reused. For the bwd kernels (single epilogue per K block) it can
sit just before kernel return / next K iteration.

**Existing barriers to update:**
- `self.epilogue_sync_bar_id = 3` exists in dkdv (line 169) and is
  already used for clear-paths. Reuse it for the new sync.
- For dq, no equivalent named barrier currently — add one (named
  barriers are cheap).

## 7. Risks and open questions

1. **`get_smem_store_op` requires the right atom kind for the dQ/dK/dV
   data type.** It works in fwd for fp16/bf16 O. Confirm it accepts the
   bwd's `q_dtype` (typically bf16) and the existing `Ld32x32b` repetition.
2. **The 2CTA cluster.** Both kernels run with `cta_group_size=2`. TMA
   bulk store from SMEM is per-CTA, so each CTA of the cluster stores
   its half. The `tma_atom_dQ` / `tma_atom_dK` / `tma_atom_dV` need to
   be built with the cluster shape in mind (mirror how `tma_atom_O` is
   built in fwd; it handles 2CTA).
3. **`autovec_copy` does an unaligned-safe predicated store.** The new
   path needs the same out-of-bounds protection at the SMEM→GMEM step.
   TMA bulk stores have built-in clipping when given OOB coordinates —
   confirm this matches the existing predicate semantics
   (`cute.elem_less(...)` checks).
4. **Quantize is currently inside the inner loop** (line 2146:
   `tSMrdQ.store(dq_vec.to(self.q_dtype))`). With SMEM staging, the
   quantize can happen at register-frag granularity (same as today) and
   the SMEM-side write goes via `cvt_copy` which does the conversion in
   the copy atom. No semantic change.
5. **dQ accumulator semantics.** dQ accumulates across N iterations
   *inside* TMEM (UMMA accumulate=True). The epilogue runs once at the
   end. So the SMEM staging buffer holds the *final* dQ for the M block
   and is written exactly once — no atomic-add concerns.
6. **Register pressure.** The new `cvt_copy` does not increase register
   usage (it consumes the same fragment that we already had). Good.

## 8. Validation plan

1. **SMEM budget check first** (compile-time print).
2. **Numerical correctness:** existing
   `pytest tests/cute/test_flash_attn.py -k "headdim_256 and not fwd"`
   covers the bwd. Run pre-/post- and diff the failing-tolerance set.
3. **ncu re-profile, same configs:**
   - Excessive sectors should drop sharply on lines 2694 and 2148. Aim
     for `<5%` of total — the residual from the LSE/dpsum stride loops
     (lines 1922/1953, the *other* identified hotspot).
   - L1TEX-dependency stall fraction should drop on dq from ~60% toward
     the dkdv healthy baseline (~30%).
4. **Bench:** rerun `agent_space/bench_pdl_delta.sh` style A/B (just the
   single-stream kernels, no dual-stream) at rep=50 on the same configs,
   target +5–15% wallclock on the affected kernels.
5. **Roofline check:** `ncu --set roofline` to confirm the kernel moves
   along the compute axis (toward higher TC throughput) rather than the
   memory axis.

## 9. Implementation order (if/when we proceed)

1. Audit SMEM budget (read SharedStorage layouts; compute a target for
   sdQ/sdK/sdV under each of the three smem-recovery options).
2. Decide on the SMEM source: prefer reuse of sP / sdST.
3. Build tma_atom_dQ / dK / dV in the kernel setup.
4. Implement variant 3a in dkdv first (smaller change; one helper to
   replace; easy to A/B).
5. Validate (correctness + ncu + bench).
6. If successful, port to dq.
7. If still significant L1TEX stall remains on dq, consider variant 3b.

---

This sketch is intentionally conservative — variant 3a is the smallest
change that resolves the headline finding, and SMEM-budget reuse is
where it can fail. Read 5 first.
