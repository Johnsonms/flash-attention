# dkdv variant 3a debug handoff (Item B → Codex)

**Branch:** `pdl-hd256-bwd`. Working tree has both dkdv and dq changes.
**File under test:** `flash_attn/cute/sm100_hd256_2cta_fmha_backward_dkdvkernel.py`
**Codex's dq plumbing is in the working tree** and now compiles (the earlier `dQ_tma.element_type.width` bug has been fixed to `self.dq_dtype.width // self.q_dtype.width` at `dqkernel.py:776`). Two older versions of the broken dq diff are in `git stash@{0}` and `git stash@{1}` — drop both after this lands.

## What's in place

The TMEM→reg→SMEM→TMA epilogue path is wired end-to-end and **compiles + runs**. Specifically:

- TMA store atoms `tma_atom_dK` / `tma_atom_dV` built at `dkdvkernel.py:546-580` (in `__call__` setup), threaded through `__call__ → dkdv_bwd → compute → epilogue`.
- SMEM staging `s_epi_dK` / `s_epi_dV` aliased onto `sP+sdST` (32 KB combined, dead at epilogue) at `dkdvkernel.py:2855-2862`. Audit confirms 32 KB ≥ 32 KB needed for one full (64, 256) bf16 tile.
- Per-thread `self.store(...)` calls at the original 2929/2942 are **replaced** with: TMEM→reg load (existing) → fp32→bf16 cast → `cute.autovec_copy(reg_cast, smem_partition)` → fence + named barrier (`epilogue_sync_bar_id=3`, 256 threads) → one warp issues `cute.copy(tma_atom_dV, smem_tma_frag, gmem_tma_frag)` → `cp_async_bulk_commit_group` → `cp_async_bulk_wait_group(0, read=True)`.

The new code is at `dkdvkernel.py:2926-3020`. Exact lines:

- `2935-2939`: `tTR_sdV / tTR_sdK = thread_t2r_*.partition_D(s_epi_*)` + `split_wg(...)`.
- `2944-2961`: GMEM tiles built via `cute.local_tile(dV_tma, ...)` then `local_tile(..., epi_tile_dKV, (0, None))` to add a unit epi-stage axis.
- `2963-2976`: `cpasync.tma_partition(tma_atom_*, 0, cute.make_layout(1), group_modes(s_epi_*, 0, 2), group_modes(gd*_tma_epi, 0, 2))`.
- `2986-2998` (dV) and `3008-3019` (dK): the actual store sequence.

## Symptom

`agent_space/ncu_hd256/correctness_check.py` (8 cases, MHA + GQA, causal + non-causal): all 8 fail. Pattern across all cases:

- `out` (forward, untouched by the new path): `max_abs ≈ 1e-3 to 1.5e-2` — bf16 noise, fine.
- `dq` (untouched — Codex's plumbing is stashed, dq still uses original per-thread store): `max_abs ≈ 1e-3 to 1.5e-2` — fine.
- `dk`: `max_abs = 0.4 to 5.5`. Off by orders of magnitude.
- `dv`: `max_abs = 0.6 to 10.3`. Off by orders of magnitude.

Forward and dQ being correct rules out infrastructure issues. The bug is **specifically in the dkdv epilogue's reg→SMEM→TMA path**.

The error magnitude is too large to be partial overlap or stale-data corruption from a single mispartitioned thread — it looks like the SMEM ends up with garbage for a substantial fraction of bytes.

## Strongest hypothesis (please verify or refute)

`thread_t2r_dV.partition_D(s_epi_dV_2d)` (line 2937) **does not place each thread's reg fragment at the SMEM offset that the SMEM swizzle expects**. The `make_smem_layout_epi`-built SMEM has a swizzle that the TMA descriptor depends on. The TMEM-load thread layout was designed for register-fragment ownership, not SMEM-position-correct writes.

This is what `copy_utils.partition_D_position_independent(thr_tmem_load, ...)` is supposed to handle in `flash_fwd_sm100.py:2740` — but that helper is **missing from our local copy_utils** (verified: not in `flash_attn/cute/copy_utils.py`, not in `/workspace/cutlass`, not in `/usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl`). The fwd kernel never reaches that line at runtime (the cute.jit lazy compile elides it for our config), so we never noticed.

## Suggested debug path

1. **Sanity check that the TMA path is the regression site.** Re-add per-thread store after the wait_group as a final overwrite. If correctness comes back, TMA path is the regression. (Quick patch: add `self.store(tTR_gdV, tTR_rdV, tTR_cdV, (K, D))` after `cp_async_bulk_wait_group(0, read=True)` for both dV and dK; mirror for the symmetric path.) This was prepped but not run before handoff.

2. **Compute the right SMEM partition.** The fwd recipe is:
   ```python
   smem_copy_atom = sm100_utils.get_smem_store_op(layout_enum, dtype, acc_dtype, tiled_t2r)
   tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_t2r)
   # then partition the SMEM tensor either via:
   #   thr_tmem_load.partition_D(s_epi_2d)              ← what we have, suspected wrong
   #   tiled_smem_store.get_slice(tidx).partition_D(s)  ← worth trying
   #   copy_utils.partition_D_position_independent(...) ← function missing locally
   ```
   For our atom (`Ld32x32bOp(Repetition(32))` → num_dp=32, num_bits=32, num_rep=32), `get_smem_store_op` falls through to `CopyUniversalOp` (per `cutlass/utils/blackwell_helpers.py:175-332`). With CopyUniversalOp the swizzle still applies via the SMEM tensor's layout, but the partition shape may differ.

3. **Alternative atom config.** `flash_bwd_sm100.py:3793-3946` has a working SM100 dK/dV TMA-store epilogue at hd=128. It uses `Ld32x32bOp(Repetition(self.dK_reduce_ncol))` (smaller rep) and a separate `tiled_copy_r2s_dKV` built via `cute.make_tiled_copy_C(...)` from a SMEM-store atom. Specifically lines 3888-3905, 3933-3946. The pattern is: separate r2s copy atom, partition SMEM with `thr_copy_r2s_dKV.partition_D(sdKV)`. This is the proven path — replicating it for hd=256 is probably the right move, even if it means changing the TMEM-load atom from `Repetition(32)` to a smaller rep that matches a stmatrix variant (so `get_smem_store_op` returns a proper StMatrix atom rather than CopyUniversalOp).

4. **If you change the TMEM-load atom for the epilogue,** the existing per-K-iteration TMEM-load atom (`Ld32x32bOp(Repetition(32))` at line 2865 of dkdvkernel.py) is *only* used in the epilogue, so replacing it has no other impact. But verify the partition shapes line up after the change.

## Validation gate

```bash
CUDA_VISIBLE_DEVICES=0 python3 agent_space/ncu_hd256/correctness_check.py
```
8/8 PASS. (User restricted GPU usage to 0–3 — please honor.)

If correctness lands, perf gate:
```bash
# rep=50 wallclock A/B vs HEAD on B200 clock-locked; target +5–15% on dkdv
# script template at agent_space/bench_pdl_delta.sh — adapt for HEAD vs WIP
```
Also ncu re-profile via `agent_space/ncu_hd256/profile.sh` — excessive sectors on the new line should drop <5%.

## Out of scope (for this debug session)

- Codex's dq plumbing fix — stashed; needs the `dQ_tma.element_type.width` → `self.q_dtype.width` one-line fix at `dqkernel.py:776` after un-stashing.
- Varlen support — `dV_tma`/`dK_tma` are TMA tensors and the manual-offset trick (`make_tensor(iter + offset, ...)`) doesn't work on them. flash_bwd_sm100.py:3837 has the same restriction (`assert not seqlen.has_cu_seqlens_k, "varlen uses non tma store path"`). Either gate the TMA path on non-varlen and keep `self.store` as varlen fallback, or handle varlen via `cute.domain_offset` like the load tensors.
