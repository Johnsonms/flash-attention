# Lessons from variant 3a (FA4 hd=256 dKdV TMA epilogue)

Captured 2026-04-26. Intended audience: future kernel work in
`flash_attn/cute/` for FA4 backward kernels with similar tile shapes.

## Where to find things

- Final commit: `d3747c0` on `hd256-bwd-epilogue-refactor` (squashed).
- Pre-squash history (5-step incremental chain): `hd256-bwd-epilogue-refactor-history`.
- Original Codex/Claude handoff doc with full debug log: `VARIANT_3A_TASKS.md`.
- Reference SM100 hd=128 backward TMA epilogue: `flash_attn/cute/flash_bwd_sm100.py:3793-3956`.

## The killer constraint: per-thread t2r N coverage at hd=256 is interleaved

For our `dkdv_kernel` with `cta_tiler = (tile_m_dkdv, tile_n_dkdv, 256)`,
`num_compute_warps = 8` (so `num_compute_wgs = 2`), and the epilogue
TMEM-load atom `tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32))`,
the per-thread t2r partition layout is:

```
tTR_cdV.layout = ((32,1),1,2):((1@1,0),0,64@1)
tTR_rdV.layout = ((32,1),1,2):((1,0),0,32)
```

Decoded — each thread reads **64 unique reg values in 2 contiguous 32-N
sub-blocks at stride 64**. Per-WG distribution (verified via cute.printf):

| tidx range | M | N coverage (per-CTA, 0..255) |
|---|---|---|
| 0..63 | tidx (0..63) | `{0..31, 64..95}` |
| 64..127 | tidx-64 (0..63) | `{128..159, 192..223}` |

Per WG covers **128 N positions interleaved across the full per-CTA hd**,
NOT a contiguous half. The complementary set `{32..63, 96..127, 160..191,
224..255}` is owned by the other WG.

This is fundamentally different from the hd=128 reference (`flash_bwd_sm100.py`),
where each WG owns a contiguous 64 N positions. Because of this, **per-WG
TMA bulk store is structurally impossible** at hd=256 — TMA reads contiguous
SMEM and writes contiguous GMEM, but per-WG SMEM can never be made to satisfy
that constraint when its data is interleaved across the full GMEM N range.

If you're writing a similar epilogue and you see contiguous per-WG data
(e.g., hd=128), the reference's per-WG TMA pattern works directly. If
you see interleaved per-WG data (e.g., hd=256 with the current MMA tile),
you need the CTA-shared design described below.

## Tools that don't work for this case

### Manual `make_tiled_copy_tv` with simple thr/val layouts

Tested 5+ variants:
- `thr (64, 2) × val (1, 8)` — reference's adaptation. Per-thread writes
  to N=`{0..7, 16..23, 32..39, 48..55, 64..71, ...}` with stride-16 between
  atoms. Doesn't match t2r's stride-1 within sub-blocks + stride-64 between
  sub-blocks.
- `thr (64, 2) ord (0, 1) × val (1, 8)`, `thr (1, 128) × val (1, 8)`,
  `thr (64, 2) × val (1, 16)`, compound `thr ((4, 16), 2)` — none expose
  the inhomogeneous (32 contiguous + stride-64-jump) atom-iter pattern.

**Root cause**: `make_tiled_copy_tv`'s atom-iter axis is single-axis with
uniform stride. There's no `make_tiled_copy_tv` parameter that can produce
"4 atoms stride 8 then jump +32 then 4 more atoms stride 8" — i.e., the
compound atom-iter pattern that t2r naturally produces.

### Auto-derived `sm100_utils.get_smem_store_op` + `make_tiled_copy_D(atom, t2r)`

For our (64, 64) per-WG SMEM slot, the auto-derived partition is:
```
tdV_sdV_r2s = ((1,32),1,(2,2)):((0,1),0,(32,4096))
```

Per thread = 128 logical entries (mode-2 inner 2 broadcasts via stride 0,
mode-2 outer 2 covers two N-halves). `cute.copy(thr_copy_r2s, source, dest)`
reads 128 source entries per thread. Our reg fragment has only 64 entries
per thread → source-size mismatch, OOB reads, garbage SMEM.

`cute.make_tensor(reg.iter, partition.layout)` (passing layout instead of
shape) preserves the stride-0 broadcasts but mode-2 outer stride 32 still
demands 64 unique reg values that t2r doesn't provide in that ordering.

### `Repetition(epi_cols)` instead of `Repetition(32)`

Tested. `Repetition(64)` collapses the per-stage axis (`tTR_rdV.shape`
becomes `((64,1),1,1)` with mode 2 size 1 instead of 2), making per-stage
slicing impossible. Keep `Repetition(32)` — matches the reference's
`dK_reduce_ncol = gcd(32, tile_hdim // 2) = 32` for hd=128.

### Single-stage TMA box `(64, 128)`

Tested. The TMA atom builder accepts a 256B inner-dim box, but
`make_smem_layout_epi` produces a swizzle pattern incompatible with it,
yielding `cudaErrorIllegalAddress` at runtime. Stick with epi_tile widths
that fit a 128B SMEM swizzle stride
(`gcd(128 // sizeof(elem), per_wg_hd)`).

## What works: CTA-shared SMEM with cooperative per-element writes

```python
# Top-level (in __call__):
epi_cols_dKV = math.gcd(128 // (dK.element_type.width // 8),
                        cta_tiler[2] // num_compute_wgs)  # 64 for bf16 hd=256
num_epi_stages_dKV = (cta_tiler[2] // num_compute_wgs) // epi_cols_dKV  # 2
epi_tile_dKV = (cta_tiler[1], epi_cols_dKV)              # (64, 64)
total_epi_stages = num_compute_wgs * num_epi_stages_dKV  # 4

sdK/V_epi_layout = make_smem_layout_epi(dtype, layout_enum,
                                        epi_tile_dKV, total_epi_stages)
```

```python
# In epilogue, after t2r + cast to bf16:

# 1. Local cdV (no global domain offset!) for SMEM indexing.
cdV_local = cute.make_identity_tensor((cta_tiler[1], cta_tiler[2]))
tTR_cdV_local = split_wg(thread_t2r_dV.partition_D(cdV_local),
                         num_warp_groups, wg_idx)

# 2. Per-element cooperative writes — both WGs walk their per-thread
#    coords and store directly to s_epi_dV at the right (m, n_within, stage):
for _i in cutlass.range_constexpr(cute.size(tTR_cdV_local, mode=[2])):
    for _j in cutlass.range_constexpr(cute.size(tTR_cdV_local[None, 0, _i])):
        c = tTR_cdV_local[None, 0, _i][_j]
        m, n = c[0], c[1]
        v = tTR_rdV_cast[None, 0, _i][_j]
        s_epi_dV[m, n % epi_cols_dKV, n // epi_cols_dKV] = v

# 3. Inter-WG barrier (CTA-level, all 256 compute threads).
cute.arch.fence_view_async_shared()
cute.arch.barrier(barrier_id=5, number_of_threads=cta_threads)

# 4. WG 0's leader warp fires the multi-stage TMA. Other threads
#    no-op cp_async_bulk_wait_group(0) since they didn't issue.
if leader_warp and wg_idx == 0:
    for _stage in cutlass.range_constexpr(total_epi_stages):
        sdV_stage = s_epi_dV[None, None, _stage]
        gdV_stage = gdV_tma_epi[None, None, _stage]
        td_sdV, td_gdV = cpasync.tma_partition(
            tma_atom_dV, 0, cute.make_layout(1),
            cute.group_modes(sdV_stage, 0, 2),
            cute.group_modes(gdV_stage, 0, 2),
        )
        cute.copy(tma_atom_dV, td_sdV, td_gdV)
        cute.arch.cp_async_bulk_commit_group()
cute.arch.cp_async_bulk_wait_group(0, read=True)
```

## Important gotchas

### `cdV` vs `cdV_local`

Existing kernel code builds:
```python
cdV = cute.domain_offset(
    (blk_coord_k * tile_shape_K, 0),
    cute.make_identity_tensor((cta_tiler[1], cta_tiler[2])),
)
```
That `domain_offset` makes per-thread `tTR_cdV[i]` return `(M = blk_coord_k * tile_shape_K + local_M, N = local_N)`. The M values reach 4032+ at large block indices.
Indexing SMEM with these blew past the 64-row buffer (illegal address).
Build a separate non-offset `cdV_local` for SMEM indexing.

### Per-element indexed store DOES honour swizzle

`s_epi_dV[m, n_within, stage] = val` with runtime `m, n_within, stage`
honours the SMEM swizzle because `s_epi_dV.iterator` carries the swizzle
(installed via `cute.recast_ptr(..., sdK_epi_layout.inner, ...)`). The
outer layout's strides resolve `(m, n_within, stage)` to a logical offset;
the iterator translates the logical offset to a swizzled address.

You can verify a swizzled iterator is being used by inspecting
`sdV_per_wg.layout` after slicing — for our case it shows
`((8,8),(64,1)):((64,512),(1,0))`, where the `(8,8)` mode 0 is the swizzle
shape (compound M decomposition).

### `partition_D` on 3D SMEM tensor breaks the per-thread shape

`thread_t2r_dV.partition_D(s_epi_dV)` where `s_epi_dV` is 3D
`(M, N_within, stage)` returns a partition with an extra trailing
dimension that doesn't fit `tTR_cdV`'s shape `((32,1),1,2)`.
Slicing `[None, 0, _i]` on it raises `Operation creation failed`.
Don't try to flatten via `cute.make_tensor(iter, custom_layout)` either —
the custom layout's stride structure must match `s_epi_dV.layout.stride[i]`'s
nested tuple form, which is brittle. Use direct per-element indexed
stores instead (per the design above).

### TMA fire from one WG only

Both WGs have a leader warp (`(warp_idx % 4) == 0` is true once per WG).
If both fire `cp_async_bulk_commit_group`, you double the TMA traffic.
Gate on `wg_idx == 0` so only WG 0 fires; other threads call
`cp_async_bulk_wait_group(0)` which is a no-op for them (they didn't
issue any copies). WG 0 leader-warp threads correctly wait on their
own outstanding copies.

### Inter-WG barrier ID

Barrier IDs 0-4 are reserved by the kernel's `__init__`. We use 5 (dV)
and 6 (dK) for the inter-WG sync. Number of threads passed to
`cute.arch.barrier` must equal `num_compute_warps * threads_per_warp = 256`.

## Debug harness (drop in before t2r copy)

```python
# Confirm shapes and per-thread coord coverage.
_csdV = cute.make_identity_tensor((cta_tiler[1],
                                   cta_tiler[2] // num_warp_groups))
if tidx == 0:
    cute.printf("[dbg] tTR_cdV.layout = {}", tTR_cdV.layout)
    cute.printf("[dbg] s_epi_dV.layout = {}", s_epi_dV.layout)
    for _i in cutlass.range_constexpr(64):
        cute.printf("[dbg] tidx=0 tTR_cdV[{}] = {}", _i, tTR_cdV[_i])
if tidx in (1, 2, 4, 16, 32, 64, 96, 127):
    cute.printf("[dbg] tidx={} tTR_cdV[0]={} tTR_cdV[31]={} tTR_cdV[32]={} tTR_cdV[63]={}",
                tidx, tTR_cdV[0], tTR_cdV[31], tTR_cdV[32], tTR_cdV[63])
```

Run with:
```bash
CUDA_VISIBLE_DEVICES=0 python3 agent_space/ncu_hd256/correctness_check.py 2>&1 \
  | grep dbg | sort -u
```

## Possible future optimizations

- **Vectorize the per-element write**. The current per-thread loop emits
  64 individual SMEM stores per thread. Combining 4-8 of them into a
  vectorized store would reduce SMEM bank traffic. Likely needs a custom
  `partition_D` over a virtual `(64, 256)` SMEM layout — earlier attempts
  hit "shape and stride must be congruent" errors because the custom
  layout's stride tuple structure didn't match `s_epi_dV.layout.stride[i]`'s
  nested form. A future attempt would need to mirror `s_epi_dV`'s nested
  shape `((8,8),(64,1),(1,4))` exactly when constructing the virtual
  `(M, (N_within, stage))` view.
- **Reduce inter-WG barrier overhead**. Currently bar 5/6 is a full
  256-thread CTA barrier between writes and TMA. If the t2r layout were
  altered upstream to give each WG contiguous coverage, per-WG TMA
  becomes possible and the inter-WG barrier disappears. Would require
  changes to the MMA tile or `Repetition` value, which cascades.
