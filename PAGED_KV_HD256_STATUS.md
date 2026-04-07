# SM100 hd256 2CTA Kernel — Status & Roadmap

## Current PR (Uncommitted Changes)

### 1. seqused_k support
Per-batch-element KV sequence lengths for production inference where each request
has a different KV cache length.

- `interface.py`: Capture `seqused_k_tensor`, add RESIDUAL_MASK detection, pass to kernel
- `sm100_hd256_2cta_fmha_forward.py`: Add `mSeqUsedK` to kernel signature/setup,
  update `seqlen_k` in all 4 warp sections, fix `batch_coord` ordering bug in MMA warp

### 2. Persistent scheduling fix
The 2CTA persistent path was broken since the kernel was written (`is_persistent`
was hardcoded `False`). Two bugs:

- **Cluster-aware scheduling** (`tile_scheduler.py`): The scheduler treated each CTA
  as an independent worker. In 2CTA, both CTAs in a cluster must work on the same
  logical tile. Fixed by sizing grids in cluster units and advancing by
  `num_persistent_clusters`.
- **CTA rank in tile coord** (`tile_scheduler.py`): `get_hier_coord` returned only
  the logical m_block, but the kernel needs `mid = m_block * cluster_shape_m + cta_rank`
  so each CTA processes its half of the tile. Fixed in `get_current_work()`.

### 3. Compile key fix
`is_persistent` was missing from `compile_key` in `interface.py`, causing different
persistent/non-persistent configs to silently share cached kernels.

### 4. Benchmark script
`benchmarks/bench_sm100_hd256_paged_tma.py` updated for DeepSeek-V2/V3/R1 MLA style
(H=128, d=256, MHA). Includes OOM guard, memory cleanup, correct einsum reference,
three benchmark sections (paged+seqused_k, paged, non-paged).

## Performance (B200, H=128 d=256 decode)

| Scenario | 2CTA vs Generic SM100 | Notes |
|----------|----------------------|-------|
| Non-paged contiguous | **1.00x – 1.09x** | Apple-to-apple: isolates 2CTA MMA advantage |
| Paged + seqused_k | 0.16x – 0.85x | Generic wins due to faster scheduler |
| Paged, no seqused_k | 0.23x – 0.87x | Same scheduler gap |

The non-paged comparison is the true apple-to-apple benchmark — both use the same
scheduling strategy, so the ~3–7% advantage comes purely from 2CTA MMA instructions.

The paged gap is **not** from 2CTA MMA being slower. It's because Generic SM100 uses
`StaticPersistentTileScheduler` (FastDivmod-based O(1) tile decomposition) while
our 2CTA kernel uses `Sm100FmhaStaticTileScheduler` (`get_hier_coord`, slower).

Current dispatch routes paged d=256 decode to Generic (optimal for now).

## Roadmap (Priority Order)

### 1. Port `StaticPersistentTileScheduler` to 2CTA kernel
**Impact**: Closes the scheduler gap; 2CTA should then win for paged decode too.

The Generic kernel's `StaticPersistentTileScheduler` uses `FastDivmodDivisor` for
O(1) tile-to-coordinate decomposition. Our `Sm100FmhaStaticTileScheduler` uses
`get_hier_coord` which is slower. Porting the FastDivmod scheduler and adapting it
for 2CTA clusters (seeding both CTAs with the same logical tile, encoding CTA rank
in `mid`) would bring scheduler parity. Then flip dispatch to route paged d=256
to 2CTA.

### 2. Split-KV for long sequences
**Impact**: Improves throughput at Sk=32K+ where single-pass is memory-bandwidth bound.

Split-KV parallelizes across K blocks, each producing partial (O, LSE), then a
combine kernel merges them. Generic SM100 already supports this via
`FlashAttentionForwardCombine`. Need to wire split-KV into the 2CTA path.

### 3. Conditional CLC scheduler
**Impact**: +8–10% at high tile count, but −9% at low tile count.

CLC (Cooperative Launch Control) is hardware tile dispatch on SM100. Currently
controlled by `FA_CLC` env var. Should auto-select based on a tile count threshold
(e.g., enable when `total_tiles > 64`).

### 4. Persistent decode tile skipping
**Impact**: Faster varlen/seqused_k with highly variable sequence lengths.

For workloads where some batch elements have very short sequences, persistent CTAs
that get tiles beyond `seqlen_q` need to skip safely. The 2CTA barrier protocol
currently doesn't support tile skipping (attempt with `continue_cond` produced
corruption). Needs a barrier-safe skip protocol or a run-through approach (compute
on garbage, gate writes only).
