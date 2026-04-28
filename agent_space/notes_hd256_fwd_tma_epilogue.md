# hd256 Forward Kernel: TMA Epilogue for O — Investigation Notes

Branch: `fa4/hd256-fwd-tma-epilogue`  
Date: 2026-04-28  
GPU: NVIDIA B200 (SM100), locked clocks 1965 MHz  
Config: B=4, S=8192, H=8, D=256, dtype=bfloat16

---

## Motivation

The hd256 2CTA forward kernel (`sm100_hd256_2cta_fmha_forward.py`) writes output O
via per-thread `autovec_copy` in `correction_epilog`. The general SM100 forward kernel
(`flash_fwd_sm100.py`) already uses TMA bulk-store for O (`use_tma_O`). The goal was
to match that pattern in the dedicated hd256 kernel and check for a performance gain.

---

## What Was Implemented

Steps 1–4 landed in three commits:

| Commit  | Description |
|---------|-------------|
| 903f415 | Add TMA S2G atom (`CopyBulkTensorTileS2GOp`) and `sO_epi_layout` to `setup()`. TMA atom prefetched in load warp. |
| a706e69 | Alias `sO_epi` staging buffer onto dead `sQ` SMEM (8 KB needed, 64 KB available, no footprint change). |
| 1135107 | Rewrite `correction_epilog` non-varlen path: stage into `sO_epi`, `fence_view_async_shared`, `barrier(id=3, 128)`, leader-warp TMA `cp.async.bulk` S2G, second barrier. Varlen path unchanged. |

Staging layout: `epi_tile_O = (64, 64)`, `num_epi_stages_O = 2` (two 64×64 tiles per
64×128 epi block). Leader warp = `warp_idx % 4 == 0` (correction warp 4).
Named barrier_id=3 (confirmed free; only barrier_id=1 used by tmem_alloc).

Correctness: 3473/3473 tests passed (non-varlen hdim=256), causal and non-causal.

---

## Benchmark Results (`python benchmarks/benchmark_attn.py --headdim 256 --fwd`)

| seqlen | causal | Baseline (ms / TFLOPS) | TMA epilogue (ms / TFLOPS) | delta |
|--------|--------|------------------------|----------------------------|-------|
| 1024   | False  | 0.29 / 941             | 0.31 / 894                 | −5.0% |
| 2048   | False  | 0.43 / 1277            | 0.45 / 1235                | −3.3% |
| 4096   | False  | 0.75 / 1457            | 0.76 / 1445                | −0.8% |
| 8192   | False  | 1.40 / 1568            | 1.40 / 1567                | ~0%   |
| 1024   | True   | 0.24 / 583             | 0.26 / 534                 | −8.4% |
| 2048   | True   | 0.31 / 894             | 0.33 / 840                 | −6.0% |
| 4096   | True   | 0.47 / 1176            | 0.48 / 1141                | −3.0% |
| 8192   | True   | 0.81 / 1361            | 0.82 / 1345                | −1.2% |

TMA epilogue is consistently slower, with regression worst at short sequences (~8% causal).

---

## NCU Analysis (seqlen=8192, non-causal)

| Metric | Baseline | TMA epilogue |
|--------|----------|--------------|
| Duration | **2.17 ms** | 2.20 ms |
| Highest pipeline | **TC 85.6%** | TMEM 84.2% |
| Executed instructions | **487M** | 492M (+1%) |
| Local mem spilling | **2.42 MB** | 8 KB |
| SMEM spilling | **0 B** | 2.98 MB |
| L1/TEX hit rate | 89.5% | 99.97% |
| L2 compression input sectors | **265K** | 4,590K (17×) |
| Block Limit Barriers | **32** | 16 |
| DRAM throughput | 3.58% | 3.56% |

### Root cause: three compounding problems

**1. Pipeline bottleneck shifted TC → TMEM.**  
The baseline is Tensor Core (TC) bound at 85.6% — the expected GEMM-saturated state.
The TMA epilogue shifted the bottleneck to the TMEM pipeline (84.2%), because the SMEM
staging writes, cp.async.bulk descriptors, and fence/barrier pairs all go through the
TMEM subsystem (same path as WGMMA accumulator LDT/STT). This creates new contention
right where the accumulator reads from the correction warps are happening.

**2. Register spilling migrated from local-mem to SMEM (2.42 MB → 2.98 MB).**  
The TMA staging code increased register pressure in the correction warps. The compiler
switched from spilling to L2 (local memory) to spilling to SMEM. SMEM spills compete
directly with the `sO_epi` staging writes and further stress the TMEM pipeline.
The L2 compression input sectors explode 17× as a symptom (SMEM spill traffic landing
in L2 at high granularity).

**3. Nothing to save on the memory side.**  
DRAM throughput is only 3.5% — the kernel is essentially pure-compute. The O store is
a tiny fraction of total memory traffic. Optimizing it with TMA provides no measurable
bandwidth savings to offset the overhead of two barriers + tma_partition + commit/wait
per 64×64 tile.

**Side effect: Block Limit Barriers halved (32 → 16).**  
The named barrier_id=3 cut the hardware barrier limit from 32 → 16 blocks/SM.
Doesn't hurt today (already 1 block/SM from regs+SMEM), but removes future headroom.

---

## Why TMA Epilogue Helps in the Backward but Not Here

In the dK/dV backward (`sm100_hd256_2cta_fmha_backward_dkdvkernel.py`), TMA helped
because the dK/dV stores are scattered across two warpgroups that each write partial
tiles — the scatter pattern was genuinely inefficient with per-thread stores.
The forward O epilog is done by four correction warps (128 threads total) all writing
a single dense 64×128 tile, which is already well-coalesced with autovec_copy.

---

## Real Bottleneck and Opportunities

The kernel is **TC-bound at 85%** with **59% "No Eligible" warp cycles**.

Actionable directions:

1. **Register spill latency (~34% of warp stalls).**  
   The CPIStall warning identifies 2.5 cycles/warp stalled on L1TEX scoreboard,
   accounting for ~34% of stall cycles. These are correction-warp register spill
   read-backs from softmax intermediates. Pruning the correction warp's register
   footprint (tighter softmax loop unroll, fewer inlined constants) could reduce
   spill-load round trips.

2. **Pipeline correction over next K-tile load.**  
   The correction epilog (tmem_load + scale + convert + store O) is sequential with
   the next KV fetch. A double-buffered epilog that overlaps correction with the
   next TMA load would hide correction latency behind I/O.

3. **No path to 2 blocks/SM.**  
   168 regs × 384 threads = 64,512 fills the 64K register file exactly.
   SMEM is 197.5 KB > 114 KB threshold for 2 blocks/SM. Both limits are hard;
   there is no viable register-reduction path here.

---

## Pre-existing Backward Test Failure (unrelated)

`test_flash_attn_varlen_output[...-256-256-128-...-15.0-...-mha-dtype0]` fails on
both origin/main and this branch. It is hdim=128 (the "256-256" in the name is
seqlen, not headdim), softcap=15.0, varlen, unpad_q=True, unpad_kv=True.
dQ max diff = 2.625 (well above tolerance). This is a pre-existing bug in the
general SM100 backward kernel (`flash_bwd_sm100.py`) for hdim=128 + varlen +
softcap — not introduced by any commit on this branch.
