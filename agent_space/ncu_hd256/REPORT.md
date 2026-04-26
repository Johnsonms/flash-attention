# ncu profile — FA4 hd256 backward (B200, branch `pdl-hd256-bwd` @ 4819cf2)

## Configs profiled

- **A**: MHA non-causal, B=8, S=16384, H=32, HKV=32, D=256
- **B**: MHA causal, same shape

Both `dq_kernel` and `dkdv_kernel` profiled per config (4 reports total) with
sections: SpeedOfLight, ComputeWorkloadAnalysis, MemoryWorkloadAnalysis,
Occupancy, WarpStateStats, SchedulerStats, SourceCounters, LaunchStats.

ncu uses base SM clocks (~1.07 GHz here), so absolute durations are not
directly comparable to the wallclock bench; ratios and percentages are.

## SOL summary

| Kernel | SM TP% | Mem TP% | DRAM% | L1 hit% | L2 hit% | IPC | Cycles/inst | Occ% |
|--------|-------:|--------:|------:|--------:|--------:|----:|------------:|-----:|
| A_dq   |  82.3  |  26.0   |  1.8  |  97.9   |  90.2   | 0.69 | **17.5** | 18.7 |
| A_dkdv |  72.1  |  62.7   |  1.3  |  88.7   |  95.8   | 0.74 | 16.2 | 18.7 |
| B_dq   |  87.2  |  32.7   |  3.4  |  96.3   |  92.4   | 0.74 | 16.2 | 18.7 |
| B_dkdv |  47.7  |  49.0   |  1.6  |  92.5   |  92.0   | 1.10 | 10.9 | 18.7 |

- All four kernels: **DRAM is not the bottleneck** (1.8–3.4% throughput).
  Data lives in L1/L2 — these kernels stream K/V/dO through the cache, not
  the HBM.
- The "compute throughput" figure is dominated by the **TMEM (tensor memory)
  pipeline** — `LDT(M)/STT(M)/UTCCP/UTCMMA/UTCSHIFT`. `dq` kernels run TMEM
  at 82–87%, near peak; the secondary instruction-count pipeline is XU
  (integer aux) at ~27% — i.e., the kernels are genuinely tensor-pipe bound
  while UMMAs are issuing.
- Occupancy is **pinned at 18.7%** for all kernels (12 active warps / 64
  SM-warp limit). `Block Limit Registers = 1` and `Block Limit Shared Mem
  = 1` — single block per SM, register- and smem-limited.

## Top stall: L1TEX scoreboard dependency

| Kernel | L1TEX stall cycles | % of total | Est. speedup if removed |
|--------|-------------------:|-----------:|------------------------:|
| A_dq   | 10.4 / 17.5 | **59.5%** | 17.7% |
| A_dkdv |  5.0 / 16.2 |  30.9%    | 27.9% |
| B_dq   |  9.2 / 16.2 | **56.9%** | 12.8% |
| B_dkdv |  —          |   —       | —     |

> "Each warp spends N cycles waiting for a scoreboard dependency on a L1TEX
> (local, global, surface, texture) operation."

`B_dkdv` is the outlier — IPC 1.10, only 10.9 cycles/inst, no dominant
stall reason. **It is already well-tuned**; the other three have headroom.

## Top theoretical opportunity: uncoalesced global accesses

ncu flags excessive L2 sector requests:

| Kernel | Excessive sectors | % of total | ncu Est. speedup |
|--------|------------------:|-----------:|-----------------:|
| A_dq   |     70 M | 51%  | 41.4% |
| A_dkdv |  2 818 M | 88%  | **87.5%** |
| B_dq   |     70 M | 51%  | 51.3% |
| B_dkdv |  2 422 M | 90%  | **89.3%** |

⚠️ **The estimates are theoretical**. With L1 hit rate 88–98%, most of these
"excessive" sectors are absorbed by L1 — the actual cost is paid in L1TEX
stall cycles (which we already see at 60%), not DRAM bandwidth. Realistic
speedup if fully fixed is closer to the L1TEX-stall estimates above
(13–28%), not the headline 41–89%.

## Branch divergence

| Kernel | Branch eff% | Avg divergent branches |
|--------|------------:|----------------------:|
| A_dq   | 99.97 |    83 |
| A_dkdv | 99.99 |   166 |
| B_dq   | 99.95 |    83 |
| B_dkdv | **99.04** | **7,255** |

`B_dkdv` has 50–100× the divergent branches of the others — the causal
mask path in dkdv generates a lot of in-warp divergence. Counter-intuitively
it's also the *fastest* kernel (IPC 1.10), so divergence isn't currently
costing much; it would be a place to look only if a future change started
to expose it.

## What to actually go after

Ranked by evidence strength × likely real impact:

1. **Reduce L1TEX-dependency stalls in `dq` kernels (both A and B).**
   60% of `dq` warp time waits on L1 hits to return. With achieved
   occupancy already pinned at theoretical max (12 warps), the lever is
   pipeline depth — more in-flight K/V tile loads, deeper TMA mbarrier
   pipeline, earlier scalar-load hoisting (`lse`, `dpsum`). Realistic win:
   10–15% on `dq` per ncu's stall-removal estimate.

2. **Trace the "excessive sectors" to source.** ncu's source-page output
   has the per-instruction breakdown but only at SASS addresses — no source
   correlation because the PTX wasn't built with `CUTE_DSL_LINEINFO=1`.
   Next step: rebuild with `CUTE_DSL_LINEINFO=1`, re-profile, find the
   `L2 Theoretical Sectors Global Excessive` hotspot lines. The dkdv
   kernels show 88–90% excess — this is unlikely to be in TMA bulk loads
   (which coalesce by construction); more plausibly per-thread loads of
   stats (`lse`, `dpsum`) or scattered accumulator I/O.

3. **Don't bother optimizing `B_dkdv`** — already at IPC 1.10 with no
   dominant stall. Causal mask + skip-tile scheduler is doing its job.

4. **`A_dkdv` is the most-bandwidth-touching kernel** (Mem Pipes Busy 69%,
   Memory Throughput 63%) — non-causal sees every K/V tile. If we want a
   bigger absolute win there, the candidate is **fewer total tile loads**
   (better re-use across heads in MHA, e.g. via larger M tiles or
   pack_gqa-style head packing — though hd=256 already limits how big tiles
   can be).

5. **Skip / deprioritize:**
   - DRAM-side optimizations (DRAM at <4%, no headroom to recover).
   - Occupancy bump via register reduction (Block Limit Registers = 1
     means we'd need to chop registers significantly; likely costs
     pipeline depth more than it gains warp count).
   - L2 compression (0% currently, but bf16 attention activations don't
     compress well; tail wag).

## Source-line attribution (lineinfo re-profile)

Re-profiled with `CUTE_DSL_LINEINFO=1` and aggregated the
`derived__memory_l2_theoretical_sectors_global_excessive` metric per source
line via the ncu Python API (`agent_space/ncu_hd256/per_line.py`). Three
hotspot lines explain ≥95% of the excess for every kernel:

### `dkdv_kernel` (A_dkdv 2.82 GB excess, B_dkdv 2.42 GB excess)

| Rank | A excess | B excess | File:Line | Source |
|----:|---------:|---------:|-----------|--------|
| 1 | 71.4% | 83.1% | `..._dkdvkernel.py:2694` | `gmem_i.store(regs_i.load().to(gmem.element_type))` — dK/dV epilogue store |
| 2 | 14.2% |  8.3% | `..._dkdvkernel.py:1922` | `cute.copy(atom_async_copy, LSE_for_copy[..., LSE_idx + i, ...], ...)` — per-K-iter LSE load |
| 3 | 14.2% |  8.3% | `..._dkdvkernel.py:1953` | `cute.copy(atom_async_copy, sum_OdO_for_copy[..., sum_OdO_idx + i, ...], ...)` — per-K-iter dpsum load |

### `dq_kernel` (A_dq 70 MB excess)

| Rank | excess | File:Line | Source |
|----:|--------:|-----------|--------|
| 1 | 95.5% | `..._dqkernel.py:2148` | `cute.autovec_copy(tSMrdQ, tTMEM_LOADgdQ_i)` — dQ epilogue store from register file |
| 2 |  2.2% | `..._dqkernel.py:1024` | (early-iteration load, low impact) |
| 3 |  2.2% | `..._dqkernel.py:1050` | (early-iteration load, low impact) |

### Two distinct root causes

**(1) Epilogue store layout (lines 2148 in dq, 2694 in dkdv) — dominant excess.**
TMEM→register→GMEM store path. The MMA accumulator in TMEM is laid out
with each thread holding strided slices of many rows, so when each thread
issues `gmem_i.store(...)`, the warp's 32 stores hit non-contiguous rows
of the output `dQ` / `dK` / `dV` tensor (hd=256 = 16 sectors per row).
Result: per-warp store generates 32× the sector traffic of a coalesced
write.

The standard fix is **TMEM → SMEM (via UTCSHIFT/UTCCP) → GMEM (via TMA
bulk store)**, which produces 128-byte coalesced stores. dq's load path
already uses TMA for K/V/dO; only the *output* path is per-thread.

**(2) Strided per-element scalar loads of LSE / dpsum (lines 1922, 1953).**
The current pattern is:
```python
LSE_idx = self.tile_shape_Q * iter_index + thread_idx * async_copy_num_elts
for i in range(async_copy_num_elts):
    cute.copy(..., LSE_for_copy[..., LSE_idx + i, ...], ...)
```
Within a single `i` iteration, thread T loads at offset
`iter*Q + T*N + i` — i.e., stride `N` between threads. For
`async_copy_num_elts=N>1`, the warp's 32 4-byte loads hit `N` sectors
instead of 1, generating `(N-1)/N` = 75–87% excess. Fix is to swap the
loop nesting so the warp axis is the inner one:
```python
for i in range(async_copy_num_elts):
    idx = self.tile_shape_Q * iter_index + i * 32 + thread_idx  # contiguous warp
    cute.copy(..., LSE_for_copy[..., idx, ...], ...)
```

### Realistic impact estimate

- The headline `Est. Speedup: 87.5%` from ncu assumes all excess sectors
  cost DRAM bandwidth — they don't (L1 hit rate 97%, DRAM <2%).
- The realistic ceiling is the L1TEX-stall fraction we measured earlier:
  ~30% of warp cycles for `A_dkdv`, ~60% for `A_dq`. Of that, the lines
  above contribute the lion's share — so a working 5–15% wallclock
  improvement on the affected kernels is plausible if both fixes land.
- The TMEM→SMEM→TMA epilogue change is the **bigger win** (71–96% of
  the excess) but a larger refactor. The strided LSE/dpsum loop reorder
  is a one-file, ≤20-line change with a smaller share of the win.

### Reports

`/tmp/ncu/{A,B}_{dq,dkdv}.ncu-rep` — initial profiles (no source).
`/tmp/ncu/{A_dq,A_dkdv,B_dkdv}_lineinfo.ncu-rep` — with source attribution.
`agent_space/ncu_hd256/per_line.py` — per-line aggregation script.

## Reports

`/tmp/ncu/{A_dq,A_dkdv,B_dq,B_dkdv}.ncu-rep` — open with
`ncu-ui` for the full interactive view, or
`ncu --import <file> --page details` for text.
