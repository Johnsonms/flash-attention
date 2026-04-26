# ncu profile — FA4 hd256 backward (B200, branch `hd256-bwd-epilogue-refactor` @ `d3747c0`)

**Profiled 2026-04-26.** Re-run of the original `agent_space/ncu_hd256/profile.sh`
with `CUTE_DSL_LINEINFO=1` for source-line attribution. Same configs as the
original baseline at `4819cf2`:

- **A**: MHA non-causal, B=8, S=16384, H=32, HKV=32, D=256
- **B**: MHA causal, same shape

Kernels covered: `dq_kernel` and `dkdv_kernel` per config (4 reports total).

## Headline result: dkdv excessive sectors → 0

Variant 3a's TMA bulk store path completely eliminates the original
"uncoalesced global access" hotspot.

| Kernel | Baseline excess sectors (`4819cf2`) | After (`d3747c0`) | Δ |
|---|--:|--:|--:|
| A_dkdv | 2 818 M | **0** | **-100%** |
| B_dkdv | 2 422 M | **0** | **-100%** |
| A_dq   | 70 M | 70 M | no change (variant 3a didn't touch `dq`) |
| B_dq   | 70 M | 70 M | no change |

The line-attribution top of the original list (now obsolete) — the per-thread
store at `dkdvkernel.py:2694` — no longer appears at all in the profile,
and neither do the strided LSE/dpsum loads at `:1922` and `:1953` (those
were already fixed by the loop-reorder commit `1748c9a`).

For `dq_kernel`, line 2143 (`cute.autovec_copy(tSMrdQ, tTMEM_LOADgdQ_i)`)
is still 95.5% of the 70 MB residual excess — variant 3a deliberately did
not touch dq; that's the next opportunity if/when we extend the refactor.

## SOL summary

| Kernel | SM TP% (baseline → after) | Mem TP% | DRAM% | L1 hit% | L2 hit% | Cycles/inst | Occ% |
|---|---|---|---|---|---|---|---|
| A_dkdv | 72.1 → **85.65** (+13.5pp) | 62.7 → 57.7 | 1.3 → low | 88.7 → **0**¹ | 95.8 → 96.6 | 16.2 → **13.45** (-17%) | 18.7 → 18.7 |
| B_dkdv | 47.7 → **61.40** (+13.7pp) | 49.0 → 44.0 | 1.6 → low | 92.5 → **0.3**¹ | 92.0 → 95.9 | 10.9 → ~10 | 18.7 → 18.7 |
| A_dq   | 82.3 → 82.4 | 26.0 → 26.1 | 1.8 | 97.9 → 97.9 | 90.2 → 90.4 | 17.5 → 17.5 | 18.7 → 18.7 |
| B_dq   | 87.2 → 87.1 | 32.7 → 32.7 | 3.4 | 96.3 → 96.3 | 92.4 → 92.4 | 16.2 → 16.2 | 18.7 → 18.7 |

¹ The L1 hit-rate collapse on dkdv is **expected and good**. Per-thread
GMEM stores routed through L1 (with the working set re-fetched, so 88-92%
of those reads hit L1 — high hit rate, but the stores themselves were the
bottleneck). The new TMA bulk store path bypasses L1 entirely; loads go
direct SMEM→L2/HBM. Hit rate of 0% on dkdv reflects "no more L1 store
traffic at all," not a cache miss issue.

## Stall breakdown

A_dkdv warp cycles per issued instruction: **13.45** (was 16.2).

- L1TEX scoreboard stall: 34.0% × 13.45 = **~4.6 cycles/inst** (was
  30.9% × 16.2 ≈ 5.0). Modest drop in absolute stall cycles, but much more
  important is that total cycles/inst dropped 17%.
- TC (tensor pipe) is now the highest-utilized pipeline at 85.7% — the
  kernel is genuinely tensor-pipe-bound, the way it should be at hd=256.

Compared to the original report's prediction:

> The realistic ceiling is the L1TEX-stall fraction we measured earlier:
> ~30% of warp cycles for `A_dkdv`, ~60% for `A_dq`. Of that, the lines
> above contribute the lion's share — so a working 5–15% wallclock
> improvement on the affected kernels is plausible if both fixes land.

Per-instruction cycles dropped 17% on A_dkdv and ~10% on B_dkdv. The
**13.5pp SM TP% bump** is consistent with the prediction; wallclock
should follow at the lower end of the 5–15% band on the dkdv-bound
configs (long-seqlen MHA-non-causal especially).

## What this means for the wallclock bench (Task 3.2)

Expectations based on these ncu numbers:

- **A_dkdv (long-seqlen MHA non-causal)**: expect a real wallclock
  improvement, maybe 5-12% on the dkdv-dominated configs. dq is unchanged
  so the overall bwd improvement depends on the dq:dkdv split — at long
  seqlens dkdv carries a larger share, so the win shows.
- **B_dkdv (MHA causal)**: B_dkdv wasn't bottlenecked the same way (IPC
  was already 1.10 in baseline). Excess sectors went to 0 but the kernel
  was already well-tuned, so wallclock improvement may be smaller and at
  risk of falling inside the noise band (±20 TFLOPS at MHA-causal mid
  seqlens per project memory).
- **dq configs**: no change expected (kernel untouched).

Goal of Task 3.1 was "excessive-sector attribution to the old hotspot
lines drops to <5% of total excess" — we're at **0%** of total excess
on dkdv, well past the goal. Architectural fix is correct; ready to
move to wallclock validation in Task 3.2.

## Reports

- `/tmp/ncu/A_dkdv.ncu-rep`, `/tmp/ncu/B_dkdv.ncu-rep`, `/tmp/ncu/A_dq.ncu-rep`, `/tmp/ncu/B_dq.ncu-rep` — open with `ncu-ui` for full interactive view.
- `agent_space/ncu_hd256/per_line.py /tmp/ncu/<tag>.ncu-rep derived__memory_l2_theoretical_sectors_global_excessive` — re-run line attribution.
