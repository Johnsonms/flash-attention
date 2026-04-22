Rebased cherry-pick of `dbb6c98` from `Johnsonms/persistent-cluster-hd256`
on top of `Johnsonms/seqused-k-hd256-v2`. Original branch was based on a
pre-merge snapshot of the hd256 tree; base commits were absorbed into
the merged hd256 PR #2412 (`27b4eb9`) and post-merge cleanup #2487
(`b21e204`).

**Stacked PR** — base branch should be `Johnsonms/seqused-k-hd256-v2`,
not `main`. Depends on that PR for the `seqused_k` + paged-KV plumbing.

## Change

Persistent scheduling amortizes CTA launch overhead by issuing a
grid-stride loop over tiles. It was hardcoded off in hd256 since the
kernel's inception because the static tile scheduler was cluster-
unaware and split 2CTA clusters across independent work tiles,
corrupting output.

### Cluster-aware fix

- `Sm100FmhaStaticTileSchedulerParams` gains a `cluster_shape_m`
  constexpr (default 1, so 1CTA kernels are unchanged).
- Grid is sized in cluster units:
  `max_ctas = (sm_count // cluster_shape_m) * cluster_shape_m`;
  problem size is multiplied by `cluster_shape_m` so `dsl_min`
  compares apples to apples.
- `num_persistent_clusters` replaces `num_persistent_sm` as the
  grid-stride step, so `advance_to_next_work` advances by one cluster
  per iteration instead of one CTA.
- In `get_current_work`, the CTA rank within the cluster is
  reconstructed from the launch `block_idx`
  (`cta_rank = blk_coord[0] % cluster_shape_m`) and spliced back into
  `mid = m_block * cluster_shape_m + cta_rank`, so both CTAs in a 2CTA
  cluster land on their half of the same tile.

### Enablement gate (work-per-tile heuristic)

Persistent's per-tile cost (cluster-rank reconstruction + grid-stride
state) only pays off when work-per-tile is small, i.e. short KV. On
B200, persistent wins at short seqlen but regresses dense prefill at
long seqlen because launch overhead is already amortized by the
hardware at high tile counts.

- **Gate:** `... and seqlen_k <= 2048`. Keeps the decode win and
  holds long-context prefill within noise of the pre-persistent
  baseline.
- Gate is folded into `interface.py`'s compile_key so configs
  straddling the threshold compile separate kernels with the correct
  persistent flag baked in.

### Files

- `flash_attn/cute/tile_scheduler.py`: add `cluster_shape_m` on
  `Sm100FmhaStaticTileSchedulerParams` and the matching updates to
  `Sm100FmhaStaticTileScheduler` and `compute_sm100_fmha_grid`.
- `flash_attn/cute/sm100_hd256_2cta_fmha_forward.py`: drop
  `self.is_persistent = False` hardcode (now honors constructor arg),
  pass `cluster_shape_mnk` to `compute_grid`, divide `blk_idx[0]` by
  `cluster_shape_mnk[0]` when constructing `FmhaStaticTileScheduler`.
- `flash_attn/cute/sm100_hd256_2cta_fmha_backward_dqkernel.py`:
  same fix applied to the bwd dQ kernel for consistency (its
  scheduler is the same `Sm100` static scheduler).
- `flash_attn/cute/interface.py`: compute `hd256_is_persistent` above
  the compile_key (gated on `seqlen_k <= 2048` in addition to
  causal/cu_seqlens_q/...); add it to the compile_key; use it at
  `fa_fwd` construction.

## Validation

### Correctness smoke

`pytest tests/cute/test_flash_attn.py` on B200, filter combines the
6 `seqused_k` tests (inherited from parent PR), the 6 paged tests
(grandparent PR), and the existing d=256 dense subset:

```
-k "seqused_k_hd256_sm100 or paged_seqused_k_hd256_sm100 or
    seqused_k_zero_hd256_sm100 or paged_hd256_sm100_tma or
    (test_flash_attn_output and 256-False-0-0.0-False-False)"
```

Result: **90 passed, 78 skipped, 0 failed** — identical pass/skip
count to `origin/main`. No new tests added by this commit (scheduler
refactor tests are covered via the existing d=256 dense parametrize
space, which hits both gate-ON and gate-OFF configurations).

### FWD perf delta vs `origin/main` (TFLOPS mean, 3 runs each)

B200, bf16, hdim=256, locked clocks @ 1755 MHz.

#### Short seqlen — **gate-ON path** (persistent scheduling active)

##### MHA 32:32

| seqlen | causal | main | this PR | Δ |
|-------:|:------:|-----:|--------:|--:|
| 1k     | F | 985  |  987 | +0.2% |
| 2k     | F | 1307 | 1301 | −0.5% |
| 1k     | T | 592  |  592 |  0.0% |
| 2k     | T | 910  |  911 | +0.1% |

##### GQA 32:2

| seqlen | causal | main | this PR | Δ |
|-------:|:------:|-----:|--------:|--:|
| 1k     | F | 1012 | 1015 | +0.3% |
| 2k     | F | 1329 | 1329 |  0.0% |
| 1k     | T |  600 |  600 |  0.0% |
| 2k     | T |  919 |  919 |  0.0% |

**Observation:** the persistent path is active but delivers essentially
**no speedup** in this bench config (32 Q-heads MHA, 32:2 GQA). The
commit message cites +10–25% wins at `seqlen_k=1024` on a different
config (8 Q-heads); with 32 Q-heads the tile count per batch is 4×
larger and launch overhead is already amortized by the hardware, so the
persistent loop's benefit is saturated out. Reproducibility is very
tight (variance <1 TFLOPS across 3 runs), so this is not run noise —
just "no regression" rather than "a win" at this config.

#### Long seqlen — **gate-OFF path** (should match main)

##### MHA 32:32

| seqlen | causal | main | this PR | Δ |
|-------:|:------:|-----:|--------:|--:|
| 4k     | F | 1474 | 1479 | +0.3% |
| 8k     | F | 1582 | 1588 | +0.4% |
| 16k    | F | 1641 | 1636 | −0.3% |
| 32k    | F | 1450 | 1442 | −0.6% |
| 64k    | F | 1417 | 1447 | **+2.1%** |
| 128k   | F | 1398 | 1423 | +1.8% |
| 4k     | T | 1215 | 1218 | +0.2% |
| 8k     | T | 1411 | 1416 | +0.4% |
| 16k    | T | 1540 | 1540 |  0.0% |
| 32k    | T | 1552 | 1619 | **+4.3%** |
| 64k    | T | 1486 | 1469 | −1.1% |
| 128k   | T | 1363 | 1383 | +1.5% |

##### GQA 32:2

| seqlen | causal | main | this PR | Δ |
|-------:|:------:|-----:|--------:|--:|
| 4k     | F | 1496 | 1504 | +0.5% |
| 8k     | F | 1601 | 1598 | −0.2% |
| 16k    | F | 1620 | 1640 | +1.2% |
| 32k    | F | 1482 | 1539 | **+3.8%** |
| 64k    | F | 1436 | 1460 | +1.7% |
| 128k   | F | 1389 | 1363 | −1.9% |
| 4k     | T | 1242 | 1244 | +0.2% |
| 8k     | T | 1434 | 1438 | +0.3% |
| 16k    | T | 1556 | 1557 | +0.1% |
| 32k    | T | 1624 | 1574 | **−3.1%** |
| 64k    | T | 1493 | 1493 |  0.0% |
| 128k   | T | 1373 | 1369 | −0.3% |

**Observation:** long-seqlen path is within ±2% on most cells. Same
long-seqlen noise pattern as the parent PRs at 32k/64k (batch-
quantization zone); deltas swing both directions, no systematic
regression.

## Caveats

- **Short-seqlen benefit is config-dependent.** The persistent path
  trades extra per-tile state for launch-overhead amortization; the
  trade only pays off when work-per-tile is small. At 8 Q-heads the
  commit's original bench showed +10–25% at 1k; at 32 Q-heads in my
  bench the benefit collapses to 0% because tile count is already
  high enough to amortize launches. If the use case is decode-style
  small-batch / few-head, the gate-ON path is expected to deliver the
  advertised win.
- Backward dQ kernel gets the same cluster-aware scheduler change
  for consistency but is not exercised by `benchmark_attn.py` in this
  PR's validation. Forward dense path is the regression-critical one
  and is covered above.
- `cluster_shape_m` currently defaults to 1 so 1CTA kernels are
  unaffected; the only callers passing `cluster_shape_m > 1` are
  hd256 forward and hd256 bwd dQ.

