Rebased cherry-pick of `94e63db` from `Johnsonms/seqused-k-hd256` on top of `Johnsonms/paged-kv-hd256-v2`. The original branch was based on a pre-merge snapshot of the hd256 tree; base commits were absorbed into the merged hd256 PR #2412 (`27b4eb9`) and post-merge cleanup #2487 (`b21e204`).

**Stacked PR** — base branch should be `Johnsonms/paged-kv-hd256-v2`, not `main`. Depends on that PR for the paged-KV kernel plumbing.

## Change

Enables variable per-batch KV sequence lengths via a `seqused_k` tensor — needed for MLA-style decode (DeepSeek-V2 / V3 / R1), where different batches have different KV cache occupancies. Works with both dense and paged K/V.

### `flash_attn/cute/sm100_hd256_2cta_fmha_forward.py`

- Drop the `mSeqUsedK` half of the `__call__` assertion; only `mSeqUsedQ` stays blocked for now.
- Build the `seqused_k` cute tensor in `__call__` alongside the `page_table` / paged-layout construction.
- Add `mSeqUsedK` to `kernel()` signature and pass `seqused_k` from `__call__`.
- Replace `seqlen_k` derivation in all 4 warp sections with a ternary: `mSeqUsedK[batch_coord] if set, else <dense/paged expression>`.
- Move `batch_coord` above `seqlen_k` in the MMA warp (second warp section) — it was declared later but now needs to be in scope for `seqused_k` indexing.
- **Zero-KV batch handling (`seqlen_k == 0`):** extend `continue_cond` in all four warp sections with `continue_cond or seqlen_k <= 0`, so load / MMA / correction / softmax warps skip in sync instead of deadlocking on `K0 / Vend / QK0 / PVend / first-stats` tiles.

### `flash_attn/cute/interface.py`

- Relax the `seqused_q is None and seqused_k is None` assertion to `seqused_q is None` (the kernel now handles `seqused_k`).
- Prefill zero-KV batches on the host with zero output and `-inf` LSE, so their output is defined even though the kernel skips them.

### `tests/cute/test_flash_attn.py`

- `test_flash_attn_seqused_k_hd256_sm100`: dense + `seqused_k` (padded K/V with per-batch valid lengths) bit-exact vs a `cu_seqlens_k` packed reference, parametrized over asymmetric per-batch lengths.
- `test_flash_attn_paged_seqused_k_hd256_sm100`: paged + `seqused_k` combined (MLA decode pattern), bit-exact vs packed reference.
- `test_flash_attn_seqused_k_zero_hd256_sm100`: `seqused_k = 0` for one batch, parametrized dense/paged. Verifies no deadlock, zero output, `-inf` LSE on the empty row, and finite output / LSE on the other row.

## Validation

### Correctness smoke

`pytest tests/cute/test_flash_attn.py` on B200, filter combines the 6 new `seqused_k` tests, the 6 paged tests from the parent PR, and the existing d=256 dense subset:

```
-k "seqused_k_hd256_sm100 or paged_seqused_k_hd256_sm100 or
    seqused_k_zero_hd256_sm100 or paged_hd256_sm100_tma or
    (test_flash_attn_output and 256-False-0-0.0-False-False)"
```

Result: **90 passed, 78 skipped, 0 failed** — 78 from the dense d=256 subset (identical pass/skip count to `origin/main`), 6 from the parent PR's `paged_hd256_sm100_tma[_gqa]` tests, and **6 from the 3 new `seqused_k_*_hd256_sm100` tests** introduced here.

### FWD perf delta vs `origin/main` (TFLOPS mean, 3 runs each)

B200, bf16, hdim=256, locked clocks @ 1755 MHz.

#### MHA 32:32

| seqlen | causal | main | this PR | Δ |
|-------:|:------:|-----:|--------:|--:|
| 4k     | F | 1474 | 1485 | +0.7% |
| 8k     | F | 1582 | 1595 | +0.8% |
| 16k    | F | 1641 | 1650 | +0.5% |
| 32k    | F | 1450 | 1488 | **+2.6%** |
| 64k    | F | 1417 | 1484 | **+4.7%** |
| 128k   | F | 1398 | 1492 | **+6.7%** |
| 4k     | T | 1215 | 1217 | +0.2% |
| 8k     | T | 1411 | 1415 | +0.3% |
| 16k    | T | 1540 | 1545 | +0.3% |
| 32k    | T | 1552 | 1615 | **+4.1%** |
| 64k    | T | 1486 | 1496 | +0.7% |
| 128k   | T | 1363 | 1370 | +0.5% |

#### GQA 32:2

| seqlen | causal | main | this PR | Δ |
|-------:|:------:|-----:|--------:|--:|
| 4k     | F | 1496 | 1511 | +1.0% |
| 8k     | F | 1601 | 1612 | +0.7% |
| 16k    | F | 1620 | 1653 | **+2.0%** |
| 32k    | F | 1482 | 1464 | −1.2% |
| 64k    | F | 1436 | 1472 | **+2.5%** |
| 128k   | F | 1389 | 1469 | **+5.8%** |
| 4k     | T | 1242 | 1244 | +0.2% |
| 8k     | T | 1434 | 1439 | +0.3% |
| 16k    | T | 1556 | 1562 | +0.4% |
| 32k    | T | 1624 | 1564 | −3.7% |
| 64k    | T | 1493 | 1507 | +0.9% |
| 128k   | T | 1373 | 1368 | −0.4% |

Aggregated means: **MHA fwd +1.8%, GQA fwd +0.7%**.

## Caveat — unexpected long-seqlen non-causal speedup

The intent of this change is correctness only (adds a ternary, a `continue_cond` extension, moves a variable declaration). None of these should improve inner-loop throughput.

Yet we observe a **reproducible** +4–7% at long-seqlen non-causal (MHA 128k F +6.7%, GQA 128k F +5.8%, MHA 64k F +4.7%, MHA 32k T +4.1%). 3-run variance per cell is tight (typically <1%), so this is not run-to-run noise. Bracketed against the parent `paged-kv-v2` PR, which measured within ±0.3% of main on the same cells, the gain is introduced specifically by this commit's source-level reorderings (likely register-allocation or instruction-scheduling artifacts from ptxas).

### Why this is a concern, not just free perf

1. **Unintended change** — we can't explain it from the diff, which means the compiler's decision hinges on something fragile (variable declaration order, kernel signature shape). A future unrelated edit could flip this back to `±0%`, or worse, regress it.
2. **Could mask a different regression.** If the reordering also subtly changed some other code path we don't benchmark (e.g. paged path with `seqused_k = None`), we wouldn't notice until production.
3. **Not portable guidance.** We can't tell future contributors "move `batch_coord` earlier to get +6%" because the mechanism isn't a deliberate optimization.

### TODO before merging

- [ ] Compare SASS between this branch and `paged-kv-v2` for the long-seqlen non-causal dense path; identify which instructions changed and whether the gain is attributable to a specific scheduling/allocation difference.
- [ ] Confirm the paged path with `seqused_k = None` isn't regressed (`benchmark_attn.py` doesn't exercise paged; add a quick paged-bench harness or run the paged tests under timing).
- [ ] Decide whether to keep the reorderings (if the mechanism is understood) or revert the non-essential ones (declaration move) to isolate the correctness change from the accidental perf gain.

## Other caveats

- Host-side prefill in `interface.py` walks zero-KV batch indices in Python. For batch sizes in the thousands with many zero-KV rows this could show up in CPU profiles; current FA4 callers aren't in that regime.
- `seqused_q` is still asserted `None` — the kernel would need a similar ternary in the Q-side iteration, out of scope for this PR.
