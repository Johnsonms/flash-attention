# PR draft: `[FA4][hd256] Add griddepcontrol_wait() to dq_kernel backward`

Branch: `pdl-hd256-bwd` on `https://github.com/Johnsonms/flash-attention`
PR-create URL: https://github.com/Johnsonms/flash-attention/pull/new/pdl-hd256-bwd
Target: `Dao-AILab/flash-attention:main`
Commit: `e3cdf9f` (1 file, +5 −0)

---

## Title

```
[FA4][hd256] Add griddepcontrol_wait() to dq_kernel backward
```

## Body

```markdown
## Summary
- Mirrors #2481 (sm90 bwd fix) for sm100 hd256 backward.
- Enables `use_pdl=True` on `dq_kernel` and adds `griddepcontrol_wait()` in
  the load warp before the first reads of `lse_log2` / `sum_OdO`.
- Without the wait, with PDL enabled, dq's load warp could read partially-
  written preprocess outputs, corrupting `dpsum` → `dS = P·(dP − dpsum)` →
  wrong `dQ` (and dK/dV through shared dependencies).

## Verification (PTX, `CUTE_DSL_KEEP_PTX=1`, sm_100a)
- `dq_kernel` emits `griddepcontrol.wait` once, gated to `load_warp_id`
  (predicate `setp.ne.s32 %p, %r, 9`), placed after `setmaxnreg.dec` and
  before the tile loop — i.e., before any GMEM read of preprocess outputs.
- `bwd_preprocess` already emits `griddepcontrol.launch_dependents` after
  the vectorized O/dO global loads (`flash_bwd_preprocess.py:316`,
  unchanged by this PR).
- `dkdv_kernel` is unchanged here (does not use PDL on this path).

## Performance
Measured on B200 (clock-locked 1755 MHz), `--headdim 256 --bwd --backend
fa4 --warmup 5 --rep 50`, comparing `main` vs this PR. Seqlens 4K–131K,
causal {true, false}, MHA (nheads=32, kv=32) and GQA (nheads=32, kv=2).

Long seqlens (≥65K, fully compute-bound) — all configurations within ±1%
of `main`, except a single MHA-causal-65K outlier at +5.1% which is paired
with an opposite-direction outlier at MHA-causal-32K (−5.8%); both are
measurement noise on this branch (run-to-run variance ±20 TFLOPS at this
size; long seqlens median Δ ≈ ±0.5%).

Mid seqlens (4K–32K), where PDL overlap is most visible:

| config       | median ΔTF | best  | worst              |
|--------------|-----------:|------:|-------------------:|
| MHA non-causal | +1.1%   | +4.1% | −1.2%              |
| MHA causal     | +0.3%   | +0.3% | −5.8% (32K, noise) |
| GQA non-causal | +1.9%   | +3.3% | −0.2%              |
| GQA causal     | +1.0%   | +3.5% | −2.2%              |

Net: small positive on GQA (consistent with free overlap from PDL),
neutral on MHA modulo measurement noise at one mid-seqlen point. No
reproducible regression at scale.

## Test plan
- [ ] `pytest tests/cute/test_flash_attn.py -k "headdim_256 and not fwd"`
- [ ] hd=256 bwd benchmark before/after on B200, one mid-seqlen
```

---

## Notes for the author (do not paste into the PR body)

- Author of `e3cdf9f` was amended from `root@<container>` to
  `Johnsonms <lizhaofu@gmail.com>` before push.
- The "Performance" section is now measured (rep=50, clock-locked B200,
  `agent_space/bench_pr_delta.sh`). Raw outputs in
  `/tmp/fa4_pr_{before,after}_{mha,gqa}.txt`. The MHA-causal mid-seqlen
  noise is real run-to-run on this hardware — flag it honestly rather
  than try to bench it away.
- The dual-stream experiment is **not** in this PR. It is stashed locally
  on this branch (`stash@{0}: dual-stream wip (pdl-hd256-bwd)`).
