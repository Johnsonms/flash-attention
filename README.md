# Overnight SM90 ↔ SM100 parity work (2026-10-17)

Every PR candidate is a branch on `Johnsonms/flash-attention` based on upstream `main` 965c9212. Each has a description file here, ready to paste into the upstream PR. Nothing has been opened upstream: review first.

Priority: accuracy and performance fixes that are small and easy to merge.

| # | branch | type | status | description |
|---|---|---|---|---|
| 01 | `johnson/sm90-bwd-sdo-hdimv` | correctness (silent wrong dQ/dK + compile error) | pushed, tested on H100 | [PR-01](PR-01-sm90-bwd-value-dim.md) |

## Survey: SM100 work with an SM90 gap (from issues + merged/open PRs, 2026-10-17)

Already covered by an open SM90 PR (not duplicated): SplitKV #2415, dynamic persistent scheduler #2754, packed-varlen K/V NaN tail #3013, padded-hdim dQ #2896, block-sparse deterministic #2567, varlen block-sparse bwd #2759, hd256 bwd perf #2889, KV128 block-sparse hdim192 #2757, learnable-sink normalization #2944.

Open candidates (verified on H100 unless noted):
- SM90 bwd (64, 128): smem overflow, 248832 > 232448 B (config keyed on head_dim only).
- SM90 GQA bwd with head_dim != head_dim_v: hard assert (SM100 supports it, e.g. (192, 128) GQA).
- Test `hdim > 192 backward: SM90 not supported yet` xfail looks stale: hd256 MHA/GQA bwd passes on H100.
- dsink precision (SM100 #2969/#2971): SM90 run of `test_flash_attn_output` sink cases at d=128: 3 failed / 387 passed. To be triaged.
- Feature ports (larger): pack_gqa backward (#2610), kv_start document masking (#2985), dropout (#2597), FP8 forward.

Working notes live in each worktree's `agent_space/` (gitignored): hypothesis ledgers, evidence files.
