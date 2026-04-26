# Handoff to next Claude Code session — 2026-04-26

You are picking up the FA4 hd=256 dKdV backward TMA bulk store optimization
on B200. The previous session completed the kernel work and validation;
**only Task 3.3 (PR + handoff blob) remains.**

## Read these first (in order)

1. This doc — high-level status and what to do next.
2. `agent_space/ncu_hd256/_recovery/README_RECOVER.md` — restore steps if not already done.
3. `agent_space/ncu_hd256/VARIANT_3A_TASKS.md` — full task contract + result log.
4. `agent_space/ncu_hd256/LESSONS_VARIANT_3A.md` — design rationale, dead-ends, gotchas.
5. `~/.claude/projects/-workspace/memory/MEMORY.md` — your persistent memory index.

## TL;DR — where we are

Two branches pushed to `fork = github.com/Johnsonms/flash-attention`:

| branch | tip | role |
|---|---|---|
| `hd256-bwd-epilogue-refactor` | `d3747c0` | **PR-ready** — kernel code, 2 commits |
| `hd256-bwd-epilogue-refactor-history` | `da0561b` | pre-squash 5-step chain (backup) |
| `hd256-bwd-epilogue-refactor-recovery` | `f9b2c67` | recovery snapshot (this) |
| `pdl-hd256-bwd` | `4819cf2` | unrelated PDL fix PR (already open) |

**Working tree on `hd256-bwd-epilogue-refactor` should be clean.** 8/8 PASS:
```bash
CUDA_VISIBLE_DEVICES=0 python3 agent_space/ncu_hd256/correctness_check.py
# expect: All cases passed.
```

## Headline numbers (already in `REPORT_after.md` + `BENCH_after.md`)

**ncu (B200, vs baseline `4819cf2`)**:
- dkdv excessive sectors: 2 818 M → **0** (-100%)
- A_dkdv SM Throughput: 72.1% → **85.65%** (+13.5pp)
- A_dkdv cycles/inst: 16.2 → **13.45** (-17%)
- dq kernels unchanged (refactor deliberately didn't touch dq).

**Wallclock (rep=50, B200 @ 1755 MHz, vs `1748c9a`)**:

| config | median ΔTF% | best | worst |
|---|---:|---:|---:|
| MHA non-causal | +5.4% | +23.2% (4K) | +0.9% (131K) |
| **MHA causal** | **+10.6%** | **+38.9%** (4K) | +2.1% (131K) |
| GQA non-causal | +1.0% | +4.4% (8K) | +0.3% (131K) |
| GQA causal | +1.5% | +3.7% (8K) | -2.3% (16K, in noise band) |

## Task 3.3 — what to do

**Goal**: open a new PR for `Johnsonms:hd256-bwd-epilogue-refactor` → `Dao-AILab:main` and hand the user a PR description blob.

You do NOT have `gh` in this env — the user will paste your blob into GitHub manually.

### Steps

1. Confirm push state:
   ```bash
   git fetch fork  # or `origin` depending on remote name
   git log --oneline main..fork/hd256-bwd-epilogue-refactor
   # Expect 2 commits: 1748c9a (loop-reorder) + d3747c0 (Variant 3a)
   ```

2. Re-run the validation gate:
   ```bash
   CUDA_VISIBLE_DEVICES=0 python3 agent_space/ncu_hd256/correctness_check.py
   ```

3. Write `agent_space/ncu_hd256/PR_UPDATE_VARIANT_3A.md` containing the PR description blob. Sections:
   - **Title**: `[FA4][hd256] dKdV backward: TMA bulk store epilogue + LSE/dpsum coalesce`
   - **Summary** (3-5 bullets): what was done, why (per-thread store was the #1 dkdv bottleneck per ncu), how (CTA-shared SMEM with cooperative WG writes; per-WG TMA was structurally infeasible due to interleaved t2r N coverage at hd=256).
   - **Wallclock bench table** — copy from `BENCH_after.md` (4 configs × 6 seqlens, plus the per-config median summary).
   - **ncu summary** — copy the headline deltas from `REPORT_after.md` (excessive sectors, SM TP, cycles/inst).
   - **Design notes** — short version of `LESSONS_VARIANT_3A.md`'s "killer constraint" + "what works" sections. Mention varlen falls back to `self.store` (mirrors `flash_bwd_sm100.py:3837`).
   - **Test plan**:
     - `[ ] CUDA_VISIBLE_DEVICES=0 python3 agent_space/ncu_hd256/correctness_check.py` → 8/8 PASS
     - `[ ] Existing pytest hd=256 backward tests pass` (run a quick `pytest tests/cute/test_flash_attn.py -k "hdim=256 and bwd"` or whatever the upstream invocation is).

4. Hand the markdown blob to the user. The user opens the PR via the GitHub UI using the URL the push prints (`https://github.com/Johnsonms/flash-attention/pull/new/hd256-bwd-epilogue-refactor`).

5. Update `agent_space/ncu_hd256/VARIANT_3A_TASKS.md` Codex result log with the 3.3 entry. Update memory accordingly.

## Conventions to honor

- **Branch hygiene**: `hd256-bwd-epilogue-refactor` is PR-ready. Don't push more commits to it without explicit user approval. The history backup branch and recovery branch are out-of-band — don't merge them into the kernel branch.
- **agent_space is gitignored** on `hd256-bwd-epilogue-refactor`. You can edit files in `agent_space/` freely; they stay as untracked working-tree files. To persist agent_space changes across containers, push them to the recovery branch (see existing recovery commits for the pattern).
- **Author identity for commits** (no global git config — pass per-command):
  ```bash
  git -c user.name="Johnsonms" -c user.email="lizhaofu@gmail.com" commit ...
  ```
- **Validation gate**: run `correctness_check.py` after any code change. 8/8 PASS = ship; anything less = stop, do not commit.
- **User style**: Together.ai kernel engineer; terse and direct; reads PTX/SASS; doesn't need GPU/CUDA concepts re-explained. Skip the preambles.

## What this work is NOT

- **`pdl-hd256-bwd` PR is separate.** That PR carries only the small PDL `griddepcontrol_wait` fix (commits `aca9620` + `4819cf2`). Do not bundle anything from variant 3a into it.
- **dq_kernel was deliberately untouched.** It still has the line-2143 hotspot (95.5% of its 70 MB residual excess). That's the next opportunity if/when someone extends this refactor — out of scope for the current PR.

## Key files (working tree, gitignored on kernel branch)

- `agent_space/ncu_hd256/VARIANT_3A_TASKS.md` — task contract / result log
- `agent_space/ncu_hd256/LESSONS_VARIANT_3A.md` — design rationale + dead-ends
- `agent_space/ncu_hd256/REPORT.md` — original ncu baseline
- `agent_space/ncu_hd256/REPORT_after.md` — ncu re-profile (Task 3.1)
- `agent_space/ncu_hd256/BENCH_after.md` — rep=50 wallclock A/B (Task 3.2)
- `agent_space/ncu_hd256/correctness_check.py` — the 8-case validation gate
- `agent_space/ncu_hd256/bench_variant3a.sh` — A/B bench script (rep=50)
- `agent_space/ncu_hd256/profile.sh` + `per_line.py` + `repro.py` — ncu tooling
- `agent_space/ncu_hd256/_recovery/` — this directory; memory snapshot + stash patches + this handoff

## Stashes (preserved as patches; usually not needed)

- `_recovery/stash_0_2.5_partition_layout.patch` — variant 3a 2.5 dead-end attempt #5 (compound mode-2 slicing).
- `_recovery/stash_1_2.5_multiple_approaches.patch` — variant 3a 2.5 dead-ends #1-4 consolidated.
- `_recovery/stash_2_dual_stream_wip.patch` — old `pdl-hd256-bwd` dual-stream wip (verdict: net-flat with MHA-causal-8K -6.6%).

These are dead-ends superseded by `d3747c0`. Apply only if you want to inspect the failed approaches; not needed for forward progress.

## Sanity checklist before declaring 3.3 done

- [ ] `git log --oneline main..fork/hd256-bwd-epilogue-refactor` shows exactly 2 commits.
- [ ] 8/8 validation gate PASS.
- [ ] `agent_space/ncu_hd256/PR_UPDATE_VARIANT_3A.md` written and ready to paste.
- [ ] User has the URL to open the PR.
- [ ] `VARIANT_3A_TASKS.md` Codex result log updated.
- [ ] Memory updated to reflect "PR opened" state.
