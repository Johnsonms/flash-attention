# Container migration recovery snapshot — 2026-04-26

User is migrating containers. This `_recovery/` directory holds everything
that lives outside the git-tracked kernel source so a new Claude Code
session in the new container can resume cleanly.

## Branches at snapshot time

| branch | tip | role |
|---|---|---|
| `hd256-bwd-epilogue-refactor` | `d3747c0` | **PR-ready**: 2 commits = loop-reorder + Variant 3a (TMA bulk store epilogue). 8/8 PASS. |
| `hd256-bwd-epilogue-refactor-history` | `da0561b` | Pre-squash 5-step incremental chain (kept as backup). |
| `hd256-bwd-epilogue-refactor-recovery` | this commit | This branch — agent_space + memory dumps for cross-container recovery. |
| `pdl-hd256-bwd` | `4819cf2` (pushed) | Original PDL fix PR; unrelated to this work. |

Both kernel branches are pushed to `fork = github.com/Johnsonms/flash-attention`.

## What lives where

- `agent_space/ncu_hd256/`: all design docs, profile results, lessons, scripts.
  - `VARIANT_3A_TASKS.md`: source-of-truth handoff doc with full task log.
  - `LESSONS_VARIANT_3A.md`: post-mortem on what worked and what didn't.
  - `REPORT.md`: original ncu profile (baseline `4819cf2`).
  - `REPORT_after.md`: ncu re-profile at `d3747c0` (Task 3.1).
  - `BENCH_after.md`: rep=50 wallclock A/B at `d3747c0` (Task 3.2).
  - `EPILOGUE_REFACTOR_DESIGN.md`, `HANDOFF_5d_DEBUG.md`: older design docs.
  - `bench_variant3a.sh`: A/B bench script.
  - `correctness_check.py`: 8-case correctness harness (validation gate).
  - `profile.sh`, `per_line.py`, `repro.py`: ncu profile + analysis tooling.
- `agent_space/ncu_hd256/_recovery/`: this directory.
  - `memory/`: snapshot of `~/.claude/projects/-workspace/memory/` (Claude's persistent memory files).
  - `stash_*.patch`: git stash contents preserved as patches.
  - `README_RECOVER.md`: this file.

## Restoring in the new container

After cloning the fork in the new container:

1. **Restore memory**:
   ```bash
   mkdir -p ~/.claude/projects/-workspace/memory/
   cp agent_space/ncu_hd256/_recovery/memory/*.md ~/.claude/projects/-workspace/memory/
   ```

2. **Optional: restore stashes** (only if revisiting the dead-end paths is useful):
   ```bash
   git checkout hd256-bwd-epilogue-refactor
   for p in agent_space/ncu_hd256/_recovery/stash_*.patch; do
       git apply --3way "$p"
       git stash push -m "$(basename $p .patch)"
   done
   ```
   The Variant 3a 2.5 stashes (`stash_0`, `stash_1`) are dead-ends superseded
   by the final `d3747c0` commit. The `stash_2` dual-stream wip is from the
   `pdl-hd256-bwd` branch, also a dead-end (verdict: net-flat, see memory).
   You probably only need these for reference, not for actual recovery.

3. **Continue from Task 3.3** (the only remaining task in the variant 3a chain):
   - Push `hd256-bwd-epilogue-refactor` to fork (already done at snapshot time, just verify).
   - Have user open a NEW PR for `Johnsonms:hd256-bwd-epilogue-refactor` → `Dao-AILab:main`.
   - Hand the user a PR description blob assembled from `BENCH_after.md` + `REPORT_after.md` + a brief design summary.

   PR description scaffold:
   - Title: `[FA4][hd256] dKdV backward: TMA bulk store epilogue + LSE/dpsum coalesce`
   - Summary: 2 commits, +13.5pp SM TP and +10.6% median wallclock on MHA-causal hd=256.
   - Body: see `BENCH_after.md` for the wallclock table, `REPORT_after.md` for ncu deltas, `LESSONS_VARIANT_3A.md` for the design rationale.
   - Test plan: `CUDA_VISIBLE_DEVICES=0 python3 agent_space/ncu_hd256/correctness_check.py` → 8/8 PASS.

## Validation gate (re-confirm in new container)

```bash
CUDA_VISIBLE_DEVICES=0 python3 agent_space/ncu_hd256/correctness_check.py
```
Expect: `All cases passed.` (8 cases).

## Author identity for any new commits

The repo has no global git config; pass identity per-command:
```bash
git -c user.name="Johnsonms" -c user.email="lizhaofu@gmail.com" commit ...
```
