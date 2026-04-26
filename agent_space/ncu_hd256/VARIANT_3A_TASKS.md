# Variant 3a Refactor — Task Handoff Doc

This is the contract for the FA4 hd=256 backward dkdv epilogue refactor on the `hd256-bwd-epilogue-refactor` branch (split off from `pdl-hd256-bwd` on 2026-04-26 — the PDL correctness fix stays on `pdl-hd256-bwd`, the larger refactor lives here). Codex implements + tests one task at a time per this doc; once Codex declares done, the integrating agent (Claude) verifies + commits + pushes. **Do not deviate from this workflow without an explicit instruction from the human.**

---

## Workflow

1. **Human** picks a task ID (e.g., "do task 2.4") and forwards it to **Codex**.
2. **Codex** reads this doc end-to-end (especially the conventions and the named task), implements the change, and runs the validation gate.
3. **Codex** writes a short result block at the bottom of this file under `## Codex result log` with: task ID, outcome (PASS / FAIL / BLOCKED), summary (3-6 lines), and any new artifacts produced (file paths). Codex does NOT commit, push, or update memory.
4. **Codex** returns to **Human** with: "task X done — please ask Claude to verify + integrate."
5. **Human** asks **Claude** to verify and integrate.
6. **Claude** does an independent verification (re-runs the validation gate, sanity-checks the diff, reads the result block), commits per the conventions below, pushes when Phase 2 chain is complete or when explicitly instructed, and updates the project memory file.

If anything is ambiguous, **stop and ask the human** rather than guess. A 30-second clarification beats a half-day on the wrong path.

---

## Conventions

### Branch + remote

- Working branch: `hd256-bwd-epilogue-refactor` (off `main`). The `pdl-hd256-bwd` branch holds the unrelated PDL correctness fix only and should NOT be touched from this workflow.
- Remote: `fork = https://github.com/Johnsonms/flash-attention.git`. Upstream `origin = Dao-AILab/flash-attention.git`.
- Tip at start of this doc: `a8315ba` (local only — not yet pushed to fork).
- Codex must NOT push. Claude pushes after Phase 2 chain validates.

### Validation gate

After EVERY code change in this doc:
```bash
CUDA_VISIBLE_DEVICES=0 python3 agent_space/ncu_hd256/correctness_check.py
```
Expected: `All cases passed.` (8 cases: MHA + GQA × causal/non-causal × seqlens 256/1024/4096).

If 8/8 does not pass, the task is FAIL — do not continue, do not commit, write the failure into the result log and stop.

### Commit conventions (when Claude integrates)

- Style: `[FA4][hd256] <imperative summary>`. For the variant 3a chain, prefix the summary with `Variant 3a (N/5):` to keep the chain readable. Body explains what changed and why.
- Author identity (no global git config — pass per-command):
  ```bash
  git -c user.name="Johnsonms" -c user.email="lizhaofu@gmail.com" commit -am "..."
  ```
- One commit per task. Never combine tasks into one commit.
- Never `git checkout` / `git reset` / `git stash drop` a file that has uncommitted work. A prior session lost 213 lines this way. If you need a clean slate for a bisect, use `git stash push -m "bisect: ..."` and pop afterward.

### Hard scope rules

- Modify only `flash_attn/cute/sm100_hd256_2cta_fmha_backward_dkdvkernel.py` (unless a task explicitly says otherwise).
- Do NOT touch `dq_kernel`. Variant 3a for dQ is a future PR.
- Do NOT touch `flash_bwd_sm100.py`, `flash_fwd_sm100.py`, or any file outside `flash_attn/cute/sm100_hd256_2cta_fmha_backward_dkdvkernel.py` and `agent_space/ncu_hd256/*`.
- Do NOT bump `num_regs_compute` past 192 without flagging the SM occupancy risk. Check via `min_blocks_per_mp` in the launch.

### Reference files (read for context, do not modify)

- `agent_space/ncu_hd256/REPORT.md` — original ncu bottleneck analysis.
- `agent_space/ncu_hd256/EPILOGUE_REFACTOR_DESIGN.md` — variant 3a design + SMEM-budget audit.
- `agent_space/ncu_hd256/HANDOFF_5d_DEBUG.md` — prior session's bisect that surfaced the suggestion-#3 r2s atom approach; also where the partition_D_position_independent mystery is documented.
- `flash_attn/cute/flash_bwd_sm100.py:3793-3956` — the proven SM100 hd=128 dKdV TMA-store epilogue. The variant 3a refactor mirrors this.

---

## Overall Goal

Replace the per-thread global stores at the end of `dkdv_kernel.epilogue` with a TMA bulk store path. Per ncu, the per-thread stores are the #1 bottleneck on hd=256 dkdv (71-96% of all "uncoalesced global access" excess sectors). ncu projection: +5-15% wallclock once the L1TEX-stall fraction drops.

---

## Phase summary

### Already done (DO NOT REDO — for context only)

| ID  | Commit    | Summary                                                                                                  |
|-----|-----------|----------------------------------------------------------------------------------------------------------|
| 0.1 | local     | Drop orphaned dq plumbing diff (was paired with lost variant 3a wiring).                                 |
| 0.2 | n/a       | 8/8 correctness verified from clean HEAD.                                                                |
| 0.3 | n/a       | Memory checkpoint at `~/.claude/projects/-workspace/memory/project_pdl_hd256_bwd.md`.                    |
| 1.1 | external  | PR opened on GitHub for `Johnsonms:pdl-hd256-bwd` (handled out-of-session by user).                      |
| 1.2 | `1748c9a` | Pop loop-reorder LSE/dpsum stash, validate 8/8, commit. ncu sectors -28.6%.                              |
| 2.1 | `d59a276` | Variant 3a (1/5): build `tma_atom_dK/dV` + `sdK/V_epi_layout` in `__call__`; thread args through call chain. No behavior change. |
| 2.2 | `2567955` | Variant 3a (2/5): build SMEM staging views `s_epi_dK/dV` aliased onto sP+sdST; per-WG slice via stage axis. Unused. |
| 2.3 | `ccfcc2a` | Variant 3a (3/5): build separate r2s tiled_copy (CopyUniversalOp 128 bits, thr (64,2)×val (1,8)) + `partition_D` per-WG SMEM view. Unused. |
| 2.4 | `a8315ba` | Variant 3a (4/5): reg→SMEM staging (bf16 cast + r2s copy + fence_view_async_shared + per-WG named barrier id 5+wg_idx) before each self.store. Key fix: **dV stages through sdOT, dK stages through sP+sdST** — using sdST for dV corrupts dK because the dK MMA is still reading sdST when dV epilogue begins. sdOT is dead post-dV-MMA. |
| 2.5 | `da0561b` | Variant 3a (5/5): TMA bulk store via **CTA-shared SMEM** (4 stages of (64, 64) aliased onto sP+sdST). Per-thread t2r N coverage is interleaved across full per-CTA hd, so per-WG TMA isn't viable — both WGs cooperatively populate one (64, 256) virtual SMEM via per-element indexed stores using `tTR_cdV_local` (cdV without domain offset), inter-WG barrier (256 threads) on bar 5/6, then WG 0 leader warp fires 4 `cp_async_bulk_commit_group` boxes mapping SMEM stages 0..3 → GMEM (64, 64) slices. Varlen falls back to self.store. 8/8 PASS. |

(Original SHAs pre-branch-split on `pdl-hd256-bwd`: `f2fa7ea`, `8fbc273`, `5bd8afd`, `71f7afa`. Rebased onto main 2026-04-26 to form `hd256-bwd-epilogue-refactor`.)

Branch is local-only; nothing pushed yet. Claude will push after Phase 2.5 lands.

### Remaining (this doc)

| ID  | Type   | Title                                                                       |
|-----|--------|-----------------------------------------------------------------------------|
| 3.1 | report | ✅ **DONE** — `REPORT_after.md`. dkdv excess sectors 2.82B→0; SM TP 72.1→85.65 (+13.5pp); cycles/inst -17% on A_dkdv. |
| 3.2 | report | ✅ **DONE** — `BENCH_after.md`. MHA causal +10.6% median; MHA non-causal +5.4% median; GQA +1-1.5% median. Best: MHA-causal-4K +38.9%. No reproducible regressions (one -2.3% at GQA-causal-16K within ±20 TF noise). |
| 3.3 | report | Push variant 3a chain + write PR-update markdown blob.                      |

---

## Task 2.4 — reg → SMEM copy + per-WG named barrier (self.store kept as final overwrite)

### What to do

In `epilogue()` (file: `flash_attn/cute/sm100_hd256_2cta_fmha_backward_dkdvkernel.py`), add the reg→SMEM staging step for both dV and dK paths. **Do not** remove the existing `self.store(tTR_g*, tTR_r*, tTR_c*, (K, D))` calls — they remain the visible GMEM write, so bytes-on-the-wire are unchanged. The SMEM contents are intentionally unread.

For each output (dV first, then dK), the sequence is:

1. (Existing) `cute.copy(tiled_t2r_*, tTR_t*, tTR_r*)` — TMEM → reg fp32.
2. (Existing for dK only) Scale loop `tTR_rdK[i] *= scale_softmax`.
3. **NEW**: bf16 reg fragment + cast:
   ```python
   tTR_r*_cast = cute.make_rmem_tensor(tTR_r*.shape, d*.element_type)
   tTR_r*_cast.store(tTR_r*.load().to(d*.element_type))
   ```
4. **NEW**: reinterpret iterator with the r2s SMEM partition's shape:
   ```python
   tTR_r*_r2s = cute.make_tensor(tTR_r*_cast.iterator, td*_sd*_r2s.shape)
   ```
5. **NEW**: reg → SMEM copy:
   ```python
   cute.copy(thr_copy_r2s_*, tTR_r*_r2s, td*_sd*_r2s)
   ```
6. **NEW**: shared-memory ordering fence:
   ```python
   cute.arch.fence_view_async_shared()
   ```
7. **NEW**: per-WG named barrier:
   ```python
   cute.arch.barrier(barrier_id=5+wg_idx, number_of_threads=128)
   ```
   Bar IDs 5/6 are free — 0-4 are reserved per `__init__`.
8. (Existing) `self.store(tTR_g*, tTR_r*, tTR_c*, (K, D))`.

Reference: `flash_bwd_sm100.py:3937-3941` (the `# RMEM -> SMEM -- copy, fence and barrier` block).

### Known blocker (must be solved in this task)

A naive copy of the above silently **corrupts dK** (dV stays correct). Bisect from session 2026-04-26 confirmed:

- Disable BOTH dV and dK r2s sequences → 8/8 PASS.
- Enable BOTH r2s sequences → dK fails (max_abs ~4 vs expected ~0.015), dV correct.
- Disable dK r2s only, keep dV r2s → dK STILL fails. (i.e., dV r2s alone breaks dK.)
- Disable just the barrier+fence in dV (keep `cute.copy(thr_copy_r2s_dV, ...)`) → dK STILL fails. (i.e., the r2s copy itself is the corruption source, not the barrier.)
- Element counts verified: per-thread = 64 elements, per-WG SMEM slot = 8192 elements.
- SMEM cosize verified: `sdK_epi_layout` cosize = 32768 bytes = exactly sP (16384) + sdST (16384) — no overrun.

**Strongest hypothesis**: register aliasing under pressure. `num_regs_compute = 128` (line ~178). The bf16 fragment `tTR_rdV_cast` (~32 32-bit regs) likely aliases part of the registers `tTR_rdK` later occupies. `tTR_rdV_cast.store(...)` corrupts those regs; the dK TMEM load may not fully overwrite them due to partition-vs-storage layout, so `self.store(tTR_gdK, ..., tTR_rdK, ...)` reads partial garbage.

**Suggested fix order** (try in sequence — first one that gives 8/8 PASS, ship it):

1. **Verify hypothesis with cute.printf** (then remove printfs before commit). Add `cute.printf` (gated on `tidx == 0`) to dump `tTR_rdK[0]`, `tTR_rdK[31]`, `tTR_rdK[63]` immediately after the dK TMEM load and immediately before `self.store(tTR_gdK, ...)`. If values change between the two points, regs are getting corrupted.
2. **Bump `self.num_regs_compute`** from 128 to 168 (or 192 if 168 is insufficient — 192 may hit SM occupancy). Re-run 8/8. If passes, accept and document the trade-off in the commit body. **Verify SM occupancy did not collapse** (check `min_blocks_per_mp=1` is still satisfied — if the kernel fails to launch at 192, back off).
3. **Cleaner restructure**: extract the per-output store sequence into a helper:
   ```python
   def _epi_store_one(self, tTR_t, tTR_r, tTR_g, tTR_c, sd_per_wg, td_sd_r2s,
                      thr_copy_r2s, tiled_t2r, dtype, K, D, per_wg_bar_id):
       cute.copy(tiled_t2r, tTR_t, tTR_r)
       # (caller already did the scale on tTR_r if needed)
       tTR_r_cast = cute.make_rmem_tensor(tTR_r.shape, dtype)
       tTR_r_cast.store(tTR_r.load().to(dtype))
       tTR_r_r2s = cute.make_tensor(tTR_r_cast.iterator, td_sd_r2s.shape)
       cute.copy(thr_copy_r2s, tTR_r_r2s, td_sd_r2s)
       cute.arch.fence_view_async_shared()
       cute.arch.barrier(barrier_id=per_wg_bar_id, number_of_threads=128)
       self.store(tTR_g, tTR_r, tTR_c, (K, D))
   ```
   Call it twice — once for dV, once for dK. Each call has its own register-lifetime scope. This mirrors how `flash_bwd_sm100.py:3315-3354` calls `epilogue_dK_or_dV_tma` separately for each output.
4. **Last resort**: drop the barrier inside the dV section (rely on the existing `cute.arch.fence_view_async_tmem_load()` between dV and dK). The barrier only matters once TMA is wired (task 2.5). Add it back in 2.5.

### Pass criteria

- `CUDA_VISIBLE_DEVICES=0 python3 agent_space/ncu_hd256/correctness_check.py` → **All cases passed.**
- No new files created. Only `flash_attn/cute/sm100_hd256_2cta_fmha_backward_dkdvkernel.py` modified.
- Diff vs `71f7afa`: roughly +30 to +50 lines if helper-function approach, or +1 line + the new sequence if just the `num_regs_compute` bump fixed it.
- Codex log entry written at the bottom of this file.

### Out of scope

- TMA bulk store (that's 2.5).
- ncu / wallclock benchmarks (that's 3.1 / 3.2).
- Touching `dq_kernel`, `flash_bwd_sm100.py`, or any other file.

---

## Task 2.5 — Replace per-thread store with TMA bulk store

### Precondition

Task 2.4 must be committed (Claude will commit after verification). Branch should be at `<2.4-commit>` with 8/8 PASS.

### What to do

Replace the `self.store(tTR_g*, tTR_r*, tTR_c*, (K, D))` calls (one per output) with the TMA bulk store path. The reg→SMEM step from 2.4 stays; only the *visible store* changes.

For each output, after the SMEM staging barrier from 2.4, do:

1. **Build the per-WG GMEM TMA tile.** Pattern from `flash_bwd_sm100.py:3839-3845`:
   ```python
   # Rebind d*_tma to a 3D layout (K, hd, HB) — the TMA tensor is rank-4 with
   # batch axis. Non-arithmetic make_tensor is safe on TMA-tensor iterators.
   md*_tma_3d = cute.make_tensor(
       d*_tma.iterator,
       cute.make_layout((K, self.cta_tiler[2], HB), stride=d*_tma.stride),
   )
   gd*_tma = cute.local_tile(md*_tma_3d, (self.cta_tiler[1], self.cta_tiler[2]),
                             (None, None, None))
   gd*_tma = gd*_tma[None, None, blk_coord_k, 0, blk_coord_batch]
   wg_split_tile = (self.cta_tiler[1], self.cta_tiler[2] // num_warp_groups)
   gd*_tma = cute.logical_divide(gd*_tma, wg_split_tile)
   gd*_tma = gd*_tma[None, (None, wg_idx)]
   gd*_tma_epi = cute.local_tile(gd*_tma, epi_tile_dKV, (0, None))
   ```
   Where `epi_tile_dKV = (self.cta_tiler[1], self.cta_tiler[2] // num_warp_groups)`.

2. **Build TMA partitions** (mirrors `flash_bwd_sm100.py:3869-3875`):
   ```python
   td*sd*_tma, td*gd*_tma = cpasync.tma_partition(
       tma_atom_d*,
       0,                       # no multicast
       cute.make_layout(1),
       cute.group_modes(sd*_per_wg, 0, 2),
       cute.group_modes(gd*_tma_epi, 0, 2),
   )
   ```

3. **Issue TMA from leader warp only** (mirrors `flash_bwd_sm100.py:3944-3956`):
   ```python
   leader_warp = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) == 0
   if leader_warp:
       cute.copy(tma_atom_d*, td*sd*_tma, td*gd*_tma[None, 0])
       cute.arch.cp_async_bulk_commit_group()
   cute.arch.cp_async_bulk_wait_group(0, read=True)
   ```

4. **Remove the `self.store(tTR_g*, tTR_r*, tTR_c*, (K, D))` call.**

### Known caveats

- **Varlen path**: `flash_bwd_sm100.py:3837` asserts `not seqlen.has_cu_seqlens_k, "varlen uses non tma store path"`. The dkdv kernel signature accepts `cumulative_s_q` / `cumulative_s_k`. **Gate the TMA path on `not varlen`** and keep `self.store(...)` as the varlen fallback. `varlen` is computed at the top of `dkdv_bwd` — thread it through to `epilogue` if not already (it currently is via `compute()`'s `varlen: bool` arg, but `epilogue` doesn't receive it — add the arg + pass it from `compute`'s epilogue call).

- **TMA tensor 3D rebind** (step 1): non-arithmetic `make_tensor` is safe on TMA-tensor iterators (only `iterator + offset` failed in earlier sessions). Mirror exactly the pattern above.

- **Last barrier**: after `cp_async_bulk_wait_group(0, read=True)`, all threads in the WG have finished consuming SMEM. The SMEM staging region (sP+sdST alias) is free for the next iteration. No extra barrier needed — sP is dead post-epilogue.

### Pass criteria

- 8/8 correctness pass.
- Diff vs the post-2.4 commit: roughly +30 to +60 lines (the `tma_partition` setup + TMA issue + commit/wait). The `self.store(...)` lines should be replaced (not just commented out) for non-varlen, and kept as the `else` branch for varlen.
- Codex log entry at the bottom of this file.

### Out of scope

- Performance benchmarking (3.1 / 3.2).
- Varlen TMA support (defer; keep self.store fallback).
- Touching dq_kernel or other files.

---

## Task 3.1 — ncu re-profile, confirm sector reduction

### Precondition

Tasks 2.4 + 2.5 committed by Claude.

### What to do

Run the ncu re-profile to verify the bottleneck moved.

```bash
cd /workspace/flash-attention
bash agent_space/ncu_hd256/profile.sh
python3 agent_space/ncu_hd256/per_line.py
```

If `profile.sh` is stale or fails, the canonical invocation is:
```bash
mkdir -p /tmp/ncu
CUTE_DSL_LINEINFO=1 CUTE_DSL_KEEP_PTX=1 ncu \
    --set full --import-source on --target-processes all \
    -f -o /tmp/ncu/A_dkdv_after \
    python3 agent_space/ncu_hd256/repro.py --config dkdv
```
(Adjust the repro.py args if it differs from the original; check by reading the file first.)

### Pass criteria

- Capture the new ncu summary into `agent_space/ncu_hd256/REPORT_after.md` with: date (2026-04-XX), commit hash of variant 3a tip, and these specific numbers:
  - Excessive global sectors total (was 2.82B before — see `REPORT.md`).
  - Excessive sectors attributable to the prior hotspot lines (`dkdvkernel.py:2694` per design doc — line numbers may have shifted post-refactor; report whatever the new attribution is).
  - Compute throughput SOL %.
  - L1TEX scoreboard dependency stall % (was 57-60% on dq, 31% on A_dkdv).
- **Goal**: excessive-sector attribution to the old hotspot lines drops to **<5%** of total excess.
- If the bottleneck did NOT move (still >20% on the hotspot lines), the refactor is wrong somehow — STOP and write a "Codex BLOCKED" entry in the result log.
- Codex log entry written.

### Out of scope

- Modifying any kernel code.
- Wallclock benchmarks (that's 3.2).

---

## Task 3.2 — rep=50 wallclock A/B vs `f2fa7ea` baseline

### Precondition

Tasks 2.4 + 2.5 committed by Claude.

### What to do

Wallclock benchmark dkdv (and dq, since they share a kernel boundary) at the variant 3a tip vs the pre-variant-3a baseline (`1748c9a` — the loop-reorder commit, first commit on `hd256-bwd-epilogue-refactor`).

**Baseline setup**: B200 GPU 0 with locked clocks at 1755 MHz. Check first if clocks are already locked:
```bash
nvidia-smi -i 0 --query-gpu=clocks.gr,clocks.max.gr --format=csv
```
If not locked: `nvidia-smi -i 0 -lgc 1755` (user has no-sudo path on this box).

**Bench grid**:
- hd=256, backward only.
- seqlens 4K, 8K, 16K, 32K, 65K, 131K (6 sizes).
- causal {true, false} (2).
- MHA (nheads=32, kv=32) and GQA (nheads=32, kv=2) (2).
- 24 rows total, rep=50.

**Reference scripts**:
```bash
ls agent_space/bench_pdl_delta.sh agent_space/bench_pr_delta.sh
```
These swap legs via `git stash` (since `agent_space/` is gitignored). Adapt as needed: A = `1748c9a`, B = variant 3a tip.

### Pass criteria

- Capture results into `agent_space/ncu_hd256/BENCH_after.md` with the row format used in `AI/PR_PDL_HD256_BWD.md`:
  | config         | median ΔTF | best  | worst |
  |----------------|-----------:|------:|------:|
  | MHA non-causal |        ... |   ... |   ... |
  | MHA causal     |        ... |   ... |   ... |
  | GQA non-causal |        ... |   ... |   ... |
  | GQA causal     |        ... |   ... |   ... |
- **Acceptance bands**:
  - Median win in [0%, +15%] across configs — ship.
  - No row regresses worse than -2% (excluding known measurement noise band: ±20 TFLOPS at MHA-causal mid-seqlens — see project memory).
  - If wallclock is flat (median ~0%) despite ncu sector reduction in 3.1, that's still acceptable to ship — the SMEM staging is the correct architecture, further wins may need concurrent improvements.
- If ANY row regresses ≥ 2% reproducibly (rep=50, two runs both negative): STOP, write "Codex BLOCKED" in the result log, escalate. Do NOT push.
- Codex log entry written.

### Out of scope

- Modifying any kernel code.
- Other configs (hd=64/128, fwd, varlen).

---

## Task 3.3 — Push variant 3a chain + write PR-update markdown blob

### Precondition

Tasks 3.1 + 3.2 reports written and acceptable.

### What to do

1. **Codex does NOT push.** Instead, write a markdown blob to `agent_space/ncu_hd256/PR_UPDATE_VARIANT_3A.md` containing:
   - A "Variant 3a epilogue refactor" section (3-5 sentences) describing the TMEM→SMEM(via r2s)→TMA bulk path and why it's the right fix per ncu.
   - The bench delta table from `BENCH_after.md`.
   - The ncu summary delta from `REPORT_after.md` (3-4 bullets: sector excess before/after, hotspot line attribution before/after, SOL % before/after).
   - A note: "Varlen path still uses per-thread self.store fallback (see flash_bwd_sm100.py:3837 for the same restriction at hd=128). TMA varlen support is deferred."
   - A test plan row: `[ ] CUDA_VISIBLE_DEVICES=0 python3 agent_space/ncu_hd256/correctness_check.py` → 8/8 PASS.

2. **Claude integrates** by:
   - Pushing local commits to `Johnsonms/flash-attention:hd256-bwd-epilogue-refactor` (a new PR will need to be opened by the human; this branch is independent of the `pdl-hd256-bwd` PR).
   - Handing the markdown blob to the human, who pastes it into the new PR description on GitHub (Codex / Claude do not have `gh` or a token in this env).
   - Updating project memory.

### Pass criteria

- `agent_space/ncu_hd256/PR_UPDATE_VARIANT_3A.md` exists and is well-formed.
- Codex log entry written.

### Out of scope

- Opening a NEW PR (the existing one stands).
- Merging.
- Updating the PR description directly (human handles this).

---

## Codex result log

After each task, append a result entry below in this format:

```
### Task X.Y — <date> — <PASS|FAIL|BLOCKED>

Working tree state at start: <commit hash>
Working tree state at end: <commit hash if same | "uncommitted, see diff" | "WIP, reverted">

Summary:
- Bullet 1 (what was done)
- Bullet 2 (what was tried, if relevant)
- Bullet 3 (any anomalies)

Validation gate result: <8/8 PASS | N/N FAIL | not run because ...>

New artifacts:
- agent_space/ncu_hd256/<filename>  (purpose)

Notes for Claude:
- <anything Claude needs to know during integration>
```

---

(append result entries below this line)

### Task 2.4 — 2026-04-26 — PASS

Working tree state at start: ccfcc2a
Working tree state at end: uncommitted, see diff

Summary:
- Added bf16 reg fragments, reg-to-SMEM r2s copies, shared fences, and per-WG named barriers before the existing dV/dK `self.store(...)` calls.
- Kept `num_regs_compute=128`; an intermediate 168-register experiment was discarded after an illegal-instruction failure.
- Fixed the documented dK corruption by staging dV through the already-consumed `sdOT` buffer, since dK MMA can still be reading `sdST` when dV epilogue starts.

Validation gate result: 8/8 PASS (`CUDA_VISIBLE_DEVICES=4,5,6,7 python agent_space/ncu_hd256/correctness_check.py`)

New artifacts:
- None

Notes for Claude:
- The requested GPU mask was 4-7; the script selects logical device 0, which mapped to physical GPU 4.
- Final code diff is limited to `flash_attn/cute/sm100_hd256_2cta_fmha_backward_dkdvkernel.py`; this log entry is the only doc change.

### Task 2.5 — 2026-04-26 — BLOCKED (Codex out-of-budget; Claude two debug rounds failed)

Working tree state at start: a8315ba
Working tree state at end: clean (a8315ba). All 2.5 work is in two stashes on `hd256-bwd-epilogue-refactor`:
- `stash@{0}` "WIP variant 3a 2.5: partition.layout + compound slicing attempt" (latest, attempt #5)
- `stash@{1}` "WIP variant 3a 2.5: TMA bulk store - multiple approaches tried, layout mismatch unresolved" (Codex's original through attempt #4)

**Summary**: 2.5 failed validation (all 8 cases FAIL) under several attempted designs. Root cause is a layout-shape mismatch between the reg fragment (full per-WG width) and the auto-derived r2s SMEM partition (per-stage width), and the manual r2s atoms tried instead don't match the t2r reg→position mapping. The 2.4 staging fix (sdOT for dV, sP+sdST for dK) is unchanged across attempts.

**Confirmed shapes (via cute.printf at the dV epilogue entry, hd=256 bf16)**:
- `tTR_rdV = ((32,1),1,2):((1,0),0,32)` — per-thread 64 fp32 elements, with mode 2 = 2 epi_stages of 32 elements each (stage-blocked: stage 0 is iter[0..31], stage 1 is iter[32..63]).
- `tdV_sdV_r2s = ((1,32),1,(2,2)):((0,1),0,(32,0))` — auto-derived (`make_tiled_copy_D(get_smem_store_op(...), tiled_t2r)`) partition over a (64, 64) per-WG SMEM slot. Per-thread 64 logical entries (32 unique × 2-broadcast). Mode 2 outer 2 stride 32 covers TWO (M, 32) sub-tiles within one (64, 64) SMEM stage.
- `tTR_tdV = (((32,32),1),1,2):(((1,65536),0),0,64)` — TMEM partition shows mode 2 size 2 = epi_stages, stride 64 (fp32 cols).
- `sdV_per_wg = ((8,8),(64,1)):((64,512),(1,0))` — (64, 64) per-WG SMEM slot.
- `gdV_tma_epi = (64,64,2):(1@1,1@0,64@0)` — 2 GMEM stages of (64, 64) TMA boxes.

**`cta_tiler[2] = 256` (full hd, NOT per-CTA — 2CTA splitting is internal). Per-WG hd width = 256/2 = 128 cols. For bf16, gcd(64, 128)=64 → num_epi_stages_dKV = 2.**

**Approaches tried, all failed (all 8 cases FAIL with dK/dV max_abs 0.4–12 vs expected 0.015)**:

1. **Codex's original**: multi-stage SMEM (`num_compute_wgs * num_epi_stages_dKV = 4` total stages of (64, 64)) + per-stage loop. Slicing `tTR_rdV_cast[None, 0, epi_stage]` gave wrong sub-fragment because the cast fragment doesn't have a 4-axis structure that this multi-index slicing assumes — only mode 2 of size 2 exists.

2. **Auto-derived atom + per-stage offset slicing**: `cute.make_tensor(cast.iterator + epi_stage*32, tdV_sdV_r2s.shape)`. Failed because `tdV_sdV_r2s` has 64 logical entries per thread (sized for both stages within one (64,64) SMEM via the (2,2) outer mode), not 32.

3. **Manual r2s atom** (CopyUniversalOp 128 bits, thr (64,2)×val (1,8), as in Phase 2.3) **+ per-stage offset slicing**. The manual atom’s thread→(M,N) mapping does NOT match t2r’s per-thread reg→(M,N) mapping. Data scrambles in SMEM. (Reference flash_bwd_sm100.py:587-599 uses thr (128,1)×val (1,8) which works there because tile_n=128 — does not directly port to our cta_tiler[1]=64.)

4. **Single-stage TMA with `epi_tile_dKV = (64, 128)` (full per-WG width, 256 B inner)**. Top-level `make_tiled_tma_atom` accepted it (assertion passed) but runtime hit `cudaErrorIllegalAddress`. Likely `make_smem_layout_epi` produces a swizzle pattern incompatible with a 256B inner-dim TMA box, OR the TMA box exceeds an SM100-specific limit not encoded in our assert.

5. **`make_tensor(reg.iter, partition.layout)` + compound mode-2 outer slicing `[None, None, (epi_stage, None)]`**. The hypothesis: `partition.layout` preserves stride-0 broadcasts, so passing it to `make_tensor` aliases reg elements correctly without OOB. Slicing `(epi_stage, None)` selects mode-2 outer (the stage) while keeping inner (the broadcast). Result: still 8/8 FAIL with same magnitudes (dK/dV max_abs 0.4–10). Conclusion: either the slicing semantics don't work as expected on compound modes, or there's a deeper issue with how the partition's mode-2 outer (stride 32) actually maps to SMEM addresses vs. how I interpreted it. Without docs/source for `cute.make_tiled_copy_D`, can't pin down the exact semantics.

**Additional findings from second debug session (Claude, 2026-04-26 evening)**:

- `flash_bwd_sm100.py:262` sets `self.dK_reduce_ncol = math.gcd(32, self.tile_hdim // 2)`. For hd=128, that's `gcd(32, 64) = 32`. So the reference's t2r atom uses **`Repetition(32)`, NOT `Repetition(epi_cols)`**. Our existing top-level was already correct in using Rep(32). Changing to Rep(epi_cols=64) collapses the per-stage axis (mode 2 size becomes 1 instead of num_epi_stages), which breaks per-stage slicing.
- The reference's `split_wg` (flash_bwd_sm100.py:2698) is more general than ours: handles rank-3/4 input, uses `cute.logical_divide` to introduce a split axis, and produces a result with rank ≥ 4 (extra axes intact). The reference then does `[None, None, 0, 0]` to drop redundant axes and `[None, epi_stage]` to slice per-stage. Our `split_wg` (line 40 of dkdvkernel.py) returns rank-3 only — slicing `[None, None, epi_stage]` works only if mode-2 size matches num_epi_stages, which it does for Rep(32) (we observed `tTR_rdV.shape = ((32,1),1,2)` mode 2 size 2 stride 32 ✓).
- The killer issue: with auto-derived `make_tiled_copy_D(get_smem_store_op(...), tiled_t2r)` on a (64, 64) per-WG SMEM slot, the **partition** has shape `((1,32),1,(2,2))` with strides `((0,1),0,(32,0))` — 128 logical entries per thread (mode-2 inner 2 broadcast via stride 0, mode-2 outer 2 covering two N-halves of the (64,64) slot). The reg fragment is `((32,1),1,2):((1,0),0,32)` = 64 actual elements per thread. `cute.make_tensor(reg.iterator, partition.shape)` uses *default compact strides* on the shape, so it expects 128 source entries (max offset 127), reading OOB. **`cute.make_tensor(reg.iterator, partition.layout)` with the partition's actual strides may fix this** — that's the next thing to try, since the partition's stride-0 broadcasts would alias multiple logical entries to the same source offset.
- The reference's manual atom (thr (128, 1) × val (1, 8)) gives a partition with NO broadcasts (per-thread 64 entries, all unique). Their `make_tensor(reg, partition.shape)` works because shape and source size match exactly. Adapting to our cta_tiler[1]=64 needs thr (64, 2) × val (1, 8) — but that doesn't match t2r's per-thread (M, N) coverage (the manual atom places thread T at (M=T//2, N=(T%2)*8) but t2r places thread T's data at TMEM positions determined by Ld32x32b, which is different).

**Phase A diagnostic findings (Claude, 2026-04-26 evening, with `cute.printf` instrumentation at the dV epilogue entry on a8315ba)**:

Per-thread t2r distribution within a WG (verified):

| tidx range | M | N coverage (per-CTA) |
|------|---|--------|
| 0..63 | tidx (0..63) | `{0..31, 64..95}` |
| 64..127 | tidx-64 (0..63) | `{128..159, 192..223}` |

Each thread reads **64 unique reg values** in 2 contiguous 32-block sub-ranges with stride 64 between them. Per WG covers **128 unique N positions interleaved across the FULL per-CTA hd=256** — `{0..31, 64..95, 128..159, 192..223}` for one WG, complementary set for the other.

This is fundamentally different from `flash_bwd_sm100.py`'s hd=128 case, where each WG owns a contiguous half of N. Our t2r distribution makes per-WG TMA bulk store **structurally impossible without redesign** — the per-WG SMEM data is interleaved across the full GMEM N range, so TMA boxes (which read contiguous SMEM and write contiguous GMEM) can't map per-WG → GMEM cleanly.

Per-thread r2s SMEM-coord targets (also verified):
- **Manual atom thr (64,2) val (1,8)**: thread 0 writes to N=`{0..7, 16..23, 32..39, 48..55, 64..71, 80..87, 96..103, 112..119}` — interleaved 8-stride within sub-blocks. Doesn't match t2r's `{0..31, 64..95}` reg layout.
- **Auto-derived atom**: thread 0 writes to N=`0..127` (all 128 positions, one per partition entry). Expects 128 unique reg values per thread; we only have 64. Source/dest size mismatch.
- Tested alternative thr_layouts `(64,2) ord (0,1)`, `(1,128) ord (1,0)`, compound `((4,16), 2)` with various val_layouts — none expose the inhomogeneous (32 + stride-64-jump + 32) atom-iter pattern that t2r produces. **`make_tiled_copy_tv`'s atom-iter axis is single, uniform-stride only.**

**Two viable paths forward (next session)**:

**Path 2 — CTA-shared (64, 256) SMEM with cooperative writes** (recommended):
- Top-level: `make_smem_layout_epi(epi_tile=(64, 64), num_stages=4)` — 4 stages of (64, 64) = (64, 256) per-CTA SMEM = 32 KB. Same total SMEM as current; aliased onto sP+sdST.
- Both WGs write their interleaved 128-N coverage into the SAME SMEM (not per-WG slots). Inter-WG barrier between r2s and TMA.
- Per WG, multiple cute.copy calls (one per sub-block of 32 N) with simple atoms. WG 0 thread 0 writes regs[0..31] → SMEM(M=0, N=0..31), regs[32..63] → SMEM(M=0, N=64..95). WG 1 fills the gaps.
- After barrier: SMEM has full (64, 256) contiguous per-CTA data. TMA fires 4 (64, 64) boxes to 4 contiguous GMEM positions.
- Estimate: ~80-120 lines of new epilogue code. ~50% risk of silent corruption from WG synchronization bugs.

**Path 4 — change t2r distribution upstream**:
- Currently `Repetition(32)` produces interleaved per-WG N. A different Rep value or different cta_tiler shape might give contiguous per-WG N.
- BUT the TMEM tile and MMA tile structures cascade — changing t2r affects the whole MMA setup. Risk of breaking the production-validated 2.4 baseline.
- Estimate: needs deep CuTeDSL/SM100 expertise. Not recommended without docs.

**Diagnostic harness** (insert after `tTR_tdV = split_wg(...)`, gated `if tidx == 0:`):
```python
_csdV = cute.make_identity_tensor((self.cta_tiler[1], self.cta_tiler[2] // num_warp_groups))
cute.printf("[dbg] tTR_cdV.layout = {}", tTR_cdV.layout)
cute.printf("[dbg] manual r2s SMEM-coord = {}", thr_copy_r2s_dV.partition_D(_csdV).layout)
for _i in cutlass.range_constexpr(64):
    cute.printf("[dbg] tidx=0 tTR_cdV[{}] = {}", _i, tTR_cdV[_i])
```
Run with `CUDA_VISIBLE_DEVICES=0 python3 agent_space/ncu_hd256/correctness_check.py 2>&1 | grep dbg | sort -u`.

**What likely needs to happen for 2.5 to land** (older notes preserved below):

a. **Follow flash_bwd_sm100.py:3793-3956 more faithfully** — that reference does PER-STAGE TMEM partitioning inside the loop. Specifically, line 3893 builds a fresh `tmem_load_atom` with `Repetition(self.dK_reduce_ncol)` (dK_reduce_ncol = epi_cols), and the per-stage TMEM/reg fragments naturally have correct sizes for the auto-derived r2s atom. Our code has the same Repetition (32) at the top, but partitions only ONCE (full-tile) and tries to slice — that path is what fails.

b. **Reference's split_wg differs from ours** (flash_bwd_sm100.py:2698) — handles rank-3/4 inputs via `cute.logical_divide` and produces a multi-axis result. Ours produces rank-3. That difference may be why our `[None, None, 0, 0][None, epi_stage]` approach doesn't work.

c. **Alternative**: build a per-stage TMEM tensor by `cute.local_tile(tdVtdV, (cta_tiler[1], epi_cols_dKV), (0, None))` BEFORE partition_S — gives a per-stage TMEM with stage axis at the end, then slice per stage and partition_S each.

d. **`Repetition(N)` value: keep at 32, NOT epi_cols.** Tested empirically — `Repetition(64)` (= epi_cols) collapses the per-stage axis (`tTR_rdV.shape` becomes `((64,1),1,1)` with mode 2 size 1 instead of 2), so per-stage slicing becomes impossible. The reference uses `Repetition(dK_reduce_ncol = gcd(32, hd//2))` = 32 for hd=128, same value we already have.

e. **Possible angle**: dump the auto-derived `tiled_copy_r2s_dV` itself (not just the per-thread slice) and inspect what the per-stage SMEM positions actually are. The mode-2 outer 2 stride 32 might be M-axis (i.e., 32 M-positions per outer entry, with outer covering M=0..31 and M=32..63 of the (64, 64) slot), not N-axis. If so, the partition naturally writes ONE epi_stage worth of data using FULL 32-elem-per-thread regs across both outer entries — and we need to provide a 64-element reg that has stage-K data duplicated/replicated. Requires real understanding of `get_smem_store_op` internals.

f. **Pragmatic alternative**: skip the auto-derived atom entirely. Build a manual r2s atom whose thread→(M,N) layout EXACTLY matches what t2r produces. This requires running `cute.printf` on `tTR_t/r/c` for several thread indices to map out the t2r's per-thread (M,N) coverage, then constructing a manual `make_tiled_copy_tv` with matching `thr_layout`. Tedious but mechanical.

**Reference pattern to copy verbatim**: flash_bwd_sm100.py:3879 (sdKV per-WG slice); :3884 (`thr_copy_r2s.partition_D(sdKV)` — single per-stage SMEM); :3893-3895 (per-stage TMEM atom build); :3897-3919 (per-stage loop with TMEM partition + slice + cast + r2s + TMA + commit/wait). Their SMEM is `num_compute_wgs` stages (one per WG, reused). Their r2s atom is manual `make_tiled_copy_tv` with thr (128,1)×val (1,8) — but their tile_n is 128. For our tile_n=64, the equivalent thread distribution needs derivation — possibly thr (64, 2) × val (1, 8) IS correct but requires a t2r whose per-thread reg→position mapping aligns. That alignment is what to verify with cute.printf.

Validation gate result: 8/8 FAIL across all 5 attempted variants. Branch reverted to a8315ba (Phase 2.4 PASS state).

New artifacts:
- `stash@{0}` on `hd256-bwd-epilogue-refactor`: attempt #5 (Rep(32), auto-derived atom, partition.layout + compound mode-2 outer slicing).
- `stash@{1}` on `hd256-bwd-epilogue-refactor`: attempts #1-4 consolidated state.

Notes for Claude (next session):
- Start by `git stash show -p stash@{0}` to inspect the last attempt; do NOT pop and validate without first redesigning per-stage TMEM partition.
- The shape printf snippet (insert after `tdV_sdV_r2s = thr_copy_r2s_dV.partition_D(sdV_per_wg)`):
  ```python
  if tidx == 0:
      cute.printf("[shape] tTR_rdV={}", tTR_rdV.layout)
      cute.printf("[shape] tdV_sdV_r2s={}", tdV_sdV_r2s.layout)
  ```
  Use it whenever changing the t2r/r2s/SMEM shape to verify expectations.
- Branch tip is `a8315ba` (Phase 2.4) — 2.4 is the stable baseline; do not destabilize it while iterating on 2.5.
