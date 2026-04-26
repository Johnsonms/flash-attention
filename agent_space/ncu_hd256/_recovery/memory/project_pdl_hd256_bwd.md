---
name: pdl-hd256-bwd PR + hd256-bwd-epilogue-refactor branch
description: Two-branch split 2026-04-26. `pdl-hd256-bwd` (PR open) carries only the PDL correctness fix. `hd256-bwd-epilogue-refactor` (off main, local, tip `d3747c0`) is squashed: 2 commits = loop-reorder + Variant 3a (TMA bulk store via CTA-shared SMEM). 8/8 PASS. Pre-squash 5-step chain preserved as `hd256-bwd-epilogue-refactor-history` (`da0561b`). Lessons doc: `agent_space/ncu_hd256/LESSONS_VARIANT_3A.md`. Remaining: 3.1 ncu, 3.2 bench, 3.3 push + new PR.
type: project
originSessionId: 5ce87721-bf30-40aa-8632-d0b628cc1cc5
---

## Branch split 2026-04-26

The single `pdl-hd256-bwd` branch was split into two so the small PDL fix can ship independently of the larger epilogue refactor:

- **`pdl-hd256-bwd`** (pushed to fork, PR open) — tip `4819cf2`. Contains only:
  - `aca9620` griddepcontrol_wait() in dq_kernel (correctness)
  - `4819cf2` PR notes/perf data
  - PR rewrites this branch on next push; `f2fa7ea` (loop-reorder) was force-pushed off this branch and is no longer on the PR.
- **`hd256-bwd-epilogue-refactor`** (local only, off main) — tip `da0561b`. Contains:
  - `1748c9a` Coalesce LSE/dpsum per-K-iter loads (was `f2fa7ea`)
  - `d59a276` Variant 3a (1/5): TMA atoms + signature threading (was `8fbc273`)
  - `2567955` Variant 3a (2/5): SMEM staging views (was `5bd8afd`)
  - `ccfcc2a` Variant 3a (3/5): r2s tiled_copy + partition_D (was `71f7afa`)
  - `a8315ba` Variant 3a (4/5): reg→SMEM staging + per-WG barrier (Codex 2026-04-26)
  - `da0561b` Variant 3a (5/5): TMA bulk store via CTA-shared SMEM (2026-04-26)

Workflow contract still lives at `/workspace/flash-attention/agent_space/ncu_hd256/VARIANT_3A_TASKS.md` (updated 2026-04-26 to point at the new branch + new SHAs). Codex implements one task; Claude verifies + commits + pushes; result log is at the bottom of that doc.

8/8 correctness verified on `hd256-bwd-epilogue-refactor` post-rebase. Working tree clean.

## Tasks pending in handoff doc

| ID  | Type   | Title |
|-----|--------|-------|
| 3.1 | report | ncu re-profile, confirm sector reduction; output to `agent_space/ncu_hd256/REPORT_after.md` |
| 3.2 | report | rep=50 wallclock A/B vs `1748c9a`; output to `agent_space/ncu_hd256/BENCH_after.md` |
| 3.3 | report | Write `agent_space/ncu_hd256/PR_UPDATE_VARIANT_3A.md`; Claude pushes new branch + hands blob to user (this is a NEW PR, separate from `pdl-hd256-bwd`) |

## Phase 2.5 design — CTA-shared SMEM with cooperative WG writes (LANDED at `da0561b`)

The fundamental obstacle was that per-thread t2r N coverage at hd=256 is interleaved across the FULL per-CTA hd (each WG owns 128 N positions distributed as 4 sub-blocks of 32 across 0..255), unlike the hd=128 reference where each WG owns a contiguous half. So per-WG TMA bulk store is structurally impossible. The fix:

1. **SMEM**: 4 stages of (64, 64) at top-level (= per-CTA (64, 256) virtual buffer aliased onto sP+sdST). Total = 32 KB, same as before.
2. **Per-element cooperative writes**: both warp-groups walk their per-thread `tTR_cdV_local` (cdV without the global domain offset) and write each value to `s_epi_dV[m, n%epi_cols, n//epi_cols] = v`. This honors the SMEM swizzle since s_epi_dV is built from a swizzle-aware iterator.
3. **Inter-WG barrier**: `cute.arch.barrier(barrier_id=5/6, number_of_threads=cta_threads=256)` separates writes from TMA reads.
4. **TMA**: WG 0 leader warp fires 4 `cp_async_bulk_commit_group` boxes per output, one per stage, mapping SMEM stage k → GMEM (M, N=k*64..(k+1)*64). Other threads call `cp_async_bulk_wait_group(0, read=True)` (no-op for them since they didn't issue).
5. **Varlen** falls back to per-thread `self.store` (mirrors flash_bwd_sm100.py:3837 restriction).

The per-element store path is correctness-first; could be vectorized later (e.g., via an explicit (M, N) → SMEM virtual layout passed through `partition_D`), but the layout/stride compatibility was tricky and per-element worked first try once `cdV_local` (no domain offset) was used.

## Phase 2.5 BLOCKED (2026-04-26) — what was tried, what to try next session

Codex hit budget mid-implementation; Claude tried **5** different approaches across two debug rounds, all failed 8/8 with dK/dV max_abs 0.4–12 (vs expected ~0.015). All attempts stashed (`stash@{0}` = round 2 attempt, `stash@{1}` = round 1 attempts).

Key shapes verified via `cute.printf` (hd=256 bf16):
- `tTR_rdV = ((32,1),1,2):((1,0),0,32)` — per-thread 64 fp32, mode 2 = 2 epi_stages of 32 elements (stage-blocked).
- `tdV_sdV_r2s = ((1,32),1,(2,2)):((0,1),0,(32,0))` — auto-derived partition over (64,64) per-WG SMEM. Mode 2 outer 2 stride 32 = covers TWO (M, 32) sub-tiles within one stage. Per-thread 64 logical entries.
- `cta_tiler[2] = 256` (full hd, 2CTA splitting is internal to the kernel — earlier note in memory said "per-CTA 128", that was WRONG). Per-WG hd width = 128 cols → num_epi_stages_dKV = 2 (not 1) for bf16.

Approaches that failed:
1. Codex's per-stage loop with `tTR_rdV_cast[None, 0, epi_stage]` slicing — wrong-rank slicing.
2. Auto-derived atom + offset slicing `cast.iterator + epi_stage*32` — partition expects 64 entries/thread, sliced source has only 32.
3. Manual r2s atom (thr (64,2)×val (1,8)) — thread→(M,N) doesn't match t2r mapping; data scrambles.
4. Single-stage TMA with `epi_tile_dKV=(64,128)` (full per-WG, 256B inner) — `cudaErrorIllegalAddress` (likely SMEM swizzle incompatibility).
5. (Round 2) `Repetition(epi_cols)` t2r — collapses per-stage axis (mode 2 size 1), then `make_tensor(reg, partition.layout)` + compound mode-2 outer slice `[None, None, (epi_stage, None)]` — still 8/8 FAIL same magnitudes. CuTeDSL slicing semantics on compound modes don't behave as I reasoned.

**Phase A diagnostic results (2026-04-26 evening, via cute.printf at a8315ba)**: per-thread t2r distribution is **interleaved across the FULL per-CTA hd=256, not contiguous per-WG**. tidx 0..63 covers M=tidx, N={0..31, 64..95}; tidx 64..127 covers M=tidx-64, N={128..159, 192..223}. Per WG → 128 N positions distributed as 4 sub-blocks of 32 across the whole hd. This is fundamentally unlike the hd=128 reference's per-WG-contiguous distribution, so per-WG TMA bulk store is structurally impossible without redesign. `make_tiled_copy_tv` API only supports uniform-stride atom-iter axes, so no manual atom layout can match t2r's compound (32 + stride-64-jump + 32) pattern.

**Path forward (next session)**: only Path 2 is viable — redesign as **CTA-shared (64, 256) SMEM** where both WGs cooperatively write into one big slot (multiple cute.copy calls per WG, one per 32-N sub-block, with inter-WG barrier). After barrier, TMA fires 4 (64, 64) boxes to 4 GMEM positions. Estimate ~80-120 lines of new epilogue code. Full diagnostic harness (cute.printf snippet) is in the handoff doc's Phase A findings section. KEEP `Repetition(32)` — `Repetition(epi_cols)` was tested and collapses the per-stage axis.

**Branch tip: `a8315ba` (Phase 2.4 stable, 8/8 PASS). Do not destabilize while iterating on 2.5.**

## Phase 2.4 resolution (2026-04-26, Codex)

The "dV r2s alone silently corrupts dK" blocker turned out to **not** be register aliasing. Actual root cause: dV was staging through `sP+sdST`, but `sdST` is still being read by the dK MMA when dV epilogue begins. Writing dV into sdST corrupts dK's input. Fix: dV stages through `sdOT` instead (consumed by the dV MMA, dead from the dV-epilogue point onward); dK continues to stage through `sP+sdST` since by the time dK epilogue runs, the dK MMA has completed.

Lesson durable across sessions: **when picking SMEM regions for epilogue staging, the region must be dead at the moment of write — including by any pipeline stage that's *concurrent with*, not just before, the staging write.** The 2.2 design doc ticked "dead post-epilogue" for sP+sdST, but missed that the dK MMA can still be reading sdST during the dV epilogue. EPILOGUE_REFACTOR_DESIGN.md SMEM audit was right about which regions are dead after the epilogue, but wrong about which are safe to write *during* the epilogue.

Also: an intermediate `num_regs_compute=168` attempt failed with illegal-instruction. The sdOT re-aliasing fix avoided the register-pressure path entirely; 128 retained.

## Conventions captured for verification

- Validation gate (run after every code change): `CUDA_VISIBLE_DEVICES=0 python3 agent_space/ncu_hd256/correctness_check.py` → expect `All cases passed.` (8 cases).
- Author identity for commits: `git -c user.name="Johnsonms" -c user.email="lizhaofu@gmail.com" commit -am "..."`. No global git config to set.
- Commit style: `[FA4][hd256] Variant 3a (N/5): <one-liner>`.
- Hard scope: only modify `flash_attn/cute/sm100_hd256_2cta_fmha_backward_dkdvkernel.py`. Never touch dq_kernel, flash_bwd_sm100.py, flash_fwd_sm100.py.
- Never `git checkout` / `git reset` a file with uncommitted work. A prior session lost 213 lines this way.

## stashes

- `stash@{0}` = dual-stream wip (deferred; net-flat verdict, MHA-causal-8K -6.6% regression). Lives on the fork only via the working repo state — not branch-attached.

## Earlier context (pre-split, retained for reference)

### Phase 2.4 first attempt (mid-session 2026-04-26) — BLOCKED, then resolved

Original blocker: dV r2s into `sdST` silently corrupted dK. Bisect at the time correctly identified that `cute.copy(thr_copy_r2s_dV, ...)` itself was the corruption source (not the barrier/fence), but the four documented hypotheses (register aliasing, num_regs_compute bump, helper restructure, drop barrier) all targeted register pressure as the suspected root cause.

Codex's later attempt confirmed register pressure was a red herring: a 168-reg variant failed with illegal-instruction. The actual fix was structural — dV stages through `sdOT` (already-consumed by dV-MMA), not `sdST` (still-being-read by dK-MMA). See "Phase 2.4 resolution" section above.

### PR perf bench (B200, clock-locked 1755 MHz, rep=50, main vs commit on `pdl-hd256-bwd`)

Long seqlens (≥65K) within ±1% per row; mid-seqlen MHA-causal has run-to-run noise of ±20 TFLOPS, the largest swing seen was −5.8% / +5.1% at adjacent points (32K / 65K) which neutralize. GQA shows small consistent positives (median +1%, peak +3.5%) — PDL overlap is working. No reproducible regression.

### Bench script gotchas

- `agent_space/bench_pdl_delta.sh` labels legs `BASELINE (no PDL)` vs `PDL`, but it's really `single-stream` vs `dual-stream` — `griddepcontrol_wait` is in both legs. Labels are misleading.
- Script uses `git stash` / `git stash pop` to swap legs — only works because `agent_space/` is gitignored.
