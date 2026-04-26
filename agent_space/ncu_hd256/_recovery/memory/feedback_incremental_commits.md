---
name: Commit incrementally on multi-step refactors; never sit on a giant uncommitted diff
description: For kernel refactors that span many file regions, commit each step (validated 8/8 correct) before moving on. A `git checkout` once destroyed 213 lines of uncommitted variant 3a wiring.
type: feedback
originSessionId: 9bcaee7b-ec78-4b93-9223-89afadb7c0f4
---
For multi-step refactors (especially kernel work in `flash_attn/cute/`), commit each step as soon as it passes the project's correctness gate. Never sit on a giant uncommitted diff that spans many files / many regions of one file.

**Why**: 2026-04-26, a `git checkout` to "revert just my edits" destroyed 213 lines of variant 3a wiring that had been sitting uncommitted in the working tree from a prior session — never committed, never stashed, no IDE local-history backup. Unrecoverable. The user explicitly noted this lesson and asked to replan with incremental commits as a hard constraint.

**How to apply**:
- Before any non-trivial multi-region edit, plan it as N commits (e.g., 5 for variant 3a) where each commit leaves the working tree in a known-good state with the project's correctness gate green.
- Use `git stash push -m "bisect: ..."` (named) for temporary debugging, never bare `git checkout`.
- If you find yourself with >100 lines of uncommitted work that hasn't been validated, stop and commit the validated subset before continuing.
- If a debugging step turns up a regression, isolate the buggy step in its own commit so future bisect is trivial.
