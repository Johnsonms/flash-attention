---
name: For variant 3a kernel work, user uses Codex-implements + Claude-verifies workflow
description: User splits multi-task kernel refactors between Codex (implement + test) and Claude (verify + integrate + commit + push + memory). Hand-off contract lives in a markdown doc in `agent_space/`.
type: feedback
originSessionId: 9bcaee7b-ec78-4b93-9223-89afadb7c0f4
---
For the variant 3a refactor on `pdl-hd256-bwd` (and likely other multi-step kernel work that follows this pattern), the user collaborates by routing tasks between Codex and Claude:

- **Codex** reads the per-task contract, implements the change, runs the validation gate, and writes a result-log entry.
- **Codex does NOT commit, push, or update memory.**
- **Claude** verifies (re-runs gate, sanity-checks diff, reads result log), commits with the project's author identity, pushes when the chain is complete or when explicitly told, and updates project memory.
- **User** routes between the two and provides direction on which task is next.

**Why**: User explicitly chose this workflow at end of session 2026-04-26 after experiencing both (a) Codex-only sessions losing work via uncommitted diffs and (b) Claude-only sessions making slow progress on long implementations. The split lets each agent play to its strength.

**How to apply**:
- When the user says "I will ask Codex to do X, you verify and integrate", do exactly that — verify first (re-run validation gate independently), then commit per the project's conventions, then push if appropriate, then update memory.
- The handoff contract lives at a known path (e.g., `agent_space/ncu_hd256/VARIANT_3A_TASKS.md` for the variant 3a work). Codex appends results to a `## Codex result log` section at the bottom of that doc.
- If Codex's result log says PASS but verification fails, do NOT silently re-do the work — surface the discrepancy to the user.
- If Codex's result log says BLOCKED, read their notes carefully and surface to the user with a recommendation (don't try to plow through the same blocker yourself).
