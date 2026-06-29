# Sprint log

A published, chronological record of how `trt-llm-langchain` evolved — one file per sprint — so
both the author and future readers can see **why** the project is shaped the way it is, not just
the final state. These are intentionally kept in version control.

See the living research and plan in [`../docs/`](../docs/):
- [`docs/01-research.md`](../docs/01-research.md) — why this is a thin client over an existing backend.
- [`docs/02-plan.md`](../docs/02-plan.md) — architecture, decisions, and the sprint roadmap.

## Convention

- One file per sprint: `sprint-NN-short-slug.md` (zero-padded, e.g. `sprint-01-manager.md`).
- Write the **Goal / Plan** at the start of the sprint, fill **What shipped / Outcomes /
  Follow-ups** as it lands. Keep it honest — record what didn't work and what got deferred.
- Date each sprint. Convert relative dates to absolute.
- Link to the commits/PRs and the files that changed.

Each sprint doc uses this skeleton:

```markdown
# Sprint NN — <title>

_Status: planned | in progress | complete · Started: YYYY-MM-DD · Landed: YYYY-MM-DD_

## Goal
One or two sentences: what "done" means for this sprint.

## Plan
The intended steps / deliverables.

## What shipped
What actually got built (files, behavior), with links.

## Decisions & discoveries
Choices made mid-sprint and anything learned that changed the plan.

## Outcomes
Did it meet the exit criteria? Evidence (commands run, output, tests).

## Follow-ups
Deferred items, new TODOs, things to verify later.
```

## Index

| Sprint | Title | Status |
|---|---|---|
| [01](sprint-01-manager.md) | Manager + connectivity | complete (live-verified) |
| [02](sprint-02-chat-trtllm.md) | `ChatTrtLlm` over `ChatOpenAI` | complete (live-verified) |
| [03](sprint-03-swap-lcel.md) | Model swapping + LCEL | complete (live-verified; restart-based swap) |
| [04](sprint-04-hardening.md) | Hardening, live tests, README | complete (live-verified) |
| [05](sprint-05-backend-contract.md) | Backend contract + integration decision (C) | complete |
| [06](sprint-06-resident-default.md) | Resident-model default + swap strategy (ADR 0002) | complete (live-verified) |
| [07](sprint-07-packaging.md) | Publish-ready packaging | complete |
| [08](sprint-08-getting-started.md) | Getting-started guide | complete (consume path live-verified) |
| [09](sprint-09-polish.md) | Post-publish polish (CI pins, structured output) | complete (live-verified) |
