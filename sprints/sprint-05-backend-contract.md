# Sprint 5 — Backend contract + integration decision

_Status: complete · Started: 2026-06-29 · Branch: `sprint-05-backend-contract`_

## Goal

Write the backend contract the client depends on, and decide (and record) how
`trt-llm-langchain` relates to the `trt-llm-explore` backend so the result is consumable by
others.

## What shipped

- **[`docs/03-backend-contract.md`](../docs/03-backend-contract.md)** — the exact HTTP surface a
  conforming backend must expose: the OpenAI chat surface (`:8003`, model key == OpenAI `id`,
  streaming), the KServe v2 control surface (`:8000`, EXPLICIT mode, index/load/unload/ready),
  the `{pipeline}_{key}` naming + vision/chat inference + decoupled-for-streaming rules, and the
  restart-based single-GPU swap constraint. Includes what a non-reference backend (incl. plain
  `trtllm-serve`) must do to conform.
- **[ADR 0001](../docs/decisions/0001-backend-integration-strategy.md)** — decision: **option C**
  (client-primary + contract + optional companion backend). Records the consequences, the
  upcoming backend **rename**, and the rename-resilience approach (refer to the backend by role;
  centralize its name).
- **Client doc updates for rename-resilience:** README now recommends
  `TRTLLM_RESTART_CMD="just -C <backend_dir> restart"` (stable across a rename) over a raw
  `docker restart <container>`; the contract doc and plan note the backend may be renamed and
  treat names/paths/containers as examples. `docs/02-plan.md` records the decision + links the ADR.
- **Backend generalization filed:** korg **WI #93** in the backend project — neutralize the
  `/ai/trtllm-poc` `TRTLLM_HOME` default, rename + update name references (docstring, README,
  proxy `owned_by`, compose-derived container name), keep `just restart`/`just swap` recipe names
  stable, ensure the quickstart starts the OpenAI proxy, and coordinate setup docs to the
  contract.

## Decisions & discoveries

- **Option C, per Ken** — the client is the product; the backend is one (cleaned-up, to-be-renamed)
  reference implementation behind a documented contract. Driven partly by the fact the client is
  already consumed standalone from `gen-ai-langchain`.
- **Rename-resilience matters now.** Since the backend will be renamed, client docs reference it by
  role and prefer the `just` recipe (rename-stable) for restarts. The single remaining name
  reference is centralized for an easy update when the new name lands.
- **Audit was reassuring** — the backend is mostly env-driven already; the env-specific surface is
  small (a POC-era default path, a few name strings, the container name). Captured in WI #93.

## Outcomes

- Docs-only sprint; `uv run pytest -q` → **23 passed, 4 deselected**; ruff clean.
- The contract is now an explicit artifact, so the client/backend can evolve independently.

## Follow-ups

- Sprint 6: publishable packaging — LICENSE, `pyproject` classifiers/keywords/URLs, CI for the
  offline suite, PyPI dry-run, and the "stand up a backend" quickstart that links the (renamed)
  reference backend.
- Coordinate with the backend rename (WI #93): update the client's single backend-name reference
  and the quickstart link when the new name is chosen.
