# Sprint 9 — Post-publish polish

_Status: complete (live-verified) · Started: 2026-06-29 · Branch: `sprint-09-polish`_

## Goal

Address the post-publish loose ends: CI action deprecation, and the real story for
tool-calling / structured output.

## What shipped

- **Action pins bumped** — `actions/checkout@v4 → v5`, `astral-sh/setup-uv@v5 → v6` in both
  `ci.yml` and `release.yml` (clears the Node 20 deprecation warning from the v0.1.0 release run).
- **Structured output, honestly** — verified live what actually works on the current backend:
  - `bind_tools` function-calling `tool_calls` are **not emitted** (qwen returns `tool_calls: []`).
  - `with_structured_output(..., method="function_calling")` (the default) **fails**.
  - `with_structured_output(..., method="json_mode")` **works** → `Person(name='Ada Lovelace', age=36)`.
  - Shipped [`examples/structured_output.py`](../examples/structured_output.py) using `json_mode`;
    added a live test (`test_structured_output_json_mode`); corrected the README limitations and
    the getting-started doc to state tool-calling isn't emitted and `json_mode` is the typed-output
    path.

## Decisions & discoveries

- **Don't ship the broken path.** Rather than claim `bind_tools`/`with_structured_output` "work,"
  the docs now point users to the one that does (`json_mode`) and name the one that doesn't.
  Tested before documenting.

## Outcomes

- `uv run ruff check` clean; `uv run pytest -q` → **28 passed, 5 deselected**.
- Live (`TRTLLM_LIVE=1`, `-k "structured_output or bind_tools"`) → **2 passed** against qwen.

## Follow-ups (not in this repo)

- **trt-llm-explore WI #93** — publish is done (GH repo exists), but three generalization/doc items
  remain: (1) `TRTLLM_HOME` default still `/ai/trtllm-poc` (justfile:11, common.py:51); (3) README
  quickstart doesn't mention `just up-openai`; (4) no "implements the backend contract" link to
  `trt-llm-langchain`. Small edits in the backend repo.
- If a tool-capable served model is added later, revisit `bind_tools` and the
  `function_calling` structured-output path.
