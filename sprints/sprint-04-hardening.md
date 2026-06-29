# Sprint 4 — Hardening, live tests, README

_Status: complete (live-verified) · Started: 2026-06-28 · Landed: 2026-06-28 · Branch: `sprint-04-hardening`_

## Goal

Make the package robust and consumable for local use: error-path coverage, an opt-in live
integration test, and a README that lets a fresh clone (with a running backend) chat.

## What shipped

- **[`README.md`](../README.md)** — quickstart (the two-line swap), how-it-works (two HTTP
  surfaces), requirements, install, the full `TRTLLM_*` config table, the restart-based swap
  section with `TRTLLM_RESTART_CMD`, the `trtllm-lc` CLI, examples, the error hierarchy, and
  status/limitations.
- **Error-path tests** — [`tests/test_errors.py`](../tests/test_errors.py): an unreachable
  backend surfaces `ServerUnavailableError` (with the URL) from `models()`; `is_healthy()` /
  `is_responsive()` return `False` rather than raising; `_wait_healthy` times out cleanly.
- **Opt-in live integration test** — [`tests/test_live.py`](../tests/test_live.py), marked `live`
  and skipped unless `TRTLLM_LIVE=1` + backend reachable. Covers `list`, `invoke` + `stream`, and
  a best-effort `bind_tools` call, all on the **currently resident** model (no forced load). A
  further-gated `test_restart_based_swap` (`TRTLLM_LIVE_SWAP=1` + `TRTLLM_RESTART_CMD`) exercises
  the real swap without running by accident.
- **pytest config** (`pyproject.toml`) — registers the `live` marker and defaults
  `addopts = -m 'not live'`, so `uv run pytest` is fast and offline by default.

## Decisions & discoveries

- **Live tests are opt-in and minimally disruptive.** Default `pytest` excludes them; when run,
  they prefer the resident model and never restart the backend unless the swap test is explicitly
  enabled. This keeps the suite safe to run anywhere while still being able to vet against real
  hardware.
- **`bind_tools` live is asserted as "doesn't error," not "emits tool_calls."** Tool-calling is
  model-dependent (the proxy has `tool_parsers.py`, but quality varies), so the test reports
  rather than requires it — avoids a flaky gate.

## Outcomes

- `uv run ruff check` clean.
- `uv run pytest -q` → **23 passed, 4 deselected** (offline, live excluded by default).
- `TRTLLM_LIVE=1 uv run pytest -m live` → **3 passed, 1 skipped** (swap gated off) against the
  running backend with llama-3_1-8b-fp16 resident: `list`, `invoke`+`stream`, `bind_tools` all
  green.

## Follow-ups

- Optional: assert actual `tool_calls` structure on a model known to support it, once we pick one.
- Phase 2 (Sprint 5): write `docs/03-backend-contract.md` and decide the `trt-llm-explore`
  relationship (ADR); (Sprint 6) publishable packaging — LICENSE, classifiers, CI, PyPI.
- CI can run the offline suite (`uv run pytest`) anywhere; the `live` marker is for a
  GPU-host job.
