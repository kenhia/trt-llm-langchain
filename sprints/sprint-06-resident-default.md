# Sprint 6 — Resident-model default + swap strategy decision

_Status: complete (live-verified) · Started: 2026-06-29 · Branch: `sprint-06-resident-default`_

> Inserted after the swap-strategy discussion; publish-ready packaging moves to Sprint 7.

## Goal

Decide the model-swap story for the published product (the "ready for others" question), and add
the `ChatTrtLlm()` no-model ergonomic.

## Decisions (ADR 0002)

The swap question reframed: the client's `TRTLLM_RESTART_CMD` is **co-located only** (it shells a
local command), which is why "swap for others" is really "where does restart authority live." The
`trt-llm-explore` agent's options (auto-swap in proxy / admin endpoint / sidecar) all put it in the
backend and require Docker/root-equiv authority.

[ADR 0002](../docs/decisions/0002-model-swap-strategy.md) — for v1:
1. **Error well by default** (`BackendRestartRequiredError`, no raw OOM) — the honest baseline,
   zero new attack surface.
2. **Keep optional local `TRTLLM_RESTART_CMD`** for the co-located single-box user.
3. **`ChatTrtLlm()` adopts the resident model** (never swaps).
4. **Document server-side swap (explore Option B) as a future/optional backend capability** in the
   contract; don't build it for v1. Reject Option A (auto-swap inside `/v1/chat/completions`) for
   the published contract.

## What shipped

- **`ChatTrtLlm()` with no model adopts the resident model.**
  - `TrtLlmManager.resident_model()` returns the single loaded key, else raises
    [`ResidentModelError`](../src/trt_llm_langchain/errors.py) (zero or >1 loaded).
  - `ChatTrtLlm.__init__` resolves the resident model before `super().__init__` when `model` is
    omitted; never triggers a swap (the adopted model is already responsive).
- **Docs:** ADR 0002; contract doc gains a "server-side swap (future, optional)" section and a
  note that `TRTLLM_RESTART_CMD` is co-located only; README documents `ChatTrtLlm()` and the
  remote-swap limitation; plan roadmap updated (packaging → Sprint 7).
- **Tests:** `resident_model` single/none/multiple (`test_manager.py`); no-model adoption and
  no-resident error (`test_chat.py`).

## Outcomes

- `uv run ruff check` clean; `uv run pytest -q` → **28 passed, 4 deselected**.
- **Live:** `ChatTrtLlm()` adopted the resident model (qwen at the time) and replied — adoption
  works end-to-end and is swap-free.

## Follow-ups

- Sprint 7 (packaging): LICENSE, `pyproject` metadata/classifiers/URLs, CI for the offline suite,
  PyPI dry-run, and the "stand up a backend" quickstart (coordinated with the backend rename,
  explore WI #93).
- If/when remote seamless swap is wanted: implement explore Option B (token-protected swap
  endpoint) + a client "control-URL swap" mode (slot reserved in the contract / ADR 0002).
