# Sprint 3 — Model swapping + LCEL

_Status: complete (live-verified) · Started: 2026-06-28 · Landed: 2026-06-28 · Branch: `sprint-03-swap-lcel`_

## Goal

Exercise an actual model swap (qwen → llama) via `ChatTrtLlm(model=...)`, plus LCEL composition
(`prompt | model`) and a `bind_tools` check.

## How the swap requirement resolved (WI #91)

`trt-llm-explore` sprint 006 confirmed: on a single GPU, **unload does not reclaim VRAM** (TRT-LLM
`cudaMallocAsync` pool retention; no Triton 25.05 knob to trim it). The reliable swap is therefore
**restart-based** — restart the backend (reclaims VRAM) then load the target. Explore shipped
`just restart` and `just swap <key>` and documented the contract for this client:
`ensure_loaded` must be restart-based, not in-place. This sprint implements exactly that.

## What shipped

- **Restart-based `ensure_loaded`** ([`manager.py`](../src/trt_llm_langchain/manager.py)). On a
  swap to a *different* model it no longer does an in-place unload→load (which OOMs). Instead:
  no-op if already responsive → else if another model is resident, restart the backend (reclaims
  VRAM) and load the target → else fresh-load.
  - Opt-in restart strategy: `restart_backend` callable, or `restart_command` /
    `TRTLLM_RESTART_CMD` (e.g. `docker restart trt-llm-explore-triton-1`). After restart the
    manager waits for health (`restart_timeout_s`). If a restart hook also loads (`just swap`),
    the post-restart responsiveness check short-circuits.
  - No restart configured + swap needed ⇒ [`BackendRestartRequiredError`](../src/trt_llm_langchain/errors.py)
    with actionable guidance — **fails fast, never attempts the OOM-ing in-place load**.
- **VRAM/OOM backstop** (from earlier in the sprint): a load that still OOMs raises
  `InsufficientVramError` with free-VRAM (nvidia-smi best-effort) instead of a raw CUDA stack.
- **Examples**: [`lcel_chain.py`](../examples/lcel_chain.py),
  [`swap_models.py`](../examples/swap_models.py) (documents `TRTLLM_RESTART_CMD`, catches both
  restart/VRAM errors).
- **Tests (offline)**: restart-based swap (`test_manager.py`: no-restart⇒raises & does no
  in-place I/O; restart-callable⇒restarts once then loads; fresh-load needs no restart),
  OOM classification (`test_vram.py`), LCEL/`Runnable`/`bind_tools` wiring (`test_lcel.py`).

## Decisions & discoveries

- **Restart-based swap is the contract, encoded in `ensure_loaded`.** The previous unload-before-
  load design is gone — it could only ever OOM on a single GPU. Restart strategy is opt-in so the
  default client stays a pure HTTP client that manages no server lifecycle; wiring
  `TRTLLM_RESTART_CMD` makes `ChatTrtLlm(model=...)` swaps seamless.
- **Restart hook semantics = "restart only" (VRAM reclaimed, nothing loaded); the manager owns
  the load.** A re-check of responsiveness after restart also tolerates a hook that loads.
- **Co-residence (fit both models, skip the swap) is out of scope** — default
  `kv_cache_free_gpu_mem_fraction` (0.90) makes the first model claim most VRAM; per-model token
  budgets would be needed. Documented, not implemented.

## Outcomes

- `uv run ruff check` clean; `uv run pytest -q` → **19 passed**.
- **Live-verified (backend up):**
  - LCEL: `prompt | model | StrOutputParser` (qwen) → coherent CUDA-stream answer.
  - **Restart-based swap** via `swap_models.py` with `TRTLLM_RESTART_CMD="docker restart
    trt-llm-explore-triton-1"`: qwen answered → container restarted → llama loaded → llama
    answered, ~36 s end-to-end. `status` after: only `llama-3_1-8b-fp16` resident, qwen's VRAM
    reclaimed.
  - **Default path** (no restart cmd, llama resident, ask qwen): raised
    `BackendRestartRequiredError` immediately, llama left undisturbed (no OOM, no churn).

## Follow-ups

- Live `bind_tools` / `with_structured_output` against served models (wiring verified offline;
  tool-calling is model-dependent) — candidate for Phase 2 or a small Sprint 3.x.
- Streaming on models other than qwen needs the backend operational step (`just setup-all` +
  `just restart`) to refresh stale `decoupled: false` configs (explore WI #90 runbook). qwen
  (hand-patched) and llama-3_1-8b-fp8 already stream; llama-3_1-8b-fp16 will after `setup-all`.
- Consider letting `ChatTrtLlm` forward a `restart_backend` callable (today: configure via
  `TRTLLM_RESTART_CMD`, or build a `TrtLlmManager` and inject it).
