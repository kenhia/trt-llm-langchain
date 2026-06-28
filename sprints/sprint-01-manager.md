# Sprint 1 — Manager + connectivity

_Status: complete (live-verified) · Started: 2026-06-28 · Landed: 2026-06-28_

## Goal

A control-plane client that can drive TRT-LLM model lifecycle from Python — list/status/load/
unload/ensure — against the existing `trt-llm-explore` KServe v2 backend (`:8000`). No LangChain
yet.

## Plan

`config.py`, `errors.py`, `manager.py`, plus a tiny `cli.py` smoke tool. Derive the model
registry and vision-vs-chat classification from the KServe repository index rather than
duplicating `models.toml`.

## What shipped

- [`config.py`](../src/trt_llm_langchain/config.py) — `TrtLlmSettings` (chat_url `:8003`,
  control_url `:8000`, timeouts), all env-overridable via `TRTLLM_*` (`from_env()`). No
  hard-coded hosts baked into logic.
- [`errors.py`](../src/trt_llm_langchain/errors.py) — `TrtLlmError` hierarchy:
  `ServerUnavailableError`, `ModelNotFoundError` (lists available), `ModelLoadError`,
  `ModelUnloadError`.
- [`manager.py`](../src/trt_llm_langchain/manager.py) — `TrtLlmManager`:
  - `models()` parses `POST /v2/repository/index` into `{key: ModelInfo}`; classifies vision by
    the presence of `multimodal_encoders_*` vs `tensorrt_llm_bls_*`.
  - `available_keys()`, `loaded_keys()`, `is_responsive()`, `is_healthy()`, `validate()`.
  - `load()` / `unload()` walk components `{pipeline}_{key}` in dependency / reverse order,
    polling each to READY; raise on failure.
  - `ensure_loaded()` = no-op if responsive, else unload other resident models (single-GPU
    swap) then load. This is the primitive `ChatTrtLlm` will call in Sprint 2.
  - Injectable `httpx.Client` seam for tests.
- [`cli.py`](../src/trt_llm_langchain/cli.py) — `trtllm-lc <list|status|load|unload|ensure>`
  console script.
- [`tests/test_manager.py`](../tests/test_manager.py) — 5 offline tests via `httpx.MockTransport`
  (index parsing, vision classification, responsive/loaded state, unknown-model error,
  ensure_loaded swap + no-op).
- `pyproject.toml` — added `httpx` dep, a `dev` group (`pytest`, `ruff`), and the `trtllm-lc`
  script (replacing the dead `:main` stub).

## Decisions & discoveries

- **Registry = the live index, not `models.toml`.** Confirmed from explore's code that the proxy
  exposes the model **key** as the OpenAI `id` (e.g. `qwen2_5-coder-7b-fp16`), and that the
  KServe index lists every component with state. So the backend is the single source of truth;
  no second registry to maintain. (Cross-referenced
  `trt-llm-explore/src/triton_client.py` and `src/openai_proxy.py`.)
- **Component contract.** Naming is `{pipeline}_{key}` with pipeline sets
  `[preprocessing, tensorrt_llm, postprocessing, tensorrt_llm_bls, ensemble]` (chat) and
  `[..., multimodal_encoders, ensemble]` (vision). Mirrored here from explore's `common.py`;
  this convention is the backend contract to formalize in Sprint 5.
- **Single-GPU swap = unload-before-load**, implemented in `ensure_loaded`. Skipped a client-side
  nvidia-smi VRAM precheck (explore does one); unloading first frees VRAM and a real OOM surfaces
  as `ModelLoadError` from the load POST. Revisit if needed.
- **Library raises, CLI prints.** Explore's helpers `sys.exit` on error; unsuitable for a library,
  so the manager raises typed exceptions and the CLI translates them to messages + exit codes.

## Outcomes

- `uv run ruff check` clean; `uv run pytest -q` → **5 passed**.
- CLI fails gracefully against the (currently down) backend:
  `ERROR: TRT-LLM backend unreachable at http://localhost:8000 (...). Is the serving stack running?`
- **Exit criterion met (live-verified 2026-06-28).** Ken brought the `trt-llm-explore` stack up
  and walked the full round-trip: `status` → `ensure qwen2_5-coder-7b-fp16` → `status` →
  `unload` → `status`. All worked. Independently reconfirmed here via `list` + `status`:
  - All 9 registry models appear in `list` even when unloaded → **the EXPLICIT-mode index lists
    non-READY models**, so no `:8003/v1/models` fallback is needed.
  - Vision classification correct: `llava-1_5-7b-fp16` and `qwen2-vl-7b-fp16` → `vision`; the
    other seven → `chat`.
  - Index `state` for never-loaded models is empty (`""`); for previously-loaded-then-unloaded
    it's `UNAVAILABLE`. Both correctly map to `responsive=no`.

## Follow-ups

- Sprint 2: `ChatTrtLlm(ChatOpenAI)` calling `ensure_loaded` lazily before the first generation.
- (Resolved) The `:8003/v1/models` fallback considered for unloaded-model discovery is **not
  needed** — the index already lists them.
