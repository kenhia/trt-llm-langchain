# Sprint 2 — `ChatTrtLlm` over `ChatOpenAI`

_Status: complete (live-verified) · Started: 2026-06-28 · Landed: 2026-06-28 · Branch: `sprint-02-chat-trtllm`_

## Goal

A `ChatTrtLlm` that is a call-site drop-in for `ChatAnthropic`/`ChatOpenAI`, with lazy model
load before the first generation. `invoke` and `stream` working end-to-end through the OpenAI
proxy (`:8003`).

## Plan

Subclass `ChatOpenAI` (inherit invoke/stream/batch/async/tools), point it at the proxy, and
override the four generate hooks to call `TrtLlmManager.ensure_loaded` first. Examples + offline
unit tests.

## What shipped

- [`chat.py`](../src/trt_llm_langchain/chat.py) — `ChatTrtLlm(ChatOpenAI)`:
  - `__init__(model, *, settings, eager_load, manager, **kw)` — points `base_url` at
    `{chat_url}/v1`, dummy `api_key`, `stream_usage=True` default; validates the model against
    the backend registry on construction (`ModelNotFoundError` if unknown).
  - Lazy `_ensure_loaded()` (sync) / `_aensure_loaded()` (async, off the event loop via
    `asyncio.to_thread`) — runs once before the first generation.
  - Overrides `_generate` / `_stream` / `_agenerate` / `_astream` to ensure-load then delegate to
    `super()`. Everything else (batch, `bind_tools`, `with_structured_output`) is inherited.
  - `manager=` injection seam for tests.
- [`examples/basic_chat.py`](../examples/basic_chat.py), [`examples/streaming.py`](../examples/streaming.py).
- [`tests/test_chat.py`](../tests/test_chat.py) — 5 offline tests (httpx MockTransport for the
  OpenAI surface + a fake manager): validates-on-construction, unknown-model raises, lazy-load
  fires once on first invoke and is idempotent, eager_load loads in ctor, `_llm_type`.
- Deps: `langchain-openai`, `langchain-core` added.

## Decisions & discoveries

- **Confirmed the ChatOpenAI contract for v1.3.x** before subclassing: field is `model_name`
  (alias `model`), `openai_api_base` (alias `base_url`), `openai_api_key` (alias `api_key`);
  `http_client` is a field (used as the test injection seam); `model_config` has `extra="ignore"`
  and `arbitrary_types_allowed=True` (so `PrivateAttr` for the manager is fine).
- **The model key IS the OpenAI model id.** The proxy's `/v1/models` returns `id` = the registry
  key (e.g. `qwen2_5-coder-7b-fp16`), so `ChatTrtLlm(model=key)` maps 1:1 with no translation.
- **Backend streaming was broken (server-side), now patched.** First live `stream()` 400'd.
  Isolated it to the backend (direct `curl` to `:8003` and to Triton `generate_stream` both 400;
  non-stream 200). Root cause: deployed `tensorrt_llm_qwen…` config had
  `model_transaction_policy { decoupled: false }`, so Triton refuses streaming. **Not a client
  bug** — our `_stream` code is correct. Per Ken's call: quick-patched the running config to
  `decoupled: true` and verified; filed **korg WI #90** (project `trt-llm-explore`) for the
  proper fix (setup_repo doesn't actually emit `decoupled: true` despite intending to).
- **VRAM is not freed on unload (cudaMallocAsync pool retention).** Reloading qwen in place after
  the config edit OOM'd repeatedly — `nvidia-smi` showed ~16.6 GB still held by the Triton process
  after a clean unload. A **container restart** reclaimed it (→ 31.5 GB free), after which load +
  streaming worked. This is a real risk for our single-GPU `ensure_loaded` swap (see Follow-ups).

## Outcomes

- `uv run ruff check` clean; `uv run pytest -q` → **10 passed**.
- **Live, end-to-end (backend up):**
  - `examples/basic_chat.py` → `def factorial(n): return 1 if n == 0 else n * factorial(n-1)`
  - `examples/streaming.py` → streamed `1\n2\n3\n4\n5`
  - The two-line swap goal works:
    `chat = ChatTrtLlm(model="qwen2_5-coder-7b-fp16")` then `chat.invoke(...)` / `chat.stream(...)`.
- Backend ops performed this sprint (reversible): started the `openai-proxy` compose service
  (`just up-openai`), patched the qwen config to decoupled, restarted the triton container.

## Follow-ups

- **VRAM-not-freed-on-unload risk to `ensure_loaded`** (also noted in WI #90 item 4). On a single
  GPU, unload→load may OOM because the pool doesn't release. Options to investigate in a later
  sprint: detect post-unload free VRAM and fail with a clear message, trigger a pool trim, or
  document that a true swap may need a backend restart. **Our offline tests don't catch this** —
  it only appears live.
- Sprint 3 will exercise an actual model **swap** (qwen → llama) via `swap_models.py`; that's
  where the VRAM behavior gets stress-tested. Expect to need the restart caveat or a mitigation.
- The proxy-down failure surfaces as a raw `openai.APIConnectionError`, not our
  `ServerUnavailableError` (ChatOpenAI owns that transport). Consider wrapping for a nicer message.
- `bind_tools` / `with_structured_output` are inherited but unverified against these served
  models — Sprint 3.
