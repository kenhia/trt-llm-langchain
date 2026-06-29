# Backend contract

The exact HTTP surface `trt-llm-langchain` depends on. Any server that satisfies this contract can
back `ChatTrtLlm` / `TrtLlmManager` — `trt-llm-explore` is the reference implementation, but it is
not special. This document is the seam that lets the client and the backend evolve independently
(Phase 2).

There are **two surfaces**, configured by `TRTLLM_CHAT_URL` (default `:8003`) and
`TRTLLM_CONTROL_URL` (default `:8000`).

> The reference backend is referred to here by role; the concrete project (`trt-llm-explore`
> today) will be cleaned up and likely renamed (ADR 0001 / korg WI #93). Treat any specific
> names, paths, or container identifiers below as examples.

## 1. Chat surface — OpenAI-compatible (`TRTLLM_CHAT_URL`)

Consumed by `ChatTrtLlm` via `langchain_openai.ChatOpenAI`. Standard OpenAI semantics.

| Method / path | Requirement |
|---|---|
| `GET /v1/models` | Returns `{"data": [{"id": <model_key>, ...}]}`. **The `id` is the model key** the client passes as `model=` (e.g. `qwen2_5-coder-7b-fp16`). |
| `POST /v1/chat/completions` | Standard OpenAI chat completion. Returns a `chat.completion` object with `choices[].message.content`. |
| `POST /v1/chat/completions` (`"stream": true`) | Server-Sent Events of `chat.completion.chunk` objects with `choices[].delta.content`. Required for `.stream()`. |

Notes:
- An API key is sent but may be any non-empty string (`TRTLLM_API_KEY`, default `trt-llm`).
- Responses must conform to the **official OpenAI schema** — `ChatOpenAI` ignores non-standard
  fields (e.g. `reasoning_content`). Models that leak channel/Harmony tokens into content will
  pass through verbatim.
- Streaming requires the underlying model to be deployed **decoupled** (see §3).

## 2. Control surface — Triton KServe v2 (`TRTLLM_CONTROL_URL`)

Consumed by `TrtLlmManager` for model lifecycle and discovery.

| Method / path | Requirement |
|---|---|
| `GET /v2/health/ready` | `200` when the server is up. Used for health + post-restart wait. |
| `POST /v2/repository/index` | Returns `[{"name": <component>, "state": <state>}, ...]` for **all** models in the repository, loaded or not. The registry is derived from this. |
| `POST /v2/repository/models/{component}/load` | `200` on success; non-`200` body is inspected for CUDA OOM (→ `InsufficientVramError`). |
| `POST /v2/repository/models/{component}/unload` | `200` on success. |
| `GET /v2/models/{component}/ready` | `200` when that component is loaded and servable. |

The server must run in Triton **EXPLICIT model-control mode** (models load/unload on request, not
all-at-startup).

## 3. Naming & model-shape conventions

A model with key `K` is served as a 5-component **ensemble**; components are named
`{pipeline}_{K}`:

- **Chat:** `preprocessing_K`, `tensorrt_llm_K`, `postprocessing_K`, `tensorrt_llm_bls_K`,
  `ensemble_K`
- **Vision:** `preprocessing_K`, `tensorrt_llm_K`, `postprocessing_K`, `multimodal_encoders_K`,
  `ensemble_K`

Rules the client relies on:
- **Load order** is the list order; **unload** is reverse order.
- **`ensemble_K`** is the inference entry point and the readiness/"is it servable" signal.
- **Vision vs chat** is inferred from the index: presence of `multimodal_encoders_K` ⇒ vision,
  `tensorrt_llm_bls_K` ⇒ chat.
- **State `READY`** on a component means loaded; the model is "responsive" when `ensemble_K` is
  `READY` / its `/ready` returns `200`. Never-loaded components may report empty state;
  previously-loaded-then-unloaded report `UNAVAILABLE` — both are non-servable.
- For streaming, the `tensorrt_llm_K` component must be configured
  `model_transaction_policy { decoupled: true }`. A non-decoupled deployment returns HTTP 400 on
  `generate_stream` and the proxy surfaces a `server_error`.

## 4. Single-GPU swap = restart-based

The hard constraint that shapes `ensure_loaded`:

- A single GPU holds **one model at a time**.
- **Unload does not free VRAM.** TensorRT-LLM's `cudaMallocAsync` pool retains freed pages at the
  process level; Triton 25.05 does not trim it. So an in-place `unload A` → `load B` fails with
  CUDA OOM on B's weights.
- **The only reliable reclaim is a backend process restart.** Therefore a conforming backend
  should provide a restart mechanism that drops all models and frees VRAM (e.g. a container
  restart, or `just restart` / `just swap <key>` in `trt-llm-explore`).
- The client treats swap as restart-based: it invokes a configured restart (`TRTLLM_RESTART_CMD`
  or a `restart_backend` callable), waits for `/v2/health/ready`, then loads the target. Without
  one, it raises `BackendRestartRequiredError` rather than attempting the OOM-ing load.

See `trt-llm-explore` sprint 006 (WI #90 streaming/decoupled, WI #91 VRAM/swap) for the
backend-side rationale and the recipes.

The client's restart command (`TRTLLM_RESTART_CMD`) is **co-located only** — it shells out a local
command, so it cannot restart a *remote* backend. A remote client gets `BackendRestartRequiredError`
and must swap server-side.

### Optional (future): server-side swap control endpoint

To make swaps remotable and seamless without client-local restart authority, a backend MAY expose
a control endpoint — e.g. `POST /admin/swap {"model": "<key>"}` that restarts + loads and returns
when healthy, protected by a shared-secret token. This is **not required** by the contract and is
**not implemented** in v1 (see [ADR 0002](decisions/0002-model-swap-strategy.md)); it's documented
so the slot is reserved. When present, the client could call it from `ensure_loaded` instead of a
local restart command. Auto-swapping inside `/v1/chat/completions` is intentionally *not* part of
the contract (it would add surprising latency/side effects to the standard OpenAI endpoint).

## 5. What a non-`trt-llm-explore` backend must do to conform

1. Expose the OpenAI chat surface (§1) with the model key as the OpenAI `id`.
2. Expose the KServe v2 control surface (§2) in EXPLICIT mode.
3. Name components `{pipeline}_{key}` per §3 and deploy `tensorrt_llm_*` decoupled for streaming.
4. Provide a restart that reclaims VRAM (§4) if model swapping is wanted; otherwise run one model
   and skip swaps.

A plain `trtllm-serve` satisfies §1 (chat/stream) but **not** §2 (no runtime load/unload) — usable
for single-model, chat-only via `ChatTrtLlm` with no swapping. Triton + `tensorrtllm_backend` in
EXPLICIT mode satisfies both.
