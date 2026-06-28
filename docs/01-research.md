# Research: TensorRT-LLM × LangChain on a single RTX 5090

_Date: 2026-06-28 · Goal: a `ChatTrtLlm(model="...")` that drop-in swaps for `ChatAnthropic(model="...")`, with model load/unload and streaming._

## TL;DR / decision

**Build a thin wrapper, install almost nothing new server-side.** There is no maintained,
dedicated LangChain chat model for TensorRT-LLM — but you don't need one to be written from
scratch, because:

1. **The chat/stream surface is a solved problem.** Point LangChain's `ChatOpenAI`
   (`langchain-openai`) at an OpenAI-compatible `/v1` endpoint. NVIDIA ships this pattern.
2. **The load/unload surface is *already built in your own `trt-llm-explore` project*.** It runs
   Triton in **EXPLICIT model-control mode** and exposes `POST /v2/repository/models/{m}/load`
   and `/unload` — independently identified by the ecosystem research as *the only
   NVIDIA-supported runtime model-swap path for TRT-LLM*.

So the new `trt-llm-langchain` package is a **client** that (a) sends chat/stream over the
existing OpenAI proxy and (b) drives load/unload over the existing KServe v2 API. The "wrapper"
the M365 plan described is real, but ~80% of what it proposed to build (a FastAPI server, a
`/chat` contract, a generate client) **already exists** in `trt-llm-explore` — see
[the M365 gap analysis](#appendix-where-the-m365-plan-missed) below.

## What already exists in your infrastructure

From `/home/ken/src/ai/trt-llm-explore` (your PRIMARY project):

| Capability | Where | Endpoint |
|---|---|---|
| OpenAI-compatible chat + streaming | `src/openai_proxy.py` | `POST :8003/v1/chat/completions` (SSE on `stream:true`), `GET :8003/v1/models` |
| Model load (VRAM-aware) | `src/triton_client.py` | `POST :8000/v2/repository/models/{component}/load` |
| Model unload | `src/triton_client.py` | `POST :8000/v2/repository/models/{component}/unload` |
| Model registry + state | `models.toml`, KServe index | `POST :8000/v2/repository/index` → `MODEL/BUILT/SETUP/LOADED/RESPONSIVE` |
| Streaming (Triton native) | `src/openai_proxy.py` | `POST :8000/v2/models/{ensemble}/generate_stream` |

Key facts:
- Models are **5-component ensembles** (`preprocessing`, `tensorrt_llm`, `postprocessing`,
  `tensorrt_llm_bls`, `ensemble`). A "load" loads all components in order; "unload" reverses.
- **EXPLICIT** control mode → models are *not* resident until explicitly loaded; you can
  load/unload at runtime **without restarting Triton**.
- Registry keys are exactly your target names, e.g. `qwen2_5-coder-7b-fp16`,
  `llama-3_1-8b-fp16`, `qwen2_5-coder-7b-fp8`, plus vision (`qwen2-vl-7b-fp16`).
- A single 5090 (32 GB) holds **one model at a time** (~5–22 GB each), so "switch model" really
  means **unload current → load next**. The registry already carries `expected_vram_gb` and
  the loader checks free VRAM before loading.
- Stack: TRT-LLM `1.3.0rc13` inside `nvcr.io/nvidia/tritonserver:25.05-trtllm-python-py3`,
  host tooling via `uv`, Python 3.11.

> Note: the OpenAI proxy on `:8003` is marked "optional" in explore. The wrapper depends on it
> for chat, so it must be running. (Alternative: talk Triton-native `generate_stream` directly —
> more code, no benefit. Recommend depending on the proxy.)

## Ecosystem findings (with sources)

### 1. `trtllm-serve` — NVIDIA's first-party OpenAI server
- Real and shipped (since TRT-LLM **v0.15.0**, 2024-12-04). Endpoints: `/v1/models`,
  `/v1/completions`, `/v1/chat/completions`, `/health`, `/metrics`, `/version`.
  <https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html>
- Streaming: supported (standard OpenAI `stream=true` SSE).
- Tool calling: via `--tool_parser`; model-dependent and currently buggy for some models
  (open issues #7163, #9256).
- **One model per process. No runtime load/unload.** This is the gap your explore project's
  Triton setup fills.
- RTX 5090 / Blackwell: works but bleeding-edge (50-series support via WSL added v0.17.0).

### 2. `ChatOpenAI` against a local `/v1` endpoint
- Works: `ChatOpenAI(model=..., base_url="http://localhost:8003/v1", api_key="trt-llm")`
  (dummy key required). Same pattern LangChain documents for vLLM.
- **Caveat (current docs):** `ChatOpenAI` "targets official OpenAI API specifications only.
  Non-standard response fields (e.g. `reasoning_content`) are not extracted… use the
  corresponding provider-specific LangChain package instead."
  <https://reference.langchain.com/python/langchain-openai/chat_models/base/ChatOpenAI>
  → Fine for your proxy, which emits standard `chat.completion` objects. Watch this if you later
  serve reasoning models that emit channel/Harmony tokens.
- Streaming token-usage is off by default for non-OpenAI `base_url`; pass `stream_usage=True`
  if the server emits usage (explore's proxy supports `stream_options.include_usage`).

### 3. `ChatNVIDIA` (`langchain-nvidia-ai-endpoints`)
- Supports self-hosting via `base_url`, **but it's NIM-shaped** (does NVIDIA `nvext` payloads +
  model discovery). Documented for **NIM**, not raw `trtllm-serve`/Triton. Pointing it at your
  proxy *might* work but is untested by NVIDIA — **don't rely on it.**
- Actively maintained (v1.4.2, 2026-06-23). Only worth adopting if you commit to running NIM
  containers.

### 4. Existing wrappers for TRT-LLM
- **No maintained, dedicated LangChain `ChatModel` for `trtllm-serve`.** Community asks are open
  and unfilled (#12474, #13975, #29547).
- Legacy `langchain-nvidia-trt` (`TritonTensorRTLLM`) exists — a Triton **gRPC** client,
  minimally maintained, superseded by `langchain-nvidia-ai-endpoints`.
- **LlamaIndex** has `LocalTensorRTLLM` (`llama-index-llms-nvidia-tensorrt`): loads an engine
  **in-process** (not a server client) and **has no streaming** (`stream_complete` raises
  `NotImplementedError`). Different model from your goal; not reusable here.

### 5. Local NIM as an alternative
- A local NIM container also gives an OpenAI `/v1` endpoint and is what `ChatNVIDIA` targets.
  Free for NVIDIA Developer Program members for personal use. But per-model 5090 support is not
  crisply documented, and you'd give up the direct Blackwell/NVFP4 control you already have.
  Not recommended given your existing Triton stack.

### 6. The load/unload reality
- `trtllm-serve`: no hot-swap (one model/process).
- NIM: one model/container.
- vLLM "sleep mode" (2025-10-26): can offload/reload, but dev-mode-flagged, not production.
- **Triton + `tensorrtllm_backend` in EXPLICIT mode: the supported runtime swap path** — and the
  one you already run. `load`/`unload` via the KServe repository API, no server restart.
- On one GPU the idiom is **unload-then-load** (can't hold two big models). Your manager just has
  to evict the current model before loading the next.

## Recommendation

| Concern | Choice |
|---|---|
| Chat / streaming / tools / structured output | Subclass **`ChatOpenAI`** → reuse all of it against `:8003/v1`. |
| Provider package (`ChatNVIDIA`)? | **No** — NIM-shaped, untested vs your proxy. |
| Run `trtllm-serve` instead? | **No** — it can't load/unload; your Triton stack already can. |
| Model load/unload / `model="..."` semantics | Thin **manager** over your existing KServe v2 `:8000` load/unload + repository index. |
| Where the registry lives | **Query the live server** (`/v2/repository/index` + `/v1/models`) as the source of truth; optionally cross-check `trt-llm-explore/models.toml`. Avoid a second hand-maintained registry. |
| New server code to write | **None.** This repo is a pure client. |

Net: `ChatTrtLlm` = `ChatOpenAI` subclass + a small `TrtLlmManager` that ensures the requested
model is loaded (unloading the current one if needed) before the first call. See
[`02-plan.md`](02-plan.md).

## Appendix: where the M365 plan missed

The M365 draft (`.scratch/m365-discussion.md`) is structurally reasonable but assumes a greenfield
build and a hypothetical server. Corrections:

- It proposes building a FastAPI server + `/chat`, `/models/load`, `/models/unload` contract.
  **All of this already exists** in `trt-llm-explore` (OpenAI proxy + KServe v2). Don't rebuild.
- Its `_generate` reimplements message conversion and HTTP by hand. **Subclass `ChatOpenAI`**
  instead and inherit streaming/batch/tools/structured-output for free.
- It hard-codes a static `MODEL_REGISTRY` with invented engine paths
  (`/models/qwen2.5-32b/engine.plan`) and wrong key style (`qwen2.5-32b`). Real keys are like
  `qwen2_5-coder-7b-fp16`, and the registry should be **queried from the server**, not duplicated.
- It loads the model in `__init__` unconditionally. Prefer **lazy load on first call** (LangChain
  is lazy; constructing an object shouldn't silently evict 14 GB of VRAM), with an opt-in eager flag.
- It omits the single-GPU constraint (unload-before-load) and VRAM accounting, which your
  existing loader already handles.
