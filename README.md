# trt-llm-langchain

Use local **TensorRT-LLM** models from **LangChain**. `ChatTrtLlm` is a call-site drop-in for
`ChatAnthropic` / `ChatOpenAI`, backed by a TensorRT-LLM server on your own GPU, with model
load/unload and streaming.

```python
# from langchain_anthropic import ChatAnthropic
# chat = ChatAnthropic(model="claude-sonnet-4-6")
from trt_llm_langchain import ChatTrtLlm
chat = ChatTrtLlm(model="qwen2_5-coder-7b-fp16")

print(chat.invoke("Write a one-line Python factorial.").content)
for chunk in chat.stream("Count to 5."):
    print(chunk.content, end="", flush=True)
```

It subclasses `langchain_openai.ChatOpenAI` (so you inherit `invoke`/`stream`/`batch`/async,
`bind_tools`, `with_structured_output`) and adds a small manager that makes the requested model
**resident before the first call**.

Or adopt whatever model is already loaded — `ChatTrtLlm()` with no `model` uses the resident
model (errors if zero or more than one is loaded) and never triggers a swap:

```python
chat = ChatTrtLlm()   # talk to whatever's currently loaded
```

## How it works

`trt-llm-langchain` is a thin **client** — it does not run models itself. It talks to two HTTP
surfaces of a [`trt-llm-explore`](#backend)-style backend:

| Surface | Default | Used for |
|---|---|---|
| OpenAI-compatible proxy | `http://localhost:8003/v1` | chat + streaming (via `ChatOpenAI`) |
| Triton KServe v2 | `http://localhost:8000/v2` | model load / unload / status |

```
ChatTrtLlm (ChatOpenAI)  ──chat/stream──▶  :8003 /v1/chat/completions
        │
        └─ TrtLlmManager  ──load/unload──▶  :8000 /v2/repository/...
```

## Requirements

- Python ≥ 3.11
- A running TensorRT-LLM backend exposing the two surfaces above with one model per *ensemble*
  named `{pipeline}_{model_key}` (the [`trt-llm-explore`](#backend) contract). Start it first —
  including the OpenAI proxy (e.g. `just up-openai` in that project).

## Quickstart: stand up a backend

The reference backend is [`trt-llm-explore`](#backend). With a built/set-up model, bring up Triton
+ the OpenAI proxy and load a model:

```bash
# in the trt-llm-explore checkout
just up            # Triton (KServe v2 on :8000)
just up-openai     # OpenAI-compatible proxy (:8003)
just load qwen2_5-coder-7b-fp16
```

Then, from your LangChain code:

```python
from trt_llm_langchain import ChatTrtLlm
print(ChatTrtLlm().invoke("Hello!").content)   # adopts the loaded model
```

See the backend's README for building/setting up engines.

## Install

```bash
uv add trt-llm-langchain      # or: pip install trt-llm-langchain
```

For local development of this package:

```bash
uv sync
uv run pytest -q
```

## Configuration

All settings come from the environment (or pass a `TrtLlmSettings`):

| Env var | Default | Meaning |
|---|---|---|
| `TRTLLM_CHAT_URL` | `http://localhost:8003` | OpenAI proxy base (the `/v1` is appended) |
| `TRTLLM_CONTROL_URL` | `http://localhost:8000` | Triton KServe v2 base |
| `TRTLLM_API_KEY` | `trt-llm` | dummy key (the OpenAI client requires one) |
| `TRTLLM_RESTART_CMD` | _unset_ | command to restart the backend for model swaps (see below) |

## Model swapping (single GPU = restart-based)

A single GPU holds **one model at a time**, and TensorRT-LLM does **not** free VRAM on unload
(`cudaMallocAsync` pool retention), so switching models requires **restarting the backend** to
reclaim VRAM, then loading the target. Wire up a restart command to make swaps automatic:

```bash
# Prefer the backend's restart recipe (stable across a backend rename):
export TRTLLM_RESTART_CMD="just -C /path/to/your/trt-llm-backend restart"
# or restart the container directly (name depends on your deployment):
#   export TRTLLM_RESTART_CMD="docker restart <triton-container-name>"
```

```python
ChatTrtLlm(model="qwen2_5-coder-7b-fp16").invoke("hi")   # loads qwen
ChatTrtLlm(model="llama-3_1-8b-fp16").invoke("hi")        # restart → load llama
```

Without `TRTLLM_RESTART_CMD`, a swap raises `BackendRestartRequiredError` with guidance instead of
failing with a raw CUDA OOM. (See `trt-llm-explore` sprint 006 / WI #91 for the full rationale.)

`TRTLLM_RESTART_CMD` is **co-located only** — it runs a local command, so it can't restart a
*remote* backend. If you run the client on a different machine than the GPU host, either load the
target model server-side (then use `ChatTrtLlm()` / `ensure`), or use one model per process.
A future optional backend swap endpoint would make this seamless — see
[ADR 0002](docs/decisions/0002-model-swap-strategy.md).

## Control-plane CLI

A small tool for driving the backend without LangChain:

```bash
uv run trtllm-lc list                           # model keys the backend knows about
uv run trtllm-lc status                          # load/responsive state for each
uv run trtllm-lc load   qwen2_5-coder-7b-fp16
uv run trtllm-lc unload qwen2_5-coder-7b-fp16
uv run trtllm-lc ensure qwen2_5-coder-7b-fp16    # make resident (restart-based swap)
```

## Examples

- [`examples/basic_chat.py`](examples/basic_chat.py) — `invoke`
- [`examples/streaming.py`](examples/streaming.py) — `stream`
- [`examples/lcel_chain.py`](examples/lcel_chain.py) — `prompt | model | parser`
- [`examples/swap_models.py`](examples/swap_models.py) — restart-based model swap

## Errors

All derive from `TrtLlmError`:

- `ServerUnavailableError` — backend unreachable
- `ModelNotFoundError` — unknown model key (lists what's available)
- `ModelLoadError` — load failed; `InsufficientVramError` (CUDA OOM) and
  `BackendRestartRequiredError` (swap needs a restart) are subclasses
- `ModelUnloadError` — unload failed

## Status & limitations

- Built and verified against a single RTX 5090 + `trt-llm-explore` (Triton 25.05 / TRT-LLM 1.3).
- Single-GPU swap is restart-based (above); co-resident multi-model is not yet supported.
- Vision models (`*-vl-*`, `llava-*`) load, but multimodal message handling isn't wrapped yet.
- `bind_tools` is inherited but tool-calling quality is model-dependent.

## Backend

The reference backend is `trt-llm-explore`. The exact HTTP contract this client depends on will be
written up in `docs/03-backend-contract.md` (Phase 2), so any conforming server can be used.
See [`docs/`](docs/) for research and the plan, and [`sprints/`](sprints/) for the build log.
