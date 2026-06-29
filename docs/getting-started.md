# Getting started

End-to-end: from a bare machine to your own LangChain app talking to a local TensorRT-LLM model.

You'll set up **two things**:

1. **The backend** — [`trt-llm-explore`](https://github.com/kenhia/trt-llm-explore): builds TRT-LLM
   engines and serves them via Triton + an OpenAI-compatible proxy.
2. **Your app** — a new project that `uv add`s the published **`trt-llm-langchain`** package and
   uses `ChatTrtLlm`.

```
your LangChain app ──▶ trt-llm-langchain (pip) ──HTTP──▶ trt-llm-explore (Triton + proxy) ──▶ GPU
```

---

## 0. Prerequisites

- **NVIDIA GPU** — built/verified on an RTX 5090 (32 GB), driver **595.58.03+**.
- **Docker** with the **NVIDIA Container Toolkit** (`nvidia-ctk`):
  ```bash
  docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi   # should print your GPU
  ```
- **[`just`](https://just.systems/)** and **[`uv`](https://docs.astral.sh/uv/)**.
- A **Hugging Face token** (`HF_TOKEN`) only if you build a *gated* model (e.g. Llama). The Qwen
  example below is **not** gated.
- Disk: each 7–8B model needs ~15–30 GB for weights + engine under `$TRTLLM_HOME`.

---

## 1. Stand up the backend (`trt-llm-explore`)

```bash
git clone https://github.com/kenhia/trt-llm-explore.git
cd trt-llm-explore
```

Create a `.env` at the repo root (loaded automatically by `just`):

```bash
# .env
TRTLLM_HOME=/ai/trtllm-explore        # any dir with room for weights + engines
HF_TOKEN=hf_your_token_here           # only needed for gated models (omit otherwise)
```

Install and pull the serving image:

```bash
uv sync
docker pull nvcr.io/nvidia/tritonserver:25.05-trtllm-python-py3
just registry-list                    # see available model keys
```

### Build a model (downloads weights + builds the engine)

`just build` downloads the Hugging Face weights **and** builds the TRT-LLM engine in one step.
Use a non-gated model to start — `qwen2_5-coder-7b-fp16` (no `HF_TOKEN` needed):

```bash
just down                             # building requires serving to be stopped
just build qwen2_5-coder-7b-fp16      # downloads to $TRTLLM_HOME/models, builds $TRTLLM_HOME/engines
just setup qwen2_5-coder-7b-fp16      # render the Triton model_repo (decoupled=true → streaming)
```

> Gated models (e.g. `llama-3_1-8b-fp16`) need `HF_TOKEN` set in `.env` first. Run
> `just registry-list` for all keys.

### Serve it

```bash
just up                               # Triton (KServe v2 on :8000)
just up-openai                        # OpenAI-compatible proxy (:8003) — trt-llm-langchain needs this
just healthy                          # wait for Triton readiness
just load qwen2_5-coder-7b-fp16       # load the model into the GPU
```

Sanity check the proxy:

```bash
curl -s localhost:8003/v1/models      # should list {"id":"qwen2_5-coder-7b-fp16", ...}
```

The backend is now serving. Leave it running.

---

## 2. Consume it from your own project

In a **separate** directory (your app — this is exactly the verified flow):

```bash
mkdir my-llm-app && cd my-llm-app
uv init
uv add trt-llm-langchain              # from PyPI
```

If your app runs on the **same machine** as the backend, the defaults (`:8003` / `:8000`) just
work. For a remote backend, set:

```bash
export TRTLLM_CHAT_URL=http://<host>:8003
export TRTLLM_CONTROL_URL=http://<host>:8000
```

`main.py`:

```python
from trt_llm_langchain import ChatTrtLlm

# Option A: adopt whatever model the backend currently has loaded.
chat = ChatTrtLlm()

# Option B: name a model explicitly (loads it if needed; see swapping below).
# chat = ChatTrtLlm(model="qwen2_5-coder-7b-fp16", temperature=0.2, max_tokens=256)

print(chat.invoke("In one sentence, what is TensorRT-LLM?").content)

for chunk in chat.stream("Count to 5."):
    print(chunk.content, end="", flush=True)
print()
```

```bash
uv run python main.py
```

That's the whole loop: a published `pip` install talking to your local GPU through LangChain.

---

## 3. Switching models (single GPU = restart-based)

A single GPU holds one model at a time, and unload doesn't free VRAM, so switching restarts the
backend to reclaim it. Point the client at the backend's `restart` recipe so swaps are automatic:

```bash
export TRTLLM_RESTART_CMD="just -C /path/to/trt-llm-explore restart"
```

```python
ChatTrtLlm(model="qwen2_5-coder-7b-fp16").invoke("hi")   # loads qwen
ChatTrtLlm(model="llama-3_1-8b-fp16").invoke("hi")        # restart → load llama
```

Without `TRTLLM_RESTART_CMD`, a swap raises `BackendRestartRequiredError` with guidance (it never
OOMs). It's **co-located only** — for a remote backend, load the target server-side (e.g.
`just swap <key>`) and use `ChatTrtLlm()` to adopt it. Full rationale:
[ADR 0002](decisions/0002-model-swap-strategy.md).

---

## 4. Use it like any LangChain chat model

`ChatTrtLlm` is a `ChatOpenAI` subclass, so LCEL, batching, async, `bind_tools`, and
`with_structured_output` all work (tool calling is **non-streaming**, on tool-capable models —
Llama/Qwen/Mistral, not Phi). Example:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from trt_llm_langchain import ChatTrtLlm

chain = ChatPromptTemplate.from_messages([("human", "{q}")]) | ChatTrtLlm() | StrOutputParser()
print(chain.invoke({"q": "Name three CUDA concepts."}))
```

See [`examples/`](../examples/) for more.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ServerUnavailableError` | Backend not running, or wrong URL. `just up` + `just up-openai`; check `TRTLLM_CONTROL_URL`/`TRTLLM_CHAT_URL`. |
| `openai.APIConnectionError` on chat | Triton is up but the **proxy** isn't — run `just up-openai` (it's a separate profile). |
| `ModelNotFoundError` | Model key not built/set up on the backend. `just registry-list`, then `just build`/`just setup`. |
| `ResidentModelError` (from `ChatTrtLlm()`) | Zero or >1 models loaded — `just load <key>` one, or pass `model=`. |
| `BackendRestartRequiredError` | Swapping models without a restart path — set `TRTLLM_RESTART_CMD`, or `just swap <key>` on the host. |
| `InsufficientVramError` / CUDA OOM | VRAM not reclaimed — `just restart` (or `just swap <key>`) on the backend. |
| Streaming returns HTTP 400 | Model deployed non-decoupled — `just setup-all && just restart` on the backend. |
| Gated model download fails | Set `HF_TOKEN` in the backend `.env` and accept the model's license on Hugging Face. |

---

## Where to go next

- [README](../README.md) — API surface, config, CLI.
- [docs/03-backend-contract.md](03-backend-contract.md) — the exact contract, if you want to back
  `ChatTrtLlm` with a different server.
- [docs/02-plan.md](02-plan.md) and [sprints/](../sprints/) — how this was built.
