# Sprint 8 — Getting-started guide

_Status: complete (consume path live-verified) · Started: 2026-06-29 · Branch: `sprint-08-getting-started`_

## Goal

A `docs/getting-started.md` that takes someone from a bare machine to their own LangChain app
talking to a local model: install the backend, download + build + load a model, then create a new
project that consumes the **published** `trt-llm-langchain` package.

## Context

Published in Sprint 7's wake: `trt-llm-langchain 0.1.0` is live on PyPI (Trusted Publishing) and
GitHub (`kenhia/trt-llm-langchain`); the backend `trt-llm-explore` is public at
`kenhia/trt-llm-explore`. So the guide can use real clone/`pip` commands.

## What shipped

- **[`docs/getting-started.md`](../docs/getting-started.md)** — prerequisites; Part 1 stand up the
  backend (clone → `.env` → `uv sync` → pull Triton image → `just build` (auto-downloads weights +
  builds engine) → `just setup` → `just up` + `just up-openai` → `just load`); Part 2 consume from
  your own project (`uv init` → `uv add trt-llm-langchain` → `ChatTrtLlm()`); switching models
  (restart-based, `TRTLLM_RESTART_CMD`); LCEL/tools; a troubleshooting table; next-steps links.
- **README** links to the guide from the quickstart.

## Decisions & discoveries

- **`just build <key>` is one step** — it downloads the HF weights *and* builds the engine
  (`src/build.py::download_weights` runs `huggingface-cli download` in-container, gated via
  `HF_TOKEN`). The guide reflects that rather than implying a separate download.
- **Used Qwen as the worked example** (`qwen2_5-coder-7b-fp16`) because it's **non-gated** — no
  `HF_TOKEN` friction for a first run; gated Llama is noted as the variant that needs a token.
- **Verified the consume path against the real PyPI package**, not the local checkout (below).

## Outcomes

- **Live-verified consume path:** fresh dir → `uv init` → `uv add trt-llm-langchain` pulled
  **0.1.0 from PyPI** → `ChatTrtLlm()` adopted the resident model (qwen) and answered
  ("TensorRT-LLM is a high-performance, optimized library for deploying large language models on
  NVIDIA GPUs."), on Python 3.13. Throwaway project built in scratch, then removed.
- Guide commands cross-checked against `trt-llm-explore`'s README, `docs/setup.md`, and `justfile`.

## Follow-ups

- When the backend's WI #93 generalization lands (neutral `TRTLLM_HOME` default, proxy in the
  quickstart), re-check the guide's backend steps still match.
- Optional: a short asciinema/GIF of the consume path for the README.
