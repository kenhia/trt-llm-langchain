# Sprint 3 — Model swapping + LCEL

_Status: in progress — backend-independent parts done; live swap blocked on WI #91 · Started: 2026-06-28 · Branch: `sprint-03-swap-lcel`_

## Goal

Exercise an actual model swap (qwen → llama) via `ChatTrtLlm(model=...)`, plus LCEL composition
(`prompt | model`) and a `bind_tools`/structured-output check.

## Scope note

Per the Sprint 2 follow-up and the split decision: the live swap depends on **WI #91** (unload
doesn't reclaim VRAM, so swap OOMs until backend restart). Ken is fixing that in
`trt-llm-explore`. This sprint lands everything that does **not** depend on the backend swap
working, and designs the client to handle the VRAM reality correctly regardless of how #91 lands.

## What shipped

- **`ensure_loaded` VRAM hardening (client-side half of WI #91).**
  - New [`InsufficientVramError(ModelLoadError)`](../src/trt_llm_langchain/errors.py).
  - [`manager.py`](../src/trt_llm_langchain/manager.py): `load()` now detects CUDA OOM in the
    failed-load response (`_looks_like_oom`) and raises `InsufficientVramError` with actionable
    guidance ("…a backend restart is usually required… See WI #91") plus best-effort free VRAM
    via a new `free_vram_gb()` (nvidia-smi). Non-OOM load failures still raise plain
    `ModelLoadError`. Callers catching `ModelLoadError` still catch both.
- **LCEL example** — [`examples/lcel_chain.py`](../examples/lcel_chain.py) (`prompt | model |
  StrOutputParser`).
- **Swap example** — [`examples/swap_models.py`](../examples/swap_models.py): constructs qwen
  then llama; catches `InsufficientVramError` and prints the guidance instead of leaking a CUDA
  stack. Completes cleanly once #91 is resolved (or the backend is restarted between models).
- **Tests (offline):**
  - [`tests/test_vram.py`](../tests/test_vram.py) — OOM body ⇒ `InsufficientVramError`;
    non-OOM ⇒ plain `ModelLoadError`; subclass relationship.
  - [`tests/test_lcel.py`](../tests/test_lcel.py) — ChatTrtLlm is a `Runnable`, LCEL pipe
    constructs without network, `bind_tools(...)` returns a runnable.

## Decisions & discoveries

- **Reactive OOM detection over a size precheck.** The manager doesn't know a model's size from
  the KServe index, and a client-side nvidia-smi precheck only works when co-located with the
  GPU. So the robust, portable approach is to detect OOM in the load failure and translate it to
  a clear, typed error — works for any backend, no registry duplication. Free VRAM is added to the
  message best-effort (nvidia-smi), not relied upon.
- **The swap example turns the #91 limitation into demonstrated, handled behavior** rather than
  hiding it — a better artifact than pretending hot-swap is seamless.

## Outcomes

- `uv run ruff check` clean; `uv run pytest -q` → **17 passed**.
- **Live-verified:** `examples/lcel_chain.py` →
  _"A CUDA stream is a sequence of operations that can be executed concurrently with other
  streams on a GPU."_ (qwen, through the LCEL pipe.)
- **Deferred to post-#91 (needs backend):** live qwen→llama swap via `swap_models.py`; live
  `bind_tools` tool-calling (model-dependent — wiring verified offline only).

## Follow-ups

- After **WI #91**: run `swap_models.py` end-to-end; confirm whether a clean in-process swap is
  achievable or whether the documented path is "restart backend to swap" (and, if the latter,
  decide whether `ensure_loaded` should surface that as the primary guidance — already does).
- Live `bind_tools` / `with_structured_output` against qwen (the proxy has `tool_parsers.py`);
  document which served models actually support tool calls.
- Requires llama-3_1-8b-fp16 engine to be **built + set up** in the backend for the swap demo
  (verify during the post-#91 run).
