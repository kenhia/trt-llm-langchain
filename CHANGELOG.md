# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions follow semver.

## [Unreleased]

## [0.1.0] - 2026-06-29

### Added
- `ChatTrtLlm` — a `langchain_openai.ChatOpenAI` subclass backed by a local TensorRT-LLM server;
  a call-site drop-in for `ChatAnthropic`/`ChatOpenAI` with `invoke`/`stream`/`batch`/async and
  inherited `bind_tools` / `with_structured_output`.
- Lazy model management: the requested model is made resident before the first call.
- `ChatTrtLlm()` with no `model` adopts the currently-resident model (never swaps).
- `TrtLlmManager` — model list/status/load/unload/`ensure_loaded` over the Triton KServe v2
  control plane; registry + vision classification derived from the repository index.
- Restart-based single-GPU model swap (`TRTLLM_RESTART_CMD` / `restart_backend`), with
  `BackendRestartRequiredError` when a swap needs a restart and none is configured.
- Typed errors: `ServerUnavailableError`, `ModelNotFoundError`, `ResidentModelError`,
  `ModelLoadError` (+ `InsufficientVramError`, `BackendRestartRequiredError`), `ModelUnloadError`.
- `trtllm-lc` control-plane CLI (`list`/`status`/`load`/`unload`/`ensure`).
- Examples (`basic_chat`, `streaming`, `lcel_chain`, `swap_models`), an opt-in live test suite,
  the backend contract (`docs/03-backend-contract.md`), and ADRs 0001–0002.

[Unreleased]: https://github.com/kenhia/trt-llm-langchain/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kenhia/trt-llm-langchain/releases/tag/v0.1.0
