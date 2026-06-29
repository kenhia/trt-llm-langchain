# Sources

Citations for [`README.md`](README.md). **[V]** = verified from the cited primary source during
research (2026-06-29); **[I]** = inferred/synthesized. Several NVIDIA & OSS doc pages show a crawl
date (often "June 2026") rather than a true publication date — noted where it matters.

## Local code (trt-llm-explore / trt-llm-poc)

- Streaming tool calls "not supported": `trt-llm-explore/docs/usage.md:497`;
  `specs/005-openai-proxy-enhancements/openai-proxy-handoff.md:190`,
  `contracts/cli-contract.md:59`. **[V]**
- Proxy tool flow: `src/openai_proxy.py` — request schema accepts `tools`/`tool_choice` (≈167-175);
  non-streaming injects + parses (≈236-329); streaming path injects but does **not** parse
  (≈332-457, final chunk always `finish_reason:"stop"`). **[V]**
- Parsers: `src/tool_parsers.py` — `_parse_llama`/`_parse_qwen`/`_parse_mistral`; phi → `[]`. **[V]**
- Template injection: `src/chat_templates.py::render_prompt(tools=…)`, with silent
  `_fallback_format` when no tokenizer chat_template is found. **[V]**
- Per-model formats: `specs/005-openai-proxy-enhancements/research.md` (≈68-73),
  `openai-proxy-handoff.md` (≈126-131); only `mistral-7b-fp16` sets `tool_format` in
  `models.toml`. **[V]**
- `trt-llm-poc`: no tool-calling implementation. **[V]**

## Architecture: tool parsing is a serving-layer feature

- Triton `tensorrtllm_backend` returns raw `text_output`, no tool parsing:
  <https://github.com/triton-inference-server/tensorrtllm_backend> ;
  <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Feature_Guide/Function_Calling/README.html> **[V]**
- Triton OpenAI **frontend** does the parsing, `--tool-call-parser {llama3|mistral}` only, added in
  **25.05** (May 2025):
  <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client_guide/openai_readme.html> ;
  release notes <https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel-25-05.html> ;
  RFE <https://github.com/triton-inference-server/server/issues/8048> (Mar 2025). **[V]**

## trtllm-serve (TRT-LLM PyTorch backend)

- `--tool_parser` values (auto, qwen3, qwen3_coder, kimi_k2, deepseek_v3/31/32, gemma4, glm4/glm47,
  minimax_m2/m3, poolside_v1) — **no llama3/mistral/hermes**; OpenAI `/v1`:
  <https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html> (updated Jun 22, 2026). **[V]**
- PyTorch backend is **default** since 1.0 ("Change default backend to PyTorch in trtllm-serve"),
  TensorRT path still available via `TLLM_USE_TRT_ENGINE=1`:
  <https://nvidia.github.io/TensorRT-LLM/release-notes.html> **[V]**
- **Streaming tool deltas DO work** — source: `parse_streaming_increment` in
  `tensorrt_llm/serve/tool_parser/base_tool_parser.py` and `DeltaToolCall` emission in
  `tensorrt_llm/serve/postprocess_handlers.py` (raw.githubusercontent, main, fetched Jun 29 2026);
  framework landed PR #8216 (Oct 29 2025), ~v1.2 line. gRPC mode does **not** support
  `--tool_parser`. **[V]** (release-version mapping **[I]**)
- 5090/sm_120 friction: trtllm-gen FMHA cubins missing for sm_120
  <https://github.com/NVIDIA/TensorRT-LLM/issues/11799> (open, Feb 28 2026); sm_120/sm_121 FMHA
  kernels added in v1.3.0rc19 (Jun 23 2026) <https://github.com/NVIDIA/TensorRT-LLM/releases>.
  Support matrix lists datacenter Blackwell only
  <https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html> (Sep 15 2025). **[V]**

## Streaming + tools fragility (cross-framework)

- TRT-LLM: gpt-oss tool info dropped <https://github.com/NVIDIA/TensorRT-LLM/issues/7163> (open,
  Aug 22 2025); Harmony tokens leak <https://github.com/NVIDIA/TensorRT-LLM/issues/9256> (open,
  Nov 18 2025); raw `<tool_call>` tags left in content
  <https://github.com/NVIDIA/TensorRT-LLM/issues/9784> (Dec 8 2025); streaming delta bug
  <https://github.com/NVIDIA/TensorRT-LLM/issues/3280> (Apr 3 2025); Triton streaming requires
  decoupled <https://github.com/triton-inference-server/tensorrtllm_backend/issues/626> (Oct 20
  2024). **[V]**
- vLLM: hermes streaming returns raw text <https://github.com/vllm-project/vllm/issues/31871>
  (Jan 7 2026); streaming `tool_calls.id` missing
  <https://github.com/vllm-project/vllm/issues/18412>; Qwen3+Hermes truncation
  <https://github.com/vllm-project/vllm/issues/19056>. **[V]**
- OpenAI streaming `delta.tool_calls` spec (index-keyed; id/name first delta; args concatenated;
  terminal `finish_reason:"tool_calls"`):
  <https://developers.openai.com/api/docs/guides/function-calling> ;
  <https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events>. **[V]**

## LangChain ChatOpenAI tool-call parsing

- Streaming merge keys on `index`; non-OpenAI backends break it (empty/`{}`/invalid tool calls):
  langchain <https://github.com/langchain-ai/langchain/issues/35514> (Mar 2 2026),
  <https://github.com/langchain-ai/langchain/issues/30563> (Mar 31 2025); langchainjs
  <https://github.com/langchain-ai/langchainjs/issues/8394>, <…/issues/9237>. Workaround
  `streaming=False`. **[V]**
- `bind_tools` + custom `base_url` returns empty tool_calls when the endpoint lacks function
  calling <https://github.com/langchain-ai/langchain/issues/21887> (closed, not planned). **[V]**
- "Use a provider-specific package" caveat:
  <https://docs.langchain.com/oss/python/integrations/chat/openai>. **[V]**

## vLLM

- Tool calling flags + ~24 parsers; streaming via `extract_tool_calls_streaming`:
  <https://docs.vllm.ai/en/latest/features/tool_calling/> (Jun 23 2026). **[V]**
- Quickstart / `vllm serve`: <https://docs.vllm.ai/en/latest/getting_started/quickstart.html>
  (May 27 2026). **[V]**
- 5090/sm_120: feature issue closed-completed 2025-06-26
  <https://github.com/vllm-project/vllm/issues/13306>; working config ~0.17.x (≈140 tok/s
  Qwen3-14B-AWQ, WSL2) <https://github.com/vllm-project/vllm/issues/37242> (Mar 17 2026); Blackwell
  needs CUDA 12.8+ <https://docs.vllm.ai/en/stable/getting_started/installation/gpu/> (May 11
  2026). **[V]**

## SGLang

- Parser registry (llama3, llama4, mistral, qwen/qwen25, deepseekv3/31/32, gpt-oss, pythonic,
  qwen3_coder, glm/glm45/glm47, kimi_k2, step3…):
  `python/sglang/srt/function_call/function_call_parser.py` (main, Jun 29 2026) ;
  <https://docs.sglang.io/advanced_features/tool_parser.html>. **[V]**
- Streaming + tools first-class; `tool_choice` needs Xgrammar backend (default); use
  `/v1/chat/completions` not `/v1/responses`:
  <https://docs.sglang.io/advanced_features/function_calling.html>. **[V]**
- 5090/sm_120 rough: block-FP8 unsupported then closed
  <https://github.com/sgl-project/sglang/issues/9233> (Aug 2025); missing sm_120 kernels
  <https://github.com/sgl-project/sglang/issues/9542>; attention-backend bug
  <https://github.com/sgl-project/sglang/issues/14814>. Current release v0.5.14 (Jun 26 2026);
  use cu129/cu13 wheels. **[V]** dates; exact "fixed-in" version **[I]**.

## NVIDIA NIM

- vLLM-backed; tool calling via `--enable-auto-tool-choice --tool-call-parser` (or
  `NIM_PASSTHROUGH_ARGS`); OpenAI `/v1` (chat/completions/responses/models):
  <https://docs.nvidia.com/nim/large-language-models/latest/advanced-use-cases/tool-calling-and-mcp.html> ;
  <https://docs.nvidia.com/nim/large-language-models/latest/reference/api-reference.html> (Jun 26
  2026). **[V]**
- RTX 5090 **not** on certified support matrix (only RTX PRO Blackwell Server); beta WSL2 "NIM on
  RTX" track for GeForce 50-series:
  <https://docs.nvidia.com/nim/large-language-models/latest/reference/support-matrix.html> (Jun 26
  2026); <https://developer.nvidia.com/blog/kickstart-your-ai-journey-on-rtx-ai-pcs-and-workstations-with-nvidia-nim-microservices/>
  (Mar 25 2025). **[V]**
- Licensing: free for dev/research up to 16 GPUs; production needs NVIDIA AI Enterprise
  (~$4,500/GPU/yr or ~$1/GPU/hr; 90-day trial): <https://docs.api.nvidia.com/nim/docs/product> ;
  <https://developer.nvidia.com/blog/access-to-nvidia-nim-now-available-free-to-developer-program-members/>
  (Jul 29 2024). **[V]**

## Model-specific tool formats

- Llama 3 prompt format:
  <https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/prompt_format.md> ;
  Hermes-2-Pro tool template:
  <https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B/discussions/13> ; llama.cpp native
  parsers <https://github.com/ggml-org/llama.cpp/pull/9639>. **[V]**

---

### Research method note

Findings synthesized from one local-code exploration agent (explore/poc) and a fan-out of web
research sub-agents (Triton/backend architecture, trtllm-serve, vLLM, SGLang, NIM, streaming+tools
issues, OpenAI delta spec, LangChain client behavior). Where two sources conflicted (e.g. "PyTorch
backend doesn't support `--tool_parser`"), the source-code read was preferred over search
summaries; such corrections are flagged in-line above.
