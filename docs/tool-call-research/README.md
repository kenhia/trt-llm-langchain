# Tool-calling research

_Date: 2026-06-29 · Scope: why tool calling doesn't work today across `trt-llm-langchain` +
`trt-llm-explore`, what the landscape offers, and the options (incl. a more-capable sibling
backend). Sources + dates in [`sources.md`](sources.md); claims tagged **[V]** verified-from-source
or **[I]** inferred._

> Companion to korg **explore #94** (investigate tool emission) and **trt-llm-langchain #95**
> (coordinate client-side testing). Context: a single RTX 5090 (Blackwell, sm_120, 32 GB).

---

## TL;DR

1. **Tool parsing is a *serving-layer* feature, not an engine feature.** The Triton
   `tensorrtllm_backend` only emits raw detokenized text; turning that text into OpenAI
   `tool_calls` happens in the layer above (Triton's OpenAI frontend, `trtllm-serve`, vLLM, NIM,
   or — in your case — the hand-rolled `openai_proxy.py`). **[V]**
2. **Your `trt-llm-explore` proxy already implements that layer — but only for non-streaming.** It
   injects tool defs into the prompt and parses `tool_calls` for `llama`/`qwen`/`mistral` on the
   non-streaming path; the **streaming path does no tool parsing at all** (always returns plain
   text + `finish_reason: "stop"`). This is **documented** in explore. **[V]**
3. **So the streaming-vs-tools split you remembered is real and recorded** — in
   `trt-llm-explore` (not poc): `docs/usage.md:497` "Streaming tool call parsing is not
   supported", and `specs/005-openai-proxy-enhancements/`. **[V]**
4. **Our failing test (`ChatTrtLlm().bind_tools(...).invoke()` on qwen-coder-7B → `tool_calls: []`)
   was non-streaming**, so the parser *did* run. The empty result points elsewhere — most likely
   the prompt template didn't inject the tools (proxy fell back to a tool-less format) and/or
   qwen-**coder**-7B is weak at tool syntax. **This is the #94 thing to verify, and it's cheap.**
   **[I]**
5. **Streaming + tools is fragile *everywhere*** (TRT-LLM, vLLM, and LangChain's own
   `ChatOpenAI` parser) — it's an industry-wide rough edge, not just your stack. The common
   workaround is `streaming=False` for tool turns. **[V]**
6. **A "more capable backend" is real and reasonable.** `trtllm-serve` (TRT-LLM's PyTorch
   backend) and **vLLM** both have first-class, configurable tool parsers and run on a 5090;
   **vLLM is the lowest-effort, most-mature tool-calling path** today. NIM is OpenAI-clean but the
   5090 is off its certified matrix (beta WSL2 track) and production needs a paid license.

**Shortest path to working tools:** verify + fix explore's **non-streaming** tool path on a
capable model (Llama-3.1-8B), keep tools non-streaming in the client. **Most capable path:** a
**vLLM sibling backend** behind the same `ChatTrtLlm`/`ChatOpenAI` client.

---

## 1. The core mechanism (why "tools don't work")

Tool calling over an OpenAI `/v1` surface is two jobs, both *above* the inference engine:

1. **Inject** the tool definitions into the prompt (via the model's chat template), so the model
   knows the tools exist and the syntax to call them.
2. **Parse** the model's emitted tool-call text (each family uses a different syntax) back into
   the structured `message.tool_calls` array.

The `tensorrtllm_backend` (C++ TensorRT engines, the 3-stage Triton ensemble) does **neither** —
it returns a `text_output` tensor of raw text. **[V]** Every OpenAI-compatible server adds these
two jobs:

- **Triton's own OpenAI frontend** — added tool calling in **25.05** (May 2025) via
  `--tool-call-parser {llama3|mistral}`; without the flag, tool calling is simply off. **[V]**
- **`trtllm-serve`** (TRT-LLM's server, PyTorch backend) — `--tool_parser` (auto, qwen3,
  deepseek_v3, …), still marked **prototype**. **[V]**
- **vLLM** — `--enable-auto-tool-choice --tool-call-parser <…>` (24 parsers as of Jun 2026). **[V]**
- **Your `openai_proxy.py`** — a bespoke implementation of the same layer.

So "tools don't work" is never the GPU/engine; it's whether the serving layer injects + parses,
for the streaming or non-streaming path, for that model's format.

---

## 2. What `trt-llm-explore` actually has today

From the code (`src/openai_proxy.py`, `src/tool_parsers.py`, `src/chat_templates.py`,
`models.toml`):

| Capability | Non-streaming path | Streaming path |
|---|---|---|
| Accepts `tools` / `tool_choice` | ✅ (request schema) | ✅ |
| **Injects** tool defs into prompt | ✅ `render_prompt(tools=…)` | ✅ (same call) |
| **Parses** `tool_calls` from output | ✅ `parse_tool_calls()` | ❌ **none** — streams raw text |
| `finish_reason: "tool_calls"` | ✅ when parsed | ❌ always `"stop"` |

**[V]** all rows. The streaming gap is **by design and documented**
(`docs/usage.md:497`, `specs/005-openai-proxy-enhancements/openai-proxy-handoff.md:190`).

**Per-model formats are already mapped** (explore `specs/005…/research.md`, `tool_parsers.py`):

| Architecture | Emitted format | Parser | Notes |
|---|---|---|---|
| `llama` (Llama 3.1) | `{"name":…,"parameters":…}<\|eot_id\|>` | `_parse_llama` | "8B has limited multi-tool reliability" |
| `qwen` (Qwen2.5) | `<tool_call>{"name":…,"arguments":…}</tool_call>` | `_parse_qwen` | Hermes-style; multiple blocks for multi-call |
| `mistral` | `[TOOL_CALLS] [{…}]` | `_parse_mistral` | needs `tool_format="mistral"` (mistral is built as a `llama` arch) |
| `phi` | none | — | returns `[]` |

Only `mistral-7b-fp16` sets `tool_format` in `models.toml`; others default to `architecture`.
`trt-llm-poc` has **no** tool support. **[V]**

### Why our qwen test still returned `tool_calls: []` (the #94 lead)

`ChatTrtLlm().bind_tools([add]).invoke(...)` is **non-streaming**, so `_parse_qwen` *ran*. Yet
empty. The likely causes, in order — to confirm in #94:

1. **Prompt injection fell back to tool-less format.** `render_prompt` uses the model's
   tokenizer `chat_template`; **if the template/config isn't found it silently falls back to
   `_fallback_format`, which doesn't inject tools** — so the model never sees them. **[I]** Check
   whether qwen-coder's tokenizer config + chat_template are actually loaded (vs fallback).
2. **Model capability.** Qwen2.5-**Coder**-7B is a code model; tool/`<tool_call>` reliability is
   weaker than instruct models. Re-test on **Llama-3.1-8B** (a first-class tool model with a
   parser already present). **[I]**
3. **`with_structured_output(method="function_calling")` also failed** (model returned prose),
   consistent with (1)/(2): the tool/function schema wasn't effectively prompted. `json_mode`
   works because it doesn't rely on tool emission. **[V] (observed)**

**Cheap first experiment (#94):** load `llama-3_1-8b-fp16`, call `bind_tools` **non-streaming**,
and inspect (a) the rendered prompt (real template vs fallback) and (b) raw `text_output`. That
isolates injection-gap vs model-capability in minutes.

---

## 3. The streaming-vs-tools tension (it's everywhere)

Your proxy's "no tool parsing while streaming" mirrors an industry-wide rough edge:

- **TRT-LLM:** Harmony control tokens leak into tool output in streaming (#9256, open);
  raw `<tool_call>` tags left in `content` instead of `tool_calls` (#9784); buggy streaming
  delta path (#3280); Triton streaming hard-requires `decoupled` mode (tensorrtllm_backend
  #626). **[V]**
- **vLLM:** streaming `tool_calls` missing `id` (#18412); Hermes parser returns raw text while
  streaming (#31871); Qwen3+Hermes truncates `}` (#19056). **[V]**
- **LangChain `ChatOpenAI` itself:** streaming tool-call assembly keys on a stable integer
  `index` with `id`/`name` only on the first delta and `arguments` concatenated across chunks
  (the OpenAI spec). Non-OpenAI backends that emit `index=null`, repeat indices, or fragment
  arguments break the merge → empty/`{}`/invalid tool calls (langchain #35514, #30563; langchainjs
  #8394, #9237). The documented workaround is **`streaming=False` for tool turns**, and LangChain
  explicitly recommends a provider-specific package over `ChatOpenAI(base_url=…)`. **[V]**

**Implication for `ChatTrtLlm`:** even with a perfect backend, *streaming* tool calls are the
fragile path. Treat **non-streaming as the supported tool path**; if we want streaming tools
later, both the backend's streaming parser *and* the client's delta assembly must be correct.

---

## 4. The two TRT-LLM backends (the "more capable" question)

TRT-LLM has two execution paths, and tool support tracks the **serving layer** in front of them:

- **TensorRT engine path (today):** `trtllm-build` → C++ engine → **Triton `tensorrtllm_backend`**.
  Raw text out; tools only if the *front* layer parses them. Your custom proxy is that front
  layer. Triton's *own* OpenAI frontend can do `--tool-call-parser {llama3|mistral}` (25.05+). **[V]**
- **PyTorch backend (`trtllm-serve`):** the newer LLM-API path that `trtllm-serve` defaults to in
  recent releases; this is where TRT-LLM's first-party `--tool_parser` (prototype) lives, with a
  broader, growing parser list. **[V/I]** This is the "more capable" *within* the TRT-LLM family.

So a "more capable sibling using a better backend" can mean either (a) **`trtllm-serve` (PyTorch
backend)** — stay in TRT-LLM, get first-party tool parsers + streaming; or (b) **leave TRT-LLM**
for vLLM/SGLang/NIM, which have more mature, better-documented tool calling.

---

## 5. Options & comparison

Effort is rough wall-clock for *you* (familiar with this stack), single 5090. **"Journey value"**
notes what you'd learn — per your note, effort is **not** a disqualifier.

| Option | Tool maturity | Streaming+tools | 5090/Blackwell | Setup effort | `ChatTrtLlm` fit | Journey value |
|---|---|---|---|---|---|---|
| **A. Fix explore proxy (non-stream)** | Medium (you own it) | No (stays non-stream) | ✅ (already running) | **S** (hours) | ✅ drop-in | Deep: you build the parser/inject layer |
| **B. + add streaming tool parsing to proxy** | Medium | Yes (you implement) | ✅ | **L** (assemble streamed deltas per-format) | ✅ | Highest learning; you implement the hard part everyone gets wrong |
| **C. Triton OpenAI frontend (`--tool-call-parser`)** | Medium (llama3/mistral only) | Partial | ✅ (Triton 25.05) | **M** (swap your proxy for Triton's frontend) | ✅ | Learn the official frontend; lose your custom control |
| **D. `trtllm-serve` (PyTorch backend) sibling** | Medium — parsers: qwen3/deepseek_v3.x/glm4/kimi_k2/… **no llama3/mistral/hermes** | **Yes — verified in source** (`parse_streaming_increment`, since ~v1.2) | ⚠️ sm_120 friction (FMHA #11799 open; kernels landing in 1.3 RCs Jun 2026) | **M–L** (new serving path; 5090 may need source build) | ✅ same `/v1` | Stay in TRT-LLM, learn the modern PyTorch path |
| **E. vLLM sibling** ⭐ | **High** (~24 parsers; hermes/llama3_json/mistral/qwen/deepseek/…) | Supported (per-parser bugs, esp. hermes) | ✅ since **0.17.0** (SM120 FP8 GEMM; CUDA 12.8/torch 2.6+) | **S–M** (`uv pip install vllm` + `vllm serve … --tool-call-parser`) | ✅ same `/v1` | Learn the most-used OSS server; easiest tools |
| **F. SGLang sibling** | High (llama3/llama4/mistral/qwen/deepseek/gpt-oss/pythonic; clean `tool_calls`) | Supported (Xgrammar backend; recent multi-tool fixes) | ⚠️ sm_120 rough (FP8-blockwise + kernel issues; needs cu129/cu13 + recent release) | **M–H** | ✅ same `/v1` | Learn a top-tier server; less of your prior context |
| **G. NIM** | High (vLLM-backed) | Documented joint support | ⚠️ 5090 **not** on certified matrix; beta WSL2 track | **M** (container + parser flags) + license | ✅ `/v1` | Least "journey" (black box); license cost for prod |

Notes/citations: tool parsers & flags **[V]** (vLLM, Triton 25.05, trtllm-serve, NIM); 5090 vLLM
sm_120 was source-build, now runs (~140 tok/s Qwen3-14B-AWQ under WSL2, #37242, Mar 2026) **[V]**;
NIM 5090 exclusion from the certified matrix **[V]**; SGLang Blackwell support **[I]** (verify).

### Tool-parser coverage by serving layer (which can structure which model)

Each model family emits a different tool-call syntax, and each serving layer ships a different set
of parsers. A model only produces structured `tool_calls` if the layer in front of it has a parser
for that family **and** the chat template injects the tools. ✅ = parser exists; ❌ = no parser
(tool text lands in `content`).

| Model family | Emitted format | Triton frontend | `trtllm-serve` | vLLM | SGLang | explore proxy (non-stream) |
|---|---|---|---|---|---|---|
| Llama 3.1/3.3 | JSON / `<\|python_tag\|>` | ✅ `llama3` | ❌ | ✅ `llama3_json` | ✅ `llama3` | ✅ `_parse_llama` |
| Qwen 2.5 / Qwen3 | `<tool_call>…</tool_call>` | ❌ | ✅ `qwen3`/`qwen3_coder` | ✅ `hermes`/`qwen` | ✅ `qwen` | ✅ `_parse_qwen` |
| Mistral / Nemo | `[TOOL_CALLS] […]` | ✅ `mistral` | ❌ | ✅ `mistral` | ✅ `mistral` | ✅ `_parse_mistral` |
| Hermes (Nous) | `<tool_call>` ChatML | ❌ | ❌ | ✅ `hermes` | ~ (qwen-like) | ~ (qwen parser may catch) |
| DeepSeek V3.x | DeepSeek template | ❌ | ✅ `deepseek_v3/31/32` | ✅ `deepseek_v3` | ✅ `deepseekv3` | ❌ |
| GLM-4.x | GLM template | ❌ | ✅ `glm4/glm47` | ✅ `glm45/glm47` | ✅ `glm` | ❌ |
| gpt-oss | Harmony channels | ❌ | ⚠️ buggy (#7163/#9256) | ⚠️ buggy | ⚠️ filters analysis | ❌ |
| Phi-3.5 | none | ❌ | ❌ | ❌ | ❌ | ❌ (returns `[]`) |

Takeaways: **vLLM has the broadest coverage** (incl. `hermes`, the one your explore proxy and
trtllm-serve both lack); **`trtllm-serve` skews to recent/Chinese-lab models and notably lacks
`llama3`/`mistral`/`hermes`**; **your explore proxy already covers llama/qwen/mistral
non-streaming** — competitive for those three families, just not streaming and not the newer
model formats. **[V]** parser lists; ~ and ⚠️ are **[I]**/known-buggy.

### Pros / cons of the leading options

**A/B — Invest in your own proxy.**
- ➕ You already have the parsers (llama/qwen/mistral) and full control; no new infra; pure
  drop-in for `ChatTrtLlm`. ➕ (B) implementing streamed tool-delta assembly is the single most
  educational piece in this whole space. ➖ You're maintaining parser logic that vLLM/SGLang
  maintain for free; (B) is genuinely hard to get right (see §3's bug list).

**D — `trtllm-serve` PyTorch backend.**
- ➕ Stays in the TRT-LLM family you've invested in; first-party tool parser + streaming;
  modern path. ➖ Prototype tool parser; different engine/serving model than your Triton setup
  (rebuild/relearn); may also inherit TRT-LLM's streaming-tool bugs (§3).

**E — vLLM sibling (recommended for capability/effort).**
- ➕ Lowest-effort path to *working, configurable* tool calling; biggest model+parser coverage;
  same `ChatOpenAI` drop-in; huge community. ➕ Also sidesteps the VRAM-on-unload swap pain
  differently (vLLM "sleep mode"). ➖ 5090 needed source builds historically (now better, verify
  your CUDA/driver); streaming-tool parser bugs still exist per-model; LangChain's "use a
  provider-specific package" caveat (mitigated since you already subclass `ChatOpenAI`).

**G — NIM.**
- ➕ Cleanest OpenAI surface, vLLM under the hood, NVIDIA-supported. ➖ 5090 off the certified
  matrix (beta WSL2); production license (~$4,500/GPU/yr); most "black box" (least journey).

---

## 6. Recommendation

Two tracks, not mutually exclusive — and given you value the journey, doing both in sequence is
the richest path:

1. **Now / shortest path (Option A):** In `trt-llm-explore`, verify and fix the **non-streaming**
   tool path on **Llama-3.1-8B** (run the §2 experiment; fix the chat-template-injection fallback
   if that's the cause). In `trt-llm-langchain`, document that **tool calling = non-streaming**
   (set/encourage `streaming=False` for tool turns) and add a live `tool_calls` assertion once
   it works. This likely makes tools work with what you already have, and teaches you the
   inject+parse layer cold. _Effort: S._
2. **Next / most capable (Option E sibling):** Stand up a **vLLM** sibling to `trt-llm-explore`
   (same OpenAI `/v1` contract, so `ChatTrtLlm` points at it unchanged) launched with
   `--enable-auto-tool-choice --tool-call-parser <family>`. Compare tool-calling reliability and
   streaming behavior head-to-head with your TRT-LLM proxy. This is the "more capable backend"
   you're already considering, and it doubles as a real evaluation. _Effort: S–M._
3. **Optional deep cut (Option B or D):** implement **streamed** tool-delta assembly in your
   proxy (B) — the highest-learning task here — or evaluate **`trtllm-serve`'s PyTorch backend**
   (D) to stay in-family with first-party parsers. _Effort: L / M._

**Why this order:** Option A is cheap and likely sufficient for non-streaming tools; it also
de-risks the diagnosis (is it injection, model, or parser?). Option E gives you a capable,
low-effort baseline and a genuine A/B against TRT-LLM. B/D are where the deep learning lives, with
no time pressure.

Cross-cutting theme (ties to WI #91 VRAM): several pain points — VRAM-not-freed-on-unload,
streaming+tools — are **Triton-25.05 + custom-proxy specific**. A vLLM/`trtllm-serve` sibling may
dissolve *both* at once, which is the strongest argument for standing one up.

---

## 7. Open questions to resolve next (for #94/#95)

- [ ] Does explore's `render_prompt` use the real chat_template (tools injected) or
      `_fallback_format` for qwen-coder and llama-3.1? (Instrument/log the rendered prompt.)
- [ ] Does **non-streaming** `bind_tools` work on **Llama-3.1-8B** today? (Capable tool model +
      existing `_parse_llama`.)
- [ ] What raw `text_output` does the model emit for a tool prompt? (Confirms inject vs parse vs
      model.)
- [ ] vLLM on this exact 5090/driver/CUDA: does a prebuilt wheel work now, or still source-build?
- [ ] SGLang Blackwell/5090 status (verify — only [I] today).
- [ ] If we want streaming tools: design the client-side delta assembly (or rely on
      `streaming=False`).

See [`sources.md`](sources.md) for all citations with dates and verified/inferred tags.
