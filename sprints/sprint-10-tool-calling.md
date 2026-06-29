# Sprint 10 — Tool calling (Option A)

_Status: complete (live-validated) · Started: 2026-06-29 · Branch: `sprint-10-tool-calling`_

> Executes "Option A" from [`docs/tool-call-research/`](../docs/tool-call-research/README.md).
> korg: explore **#94** (resolved, PR #2), langchain **#95** (resolved).

## Goal

Make tool calling actually work, validate which models support it, and fold the result into both
repos.

## Diagnosis (on the GPU host)

The research's leading hypothesis (prompt-injection fallback) was **wrong** — injection worked
all along. The real causes:

1. **We had tested the wrong model.** Llama-3.1-8B did tools end-to-end immediately; the original
   failure was `qwen2_5-coder-7b`.
2. **Parsers were too strict** for what models actually emit, and the proxy rejected object-form
   `tool_choice`. Captured raw outputs: qwen-coder emits `<function_call>` / fenced JSON (not
   `<tool_call>`); mistral emits a bare `[{...}]` array (no `[TOOL_CALLS]`); llama-fp16 re-emits
   the call (duplicate).

## What shipped

**trt-llm-explore** (PR #2, `feat/tool-calling-fix`, `specs/007-tool-calling-fixes/`):
- `tool_parsers.py`: `<function_call>` + fenced/bare-JSON fallback (qwen), bare-array fallback
  (mistral), accept `arguments|parameters`, **dedup** exact duplicates. +5 unit tests (237 pass).
- `openai_proxy.py`/`chat_templates.py`: `tool_choice` accepts string **or** object (forced
  function) → `with_structured_output(method="function_calling")` no longer 422s.

**trt-llm-langchain** (this branch):
- [`examples/tool_calling.py`](../examples/tool_calling.py) — `bind_tools` walkthrough.
- `test_live.py`: `test_bind_tools_live` now **asserts** `tool_calls` (skips on tool-less models);
  json_mode structured test hardened (directive prompt, skip on non-adherence).
- Docs corrected: README limitations + getting-started now say tool calling works (non-streaming,
  tool-capable models); structured-output example notes both methods work.
- Research doc gets a "Option A shipped" outcome banner.

## Outcomes — validation matrix (live, non-streaming, `ChatTrtLlm.bind_tools`)

| Model | tool_calls | Note |
|---|---|---|
| llama-3_1-8b-fp16 | ✅ | was duplicated → deduped |
| llama-3_1-8b-fp8 | ✅ | clean |
| llama-3_1-8b-awq | ✅ | clean |
| qwen2_5-coder-7b-fp16 | ✅ | via `<function_call>` |
| qwen2_5-coder-7b-fp8 | ✅ | via fenced-JSON fallback |
| mistral-7b-fp16 | ✅ | via bare-array fallback |
| phi-3_5-mini-fp16 | ❌ | no native tool support (model limitation) |

Also: `with_structured_output(method="function_calling")` → `Person(name='Ada Lovelace', age=36)`
on Llama. Offline `pytest` → 28 passed; live tool tests pass; ruff clean.

## Decisions & discoveries

- **The fix is parser-tolerance, not model-perfection.** Real models emit messy/variant tool
  syntax; meeting them where they are (extra tag names, fenced/bare JSON, dedup) is higher-value
  than chasing one canonical format.
- **Streaming tools stay out of scope** — the backend's streaming path emits text only. Tool turns
  must be non-streaming (consistent with the cross-framework fragility in the research).

## Follow-ups

- Optional deep cut: streamed `delta.tool_calls` assembly in the proxy (the hard part everyone
  gets wrong) — separate sprint.
- The "most capable" path (vLLM sibling) from the research remains open if broader model/parser
  coverage or streaming tools become priorities.
- Phi tool support is a model limitation; nothing to fix client- or proxy-side.
