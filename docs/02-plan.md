# Plan: `trt-llm-langchain` — a `ChatTrtLlm` for LangChain

_Companion to [`01-research.md`](01-research.md). Target: drop-in swap for `ChatAnthropic`._

```python
from langchain_anthropic import ChatAnthropic
chat = ChatAnthropic(model="claude-sonnet-4-6")      # before
# ───────────────────────────────────────────────────────────
from trt_llm_langchain import ChatTrtLlm
chat = ChatTrtLlm(model="qwen2_5-coder-7b-fp16")     # after — same call-site surface
```

## Architecture

```
   your LangChain code
          │  invoke / stream / batch / bind_tools
          ▼
   ┌──────────────────────┐
   │   ChatTrtLlm         │  subclass of langchain_openai.ChatOpenAI
   │  (chat.py)           │  • ensures model is loaded before each call
   └───────┬──────────────┘
           │ holds
           ▼
   ┌──────────────────────┐      load/unload/status (KServe v2)
   │  TrtLlmManager       │ ───────────────────────────────────► :8000  Triton
   │  (manager.py)        │      POST /v2/repository/models/{c}/load|unload
   └───────┬──────────────┘      POST /v2/repository/index
           │ chat/stream (OpenAI)
           └──────────────────────────────────────────────────► :8003  OpenAI proxy
                                                                 POST /v1/chat/completions
```

Two HTTP surfaces, both **already running** in `trt-llm-explore`:
- **`:8003/v1`** — chat + streaming (inherited from `ChatOpenAI`, zero new transport code).
- **`:8000/v2/repository`** — load / unload / index (the manager).

The single new package is a pure client. No server code.

## Package layout

```
src/trt_llm_langchain/
  __init__.py        # exports ChatTrtLlm, TrtLlmManager, errors
  config.py          # TrtLlmSettings: base_url(8003), control_url(8000), timeouts, env overrides
  errors.py          # TrtLlmError, ModelNotFoundError, ModelLoadError, ServerUnavailableError
  manager.py         # TrtLlmManager: list/status/load/unload/ensure_loaded over KServe v2
  chat.py            # ChatTrtLlm(ChatOpenAI): lazy ensure_loaded + delegated chat/stream
examples/
  basic_chat.py      # ChatTrtLlm(model=...).invoke([...])
  streaming.py       # for chunk in chat.stream(...)
  lcel_chain.py      # prompt | ChatTrtLlm(...)
  swap_models.py     # construct two models, show unload→load on switch
tests/
  test_manager.py    # unit: registry resolution, ensure_loaded logic (mock KServe)
  test_chat.py       # unit: lazy-load hook fires once; message passthrough (mock proxy)
  test_live.py       # integration (opt-in marker): real server smoke test
```

Dependencies to add: `langchain-core`, `langchain-openai`, `httpx`. (Drop the `trt-llm-langchain`
console-script stub in `pyproject.toml` unless you want a CLI.)

## Design decisions

1. **Subclass `ChatOpenAI`, don't reimplement `BaseChatModel`.** Inherit `invoke/stream/batch/
   ainvoke/astream`, `bind_tools`, `with_structured_output`. Override `__init__` to inject
   `base_url`/`api_key` and capture the trt model + manager; override the four generate hooks
   (`_generate`, `_agenerate`, `_stream`, `_astream`) to call `self._ensure_loaded()` then
   `super()`. This is the smallest correct surface.

2. **Lazy load on first call, not in `__init__`.** Constructing the object shouldn't silently
   evict 14 GB of VRAM. `_ensure_loaded()` runs once (idempotent) before the first generation.
   Provide `eager_load=True` to opt into load-at-construction for parity with a "warm" model.

3. **Server is the registry.** Resolve/validate `model=` against the live KServe repository index
   (`POST /v2/repository/index`) and/or `GET /v1/models`. Unknown model → `ModelNotFoundError`
   listing what's available. No second hand-maintained registry. (Optional: cross-check
   `trt-llm-explore/models.toml` for nicer error messages / capabilities.)

4. **Ensemble-aware load/unload.** A model key maps to its 5 components
   (`preprocessing`, `tensorrt_llm`, `postprocessing`, `tensorrt_llm_bls`, `ensemble`,
   prefixed/suffixed per explore's convention). Reuse explore's load ordering + readiness poll;
   unload in reverse. Encapsulate this so callers only ever say `ensure_loaded("qwen2_5-coder-7b-fp16")`.

5. **Single-GPU swap is RESTART-based** (revised in Sprint 3 after explore WI #91). A clean unload
   does *not* reclaim VRAM (TRT-LLM `cudaMallocAsync` pool retention), so in-place unload→load
   OOMs. `ensure_loaded(target)`: no-op if `target` is RESPONSIVE; else if a *different* model is
   resident, restart the backend (reclaims VRAM) then load `target`; else fresh-load. Restart is
   an opt-in strategy (`restart_backend` callable or `TRTLLM_RESTART_CMD`); without it a swap
   raises `BackendRestartRequiredError` rather than attempting the OOM-ing load. A reactive
   `InsufficientVramError` backstops any load that still OOMs. See
   [`sprints/sprint-03-swap-lcel.md`](../sprints/sprint-03-swap-lcel.md).

6. **Streaming is free.** Inherited from `ChatOpenAI`. Set `stream_usage=True` by default since
   explore's proxy supports `stream_options.include_usage` (verify in smoke test).

7. **Fail loud on a down server.** `_ensure_loaded` / first call should raise
   `ServerUnavailableError` with the URL if `:8000`/`:8003` aren't reachable — no silent hangs.

## `ChatTrtLlm` sketch (target API)

```python
class ChatTrtLlm(ChatOpenAI):
    def __init__(self, model: str, *, eager_load: bool = False,
                 settings: TrtLlmSettings | None = None, **kw):
        s = settings or TrtLlmSettings.from_env()
        super().__init__(model=model, base_url=f"{s.base_url}/v1",
                         api_key="trt-llm", stream_usage=True, **kw)
        self._manager = TrtLlmManager(s)
        self._manager.validate(model)            # ModelNotFoundError if unknown
        self._loaded = False
        if eager_load:
            self._manager.ensure_loaded(model); self._loaded = True

    def _ensure_loaded(self):
        if not self._loaded:
            self._manager.ensure_loaded(self.model_name)   # unload current → load this
            self._loaded = True

    def _generate(self, *a, **k):  self._ensure_loaded(); return super()._generate(*a, **k)
    def _stream(self, *a, **k):    self._ensure_loaded(); yield from super()._stream(*a, **k)
    async def _agenerate(self, *a, **k): self._ensure_loaded(); return await super()._agenerate(*a, **k)
    async def _astream(self, *a, **k):
        self._ensure_loaded()
        async for c in super()._astream(*a, **k): yield c
```
_(Pydantic-field details — storing `_manager`/`_loaded` as private attrs on a pydantic model —
get sorted during implementation; `model_config`/`PrivateAttr` as needed.)_

## The bigger goal: a package others can consume

This isn't just a personal tool. The end state is something **anyone wanting to use LangChain
with TensorRT-LLM can install and run**. That has two parts:

1. **The client** (`trt-llm-langchain`) — pip-installable, lean, well-documented.
2. **A serving backend** — today that's `trt-llm-explore`, which is **not yet published** and
   almost certainly contains Ken-specific assumptions (hard-coded paths, `$TRTLLM_HOME` layout,
   `just`/compose wiring, host details). For others to consume the whole experience, the backend
   has to be generalized too.

So Phase 2 (after the thing works locally) tackles **packaging + generalization + the
client/server relationship decision**.

### The client ↔ `trt-llm-explore` relationship (decision for Sprint 5)

How should the two repos relate so the result is cohesive for outsiders? Options:

- **A. Merge / vendor the server into this repo.** One clone + `docker compose up` = full stack.
  - ✅ Single cohesive artifact; easiest "it just works" story.
  - ❌ Couples client releases to heavy server tooling (Docker, engine builds); bloats a repo
    that LangChain users may want to `pip install` thin; duplicates explore if it lives on.
- **B. Two separate published repos joined by a documented contract.** `trt-llm-explore`
  (generalized) is the reference server; `trt-llm-langchain` is a lean client targeting the
  contract.
  - ✅ Clean separation; lean pip install; server reusable by non-LangChain users.
  - ❌ Two things to install/coordinate; cohesion lives in docs + a quickstart.
- **C. Hybrid (leaning recommendation): client is primary + a defined backend contract + an
    optional companion server.** Define the **backend contract** (OpenAI `/v1` chat/stream +
    KServe v2 load/unload + the `{pipeline}_{key}` naming convention). The client depends only on
    the contract, so it works against explore *or* any conforming server (incl. "bring your own"
    `trtllm-serve` for chat-only, no swap). Ship a one-command quickstart (compose/submodule
    pointer) that stands up explore as the default backend.
  - ✅ Lean client, broad compatibility, still a cohesive quickstart; decouples release cadence.
  - ❌ Must write + maintain the contract spec and keep explore conformant.

**Decided (Sprint 5): option C** — see [ADR 0001](decisions/0001-backend-integration-strategy.md).
The client is the published product; it depends only on the
[backend contract](03-backend-contract.md). The reference backend (`trt-llm-explore`) will be
cleaned up, likely **renamed**, and published with setup docs coordinated to the contract
(tracked in that project, korg WI #93). Client docs refer to the backend by role so a rename
doesn't break them.

## Build sprints

Each sprint is logged in [`sprints/`](../sprints/) (see that dir's README for the convention) —
a dated, published narrative of goal → decisions → what shipped → outcomes → follow-ups, so the
project's evolution is legible to Ken and to future readers.

### Phase 1 — Make it work locally (Sprints 1–4)

**Sprint 1 — Manager + connectivity (no LangChain yet).**
`config.py`, `errors.py`, `manager.py`. `list_models()`, `status()`, `load()`, `unload()`,
`ensure_loaded()` against the live `:8000` API, deriving the registry + vision classification from
the KServe repository index. CLI/REPL smoke test: load qwen, check RESPONSIVE, unload.
_Exit: can drive load/unload from Python against the running explore stack._

**Sprint 2 — `ChatTrtLlm` over `ChatOpenAI`.**
`chat.py` with lazy `_ensure_loaded`. `invoke` and `stream` work end-to-end through `:8003`.
`examples/basic_chat.py`, `streaming.py`. _Exit: the two-line swap at the top of this doc runs._

**Sprint 3 — Model swapping + LCEL.**
`examples/swap_models.py` (construct qwen then llama, observe unload→load), `lcel_chain.py`
(`prompt | model`). Confirm `bind_tools` / `with_structured_output` behavior (model-dependent;
document what works on your served models). _Exit: switching models and LCEL both work._

**Sprint 4 — Hardening + tests + local docs.**
Unit tests with mocked HTTP (`test_manager.py`, `test_chat.py`), opt-in live test
(`test_live.py`), error-path coverage (down server, unknown model, OOM on load). README with the
quickstart and the "start the explore stack first" prerequisite. _Exit: `uv run pytest` green;
README lets a fresh clone (with a running backend) run a chat._

### Phase 2 — Make it consumable by others (Sprints 5–6)

**Sprint 5 — Backend contract + the explore relationship decision.**
Write `docs/03-backend-contract.md`: the exact HTTP surface the client requires (chat/stream
endpoints, load/unload/index endpoints, component naming, model-id semantics) so any conforming
server works. Decide A/B/C above and record it in an ADR (`docs/decisions/`). Audit
`trt-llm-explore` for Ken-specific assumptions (paths, `$TRTLLM_HOME`, host wiring) and file the
generalization work. _Exit: a written contract + a chosen, recorded integration strategy._

**Sprint 6 — Publish-ready packaging.**
De-Ken-ify config (env/`base_url`/`control_url` overrides, no hard-coded hosts), polished README
with a copy-paste quickstart, `LICENSE`, classifiers/keywords in `pyproject.toml`, CI
(lint + unit tests), and a tagged release / PyPI dry-run. Generalize + publish the backend per the
Sprint 5 decision, with a one-command "stand up a backend" path. _Exit: a stranger can install the
client, stand up a backend, and run the two-line swap from a clean machine._

## Open questions / things to verify during Sprint 1

- **Proxy port & path:** confirm explore's OpenAI proxy is on `:8003` and started (it's "optional"
  there). If you usually run only Triton-native, we either start the proxy or add a Triton-native
  transport fallback. _Recommend: depend on the proxy; add a `just`/compose note to bring it up._
- **Component naming:** confirm the exact component names per model key (explore uses
  `ensemble_{key}` etc.; the POC uses `tensorrt_llm_qwen` style). The manager should derive these
  the same way explore's `triton_client.py` does — reuse that logic rather than re-deriving.
- **`stream_usage`:** verify the proxy emits usage in the final SSE chunk; if not, default it off.
- **Reasoning/tool models:** if you serve models that emit non-OpenAI fields, revisit the
  `ChatOpenAI` "official-spec-only" caveat (research §2) before relying on tool calls.

## Out of scope (for now)

- Auto-starting the Triton/proxy stack from Python (assume it's running; document the prereq).
  A one-command quickstart to *stand up* a backend is in Phase 2, but the client won't manage the
  server lifecycle at runtime.
- Multi-GPU / holding two models resident (single 5090 → one at a time).
- Vision models (`qwen2-vl-7b-fp16`) — the chat path may work, but multimodal message handling is
  a separate sprint.

_(Publishing to PyPI and generalizing the backend moved **into** scope as Phase 2 — see the
sprints above.)_
