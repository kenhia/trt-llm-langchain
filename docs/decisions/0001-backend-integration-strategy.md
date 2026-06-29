# ADR 0001 — Backend integration strategy

_Status: accepted · Date: 2026-06-29 · Sprint: 5_

## Context

`trt-llm-langchain` (the LangChain client) needs a TensorRT-LLM serving backend. The reference
backend today is `trt-llm-explore`, which is not yet published and contains environment-specific
assumptions. For Phase 2 (a package others can consume) we must decide how the two relate. Options
considered (see [`../02-plan.md`](../02-plan.md)): **A** merge/vendor the server into this repo,
**B** two separate published repos joined by a contract, **C** client-primary + a defined backend
contract + an optional companion server.

A key data point: the client is already consumed standalone — installed into a separate
"Learn LangChain" project (`gen-ai-langchain`) and used from a notebook successfully. The client's
value is independent of any one backend.

## Decision

**Option C.** `trt-llm-langchain` is the primary, published, pip-installable product. It depends
only on the **backend contract** ([`../03-backend-contract.md`](../03-backend-contract.md)), so it
works against the reference backend, any conforming server, or a plain `trtllm-serve` (chat-only,
no swap). We will ship a one-command quickstart that stands up the reference backend as the
default. The repos stay separate; cohesion lives in the contract + quickstart + coordinated setup
docs.

## Consequences

- The client never bundles Docker/engine-build tooling; it stays lean.
- The contract doc is the load-bearing artifact — it must stay accurate as the backend evolves.
- **The reference backend (`trt-llm-explore`) will be cleaned up for publish** (~~and likely
  renamed~~ — rename dropped, see Update below). Therefore:
  - Environment-specific identifiers the client documents (e.g. the container name in
    `TRTLLM_RESTART_CMD` examples, `$TRTLLM_HOME`) are presented as *examples*, not fixed — they
    depend on the user's deployment.
- **Setup docs must be coordinated across the two projects** so a user who wants to run Ken's
  TRT-LLM server gets a clear, single path: the client README links to the backend's setup; the
  backend documents the contract it implements and the `up` / `restart` / `swap` recipes the
  client expects.
- Generalization work on the backend (de-Ken-ify paths, rename, publish, setup docs) is tracked in
  the backend's project, not here.

## Follow-ups

- Backend (separate project): generalize environment assumptions, publish, and write setup docs
  aligned to the contract. Tracked via korg WI #93 in that project.
- Client (Sprint 6, done): `ChatTrtLlm()` adopts the resident model; (Sprint 7) the quickstart
  that stands up the backend.

## Update (2026-06-29, Sprint 7)

After further discussion, Ken decided to **keep the `trt-llm-explore` name** — the rename is
**dropped**. The integration strategy (option C) is unchanged; the backend is still generalized for
publish (WI #93, minus the rename). Client docs now use the name `trt-llm-explore` directly while
still treating deployment-specific paths/container names as examples. (WI #93 should be updated to
drop its rename item.)
