# ADR 0002 — Model swap strategy for the published product

_Status: accepted · Date: 2026-06-29 · Sprint: 6_

## Context

On a single GPU, swapping models requires a backend restart to reclaim VRAM (ADR-adjacent: see
`../03-backend-contract.md` §4 and `trt-llm-explore` WI #91). The client today does restart-based
swaps by shelling out a **local** command (`TRTLLM_RESTART_CMD`), which only works when the client
is co-located with the GPU host. For "ready for others," the question is whether swapping should be
made remotable (and seamless) — and if so, where the restart authority lives.

The `trt-llm-explore` agent proposed backend-side options (`.scratch/restart-discussion.md`):
**A** auto-swap inside `/v1/chat/completions`, **B** an explicit `POST /admin/swap` admin
endpoint, **C** a control sidecar — all requiring the backend to gain Docker/host restart
authority (`docker.sock` ≈ root-equiv; a real security surface). A fourth option is to **do
nothing** beyond a clear error.

## Decision

For the initial published product:

1. **Default = error well.** Without a configured swap mechanism, a swap raises
   `BackendRestartRequiredError` with actionable guidance (never a raw CUDA OOM). This is the
   honest baseline and adds no attack surface.
2. **Keep the optional local restart command** (`TRTLLM_RESTART_CMD`) for the co-located
   single-box user — seamless swap with no backend changes and no `docker.sock` in the server.
3. **`ChatTrtLlm()` with no model adopts the resident model** (never swaps) — the ergonomic
   "talk to whatever's loaded" path that pairs with a load-it-yourself workflow.
4. **Document server-side swap (Option B) as an OPTIONAL, FUTURE backend capability** in the
   contract — a token-protected control endpoint the client could call over HTTP to make swaps
   remotable. **Do not build it for v1.**

**Rejected for v1:**
- **Option A** (auto-swap hidden inside `/v1/chat/completions`) for a *published* contract:
  surprising ~restart latency and non-standard side effects on the standard OpenAI endpoint (a
  plain OpenAI client pointed at it would trigger swaps unexpectedly).
- **Building B/C now:** they require giving the backend restart authority (security cost) and are
  mostly backend work; not justified until remote/multi-user demand is real.

## Consequences

- The common consumers are covered: single-box local (optional `TRTLLM_RESTART_CMD`), one-model
  users (nothing needed), and load-it-yourself (`ChatTrtLlm()`). Remote clients get a clear error
  rather than silent failure.
- The client stays a pure HTTP client with no required server-lifecycle authority (consistent with
  ADR 0001 option C).
- When remote seamless swap is wanted, the path is pre-decided: backend adds a token-protected
  swap endpoint (Option B); the client gains a "control-URL swap" mode that calls it. The contract
  already reserves the slot.

## Follow-ups

- Backend project: if/when remote swap is wanted, implement Option B (token-protected
  `POST /admin/swap`, serialized with a lock, restart authority via a privileged swapper).
- Client: add a `swap via control endpoint` mode to `ensure_loaded` at that time.
