# Sprint 7 — Publish-ready packaging

_Status: complete · Started: 2026-06-29 · Branch: `sprint-07-packaging`_

## Goal

Make the package publish-ready: license, metadata, CI, changelog, a "stand up a backend"
quickstart, and a build/metadata dry-run. (Final-name decision resolved — `trt-llm-explore` is
**kept**, so no placeholder needed.)

## What shipped

- **[`LICENSE`](../LICENSE)** — MIT, © 2026 Ken Hiatt (matches `trt-llm-explore`).
- **`pyproject.toml` metadata** — SPDX `license = "MIT"` + `license-files`, `keywords`, and
  `classifiers` (Beta, Py 3.11/3.12, Linux, AI, Pydantic). `[project.urls]` left commented until
  there's a public repo.
- **[`.github/workflows/ci.yml`](../.github/workflows/ci.yml)** — uv-based CI on push/PR, Python
  3.11 + 3.12 matrix: `uv sync` → `ruff check` → `pytest` (offline; live suite excluded by
  default).
- **[`CHANGELOG.md`](../CHANGELOG.md)** — Keep-a-Changelog style, `Unreleased` section covering the
  Phase 1–2 feature set.
- **README quickstart** — a "stand up a backend" section with the `trt-llm-explore` commands
  (`just up` / `just up-openai` / `just load`) and the `ChatTrtLlm()` adopt-resident one-liner.
- **Rename reconciliation** — Ken decided to keep the `trt-llm-explore` name; updated ADR 0001
  (addendum), the contract doc note, and korg WI #93 (dropped the rename item).

## Decisions & discoveries

- **PEP 639 license metadata.** Used the SPDX `license = "MIT"` expression + `license-files`
  (no deprecated `License ::` classifier), matching the modern `uv_build` backend. The built
  wheel METADATA shows `License-Expression: MIT` / `License-File: LICENSE`.
- **URLs deferred.** No public repo yet (local only), so `[project.urls]` is commented with the
  intended GitHub paths — uncomment when published.

## Outcomes

- `uv run ruff check` clean; `uv run pytest -q` → **28 passed, 4 deselected**.
- `uv build` → sdist + wheel built; **`twine check` PASSED** for both.
- Wheel sanity: `cli.py` and `LICENSE` bundled; METADATA carries license, keywords, classifiers,
  `Requires-Python >=3.11`, and the three runtime deps.

## Remaining to actually publish (needs Ken / external accounts)

- Create the public GitHub repo, push, and uncomment `[project.urls]`.
- Set the release version (drop `Unreleased` → `0.1.0`, tag), then `uv publish` with a PyPI token.
- Coordinate with `trt-llm-explore` publish (WI #93) so the quickstart link resolves.

## Follow-ups

- Optional: live `tool_calls` assertion on a tool-capable model; a `with_structured_output`
  example.
- If remote seamless swap is wanted later: explore Option B endpoint + client control-URL swap
  mode (ADR 0002).
