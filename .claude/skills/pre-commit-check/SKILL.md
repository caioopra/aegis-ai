---
name: pre-commit-check
description: Run the full pre-commit gate for AegisNode — unit tests, ruff check, and ruff format — and report whether the working tree is safe to commit. Use before any git commit.
allowed-tools: Bash, Read
---

# Pre-Commit Check

A single gate that must pass before creating a commit. Runs the unit
test suite and ruff, then reports a clear GO / NO-GO verdict.

## Steps

Run these in sequence and collect results:

1. **Unit tests** (fast path, no integration):
   ```bash
   uv run pytest -m "not integration" -q
   ```

2. **Lint**:
   ```bash
   uv run ruff check src/ tests/ scripts/
   ```

3. **Format check** (apply, then inspect `git status`):
   ```bash
   uv run ruff format src/ tests/ scripts/
   ```

## Verdict

- **GO** — all three commands succeed. Report: "Safe to commit.
  N unit tests passing, lint clean, format clean."
- **NO-GO** — any command fails. Report:
  - Which stage failed.
  - Short summary of the first failure (stop at the first red flag,
    don't run subsequent checks blindly).
  - Which specialist should own the fix (see escalation list below).
  - The exact command the user can run locally to reproduce.

## Escalation

| Failure type | Specialist |
|---|---|
| Test failure in `tests/test_llm.py` | llm-prompt-specialist |
| Test failure in `tests/test_agent_*.py` | agent-graph-architect |
| Test failure in `tests/test_rag.py` | rag-retrieval-specialist |
| Test failure in `tests/test_fhir.py` / `tests/test_mcp_server.py` | mcp-fhir-specialist |
| Test fixture / conftest issue | test-quality-engineer |
| Ruff rule violation | the specialist who owns the file |
| Format-only changes | auto-apply and continue (not a blocker) |

## What NOT to do

- Do NOT bypass failing tests with `-k 'not broken'`.
- Do NOT run `ruff check --fix` — auto-fix can change behavior.
- Do NOT mark a commit safe just because format changes were applied;
  tests and lint must also be green.
