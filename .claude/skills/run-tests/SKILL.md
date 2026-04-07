---
name: run-tests
description: Run the fast AegisNode unit test suite (mocked, no Ollama required) and report pass/fail counts plus any failures. Use this before committing code or when the user asks to check test status.
allowed-tools: Bash, Read
---

# Run Tests — Fast Unit Path

Run the mocked unit test suite and report results.

## What to do

1. Run the fast path (no integration tests, no network):

   ```bash
   uv run pytest -m "not integration" -q
   ```

2. Parse the output. Report:
   - Total passed / failed / skipped / deselected counts.
   - Duration.
   - If there are failures, read the first 3-5 and summarize each in
     one line: `tests/test_X.py::test_name — short reason`.
   - If all pass, report "All N unit tests passing".

3. If any test failed, do NOT try to fix it automatically. Surface the
   failure and hand off to the relevant specialist:
   - `tests/test_llm.py` → **llm-prompt-specialist**
   - `tests/test_agent_*.py` → **agent-graph-architect**
   - `tests/test_rag.py` → **rag-retrieval-specialist**
   - `tests/test_fhir.py` / `tests/test_mcp_server.py` → **mcp-fhir-specialist**
   - `tests/conftest.py` or fixture issues → **test-quality-engineer**

## What NOT to do

- Do not run `uv run pytest -m integration` — it requires a live
  Ollama instance and is out of scope for this skill.
- Do not run `uv run pytest` (without the marker) for the same reason.
- Do not edit test files to make them pass.

## Useful flags for diagnosis

```bash
uv run pytest -m "not integration" tests/test_llm.py -v     # single file verbose
uv run pytest -m "not integration" --durations=10           # find slow tests
uv run pytest -m "not integration" -x                        # stop at first failure
uv run pytest --collect-only -q                              # shape check only
```
