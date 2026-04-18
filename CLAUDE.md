# AegisNode — Project Conventions

## Overview
AegisNode is a learning-focused agentic clinical copilot. It takes simulated doctor's
voice notes, retrieves clinical guidelines via RAG, interacts with patient records
through MCP, and generates structured medical reports.

## Tech Stack
- **Python 3.10+** with `uv` for package management
- **Ollama** for local LLM inference (Mistral / Llama 3.2)
- **Qdrant** for vector storage (RAG)
- **FastMCP** for the Model Context Protocol server
- **LangGraph** for agent orchestration

## Project Layout
- `src/aegis/` — main package
- `scripts/` — CLI entry points (ingest, run agent, generate data)
- `tests/` — pytest test suite
- `data/guidelines/` — clinical guideline documents (checked in)
- `data/synthea/` — generated FHIR patient data (gitignored)

## Commands
- Install deps: `uv sync`
- Run tests: `uv run pytest`
- Run single test: `uv run pytest tests/test_mcp_server.py -v`
- Lint: `uv run ruff check src/ tests/`
- Format: `uv run ruff format src/ tests/ scripts/`
- Run MCP server: `uv run python -m aegis.mcp_server`
- Run agent: `uv run python scripts/run_agent.py --note "..."`
- Ingest guidelines: `uv run python scripts/ingest_guidelines.py`

## Code Style
- Use type hints everywhere
- Pydantic models for data structures and settings
- Async functions for I/O-bound operations (MCP, Ollama calls)
- Keep functions small and focused
- Docstrings on public functions only
- **Before finishing a work session**, always run `uv run ruff format src/ tests/ scripts/` to format all code. CI enforces formatting and will fail otherwise.

## PII & LGPD Handling
Clinical notes may contain Brazilian PII — notably **CPF** (Cadastro de Pessoas Físicas) and patient names. CPF is classified as sensitive personal data under LGPD (Lei Geral de Proteção de Dados, Lei 13.709/2018), and health data is additionally classified as *dado pessoal sensível* (Art. 5 II).

- **All patient data in this repository is synthetic.** `data/synthea/sample_*.json` files contain fictitious patients only. Sample CPFs use structurally-invalid sentinel values (e.g., `111.111.111-11`) that are rejected by the Receita Federal validator.
- **Local mode (Ollama)**: all inference runs on-device; notes never leave the machine.
- **Cloud mode (Gemini — Phase 16+)**: the clinical note (including any CPF it contains) is transmitted **verbatim** to the remote LLM provider via `REPORT_PROMPT` and `EVALUATE_REPORT_PROMPT`. Under LGPD Art. 33 (international transfer of sensitive data), this requires an explicit legal basis — typically a Data Processing Agreement with the provider or explicit patient consent. **Cloud mode must only be used with synthetic data until a DPA or redaction pass is in place.**
- **CPF logging**: CPF values are never logged in plaintext. The CPF-based patient matcher (`_match_patient_id` in `src/aegis/agent/nodes.py`) logs only `"CPF extraído da nota, tentando lookup"` at DEBUG level, with no value interpolation.
- **CPF in state**: the CPF itself is never stored in `AgentState`. Only `patient_id_match_type == "cpf"` signals that CPF-based matching succeeded.
