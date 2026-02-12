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
- Run MCP server: `uv run python -m aegis.mcp_server`
- Run agent: `uv run python scripts/run_agent.py --note "..."`
- Ingest guidelines: `uv run python scripts/ingest_guidelines.py`

## Code Style
- Use type hints everywhere
- Pydantic models for data structures and settings
- Async functions for I/O-bound operations (MCP, Ollama calls)
- Keep functions small and focused
- Docstrings on public functions only
