# AegisNode: Agentic Clinical Copilot

An AI agent that processes simulated doctor's voice notes, retrieves clinical guidelines via RAG, accesses patient records through MCP, and generates structured medical reports.

Built for learning key LLM engineering concepts: **prompt engineering, RAG, MCP, agent orchestration, and fine-tuning**.

## Architecture

```
Doctor's Note ──▶ Parse Entities ──▶ Retrieve Guidelines (RAG)
                                          │
                    Fetch Patient Data (MCP) ◀──┘
                                          │
                    Generate Report ◀─────┘
                          │
                    Evaluate Report ──▶ Structured Output
```

## Quick Start

```bash
# Install dependencies
uv sync

# Pull a model with Ollama
ollama pull mistral

# Start Qdrant (for RAG)
docker run -p 6333:6333 qdrant/qdrant

# Ingest clinical guidelines
uv run python scripts/ingest_guidelines.py

# Run the agent
uv run python scripts/run_agent.py --note "Paciente João, 65a, dispneia aos esforços, PA 150x95"
```

## Phases

1. **Foundation + LLM** — Ollama integration, prompt engineering
2. **MCP Server** — FHIR patient data access via Model Context Protocol
3. **RAG System** — Clinical guideline retrieval with Qdrant
4. **Agent Orchestration** — LangGraph pipeline combining all components
5. **Fine-tuning** — QLoRA training pipeline (cloud-ready)
