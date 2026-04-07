---
name: ingest-guidelines
description: Re-ingest all Brazilian clinical guidelines from data/guidelines/ into Qdrant (dense + BM25 sparse vectors). Use after adding new guidelines, changing chunking parameters, or fixing the BM25 hasher.
allowed-tools: Bash, Read
---

# Ingest Guidelines — Re-build the Hybrid Index

Re-ingest all clinical guideline files into Qdrant with both dense
(nomic-embed-text) and sparse (BM25) vectors.

## Prerequisites

- Ollama is running locally with the `nomic-embed-text` model pulled.
  Verify with:
  ```bash
  curl -s http://localhost:11434/api/tags | grep nomic-embed-text
  ```
- Qdrant is reachable at `http://localhost:6333` (or whatever
  `settings.qdrant_url` is set to).

If either is missing, STOP and instruct the user to start them
(delegate to **infra-deployment-engineer** if config is unclear).

## Steps

1. List the guideline files that will be ingested:

   ```bash
   ls data/guidelines/
   ```

2. Run the ingest script:

   ```bash
   uv run python scripts/ingest_guidelines.py
   ```

3. Report:
   - Number of files processed.
   - Number of chunks produced.
   - Path to the BM25 stats file written (`data/bm25_stats.json`).
   - Any warnings or errors from the script.

4. Smoke test the retriever with a known query:

   ```bash
   uv run python -c "from aegis.rag.retriever import retrieve; \
     import json; print(json.dumps(retrieve('tratamento hipertensão', top_k=3), indent=2, ensure_ascii=False))"
   ```

## When to re-ingest

- After adding or editing files in `data/guidelines/`.
- After changing `chunk_size`, `chunk_overlap`, or tokenization in
  `src/aegis/rag/ingest.py` or `src/aegis/rag/sparse.py`.
- After changing the embedding model or dimension.
- After upgrading `BM25Vectorizer` (the `_term_hash` must stay
  deterministic — MD5, not Python's `hash()`).
- After a fresh clone where `data/bm25_stats.json` doesn't exist.

## What NOT to do

- Do not edit `data/bm25_stats.json` by hand — it's generated.
- Do not commit `data/bm25_stats.json` — it is gitignored.
- Do not run this skill during a test run; it mutates the Qdrant
  collection and will race with test fixtures.
