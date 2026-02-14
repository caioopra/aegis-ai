"""Retriever: query Qdrant for relevant clinical guideline chunks."""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient

from aegis.config import settings
from aegis.rag.ingest import embed_text, get_qdrant_client

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

DEFAULT_TOP_K = 5
DEFAULT_SCORE_THRESHOLD = 0.3


def retrieve(
    query: str,
    client: QdrantClient | None = None,
    collection: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> list[dict[str, Any]]:
    """Search for chunks most relevant to *query*.

    Returns a list of dicts with ``text``, ``source``, ``chunk_index``,
    and ``score``, sorted by descending relevance.
    """
    client = client or get_qdrant_client()
    collection = collection or settings.qdrant_collection

    query_vector = embed_text(query)

    results = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=top_k,
        score_threshold=score_threshold,
    )

    return [
        {
            "text": point.payload["text"],
            "source": point.payload["source"],
            "chunk_index": point.payload.get("chunk_index", 0),
            "score": point.score,
        }
        for point in results.points
    ]


def format_context(results: list[dict[str, Any]]) -> str:
    """Format retrieved chunks as a single text block for the LLM prompt."""
    if not results:
        return "Nenhuma diretriz relevante encontrada."

    sections = []
    for i, r in enumerate(results, 1):
        sections.append(
            f"[Fonte: {r['source']} | Relevância: {r['score']:.2f}]\n{r['text']}"
        )
    return "\n\n---\n\n".join(sections)
