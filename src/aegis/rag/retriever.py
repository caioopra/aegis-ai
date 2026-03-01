"""Retriever: query Qdrant for relevant clinical guideline chunks."""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Fusion,
    FusionQuery,
    Prefetch,
    SparseVector,
)

from aegis.config import settings
from aegis.rag.ingest import embed_text, get_qdrant_client
from aegis.rag.sparse import BM25Vectorizer

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

DEFAULT_TOP_K = 5
DEFAULT_SCORE_THRESHOLD = 0.3
_PREFETCH_LIMIT = 20  # candidates per branch before fusion


def retrieve(
    query: str,
    client: QdrantClient | None = None,
    collection: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    mode: str = "hybrid",
) -> list[dict[str, Any]]:
    """Search for chunks most relevant to *query*.

    ``mode`` controls the search strategy:
    - ``"dense"``: cosine similarity on embeddings only (original behaviour)
    - ``"hybrid"``: dense + BM25 sparse search fused with RRF (default)

    Returns a list of dicts with ``text``, ``source``, ``chunk_index``,
    and ``score``, sorted by descending relevance.
    """
    client = client or get_qdrant_client()
    collection = collection or settings.qdrant_collection

    if mode == "hybrid":
        return _retrieve_hybrid(query, client, collection, top_k, score_threshold)
    return _retrieve_dense(query, client, collection, top_k, score_threshold)


def _retrieve_dense(
    query: str,
    client: QdrantClient,
    collection: str,
    top_k: int,
    score_threshold: float,
) -> list[dict[str, Any]]:
    """Dense-only retrieval using cosine similarity."""
    query_vector = embed_text(query)

    results = client.query_points(
        collection_name=collection,
        query=query_vector,
        using="dense",
        limit=top_k,
        score_threshold=score_threshold,
    )

    return _format_results(results.points)


def _retrieve_hybrid(
    query: str,
    client: QdrantClient,
    collection: str,
    top_k: int,
    score_threshold: float,
) -> list[dict[str, Any]]:
    """Hybrid retrieval: dense + sparse fused with Reciprocal Rank Fusion."""
    query_vector = embed_text(query)

    # Load BM25 stats and compute sparse query vector
    bm25 = BM25Vectorizer.load(settings.bm25_stats_path)
    indices, values = bm25.encode_query(query)

    results = client.query_points(
        collection_name=collection,
        prefetch=[
            Prefetch(
                query=query_vector,
                using="dense",
                limit=_PREFETCH_LIMIT,
            ),
            Prefetch(
                query=SparseVector(indices=indices, values=values),
                using="sparse",
                limit=_PREFETCH_LIMIT,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        score_threshold=score_threshold,
    )

    return _format_results(results.points)


def _format_results(points: list) -> list[dict[str, Any]]:
    """Convert Qdrant ScoredPoints to result dicts."""
    return [
        {
            "text": point.payload["text"],
            "source": point.payload["source"],
            "chunk_index": point.payload.get("chunk_index", 0),
            "score": point.score,
        }
        for point in points
    ]


def format_context(results: list[dict[str, Any]]) -> str:
    """Format retrieved chunks as a single text block for the LLM prompt."""
    if not results:
        return "Nenhuma diretriz relevante encontrada."

    sections = []
    for r in results:
        sections.append(f"[Fonte: {r['source']} | Relevância: {r['score']:.2f}]\n{r['text']}")
    return "\n\n---\n\n".join(sections)
