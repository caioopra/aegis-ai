"""Integration tests for the RAG layer (requires Ollama with nomic-embed-text)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aegis.rag.retriever import format_context, retrieve
from tests.conftest import timed


@pytest.mark.integration
class TestRAGIntegration:
    """End-to-end tests for ingestion + retrieval with real embeddings."""

    @pytest.fixture(autouse=True)
    def _patch_settings(self, ingested_rag_env):
        client, _collection, bm25_path = ingested_rag_env
        self.client = client
        # ingest_guidelines uses settings.qdrant_collection internally
        self.collection = "clinical_guidelines"
        with (
            patch("aegis.rag.retriever.settings") as mock_settings,
            patch("aegis.rag.retriever._bm25", None),
        ):
            mock_settings.bm25_stats_path = bm25_path
            mock_settings.qdrant_collection = self.collection
            yield

    def test_ingest_chunk_count(self) -> None:
        """Ingestion should produce at least 20 chunks across 6 guidelines."""
        info = self.client.get_collection(self.collection)
        assert info.points_count >= 20

    def test_dense_retrieval_hipertensao(self) -> None:
        """Dense search for hypertension should return chunks from the HAS guideline."""
        results = retrieve(
            "tratamento hipertensão estágio 2",
            client=self.client,
            collection=self.collection,
            mode="dense",
        )
        assert len(results) > 0
        sources = [r["source"] for r in results]
        assert any("hipertensao" in s for s in sources)

    def test_hybrid_retrieval_diabetes(self) -> None:
        """Hybrid search for diabetes should return chunks from diabetes_tipo2.txt."""
        results = retrieve(
            "metas glicêmicas diabetes tipo 2",
            client=self.client,
            collection=self.collection,
            mode="hybrid",
        )
        assert any(r["source"] == "diabetes_tipo2.txt" for r in results)

    def test_hybrid_retrieval_insuficiencia_cardiaca(self) -> None:
        """Hybrid search for heart failure should hit insuficiencia_cardiaca.txt."""
        results = retrieve(
            "tratamento insuficiência cardíaca fração ejeção reduzida",
            client=self.client,
            collection=self.collection,
            mode="hybrid",
        )
        assert any(r["source"] == "insuficiencia_cardiaca.txt" for r in results)

    def test_hybrid_retrieval_asma(self) -> None:
        """Hybrid search for asthma crisis should hit asma.txt."""
        results = retrieve(
            "broncodilatador asma crise",
            client=self.client,
            collection=self.collection,
            mode="hybrid",
        )
        assert any(r["source"] == "asma.txt" for r in results)

    def test_hybrid_retrieval_dpoc(self) -> None:
        """Hybrid search for COPD exacerbation should hit dpoc.txt."""
        results = retrieve(
            "exacerbação DPOC GOLD",
            client=self.client,
            collection=self.collection,
            mode="hybrid",
        )
        assert any(r["source"] == "dpoc.txt" for r in results)

    def test_hybrid_retrieval_avc(self) -> None:
        """Hybrid search for stroke thrombolysis should hit avc.txt."""
        results = retrieve(
            "trombólise AVC isquêmico",
            client=self.client,
            collection=self.collection,
            mode="hybrid",
        )
        assert any(r["source"] == "avc.txt" for r in results)

    def test_cross_topic_query(self) -> None:
        """A cross-topic query should return results from at least 2 sources."""
        results = retrieve(
            "comorbidades hipertensão diabetes",
            client=self.client,
            collection=self.collection,
            mode="hybrid",
        )
        unique_sources = {r["source"] for r in results}
        assert len(unique_sources) >= 2

    def test_format_context_integration(self) -> None:
        """format_context should produce a non-empty string with source markers."""
        results = retrieve(
            "tratamento hipertensão",
            client=self.client,
            collection=self.collection,
            mode="hybrid",
        )
        context = format_context(results)
        assert isinstance(context, str)
        assert len(context) > 0
        assert "[Fonte:" in context

    def test_retrieval_timing(self) -> None:
        """A single retrieve call should complete in under 5 seconds."""
        with timed("hybrid_retrieve"):
            results = retrieve(
                "manejo insuficiência cardíaca",
                client=self.client,
                collection=self.collection,
                mode="hybrid",
            )
        assert results is not None  # sanity — timing is the real check

    def test_garbage_query_no_crash(self) -> None:
        """A nonsense query should return empty or very-low-score results, not crash."""
        results = retrieve(
            "xyzabc123 nonsense",
            client=self.client,
            collection=self.collection,
            mode="hybrid",
            score_threshold=0.5,
        )
        # Either empty or all scores below threshold
        assert isinstance(results, list)
        for r in results:
            assert r["score"] <= 0.5
