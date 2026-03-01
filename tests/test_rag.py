"""Tests for aegis.rag — ingestion and retrieval pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from qdrant_client import QdrantClient

from aegis.rag.ingest import (
    EMBEDDING_DIM,
    chunk_documents,
    chunk_text,
    ensure_collection,
    get_qdrant_client,
    load_all_documents,
    load_document,
    load_text_file,
    store_chunks,
)
from aegis.rag.retriever import format_context, retrieve
from aegis.rag.sparse import BM25Vectorizer, tokenize

GUIDELINES_DIR = Path("data/guidelines")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_embedding(text: str) -> list[float]:
    """Deterministic fake embedding based on text hash (768-dim)."""
    import hashlib

    h = hashlib.sha256(text.encode()).hexdigest()
    # Convert hex to floats in [-1, 1] range
    values = []
    for i in range(0, min(len(h) * 2, EMBEDDING_DIM), 1):
        byte_val = int(h[i % len(h)], 16) / 15.0  # 0..1
        values.append(byte_val * 2 - 1)  # -1..1
    # Pad to EMBEDDING_DIM
    while len(values) < EMBEDDING_DIM:
        values.append(0.0)
    return values[:EMBEDDING_DIM]


def _fake_sparse(text: str) -> tuple[list[int], list[float]]:
    """Deterministic fake sparse vector from text hash."""
    import hashlib

    h = hashlib.sha256(text.encode()).hexdigest()
    indices = [int(h[i : i + 4], 16) % 30000 for i in range(0, 20, 4)]
    values = [int(h[i], 16) / 15.0 + 0.1 for i in range(5)]
    # Ensure indices are unique and sorted
    unique: dict[int, float] = {}
    for idx, val in zip(indices, values):
        unique[idx] = val
    sorted_items = sorted(unique.items())
    return [i for i, _ in sorted_items], [v for _, v in sorted_items]


@pytest.fixture
def qdrant_memory() -> QdrantClient:
    """An in-memory Qdrant client."""
    return QdrantClient(":memory:")


@pytest.fixture
def sample_chunks() -> list[dict]:
    """Pre-chunked sample data with fake embeddings and sparse vectors."""
    texts = [
        "Hipertensão arterial sistêmica é uma condição multifatorial.",
        "O tratamento de primeira linha inclui IECA, BRA, BCC e diuréticos.",
        "Metformina é o medicamento de primeira linha para diabetes tipo 2.",
    ]
    chunks = []
    for i, t in enumerate(texts):
        s_indices, s_values = _fake_sparse(t)
        chunks.append({
            "text": t,
            "source": "test.txt",
            "chunk_index": i,
            "embedding": _fake_embedding(t),
            "sparse_indices": s_indices,
            "sparse_values": s_values,
        })
    return chunks


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------


class TestLoadDocument:
    """Verify document loading functions."""

    def test_load_text_file(self):
        text = load_text_file(GUIDELINES_DIR / "hipertensao_arterial.txt")
        assert "hipertensão" in text.lower()
        assert len(text) > 100

    def test_load_all_documents(self):
        docs = load_all_documents(GUIDELINES_DIR)
        assert len(docs) == 6
        sources = {d["source"] for d in docs}
        assert "hipertensao_arterial.txt" in sources
        assert "diabetes_tipo2.txt" in sources
        assert "insuficiencia_cardiaca.txt" in sources
        assert "asma.txt" in sources
        assert "dpoc.txt" in sources
        assert "avc.txt" in sources

    def test_each_document_has_text(self):
        docs = load_all_documents(GUIDELINES_DIR)
        for doc in docs:
            assert len(doc["text"]) > 100

    def test_unsupported_format_raises(self, tmp_path: Path):
        bad = tmp_path / "data.csv"
        bad.write_text("a,b,c")
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_document(bad)

    def test_empty_directory(self, tmp_path: Path):
        docs = load_all_documents(tmp_path)
        assert docs == []


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


class TestChunking:
    """Verify text chunking."""

    def test_chunks_created(self):
        text = "A " * 1000  # Long text
        chunks = chunk_text(text, source="test.txt")
        assert len(chunks) > 1

    def test_chunk_metadata(self):
        text = "A " * 1000
        chunks = chunk_text(text, source="myfile.txt")
        for i, chunk in enumerate(chunks):
            assert chunk["source"] == "myfile.txt"
            assert chunk["chunk_index"] == i
            assert "text" in chunk

    def test_chunk_size_respected(self):
        text = "A " * 1000
        chunks = chunk_text(text, source="test.txt", chunk_size=200)
        for chunk in chunks:
            assert len(chunk["text"]) <= 200 + 50  # some tolerance

    def test_short_text_single_chunk(self):
        chunks = chunk_text("Hello world.", source="test.txt")
        assert len(chunks) == 1
        assert chunks[0]["text"] == "Hello world."

    def test_chunk_documents_multiple_sources(self):
        docs = [
            {"source": "a.txt", "text": "A " * 500},
            {"source": "b.txt", "text": "B " * 500},
        ]
        chunks = chunk_documents(docs)
        sources = {c["source"] for c in chunks}
        assert "a.txt" in sources
        assert "b.txt" in sources

    def test_real_guidelines_chunk_count(self):
        docs = load_all_documents(GUIDELINES_DIR)
        chunks = chunk_documents(docs)
        # 6 documents → expect ~30-60 chunks
        assert len(chunks) >= 20
        assert all(c["text"] for c in chunks)


# ---------------------------------------------------------------------------
# Qdrant storage
# ---------------------------------------------------------------------------


class TestQdrantStorage:
    """Verify Qdrant collection and storage operations."""

    def test_ensure_collection_creates(self, qdrant_memory: QdrantClient):
        ensure_collection(qdrant_memory, "test_collection")
        assert qdrant_memory.collection_exists("test_collection")

    def test_ensure_collection_idempotent(self, qdrant_memory: QdrantClient):
        ensure_collection(qdrant_memory, "test_collection")
        ensure_collection(qdrant_memory, "test_collection")  # no error
        assert qdrant_memory.collection_exists("test_collection")

    def test_store_chunks(self, qdrant_memory: QdrantClient, sample_chunks: list):
        count = store_chunks(qdrant_memory, sample_chunks, collection="test_coll")
        assert count == 3

    def test_stored_data_retrievable(self, qdrant_memory: QdrantClient, sample_chunks: list):
        store_chunks(qdrant_memory, sample_chunks, collection="test_coll")
        info = qdrant_memory.get_collection("test_coll")
        assert info.points_count == 3

    def test_get_qdrant_client_memory(self, monkeypatch):
        monkeypatch.setenv("QDRANT_URL", ":memory:")
        from aegis.config import Settings

        with patch("aegis.rag.ingest.settings", Settings()):
            client = get_qdrant_client()
        assert client is not None


# ---------------------------------------------------------------------------
# Retrieval (with mocked embeddings) — dense mode
# ---------------------------------------------------------------------------


class TestRetrieval:
    """Verify retrieval with in-memory Qdrant and fake embeddings."""

    @pytest.fixture
    def populated_qdrant(self, qdrant_memory: QdrantClient, sample_chunks: list) -> QdrantClient:
        store_chunks(qdrant_memory, sample_chunks, collection="test_coll")
        return qdrant_memory

    def test_retrieve_returns_results(self, populated_qdrant: QdrantClient):
        query_vec = _fake_embedding("hipertensão tratamento")
        with patch("aegis.rag.retriever.embed_text", return_value=query_vec):
            results = retrieve(
                "hipertensão tratamento",
                client=populated_qdrant,
                collection="test_coll",
                score_threshold=0.0,
                mode="dense",
            )
        assert len(results) > 0

    def test_result_has_expected_fields(self, populated_qdrant: QdrantClient):
        query_vec = _fake_embedding("diabetes metformina")
        with patch("aegis.rag.retriever.embed_text", return_value=query_vec):
            results = retrieve(
                "diabetes metformina",
                client=populated_qdrant,
                collection="test_coll",
                score_threshold=0.0,
                mode="dense",
            )
        for r in results:
            assert "text" in r
            assert "source" in r
            assert "score" in r
            assert isinstance(r["score"], float)

    def test_top_k_limits_results(self, populated_qdrant: QdrantClient):
        query_vec = _fake_embedding("tratamento")
        with patch("aegis.rag.retriever.embed_text", return_value=query_vec):
            results = retrieve(
                "tratamento",
                client=populated_qdrant,
                collection="test_coll",
                top_k=2,
                score_threshold=0.0,
                mode="dense",
            )
        assert len(results) <= 2


# ---------------------------------------------------------------------------
# Retrieval — hybrid mode
# ---------------------------------------------------------------------------


class TestHybridRetrieval:
    """Verify hybrid retrieval with RRF fusion."""

    @pytest.fixture
    def populated_qdrant(self, qdrant_memory: QdrantClient, sample_chunks: list) -> QdrantClient:
        store_chunks(qdrant_memory, sample_chunks, collection="test_coll")
        return qdrant_memory

    @pytest.fixture
    def bm25_stats(self, tmp_path: Path, sample_chunks: list) -> Path:
        """Fit BM25 on sample chunks and save stats."""
        texts = [c["text"] for c in sample_chunks]
        bm25 = BM25Vectorizer().fit(texts)
        path = tmp_path / "bm25_stats.json"
        bm25.save(path)
        return path

    def test_hybrid_returns_results(
        self, populated_qdrant: QdrantClient, bm25_stats: Path
    ):
        query_vec = _fake_embedding("hipertensão tratamento")
        with (
            patch("aegis.rag.retriever.embed_text", return_value=query_vec),
            patch("aegis.rag.retriever.settings") as mock_settings,
        ):
            mock_settings.qdrant_collection = "test_coll"
            mock_settings.bm25_stats_path = bm25_stats
            results = retrieve(
                "hipertensão tratamento",
                client=populated_qdrant,
                collection="test_coll",
                score_threshold=0.0,
                mode="hybrid",
            )
        assert len(results) > 0

    def test_hybrid_result_has_fields(
        self, populated_qdrant: QdrantClient, bm25_stats: Path
    ):
        query_vec = _fake_embedding("diabetes")
        with (
            patch("aegis.rag.retriever.embed_text", return_value=query_vec),
            patch("aegis.rag.retriever.settings") as mock_settings,
        ):
            mock_settings.qdrant_collection = "test_coll"
            mock_settings.bm25_stats_path = bm25_stats
            results = retrieve(
                "diabetes",
                client=populated_qdrant,
                collection="test_coll",
                score_threshold=0.0,
                mode="hybrid",
            )
        for r in results:
            assert "text" in r
            assert "source" in r
            assert "score" in r

    def test_hybrid_top_k_limits(
        self, populated_qdrant: QdrantClient, bm25_stats: Path
    ):
        query_vec = _fake_embedding("tratamento")
        with (
            patch("aegis.rag.retriever.embed_text", return_value=query_vec),
            patch("aegis.rag.retriever.settings") as mock_settings,
        ):
            mock_settings.qdrant_collection = "test_coll"
            mock_settings.bm25_stats_path = bm25_stats
            results = retrieve(
                "tratamento",
                client=populated_qdrant,
                collection="test_coll",
                top_k=1,
                score_threshold=0.0,
                mode="hybrid",
            )
        assert len(results) <= 1


# ---------------------------------------------------------------------------
# Format context
# ---------------------------------------------------------------------------


class TestFormatContext:
    """Verify format_context output."""

    def test_empty_results(self):
        result = format_context([])
        assert "Nenhuma diretriz" in result

    def test_formats_single_result(self):
        results = [
            {"text": "Texto da diretriz.", "source": "has.txt", "chunk_index": 0, "score": 0.85}
        ]
        output = format_context(results)
        assert "has.txt" in output
        assert "0.85" in output
        assert "Texto da diretriz." in output

    def test_formats_multiple_results(self):
        results = [
            {"text": "Chunk 1", "source": "a.txt", "chunk_index": 0, "score": 0.9},
            {"text": "Chunk 2", "source": "b.txt", "chunk_index": 1, "score": 0.7},
        ]
        output = format_context(results)
        assert "Chunk 1" in output
        assert "Chunk 2" in output
        assert "---" in output  # separator


# ---------------------------------------------------------------------------
# BM25 sparse vectorizer
# ---------------------------------------------------------------------------


class TestTokenize:
    """Verify Portuguese-aware tokenizer."""

    def test_lowercases(self):
        tokens = tokenize("Hipertensão Arterial")
        assert "hipertensão" in tokens
        assert "arterial" in tokens

    def test_removes_stop_words(self):
        tokens = tokenize("O tratamento de primeira linha para hipertensão")
        assert "o" not in tokens
        assert "de" not in tokens
        assert "para" not in tokens
        assert "tratamento" in tokens

    def test_removes_single_chars(self):
        tokens = tokenize("A B C hipertensão")
        assert "a" not in tokens
        assert "hipertensão" in tokens

    def test_splits_on_punctuation(self):
        tokens = tokenize("IECA, BRA, BCC — diuréticos.")
        assert "ieca" in tokens
        assert "bra" in tokens
        assert "diuréticos" in tokens


class TestBM25Vectorizer:
    """Verify BM25 fitting, encoding, and persistence."""

    @pytest.fixture
    def corpus(self) -> list[str]:
        return [
            "Hipertensão arterial sistêmica é uma condição multifatorial.",
            "O tratamento de primeira linha inclui IECA, BRA, BCC e diuréticos.",
            "Metformina é o medicamento de primeira linha para diabetes tipo 2.",
            "A asma é uma doença inflamatória crônica das vias aéreas.",
            "DPOC é causada principalmente pelo tabagismo crônico.",
        ]

    @pytest.fixture
    def fitted_bm25(self, corpus: list[str]) -> BM25Vectorizer:
        return BM25Vectorizer().fit(corpus)

    def test_fit_sets_doc_count(self, fitted_bm25: BM25Vectorizer):
        assert fitted_bm25.doc_count == 5

    def test_fit_sets_avg_doc_len(self, fitted_bm25: BM25Vectorizer):
        assert fitted_bm25.avg_doc_len > 0

    def test_fit_populates_doc_freq(self, fitted_bm25: BM25Vectorizer):
        assert len(fitted_bm25.doc_freq) > 0

    def test_encode_document_returns_sparse(self, fitted_bm25: BM25Vectorizer):
        indices, values = fitted_bm25.encode_document("hipertensão arterial tratamento")
        assert len(indices) > 0
        assert len(indices) == len(values)
        assert all(isinstance(i, int) for i in indices)
        assert all(isinstance(v, float) for v in values)
        assert all(v > 0 for v in values)

    def test_encode_document_indices_sorted(self, fitted_bm25: BM25Vectorizer):
        indices, _ = fitted_bm25.encode_document("hipertensão tratamento IECA")
        assert indices == sorted(indices)

    def test_encode_query_returns_sparse(self, fitted_bm25: BM25Vectorizer):
        indices, values = fitted_bm25.encode_query("hipertensão tratamento")
        assert len(indices) > 0
        assert len(indices) == len(values)
        assert all(v > 0 for v in values)

    def test_encode_query_indices_sorted(self, fitted_bm25: BM25Vectorizer):
        indices, _ = fitted_bm25.encode_query("diabetes metformina primeira linha")
        assert indices == sorted(indices)

    def test_save_and_load(self, fitted_bm25: BM25Vectorizer, tmp_path: Path):
        path = tmp_path / "bm25.json"
        fitted_bm25.save(path)

        loaded = BM25Vectorizer.load(path)
        assert loaded.doc_count == fitted_bm25.doc_count
        assert loaded.avg_doc_len == fitted_bm25.avg_doc_len
        assert loaded.doc_freq == fitted_bm25.doc_freq
        assert loaded.k1 == fitted_bm25.k1
        assert loaded.b == fitted_bm25.b

    def test_loaded_produces_same_vectors(self, fitted_bm25: BM25Vectorizer, tmp_path: Path):
        path = tmp_path / "bm25.json"
        fitted_bm25.save(path)
        loaded = BM25Vectorizer.load(path)

        query = "tratamento diabetes"
        i1, v1 = fitted_bm25.encode_query(query)
        i2, v2 = loaded.encode_query(query)
        assert i1 == i2
        assert v1 == v2

    def test_different_texts_produce_different_vectors(self, fitted_bm25: BM25Vectorizer):
        i1, v1 = fitted_bm25.encode_document("hipertensão arterial")
        i2, v2 = fitted_bm25.encode_document("diabetes metformina")
        assert (i1, v1) != (i2, v2)

    def test_empty_text_returns_empty(self, fitted_bm25: BM25Vectorizer):
        indices, values = fitted_bm25.encode_document("")
        assert indices == []
        assert values == []


# ---------------------------------------------------------------------------
# Integration tests (need Ollama + nomic-embed-text)
# ---------------------------------------------------------------------------


def _is_embed_model_available() -> bool:
    try:
        import ollama as _ollama

        models = _ollama.list().get("models", [])
        return any("nomic-embed-text" in (m.get("name", "") or m.get("model", "")) for m in models)
    except Exception:
        return False


@pytest.mark.integration
class TestRAGIntegration:
    """End-to-end RAG tests with real Ollama embeddings and in-memory Qdrant."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_embed_model(self):
        if not _is_embed_model_available():
            pytest.skip("nomic-embed-text model not available in Ollama")

    def test_embed_text_returns_vector(self):
        from aegis.rag.ingest import embed_text

        vec = embed_text("hipertensão arterial")
        assert isinstance(vec, list)
        assert len(vec) == EMBEDDING_DIM
        assert all(isinstance(v, float) for v in vec)

    def test_full_ingest_and_retrieve(self, tmp_path: Path):
        from aegis.rag.ingest import ingest_guidelines
        from aegis.rag.retriever import retrieve

        client = QdrantClient(":memory:")
        bm25_path = tmp_path / "bm25_stats.json"
        with patch("aegis.rag.ingest.settings.qdrant_collection", "test_integration"):
            count = ingest_guidelines(GUIDELINES_DIR, client=client, bm25_path=bm25_path)
        assert count > 0
        assert bm25_path.exists()

        # Test dense retrieval
        results = retrieve(
            "tratamento hipertensão estágio 2",
            client=client,
            collection="test_integration",
            mode="dense",
        )
        assert len(results) > 0
        combined = " ".join(r["text"].lower() for r in results)
        assert "hipertensão" in combined or "pressão" in combined

        # Test hybrid retrieval
        with patch("aegis.rag.retriever.settings") as mock_settings:
            mock_settings.bm25_stats_path = bm25_path
            mock_settings.qdrant_collection = "test_integration"
            results_hybrid = retrieve(
                "tratamento hipertensão estágio 2",
                client=client,
                collection="test_integration",
                mode="hybrid",
            )
        assert len(results_hybrid) > 0
