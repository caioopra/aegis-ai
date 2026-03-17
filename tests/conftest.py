"""Shared fixtures and markers for the test suite."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from aegis.fhir import FHIRStore

logger = logging.getLogger("aegis.integration")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests that require live services (Ollama, Qdrant, etc.)",
    )


def is_ollama_available() -> bool:
    """Check if the Ollama server is reachable and has a model."""
    try:
        import ollama

        models = ollama.list().get("models", [])
        return len(models) > 0
    except Exception:
        return False


def is_embed_model_available() -> bool:
    """Check if the embedding model is available in Ollama."""
    try:
        import ollama

        models = ollama.list().get("models", [])
        return any("nomic-embed-text" in m.get("model", "") for m in models)
    except Exception:
        return False


@contextmanager
def timed(label: str):
    """Context manager for timing test sections and logging results."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logger.info("[TIMING] %s: %.2fs", label, elapsed)


# ---------------------------------------------------------------------------
# Shared data fixtures
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATIENT_PATH = PROJECT_ROOT / "data" / "synthea" / "sample_patient_joao.json"
GUIDELINES_DIR = PROJECT_ROOT / "data" / "guidelines"


@pytest.fixture
def sample_note() -> str:
    return "Paciente João, 65a, c/ dispneia aos esforços, DPN, edema MMII. PA 150x95. Uso de losartana 50mg e HCTZ 25mg."


@pytest.fixture
def sample_note_en() -> str:
    return (
        "Pt 72M, SOB on exertion, PND, bilateral LE edema. BP 180/110. On losartan 50mg, HCTZ 25mg."
    )


@pytest.fixture(scope="session")
def fhir_store() -> FHIRStore:
    """Load the sample patient into a FHIRStore (session-scoped, no LLM needed)."""
    store = FHIRStore()
    if SAMPLE_PATIENT_PATH.exists():
        store.load_bundle(SAMPLE_PATIENT_PATH)
    return store


@pytest.fixture(scope="session")
def ingested_rag_env(tmp_path_factory):
    """Session-scoped fixture: ingest all guidelines with real embeddings.

    Returns ``(client, collection_name, bm25_path)``.
    Skips if ``nomic-embed-text`` is not available.
    """
    if not is_embed_model_available():
        pytest.skip("nomic-embed-text model not available in Ollama")

    from qdrant_client import QdrantClient

    from aegis.rag.ingest import ingest_guidelines

    client = QdrantClient(":memory:")
    collection = "test_integration"
    bm25_dir = tmp_path_factory.mktemp("bm25")
    bm25_path = bm25_dir / "bm25_stats.json"

    ingest_guidelines(
        directory=GUIDELINES_DIR,
        client=client,
        bm25_path=bm25_path,
    )

    return client, collection, bm25_path
