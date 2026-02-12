"""Shared fixtures and markers for the test suite."""

import pytest


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


@pytest.fixture
def sample_note() -> str:
    return "Paciente João, 65a, c/ dispneia aos esforços, DPN, edema MMII. PA 150x95. Uso de losartana 50mg e HCTZ 25mg."


@pytest.fixture
def sample_note_en() -> str:
    return "Pt 72M, SOB on exertion, PND, bilateral LE edema. BP 180/110. On losartan 50mg, HCTZ 25mg."
