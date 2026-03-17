"""Integration tests for the full AegisNode clinical agent pipeline and edge cases.

These tests exercise the entire LangGraph flow end-to-end with real LLM calls
(Ollama/Mistral), real embeddings, and real FHIR data.  They require live
Ollama with ``mistral`` and ``nomic-embed-text`` models, so every test is
marked ``@pytest.mark.integration``.
"""

from __future__ import annotations

import time
import warnings
from unittest.mock import patch

import pytest

from aegis.agent.graph import build_graph
from tests.conftest import is_ollama_available

# ---------------------------------------------------------------------------
# Clinical notes
# ---------------------------------------------------------------------------

NOTE_HAS = (
    "Paciente João, 65 anos, PA 170x100 mmHg, cefaleia occipital. "
    "Uso de losartana 50mg e HCTZ 25mg."
)
NOTE_ICC = (
    "Paciente João, 65a, dispneia aos esforços progressiva, DPN, edema MMII bilateral. "
    "PA 150x95 mmHg. Uso de losartana 50mg, HCTZ 25mg, metformina 850mg."
)
NOTE_DM2 = "Maria, 58a, retorno DM2, glicemia jejum 180, HbA1c 8.5%. Metformina 850mg 2x/dia."
NOTE_ROTINA = "Retorno de rotina, paciente estável, sem queixas novas. PA 130x80. Manter conduta."

EXPECTED_STATE_KEYS = {
    "patient_note",
    "extracted_entities",
    "patient_id",
    "needs_retrieval",
    "retrieval_queries",
    "report",
    "evaluation",
}

REPORT_EXPECTED_KEYS = {"patient_summary", "findings", "assessment", "plan"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _invoke(note: str) -> dict:
    """Build a fresh graph and invoke it with the given note."""
    graph = build_graph()
    return graph.invoke({"patient_note": note})


# ---------------------------------------------------------------------------
# Full Pipeline Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFullPipelineIntegration:
    """End-to-end tests that run the complete agent graph."""

    @pytest.fixture(autouse=True)
    def _require_services(self) -> None:
        if not is_ollama_available():
            pytest.skip("Ollama not available")

    @pytest.fixture(autouse=True)
    def _patch_all(self, fhir_store, ingested_rag_env):
        client, _collection, bm25_path = ingested_rag_env
        actual_collection = "clinical_guidelines"
        with (
            patch("aegis.agent.nodes._store", fhir_store),
            patch("aegis.agent.nodes._ensure_store", lambda: None),
            patch("aegis.mcp_server._store", fhir_store),
            patch("aegis.mcp_server._load_store", lambda: None),
            patch("aegis.rag.retriever.get_qdrant_client", return_value=client),
            patch("aegis.rag.retriever.settings") as mock_settings,
        ):
            mock_settings.bm25_stats_path = bm25_path
            mock_settings.qdrant_collection = actual_collection
            yield

    # -- individual tests ---------------------------------------------------

    def test_full_pipeline_has_note(self) -> None:
        result = _invoke(NOTE_HAS)

        for key in EXPECTED_STATE_KEYS:
            assert key in result, f"Missing state key: {key}"

        assert result["patient_id"] == "patient-joao-001"
        assert isinstance(result["report"], dict)
        assert isinstance(result["evaluation"], dict)

    def test_full_pipeline_icc_note(self) -> None:
        result = _invoke(NOTE_ICC)

        assert result["patient_id"] == "patient-joao-001"
        assert "report" in result
        assert "evaluation" in result

    def test_full_pipeline_unmatched_patient(self) -> None:
        result = _invoke(NOTE_DM2)

        # Maria is not in the FHIR store; should fall back to first patient
        assert result["patient_id"] != ""
        assert "report" in result
        assert "evaluation" in result

    def test_full_pipeline_routine_note(self) -> None:
        result = _invoke(NOTE_ROTINA)

        assert "report" in result
        assert "evaluation" in result

    def test_pipeline_timing(self) -> None:
        start = time.perf_counter()
        result = _invoke(NOTE_HAS)
        elapsed = time.perf_counter() - start

        assert "report" in result, "Pipeline did not produce a report"
        assert elapsed < 300, f"Pipeline took {elapsed:.1f}s — exceeds 300s limit"

    def test_guidelines_populated_when_retrieved(self) -> None:
        result = _invoke(NOTE_HAS)

        if result.get("needs_retrieval"):
            guidelines = result.get("guidelines", "")
            assert guidelines, "needs_retrieval=True but guidelines is empty"
            assert "[Fonte:" in guidelines, "Guidelines should contain '[Fonte:' source markers"
        else:
            guidelines = result.get("guidelines", "")
            if guidelines:
                warnings.warn(
                    "LLM decided retrieval was not needed for a HAS note — "
                    "guidelines may be a skip message."
                )

    def test_report_has_minimum_structure(self) -> None:
        result = _invoke(NOTE_HAS)
        report = result.get("report", {})
        assert isinstance(report, dict), "report should be a dict"

        matched = REPORT_EXPECTED_KEYS & set(report.keys())
        if not matched:
            warnings.warn(
                f"Report keys {list(report.keys())} don't overlap with "
                f"expected keys {REPORT_EXPECTED_KEYS}. LLM output may vary."
            )

    def test_evaluation_scores_valid(self) -> None:
        result = _invoke(NOTE_HAS)
        evaluation = result.get("evaluation", {})
        assert isinstance(evaluation, dict), "evaluation should be a dict"

        overall = evaluation.get("overall", {})
        if isinstance(overall, dict) and "score" in overall:
            score = overall["score"]
            assert isinstance(score, int), f"Score should be int, got {type(score)}"
            assert 1 <= score <= 5, f"Score {score} is outside 1-5 range"
            if score < 3:
                warnings.warn(
                    f"Report quality score is low ({score}/5). Review LLM prompts or model quality."
                )


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEdgeCasesIntegration:
    """Edge-case tests that stress unusual inputs through the full pipeline."""

    @pytest.fixture(autouse=True)
    def _require_services(self) -> None:
        if not is_ollama_available():
            pytest.skip("Ollama not available")

    @pytest.fixture(autouse=True)
    def _patch_all(self, fhir_store, ingested_rag_env):
        client, _collection, bm25_path = ingested_rag_env
        actual_collection = "clinical_guidelines"
        with (
            patch("aegis.agent.nodes._store", fhir_store),
            patch("aegis.agent.nodes._ensure_store", lambda: None),
            patch("aegis.mcp_server._store", fhir_store),
            patch("aegis.mcp_server._load_store", lambda: None),
            patch("aegis.rag.retriever.get_qdrant_client", return_value=client),
            patch("aegis.rag.retriever.settings") as mock_settings,
        ):
            mock_settings.bm25_stats_path = bm25_path
            mock_settings.qdrant_collection = actual_collection
            yield

    # -- individual tests ---------------------------------------------------

    def test_empty_note(self) -> None:
        result = _invoke("")

        assert "report" in result, "Pipeline should produce a report even for empty input"

    def test_very_long_note(self) -> None:
        base = (
            "Paciente João, 65 anos, hipertenso, diabético, em uso de losartana "
            "50mg, HCTZ 25mg, metformina 850mg. PA 160x95 mmHg, FC 78bpm. "
        )
        long_note = base * 20  # ~2500 characters
        assert len(long_note) > 2000

        result = _invoke(long_note)

        assert "report" in result, "Pipeline should handle long notes without crashing"
        assert isinstance(result["report"], dict)

    def test_abbreviation_heavy_note(self) -> None:
        note = "Pct M 72a HAS DM2 ICC CF III FA crôn. BRA + HCTZ + BB + iSGLT2."

        result = _invoke(note)

        entities = result.get("extracted_entities", [])
        if not entities:
            warnings.warn("LLM did not extract entities from abbreviation-heavy note.")
        assert "report" in result

    def test_english_note(self) -> None:
        note = "72M patient with uncontrolled hypertension, BP 180/110, on losartan 50mg."

        result = _invoke(note)

        assert "report" in result, "Pipeline should handle English input"
        assert "evaluation" in result
