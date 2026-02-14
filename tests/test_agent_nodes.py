"""Unit tests for aegis.agent.nodes — processing node functions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from aegis.agent.nodes import (
    _match_patient_id,
    decide_retrieval,
    evaluate_report,
    fetch_patient_data,
    generate_report,
    parse_note,
    retrieve_guidelines,
)
from aegis.fhir import FHIRStore

SAMPLE_FILE = Path("data/synthea/sample_patient_joao.json")
PATIENT_ID = "patient-joao-001"


@pytest.fixture
def loaded_store() -> FHIRStore:
    store = FHIRStore()
    store.load_bundle(SAMPLE_FILE)
    return store


def _patch_store(store: FHIRStore):
    return patch("aegis.agent.nodes._store", store)


def _patch_ensure():
    return patch("aegis.agent.nodes._ensure_store")


# ------------------------------------------------------------------
# parse_note
# ------------------------------------------------------------------


class TestParseNote:
    """Verify parse_note extracts entities and matches patient."""

    def test_returns_entities_and_patient_id(self, loaded_store: FHIRStore):
        mock_result = {
            "entities": [
                {"text": "João", "type": "patient", "normalized": "João"},
                {"text": "dispneia", "type": "symptom", "normalized": "dispneia"},
            ]
        }
        with (
            patch("aegis.agent.nodes.extract_entities", return_value=mock_result),
            _patch_store(loaded_store),
            _patch_ensure(),
        ):
            state = {"patient_note": "Paciente João, 65a, dispneia"}
            result = parse_note(state)

        assert "extracted_entities" in result
        assert len(result["extracted_entities"]) == 2
        assert result["patient_id"] == PATIENT_ID

    def test_returns_empty_id_when_no_patients(self):
        mock_result = {"entities": [{"text": "dor", "type": "symptom", "normalized": "dor"}]}
        with (
            patch("aegis.agent.nodes.extract_entities", return_value=mock_result),
            _patch_store(FHIRStore()),
            _patch_ensure(),
        ):
            state = {"patient_note": "Paciente com dor"}
            result = parse_note(state)

        assert result["patient_id"] == ""


# ------------------------------------------------------------------
# _match_patient_id
# ------------------------------------------------------------------


class TestMatchPatientId:
    """Verify patient ID matching from entities."""

    def test_matches_by_name(self, loaded_store: FHIRStore):
        entities = [{"text": "João", "type": "patient", "normalized": "João Silva"}]
        with _patch_store(loaded_store), _patch_ensure():
            result = _match_patient_id(entities)
        assert result == PATIENT_ID

    def test_fallback_to_first_patient(self, loaded_store: FHIRStore):
        entities = [{"text": "Maria", "type": "patient", "normalized": "Maria Santos"}]
        with _patch_store(loaded_store), _patch_ensure():
            result = _match_patient_id(entities)
        # No match for Maria, falls back to first available
        assert result == PATIENT_ID

    def test_returns_empty_when_no_patients(self):
        entities = [{"text": "João", "type": "patient", "normalized": "João"}]
        with _patch_store(FHIRStore()), _patch_ensure():
            result = _match_patient_id(entities)
        assert result == ""


# ------------------------------------------------------------------
# decide_retrieval
# ------------------------------------------------------------------


class TestDecideRetrieval:
    """Verify decide_retrieval returns needs_retrieval and queries."""

    def test_needs_retrieval_true(self):
        mock_result = {
            "needs_retrieval": True,
            "queries": ["tratamento HAS estágio 2", "meta PA diabéticos"],
        }
        with patch("aegis.agent.nodes.llm_decide_retrieval", return_value=mock_result):
            state = {
                "patient_note": "Paciente com HAS descompensada",
                "extracted_entities": [{"text": "HAS", "type": "condition"}],
            }
            result = decide_retrieval(state)

        assert result["needs_retrieval"] is True
        assert len(result["retrieval_queries"]) == 2

    def test_needs_retrieval_false(self):
        mock_result = {"needs_retrieval": False, "queries": []}
        with patch("aegis.agent.nodes.llm_decide_retrieval", return_value=mock_result):
            state = {
                "patient_note": "Retorno de rotina, sem queixas",
                "extracted_entities": [],
            }
            result = decide_retrieval(state)

        assert result["needs_retrieval"] is False
        assert result["retrieval_queries"] == []


# ------------------------------------------------------------------
# retrieve_guidelines
# ------------------------------------------------------------------


class TestRetrieveGuidelines:
    """Verify retrieve_guidelines calls RAG and formats context."""

    def test_retrieves_and_formats(self):
        fake_results = [
            {
                "text": "Tratamento HAS estágio 2...",
                "source": "has.txt",
                "chunk_index": 0,
                "score": 0.9,
            },
        ]
        with patch("aegis.agent.nodes.retrieve", return_value=fake_results):
            state = {"retrieval_queries": ["tratamento HAS"]}
            result = retrieve_guidelines(state)

        assert "Tratamento HAS" in result["guidelines"]
        assert "has.txt" in result["guidelines"]

    def test_deduplicates_results(self):
        same_chunk = {"text": "Same chunk", "source": "a.txt", "chunk_index": 0, "score": 0.8}
        with patch("aegis.agent.nodes.retrieve", return_value=[same_chunk]):
            state = {"retrieval_queries": ["query1", "query2"]}
            result = retrieve_guidelines(state)

        # Same text should appear only once
        assert result["guidelines"].count("Same chunk") == 1

    def test_empty_queries(self):
        state = {"retrieval_queries": []}
        result = retrieve_guidelines(state)
        assert "Nenhuma" in result["guidelines"]


# ------------------------------------------------------------------
# fetch_patient_data
# ------------------------------------------------------------------


class TestFetchPatientData:
    """Verify fetch_patient_data calls MCP tools."""

    def test_fetches_all_sections(self, loaded_store: FHIRStore):
        with (
            patch("aegis.mcp_server._store", loaded_store),
            patch("aegis.mcp_server._load_store"),
        ):
            state = {"patient_id": PATIENT_ID}
            result = fetch_patient_data(state)

        data = result["patient_data"]
        assert "João Carlos Silva" in data
        assert "Hipertensão" in data
        assert "Losartana" in data
        assert "Pressão arterial" in data

    def test_no_patient_id(self):
        state = {"patient_id": ""}
        result = fetch_patient_data(state)
        assert "não identificado" in result["patient_data"]


# ------------------------------------------------------------------
# generate_report
# ------------------------------------------------------------------


class TestGenerateReport:
    """Verify generate_report calls the LLM with all context."""

    def test_passes_all_context(self):
        mock_report = {
            "patient_summary": "João, 65a, masculino",
            "findings": ["HAS descompensada"],
            "assessment": "Necessita ajuste terapêutico",
            "plan": ["Associar BCC"],
        }
        with patch("aegis.agent.nodes.llm_generate_report", return_value=mock_report) as mock:
            state = {
                "patient_note": "Nota clínica",
                "patient_data": "Dados do paciente",
                "guidelines": "Diretrizes relevantes",
            }
            result = generate_report(state)

            mock.assert_called_once_with(
                note="Nota clínica",
                patient_data="Dados do paciente",
                guidelines="Diretrizes relevantes",
            )
        assert result["report"] == mock_report

    def test_handles_missing_context(self):
        mock_report = {"findings": []}
        with patch("aegis.agent.nodes.llm_generate_report", return_value=mock_report):
            state = {"patient_note": "Nota clínica"}
            result = generate_report(state)

        assert "report" in result


# ------------------------------------------------------------------
# evaluate_report
# ------------------------------------------------------------------


class TestEvaluateReport:
    """Verify evaluate_report calls the LLM self-evaluator."""

    def test_returns_evaluation(self):
        mock_eval = {
            "completeness": {"score": 4, "feedback": "Bom"},
            "overall": {"score": 4, "feedback": "Adequado"},
        }
        with patch("aegis.agent.nodes.llm_evaluate_report", return_value=mock_eval):
            state = {"report": {"findings": ["HAS"], "plan": ["Ajustar medicação"]}}
            result = evaluate_report(state)

        assert result["evaluation"]["overall"]["score"] == 4

    def test_handles_empty_report(self):
        state = {"report": {}}
        result = evaluate_report(state)
        assert result["evaluation"]["overall"]["score"] == 0
