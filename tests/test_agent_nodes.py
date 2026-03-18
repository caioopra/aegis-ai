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
        assert result["patient_id_match_type"] == "none"

    def test_sets_match_type_fallback(self, loaded_store: FHIRStore):
        mock_result = {
            "entities": [{"text": "Maria", "type": "patient", "normalized": "Maria Santos"}]
        }
        with (
            patch("aegis.agent.nodes.extract_entities", return_value=mock_result),
            _patch_store(loaded_store),
            _patch_ensure(),
        ):
            state = {"patient_note": "Paciente Maria"}
            result = parse_note(state)

        assert result["patient_id_match_type"] == "fallback"
        assert any("fallback" in w for w in result["warnings"])

    def test_recovers_from_llm_failure(self, loaded_store: FHIRStore):
        with (
            patch("aegis.agent.nodes.extract_entities", side_effect=ValueError("LLM down")),
            _patch_store(loaded_store),
            _patch_ensure(),
        ):
            state = {"patient_note": "Paciente João"}
            result = parse_note(state)

        assert result["extracted_entities"] == []
        assert any("falha" in w for w in result["warnings"])
        # Should still have a patient_id (fallback)
        assert result["patient_id"] != ""

    def test_handles_non_list_entities(self, loaded_store: FHIRStore):
        mock_result = {"entities": "not a list"}
        with (
            patch("aegis.agent.nodes.extract_entities", return_value=mock_result),
            _patch_store(loaded_store),
            _patch_ensure(),
        ):
            state = {"patient_note": "Paciente com dor"}
            result = parse_note(state)

        assert result["extracted_entities"] == []
        assert any("lista" in w for w in result["warnings"])


# ------------------------------------------------------------------
# _match_patient_id
# ------------------------------------------------------------------


class TestMatchPatientId:
    """Verify patient ID matching from entities."""

    def test_matches_by_name(self, loaded_store: FHIRStore):
        entities = [{"text": "João", "type": "patient", "normalized": "João Silva"}]
        with _patch_store(loaded_store), _patch_ensure():
            patient_id, match_type = _match_patient_id(entities)
        assert patient_id == PATIENT_ID
        assert match_type in ("exact", "partial")

    def test_fallback_to_first_patient(self, loaded_store: FHIRStore):
        entities = [{"text": "Maria", "type": "patient", "normalized": "Maria Santos"}]
        with _patch_store(loaded_store), _patch_ensure():
            patient_id, match_type = _match_patient_id(entities)
        # No match for Maria, falls back to first available
        assert patient_id == PATIENT_ID
        assert match_type == "fallback"

    def test_returns_none_when_no_patients(self):
        entities = [{"text": "João", "type": "patient", "normalized": "João"}]
        with _patch_store(FHIRStore()), _patch_ensure():
            patient_id, match_type = _match_patient_id(entities)
        assert patient_id == ""
        assert match_type == "none"


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

    def test_recovers_from_llm_failure(self):
        with patch("aegis.agent.nodes.llm_decide_retrieval", side_effect=ValueError("LLM error")):
            state = {
                "patient_note": "Paciente com HAS",
                "extracted_entities": [],
            }
            result = decide_retrieval(state)

        # Should default to retrieval on failure
        assert result["needs_retrieval"] is True
        assert len(result["retrieval_queries"]) > 0
        assert any("falha" in w for w in result["warnings"])


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
        assert result["retrieval_confidence"] == 0.9

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
        assert result["retrieval_confidence"] == 0.0

    def test_low_confidence_adds_warning(self):
        fake_results = [
            {"text": "Vague result", "source": "x.txt", "chunk_index": 0, "score": 0.3},
        ]
        with patch("aegis.agent.nodes.retrieve", return_value=fake_results):
            state = {"retrieval_queries": ["obscure query"]}
            result = retrieve_guidelines(state)

        assert result["retrieval_confidence"] == 0.3
        assert any("confiança baixa" in w for w in result["warnings"])

    def test_recovers_from_rag_failure(self):
        with patch("aegis.agent.nodes.retrieve", side_effect=Exception("Qdrant down")):
            state = {"retrieval_queries": ["tratamento HAS"]}
            result = retrieve_guidelines(state)

        assert "indisponíveis" in result["guidelines"]
        assert result["retrieval_confidence"] == 0.0
        assert any("falha" in w for w in result["warnings"])


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

    def test_recovers_from_partial_tool_failure(self, loaded_store: FHIRStore):
        with (
            patch("aegis.mcp_server._store", loaded_store),
            patch("aegis.mcp_server._load_store"),
            patch(
                "aegis.agent.nodes.consultar_sinais_vitais",
                side_effect=Exception("Timeout"),
            ),
        ):
            state = {"patient_id": PATIENT_ID}
            result = fetch_patient_data(state)

        data = result["patient_data"]
        # Other sections should still be present
        assert "João Carlos Silva" in data
        # Failed section should have fallback message
        assert "indisponíveis" in data
        assert any("falhou" in w for w in result["warnings"])


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
                refinement_context="",
            )
        assert result["report"] == mock_report

    def test_handles_missing_context(self):
        mock_report = {"findings": []}
        with patch("aegis.agent.nodes.llm_generate_report", return_value=mock_report):
            state = {"patient_note": "Nota clínica"}
            result = generate_report(state)

        assert "report" in result

    def test_passes_refinement_context_on_retry(self):
        mock_report = {"findings": ["improved"]}
        with patch("aegis.agent.nodes.llm_generate_report", return_value=mock_report) as mock:
            state = {
                "patient_note": "Nota clínica",
                "retry_count": 1,
                "evaluation": {
                    "completeness": {"score": 2, "feedback": "Faltam achados"},
                    "overall": {"score": 2, "feedback": "Insuficiente"},
                },
            }
            result = generate_report(state)

            call_kwargs = mock.call_args.kwargs
            assert "Faltam achados" in call_kwargs["refinement_context"]
            assert "completeness" in call_kwargs["refinement_context"]
        assert result["report"] == mock_report

    def test_recovers_from_llm_failure(self):
        with patch("aegis.agent.nodes.llm_generate_report", side_effect=ValueError("LLM error")):
            state = {"patient_note": "Nota clínica"}
            result = generate_report(state)

        assert "error" in result["report"]
        assert any("falha" in w for w in result["warnings"])


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
            state = {
                "report": {"findings": ["HAS"], "plan": ["Ajustar medicação"]},
                "patient_note": "Test note",
                "patient_data": "Test data",
            }
            result = evaluate_report(state)

        assert result["evaluation"]["overall"]["score"] == 4

    def test_handles_empty_report(self):
        state = {"report": {}}
        result = evaluate_report(state)
        assert result["evaluation"]["overall"]["score"] == 0

    def test_handles_error_report(self):
        state = {"report": {"error": "LLM failed"}}
        result = evaluate_report(state)
        assert result["evaluation"]["overall"]["score"] == 0

    def test_passes_note_and_patient_data(self):
        mock_eval = {"overall": {"score": 4, "feedback": "ok"}}
        with patch("aegis.agent.nodes.llm_evaluate_report", return_value=mock_eval) as mock:
            state = {
                "report": {"findings": ["test"]},
                "patient_note": "Original note",
                "patient_data": "Patient context",
            }
            evaluate_report(state)

            mock.assert_called_once()
            call_kwargs = mock.call_args.kwargs
            assert call_kwargs["note"] == "Original note"
            assert call_kwargs["patient_data"] == "Patient context"

    def test_recovers_from_llm_failure(self):
        with patch("aegis.agent.nodes.llm_evaluate_report", side_effect=ValueError("LLM error")):
            state = {
                "report": {"findings": ["test"]},
                "patient_note": "Note",
            }
            result = evaluate_report(state)

        # Should return a default score rather than crashing
        assert result["evaluation"]["overall"]["score"] == 3
        assert any("falha" in w for w in result["warnings"])
