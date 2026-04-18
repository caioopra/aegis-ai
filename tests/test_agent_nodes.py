"""Unit tests for aegis.agent.nodes — processing node functions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from aegis.agent.nodes import (
    _extract_medication_names,
    _match_patient_id,
    _select_dynamic_tools,
    check_allergy_safety,
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
    return patch("aegis.fhir.get_store", return_value=store)


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
        ):
            state = {"patient_note": "Paciente com dor"}
            result = parse_note(state)

        assert result["patient_id"] == ""
        assert result["patient_id_match_type"] == "none"

    def test_sets_match_type_none_when_no_match(self, loaded_store: FHIRStore):
        mock_result = {
            "entities": [{"text": "Maria", "type": "patient", "normalized": "Maria Santos"}]
        }
        with (
            patch("aegis.agent.nodes.extract_entities", return_value=mock_result),
            _patch_store(loaded_store),
        ):
            state = {"patient_note": "Paciente Maria"}
            result = parse_note(state)

        assert result["patient_id"] == ""
        assert result["patient_id_match_type"] == "none"
        assert any("não identificado" in w for w in result["warnings"])

    def test_recovers_from_llm_failure(self, loaded_store: FHIRStore):
        with (
            patch("aegis.agent.nodes.extract_entities", side_effect=ValueError("LLM down")),
            _patch_store(loaded_store),
        ):
            state = {"patient_note": "Paciente João"}
            result = parse_note(state)

        assert result["extracted_entities"] == []
        assert any("falha" in w for w in result["warnings"])
        # Should still have a patient_id (matched from note text, not entities)
        assert result["patient_id"] != ""

    def test_handles_non_list_entities(self, loaded_store: FHIRStore):
        mock_result = {"entities": "not a list"}
        with (
            patch("aegis.agent.nodes.extract_entities", return_value=mock_result),
            _patch_store(loaded_store),
        ):
            state = {"patient_note": "Paciente com dor"}
            result = parse_note(state)

        assert result["extracted_entities"] == []
        assert any("lista" in w for w in result["warnings"])

    def test_matches_patient_from_note_even_without_entity(self, loaded_store: FHIRStore):
        """When LLM doesn't extract patient name as entity, note text is still used."""
        mock_result = {
            "entities": [
                {"text": "HAS", "type": "condition", "normalized": "Hipertensão arterial"},
            ]
        }
        with (
            patch("aegis.agent.nodes.extract_entities", return_value=mock_result),
            _patch_store(loaded_store),
        ):
            state = {"patient_note": "Paciente João, 65a, com HAS descompensada"}
            result = parse_note(state)

        # Should match João from the note text itself
        assert result["patient_id"] == PATIENT_ID
        assert result["patient_id_match_type"] in ("exact", "partial")


# ------------------------------------------------------------------
# _match_patient_id
# ------------------------------------------------------------------


class TestMatchPatientId:
    """Verify patient ID matching from entities and note text."""

    def test_matches_by_name_in_entities(self, loaded_store: FHIRStore):
        entities = [{"text": "João", "type": "patient", "normalized": "João Silva"}]
        with _patch_store(loaded_store):
            patient_id, match_type = _match_patient_id(entities)
        assert patient_id == PATIENT_ID
        assert match_type in ("exact", "partial")

    def test_matches_by_name_in_note(self, loaded_store: FHIRStore):
        """Name in the note text (not in entities) should still match."""
        entities = [{"text": "HAS", "type": "condition", "normalized": "hipertensão"}]
        with _patch_store(loaded_store):
            patient_id, match_type = _match_patient_id(entities, note="Paciente João, 65a, HAS")
        assert patient_id == PATIENT_ID
        assert match_type in ("exact", "partial")

    def test_matches_first_name_only_from_note(self, loaded_store: FHIRStore):
        """Just 'João' in the note should match (partial)."""
        with _patch_store(loaded_store):
            patient_id, match_type = _match_patient_id([], note="Paciente João com queixas")
        assert patient_id == PATIENT_ID
        assert match_type in ("exact", "partial")

    def test_returns_none_when_no_name_matches(self, loaded_store: FHIRStore):
        entities = [{"text": "Maria", "type": "patient", "normalized": "Maria Santos"}]
        with _patch_store(loaded_store):
            patient_id, match_type = _match_patient_id(entities, note="Paciente Maria")
        # No match for Maria — returns empty instead of wrong patient
        assert patient_id == ""
        assert match_type == "none"

    def test_returns_none_when_no_patients(self):
        entities = [{"text": "João", "type": "patient", "normalized": "João"}]
        with _patch_store(FHIRStore()):
            patient_id, match_type = _match_patient_id(entities)
        assert patient_id == ""
        assert match_type == "none"

    def test_matches_full_name_from_note(self, loaded_store: FHIRStore):
        """Full name 'João Carlos Silva' in the note should be exact match."""
        with _patch_store(loaded_store):
            patient_id, match_type = _match_patient_id(
                [], note="Paciente João Carlos Silva, 65 anos"
            )
        assert patient_id == PATIENT_ID
        assert match_type == "exact"

    def test_word_boundary_prevents_false_positive(self):
        """'Ana' in entity should NOT match a patient named 'Anamnese Teste'."""
        store = FHIRStore()
        # Manually add a patient whose name contains 'Anamnese' — substring
        # of 'Ana' should NOT match thanks to word-boundary regex.
        store._patients["patient-anamnese"] = {
            "resourceType": "Patient",
            "id": "patient-anamnese",
            "name": [{"use": "official", "given": ["Anamnese"], "family": "Teste"}],
            "birthDate": "1990-01-01",
            "gender": "female",
        }
        entities = [{"text": "Ana", "type": "patient", "normalized": "Ana"}]
        with _patch_store(store):
            patient_id, match_type = _match_patient_id(entities, note="Paciente Ana")
        # 'Ana' should NOT match 'Anamnese' — word boundary prevents it
        assert patient_id == ""
        assert match_type == "none"

    def test_word_boundary_allows_correct_match(self):
        """'Ana' should match patient 'Ana Maria Santos'."""
        store = FHIRStore()
        store._patients["patient-ana"] = {
            "resourceType": "Patient",
            "id": "patient-ana",
            "name": [{"use": "official", "given": ["Ana", "Maria"], "family": "Santos"}],
            "birthDate": "1985-05-15",
            "gender": "female",
        }
        entities = [{"text": "Ana", "type": "patient", "normalized": "Ana"}]
        with _patch_store(store):
            patient_id, match_type = _match_patient_id(entities, note="Paciente Ana")
        assert patient_id == "patient-ana"
        assert match_type in ("exact", "partial")


# ------------------------------------------------------------------
# _select_dynamic_tools
# ------------------------------------------------------------------


class TestSelectDynamicTools:
    """Verify entity-driven dynamic tool selection."""

    def test_returns_empty_for_no_matching_entities(self):
        entities = [{"text": "HAS", "type": "condition", "normalized": "hipertensão"}]
        result = _select_dynamic_tools(entities)
        assert result == []

    def test_selects_procedimentos_by_keyword(self):
        entities = [
            {"text": "ecocardiograma", "type": "procedure", "normalized": "ecocardiograma"},
        ]
        result = _select_dynamic_tools(entities)
        assert "consultar_procedimentos" in result

    def test_selects_exames_by_keyword(self):
        entities = [
            {"text": "HbA1c", "type": "lab", "normalized": "hemoglobina glicada"},
        ]
        result = _select_dynamic_tools(entities)
        assert "consultar_exames" in result

    def test_selects_encontros_by_keyword(self):
        entities = [
            {"text": "internação", "type": "event", "normalized": "internação hospitalar"},
        ]
        result = _select_dynamic_tools(entities)
        assert "consultar_encontros" in result

    def test_selects_imunizacoes_by_keyword(self):
        entities = [
            {"text": "vacina", "type": "procedure", "normalized": "vacinação"},
        ]
        result = _select_dynamic_tools(entities)
        assert "consultar_imunizacoes" in result

    def test_alergias_not_in_dynamic_tools(self):
        """consultar_alergias is now a base tool — the dynamic selector never returns it."""
        entities = [
            {"text": "alergia a penicilina", "type": "condition", "normalized": "alergia"},
        ]
        result = _select_dynamic_tools(entities)
        assert "consultar_alergias" not in result

    def test_selects_multiple_tools(self):
        entities = [
            {"text": "ecocardiograma", "type": "procedure", "normalized": "ecocardiograma"},
            {"text": "hemograma", "type": "lab", "normalized": "hemograma completo"},
        ]
        result = _select_dynamic_tools(entities)
        assert "consultar_procedimentos" in result
        assert "consultar_exames" in result
        assert "consultar_alergias" not in result

    def test_selects_by_entity_type(self):
        entities = [{"text": "ECG", "type": "procedure", "normalized": "eletrocardiograma"}]
        result = _select_dynamic_tools(entities)
        assert "consultar_procedimentos" in result

    def test_selects_exames_by_lab_result_type(self):
        entities = [
            {"text": "HbA1c 8.2", "type": "lab_result", "normalized": "Hemoglobina glicada 8,2%"},
        ]
        result = _select_dynamic_tools(entities)
        assert "consultar_exames" in result

    def test_allergy_entity_type_not_in_dynamic_tools(self):
        """Entity type 'allergy' no longer triggers dynamic tool selection."""
        entities = [
            {"text": "penicilina", "type": "allergy", "normalized": "Alergia a penicilina"},
        ]
        result = _select_dynamic_tools(entities)
        assert "consultar_alergias" not in result

    def test_returns_sorted(self):
        entities = [
            {"text": "vacina covid", "type": "procedure", "normalized": "vacinação"},
            {"text": "alergia", "type": "condition", "normalized": "alergia"},
        ]
        result = _select_dynamic_tools(entities)
        assert result == sorted(result)


# ------------------------------------------------------------------
# _extract_medication_names
# ------------------------------------------------------------------


class TestExtractMedicationNames:
    """Verify medication name extraction from entities."""

    def test_extracts_medications(self):
        entities = [
            {"text": "losartana 50mg", "type": "medication", "normalized": "Losartana 50 mg"},
            {"text": "HAS", "type": "condition", "normalized": "hipertensão"},
            {"text": "HCTZ 25mg", "type": "medication", "normalized": "Hidroclorotiazida 25 mg"},
        ]
        result = _extract_medication_names(entities)
        assert len(result) == 2
        assert "Losartana 50 mg" in result
        assert "Hidroclorotiazida 25 mg" in result

    def test_returns_empty_for_no_meds(self):
        entities = [{"text": "HAS", "type": "condition", "normalized": "hipertensão"}]
        assert _extract_medication_names(entities) == []

    def test_prefers_normalized_name(self):
        entities = [
            {"text": "losartana", "type": "medication", "normalized": "Losartana potássica"},
        ]
        result = _extract_medication_names(entities)
        assert result == ["Losartana potássica"]

    def test_falls_back_to_text(self):
        entities = [
            {"text": "losartana 50mg", "type": "medication", "normalized": ""},
        ]
        result = _extract_medication_names(entities)
        assert result == ["losartana 50mg"]


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

    def test_safety_net_forces_retrieval_on_condition(self):
        """LLM says no, but a condition is present — safety net overrides."""
        mock_result = {"needs_retrieval": False, "queries": []}
        with patch("aegis.agent.nodes.llm_decide_retrieval", return_value=mock_result):
            state = {
                "patient_note": "Paciente com HAS leve",
                "extracted_entities": [
                    {"text": "HAS", "type": "condition", "normalized": "Hipertensão arterial"},
                ],
            }
            result = decide_retrieval(state)

        assert result["needs_retrieval"] is True
        assert any("segurança" in w for w in result["warnings"])
        # Safety net should seed queries from the clinical entity
        assert result["retrieval_queries"]
        assert any("Hipertensão" in q for q in result["retrieval_queries"])

    def test_safety_net_forces_retrieval_on_medication(self):
        """LLM says no, but a medication is present — safety net overrides."""
        mock_result = {"needs_retrieval": False, "queries": []}
        with patch("aegis.agent.nodes.llm_decide_retrieval", return_value=mock_result):
            state = {
                "patient_note": "Retorno em uso de losartana",
                "extracted_entities": [
                    {
                        "text": "losartana",
                        "type": "medication",
                        "normalized": "Losartana 50 mg",
                    },
                ],
            }
            result = decide_retrieval(state)

        assert result["needs_retrieval"] is True
        assert any("segurança" in w for w in result["warnings"])

    def test_safety_net_does_not_fire_without_clinical_entities(self):
        """Routine note with only a symptom — LLM decision (False) is respected."""
        mock_result = {"needs_retrieval": False, "queries": []}
        with patch("aegis.agent.nodes.llm_decide_retrieval", return_value=mock_result):
            state = {
                "patient_note": "Paciente refere leve cefaleia passageira",
                "extracted_entities": [
                    {"text": "cefaleia", "type": "symptom", "normalized": "Cefaleia"},
                ],
            }
            result = decide_retrieval(state)

        assert result["needs_retrieval"] is False
        assert result["retrieval_queries"] == []
        assert not any("segurança" in w for w in result.get("warnings", []))


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
    """Verify fetch_patient_data calls base + dynamic MCP tools."""

    def test_fetches_all_base_sections(self, loaded_store: FHIRStore):
        with (
            _patch_store(loaded_store),
        ):
            state = {"patient_id": PATIENT_ID}
            result = fetch_patient_data(state)

        data = result["patient_data"]
        assert "João Carlos Silva" in data
        assert "Hipertensão" in data
        assert "Losartana" in data
        assert "Pressão arterial" in data
        # Should track tools called
        assert "consultar_paciente" in result["tools_called"]
        assert "consultar_condicoes" in result["tools_called"]

    def test_no_patient_id(self):
        state = {"patient_id": ""}
        result = fetch_patient_data(state)
        assert "não identificado" in result["patient_data"]
        assert "não recuperados" in result["patient_data"]
        assert result["tools_called"] == []

    def test_recovers_from_partial_tool_failure(self, loaded_store: FHIRStore):
        with (
            _patch_store(loaded_store),
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

    def test_calls_dynamic_tools_for_procedure_entities(self, loaded_store: FHIRStore):
        with (
            _patch_store(loaded_store),
        ):
            state = {
                "patient_id": PATIENT_ID,
                "extracted_entities": [
                    {"text": "ecocardiograma", "type": "procedure", "normalized": "ecocardiograma"},
                ],
            }
            result = fetch_patient_data(state)

        assert "consultar_procedimentos" in result["tools_called"]
        assert "Ecocardiograma" in result["patient_data"]

    def test_calls_dynamic_tools_for_exam_entities(self, loaded_store: FHIRStore):
        with (
            _patch_store(loaded_store),
        ):
            state = {
                "patient_id": PATIENT_ID,
                "extracted_entities": [
                    {"text": "HbA1c", "type": "lab", "normalized": "hemoglobina glicada"},
                ],
            }
            result = fetch_patient_data(state)

        assert "consultar_exames" in result["tools_called"]
        assert "HbA1c" in result["patient_data"]

    def test_allergy_tool_always_called_as_base(self, loaded_store: FHIRStore):
        """consultar_alergias is a base tool — called even with no allergy-related entities."""
        with (
            _patch_store(loaded_store),
        ):
            state = {
                "patient_id": PATIENT_ID,
                "extracted_entities": [
                    {"text": "HAS", "type": "condition", "normalized": "hipertensão"},
                ],
            }
            result = fetch_patient_data(state)

        assert "consultar_alergias" in result["tools_called"]
        assert "Penicilina" in result["patient_data"]

    def test_checks_medication_interactions(self, loaded_store: FHIRStore):
        with (
            _patch_store(loaded_store),
        ):
            state = {
                "patient_id": PATIENT_ID,
                "extracted_entities": [
                    {"text": "losartana 50mg", "type": "medication", "normalized": "Losartana"},
                    {
                        "text": "espironolactona",
                        "type": "medication",
                        "normalized": "Espironolactona",
                    },
                ],
            }
            result = fetch_patient_data(state)

        # Should have called interaction check
        interaction_calls = [t for t in result["tools_called"] if "interacao" in t]
        assert len(interaction_calls) == 1
        assert "hipercalemia" in result["patient_data"].lower()

    def test_no_dynamic_tools_for_simple_note(self, loaded_store: FHIRStore):
        with (
            _patch_store(loaded_store),
        ):
            state = {
                "patient_id": PATIENT_ID,
                "extracted_entities": [
                    {"text": "HAS", "type": "condition", "normalized": "hipertensão"},
                ],
            }
            result = fetch_patient_data(state)

        # Only the 5 base tools should have been called (allergy is now always-on)
        assert len(result["tools_called"]) == 5


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

    def test_generate_report_truncates_large_context(self):
        mock_report = {
            "patient_summary": "Resumo",
            "findings": ["achado"],
            "assessment": "avaliação",
            "plan": ["plano"],
        }
        with patch("aegis.agent.nodes.llm_generate_report", return_value=mock_report) as mock:
            state = {
                "patient_note": "Paciente João",
                "patient_data": "x" * 30000,  # huge
                "guidelines": "y" * 30000,  # huge
                "retry_count": 0,
            }
            result = generate_report(state)

            # The function should still succeed (truncated input)
            assert result["report"] == mock_report
            # It should add a warning about truncation
            assert any("truncado" in w for w in result["warnings"])
            # The actual data passed to llm should be truncated
            call_kwargs = mock.call_args.kwargs
            assert len(call_kwargs["patient_data"]) < 30000
            assert len(call_kwargs["guidelines"]) < 30000

    def test_generate_report_no_truncation_for_small_context(self):
        mock_report = {"findings": []}
        with patch("aegis.agent.nodes.llm_generate_report", return_value=mock_report):
            state = {
                "patient_note": "Nota curta",
                "patient_data": "Dados curtos",
                "guidelines": "Diretrizes curtas",
                "retry_count": 0,
            }
            result = generate_report(state)

            # No truncation warning expected
            assert not any("truncado" in w for w in result["warnings"])

    def test_attaches_ai_disclaimer(self):
        """Every report must carry the hardcoded AI disclaimer."""
        from aegis.agent.nodes import AI_DISCLAIMER

        mock_report = {"findings": ["achado"], "plan": ["plano"]}
        with patch("aegis.agent.nodes.llm_generate_report", return_value=mock_report):
            state = {"patient_note": "Nota"}
            result = generate_report(state)

        assert result["report"]["disclaimer"] == AI_DISCLAIMER
        assert "IA" in AI_DISCLAIMER
        assert "NÃO substitui" in AI_DISCLAIMER

    def test_disclaimer_present_on_error_path(self):
        """Even an error-path fallback report must carry the disclaimer."""
        from aegis.agent.nodes import AI_DISCLAIMER

        with patch("aegis.agent.nodes.llm_generate_report", side_effect=ValueError("boom")):
            state = {"patient_note": "Nota"}
            result = generate_report(state)

        assert result["report"]["disclaimer"] == AI_DISCLAIMER
        assert "error" in result["report"]


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


# ------------------------------------------------------------------
# check_allergy_safety
# ------------------------------------------------------------------


class TestCheckAllergySafety:
    """Verify allergy-prescription conflict detection."""

    def _make_report(
        self,
        plan: list[str] | None = None,
        sinais_alarme: list[str] | None = None,
    ) -> dict:
        return {
            "patient_summary": "Paciente teste",
            "findings": [],
            "assessment": "avaliação",
            "plan": plan or [],
            "sinais_alarme": sinais_alarme or [],
            "disclaimer": "Este relatório foi gerado por IA.",
        }

    def _make_state(self, report: dict, patient_data: str = "") -> dict:
        return {"report": report, "patient_data": patient_data}

    def test_no_conflict_no_warning(self):
        """Plan with a non-allergy drug and a penicillin allergy → no warning."""
        report = self._make_report(
            plan=["Manter losartana 50mg 1x/dia"],
            sinais_alarme=["Monitorar PA"],
        )
        state = self._make_state(report, patient_data="Alergia: Penicilina (SNOMED 91936005)")
        result = check_allergy_safety(state)

        assert result["warnings"] == []
        assert result["report"]["sinais_alarme"] == ["Monitorar PA"]

    def test_penicillin_conflict(self):
        """Plan with amoxicilina and patient allergic to penicilina → warning."""
        report = self._make_report(
            plan=["Iniciar amoxicilina 500mg 8/8h por 7 dias"],
        )
        state = self._make_state(
            report, patient_data="Alergia registrada: Penicilina (reação grave)"
        )
        result = check_allergy_safety(state)

        assert len(result["warnings"]) == 1
        assert "penicilina" in result["warnings"][0].lower()
        assert "amoxicilina" in result["warnings"][0].lower()
        assert result["warnings"][0] in result["report"]["sinais_alarme"]

    def test_sulfa_conflict(self):
        """Plan with sulfametoxazol and patient allergic to sulfa → warning."""
        report = self._make_report(
            plan=["Prescrever sulfametoxazol+trimetoprima 2x/dia"],
        )
        state = self._make_state(report, patient_data="Histórico: alergia a sulfa com rash cutâneo")
        result = check_allergy_safety(state)

        assert len(result["warnings"]) >= 1
        assert any("sulfa" in w.lower() for w in result["warnings"])

    def test_aine_conflict(self):
        """Plan with ibuprofeno and patient allergic to AINE → warning."""
        report = self._make_report(
            plan=["Iniciar ibuprofeno 400mg 8/8h para dor"],
        )
        state = self._make_state(
            report,
            patient_data="Alergia: AINE (ibuprofeno, diclofenaco) — broncoespasmo",
        )
        result = check_allergy_safety(state)

        assert len(result["warnings"]) >= 1
        assert any("aine" in w.lower() for w in result["warnings"])

    def test_cefalosporina_conflict(self):
        """Plan with cefalexina and patient allergic to cefalosporina → warning."""
        report = self._make_report(
            plan=["Iniciar cefalexina 500mg 6/6h por 10 dias"],
        )
        state = self._make_state(
            report, patient_data="Alergia conhecida: cefalosporina de primeira geração"
        )
        result = check_allergy_safety(state)

        assert len(result["warnings"]) >= 1
        assert any("cefalosporina" in w.lower() for w in result["warnings"])

    def test_empty_plan_early_return(self):
        """Empty plan → no processing, no warnings."""
        report = self._make_report(plan=[])
        state = self._make_state(report, patient_data="Alergia: Penicilina")
        result = check_allergy_safety(state)

        assert result["warnings"] == []
        assert result["report"] is report  # unchanged reference

    def test_empty_allergies_early_return(self):
        """patient_data has no allergy mentions → no warnings, early return."""
        report = self._make_report(
            plan=["Iniciar amoxicilina 500mg"],
        )
        state = self._make_state(report, patient_data="Sem alergias conhecidas. PA 130/85.")
        result = check_allergy_safety(state)

        assert result["warnings"] == []

    def test_error_report_skipped(self):
        """Report with 'error' key → node returns early, no modification."""
        report = {
            "error": "LLM failure",
            "plan": ["Iniciar amoxicilina"],
            "disclaimer": "aviso IA",
        }
        state = self._make_state(report, patient_data="Alergia: Penicilina")
        result = check_allergy_safety(state)

        assert result["warnings"] == []
        assert result["report"] is report

    def test_existing_sinais_alarme_preserved(self):
        """Warning is inserted at index 0; pre-existing entries remain."""
        existing = ["Monitorar glicemia", "Controlar PA"]
        report = self._make_report(
            plan=["Iniciar amoxicilina 875mg"],
            sinais_alarme=list(existing),
        )
        state = self._make_state(report, patient_data="Alergia: Penicilina grave")
        result = check_allergy_safety(state)

        sinais = result["report"]["sinais_alarme"]
        assert len(sinais) == 3  # 1 new + 2 existing
        # Warning is first
        assert "ALERTA DE ALERGIA" in sinais[0]
        # Existing entries are still there
        assert "Monitorar glicemia" in sinais
        assert "Controlar PA" in sinais

    def test_multiple_conflicts(self):
        """Plan with amoxicilina + ibuprofeno, allergic to penicilina AND AINE → two warnings."""
        report = self._make_report(
            plan=[
                "Iniciar amoxicilina 500mg 8/8h",
                "Ibuprofeno 400mg se dor",
            ],
        )
        state = self._make_state(
            report,
            patient_data=(
                "Alergias: Penicilina (reação anafilática), AINE (urticária com diclofenaco)"
            ),
        )
        result = check_allergy_safety(state)

        assert len(result["warnings"]) == 2
        assert len(result["report"]["sinais_alarme"]) == 2
        warning_text = " ".join(result["warnings"])
        assert "penicilina" in warning_text.lower()
        assert "aine" in warning_text.lower()

    def test_aine_conflict_with_aas(self):
        """Plan with AAS and patient allergic to AINE → warning (B4 fix)."""
        report = self._make_report(
            plan=["Iniciar AAS 100mg 1x/dia para antiagregação plaquetária"],
        )
        state = self._make_state(
            report,
            patient_data="Alergia registrada: AINE (broncoespasmo com uso de anti-inflamatório)",
        )
        result = check_allergy_safety(state)

        assert len(result["warnings"]) >= 1
        assert any("aine" in w.lower() for w in result["warnings"])
        assert any("aas" in w.lower() for w in result["warnings"])

    def test_aine_conflict_with_meloxicam(self):
        """Plan with meloxicam and patient allergic to AINE → warning (B4 fix)."""
        report = self._make_report(
            plan=["Prescrever meloxicam 15mg 1x/dia para artralgia"],
        )
        state = self._make_state(
            report,
            patient_data="Histórico de alergia a AINE com urticária",
        )
        result = check_allergy_safety(state)

        assert len(result["warnings"]) >= 1
        assert any("aine" in w.lower() for w in result["warnings"])
        assert any("meloxicam" in w.lower() for w in result["warnings"])


# ------------------------------------------------------------------
# _match_patient_id — CPF matching
# ------------------------------------------------------------------


class TestMatchPatientIdCpf:
    """Verify CPF-based patient identification in _match_patient_id."""

    @pytest.fixture
    def loaded_store(self) -> FHIRStore:
        from pathlib import Path

        store = FHIRStore()
        store.load_bundle(Path("data/synthea/sample_patient_joao.json"))
        return store

    def test_cpf_exact_formatted(self, loaded_store: FHIRStore):
        """Formatted CPF 111.111.111-11 in the note resolves to João."""
        with _patch_store(loaded_store):
            patient_id, match_type = _match_patient_id(
                [], note="Paciente CPF 111.111.111-11 com HAS e ICC"
            )
        assert patient_id == PATIENT_ID
        assert match_type == "cpf"

    def test_cpf_unformatted(self, loaded_store: FHIRStore):
        """Bare digit CPF 11111111111 in the note resolves to João."""
        with _patch_store(loaded_store):
            patient_id, match_type = _match_patient_id(
                [], note="CPF: 11111111111 em acompanhamento"
            )
        assert patient_id == PATIENT_ID
        assert match_type == "cpf"

    def test_cpf_partial_format(self, loaded_store: FHIRStore):
        """CPF with dashes only (no dots) resolves correctly."""
        with _patch_store(loaded_store):
            patient_id, match_type = _match_patient_id([], note="Paciente 111111111-11 retorno")
        assert patient_id == PATIENT_ID
        assert match_type == "cpf"

    def test_cpf_in_entity(self, loaded_store: FHIRStore):
        """CPF present only in entity text (not in note) resolves to João."""
        entities = [{"text": "111.111.111-11", "type": "identifier", "normalized": "CPF"}]
        with _patch_store(loaded_store):
            patient_id, match_type = _match_patient_id(entities, note="Retorno clínico")
        assert patient_id == PATIENT_ID
        assert match_type == "cpf"

    def test_cpf_not_found_falls_back_to_name(self, loaded_store: FHIRStore):
        """Invalid CPF → no CPF hit → falls through to name matching."""
        with _patch_store(loaded_store):
            patient_id, match_type = _match_patient_id(
                [],
                note="Paciente João, CPF 999.999.999-99, com HAS",
            )
        # CPF 999.999.999-99 has no matching patient → falls back to name "João"
        assert patient_id == PATIENT_ID
        assert match_type in ("exact", "partial")

    def test_no_cpf_no_name_returns_none(self, loaded_store: FHIRStore):
        """No CPF and no name in note → ("", "none")."""
        with _patch_store(loaded_store):
            patient_id, match_type = _match_patient_id(
                [], note="Paciente com cefaleia intensa e febre"
            )
        assert patient_id == ""
        assert match_type == "none"

    def test_cpf_takes_precedence_over_name(self, loaded_store: FHIRStore):
        """Valid CPF and a name that would also match → CPF wins, match_type is 'cpf'."""
        with _patch_store(loaded_store):
            patient_id, match_type = _match_patient_id(
                [],
                note="Paciente João Carlos Silva, CPF 111.111.111-11, retorno",
            )
        assert patient_id == PATIENT_ID
        assert match_type == "cpf"


# ------------------------------------------------------------------
# parse_note — CPF name-consistency check
# ------------------------------------------------------------------


class TestParseNoteCpfNameMismatch:
    """Verify parse_note emits a warning when the CPF-resolved patient's name
    is absent from the note (potential data-mix-up)."""

    @pytest.fixture
    def loaded_store(self) -> FHIRStore:
        store = FHIRStore()
        store.load_bundle(Path("data/synthea/sample_patient_joao.json"))
        return store

    def test_cpf_match_name_mismatch_warns(self, loaded_store: FHIRStore):
        """CPF resolves to João but the note names 'Maria' → consistency warning."""
        mock_result = {
            "entities": [
                {"text": "Maria", "type": "patient", "normalized": "Maria"},
            ]
        }
        with (
            patch("aegis.agent.nodes.extract_entities", return_value=mock_result),
            _patch_store(loaded_store),
        ):
            # Note mentions Maria and contains João's CPF — suspicious mix-up
            state = {"patient_note": "Paciente Maria, CPF 111.111.111-11, queixa de dor torácica"}
            result = parse_note(state)

        # Should have resolved to João via CPF
        assert result["patient_id"] == PATIENT_ID
        assert result["patient_id_match_type"] == "cpf"
        # Should have warned about the name mismatch
        assert any("CPF resolveu" in w for w in result["warnings"])
        assert any("nome não consta" in w for w in result["warnings"])

    def test_cpf_match_name_matches_no_extra_warning(self, loaded_store: FHIRStore):
        """CPF resolves to João and note contains 'João' → no mismatch warning."""
        mock_result = {
            "entities": [
                {"text": "HAS", "type": "condition", "normalized": "hipertensão"},
            ]
        }
        with (
            patch("aegis.agent.nodes.extract_entities", return_value=mock_result),
            _patch_store(loaded_store),
        ):
            state = {"patient_note": "Paciente João, CPF 111.111.111-11, retorno para HAS"}
            result = parse_note(state)

        assert result["patient_id"] == PATIENT_ID
        assert result["patient_id_match_type"] == "cpf"
        # No mismatch warning — name is present
        assert not any("CPF resolveu" in w for w in result["warnings"])
