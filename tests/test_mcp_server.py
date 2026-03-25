"""Unit tests for aegis.mcp_server — MCP tool functions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from aegis.mcp_server import (
    _format_allergy,
    _format_condition,
    _format_diagnostic_report,
    _format_encounter,
    _format_immunization,
    _format_medication,
    _format_observation,
    _format_patient,
    _format_procedure,
    _normalize_drug_name,
    consultar_alergias,
    consultar_condicoes,
    consultar_encontros,
    consultar_exames,
    consultar_imunizacoes,
    consultar_medicamentos,
    consultar_paciente,
    consultar_procedimentos,
    consultar_sinais_vitais,
    listar_pacientes,
    verificar_interacao_medicamentosa,
)
from aegis.fhir import FHIRStore

SAMPLE_FILE = Path("data/synthea/sample_patient_joao.json")
PATIENT_ID = "patient-joao-001"


@pytest.fixture
def loaded_store() -> FHIRStore:
    """A FHIRStore pre-loaded with the sample patient."""
    store = FHIRStore()
    store.load_bundle(SAMPLE_FILE)
    return store


def _patch_store(store: FHIRStore):
    """Return a patch that makes get_store() return the given store."""
    return patch("aegis.fhir.get_store", return_value=store)


# ------------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------------


class TestFormatPatient:
    """Verify _format_patient output."""

    def test_includes_name(self, loaded_store: FHIRStore):
        patient = loaded_store.get_patient(PATIENT_ID)
        text = _format_patient(patient)
        assert "João Carlos Silva" in text

    def test_includes_gender(self, loaded_store: FHIRStore):
        patient = loaded_store.get_patient(PATIENT_ID)
        text = _format_patient(patient)
        assert "Masculino" in text

    def test_includes_birth_date(self, loaded_store: FHIRStore):
        patient = loaded_store.get_patient(PATIENT_ID)
        text = _format_patient(patient)
        assert "1960-03-15" in text

    def test_includes_age(self, loaded_store: FHIRStore):
        patient = loaded_store.get_patient(PATIENT_ID)
        text = _format_patient(patient)
        assert "anos)" in text

    def test_includes_address(self, loaded_store: FHIRStore):
        patient = loaded_store.get_patient(PATIENT_ID)
        text = _format_patient(patient)
        assert "São Paulo" in text


class TestFormatCondition:
    """Verify _format_condition output."""

    def test_includes_text(self):
        condition = {
            "code": {"text": "Hipertensão arterial sistêmica"},
            "clinicalStatus": {"coding": [{"code": "active"}]},
            "onsetDateTime": "2015-06-10",
        }
        result = _format_condition(condition)
        assert "Hipertensão arterial sistêmica" in result
        assert "active" in result
        assert "2015-06-10" in result

    def test_falls_back_to_display(self):
        condition = {
            "code": {"coding": [{"display": "Hypertension"}]},
        }
        result = _format_condition(condition)
        assert "Hypertension" in result

    def test_unknown_when_no_code(self):
        result = _format_condition({"code": {}})
        assert "Desconhecido" in result


class TestFormatMedication:
    """Verify _format_medication output."""

    def test_includes_name_and_dosage(self):
        med = {
            "medicationCodeableConcept": {"text": "Losartana 50mg"},
            "dosageInstruction": [{"text": "1 comprimido por dia"}],
            "status": "active",
        }
        result = _format_medication(med)
        assert "Losartana 50mg" in result
        assert "1 comprimido por dia" in result
        assert "active" in result


class TestFormatObservation:
    """Verify _format_observation output."""

    def test_simple_value(self):
        obs = {
            "code": {"text": "Frequência cardíaca"},
            "valueQuantity": {"value": 88, "unit": "bpm"},
        }
        result = _format_observation(obs)
        assert "Frequência cardíaca" in result
        assert "88" in result
        assert "bpm" in result

    def test_component_value(self):
        obs = {
            "code": {"text": "Pressão arterial"},
            "component": [
                {
                    "code": {"text": "Pressão sistólica"},
                    "valueQuantity": {"value": 150, "unit": "mmHg"},
                },
                {
                    "code": {"text": "Pressão diastólica"},
                    "valueQuantity": {"value": 95, "unit": "mmHg"},
                },
            ],
        }
        result = _format_observation(obs)
        assert "Pressão arterial" in result
        assert "150" in result
        assert "95" in result


# ------------------------------------------------------------------
# MCP Tools — listar_pacientes
# ------------------------------------------------------------------


class TestListarPacientes:
    """Verify the listar_pacientes tool."""

    def test_lists_sample_patient(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store):
            result = listar_pacientes()
        assert "João Carlos Silva" in result
        assert PATIENT_ID in result

    def test_shows_count(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store):
            result = listar_pacientes()
        assert "1)" in result

    def test_empty_store(self):
        with _patch_store(FHIRStore()):
            result = listar_pacientes()
        assert "Nenhum paciente" in result


# ------------------------------------------------------------------
# MCP Tools — consultar_paciente
# ------------------------------------------------------------------


class TestConsultarPaciente:
    """Verify the consultar_paciente tool."""

    def test_returns_demographics(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store):
            result = consultar_paciente(PATIENT_ID)
        assert "João Carlos Silva" in result
        assert "Masculino" in result
        assert "1960-03-15" in result

    def test_not_found(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store):
            result = consultar_paciente("nonexistent")
        assert "não encontrado" in result


# ------------------------------------------------------------------
# MCP Tools — consultar_condicoes
# ------------------------------------------------------------------


class TestConsultarCondicoes:
    """Verify the consultar_condicoes tool."""

    def test_lists_conditions(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store):
            result = consultar_condicoes(PATIENT_ID)
        assert "Hipertensão arterial sistêmica" in result
        assert "Diabetes mellitus tipo 2" in result
        assert "Insuficiência cardíaca congestiva" in result

    def test_shows_count(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store):
            result = consultar_condicoes(PATIENT_ID)
        assert "3)" in result

    def test_empty_for_unknown_patient(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store):
            result = consultar_condicoes("nonexistent")
        assert "Nenhuma condição" in result


# ------------------------------------------------------------------
# MCP Tools — consultar_medicamentos
# ------------------------------------------------------------------


class TestConsultarMedicamentos:
    """Verify the consultar_medicamentos tool."""

    def test_lists_medications(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store):
            result = consultar_medicamentos(PATIENT_ID)
        assert "Losartana 50mg" in result
        assert "Hidroclorotiazida 25mg" in result
        assert "Metformina 850mg" in result

    def test_includes_dosage(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store):
            result = consultar_medicamentos(PATIENT_ID)
        assert "comprimido" in result

    def test_empty_for_unknown_patient(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store):
            result = consultar_medicamentos("nonexistent")
        assert "Nenhum medicamento" in result


# ------------------------------------------------------------------
# MCP Tools — consultar_sinais_vitais
# ------------------------------------------------------------------


class TestConsultarSinaisVitais:
    """Verify the consultar_sinais_vitais tool."""

    def test_lists_vitals(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store):
            result = consultar_sinais_vitais(PATIENT_ID)
        assert "Pressão arterial" in result
        assert "Frequência cardíaca" in result
        assert "Peso" in result
        assert "Altura" in result

    def test_bp_has_components(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store):
            result = consultar_sinais_vitais(PATIENT_ID)
        assert "150" in result
        assert "95" in result

    def test_simple_vitals_have_values(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store):
            result = consultar_sinais_vitais(PATIENT_ID)
        assert "88" in result  # HR
        assert "92" in result  # Weight
        assert "172" in result  # Height

    def test_empty_for_unknown_patient(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store):
            result = consultar_sinais_vitais("nonexistent")
        assert "Nenhum sinal vital" in result


# ------------------------------------------------------------------
# MCP Tools — verificar_interacao_medicamentosa
# ------------------------------------------------------------------


class TestVerificarInteracao:
    """Verify the verificar_interacao_medicamentosa tool."""

    def test_known_interaction(self):
        result = verificar_interacao_medicamentosa("losartana", "espironolactona")
        assert "hipercalemia" in result.lower()

    def test_case_insensitive(self):
        result = verificar_interacao_medicamentosa("Losartana", "ESPIRONOLACTONA")
        assert "hipercalemia" in result.lower()

    def test_order_independent(self):
        r1 = verificar_interacao_medicamentosa("losartana", "ibuprofeno")
        r2 = verificar_interacao_medicamentosa("ibuprofeno", "losartana")
        # Both find the same interaction, only drug name order in the message differs
        assert "AINEs" in r1
        assert "AINEs" in r2

    def test_no_interaction(self):
        result = verificar_interacao_medicamentosa("losartana", "paracetamol")
        assert "Nenhuma interação" in result

    def test_metformin_contrast(self):
        result = verificar_interacao_medicamentosa("metformina", "contraste iodado")
        assert "acidose lática" in result.lower()


# ------------------------------------------------------------------
# Normalize helper
# ------------------------------------------------------------------


class TestNormalizeDrugName:
    """Verify _normalize_drug_name."""

    def test_strips_whitespace(self):
        assert _normalize_drug_name("  losartana  ") == "losartana"

    def test_lowercases(self):
        assert _normalize_drug_name("Losartana") == "losartana"


# ------------------------------------------------------------------
# Formatting helpers — new resource types (Phase 6)
# ------------------------------------------------------------------


class TestFormatProcedure:
    """Verify _format_procedure output."""

    def test_includes_text_and_date(self):
        proc = {
            "code": {"text": "Ecocardiograma transtorácico"},
            "status": "completed",
            "performedDateTime": "2023-11-06T14:00:00-03:00",
        }
        result = _format_procedure(proc)
        assert "Ecocardiograma transtorácico" in result
        assert "completed" in result
        assert "2023-11-06" in result

    def test_falls_back_to_display(self):
        proc = {"code": {"coding": [{"display": "ECG"}]}, "status": "completed"}
        result = _format_procedure(proc)
        assert "ECG" in result

    def test_performed_period_fallback(self):
        proc = {
            "code": {"text": "Cirurgia"},
            "performedPeriod": {"start": "2024-01-01", "end": "2024-01-02"},
        }
        result = _format_procedure(proc)
        assert "2024-01-01" in result

    def test_unknown_when_no_code(self):
        result = _format_procedure({"code": {}})
        assert "Desconhecido" in result


class TestFormatDiagnosticReport:
    """Verify _format_diagnostic_report output."""

    def test_includes_text_and_conclusion(self):
        report = {
            "code": {"text": "Hemograma completo"},
            "effectiveDateTime": "2025-01-10",
            "conclusion": "Dentro da normalidade.",
        }
        result = _format_diagnostic_report(report)
        assert "Hemograma completo" in result
        assert "2025-01-10" in result
        assert "Dentro da normalidade." in result

    def test_falls_back_to_display(self):
        report = {"code": {"coding": [{"display": "CBC"}]}}
        result = _format_diagnostic_report(report)
        assert "CBC" in result

    def test_no_conclusion(self):
        report = {"code": {"text": "Exame"}, "effectiveDateTime": "2025-01-01"}
        result = _format_diagnostic_report(report)
        assert "Conclusão" not in result

    def test_unknown_when_no_code(self):
        result = _format_diagnostic_report({"code": {}})
        assert "Desconhecido" in result


class TestFormatEncounter:
    """Verify _format_encounter output."""

    def test_includes_type_and_dates(self):
        enc = {
            "type": [{"text": "Consulta de rotina"}],
            "class": {"code": "AMB", "display": "ambulatory"},
            "period": {"start": "2025-01-10T09:00:00", "end": "2025-01-10T10:30:00"},
            "reasonCode": [{"text": "Acompanhamento"}],
        }
        result = _format_encounter(enc)
        assert "Consulta de rotina" in result
        assert "ambulatory" in result
        assert "2025-01-10" in result
        assert "Acompanhamento" in result

    def test_date_range_different_days(self):
        enc = {
            "type": [{"text": "Internação"}],
            "class": {"code": "IMP"},
            "period": {"start": "2023-11-05", "end": "2023-11-12"},
        }
        result = _format_encounter(enc)
        assert "2023-11-05" in result
        assert "2023-11-12" in result

    def test_same_day_no_range(self):
        enc = {
            "type": [{"text": "Consulta"}],
            "class": {"code": "AMB"},
            "period": {"start": "2025-01-10", "end": "2025-01-10"},
        }
        result = _format_encounter(enc)
        assert "2025-01-10" in result
        assert " a " not in result

    def test_no_type_defaults(self):
        enc = {"class": {"code": "AMB"}, "period": {"start": "2025-01-10"}}
        result = _format_encounter(enc)
        assert "Encontro" in result

    def test_falls_back_to_coding_display(self):
        enc = {
            "type": [{"coding": [{"display": "Check up"}]}],
            "class": {"code": "AMB"},
            "period": {"start": "2025-01-10"},
        }
        result = _format_encounter(enc)
        assert "Check up" in result


class TestFormatImmunization:
    """Verify _format_immunization output."""

    def test_includes_vaccine_and_date(self):
        imm = {
            "vaccineCode": {"text": "COVID-19 (Coronavac)"},
            "status": "completed",
            "occurrenceDateTime": "2021-04-15",
        }
        result = _format_immunization(imm)
        assert "COVID-19 (Coronavac)" in result
        assert "completed" in result
        assert "2021-04-15" in result

    def test_falls_back_to_display(self):
        imm = {
            "vaccineCode": {"coding": [{"display": "Influenza"}]},
            "status": "completed",
        }
        result = _format_immunization(imm)
        assert "Influenza" in result

    def test_unknown_when_no_code(self):
        result = _format_immunization({"vaccineCode": {}})
        assert "Desconhecido" in result


# ------------------------------------------------------------------
# MCP Tools — consultar_procedimentos
# ------------------------------------------------------------------


class TestConsultarProcedimentos:
    """Verify the consultar_procedimentos tool."""

    def test_lists_procedures(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store):
            result = consultar_procedimentos(PATIENT_ID)
        assert "Ecocardiograma transtorácico" in result
        assert "Eletrocardiograma" in result

    def test_shows_count(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store):
            result = consultar_procedimentos(PATIENT_ID)
        assert "2)" in result

    def test_empty_for_unknown_patient(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store):
            result = consultar_procedimentos("nonexistent")
        assert "Nenhum procedimento" in result


# ------------------------------------------------------------------
# MCP Tools — consultar_exames
# ------------------------------------------------------------------


class TestConsultarExames:
    """Verify the consultar_exames tool."""

    def test_lists_reports(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store):
            result = consultar_exames(PATIENT_ID)
        assert "Hemograma completo" in result
        assert "Hemoglobina glicada" in result

    def test_includes_conclusion(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store):
            result = consultar_exames(PATIENT_ID)
        assert "HbA1c: 7.8%" in result

    def test_shows_count(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store):
            result = consultar_exames(PATIENT_ID)
        assert "2)" in result

    def test_empty_for_unknown_patient(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store):
            result = consultar_exames("nonexistent")
        assert "Nenhum exame" in result


# ------------------------------------------------------------------
# MCP Tools — consultar_encontros
# ------------------------------------------------------------------


class TestConsultarEncontros:
    """Verify the consultar_encontros tool."""

    def test_lists_encounters(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store):
            result = consultar_encontros(PATIENT_ID)
        assert "Consulta de rotina" in result
        assert "Internação hospitalar" in result

    def test_includes_reason(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store):
            result = consultar_encontros(PATIENT_ID)
        assert "Acompanhamento de hipertensão e diabetes" in result

    def test_shows_count(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store):
            result = consultar_encontros(PATIENT_ID)
        assert "2)" in result

    def test_empty_for_unknown_patient(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store):
            result = consultar_encontros("nonexistent")
        assert "Nenhum encontro" in result


# ------------------------------------------------------------------
# MCP Tools — consultar_imunizacoes
# ------------------------------------------------------------------


class TestConsultarImunizacoes:
    """Verify the consultar_imunizacoes tool."""

    def test_lists_immunizations(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store):
            result = consultar_imunizacoes(PATIENT_ID)
        assert "COVID-19 (Coronavac)" in result
        assert "Influenza sazonal" in result

    def test_shows_count(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store):
            result = consultar_imunizacoes(PATIENT_ID)
        assert "2)" in result

    def test_empty_for_unknown_patient(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store):
            result = consultar_imunizacoes("nonexistent")
        assert "Nenhuma imunização" in result


# ------------------------------------------------------------------
# _format_allergy
# ------------------------------------------------------------------


class TestFormatAllergy:
    """Verify _format_allergy output."""

    def test_formats_allergy_with_all_fields(self):
        allergy = {
            "code": {"text": "Penicilina"},
            "clinicalStatus": {"coding": [{"code": "active"}]},
            "category": ["medication"],
            "criticality": "high",
        }
        result = _format_allergy(allergy)
        assert "Penicilina" in result
        assert "active" in result
        assert "medication" in result
        assert "high" in result

    def test_formats_allergy_minimal(self):
        allergy = {"code": {"text": "Látex"}}
        result = _format_allergy(allergy)
        assert "Látex" in result

    def test_uses_coding_display_as_fallback(self):
        allergy = {
            "code": {"coding": [{"display": "Penicillin"}]},
        }
        result = _format_allergy(allergy)
        assert "Penicillin" in result

    def test_unknown_when_no_code(self):
        allergy = {"code": {}}
        result = _format_allergy(allergy)
        assert "Desconhecido" in result


# ------------------------------------------------------------------
# consultar_alergias
# ------------------------------------------------------------------


class TestConsultarAlergias:
    """Verify the consultar_alergias tool."""

    def test_lists_allergies(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store):
            result = consultar_alergias(PATIENT_ID)
        assert "Penicilina" in result
        assert "Lactose" in result

    def test_shows_count(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store):
            result = consultar_alergias(PATIENT_ID)
        assert "2)" in result

    def test_shows_criticality(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store):
            result = consultar_alergias(PATIENT_ID)
        assert "high" in result

    def test_empty_for_unknown_patient(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store):
            result = consultar_alergias("nonexistent")
        assert "Nenhuma alergia" in result
