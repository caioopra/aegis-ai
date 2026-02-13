"""Unit tests for aegis.mcp_server — MCP tool functions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from aegis.mcp_server import (
    _format_condition,
    _format_medication,
    _format_observation,
    _format_patient,
    _normalize_drug_name,
    consultar_condicoes,
    consultar_medicamentos,
    consultar_paciente,
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
    """Return a patch that replaces the module-level _store."""
    return patch("aegis.mcp_server._store", store)


def _patch_load():
    """Return a patch that disables _load_store (data already loaded)."""
    return patch("aegis.mcp_server._load_store")


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
        with _patch_store(loaded_store), _patch_load():
            result = listar_pacientes()
        assert "João Carlos Silva" in result
        assert PATIENT_ID in result

    def test_shows_count(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store), _patch_load():
            result = listar_pacientes()
        assert "1)" in result

    def test_empty_store(self):
        with _patch_store(FHIRStore()), _patch_load():
            result = listar_pacientes()
        assert "Nenhum paciente" in result


# ------------------------------------------------------------------
# MCP Tools — consultar_paciente
# ------------------------------------------------------------------


class TestConsultarPaciente:
    """Verify the consultar_paciente tool."""

    def test_returns_demographics(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store), _patch_load():
            result = consultar_paciente(PATIENT_ID)
        assert "João Carlos Silva" in result
        assert "Masculino" in result
        assert "1960-03-15" in result

    def test_not_found(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store), _patch_load():
            result = consultar_paciente("nonexistent")
        assert "não encontrado" in result


# ------------------------------------------------------------------
# MCP Tools — consultar_condicoes
# ------------------------------------------------------------------


class TestConsultarCondicoes:
    """Verify the consultar_condicoes tool."""

    def test_lists_conditions(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store), _patch_load():
            result = consultar_condicoes(PATIENT_ID)
        assert "Hipertensão arterial sistêmica" in result
        assert "Diabetes mellitus tipo 2" in result
        assert "Insuficiência cardíaca congestiva" in result

    def test_shows_count(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store), _patch_load():
            result = consultar_condicoes(PATIENT_ID)
        assert "3)" in result

    def test_empty_for_unknown_patient(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store), _patch_load():
            result = consultar_condicoes("nonexistent")
        assert "Nenhuma condição" in result


# ------------------------------------------------------------------
# MCP Tools — consultar_medicamentos
# ------------------------------------------------------------------


class TestConsultarMedicamentos:
    """Verify the consultar_medicamentos tool."""

    def test_lists_medications(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store), _patch_load():
            result = consultar_medicamentos(PATIENT_ID)
        assert "Losartana 50mg" in result
        assert "Hidroclorotiazida 25mg" in result
        assert "Metformina 850mg" in result

    def test_includes_dosage(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store), _patch_load():
            result = consultar_medicamentos(PATIENT_ID)
        assert "comprimido" in result

    def test_empty_for_unknown_patient(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store), _patch_load():
            result = consultar_medicamentos("nonexistent")
        assert "Nenhum medicamento" in result


# ------------------------------------------------------------------
# MCP Tools — consultar_sinais_vitais
# ------------------------------------------------------------------


class TestConsultarSinaisVitais:
    """Verify the consultar_sinais_vitais tool."""

    def test_lists_vitals(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store), _patch_load():
            result = consultar_sinais_vitais(PATIENT_ID)
        assert "Pressão arterial" in result
        assert "Frequência cardíaca" in result
        assert "Peso" in result
        assert "Altura" in result

    def test_bp_has_components(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store), _patch_load():
            result = consultar_sinais_vitais(PATIENT_ID)
        assert "150" in result
        assert "95" in result

    def test_simple_vitals_have_values(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store), _patch_load():
            result = consultar_sinais_vitais(PATIENT_ID)
        assert "88" in result  # HR
        assert "92" in result  # Weight
        assert "172" in result  # Height

    def test_empty_for_unknown_patient(self, loaded_store: FHIRStore):
        with _patch_store(loaded_store), _patch_load():
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
