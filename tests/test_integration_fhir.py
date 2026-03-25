"""Integration tests for MCP tools against real FHIR Bundle data."""

from __future__ import annotations

import re
from unittest.mock import patch

import pytest

from aegis.fhir import FHIRStore
from aegis.mcp_server import (
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

PATIENT_ID = "patient-joao-001"

# All query tools that accept a patient_id
_PATIENT_TOOLS = [
    consultar_paciente,
    consultar_condicoes,
    consultar_medicamentos,
    consultar_sinais_vitais,
    consultar_procedimentos,
    consultar_exames,
    consultar_encontros,
    consultar_imunizacoes,
]


class TestMCPFHIRIntegration:
    """MCP tool tests backed by the real sample_patient_joao.json bundle."""

    @pytest.fixture(autouse=True)
    def _patch_store(self, fhir_store: FHIRStore) -> None:
        with patch("aegis.fhir.get_store", return_value=fhir_store):
            yield

    def test_listar_pacientes(self) -> None:
        result = listar_pacientes()
        assert "João" in result
        assert "ID:" in result

    def test_consultar_paciente_joao(self) -> None:
        result = consultar_paciente(PATIENT_ID)
        assert "João Carlos Silva" in result
        assert "Masculino" in result

    def test_consultar_condicoes_joao(self) -> None:
        result = consultar_condicoes(PATIENT_ID)
        assert re.search(r"[Hh]ipertensão", result)

    def test_consultar_medicamentos_joao(self) -> None:
        result = consultar_medicamentos(PATIENT_ID)
        assert re.search(r"[Ll]osartana", result)

    def test_consultar_sinais_vitais_joao(self) -> None:
        result = consultar_sinais_vitais(PATIENT_ID)
        assert re.search(r"\d", result), "Expected numeric values in vital signs"
        assert re.search(r"(mmHg|bpm|kg)", result), "Expected a unit like mmHg, bpm, or kg"

    def test_consultar_procedimentos_joao(self) -> None:
        result = consultar_procedimentos(PATIENT_ID)
        assert len(result.strip()) > 0

    def test_consultar_exames_joao(self) -> None:
        result = consultar_exames(PATIENT_ID)
        assert len(result.strip()) > 0

    def test_consultar_encontros_joao(self) -> None:
        result = consultar_encontros(PATIENT_ID)
        assert len(result.strip()) > 0

    def test_consultar_imunizacoes_joao(self) -> None:
        result = consultar_imunizacoes(PATIENT_ID)
        assert len(result.strip()) > 0

    def test_verificar_interacao_losartana_ibuprofeno(self) -> None:
        result = verificar_interacao_medicamentosa("losartana", "ibuprofeno")
        assert "AINEs" in result

    def test_consultar_paciente_not_found(self) -> None:
        result = consultar_paciente("nonexistent")
        assert "não encontrado" in result

    def test_all_tools_no_exceptions(self) -> None:
        for tool_fn in _PATIENT_TOOLS:
            result = tool_fn(PATIENT_ID)
            assert isinstance(result, str)
