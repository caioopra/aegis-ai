"""Unit tests for aegis.agent.graph — LangGraph wiring and execution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from aegis.agent.graph import _route_retrieval, build_graph
from aegis.fhir import FHIRStore

SAMPLE_FILE = Path("data/synthea/sample_patient_joao.json")
PATIENT_ID = "patient-joao-001"


@pytest.fixture
def loaded_store() -> FHIRStore:
    store = FHIRStore()
    store.load_bundle(SAMPLE_FILE)
    return store


# ------------------------------------------------------------------
# Routing logic
# ------------------------------------------------------------------


class TestRouteRetrieval:
    """Verify the conditional edge function."""

    def test_routes_to_retrieve_when_needed(self):
        state = {"needs_retrieval": True}
        assert _route_retrieval(state) == "retrieve_guidelines"

    def test_routes_to_fetch_when_not_needed(self):
        state = {"needs_retrieval": False}
        assert _route_retrieval(state) == "fetch_patient_data"

    def test_routes_to_fetch_when_key_missing(self):
        state = {}
        assert _route_retrieval(state) == "fetch_patient_data"


# ------------------------------------------------------------------
# Graph structure
# ------------------------------------------------------------------


class TestGraphStructure:
    """Verify the graph builds correctly with all nodes and edges."""

    def test_graph_builds_without_error(self):
        graph = build_graph()
        assert graph is not None

    def test_graph_has_all_nodes(self):
        graph = build_graph()
        node_names = set(graph.get_graph().nodes.keys())
        expected = {
            "__start__",
            "__end__",
            "parse_note",
            "decide_retrieval",
            "retrieve_guidelines",
            "fetch_patient_data",
            "generate_report",
            "evaluate_report",
        }
        assert expected == node_names


# ------------------------------------------------------------------
# End-to-end with mocks (no LLM, no Qdrant)
# ------------------------------------------------------------------


class TestGraphExecution:
    """Verify the graph runs end-to-end with fully mocked nodes."""

    def _mock_parse_note(self, state):
        return {
            "extracted_entities": [{"text": "dispneia", "type": "symptom"}],
            "patient_id": PATIENT_ID,
        }

    def _mock_decide_retrieval_yes(self, state):
        return {
            "needs_retrieval": True,
            "retrieval_queries": ["tratamento HAS"],
        }

    def _mock_decide_retrieval_no(self, state):
        return {
            "needs_retrieval": False,
            "retrieval_queries": [],
        }

    def _mock_retrieve_guidelines(self, state):
        return {"guidelines": "Diretriz: tratar HAS com IECA ou BRA."}

    def _mock_fetch_patient_data(self, state):
        return {"patient_data": "João, 65a, HAS + DM2 + ICC"}

    def _mock_generate_report(self, state):
        return {
            "report": {
                "patient_summary": "João, 65a",
                "findings": ["HAS descompensada"],
                "assessment": "Ajuste terapêutico necessário",
                "plan": ["Associar BCC"],
            }
        }

    def _mock_evaluate_report(self, state):
        return {"evaluation": {"overall": {"score": 4, "feedback": "Bom"}}}

    def test_full_run_with_retrieval(self, loaded_store: FHIRStore):
        with (
            patch(
                "aegis.agent.nodes.extract_entities",
                side_effect=lambda n: {"entities": [{"text": "João", "type": "patient"}]},
            ),
            patch(
                "aegis.agent.nodes.llm_decide_retrieval",
                return_value={"needs_retrieval": True, "queries": ["HAS"]},
            ),
            patch(
                "aegis.agent.nodes.retrieve",
                return_value=[
                    {"text": "Tratar HAS", "source": "has.txt", "chunk_index": 0, "score": 0.9}
                ],
            ),
            patch(
                "aegis.agent.nodes.llm_generate_report",
                return_value={"findings": ["HAS"], "plan": ["BCC"]},
            ),
            patch(
                "aegis.agent.nodes.llm_evaluate_report",
                return_value={"overall": {"score": 4, "feedback": "OK"}},
            ),
            patch("aegis.agent.nodes._store", loaded_store),
            patch("aegis.agent.nodes._ensure_store"),
            patch("aegis.mcp_server._store", loaded_store),
            patch("aegis.mcp_server._load_store"),
        ):
            graph = build_graph()
            result = graph.invoke({"patient_note": "Paciente João, 65a, dispneia"})

        assert result["patient_id"] == PATIENT_ID
        assert result["needs_retrieval"] is True
        assert "guidelines" in result
        assert "patient_data" in result
        assert "report" in result
        assert "evaluation" in result

    def test_full_run_without_retrieval(self, loaded_store: FHIRStore):
        with (
            patch(
                "aegis.agent.nodes.extract_entities",
                side_effect=lambda n: {"entities": [{"text": "João", "type": "patient"}]},
            ),
            patch(
                "aegis.agent.nodes.llm_decide_retrieval",
                return_value={"needs_retrieval": False, "queries": []},
            ),
            patch(
                "aegis.agent.nodes.llm_generate_report", return_value={"findings": [], "plan": []}
            ),
            patch(
                "aegis.agent.nodes.llm_evaluate_report",
                return_value={"overall": {"score": 3, "feedback": "OK"}},
            ),
            patch("aegis.agent.nodes._store", loaded_store),
            patch("aegis.agent.nodes._ensure_store"),
            patch("aegis.mcp_server._store", loaded_store),
            patch("aegis.mcp_server._load_store"),
        ):
            graph = build_graph()
            result = graph.invoke({"patient_note": "Retorno de rotina, sem queixas"})

        assert result["needs_retrieval"] is False
        # guidelines should not be set (retrieval was skipped)
        assert result.get("guidelines") is None
        assert "report" in result
