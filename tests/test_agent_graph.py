"""Unit tests for aegis.agent.graph — LangGraph wiring and execution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from aegis.agent.graph import (
    MAX_RETRIES,
    MIN_ACCEPTABLE_SCORE,
    _route_after_evaluation,
    _route_retrieval,
    build_graph,
)
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
    """Verify the fan-out conditional edge function."""

    def test_fan_out_when_retrieval_needed(self):
        state = {"needs_retrieval": True}
        result = _route_retrieval(state)
        assert set(result) == {"retrieve_guidelines", "fetch_patient_data"}

    def test_fetch_only_when_not_needed(self):
        state = {"needs_retrieval": False}
        assert _route_retrieval(state) == ["fetch_patient_data"]

    def test_fetch_only_when_key_missing(self):
        state = {}
        assert _route_retrieval(state) == ["fetch_patient_data"]


class TestRouteAfterEvaluation:
    """Verify the retry/finish conditional edge."""

    def test_retries_when_score_low_and_retries_available(self):
        state = {
            "evaluation": {"overall": {"score": 2, "feedback": "poor"}},
            "retry_count": 0,
        }
        assert _route_after_evaluation(state) == "increment_retry"

    def test_finishes_when_score_acceptable(self):
        state = {
            "evaluation": {"overall": {"score": 4, "feedback": "good"}},
            "retry_count": 0,
        }
        assert _route_after_evaluation(state) == "__end__"

    def test_finishes_when_score_equals_threshold(self):
        state = {
            "evaluation": {"overall": {"score": MIN_ACCEPTABLE_SCORE, "feedback": "ok"}},
            "retry_count": 0,
        }
        assert _route_after_evaluation(state) == "__end__"

    def test_finishes_when_max_retries_reached(self):
        state = {
            "evaluation": {"overall": {"score": 1, "feedback": "terrible"}},
            "retry_count": MAX_RETRIES,
        }
        assert _route_after_evaluation(state) == "__end__"

    def test_finishes_when_no_evaluation(self):
        state = {"retry_count": 0}
        assert _route_after_evaluation(state) == "__end__"

    def test_finishes_when_overall_not_dict(self):
        state = {
            "evaluation": {"overall": "some string"},
            "retry_count": 0,
        }
        assert _route_after_evaluation(state) == "__end__"


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
            "increment_retry",
        }
        assert expected == node_names

    def test_retrieve_and_fetch_both_lead_to_generate(self):
        """Both parallel branches should converge into generate_report."""
        graph = build_graph()
        graph_data = graph.get_graph()
        # Check that both retrieve_guidelines and fetch_patient_data
        # have edges leading to generate_report
        edges = [(e.source, e.target) for e in graph_data.edges]
        assert ("retrieve_guidelines", "generate_report") in edges
        assert ("fetch_patient_data", "generate_report") in edges


# ------------------------------------------------------------------
# End-to-end with mocks (no LLM, no Qdrant)
# ------------------------------------------------------------------


class TestGraphExecution:
    """Verify the graph runs end-to-end with fully mocked nodes."""

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
            patch("aegis.fhir.get_store", return_value=loaded_store),
        ):
            graph = build_graph()
            result = graph.invoke({"patient_note": "Paciente João, 65a, dispneia"})

        assert result["patient_id"] == PATIENT_ID
        assert result["needs_retrieval"] is True
        assert "guidelines" in result
        assert "patient_data" in result
        assert "report" in result
        assert "evaluation" in result
        assert result.get("retry_count", 0) == 0
        # Parallel execution: both branches ran
        assert "tools_called" in result
        assert len(result["tools_called"]) >= 4  # at least 4 base tools

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
                "aegis.agent.nodes.llm_generate_report",
                return_value={"findings": [], "plan": []},
            ),
            patch(
                "aegis.agent.nodes.llm_evaluate_report",
                return_value={"overall": {"score": 3, "feedback": "OK"}},
            ),
            patch("aegis.fhir.get_store", return_value=loaded_store),
        ):
            graph = build_graph()
            result = graph.invoke({"patient_note": "Retorno de rotina, sem queixas"})

        assert result["needs_retrieval"] is False
        # guidelines should not be set (retrieval was skipped)
        assert result.get("guidelines") is None
        assert "report" in result

    def test_retry_loop_triggers_on_low_score(self, loaded_store: FHIRStore):
        call_count = {"generate": 0, "evaluate": 0}

        def mock_generate(note, patient_data="", guidelines="", refinement_context=""):
            call_count["generate"] += 1
            return {"findings": [f"attempt {call_count['generate']}"], "plan": ["BCC"]}

        def mock_evaluate(report, note="", patient_data=""):
            call_count["evaluate"] += 1
            # First evaluation: low score → triggers retry
            # Second evaluation: acceptable score → finishes
            score = 2 if call_count["evaluate"] == 1 else 4
            return {"overall": {"score": score, "feedback": f"score {score}"}}

        with (
            patch(
                "aegis.agent.nodes.extract_entities",
                side_effect=lambda n: {"entities": [{"text": "João", "type": "patient"}]},
            ),
            patch(
                "aegis.agent.nodes.llm_decide_retrieval",
                return_value={"needs_retrieval": False, "queries": []},
            ),
            patch("aegis.agent.nodes.llm_generate_report", side_effect=mock_generate),
            patch("aegis.agent.nodes.llm_evaluate_report", side_effect=mock_evaluate),
            patch("aegis.fhir.get_store", return_value=loaded_store),
        ):
            graph = build_graph()
            result = graph.invoke({"patient_note": "Paciente João, 65a, HAS"})

        assert call_count["generate"] == 2, "Should have generated twice (original + retry)"
        assert call_count["evaluate"] == 2, "Should have evaluated twice"
        assert result["retry_count"] == 1
        assert result["evaluation"]["overall"]["score"] == 4

    def test_retry_loop_caps_at_max_retries(self, loaded_store: FHIRStore):
        call_count = {"generate": 0}

        def mock_generate(note, patient_data="", guidelines="", refinement_context=""):
            call_count["generate"] += 1
            return {"findings": [], "plan": []}

        with (
            patch(
                "aegis.agent.nodes.extract_entities",
                side_effect=lambda n: {"entities": [{"text": "João", "type": "patient"}]},
            ),
            patch(
                "aegis.agent.nodes.llm_decide_retrieval",
                return_value={"needs_retrieval": False, "queries": []},
            ),
            patch("aegis.agent.nodes.llm_generate_report", side_effect=mock_generate),
            patch(
                "aegis.agent.nodes.llm_evaluate_report",
                return_value={"overall": {"score": 1, "feedback": "always bad"}},
            ),
            patch("aegis.fhir.get_store", return_value=loaded_store),
        ):
            graph = build_graph()
            result = graph.invoke({"patient_note": "Paciente João, 65a, HAS"})

        # 1 original + MAX_RETRIES retries
        assert call_count["generate"] == 1 + MAX_RETRIES
        assert result["retry_count"] == MAX_RETRIES
        assert result["evaluation"]["overall"]["score"] == 1

    def test_parallel_execution_both_branches_complete(self, loaded_store: FHIRStore):
        """When retrieval is needed, both retrieve_guidelines and fetch_patient_data run."""
        with (
            patch(
                "aegis.agent.nodes.extract_entities",
                side_effect=lambda n: {
                    "entities": [
                        {"text": "João", "type": "patient"},
                        {"text": "HAS", "type": "condition"},
                    ]
                },
            ),
            patch(
                "aegis.agent.nodes.llm_decide_retrieval",
                return_value={"needs_retrieval": True, "queries": ["HAS tratamento"]},
            ),
            patch(
                "aegis.agent.nodes.retrieve",
                return_value=[
                    {"text": "Diretriz HAS", "source": "has.txt", "chunk_index": 0, "score": 0.85}
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
            patch("aegis.fhir.get_store", return_value=loaded_store),
        ):
            graph = build_graph()
            result = graph.invoke(
                {"patient_note": "Paciente João, 65a, HAS descompensada, PA 170x100"}
            )

        # Both branches should have produced output
        assert result.get("guidelines") is not None
        assert "Diretriz HAS" in result["guidelines"]
        assert result.get("patient_data") is not None
        assert "João Carlos Silva" in result["patient_data"]
        assert result["retrieval_confidence"] == 0.85


class TestGraphWithErrors:
    """Verify the graph handles node failures gracefully."""

    def test_survives_entity_extraction_failure(self, loaded_store: FHIRStore):
        with (
            patch("aegis.agent.nodes.extract_entities", side_effect=Exception("LLM down")),
            patch(
                "aegis.agent.nodes.llm_decide_retrieval",
                return_value={"needs_retrieval": False, "queries": []},
            ),
            patch(
                "aegis.agent.nodes.llm_generate_report",
                return_value={"findings": [], "plan": []},
            ),
            patch(
                "aegis.agent.nodes.llm_evaluate_report",
                return_value={"overall": {"score": 3, "feedback": "OK"}},
            ),
            patch("aegis.fhir.get_store", return_value=loaded_store),
        ):
            graph = build_graph()
            result = graph.invoke({"patient_note": "Paciente com dor torácica"})

        # Pipeline should complete despite entity extraction failure
        assert "report" in result
        assert "evaluation" in result
        assert len(result.get("warnings", [])) > 0


# ------------------------------------------------------------------
# Runner module (run_pipeline / stream_pipeline)
# ------------------------------------------------------------------


class TestRunnerModule:
    """Verify the programmatic runner entry points."""

    def test_run_pipeline_returns_state(self, loaded_store: FHIRStore):
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
                    {
                        "text": "Tratar HAS",
                        "source": "has.txt",
                        "chunk_index": 0,
                        "score": 0.9,
                    }
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
            patch("aegis.fhir.get_store", return_value=loaded_store),
        ):
            from aegis.agent.runner import run_pipeline

            result = run_pipeline("Paciente João, 65a, HAS")

        assert "report" in result
        assert "evaluation" in result
        assert result["report"]["findings"] == ["HAS"]
        assert result["report"]["plan"] == ["BCC"]
        # generate_report node attaches a mandatory AI disclaimer
        assert "disclaimer" in result["report"]
        assert result["evaluation"]["overall"]["score"] == 4

    def test_stream_pipeline_yields_steps(self, loaded_store: FHIRStore):
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
                "aegis.agent.nodes.llm_generate_report",
                return_value={"findings": [], "plan": []},
            ),
            patch(
                "aegis.agent.nodes.llm_evaluate_report",
                return_value={"overall": {"score": 3, "feedback": "OK"}},
            ),
            patch("aegis.fhir.get_store", return_value=loaded_store),
        ):
            from aegis.agent.runner import stream_pipeline

            steps = list(stream_pipeline("Paciente João, 65a, retorno"))

        # Each step is a (node_name, output_dict, elapsed_float) tuple
        assert len(steps) >= 4  # at least parse, decide, fetch, generate, evaluate
        node_names = [name for name, _output, _elapsed in steps]
        assert "parse_note" in node_names
        assert "generate_report" in node_names
        assert "evaluate_report" in node_names
        for _name, _output, elapsed in steps:
            assert isinstance(elapsed, float)
            assert elapsed >= 0.0
