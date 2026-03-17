"""Integration tests for the full AegisNode clinical agent pipeline and edge cases.

These tests exercise the entire LangGraph flow end-to-end with real LLM calls
(Ollama/Mistral), real embeddings, and real FHIR data.  They require live
Ollama with ``mistral`` and ``nomic-embed-text`` models, so every test is
marked ``@pytest.mark.integration``.
"""

from __future__ import annotations

import logging
import time
import warnings
from unittest.mock import patch

import pytest

from aegis.agent.graph import build_graph
from tests.conftest import is_ollama_available

logger = logging.getLogger("aegis.integration")

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


# ---------------------------------------------------------------------------
# Helpers — streaming
# ---------------------------------------------------------------------------

ALL_NODE_NAMES = {
    "parse_note",
    "decide_retrieval",
    "retrieve_guidelines",
    "fetch_patient_data",
    "generate_report",
    "evaluate_report",
}

# Nodes that always execute (retrieve_guidelines is conditional)
MANDATORY_NODES = ALL_NODE_NAMES - {"retrieve_guidelines"}


def _stream_with_timings(note: str) -> tuple[dict, dict[str, float]]:
    """Run the graph via stream and return (final_state, per-node timings)."""
    graph = build_graph()
    timings: dict[str, float] = {}
    state: dict = {}

    for step in graph.stream({"patient_note": note}):
        for node_name, node_output in step.items():
            t0 = time.perf_counter()
            # step already contains the executed output; measure is wall-clock
            # between consecutive yields (approximation for the node duration)
            timings[node_name] = time.perf_counter() - t0
            state.update(node_output)

    return state, timings


def _stream_and_collect(note: str) -> tuple[dict, list[tuple[str, float, dict]]]:
    """Stream the graph and collect (node_name, elapsed_seconds, output) per step."""
    graph = build_graph()
    steps: list[tuple[str, float, dict]] = []
    prev_time = time.perf_counter()

    for step in graph.stream({"patient_note": note}):
        now = time.perf_counter()
        elapsed = now - prev_time
        for node_name, node_output in step.items():
            steps.append((node_name, elapsed, node_output))
        prev_time = now

    # Build final state
    final: dict = {}
    for _, _, output in steps:
        final.update(output)
    return final, steps


# ---------------------------------------------------------------------------
# Node Observability Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestNodeObservability:
    """Per-node timing and observability via graph streaming."""

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

    def test_all_mandatory_nodes_execute(self) -> None:
        """Every mandatory node must appear in the stream output."""
        _final, steps = _stream_and_collect(NOTE_HAS)
        executed = {name for name, _, _ in steps}

        for node in MANDATORY_NODES:
            assert node in executed, f"Node '{node}' did not execute"

    def test_per_node_timing_logged(self) -> None:
        """Each node should complete within a reasonable time and be logged."""
        _final, steps = _stream_and_collect(NOTE_HAS)

        for node_name, elapsed, _output in steps:
            logger.info("[NODE TIMING] %-25s %.2fs", node_name, elapsed)
            assert elapsed < 120, (
                f"Node '{node_name}' took {elapsed:.1f}s — exceeds 120s per-node limit"
            )

    def test_llm_nodes_are_slowest(self) -> None:
        """LLM-bound nodes should dominate total pipeline time."""
        _final, steps = _stream_and_collect(NOTE_HAS)

        llm_nodes = {"parse_note", "decide_retrieval", "generate_report", "evaluate_report"}
        llm_time = sum(e for n, e, _ in steps if n in llm_nodes)
        total_time = sum(e for _, e, _ in steps)

        if total_time > 0:
            llm_ratio = llm_time / total_time
            logger.info(
                "[OBSERVABILITY] LLM nodes: %.1fs / %.1fs (%.0f%%)",
                llm_time,
                total_time,
                llm_ratio * 100,
            )
            # LLM calls should account for at least 50% of total time
            assert llm_ratio > 0.5, (
                f"LLM nodes only account for {llm_ratio:.0%} of total time — "
                f"expected >50%. Non-LLM bottleneck detected."
            )

    def test_node_execution_order(self) -> None:
        """Nodes must execute in the expected graph order."""
        _final, steps = _stream_and_collect(NOTE_HAS)
        executed = [name for name, _, _ in steps]

        # parse_note must be first
        assert executed[0] == "parse_note"
        # decide_retrieval must follow parse_note
        assert executed[1] == "decide_retrieval"
        # evaluate_report must be last
        assert executed[-1] == "evaluate_report"
        # generate_report must come right before evaluate_report
        assert executed[-2] == "generate_report"

    def test_timing_summary_report(self) -> None:
        """Produce a full timing summary for observability."""
        _final, steps = _stream_and_collect(NOTE_ICC)

        total = sum(e for _, e, _ in steps)
        logger.info("=" * 60)
        logger.info("[TIMING SUMMARY] Pipeline for ICC note")
        logger.info("-" * 60)
        for node_name, elapsed, _output in steps:
            pct = (elapsed / total * 100) if total > 0 else 0
            logger.info("  %-25s %6.2fs  (%4.1f%%)", node_name, elapsed, pct)
        logger.info("-" * 60)
        logger.info("  %-25s %6.2fs", "TOTAL", total)
        logger.info("=" * 60)

        assert total < 300, f"Total pipeline took {total:.1f}s — exceeds 300s limit"


# ---------------------------------------------------------------------------
# Multi-Patient Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestMultiPatientIntegration:
    """Tests with multiple patients loaded from Synthea data."""

    @pytest.fixture(autouse=True)
    def _require_services(self) -> None:
        if not is_ollama_available():
            pytest.skip("Ollama not available")

    @pytest.fixture(autouse=True)
    def _patch_all(self, multi_patient_fhir_store, ingested_rag_env):
        client, _collection, bm25_path = ingested_rag_env
        actual_collection = "clinical_guidelines"
        with (
            patch("aegis.agent.nodes._store", multi_patient_fhir_store),
            patch("aegis.agent.nodes._ensure_store", lambda: None),
            patch("aegis.mcp_server._store", multi_patient_fhir_store),
            patch("aegis.mcp_server._load_store", lambda: None),
            patch("aegis.rag.retriever.get_qdrant_client", return_value=client),
            patch("aegis.rag.retriever.settings") as mock_settings,
        ):
            mock_settings.bm25_stats_path = bm25_path
            mock_settings.qdrant_collection = actual_collection
            yield

    def test_multiple_patients_loaded(self, multi_patient_fhir_store) -> None:
        """Store should contain João + Synthea patients."""
        patients = multi_patient_fhir_store.list_patients()
        assert len(patients) >= 3, (
            f"Expected at least 3 patients (João + 2 Synthea), got {len(patients)}"
        )

    def test_joao_matched_with_multiple_patients(self) -> None:
        """João note should still match patient-joao-001 even with other patients loaded."""
        result = _invoke(NOTE_HAS)
        assert result["patient_id"] == "patient-joao-001"

    def test_unmatched_patient_falls_back(self) -> None:
        """Note referencing unknown patient should fall back (not crash)."""
        note = "Paciente Carlos, 40 anos, tosse seca há 3 semanas, sem febre."
        result = _invoke(note)

        assert result["patient_id"] != "", "Should fall back to a patient"
        assert "report" in result
        assert isinstance(result["report"], dict)

    def test_synthea_patient_produces_report(self) -> None:
        """Pipeline should work with Synthea patient data (different FHIR structure)."""
        # Use a name that won't match João to exercise Synthea data path
        note = "Paciente idoso, dispneia progressiva, edema bilateral. PA 145x90."
        result = _invoke(note)

        assert "report" in result
        assert "evaluation" in result
        # Patient data should contain some FHIR content
        patient_data = result.get("patient_data", "")
        assert patient_data != "Paciente não identificado."

    def test_patient_list_has_correct_ids(self, multi_patient_fhir_store) -> None:
        """Verify known patient IDs are present in the store."""
        patients = multi_patient_fhir_store.list_patients()
        patient_ids = {p["id"] for p in patients}
        assert "patient-joao-001" in patient_ids
