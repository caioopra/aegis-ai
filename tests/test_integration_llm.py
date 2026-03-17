"""Integration tests for the LLM layer — requires a live Ollama server."""

from __future__ import annotations

import time
import warnings

import pytest

from aegis.llm import decide_retrieval, evaluate_report, extract_entities, generate_report
from tests.conftest import is_ollama_available, timed

# ---------------------------------------------------------------------------
# Clinical notes (pt-BR)
# ---------------------------------------------------------------------------

NOTE_HAS = (
    "Paciente João, 65 anos, masculino. Queixa de cefaleia occipital há 3 dias. "
    "PA 170x100 mmHg no consultório. Uso regular de losartana 50mg e HCTZ 25mg. "
    "Nega dor torácica, dispneia ou edema."
)
NOTE_DM2 = (
    "Maria, 58 anos, feminina. Retorno para controle de DM2. "
    "Glicemia de jejum 180 mg/dL, HbA1c 8.5%. Metformina 850mg 2x/dia. "
    "Queixa de polidipsia e poliúria há 2 semanas."
)
NOTE_ROTINA = (
    "Retorno de rotina, paciente estável, sem queixas novas. "
    "PA 130x80. Exames laboratoriais dentro da normalidade. Manter conduta."
)


@pytest.mark.integration
class TestLLMIntegration:
    @pytest.fixture(autouse=True)
    def _require_ollama(self) -> None:
        if not is_ollama_available():
            pytest.skip("Ollama not available")

    # -- Entity extraction ---------------------------------------------------

    def test_extract_entities_has_note(self) -> None:
        result = extract_entities(NOTE_HAS)
        assert isinstance(result, dict)
        assert "entities" in result
        entities = result["entities"]
        assert isinstance(entities, list)
        assert len(entities) >= 2
        assert any("type" in e for e in entities)

    def test_extract_entities_dm2_note(self) -> None:
        result = extract_entities(NOTE_DM2)
        assert isinstance(result, dict)
        entities = result.get("entities", [])
        assert len(entities) > 0

        dm2_terms = {"diabetes", "glicemia", "metformina", "dm2", "hba1c"}
        # LLM may return "normalized" as a string or list — coerce to str
        parts = []
        for e in entities:
            parts.append(str(e.get("text", "")))
            norm = e.get("normalized", "")
            parts.append(str(norm) if not isinstance(norm, list) else " ".join(norm))
        texts = " ".join(parts).lower()
        if not any(t in texts for t in dm2_terms):
            warnings.warn(
                "Nenhuma entidade menciona diabetes/glicemia/metformina",
                stacklevel=1,
            )

    # -- Retrieval decision --------------------------------------------------

    def test_decide_retrieval_complex_note(self) -> None:
        entities_result = extract_entities(NOTE_HAS)
        entities = entities_result.get("entities", [])

        result = decide_retrieval(NOTE_HAS, entities)
        assert "needs_retrieval" in result
        assert isinstance(result["needs_retrieval"], bool)

        if not result["needs_retrieval"]:
            warnings.warn(
                "needs_retrieval=False para nota clínica complexa (HAS)",
                stacklevel=1,
            )

    def test_decide_retrieval_routine_note(self) -> None:
        result = decide_retrieval(NOTE_ROTINA, [])
        assert "needs_retrieval" in result
        assert isinstance(result["needs_retrieval"], bool)

    # -- Report generation ---------------------------------------------------

    def test_generate_report(self) -> None:
        result = generate_report(
            note=NOTE_HAS,
            patient_data="João, 65a, HAS + DM2, PA 170x100",
            guidelines="Diretriz HAS: meta PA < 130x80 para diabéticos",
        )
        assert isinstance(result, dict)
        expected_keys = {"patient_summary", "findings", "assessment", "plan"}
        present = expected_keys & set(result.keys())
        assert len(present) >= 2, f"Only found keys: {present}"

    # -- Report evaluation ---------------------------------------------------

    def test_evaluate_report(self) -> None:
        report = {
            "patient_summary": "João, 65a",
            "findings": ["HAS descompensada"],
            "assessment": "Ajuste terapêutico",
            "plan": ["Aumentar losartana"],
        }
        result = evaluate_report(report)
        assert isinstance(result, dict)
        assert "overall" in result

        overall = result["overall"]
        if isinstance(overall, dict) and "score" in overall:
            score = overall["score"]
            # LLM may return score as int, float, or string — coerce to number
            if isinstance(score, str):
                score = float(score)
            assert 1 <= score <= 5

    # -- Timing --------------------------------------------------------------

    def test_entity_extraction_timing(self) -> None:
        with timed("extract_entities"):
            start = time.perf_counter()
            extract_entities(NOTE_HAS)
            elapsed = time.perf_counter() - start
        assert elapsed < 60, f"extract_entities took {elapsed:.1f}s (limit 60s)"

    def test_report_generation_timing(self) -> None:
        with timed("generate_report"):
            start = time.perf_counter()
            generate_report(
                note=NOTE_HAS,
                patient_data="João, 65a, HAS + DM2, PA 170x100",
                guidelines="Diretriz HAS: meta PA < 130x80 para diabéticos",
            )
            elapsed = time.perf_counter() - start
        assert elapsed < 60, f"generate_report took {elapsed:.1f}s (limit 60s)"
