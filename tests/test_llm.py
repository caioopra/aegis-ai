"""Unit and integration tests for aegis.llm — LLM client and prompt templates."""

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from aegis.llm import (
    ENTITY_EXTRACTION_PROMPT,
    EVALUATE_REPORT_PROMPT,
    EXPAND_NOTE_PROMPT,
    MAX_INPUT_TOKENS,
    REPORT_PROMPT,
    SELF_RAG_DECISION_PROMPT,
    SYSTEM_ENTITY_EXTRACTION,
    SYSTEM_MEDICAL,
    SYSTEM_RAG_DECISION,
    SYSTEM_REPORT_EVALUATION,
    SYSTEM_REPORT_GENERATION,
    TEMP_EVALUATION,
    TEMP_EXTRACTION,
    TEMP_RAG_DECISION,
    TEMP_REPORT,
    AcompanhamentoSection,
    ClinicalReport,
    EntityExtractionResult,
    ExtractedEntity,
    ReportEvaluation,
    RetrievalDecision,
    _extract_json,
    decide_retrieval,
    estimate_tokens,
    evaluate_report,
    expand_note,
    extract_entities,
    generate,
    generate_json,
    generate_report,
    truncate_to_budget,
)


# ── Prompt template tests ────────────────────────────────────────────────


class TestPromptTemplates:
    """Verify prompt templates have the right placeholders and structure."""

    def test_expand_note_prompt_has_note_placeholder(self):
        assert "{note}" in EXPAND_NOTE_PROMPT

    def test_expand_note_prompt_requests_json(self):
        assert "JSON" in EXPAND_NOTE_PROMPT

    def test_entity_extraction_prompt_has_note_placeholder(self):
        assert "{note}" in ENTITY_EXTRACTION_PROMPT

    def test_entity_extraction_prompt_has_example(self):
        assert "HAS" in ENTITY_EXTRACTION_PROMPT
        assert "Hipertensão arterial sistêmica" in ENTITY_EXTRACTION_PROMPT

    def test_report_prompt_has_all_placeholders(self):
        assert "{note}" in REPORT_PROMPT
        assert "{patient_data}" in REPORT_PROMPT
        assert "{guidelines}" in REPORT_PROMPT
        assert "{refinement_context}" in REPORT_PROMPT

    def test_report_prompt_instructs_on_missing_data(self):
        assert "Não disponível" in REPORT_PROMPT

    def test_self_rag_prompt_has_placeholders(self):
        assert "{note}" in SELF_RAG_DECISION_PROMPT
        assert "{entities}" in SELF_RAG_DECISION_PROMPT

    def test_evaluate_report_prompt_has_placeholders(self):
        assert "{report}" in EVALUATE_REPORT_PROMPT
        assert "{note}" in EVALUATE_REPORT_PROMPT
        assert "{patient_data}" in EVALUATE_REPORT_PROMPT

    def test_evaluate_report_prompt_has_rubric(self):
        assert "1 =" in EVALUATE_REPORT_PROMPT
        assert "5 =" in EVALUATE_REPORT_PROMPT

    def test_system_medical_is_nonempty(self):
        assert len(SYSTEM_MEDICAL) > 50

    def test_task_specific_system_prompts_in_pt_br(self):
        # All four task-specific prompts must be substantial pt-BR strings.
        for prompt in (
            SYSTEM_ENTITY_EXTRACTION,
            SYSTEM_RAG_DECISION,
            SYSTEM_REPORT_GENERATION,
            SYSTEM_REPORT_EVALUATION,
        ):
            assert len(prompt) > 50
            # pt-BR sanity check: contains "Você" or accented chars
            assert "Você" in prompt or "ê" in prompt or "á" in prompt

    def test_all_prompts_pt_br(self):
        # Each user-facing prompt template must contain pt-BR markers.
        assert "Nota" in EXPAND_NOTE_PROMPT
        assert "Retorne" in EXPAND_NOTE_PROMPT
        assert "Extraia" in ENTITY_EXTRACTION_PROMPT
        assert "Dados do Paciente" in REPORT_PROMPT
        assert "Diretrizes Relevantes" in REPORT_PROMPT
        assert "Nota Clínica" in SELF_RAG_DECISION_PROMPT
        assert "Avalie" in EVALUATE_REPORT_PROMPT

    def test_expand_note_prompt_formats_correctly(self, sample_note):
        rendered = EXPAND_NOTE_PROMPT.format(note=sample_note)
        assert sample_note in rendered
        assert "{note}" not in rendered

    def test_report_prompt_formats_correctly(self, sample_note):
        rendered = REPORT_PROMPT.format(
            note=sample_note,
            patient_data="John Doe, 65y",
            guidelines="Treat hypertension per JNC8",
            refinement_context="",
        )
        assert sample_note in rendered
        assert "John Doe" in rendered
        assert "JNC8" in rendered

    def test_evaluate_report_prompt_formats_correctly(self):
        rendered = EVALUATE_REPORT_PROMPT.format(
            report='{"findings": []}',
            note="Test note",
            patient_data="Test data",
        )
        assert "Test note" in rendered
        assert "Test data" in rendered


# ── _extract_json tests ──────────────────────────────────────────────────


class TestExtractJson:
    """Unit tests for the JSON extraction fallback helper."""

    def test_extracts_raw_json(self):
        text = '{"expanded_note": "test", "entities": []}'
        result = _extract_json(text)
        assert result == {"expanded_note": "test", "entities": []}

    def test_extracts_json_from_fenced_block(self):
        text = 'Here is the result:\n```json\n{"key": "value"}\n```\nDone.'
        result = _extract_json(text)
        assert result == {"key": "value"}

    def test_extracts_json_from_unfenced_block(self):
        text = '```\n{"key": "value"}\n```'
        result = _extract_json(text)
        assert result == {"key": "value"}

    def test_extracts_json_with_surrounding_text(self):
        text = 'The answer is: {"a": 1, "b": 2}. Hope that helps!'
        result = _extract_json(text)
        assert result == {"a": 1, "b": 2}

    def test_raises_on_no_json(self):
        with pytest.raises(ValueError, match="Could not extract JSON"):
            _extract_json("No JSON here at all.")

    def test_raises_on_invalid_json(self):
        with pytest.raises((ValueError, json.JSONDecodeError)):
            _extract_json("{invalid json!!!}")

    def test_handles_nested_json(self):
        text = '{"outer": {"inner": [1, 2, 3]}}'
        result = _extract_json(text)
        assert result["outer"]["inner"] == [1, 2, 3]

    def test_handles_multiline_json(self):
        text = """Here:
```json
{
  "expanded_note": "Patient presents with dyspnea",
  "entities": [
    {"text": "dyspnea", "type": "symptom", "original": "dispneia"}
  ]
}
```"""
        result = _extract_json(text)
        assert "expanded_note" in result
        assert len(result["entities"]) == 1

    def test_extracts_first_json_from_multiple(self):
        text = 'Result: {"a": 1} And also {"b": 2}'
        result = _extract_json(text)
        assert result == {"a": 1}


# ── generate() mocked tests ─────────────────────────────────────────────


class TestGenerateMocked:
    """Unit tests for generate() with the chat provider mocked out."""

    @patch("aegis.llm._get_chat")
    def test_generate_returns_content(self, mock_get_chat):
        mock_provider = MagicMock()
        mock_provider.chat.return_value = "Hello, doctor."
        mock_get_chat.return_value = mock_provider
        result = generate("Test prompt")
        assert result == "Hello, doctor."

    @patch("aegis.llm._get_chat")
    def test_generate_passes_system_prompt(self, mock_get_chat):
        mock_provider = MagicMock()
        mock_provider.chat.return_value = "ok"
        mock_get_chat.return_value = mock_provider
        generate("Test", system_prompt="Custom system")
        call_args = mock_provider.chat.call_args
        assert call_args.kwargs["system_prompt"] == "Custom system"

    @patch("aegis.llm._get_chat")
    def test_generate_passes_user_prompt(self, mock_get_chat):
        mock_provider = MagicMock()
        mock_provider.chat.return_value = "ok"
        mock_get_chat.return_value = mock_provider
        generate("My user prompt")
        call_args = mock_provider.chat.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "My user prompt"

    @patch("aegis.llm._get_chat")
    def test_generate_uses_low_temperature(self, mock_get_chat):
        mock_provider = MagicMock()
        mock_provider.chat.return_value = "ok"
        mock_get_chat.return_value = mock_provider
        generate("Test")
        call_args = mock_provider.chat.call_args
        assert call_args.kwargs["temperature"] == 0.3


# ── generate_json() mocked tests ────────────────────────────────────────


class TestGenerateJsonMocked:
    """Unit tests for generate_json() with the chat provider mocked."""

    @patch("aegis.llm._get_chat")
    def test_returns_parsed_json(self, mock_get_chat):
        mock_provider = MagicMock()
        payload = {"expanded_note": "test", "entities": []}
        mock_provider.chat.return_value = json.dumps(payload)
        mock_get_chat.return_value = mock_provider
        result = generate_json("Test")
        assert result == payload

    @patch("aegis.llm._get_chat")
    def test_uses_json_mode(self, mock_get_chat):
        mock_provider = MagicMock()
        mock_provider.chat.return_value = '{"key": "val"}'
        mock_get_chat.return_value = mock_provider
        generate_json("Test")
        call_args = mock_provider.chat.call_args
        assert call_args.kwargs["json_mode"] is True

    @patch("aegis.llm._get_chat")
    def test_forwards_temperature_to_chat(self, mock_get_chat):
        mock_provider = MagicMock()
        mock_provider.chat.return_value = '{"ok": true}'
        mock_get_chat.return_value = mock_provider
        generate_json("Test", temperature=0.05)
        call_args = mock_provider.chat.call_args
        assert call_args.kwargs["temperature"] == 0.05

    @patch("aegis.llm._get_chat")
    def test_forwards_system_prompt_to_chat(self, mock_get_chat):
        mock_provider = MagicMock()
        mock_provider.chat.return_value = '{"ok": true}'
        mock_get_chat.return_value = mock_provider
        generate_json("Test", system_prompt="Custom system")
        call_args = mock_provider.chat.call_args
        assert call_args.kwargs["system_prompt"] == "Custom system"

    @patch("aegis.llm.time.sleep")
    @patch("aegis.llm._get_chat")
    def test_retries_on_json_decode_error(self, mock_get_chat, mock_sleep):
        mock_provider = MagicMock()
        # First 2 calls return invalid JSON, third succeeds
        mock_provider.chat.side_effect = [
            "not json",
            "still not json",
            '{"ok": true}',
        ]
        mock_get_chat.return_value = mock_provider
        result = generate_json("Test", max_retries=3)
        assert result == {"ok": True}
        assert mock_provider.chat.call_count == 3

    @patch("aegis.llm.time.sleep")
    @patch("aegis.llm._get_chat")
    def test_falls_back_to_extract_after_all_retries(self, mock_get_chat, mock_sleep):
        mock_provider = MagicMock()
        # All retries fail with json_mode, final fallback without json_mode succeeds
        mock_provider.chat.side_effect = [
            "bad",  # retry 1
            "bad",  # retry 2
            "bad",  # retry 3
            '{"fallback": true}',  # fallback generate()
        ]
        mock_get_chat.return_value = mock_provider
        result = generate_json("Test", max_retries=3)
        assert result == {"fallback": True}

    @patch("aegis.llm.time.sleep")
    @patch("aegis.llm._get_chat")
    def test_raises_after_all_retries_and_fallback_fail(self, mock_get_chat, mock_sleep):
        mock_provider = MagicMock()
        mock_provider.chat.side_effect = [
            "bad",
            "bad",
            "bad",
            "still no json",
        ]
        mock_get_chat.return_value = mock_provider
        with pytest.raises(ValueError, match="All 3 attempts failed"):
            generate_json("Test", max_retries=3)

    @patch("aegis.llm.time.sleep")
    @patch("aegis.llm._get_chat")
    def test_retries_on_connection_error(self, mock_get_chat, mock_sleep):
        mock_provider = MagicMock()
        mock_provider.chat.side_effect = [
            ConnectionError("Provider down"),
            '{"ok": true}',
        ]
        mock_get_chat.return_value = mock_provider
        result = generate_json("Test", max_retries=2)
        assert result == {"ok": True}


# ── High-level functions mocked tests ────────────────────────────────────


class TestHighLevelFunctionsMocked:
    """Verify high-level functions call generate_json with correct prompts."""

    @patch("aegis.llm.generate_json")
    def test_expand_note_calls_generate_json(self, mock_gen, sample_note):
        mock_gen.return_value = {"expanded_note": "...", "entities": []}
        result = expand_note(sample_note)
        mock_gen.assert_called_once()
        prompt_arg = mock_gen.call_args[0][0]
        assert sample_note in prompt_arg
        assert "expanded_note" in result

    @patch("aegis.llm.generate_json")
    def test_extract_entities_calls_generate_json(self, mock_gen, sample_note):
        mock_gen.return_value = {"entities": [{"text": "dispneia", "type": "symptom"}]}
        result = extract_entities(sample_note)
        mock_gen.assert_called_once()
        assert "entities" in result

    @patch("aegis.llm.generate_json")
    def test_generate_report_calls_generate_json(self, mock_gen, sample_note):
        mock_gen.return_value = {
            "patient_summary": "...",
            "findings": [],
            "assessment": "...",
            "plan": "...",
            "guideline_references": [],
        }
        generate_report(sample_note, patient_data="age 65", guidelines="JNC8")
        mock_gen.assert_called_once()
        prompt_arg = mock_gen.call_args[0][0]
        assert sample_note in prompt_arg
        assert "age 65" in prompt_arg
        assert "JNC8" in prompt_arg

    @patch("aegis.llm.generate_json")
    def test_generate_report_defaults(self, mock_gen, sample_note):
        mock_gen.return_value = {}
        generate_report(sample_note)
        prompt_arg = mock_gen.call_args[0][0]
        assert "Não disponível" in prompt_arg

    @patch("aegis.llm.generate_json")
    def test_generate_report_with_refinement_context(self, mock_gen, sample_note):
        mock_gen.return_value = {}
        generate_report(sample_note, refinement_context="Fix completeness")
        prompt_arg = mock_gen.call_args[0][0]
        assert "Fix completeness" in prompt_arg

    @patch("aegis.llm.generate_json")
    def test_decide_retrieval_calls_generate_json(self, mock_gen, sample_note):
        entities = [{"text": "dispneia", "type": "symptom"}]
        mock_gen.return_value = {"needs_retrieval": True, "queries": ["dyspnea treatment"]}
        result = decide_retrieval(sample_note, entities)
        mock_gen.assert_called_once()
        assert result["needs_retrieval"] is True

    @patch("aegis.llm.generate_json")
    def test_evaluate_report_calls_generate_json(self, mock_gen):
        report = {"patient_summary": "test", "findings": [], "plan": "test"}
        mock_gen.return_value = {
            "completeness": {"score": 4, "feedback": "good"},
            "overall": {"score": 4, "feedback": "good"},
        }
        result = evaluate_report(report, note="Test note", patient_data="Test data")
        mock_gen.assert_called_once()
        prompt_arg = mock_gen.call_args[0][0]
        assert "Test note" in prompt_arg
        assert "Test data" in prompt_arg
        assert "completeness" in result

    @patch("aegis.llm.generate_json")
    def test_evaluate_report_defaults_missing_context(self, mock_gen):
        mock_gen.return_value = {"overall": {"score": 3, "feedback": "ok"}}
        evaluate_report({"findings": []})
        prompt_arg = mock_gen.call_args[0][0]
        assert "Não disponível" in prompt_arg

    # ── Task-specific system prompt + temperature wiring ──────────────────

    @patch("aegis.llm.generate_json")
    def test_extract_entities_uses_extraction_system_prompt(self, mock_gen, sample_note):
        mock_gen.return_value = {"entities": []}
        extract_entities(sample_note)
        kwargs = mock_gen.call_args.kwargs
        assert kwargs["system_prompt"] == SYSTEM_ENTITY_EXTRACTION
        assert kwargs["temperature"] == TEMP_EXTRACTION

    @patch("aegis.llm.generate_json")
    def test_expand_note_uses_extraction_system_prompt(self, mock_gen, sample_note):
        mock_gen.return_value = {"expanded_note": "...", "entities": []}
        expand_note(sample_note)
        kwargs = mock_gen.call_args.kwargs
        assert kwargs["system_prompt"] == SYSTEM_ENTITY_EXTRACTION
        assert kwargs["temperature"] == TEMP_EXTRACTION

    @patch("aegis.llm.generate_json")
    def test_decide_retrieval_uses_rag_system_prompt(self, mock_gen, sample_note):
        mock_gen.return_value = {"needs_retrieval": False, "queries": []}
        decide_retrieval(sample_note, [])
        kwargs = mock_gen.call_args.kwargs
        assert kwargs["system_prompt"] == SYSTEM_RAG_DECISION
        assert kwargs["temperature"] == TEMP_RAG_DECISION

    @patch("aegis.llm.generate_json")
    def test_generate_report_uses_report_system_prompt(self, mock_gen, sample_note):
        mock_gen.return_value = {}
        generate_report(sample_note)
        kwargs = mock_gen.call_args.kwargs
        assert kwargs["system_prompt"] == SYSTEM_REPORT_GENERATION
        assert kwargs["temperature"] == TEMP_REPORT

    @patch("aegis.llm.generate_json")
    def test_evaluate_report_uses_evaluation_system_prompt(self, mock_gen):
        mock_gen.return_value = {"overall": {"score": 4, "feedback": "ok"}}
        evaluate_report({"findings": []})
        kwargs = mock_gen.call_args.kwargs
        assert kwargs["system_prompt"] == SYSTEM_REPORT_EVALUATION
        assert kwargs["temperature"] == TEMP_EVALUATION


# ── Token estimation tests ───────────────────────────────────────────────


class TestTokenEstimation:
    """Verify token estimation and truncation utilities."""

    def test_estimate_tokens_basic(self):
        assert estimate_tokens("") == 0
        assert estimate_tokens("abcd") == 1
        assert estimate_tokens("a" * 400) == 100

    def test_truncate_to_budget_within_budget(self):
        text = "short text"
        result = truncate_to_budget(text, 100)
        assert result == text  # no truncation

    def test_truncate_to_budget_exceeds(self):
        text = "a" * 800  # ~200 tokens
        result = truncate_to_budget(text, 50, "teste")
        assert len(result) < 800
        assert "[teste truncado" in result

    def test_truncate_to_budget_preserves_label(self):
        text = "a" * 800
        result = truncate_to_budget(text, 50, "dados do paciente")
        assert "dados do paciente" in result

    def test_max_input_tokens_is_positive(self):
        assert MAX_INPUT_TOKENS > 0


# ── Prompt structure enrichment tests ───────────────────────────────────


class TestReportPromptEnrichment:
    """Verify REPORT_PROMPT includes the 5 new section keys and instructions."""

    def test_report_prompt_has_new_section_keys(self):
        for key in (
            "diagnosticos_diferenciais",
            "sinais_alarme",
            "acompanhamento",
            "interacoes_medicamentosas",
            "limitacoes",
        ):
            assert key in REPORT_PROMPT, f"Missing key '{key}' in REPORT_PROMPT"

    def test_report_prompt_has_all_original_section_keys(self):
        for key in ("patient_summary", "findings", "assessment", "plan", "guideline_references"):
            assert key in REPORT_PROMPT, f"Missing original key '{key}' in REPORT_PROMPT"

    def test_report_prompt_has_acompanhamento_subfields(self):
        assert "proxima_visita" in REPORT_PROMPT
        assert "exames_a_repetir" in REPORT_PROMPT
        assert "sinais_para_escalar" in REPORT_PROMPT

    def test_report_prompt_instructs_on_10_sections(self):
        # The prompt should enumerate 10 sections
        assert "10" in REPORT_PROMPT

    def test_report_prompt_renders_with_all_placeholders(self, sample_note):
        rendered = REPORT_PROMPT.format(
            note=sample_note,
            patient_data="João, 65a, HAS",
            guidelines="Diretriz HAS 2023",
            refinement_context="",
        )
        assert sample_note in rendered
        assert "João" in rendered
        assert "diagnosticos_diferenciais" in rendered
        assert "sinais_alarme" in rendered
        assert "acompanhamento" in rendered

    def test_report_prompt_is_pt_br(self):
        assert "Diretrizes" in REPORT_PROMPT
        assert "Nota Clínica" in REPORT_PROMPT
        assert "sinais de alerta" in REPORT_PROMPT or "sinais_alarme" in REPORT_PROMPT


class TestEvaluateReportPromptEnrichment:
    """Verify EVALUATE_REPORT_PROMPT includes the two new rubric dimensions."""

    def test_evaluate_prompt_has_safety_dimension(self):
        assert '"safety"' in EVALUATE_REPORT_PROMPT

    def test_evaluate_prompt_has_follow_up_quality_dimension(self):
        assert '"follow_up_quality"' in EVALUATE_REPORT_PROMPT

    def test_evaluate_prompt_has_all_original_dimensions(self):
        for dim in ("completeness", "accuracy", "guideline_adherence", "clarity", "overall"):
            assert dim in EVALUATE_REPORT_PROMPT, (
                f"Missing dimension '{dim}' in EVALUATE_REPORT_PROMPT"
            )

    def test_evaluate_prompt_mentions_seven_dimensions(self):
        assert "7" in EVALUATE_REPORT_PROMPT

    def test_evaluate_prompt_safety_explains_sinais_alarme(self):
        assert "sinais_alarme" in EVALUATE_REPORT_PROMPT

    def test_evaluate_prompt_follow_up_quality_explains_acompanhamento(self):
        assert "acompanhamento" in EVALUATE_REPORT_PROMPT

    def test_evaluate_prompt_renders_correctly(self):
        rendered = EVALUATE_REPORT_PROMPT.format(
            report='{"findings": []}',
            note="Paciente com HAS",
            patient_data="João, 65a",
        )
        assert "safety" in rendered
        assert "follow_up_quality" in rendered


# ── Pydantic model tests ─────────────────────────────────────────────────


class TestPydanticModels:
    """Happy-path and fallback tests for Pydantic output schemas."""

    # -- ExtractedEntity --------------------------------------------------

    def test_extracted_entity_valid(self):
        e = ExtractedEntity(
            text="HAS", type="condition", normalized="Hipertensão arterial sistêmica"
        )
        assert e.text == "HAS"
        assert e.type == "condition"
        assert e.normalized == "Hipertensão arterial sistêmica"

    def test_extracted_entity_normalized_optional(self):
        e = ExtractedEntity(text="dispneia", type="symptom")
        assert e.normalized is None

    def test_extracted_entity_invalid_type_raises(self):
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            ExtractedEntity(text="x", type="unknown_type")

    # -- EntityExtractionResult ------------------------------------------

    def test_entity_extraction_result_happy_path(self):
        data = {
            "entities": [
                {
                    "text": "HAS",
                    "type": "condition",
                    "normalized": "Hipertensão arterial sistêmica",
                },
                {
                    "text": "PA 150x95",
                    "type": "vital_sign",
                    "normalized": "Pressão arterial 150x95 mmHg",
                },
            ]
        }
        result = EntityExtractionResult.model_validate(data)
        assert len(result.entities) == 2
        dumped = result.model_dump()
        assert dumped["entities"][0]["text"] == "HAS"

    def test_entity_extraction_result_empty_entities(self):
        result = EntityExtractionResult.model_validate({"entities": []})
        assert result.entities == []

    # -- RetrievalDecision -----------------------------------------------

    def test_retrieval_decision_happy_path(self):
        data = {"needs_retrieval": True, "queries": ["HAS tratamento"], "reasoning": "Há condição"}
        rd = RetrievalDecision.model_validate(data)
        assert rd.needs_retrieval is True
        assert rd.queries == ["HAS tratamento"]
        assert rd.reasoning == "Há condição"

    def test_retrieval_decision_defaults(self):
        rd = RetrievalDecision.model_validate({"needs_retrieval": False})
        assert rd.queries == []
        assert rd.reasoning == ""

    # -- AcompanhamentoSection -------------------------------------------

    def test_acompanhamento_section_defaults(self):
        a = AcompanhamentoSection()
        assert a.proxima_visita == ""
        assert a.exames_a_repetir == []
        assert a.sinais_para_escalar == []

    def test_acompanhamento_section_full(self):
        a = AcompanhamentoSection(
            proxima_visita="Retorno em 30 dias",
            exames_a_repetir=["HbA1c", "creatinina"],
            sinais_para_escalar=["Dispneia súbita"],
        )
        assert a.proxima_visita == "Retorno em 30 dias"
        assert len(a.exames_a_repetir) == 2

    # -- ClinicalReport --------------------------------------------------

    def test_clinical_report_happy_path_all_10_fields(self):
        data = {
            "patient_summary": "João, 65a, HAS, DM2",
            "findings": ["PA 150x95", "edema MMII"],
            "assessment": "HAS não controlada",
            "plan": ["Ajustar losartana para 100mg"],
            "guideline_references": ["Diretriz HAS 2023 — meta < 130x80"],
            "diagnosticos_diferenciais": ["ICC descompensada — edema + dispneia"],
            "sinais_alarme": ["Piora súbita da dispneia"],
            "acompanhamento": {
                "proxima_visita": "Retorno em 30 dias",
                "exames_a_repetir": ["creatinina", "potássio"],
                "sinais_para_escalar": ["Dispneia em repouso"],
            },
            "interacoes_medicamentosas": [],
            "limitacoes": ["Sem resultado de HbA1c recente"],
        }
        report = ClinicalReport.model_validate(data)
        dumped = report.model_dump()
        assert dumped["patient_summary"] == "João, 65a, HAS, DM2"
        assert dumped["diagnosticos_diferenciais"] == ["ICC descompensada — edema + dispneia"]
        assert dumped["sinais_alarme"] == ["Piora súbita da dispneia"]
        assert dumped["acompanhamento"]["proxima_visita"] == "Retorno em 30 dias"
        assert dumped["interacoes_medicamentosas"] == []
        assert dumped["limitacoes"] == ["Sem resultado de HbA1c recente"]
        assert dumped["disclaimer"] is None

    def test_clinical_report_disclaimer_field(self):
        report = ClinicalReport.model_validate(
            {"disclaimer": "Este relatório foi gerado por IA e requer revisão médica."}
        )
        assert "IA" in report.disclaimer

    def test_clinical_report_defaults_all_fields(self):
        report = ClinicalReport.model_validate({})
        assert report.patient_summary == ""
        assert report.findings == []
        assert report.diagnosticos_diferenciais == []
        assert report.sinais_alarme == []
        assert report.interacoes_medicamentosas == []
        assert report.limitacoes == []
        assert report.disclaimer is None

    def test_clinical_report_roundtrip(self):
        data = {
            "patient_summary": "Teste",
            "findings": ["achado 1"],
            "assessment": "avaliação",
            "plan": ["ação 1"],
            "guideline_references": ["ref 1"],
            "diagnosticos_diferenciais": ["DD 1"],
            "sinais_alarme": ["alerta 1"],
            "acompanhamento": {
                "proxima_visita": "30 dias",
                "exames_a_repetir": [],
                "sinais_para_escalar": [],
            },
            "interacoes_medicamentosas": ["losartana + ibuprofeno"],
            "limitacoes": ["limitação 1"],
        }
        dumped = ClinicalReport.model_validate(data).model_dump()
        assert dumped["patient_summary"] == "Teste"
        assert dumped["interacoes_medicamentosas"] == ["losartana + ibuprofeno"]

    # -- ReportEvaluation (nested DimensionScore shape) -------------------

    def test_report_evaluation_happy_path_7_dimensions(self):
        data = {
            "completeness": {"score": 4, "feedback": "Completo"},
            "accuracy": {"score": 5, "feedback": "Preciso"},
            "guideline_adherence": {"score": 3, "feedback": "Parcial"},
            "clarity": {"score": 4, "feedback": "Claro"},
            "safety": {"score": 4, "feedback": "Sinais presentes"},
            "follow_up_quality": {"score": 3, "feedback": "Acompanhamento básico"},
            "overall": {"score": 4, "feedback": "Bom relatório"},
        }
        ev = ReportEvaluation.model_validate(data)
        dumped = ev.model_dump()
        # model_dump() must produce the same nested shape that nodes.py consumes
        assert dumped["safety"]["score"] == 4
        assert dumped["safety"]["feedback"] == "Sinais presentes"
        assert dumped["follow_up_quality"]["score"] == 3
        assert dumped["overall"]["score"] == 4
        assert dumped["overall"]["feedback"] == "Bom relatório"

    def test_report_evaluation_feedback_defaults_to_empty_string(self):
        # feedback is optional — omitting it should default to ""
        data = {
            "completeness": {"score": 3},
            "accuracy": {"score": 3},
            "guideline_adherence": {"score": 3},
            "clarity": {"score": 3},
            "safety": {"score": 3},
            "follow_up_quality": {"score": 3},
            "overall": {"score": 3},
        }
        ev = ReportEvaluation.model_validate(data)
        assert ev.completeness.feedback == ""
        assert ev.overall.feedback == ""

    def test_report_evaluation_score_boundary_min_max(self):
        from pydantic import ValidationError as PydanticValidationError

        # score=1 and score=5 must be accepted
        valid_dim = {"score": 1, "feedback": ""}
        data_min = {
            dim: valid_dim
            for dim in (
                "completeness",
                "accuracy",
                "guideline_adherence",
                "clarity",
                "safety",
                "follow_up_quality",
                "overall",
            )
        }
        ev_min = ReportEvaluation.model_validate(data_min)
        assert ev_min.completeness.score == 1

        data_max = {
            dim: {"score": 5, "feedback": ""}
            for dim in (
                "completeness",
                "accuracy",
                "guideline_adherence",
                "clarity",
                "safety",
                "follow_up_quality",
                "overall",
            )
        }
        ev_max = ReportEvaluation.model_validate(data_max)
        assert ev_max.overall.score == 5

        # score=0 must be rejected
        with pytest.raises(PydanticValidationError):
            ReportEvaluation.model_validate(
                {
                    "completeness": {"score": 0},
                    "accuracy": {"score": 3},
                    "guideline_adherence": {"score": 3},
                    "clarity": {"score": 3},
                    "safety": {"score": 3},
                    "follow_up_quality": {"score": 3},
                    "overall": {"score": 3},
                }
            )

        # score=6 must be rejected
        with pytest.raises(PydanticValidationError):
            ReportEvaluation.model_validate(
                {
                    "completeness": {"score": 6},
                    "accuracy": {"score": 3},
                    "guideline_adherence": {"score": 3},
                    "clarity": {"score": 3},
                    "safety": {"score": 3},
                    "follow_up_quality": {"score": 3},
                    "overall": {"score": 3},
                }
            )

    def test_report_evaluation_roundtrip_matches_nodes_consumer_shape(self):
        # Verifies the dict produced by model_dump() is exactly what
        # nodes.py:444-454 expects: each dim is a dict with "score" and "feedback".
        data = {
            "completeness": {"score": 4, "feedback": "ok"},
            "accuracy": {"score": 4, "feedback": "ok"},
            "guideline_adherence": {"score": 3, "feedback": "parcial"},
            "clarity": {"score": 4, "feedback": "ok"},
            "safety": {"score": 3, "feedback": "sem interações"},
            "follow_up_quality": {"score": 3, "feedback": "retorno em 30d"},
            "overall": {"score": 4, "feedback": "bom"},
        }
        dumped = ReportEvaluation.model_validate(data).model_dump()
        for dim, orig in data.items():
            assert isinstance(dumped[dim], dict), f"{dim} should be a dict"
            assert dumped[dim]["score"] == orig["score"]
            assert dumped[dim]["feedback"] == orig["feedback"]

    def test_report_evaluation_flat_int_fails_validation(self):
        # Flat int input no longer matches the nested DimensionScore schema
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            ReportEvaluation.model_validate(
                {
                    "completeness": 4,
                    "accuracy": 4,
                    "guideline_adherence": 3,
                    "clarity": 4,
                    "safety": 4,
                    "follow_up_quality": 3,
                    "overall": 4,
                }
            )


# ── Pydantic validation gate in helper functions ─────────────────────────


class TestPydanticValidationGate:
    """Verify that helpers validate and fall back gracefully."""

    # -- extract_entities ------------------------------------------------

    @patch("aegis.llm.generate_json")
    def test_extract_entities_valid_returns_model_dump(self, mock_gen):
        mock_gen.return_value = {
            "entities": [
                {"text": "HAS", "type": "condition", "normalized": "Hipertensão arterial sistêmica"}
            ]
        }
        result = extract_entities("Paciente com HAS")
        assert isinstance(result, dict)
        assert "entities" in result
        assert result["entities"][0]["text"] == "HAS"

    @patch("aegis.llm.generate_json")
    def test_extract_entities_invalid_returns_raw(self, mock_gen, caplog):
        bad_raw = {"entities": [{"text": "x", "type": "invalid_type_xyz"}]}
        mock_gen.return_value = bad_raw
        with caplog.at_level(logging.WARNING, logger="aegis.llm"):
            result = extract_entities("nota")
        assert result is bad_raw
        assert any("extract_entities" in r.message for r in caplog.records)

    # -- decide_retrieval ------------------------------------------------

    @patch("aegis.llm.generate_json")
    def test_decide_retrieval_valid_returns_model_dump(self, mock_gen):
        mock_gen.return_value = {"needs_retrieval": True, "queries": ["HAS"]}
        result = decide_retrieval("nota com HAS", [])
        assert result["needs_retrieval"] is True
        assert result["queries"] == ["HAS"]
        # reasoning default is populated by model_dump
        assert "reasoning" in result

    @patch("aegis.llm.generate_json")
    def test_decide_retrieval_invalid_returns_raw(self, mock_gen, caplog):
        bad_raw = {"needs_retrieval": "maybe"}  # not a bool — fails validation
        mock_gen.return_value = bad_raw
        with caplog.at_level(logging.WARNING, logger="aegis.llm"):
            result = decide_retrieval("nota", [])
        assert result is bad_raw
        assert any("decide_retrieval" in r.message for r in caplog.records)

    # -- generate_report -------------------------------------------------

    @patch("aegis.llm.generate_json")
    def test_generate_report_valid_returns_all_10_fields(self, mock_gen, sample_note):
        mock_gen.return_value = {
            "patient_summary": "João, 65a, HAS",
            "findings": ["PA elevada"],
            "assessment": "HAS não controlada",
            "plan": ["Manter losartana"],
            "guideline_references": ["Diretriz HAS"],
            "diagnosticos_diferenciais": ["ICC"],
            "sinais_alarme": ["Dispneia súbita"],
            "acompanhamento": {
                "proxima_visita": "30 dias",
                "exames_a_repetir": ["creatinina"],
                "sinais_para_escalar": ["Dispneia em repouso"],
            },
            "interacoes_medicamentosas": [],
            "limitacoes": [],
        }
        result = generate_report(sample_note)
        for field in (
            "patient_summary",
            "findings",
            "assessment",
            "plan",
            "guideline_references",
            "diagnosticos_diferenciais",
            "sinais_alarme",
            "acompanhamento",
            "interacoes_medicamentosas",
            "limitacoes",
        ):
            assert field in result, f"Missing field '{field}' in generate_report output"

    @patch("aegis.llm.generate_json")
    def test_generate_report_invalid_returns_raw(self, mock_gen, sample_note, caplog):
        # Pass a dict with a non-coercible type to trigger ValidationError
        bad_raw = {"findings": "not-a-list", "plan": 42}
        mock_gen.return_value = bad_raw
        with caplog.at_level(logging.WARNING, logger="aegis.llm"):
            result = generate_report(sample_note)
        assert result is bad_raw
        assert any("generate_report" in r.message for r in caplog.records)

    @patch("aegis.llm.generate_json")
    def test_generate_report_disclaimer_none_by_default(self, mock_gen, sample_note):
        mock_gen.return_value = {
            "patient_summary": "Teste",
            "findings": [],
            "assessment": "",
            "plan": [],
            "guideline_references": [],
            "diagnosticos_diferenciais": [],
            "sinais_alarme": [],
            "acompanhamento": {},
            "interacoes_medicamentosas": [],
            "limitacoes": [],
        }
        result = generate_report(sample_note)
        assert result.get("disclaimer") is None

    # -- evaluate_report -------------------------------------------------

    @patch("aegis.llm.generate_json")
    def test_evaluate_report_valid_returns_7_dimensions(self, mock_gen):
        # Nested {score, feedback} shape — matches EVALUATE_REPORT_PROMPT contract
        mock_gen.return_value = {
            "completeness": {"score": 4, "feedback": "Completo"},
            "accuracy": {"score": 4, "feedback": "Preciso"},
            "guideline_adherence": {"score": 3, "feedback": "Parcial"},
            "clarity": {"score": 4, "feedback": "Claro"},
            "safety": {"score": 4, "feedback": "Sinais presentes"},
            "follow_up_quality": {"score": 3, "feedback": "Retorno em 30 dias"},
            "overall": {"score": 4, "feedback": "Bom"},
        }
        result = evaluate_report({"findings": []}, note="nota", patient_data="dados")
        for dim in (
            "completeness",
            "accuracy",
            "guideline_adherence",
            "clarity",
            "safety",
            "follow_up_quality",
            "overall",
        ):
            assert dim in result, f"Missing dimension '{dim}' in evaluate_report output"
            # Each dimension must be a nested dict (model_dump of DimensionScore)
            assert isinstance(result[dim], dict)
            assert "score" in result[dim]
            assert "feedback" in result[dim]

    @patch("aegis.llm.generate_json")
    def test_evaluate_report_missing_required_dimension_returns_raw(self, mock_gen, caplog):
        # A response missing required dimensions fails validation → raw dict fallback.
        # (The old test_evaluate_report_invalid_returns_raw documented the now-fixed
        # flat-int mismatch; this replaces it with a genuine invalid case.)
        incomplete_raw = {
            "completeness": {"score": 4, "feedback": "ok"},
            # missing: accuracy, guideline_adherence, clarity, safety,
            #          follow_up_quality, overall — all required
        }
        mock_gen.return_value = incomplete_raw
        with caplog.at_level(logging.WARNING, logger="aegis.llm"):
            result = evaluate_report({"findings": []})
        assert result is incomplete_raw
        assert any("evaluate_report" in r.message for r in caplog.records)

    @patch("aegis.llm.generate_json")
    def test_evaluate_report_uses_correct_system_prompt_and_temp(self, mock_gen):
        mock_gen.return_value = {
            "completeness": {"score": 3, "feedback": ""},
            "accuracy": {"score": 3, "feedback": ""},
            "guideline_adherence": {"score": 3, "feedback": ""},
            "clarity": {"score": 3, "feedback": ""},
            "safety": {"score": 3, "feedback": ""},
            "follow_up_quality": {"score": 3, "feedback": ""},
            "overall": {"score": 3, "feedback": ""},
        }
        evaluate_report({"findings": []})
        kwargs = mock_gen.call_args.kwargs
        assert kwargs["system_prompt"] == SYSTEM_REPORT_EVALUATION
        assert kwargs["temperature"] == TEMP_EVALUATION

    @patch("aegis.llm.generate_json")
    def test_generate_report_uses_correct_system_prompt_and_temp(self, mock_gen, sample_note):
        mock_gen.return_value = {}
        generate_report(sample_note)
        kwargs = mock_gen.call_args.kwargs
        assert kwargs["system_prompt"] == SYSTEM_REPORT_GENERATION
        assert kwargs["temperature"] == TEMP_REPORT


# ── Integration tests (require live Ollama) ──────────────────────────────


@pytest.mark.integration
class TestLLMIntegration:
    """Integration tests that hit the real Ollama server.

    Run with: pytest -m integration
    Skip with: pytest -m "not integration"
    """

    @pytest.fixture(autouse=True)
    def _skip_if_no_ollama(self):
        from tests.conftest import is_ollama_available

        if not is_ollama_available():
            pytest.skip("Ollama not available or no models loaded")

    def test_generate_returns_nonempty_string(self):
        result = generate("Say 'hello' in one word.")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_json_returns_dict(self):
        result = generate_json('Return a JSON object with a single key "status" set to "ok".')
        assert isinstance(result, dict)

    def test_expand_note_returns_expected_keys(self, sample_note):
        result = expand_note(sample_note)
        assert isinstance(result, dict)
        # The model should produce at least one of these keys
        assert "expanded_note" in result or "entities" in result

    def test_extract_entities_finds_medical_terms(self, sample_note):
        result = extract_entities(sample_note)
        assert "entities" in result
        assert isinstance(result["entities"], list)
        assert len(result["entities"]) > 0

    def test_decide_retrieval_returns_boolean(self, sample_note):
        entities = [{"text": "dispneia", "type": "symptom"}]
        result = decide_retrieval(sample_note, entities)
        assert "needs_retrieval" in result
        assert isinstance(result["needs_retrieval"], bool)
