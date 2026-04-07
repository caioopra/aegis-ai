"""Unit and integration tests for aegis.llm — LLM client and prompt templates."""

import json
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
