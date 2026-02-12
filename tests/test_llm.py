"""Unit and integration tests for aegis.llm — LLM client and prompt templates."""

import json
from unittest.mock import patch

import pytest

from aegis.llm import (
    EXPAND_NOTE_PROMPT,
    ENTITY_EXTRACTION_PROMPT,
    REPORT_PROMPT,
    SELF_RAG_DECISION_PROMPT,
    EVALUATE_REPORT_PROMPT,
    SYSTEM_MEDICAL,
    _extract_json,
    generate,
    generate_json,
    expand_note,
    extract_entities,
    generate_report,
    decide_retrieval,
    evaluate_report,
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

    def test_report_prompt_has_all_placeholders(self):
        assert "{note}" in REPORT_PROMPT
        assert "{patient_data}" in REPORT_PROMPT
        assert "{guidelines}" in REPORT_PROMPT

    def test_self_rag_prompt_has_placeholders(self):
        assert "{note}" in SELF_RAG_DECISION_PROMPT
        assert "{entities}" in SELF_RAG_DECISION_PROMPT

    def test_evaluate_report_prompt_has_placeholder(self):
        assert "{report}" in EVALUATE_REPORT_PROMPT

    def test_system_medical_is_nonempty(self):
        assert len(SYSTEM_MEDICAL) > 50

    def test_expand_note_prompt_formats_correctly(self, sample_note):
        rendered = EXPAND_NOTE_PROMPT.format(note=sample_note)
        assert sample_note in rendered
        assert "{note}" not in rendered

    def test_report_prompt_formats_correctly(self, sample_note):
        rendered = REPORT_PROMPT.format(
            note=sample_note,
            patient_data="John Doe, 65y",
            guidelines="Treat hypertension per JNC8",
        )
        assert sample_note in rendered
        assert "John Doe" in rendered
        assert "JNC8" in rendered


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


# ── generate() mocked tests ─────────────────────────────────────────────


class TestGenerateMocked:
    """Unit tests for generate() with Ollama mocked out."""

    @patch("aegis.llm.ollama.chat")
    def test_generate_returns_content(self, mock_chat):
        mock_chat.return_value = {"message": {"content": "Hello, doctor."}}
        result = generate("Test prompt")
        assert result == "Hello, doctor."

    @patch("aegis.llm.ollama.chat")
    def test_generate_passes_system_prompt(self, mock_chat):
        mock_chat.return_value = {"message": {"content": "ok"}}
        generate("Test", system_prompt="Custom system")
        call_args = mock_chat.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Custom system"

    @patch("aegis.llm.ollama.chat")
    def test_generate_passes_user_prompt(self, mock_chat):
        mock_chat.return_value = {"message": {"content": "ok"}}
        generate("My user prompt")
        call_args = mock_chat.call_args
        messages = call_args.kwargs["messages"]
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "My user prompt"

    @patch("aegis.llm.ollama.chat")
    def test_generate_uses_configured_model(self, mock_chat):
        mock_chat.return_value = {"message": {"content": "ok"}}
        generate("Test")
        call_args = mock_chat.call_args
        assert call_args.kwargs["model"] == "mistral"

    @patch("aegis.llm.ollama.chat")
    def test_generate_uses_low_temperature(self, mock_chat):
        mock_chat.return_value = {"message": {"content": "ok"}}
        generate("Test")
        call_args = mock_chat.call_args
        assert call_args.kwargs["options"]["temperature"] == 0.3


# ── generate_json() mocked tests ────────────────────────────────────────


class TestGenerateJsonMocked:
    """Unit tests for generate_json() with Ollama mocked."""

    @patch("aegis.llm.ollama.chat")
    def test_returns_parsed_json(self, mock_chat):
        payload = {"expanded_note": "test", "entities": []}
        mock_chat.return_value = {"message": {"content": json.dumps(payload)}}
        result = generate_json("Test")
        assert result == payload

    @patch("aegis.llm.ollama.chat")
    def test_uses_json_format(self, mock_chat):
        mock_chat.return_value = {"message": {"content": '{"key": "val"}'}}
        generate_json("Test")
        call_args = mock_chat.call_args
        assert call_args.kwargs["format"] == "json"

    @patch("aegis.llm.ollama.chat")
    def test_falls_back_on_invalid_json(self, mock_chat):
        # First call (with format=json) returns invalid JSON → JSONDecodeError
        # Second call (fallback without format=json) returns wrapped JSON
        mock_chat.side_effect = [
            {"message": {"content": "not json"}},
            {"message": {"content": '{"fallback": true}'}},
        ]
        result = generate_json("Test")
        assert result == {"fallback": True}


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
        assert "Not available" in prompt_arg

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
        result = evaluate_report(report)
        mock_gen.assert_called_once()
        assert "completeness" in result


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
