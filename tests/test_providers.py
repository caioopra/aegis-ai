"""Unit tests for aegis.providers — provider abstraction layer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aegis.providers.ollama import OllamaChatProvider, OllamaEmbedProvider


# ── OllamaChatProvider tests ─────────────────────────────────────────────


class TestOllamaChatProvider:
    """Unit tests for OllamaChatProvider with ollama mocked."""

    @patch("aegis.providers.ollama._ollama.Client")
    def test_chat_returns_content(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.chat.return_value = {"message": {"content": "Hello"}}
        provider = OllamaChatProvider(model="mistral")
        result = provider.chat(messages=[{"role": "user", "content": "Hi"}])
        assert result == "Hello"

    @patch("aegis.providers.ollama._ollama.Client")
    def test_chat_prepends_system_prompt(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.chat.return_value = {"message": {"content": "ok"}}
        provider = OllamaChatProvider(model="mistral")
        provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
            system_prompt="You are a doctor.",
        )
        call_kwargs = mock_client.chat.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a doctor."
        assert messages[1]["role"] == "user"

    @patch("aegis.providers.ollama._ollama.Client")
    def test_chat_without_system_prompt(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.chat.return_value = {"message": {"content": "ok"}}
        provider = OllamaChatProvider(model="mistral")
        provider.chat(messages=[{"role": "user", "content": "Hi"}], system_prompt="")
        call_kwargs = mock_client.chat.call_args[1]
        messages = call_kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    @patch("aegis.providers.ollama._ollama.Client")
    def test_chat_passes_model(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.chat.return_value = {"message": {"content": "ok"}}
        provider = OllamaChatProvider(model="llama3.2")
        provider.chat(messages=[{"role": "user", "content": "Hi"}])
        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["model"] == "llama3.2"

    @patch("aegis.providers.ollama._ollama.Client")
    def test_chat_passes_temperature(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.chat.return_value = {"message": {"content": "ok"}}
        provider = OllamaChatProvider(model="mistral")
        provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.7,
        )
        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["options"]["temperature"] == 0.7

    @patch("aegis.providers.ollama._ollama.Client")
    def test_chat_json_mode_sets_format(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.chat.return_value = {"message": {"content": '{"key": "val"}'}}
        provider = OllamaChatProvider(model="mistral")
        provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
            json_mode=True,
        )
        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["format"] == "json"

    @patch("aegis.providers.ollama._ollama.Client")
    def test_chat_no_json_mode_omits_format(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.chat.return_value = {"message": {"content": "ok"}}
        provider = OllamaChatProvider(model="mistral")
        provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
            json_mode=False,
        )
        call_kwargs = mock_client.chat.call_args[1]
        assert "format" not in call_kwargs

    @patch("aegis.providers.ollama._ollama.Client")
    def test_chat_client_receives_base_url(self, mock_client_cls):
        mock_client_cls.return_value = MagicMock()
        OllamaChatProvider(model="mistral", base_url="http://remote:11434")
        mock_client_cls.assert_called_once_with(host="http://remote:11434")


# ── OllamaEmbedProvider tests ───────────────────────────────────────────


class TestOllamaEmbedProvider:
    """Unit tests for OllamaEmbedProvider with ollama mocked."""

    @patch("aegis.providers.ollama._ollama.Client")
    def test_embed_returns_vector(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.embed.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        provider = OllamaEmbedProvider(model="nomic-embed-text")
        result = provider.embed("test text")
        assert result == [0.1, 0.2, 0.3]

    @patch("aegis.providers.ollama._ollama.Client")
    def test_embed_passes_model(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.embed.return_value = {"embeddings": [[0.1]]}
        provider = OllamaEmbedProvider(model="custom-embed")
        provider.embed("test text")
        call_kwargs = mock_client.embed.call_args[1]
        assert call_kwargs["model"] == "custom-embed"

    @patch("aegis.providers.ollama._ollama.Client")
    def test_embedding_dim_returns_configured_value(self, mock_client_cls):
        mock_client_cls.return_value = MagicMock()
        provider = OllamaEmbedProvider(model="nomic-embed-text", dim=1024)
        assert provider.embedding_dim == 1024

    @patch("aegis.providers.ollama._ollama.Client")
    def test_embedding_dim_default(self, mock_client_cls):
        mock_client_cls.return_value = MagicMock()
        provider = OllamaEmbedProvider()
        assert provider.embedding_dim == 768

    @patch("aegis.providers.ollama._ollama.Client")
    def test_embed_client_receives_base_url(self, mock_client_cls):
        mock_client_cls.return_value = MagicMock()
        OllamaEmbedProvider(model="nomic-embed-text", base_url="http://remote:11434")
        mock_client_cls.assert_called_once_with(host="http://remote:11434")


# ── Factory function tests ──────────────────────────────────────────────


class TestFactoryFunctions:
    """Unit tests for get_chat_provider() and get_embed_provider() factories."""

    @patch("aegis.config.settings")
    def test_get_chat_provider_returns_ollama(self, mock_settings):
        from aegis.providers import get_chat_provider

        mock_settings.llm_provider = "ollama"
        mock_settings.ollama_model = "mistral"
        mock_settings.ollama_base_url = "http://localhost:11434"
        provider = get_chat_provider()
        assert isinstance(provider, OllamaChatProvider)

    @patch("aegis.config.settings")
    def test_get_embed_provider_returns_ollama(self, mock_settings):
        from aegis.providers import get_embed_provider

        mock_settings.llm_provider = "ollama"
        mock_settings.ollama_embed_model = "nomic-embed-text"
        mock_settings.ollama_base_url = "http://localhost:11434"
        mock_settings.embedding_dim = 768
        provider = get_embed_provider()
        assert isinstance(provider, OllamaEmbedProvider)

    @patch("aegis.config.settings")
    def test_get_chat_provider_unknown_raises(self, mock_settings):
        from aegis.providers import get_chat_provider

        mock_settings.llm_provider = "unknown"
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_chat_provider()

    @patch("aegis.config.settings")
    def test_get_embed_provider_unknown_raises(self, mock_settings):
        from aegis.providers import get_embed_provider

        mock_settings.llm_provider = "unknown"
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            get_embed_provider()
