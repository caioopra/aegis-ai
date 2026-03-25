"""LLM and embedding provider factories."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aegis.providers.base import ChatProvider, EmbedProvider


def get_chat_provider() -> ChatProvider:
    """Return a ChatProvider based on the configured llm_provider setting."""
    from aegis.config import settings

    if settings.llm_provider == "ollama":
        from aegis.providers.ollama import OllamaChatProvider

        return OllamaChatProvider(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
        )
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider!r}. Supported: 'ollama'")


def get_embed_provider() -> EmbedProvider:
    """Return an EmbedProvider based on the configured llm_provider setting."""
    from aegis.config import settings

    if settings.llm_provider == "ollama":
        from aegis.providers.ollama import OllamaEmbedProvider

        return OllamaEmbedProvider(
            model=settings.ollama_embed_model,
            base_url=settings.ollama_base_url,
            dim=settings.embedding_dim,
        )
    raise ValueError(f"Unknown embedding provider: {settings.llm_provider!r}. Supported: 'ollama'")
