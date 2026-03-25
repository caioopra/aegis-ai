"""Ollama provider implementations for chat and embeddings."""

from __future__ import annotations

from typing import Any

import ollama as _ollama


class OllamaChatProvider:
    """Ollama implementation of the ChatProvider protocol."""

    def __init__(self, model: str = "mistral", base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.base_url = base_url
        self._client = _ollama.Client(host=base_url)

    def chat(
        self,
        messages: list[dict[str, str]],
        system_prompt: str = "",
        temperature: float = 0.3,
        json_mode: bool = False,
    ) -> str:
        all_messages: list[dict[str, str]] = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        all_messages.extend(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": all_messages,
            "options": {"temperature": temperature},
        }
        if json_mode:
            kwargs["format"] = "json"

        response = self._client.chat(**kwargs)
        return response["message"]["content"]


class OllamaEmbedProvider:
    """Ollama implementation of the EmbedProvider protocol."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        dim: int = 768,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self._dim = dim
        self._client = _ollama.Client(host=base_url)

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        response = self._client.embed(model=self.model, input=text)
        return response["embeddings"][0]
