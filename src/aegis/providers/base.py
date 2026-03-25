"""Provider protocols -- the contracts that all LLM/embedding backends must satisfy."""

from __future__ import annotations

from typing import Protocol


class ChatProvider(Protocol):
    """Protocol for LLM chat completion providers."""

    def chat(
        self,
        messages: list[dict[str, str]],
        system_prompt: str = "",
        temperature: float = 0.3,
        json_mode: bool = False,
    ) -> str:
        """Send messages and return the response text content."""
        ...


class EmbedProvider(Protocol):
    """Protocol for text embedding providers."""

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the embedding vectors produced."""
        ...

    def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text."""
        ...
