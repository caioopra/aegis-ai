"""Application settings loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration — values come from .env or environment variables."""

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral"
    ollama_embed_model: str = "nomic-embed-text"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "clinical_guidelines"

    # Data paths (relative to project root)
    synthea_data_dir: Path = Path("data/synthea")
    guidelines_dir: Path = Path("data/guidelines")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
