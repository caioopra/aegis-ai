"""Unit tests for aegis.config — Settings loading and defaults."""

from pathlib import Path


class TestSettingsDefaults:
    """Verify that Settings loads correct defaults when no env is set."""

    def test_default_ollama_model(self):
        from aegis.config import Settings

        s = Settings()
        assert s.ollama_model == "mistral"

    def test_default_ollama_base_url(self):
        from aegis.config import Settings

        s = Settings()
        assert s.ollama_base_url == "http://localhost:11434"

    def test_default_embed_model(self):
        from aegis.config import Settings

        s = Settings()
        assert s.ollama_embed_model == "nomic-embed-text"

    def test_default_qdrant_url(self):
        from aegis.config import Settings

        s = Settings()
        assert s.qdrant_url == "http://localhost:6333"

    def test_default_qdrant_collection(self):
        from aegis.config import Settings

        s = Settings()
        assert s.qdrant_collection == "clinical_guidelines"

    def test_default_data_paths(self):
        from aegis.config import Settings

        s = Settings()
        assert s.synthea_data_dir == Path("data/synthea")
        assert s.guidelines_dir == Path("data/guidelines")


class TestSettingsOverride:
    """Verify that Settings picks up environment variable overrides."""

    def test_override_ollama_model(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODEL", "llama3.2")
        from aegis.config import Settings

        s = Settings()
        assert s.ollama_model == "llama3.2"

    def test_override_qdrant_url(self, monkeypatch):
        monkeypatch.setenv("QDRANT_URL", "http://remote-host:6333")
        from aegis.config import Settings

        s = Settings()
        assert s.qdrant_url == "http://remote-host:6333"

    def test_override_data_paths(self, monkeypatch):
        monkeypatch.setenv("SYNTHEA_DATA_DIR", "/tmp/synthea")
        monkeypatch.setenv("GUIDELINES_DIR", "/tmp/guidelines")
        from aegis.config import Settings

        s = Settings()
        assert s.synthea_data_dir == Path("/tmp/synthea")
        assert s.guidelines_dir == Path("/tmp/guidelines")


class TestSingletonSettings:
    """Verify the module-level settings instance is accessible."""

    def test_settings_instance_exists(self):
        from aegis.config import settings

        assert settings is not None
        assert hasattr(settings, "ollama_model")
