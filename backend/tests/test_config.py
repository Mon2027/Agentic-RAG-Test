"""Tests for configuration module."""


class TestConfig:
    """Test cases for configuration."""

    def test_settings_defaults(self, monkeypatch):
        """Test code defaults without inheriting developer .env overrides."""
        from app.core.config import Settings

        for variable in (
            "APP_NAME",
            "LLM_MODEL",
            "EMBEDDING_MODEL",
            "CHUNK_SIZE",
            "CHUNK_OVERLAP",
            "RETRIEVAL_TOP_K",
            "RETRIEVAL_SCORE_THRESHOLD",
            "RETRIEVAL_CANDIDATE_MULTIPLIER",
            "HYBRID_SEARCH_ENABLED",
            "QUERY_REWRITE_ENABLED",
            "QUERY_REWRITE_MAX_VARIANTS",
            "RAG_RELEVANCE_THRESHOLD",
        ):
            monkeypatch.delenv(variable, raising=False)

        settings = Settings(_env_file=None)

        assert settings.app_name == "report-analysis-agent"
        assert settings.app_env in ["development", "production", "testing"]
        assert settings.llm_model == "glm-5"
        assert settings.embedding_model == "BAAI/bge-m3"
        assert settings.chunk_size == 1000
        assert settings.chunk_overlap == 200
        assert settings.retrieval_top_k == 5
        assert settings.retrieval_score_threshold == 0.0
        assert settings.retrieval_candidate_multiplier == 4
        assert settings.hybrid_search_enabled is True
        assert settings.query_rewrite_enabled is True
        assert settings.query_rewrite_max_variants == 3
        assert settings.rag_relevance_threshold == 0.5

    def test_get_settings_cached(self):
        """Test that get_settings returns cached instance."""
        from app.core.config import get_settings

        settings1 = get_settings()
        settings2 = get_settings()

        # Should be the same instance (cached)
        assert settings1 is settings2

    def test_paths_are_absolute(self):
        """Test that paths are converted to absolute."""
        from app.core.config import get_settings

        settings = get_settings()

        assert settings.vector_store_path.is_absolute()
        assert settings.reports_path.is_absolute()
        assert settings.uploads_path.is_absolute()

    def test_api_key_from_env(self, monkeypatch):
        """Test API key loading from environment."""
        from app.core.config import Settings

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test_key_123")

        settings = Settings()
        assert settings.anthropic_api_key == "test_key_123"
