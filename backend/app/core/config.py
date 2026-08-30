"""Core configuration module."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_name: str = "report-analysis-agent"
    app_env: Literal["development", "production", "testing"] = "development"
    debug: bool = False

    # API Keys
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")

    # Anthropic API Configuration (for DashScope or custom endpoints)
    anthropic_auth_token: str = Field(default="", alias="ANTHROPIC_AUTH_TOKEN")
    anthropic_base_url: str = Field(default="", alias="ANTHROPIC_BASE_URL")
    api_timeout_ms: int = Field(default=300000, alias="API_TIMEOUT_MS")
    anthropic_default_model: str = Field(default="glm-5", alias="ANTHROPIC_DEFAULT_SONNET_MODEL")

    # Paths
    base_dir: Path = Field(default=Path(__file__).parent.parent.parent, exclude=True)
    vector_store_path: Path = Field(default=Path("./data/vectorstore"), alias="VECTOR_STORE_PATH")
    reports_path: Path = Field(default=Path("./data/reports"), alias="REPORTS_PATH")
    uploads_path: Path = Field(default=Path("./data/uploads"), alias="UPLOADS_PATH")

    # Models
    llm_model: str = Field(default="glm-5", alias="LLM_MODEL")
    embedding_model: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL")

    # RAG Settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_top_k: int = 5
    retrieval_score_threshold: float = 0.0
    retrieval_candidate_multiplier: int = 4
    hybrid_search_enabled: bool = True
    query_rewrite_enabled: bool = True
    query_rewrite_max_variants: int = 3
    rag_relevance_threshold: float = 0.5

    def model_post_init(self, __context) -> None:
        """Ensure paths are absolute and directories exist."""
        for path_attr in ["vector_store_path", "reports_path", "uploads_path"]:
            path = getattr(self, path_attr)
            if not path.is_absolute():
                object.__setattr__(self, path_attr, self.base_dir / path)
            getattr(self, path_attr).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
