"""配置模块的测试。

重点验证配置默认值、缓存行为、路径规范化，以及从环境变量读取 API Key。
涉及环境变量的用例使用 pytest 的 monkeypatch 临时修改进程环境；用例结束后，
monkeypatch 会自动恢复原值，避免不同测试互相影响。
"""


class TestConfig:
    """集中测试 app.core.config 中的 Settings 和 get_settings。"""

    def test_settings_defaults(self, monkeypatch):
        """排除外部配置后，检查 Settings 在代码中声明的默认值。"""
        from app.core.config import Settings

        # 逐个清除可能覆盖默认值的同名环境变量。
        # raising=False 表示变量原本不存在时也不报错；测试结束后会自动恢复。
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

        # 不读取项目的 .env 文件，确保下面的断言检验的是代码默认值。
        settings = Settings(_env_file=None)

        # APP_ENV 由全局 conftest.py 设置为 testing，因此这里允许三种合法值。
        assert settings.app_name == "report-analysis-agent"
        assert settings.app_env in ["development", "production", "testing"]
        # 模型、分块与检索参数分别检查，配置默认值发生变化时便于定位。
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
        """验证 get_settings 多次调用会返回同一个缓存实例。"""
        from app.core.config import get_settings

        # get_settings 使用 lru_cache；两次调用不应重复创建 Settings。
        settings1 = get_settings()
        settings2 = get_settings()

        # is 比较对象身份，能够证明二者是同一个实例，而不只是字段值相等。
        assert settings1 is settings2

    def test_paths_are_absolute(self):
        """验证数据目录在配置初始化后都成为绝对路径。"""
        from app.core.config import get_settings

        settings = get_settings()

        # Settings.model_post_init 会规范化这三个路径，避免依赖运行时工作目录。
        assert settings.vector_store_path.is_absolute()
        assert settings.reports_path.is_absolute()
        assert settings.uploads_path.is_absolute()

    def test_api_key_from_env(self, monkeypatch):
        """验证 Settings 能从环境变量读取 Anthropic API Key。"""
        from app.core.config import Settings

        # 只在当前测试中设置一个假凭证，不使用开发者的真实 API Key。
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test_key_123")

        # 直接新建 Settings，避免 get_settings 的缓存遮蔽刚设置的环境变量。
        settings = Settings()
        assert settings.anthropic_api_key == "test_key_123"
