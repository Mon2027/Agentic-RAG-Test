"""Tests for RAG analyst tools."""

from unittest.mock import MagicMock, patch

from app.agents.rag_analyst import (
    check_rag_relevance,
    create_rag_analyst_tools,
    list_available_reports,
    search_reports,
)


class TestSearchReports:
    """Test cases for search_reports tool."""

    @patch("app.agents.rag_analyst.get_retriever")
    def test_search_reports_no_results(self, mock_get_retriever):
        """Test search when no results found."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = MagicMock(results=[], sources=[])
        mock_get_retriever.return_value = mock_retriever

        result = search_reports.invoke({"query": "不存在的内容", "top_k": 5})

        assert "未找到" in result or "没有" in result

    @patch("app.agents.rag_analyst.get_retriever")
    def test_search_reports_with_results(self, mock_get_retriever):
        """Test search with results."""
        from app.rag.retriever import RetrievalResult, RetrievedContext

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = RetrievedContext(
            query="测试",
            results=[
                RetrievalResult(
                    content="这是测试内容",
                    metadata={"file_name": "test.pdf", "page_number": 1},
                    score=0.9
                )
            ],
            sources=[{"file_id": "test", "file_name": "test.pdf"}]
        )
        mock_get_retriever.return_value = mock_retriever

        result = search_reports.invoke({"query": "测试", "top_k": 5})

        assert "日期、年份只能支持同一句或同一项目中明确关联的事实" in result
        assert "研报未披露该事件开始年份" in result

        assert "测试内容" in result
        assert "test.pdf" in result

    @patch("app.agents.rag_analyst.get_retriever")
    def test_search_reports_passes_topic_to_retriever(self, mock_get_retriever):
        """The Agent-facing search tool must expose the production topic filter."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = MagicMock(results=[], sources=[])
        mock_get_retriever.return_value = mock_retriever

        search_reports.invoke({
            "query": "凌云光财务表现",
            "top_k": 5,
            "topic": "embodied_intelligence",
        })

        mock_retriever.retrieve.assert_called_once_with(
            query="凌云光财务表现",
            top_k=5,
            topic="embodied_intelligence",
        )


class TestListAvailableReports:
    """Test cases for list_available_reports tool."""

    @patch("app.agents.rag_analyst.get_vector_store")
    @patch("app.agents.rag_analyst.get_settings")
    def test_list_empty(self, mock_get_settings, mock_get_vector_store):
        """Test listing when no reports available."""
        mock_settings = MagicMock()
        mock_settings.reports_path.glob.return_value = []
        mock_get_settings.return_value = mock_settings

        mock_store = MagicMock()
        mock_store.list_files.return_value = []
        mock_get_vector_store.return_value = mock_store

        result = list_available_reports.invoke({})

        assert "没有" in result or "空" in result

    @patch("app.agents.rag_analyst.get_vector_store")
    @patch("app.agents.rag_analyst.get_settings")
    def test_list_with_reports(self, mock_get_settings, mock_get_vector_store):
        """Test listing with reports."""
        mock_settings = MagicMock()
        mock_settings.reports_path.glob.return_value = []
        mock_get_settings.return_value = mock_settings

        mock_store = MagicMock()
        mock_store.list_files.return_value = [
            {"file_id": "test123", "file_name": "test.pdf", "chunk_count": 10}
        ]
        mock_get_vector_store.return_value = mock_store

        result = list_available_reports.invoke({})

        assert "test.pdf" in result or "test123" in result


class TestCheckRAGRelevance:
    """Test cases for check_rag_relevance tool."""

    @patch("app.agents.rag_analyst.get_settings")
    @patch("app.agents.rag_analyst.get_retriever")
    def test_check_relevant(self, mock_get_retriever, mock_get_settings):
        """Test when content is relevant."""
        mock_settings = MagicMock()
        mock_settings.rag_relevance_threshold = 0.6
        mock_get_settings.return_value = mock_settings
        mock_retriever = MagicMock()
        mock_retriever.check_relevance.return_value = (True, 0.8)
        mock_get_retriever.return_value = mock_retriever

        result = check_rag_relevance.invoke({"query": "测试查询"})

        assert "相关" in result
        assert "0.8" in result
        mock_retriever.check_relevance.assert_called_once_with(
            "测试查询",
            min_relevant_score=0.6,
            topic=None,
        )

    @patch("app.agents.rag_analyst.get_settings")
    @patch("app.agents.rag_analyst.get_retriever")
    def test_check_relevance_passes_topic(self, mock_get_retriever, mock_get_settings):
        """Relevance pre-check and retrieval must use the same topic boundary."""
        mock_get_settings.return_value.rag_relevance_threshold = 0.6
        mock_retriever = MagicMock()
        mock_retriever.check_relevance.return_value = (True, 0.8)
        mock_get_retriever.return_value = mock_retriever

        check_rag_relevance.invoke({
            "query": "具身智能产业链",
            "topic": "embodied_intelligence",
        })

        mock_retriever.check_relevance.assert_called_once_with(
            "具身智能产业链",
            min_relevant_score=0.6,
            topic="embodied_intelligence",
        )

    @patch("app.agents.rag_analyst.get_retriever")
    def test_check_not_relevant(self, mock_get_retriever):
        """Test when content is not relevant."""
        mock_retriever = MagicMock()
        mock_retriever.check_relevance.return_value = (False, 0.3)
        mock_get_retriever.return_value = mock_retriever

        result = check_rag_relevance.invoke({"query": "测试查询"})

        assert "不相关" in result or "联网搜索" in result


class TestCreateRAGAnalystTools:
    """Test tool creation."""

    def test_creates_all_tools(self):
        """Test that all tools are created."""
        tools = create_rag_analyst_tools()

        assert len(tools) == 4
        tool_names = [t.name for t in tools]
        assert "search_reports" in tool_names
        assert "get_report_summary" in tool_names
        assert "list_available_reports" in tool_names
        assert "check_rag_relevance" in tool_names
