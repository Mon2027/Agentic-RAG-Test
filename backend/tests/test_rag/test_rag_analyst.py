"""RAG 分析智能体工具的单元测试。

这里验证“检索器和向量库能力如何包装成智能体可调用的工具”：
搜索结果的文字格式、报告列表、相关性判断，以及 topic 等参数是否传到底层。
通过 patch 替换工具模块中使用的工厂函数，避免连接真实向量库或嵌入模型。
"""

from unittest.mock import MagicMock, patch

from app.agents.rag_analyst import (
    check_rag_relevance,
    create_rag_analyst_tools,
    list_available_reports,
    search_reports,
)


class TestSearchReports:
    """验证搜索工具在无结果、有结果及带主题过滤时的行为。"""

    @patch("app.agents.rag_analyst.get_retriever")
    def test_search_reports_no_results(self, mock_get_retriever):
        """检索器没有返回分块时，工具应给出明确的未找到提示。"""
        # patch 的目标是工具实现实际查找 get_retriever 的模块位置。
        # 工厂函数返回模拟检索器，retrieve 再返回一个空结果对象。
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = MagicMock(results=[], sources=[])
        mock_get_retriever.return_value = mock_retriever

        # @tool 包装后的对象通过 invoke 接收参数字典，而不是直接按普通函数调用。
        result = search_reports.invoke({"query": "不存在的内容", "top_k": 5})

        # 允许两种自然语言提示，重点是用户能看懂“没有命中”。
        assert "未找到" in result or "没有" in result

    @patch("app.agents.rag_analyst.get_retriever")
    def test_search_reports_with_results(self, mock_get_retriever):
        """搜索有命中时，应输出证据内容、来源和证据使用约束。"""
        from app.rag.retriever import RetrievalResult, RetrievedContext

        # 与空结果用例不同，这里构造真实 RetrievedContext，
        # 使工具调用它的 format_context 时执行真正的格式化逻辑。
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

        # 前两条检查日期、年份不能被错误嫁接；后两条检查证据和文件名可见。
        assert "日期、年份只能支持同一句或同一项目中明确关联的事实" in result
        assert "研报未披露该事件开始年份" in result

        assert "测试内容" in result
        assert "test.pdf" in result

    @patch("app.agents.rag_analyst.get_retriever")
    def test_search_reports_passes_topic_to_retriever(self, mock_get_retriever):
        """智能体传入的 topic 应原样交给检索器，保持主题边界。"""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = MagicMock(results=[], sources=[])
        mock_get_retriever.return_value = mock_retriever

        search_reports.invoke({
            "query": "凌云光财务表现",
            "top_k": 5,
            "topic": "embodied_intelligence",
        })

        # 直接检查底层调用参数，比只检查工具返回的提示文字更能证明过滤生效。
        mock_retriever.retrieve.assert_called_once_with(
            query="凌云光财务表现",
            top_k=5,
            topic="embodied_intelligence",
        )


class TestListAvailableReports:
    """验证工具如何合并向量库中已索引的报告与磁盘上的报告。"""

    # 两层 patch 从下往上注入：get_settings 对应第一个参数，
    # get_vector_store 对应第二个参数；测试中两条数据来源都可单独控制。
    @patch("app.agents.rag_analyst.get_vector_store")
    @patch("app.agents.rag_analyst.get_settings")
    def test_list_empty(self, mock_get_settings, mock_get_vector_store):
        """磁盘目录与向量库都为空时，应提示尚无可用研报。"""
        # reports_path.glob 模拟磁盘上没有 PDF；list_files 模拟没有已索引文件。
        mock_settings = MagicMock()
        mock_settings.reports_path.glob.return_value = []
        mock_get_settings.return_value = mock_settings

        mock_store = MagicMock()
        mock_store.list_files.return_value = []
        mock_get_vector_store.return_value = mock_store

        result = list_available_reports.invoke({})

        # 无参数工具也使用空字典调用 invoke。
        assert "没有" in result or "空" in result

    @patch("app.agents.rag_analyst.get_vector_store")
    @patch("app.agents.rag_analyst.get_settings")
    def test_list_with_reports(self, mock_get_settings, mock_get_vector_store):
        """向量库有已索引报告时，列表应显示其名称或文件 ID。"""
        # 磁盘目录保持为空，隔离验证向量库 list_files 这一数据来源。
        mock_settings = MagicMock()
        mock_settings.reports_path.glob.return_value = []
        mock_get_settings.return_value = mock_settings

        mock_store = MagicMock()
        mock_store.list_files.return_value = [
            {"file_id": "test123", "file_name": "test.pdf", "chunk_count": 10}
        ]
        mock_get_vector_store.return_value = mock_store

        result = list_available_reports.invoke({})

        # 此用例只检查报告能被列出，不约束状态图标或完整排版。
        assert "test.pdf" in result or "test123" in result


class TestCheckRAGRelevance:
    """验证相关性预检查的阈值、主题参数及推荐文字。"""

    # 下方装饰器先提供检索器 mock；上方装饰器再提供配置 mock。
    @patch("app.agents.rag_analyst.get_settings")
    @patch("app.agents.rag_analyst.get_retriever")
    def test_check_relevant(self, mock_get_retriever, mock_get_settings):
        """高于阈值的结果应提示相关，并返回最佳分数。"""
        # 阈值由配置提供；模拟检索器返回 (是否相关, 最佳分数)。
        mock_settings = MagicMock()
        mock_settings.rag_relevance_threshold = 0.6
        mock_get_settings.return_value = mock_settings
        mock_retriever = MagicMock()
        mock_retriever.check_relevance.return_value = (True, 0.8)
        mock_get_retriever.return_value = mock_retriever

        result = check_rag_relevance.invoke({"query": "测试查询"})

        # 同时断言用户可见文字和调用参数，证明 0.6 阈值确实传到底层。
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
        """预检查也必须传递 topic，与正式检索使用同一主题范围。"""
        mock_get_settings.return_value.rag_relevance_threshold = 0.6
        mock_retriever = MagicMock()
        mock_retriever.check_relevance.return_value = (True, 0.8)
        mock_get_retriever.return_value = mock_retriever

        check_rag_relevance.invoke({
            "query": "具身智能产业链",
            "topic": "embodied_intelligence",
        })

        # 如果预检查丢失 topic，可能用其他主题的高分结果错误判断为相关。
        mock_retriever.check_relevance.assert_called_once_with(
            "具身智能产业链",
            min_relevant_score=0.6,
            topic="embodied_intelligence",
        )

    @patch("app.agents.rag_analyst.get_retriever")
    def test_check_not_relevant(self, mock_get_retriever):
        """相关性不足时，应提示本地证据不足或给出下一步建议。"""
        mock_retriever = MagicMock()
        mock_retriever.check_relevance.return_value = (False, 0.3)
        mock_get_retriever.return_value = mock_retriever

        result = check_rag_relevance.invoke({"query": "测试查询"})

        # 这里只检查对用户的提示，不断言配置阈值或底层调用细节。
        assert "不相关" in result or "联网搜索" in result


class TestCreateRAGAnalystTools:
    """验证提供给主智能体的 RAG 工具清单。"""

    def test_creates_all_tools(self):
        """工厂应返回预期的四个工具，避免遗漏某项能力。"""
        tools = create_rag_analyst_tools()

        # 通过工具 name 而非对象顺序判断，减少列表调整对测试的影响。
        assert len(tools) == 4
        tool_names = [t.name for t in tools]
        assert "search_reports" in tool_names
        assert "get_report_summary" in tool_names
        assert "list_available_reports" in tool_names
        assert "check_rag_relevance" in tool_names
