"""Tavily Web 搜索工具的数据模型、HTTP 调用和证据格式测试。

本文件不访问真实 Tavily 服务，而是分层使用 mock：

* 数据模型测试验证单条结果和完整响应的默认字段；
* ``_tavily_search`` 测试 mock ``httpx.Client``，验证底层 HTTP 适配逻辑；
* ``web_search``/``web_search_quick`` 测试 mock ``_tavily_search``，聚焦工具输出；
* 行情测试验证带截止日查询会补充“历史行情/收盘价/交易日期”语义，并在首轮缺少
  带日期历史表时进行一次更窄的补查。

核心安全契约是：即使 Tavily 返回 AI 摘要，工具也必须保留直接 URL、发布日期、
正文日期候选和用户截止日，不能把不可追溯摘要当作精确日期或行情数字的唯一证据。
"""

import pytest
from unittest.mock import MagicMock, patch


class TestTavilySearchResult:
    """Tavily 单条搜索结果 Pydantic 模型测试。"""

    def test_search_result_creation(self):
        """必填字段应原样保存，未提供发布日期时默认为 None。"""
        from app.tools.web_search import TavilySearchResult

        result = TavilySearchResult(
            title="Test Title",
            url="https://example.com",
            content="Test content",
            score=0.95,
        )

        # 逐字段断言同时验证 Pydantic 没有意外转换 URL、正文或浮点分数。
        assert result.title == "Test Title"
        assert result.url == "https://example.com"
        assert result.content == "Test content"
        assert result.score == 0.95
        assert result.published_date is None


class TestTavilySearchResponse:
    """Tavily 完整响应模型及可选 AI 摘要测试。"""

    def test_response_with_answer(self):
        """包含 AI 摘要时，应同时保留原查询和直接来源结果。"""
        from app.tools.web_search import TavilySearchResponse, TavilySearchResult

        response = TavilySearchResponse(
            query="test query",
            results=[
                TavilySearchResult(
                    title="Result 1",
                    url="https://example.com",
                    content="Content 1",
                    score=0.9,
                )
            ],
            answer="AI generated answer",
        )

        assert response.query == "test query"
        assert len(response.results) == 1
        assert response.answer == "AI generated answer"

    def test_response_without_answer(self):
        """服务未返回 AI 摘要时，answer 应保持可选的 None。"""
        from app.tools.web_search import TavilySearchResponse

        response = TavilySearchResponse(
            query="test query",
            results=[],
        )

        assert response.answer is None


class TestTavilySearch:
    """底层 _tavily_search HTTP 适配器测试。"""

    # 两层 patch 从下往上应用，所以测试参数依次为 settings mock、Client mock。
    @patch("httpx.Client")
    @patch("app.tools.web_search.get_settings")
    def test_tavily_search_success(self, mock_get_settings, mock_client):
        """Tavily 请求成功时，应检查 HTTP 状态并返回反序列化 JSON。"""
        # Arrange：提供测试 API Key，阻断真实配置和环境依赖。
        mock_settings = MagicMock()
        mock_settings.tavily_api_key = "test_api_key"
        mock_get_settings.return_value = mock_settings

        # mock_response 模拟 httpx.Response 的 json() 与 raise_for_status() 接口。
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"title": "Test", "url": "https://example.com", "content": "Content", "score": 0.9}
            ],
            "answer": "Test answer",
        }
        mock_response.raise_for_status = MagicMock()

        # 生产代码使用 ``with httpx.Client(...) as client``，因此 mock 必须实现
        # __enter__/__exit__，并让 post 返回上面的响应对象。
        mock_http_client = MagicMock()
        mock_http_client.__enter__ = MagicMock(return_value=mock_http_client)
        mock_http_client.__exit__ = MagicMock(return_value=False)
        mock_http_client.post.return_value = mock_response
        mock_client.return_value = mock_http_client

        from app.tools.web_search import _tavily_search

        # Act：底层函数返回原始字典，证据格式化由更上层工具负责。
        result = _tavily_search("test query", max_results=5)

        assert "results" in result
        assert len(result["results"]) == 1
        assert result["answer"] == "Test answer"

    @patch("app.tools.web_search.get_settings")
    def test_tavily_search_no_api_key(self, mock_get_settings):
        """环境变量和配置均无 API Key 时，应在发请求前抛出明确 ValueError。"""
        import os

        # 暂存并移除环境变量，避免开发者机器或 CI 的真实 Key 让此用例误走成功路径。
        old_val = os.environ.pop("TAVILY_API_KEY", None)

        mock_settings = MagicMock()
        mock_settings.tavily_api_key = None
        mock_get_settings.return_value = mock_settings

        from app.tools.web_search import _tavily_search

        with pytest.raises(ValueError, match="TAVILY_API_KEY not configured"):
            _tavily_search("test query")

        # 用例结束前恢复原值，避免改变同一 pytest 进程内后续测试的环境。
        if old_val:
            os.environ["TAVILY_API_KEY"] = old_val


class TestWebSearch:
    """完整 web_search LangChain 工具的结果、空结果和异常测试。"""

    @patch("app.tools.web_search._tavily_search")
    def test_web_search_with_results(self, mock_tavily):
        """有结果和 AI 摘要时，应输出可追溯来源记录及证据使用警告。"""
        # 这里 mock 的是内部搜索函数，所以测试只关注工具层格式化和文本契约。
        mock_tavily.return_value = {
            "results": [
                {
                    "title": "AI Development",
                    "url": "https://example.com/ai",
                    "content": "AI is growing rapidly",
                    "score": 0.95,
                },
                {
                    "title": "Machine Learning",
                    "url": "https://example.com/ml",
                    "content": "ML is a subset of AI",
                    "score": 0.88,
                },
            ],
            "answer": "AI stands for Artificial Intelligence.",
        }

        from app.tools.web_search import web_search

        # @tool 包装后使用 invoke(dict) 调用，字典会经过工具参数 schema 校验。
        result = web_search.invoke({"query": "what is AI"})

        # AI 摘要必须明确标成未核验，并与标题、URL、日期等直接来源并存。
        assert "搜索服务 AI 摘要（未核验）" in result
        assert "AI stands for Artificial Intelligence" in result
        assert "来源记录" in result
        assert "AI Development" in result
        assert "Machine Learning" in result
        assert "URL：https://example.com/ai" in result
        assert "发布日期：搜索服务未提供" in result
        assert "AI摘要不能作为精确日期" in result

    @patch("app.tools.web_search._tavily_search")
    def test_web_search_no_results(self, mock_tavily):
        """结果与摘要均为空时，应返回友好的未找到提示。"""
        mock_tavily.return_value = {
            "results": [],
            "answer": None,
        }

        from app.tools.web_search import web_search

        result = web_search.invoke({"query": "nonexistent topic xyz123"})

        assert "未找到" in result

    @patch("app.tools.web_search._tavily_search")
    def test_web_search_with_error(self, mock_tavily):
        """配置异常应被工具捕获并转成 Agent 可读文本，而不是向上抛出。"""
        # side_effect 让内部调用稳定抛出与缺少 API Key 相同的配置错误。
        mock_tavily.side_effect = ValueError("API key not configured")

        from app.tools.web_search import web_search

        result = web_search.invoke({"query": "test query"})

        assert "配置错误" in result or "API key" in result


class TestWebSearchQuick:
    """快速搜索、来源保留和行情历史表补查测试。"""

    @patch("app.tools.web_search._tavily_search")
    def test_quick_search_with_answer(self, mock_tavily):
        """快速搜索只有 AI 摘要时，仍应返回摘要内容。"""
        mock_tavily.return_value = {
            "results": [],
            "answer": "Paris is the capital of France.",
        }

        from app.tools.web_search import web_search_quick

        result = web_search_quick.invoke({"query": "capital of France"})

        assert "Paris" in result

    @patch("app.tools.web_search._tavily_search")
    def test_quick_search_without_answer(self, mock_tavily):
        """没有 AI 摘要时，应回退到直接搜索结果并保留 URL。"""
        mock_tavily.return_value = {
            "results": [
                {
                    "title": "France",
                    "url": "https://example.com",
                    "content": "Paris is the capital city of France.",
                    "score": 0.9,
                }
            ],
            "answer": None,
        }

        from app.tools.web_search import web_search_quick

        result = web_search_quick.invoke({"query": "capital of France"})

        assert "Paris" in result
        assert "URL：https://example.com" in result

    @patch("app.tools.web_search._tavily_search")
    def test_quick_search_keeps_url_and_dates_when_answer_exists(self, mock_tavily):
        """即使已有摘要，快速查询也不能丢失 URL、发布日期和正文日期候选。"""
        # published_date 是来源元数据；content 内的 2026-08-07 只是“日期候选”，
        # 两者语义不同，格式化输出必须分别展示。
        mock_tavily.return_value = {
            "results": [{
                "title": "Historical quote",
                "url": "https://example.com/history",
                "content": "2026-08-07 close 14.20",
                "published_date": "2026-08-08",
                "score": 0.95,
            }],
            "answer": "14.20",
        }

        from app.tools.web_search import web_search_quick

        result = web_search_quick.invoke({
            "query": "截至2026年8月8日的最新股价"
        })

        # 原始查询中的截止日也必须保留，供上层判断来源是否晚于用户时间边界。
        assert "URL：https://example.com/history" in result
        assert "发布日期：2026-08-08" in result
        assert "内容日期候选：2026-08-07" in result
        assert "用户截止日：2026-08-08" in result
        assert "搜索服务 AI 摘要（未核验）" in result

    @patch("app.tools.web_search._tavily_search")
    def test_finance_search_enriches_query_with_historical_semantics(self, mock_tavily):
        """带截止日的股价查询应自动补充历史收盘价和交易日期检索词。"""
        mock_tavily.return_value = {"results": [], "answer": None}

        from app.tools.web_search import web_search

        web_search.invoke({"query": "奥普特截至2026年8月8日最新股价"})

        # 从 mock 调用历史读取真正发给 Tavily 的 effective query，而非用户原始 query。
        sent_query = mock_tavily.call_args_list[0].args[0]
        assert "2026年8月8日" in sent_query
        assert "历史行情" in sent_query
        assert "收盘价" in sent_query
        assert "交易日期" in sent_query

    @patch("app.tools.web_search._tavily_search")
    def test_finance_search_retries_for_dated_history_table(self, mock_tavily):
        """首轮缺少带日期行情表时，应进行一次更窄的历史数据补查。"""
        # side_effect 列表让连续两次调用依次返回“实时快照”和“历史行情表”。
        mock_tavily.side_effect = [
            {
                "results": [{
                    "title": "Live quote",
                    "url": "https://example.com/live",
                    "content": "Current quote 118.14",
                    "score": 0.9,
                }],
                "answer": "118.14",
            },
            {
                "results": [{
                    "title": "Historical data",
                    "url": "https://example.com/history",
                    "content": (
                        "| 日期 | 收盘 | 开盘 | 高 | 低 | 交易量 | 涨跌幅 |\n"
                        "| 2026年08月07日 | 120.00 | 117.50 | 120.68 | "
                        "114.80 | 1.75M | +3.47% |"
                    ),
                    "score": 0.95,
                }],
                "answer": "120.00",
            },
        ]

        from app.tools.web_search import web_search

        result = web_search.invoke({
            "query": "奥普特截至2026年8月8日最新股价"
        })

        # 只允许一次补查，避免搜索循环失控和额外 API 成本。
        assert mock_tavily.call_count == 2
        retry_query = mock_tavily.call_args_list[1].args[0]
        assert "2026年8月7日" in retry_query
        assert "股票历史数据表" in retry_query
        # 合并结果时补查证据优先；同时按 URL 去重，因此历史表来源排在实时快照前。
        assert result.index("https://example.com/history") < result.index(
            "https://example.com/live"
        )

    @patch("app.tools.web_search._tavily_search")
    def test_quick_search_empty(self, mock_tavily):
        """快速搜索没有结果和摘要时，应返回未找到提示。"""
        mock_tavily.return_value = {
            "results": [],
            "answer": None,
        }

        from app.tools.web_search import web_search_quick

        result = web_search_quick.invoke({"query": "nonexistent"})

        assert "未找到" in result


class TestCreateWebSearchTool:
    """Web 搜索工具工厂注册清单测试。"""

    def test_creates_tools(self):
        """默认应同时注册完整搜索和快速搜索两个工具。"""
        from app.tools.web_search import create_web_search_tool

        tools = create_web_search_tool()

        assert len(tools) == 2
        # Agent 根据 tool.name 识别工具，因此按名称验证比比较函数对象更贴近运行时。
        tool_names = [t.name for t in tools]
        assert "web_search" in tool_names
        assert "web_search_quick" in tool_names
