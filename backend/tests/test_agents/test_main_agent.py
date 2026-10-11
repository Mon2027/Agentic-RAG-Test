"""主 Agent 的提示词、模型参数、工具和子 Agent 配置测试。

这些测试不调用真实大模型，而是通过 patch 替换配置、ChatAnthropic、工具工厂和
``deepagents.create_deep_agent``，再检查主 Agent 创建过程中传递的参数。
因此这里验证的是“创建时如何组装依赖”，不是大模型实际回答的质量。

重点学习内容：

* 提示词中的关键路由规则如何通过字符串断言防止意外删除；
* 多层 ``@patch`` 装饰器如何从下往上应用，并按相反视觉顺序注入 mock 参数；
* 如何用 ``patch.dict(sys.modules)`` 模拟可选第三方模块；
* 主 Agent、data-analyst 和 rag-analyst 各自应该看到哪些工具；
* 如何验证全局单例只初始化一次。
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class TestMainAgentPrompt:
    """主 Agent 与两个业务子 Agent 的提示词契约测试。"""

    def test_main_agent_prompt_content(self):
        """主提示词必须保留任务路由、来源标注和实时行情约束。"""
        from app.agents.main_agent import MAIN_AGENT_PROMPT

        # 这些是行为契约而非文案快照：只检查关键规则存在，允许其他措辞继续演进。
        assert "研报问答" in MAIN_AGENT_PROMPT
        assert "数据分析" in MAIN_AGENT_PROMPT
        assert "联网搜索" in MAIN_AGENT_PROMPT
        assert "决策逻辑" in MAIN_AGENT_PROMPT
        assert "来源标签只能追加在实质性正文末尾" in MAIN_AGENT_PROMPT
        assert "不能只回复" in MAIN_AGENT_PROMPT
        assert "最近交易日的收盘价" in MAIN_AGENT_PROMPT
        assert "不得照抄网页标题或摘要中的" in MAIN_AGENT_PROMPT
        assert "不能只写 `🌐 来源：联网搜索`" in MAIN_AGENT_PROMPT

    def test_data_analyst_prompt_content(self):
        """数据分析子 Agent 提示词应覆盖清洗、统计、可视化及必需工具。"""
        from app.agents.main_agent import DATA_ANALYST_PROMPT

        # 对完整提示词做成员检查：缺少任一能力或工具名称时，对应断言会指出问题。
        assert "数据清洗" in DATA_ANALYST_PROMPT
        assert "统计分析" in DATA_ANALYST_PROMPT
        assert "数据可视化" in DATA_ANALYST_PROMPT
        assert "read_csv_file" in DATA_ANALYST_PROMPT

    def test_rag_analyst_prompt_content(self):
        """RAG 子 Agent 提示词应包含检索预算、主题和证据年份/页码规则。"""
        from app.agents.main_agent import RAG_ANALYST_PROMPT

        # 这些断言只检查规则是否写入提示词，不会调用模型判断规则执行效果。
        assert "语义检索" in RAG_ANALYST_PROMPT
        assert "信息整合" in RAG_ANALYST_PROMPT
        assert "search_reports" in RAG_ANALYST_PROMPT
        assert "check_rag_relevance" in RAG_ANALYST_PROMPT
        assert "最多调用 4 次 search_reports" in RAG_ANALYST_PROMPT
        assert "已上传研报未提供该信息" in RAG_ANALYST_PROMPT
        assert 'topic="embodied_intelligence"' in RAG_ANALYST_PROMPT
        assert "不得把收购、并表、报告发布日期、财务年度或相邻事件的年份" in RAG_ANALYST_PROMPT
        assert "研报未披露该事件开始年份" in RAG_ANALYST_PROMPT
        assert "每个具体年月必须由最终来源列表中的对应研报页码直接支持" in RAG_ANALYST_PROMPT


class TestCreateMainAgent:
    """create_main_agent 的依赖组装和扩展参数测试。"""

    # patch 从最靠近函数的 get_settings 开始应用，因此 mock 参数顺序为：
    # settings、ChatAnthropic、RAG 工具、数据工具、Web 工具。
    @patch("app.agents.main_agent.create_web_search_tool")
    @patch("app.agents.main_agent.create_data_analyst_tools")
    @patch("app.agents.main_agent.create_rag_analyst_tools")
    @patch("app.agents.main_agent.ChatAnthropic")
    @patch("app.agents.main_agent.get_settings")
    def test_disables_deepagents_general_purpose_subagent(
        self,
        mock_get_settings,
        mock_chat_anthropic,
        mock_rag_tools,
        mock_data_tools,
        mock_web_tools,
    ):
        """只能路由到两个显式业务子 Agent，必须关闭通用子 Agent。"""
        # Arrange：构造创建模型所需的最小配置，并让所有工具工厂返回空列表。
        mock_settings = MagicMock()
        mock_settings.llm_model = "glm-5"
        mock_settings.anthropic_api_key = "test_api_key"
        mock_settings.anthropic_auth_token = None
        mock_settings.anthropic_base_url = None
        mock_settings.api_timeout_ms = None
        mock_get_settings.return_value = mock_settings
        mock_chat_anthropic.return_value = MagicMock()
        mock_web_tools.return_value = []
        mock_data_tools.return_value = []
        mock_rag_tools.return_value = []

        # deepagents 是可选依赖；用具备目标属性的 mock 模块完整模拟其配置 API。
        mock_create_deep_agent = MagicMock(return_value=MagicMock())
        mock_register_harness_profile = MagicMock()
        mock_deepagents = MagicMock(
            create_deep_agent=mock_create_deep_agent,
            register_harness_profile=mock_register_harness_profile,
        )
        # side_effect 让“构造函数 mock”接收实参并返回可读取字段的简单对象。
        # 若只用默认 MagicMock 返回值，后面的 enabled/excluded_tools 断言就
        # 无法核对创建代码实际传入的配置。
        mock_deepagents.GeneralPurposeSubagentProfile.side_effect = (
            lambda *, enabled: SimpleNamespace(enabled=enabled)
        )
        mock_deepagents.HarnessProfile.side_effect = (
            lambda *, tool_description_overrides, excluded_tools, excluded_middleware,
            extra_middleware, general_purpose_subagent: SimpleNamespace(
                tool_description_overrides=tool_description_overrides,
                excluded_tools=excluded_tools,
                excluded_middleware=excluded_middleware,
                extra_middleware=extra_middleware,
                general_purpose_subagent=general_purpose_subagent,
            )
        )

        # patch sys.modules 后，函数内部的 ``import deepagents`` 会取得这个测试模块。
        with patch.dict("sys.modules", {"deepagents": mock_deepagents}):
            from app.agents.main_agent import create_main_agent

            create_main_agent()

        # call_args.args 保存最后一次调用的位置参数，这里解包出注册键与配置对象。
        # 验证 harness profile 关闭 general-purpose，并隐藏文件系统/执行类工具和中间件。
        mock_register_harness_profile.assert_called_once()
        profile_key, profile = mock_register_harness_profile.call_args.args
        assert profile_key == "anthropic:glm-5"
        assert profile.general_purpose_subagent.enabled is False
        task_description = profile.tool_description_overrides["task"]
        assert "{available_agents}" in task_description
        assert "不得使用 `general-purpose`" in task_description
        # frozenset 比较不依赖元素顺序，且会发现多排除或少排除的工具。
        assert profile.excluded_tools == frozenset({
            "write_todos",
            "ls",
            "read_file",
            "write_file",
            "edit_file",
            "glob",
            "grep",
            "execute",
        })
        # task 负责路由子 Agent；联网搜索也应继续可用。
        assert "task" not in profile.excluded_tools
        assert "web_search" not in profile.excluded_tools

        from langchain.agents.middleware import TodoListMiddleware

        from app.agents.tool_boundary import BusinessToolBoundaryMiddleware

        assert profile.excluded_middleware == frozenset({TodoListMiddleware})
        # extra_middleware 是返回中间件列表的工厂，因此要先调用再检查结果。
        boundary_middleware = profile.extra_middleware()
        assert len(boundary_middleware) == 1
        assert isinstance(boundary_middleware[0], BusinessToolBoundaryMiddleware)
        assert boundary_middleware[0].disabled_tools == profile.excluded_tools

        # call_args.kwargs 保存最后一次调用的关键字参数；取出其中的子 Agent 列表。
        # 这里比较整个名称列表，同时验证数量和默认角色的排列顺序。
        subagent_names = [
            subagent["name"]
            for subagent in mock_create_deep_agent.call_args.kwargs["subagents"]
        ]
        assert subagent_names == ["data-analyst", "rag-analyst"]

    @patch("app.agents.main_agent.create_web_search_tool")
    @patch("app.agents.main_agent.create_data_analyst_tools")
    @patch("app.agents.main_agent.create_rag_analyst_tools")
    @patch("app.agents.main_agent.ChatAnthropic")
    @patch("app.agents.main_agent.get_settings")
    def test_create_main_agent_success(
        self,
        mock_get_settings,
        mock_chat_anthropic,
        mock_rag_tools,
        mock_data_tools,
        mock_web_tools,
    ):
        """默认配置应正确创建模型、业务工具和 DeepAgent。"""
        # Arrange：模拟应用配置，隔离真实 API 密钥和网络模型初始化。
        mock_settings = MagicMock()
        mock_settings.llm_model = "claude-3-sonnet-20240229"
        mock_settings.anthropic_api_key = "test_api_key"
        mock_settings.anthropic_auth_token = None
        mock_settings.anthropic_base_url = None
        mock_settings.api_timeout_ms = None
        mock_get_settings.return_value = mock_settings

        mock_model = MagicMock()
        mock_chat_anthropic.return_value = mock_model

        mock_web_tools.return_value = [MagicMock(name="web_search")]
        mock_data_tools.return_value = [MagicMock(name="read_csv_file")]
        mock_rag_tools.return_value = [MagicMock(name="search_reports")]

        # 模拟 deepagents 模块，并记录 create_deep_agent 的构造参数。
        mock_agent = MagicMock()
        mock_create_deep_agent = MagicMock(return_value=mock_agent)
        mock_deepagents = MagicMock(create_deep_agent=mock_create_deep_agent)

        with patch.dict("sys.modules", {"deepagents": mock_deepagents}):
            from app.agents.main_agent import create_main_agent

            create_main_agent()

        # Assert：检查模型名称、最大输出 token 和温度等关键推理配置。
        mock_chat_anthropic.assert_called_once()
        call_kwargs = mock_chat_anthropic.call_args.kwargs
        assert call_kwargs["model"] == "claude-3-sonnet-20240229"
        assert call_kwargs["max_tokens"] == 4096
        assert call_kwargs["temperature"] == 0.7

        # 这里只核对工厂调用次数；具体子 Agent 配置由下方独立用例检查。
        mock_create_deep_agent.assert_called_once()

    @patch("app.agents.main_agent.create_web_search_tool")
    @patch("app.agents.main_agent.create_data_analyst_tools")
    @patch("app.agents.main_agent.create_rag_analyst_tools")
    @patch("app.agents.main_agent.ChatAnthropic")
    @patch("app.agents.main_agent.get_settings")
    def test_create_main_agent_with_custom_base_url(
        self,
        mock_get_settings,
        mock_chat_anthropic,
        mock_rag_tools,
        mock_data_tools,
        mock_web_tools,
    ):
        """自定义 Anthropic 兼容地址和毫秒超时应正确转换给模型客户端。"""
        mock_settings = MagicMock()
        mock_settings.llm_model = "claude-3-sonnet-20240229"
        mock_settings.anthropic_api_key = None
        mock_settings.anthropic_auth_token = "test_token"
        mock_settings.anthropic_base_url = "https://custom-api.example.com"
        mock_settings.api_timeout_ms = 30000
        mock_get_settings.return_value = mock_settings

        mock_model = MagicMock()
        mock_chat_anthropic.return_value = mock_model

        mock_web_tools.return_value = []
        mock_data_tools.return_value = []
        mock_rag_tools.return_value = []

        mock_agent = MagicMock()
        mock_create_deep_agent = MagicMock(return_value=mock_agent)
        mock_deepagents = MagicMock(create_deep_agent=mock_create_deep_agent)

        with patch.dict("sys.modules", {"deepagents": mock_deepagents}):
            from app.agents.main_agent import create_main_agent

            create_main_agent()

        # 配置值采用毫秒，模型客户端的 timeout 参数采用秒：30000 / 1000 = 30.0。
        call_kwargs = mock_chat_anthropic.call_args.kwargs
        assert call_kwargs["base_url"] == "https://custom-api.example.com"
        assert call_kwargs["timeout"] == 30.0

    @patch("app.agents.main_agent.create_web_search_tool")
    @patch("app.agents.main_agent.create_data_analyst_tools")
    @patch("app.agents.main_agent.create_rag_analyst_tools")
    @patch("app.agents.main_agent.ChatAnthropic")
    @patch("app.agents.main_agent.get_settings")
    def test_create_main_agent_with_additional_tools(
        self,
        mock_get_settings,
        mock_chat_anthropic,
        mock_rag_tools,
        mock_data_tools,
        mock_web_tools,
    ):
        """调用方传入的额外工具应与默认 Web 工具合并，而不是覆盖。"""
        mock_settings = MagicMock()
        mock_settings.llm_model = "claude-3-sonnet-20240229"
        mock_settings.anthropic_api_key = "test_key"
        mock_settings.anthropic_auth_token = None
        mock_settings.anthropic_base_url = None
        mock_settings.api_timeout_ms = None
        mock_get_settings.return_value = mock_settings

        mock_model = MagicMock()
        mock_chat_anthropic.return_value = mock_model

        mock_web_tools.return_value = [MagicMock(name="web_search")]
        mock_data_tools.return_value = []
        mock_rag_tools.return_value = []

        mock_agent = MagicMock()
        mock_create_deep_agent = MagicMock(return_value=mock_agent)
        mock_deepagents = MagicMock(create_deep_agent=mock_create_deep_agent)

        with patch.dict("sys.modules", {"deepagents": mock_deepagents}):
            from app.agents.main_agent import create_main_agent

            # 额外工具代表业务方在默认配置之外进行的可插拔扩展。
            additional_tool = MagicMock(name="custom_tool")
            create_main_agent(tools=[additional_tool])

        # 工厂收到的 tools 数量应为默认 Web 工具 1 个加上自定义工具 1 个。
        # 此处只验证数量，未逐一验证列表中对象的身份。
        call_kwargs = mock_create_deep_agent.call_args.kwargs
        assert len(call_kwargs["tools"]) == 2

    @patch("app.agents.main_agent.create_web_search_tool")
    @patch("app.agents.main_agent.create_data_analyst_tools")
    @patch("app.agents.main_agent.create_rag_analyst_tools")
    @patch("app.agents.main_agent.ChatAnthropic")
    @patch("app.agents.main_agent.get_settings")
    def test_create_main_agent_with_additional_subagents(
        self,
        mock_get_settings,
        mock_chat_anthropic,
        mock_rag_tools,
        mock_data_tools,
        mock_web_tools,
    ):
        """调用方传入的额外子 Agent 应追加到两个默认业务子 Agent 后。"""
        mock_settings = MagicMock()
        mock_settings.llm_model = "claude-3-sonnet-20240229"
        mock_settings.anthropic_api_key = "test_key"
        mock_settings.anthropic_auth_token = None
        mock_settings.anthropic_base_url = None
        mock_settings.api_timeout_ms = None
        mock_get_settings.return_value = mock_settings

        mock_model = MagicMock()
        mock_chat_anthropic.return_value = mock_model

        mock_web_tools.return_value = []
        mock_data_tools.return_value = []
        mock_rag_tools.return_value = []

        mock_agent = MagicMock()
        mock_create_deep_agent = MagicMock(return_value=mock_agent)
        mock_deepagents = MagicMock(create_deep_agent=mock_create_deep_agent)

        with patch.dict("sys.modules", {"deepagents": mock_deepagents}):
            from app.agents.main_agent import create_main_agent

            # 子 Agent 配置是包含名称、描述、提示词、工具和模型的字典。
            additional_subagent = {
                "name": "custom-agent",
                "description": "Custom subagent",
                "system_prompt": "Custom prompt",
                "tools": [],
                "model": mock_model,
            }
            create_main_agent(subagents=[additional_subagent])

        # 默认 2 个 + 自定义 1 个，共 3 个；这里只验证数量，不检查排列顺序。
        call_kwargs = mock_create_deep_agent.call_args.kwargs
        assert len(call_kwargs["subagents"]) == 3


class TestGetMainAgent:
    """get_main_agent 模块级单例行为测试。"""

    @patch("app.agents.main_agent.create_main_agent")
    def test_get_main_agent_singleton(self, mock_create):
        """连续调用应返回同一实例，并且底层创建函数只执行一次。"""
        mock_agent = MagicMock()
        mock_create.return_value = mock_agent

        # 单例是模块全局状态；用例前必须重置，避免受其他测试调用顺序影响。
        import app.agents.main_agent as main_agent_module
        from app.agents.main_agent import get_main_agent
        main_agent_module._main_agent = None

        agent1 = get_main_agent()
        agent2 = get_main_agent()

        # is 比较是否为同一个对象；两个不同但内容相等的对象不能通过这个断言。
        assert agent1 is agent2
        # 第二次 get_main_agent 应直接复用缓存。
        mock_create.assert_called_once()

    @patch("app.agents.main_agent.create_main_agent")
    def test_get_main_agent_creates_new_if_none(self, mock_create):
        """单例缓存为空时，应创建并返回一个新 Agent。"""
        mock_agent = MagicMock()
        mock_create.return_value = mock_agent

        # 显式制造“首次调用”条件。
        import app.agents.main_agent as main_agent_module
        from app.agents.main_agent import get_main_agent
        main_agent_module._main_agent = None

        agent = get_main_agent()

        assert agent is mock_agent
        mock_create.assert_called_once()


class TestSubagentConfiguration:
    """两个业务子 Agent 的名称、描述和工具隔离测试。"""

    @patch("app.agents.main_agent.create_web_search_tool")
    @patch("app.agents.main_agent.create_data_analyst_tools")
    @patch("app.agents.main_agent.create_rag_analyst_tools")
    @patch("app.agents.main_agent.ChatAnthropic")
    @patch("app.agents.main_agent.get_settings")
    def test_data_analyst_subagent_config(
        self,
        mock_get_settings,
        mock_chat_anthropic,
        mock_rag_tools,
        mock_data_tools,
        mock_web_tools,
    ):
        """data-analyst 应获得数据分析工具和正确业务描述。"""
        mock_settings = MagicMock()
        mock_settings.llm_model = "claude-3-sonnet-20240229"
        mock_settings.anthropic_api_key = "test_key"
        mock_settings.anthropic_auth_token = None
        mock_settings.anthropic_base_url = None
        mock_settings.api_timeout_ms = None
        mock_get_settings.return_value = mock_settings

        mock_model = MagicMock()
        mock_chat_anthropic.return_value = mock_model

        mock_tool = MagicMock(name="read_csv_file")
        mock_web_tools.return_value = []
        mock_data_tools.return_value = [mock_tool]
        mock_rag_tools.return_value = []

        mock_agent = MagicMock()
        mock_create_deep_agent = MagicMock(return_value=mock_agent)
        mock_deepagents = MagicMock(create_deep_agent=mock_create_deep_agent)

        with patch.dict("sys.modules", {"deepagents": mock_deepagents}):
            from app.agents.main_agent import create_main_agent

            create_main_agent()

        call_kwargs = mock_create_deep_agent.call_args.kwargs
        subagents = call_kwargs["subagents"]

        # next 取生成器中的第一个匹配项；若找不到该名称，会抛出 StopIteration，
        # 测试因此失败。不依赖列表位置，新增其他子 Agent 不会影响本用例。
        data_analyst = next(s for s in subagents if s["name"] == "data-analyst")

        assert data_analyst["name"] == "data-analyst"
        assert "数据分析" in data_analyst["description"]
        assert data_analyst["tools"] == [mock_tool]

    @patch("app.agents.main_agent.create_web_search_tool")
    @patch("app.agents.main_agent.create_data_analyst_tools")
    @patch("app.agents.main_agent.create_rag_analyst_tools")
    @patch("app.agents.main_agent.ChatAnthropic")
    @patch("app.agents.main_agent.get_settings")
    def test_rag_analyst_subagent_config(
        self,
        mock_get_settings,
        mock_chat_anthropic,
        mock_rag_tools,
        mock_data_tools,
        mock_web_tools,
    ):
        """rag-analyst 应获得 RAG 工具和正确业务描述。"""
        mock_settings = MagicMock()
        mock_settings.llm_model = "claude-3-sonnet-20240229"
        mock_settings.anthropic_api_key = "test_key"
        mock_settings.anthropic_auth_token = None
        mock_settings.anthropic_base_url = None
        mock_settings.api_timeout_ms = None
        mock_get_settings.return_value = mock_settings

        mock_model = MagicMock()
        mock_chat_anthropic.return_value = mock_model

        mock_tool = MagicMock(name="search_reports")
        mock_web_tools.return_value = []
        mock_data_tools.return_value = []
        mock_rag_tools.return_value = [mock_tool]

        mock_agent = MagicMock()
        mock_create_deep_agent = MagicMock(return_value=mock_agent)
        mock_deepagents = MagicMock(create_deep_agent=mock_create_deep_agent)

        with patch.dict("sys.modules", {"deepagents": mock_deepagents}):
            from app.agents.main_agent import create_main_agent

            create_main_agent()

        call_kwargs = mock_create_deep_agent.call_args.kwargs
        subagents = call_kwargs["subagents"]

        # 同样按 name 查找，验证工具没有错误分配给另一个子 Agent。
        rag_analyst = next(s for s in subagents if s["name"] == "rag-analyst")

        assert rag_analyst["name"] == "rag-analyst"
        assert "研报检索" in rag_analyst["description"]
        assert rag_analyst["tools"] == [mock_tool]
