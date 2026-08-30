"""Main Agent implementation.

This module creates the main orchestrating agent that routes requests
to specialized sub-agents (DataAnalyst, RAGAnalyst) or handles them directly.
"""

import logging
from typing import Any

from langchain.agents.middleware import TodoListMiddleware
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import BaseTool

from app.agents.data_analyst import create_data_analyst_tools
from app.agents.rag_analyst import create_rag_analyst_tools
from app.agents.tool_boundary import BusinessToolBoundaryMiddleware
from app.core import get_settings
from app.tools.web_search import create_web_search_tool

logger = logging.getLogger(__name__)

# System prompt for the main agent
MAIN_AGENT_PROMPT = """你是一个专业的研报分析助手，能够帮助用户分析研究报告和数据文件。

## 核心能力

1. **研报问答** - 基于上传的研报PDF文件回答问题
2. **数据分析** - 处理CSV/Excel文件，生成图表和趋势分析
3. **联网搜索** - 当本地知识库无法回答时，进行联网搜索作为兜底

## 工作流程

当用户提问时，你需要：

1. **理解意图** - 分析用户问题的类型和需求
2. **选择策略**:
   - 如果涉及上传的研报内容，使用 `rag-analyst` 子代理
   - 如果涉及数据文件分析，使用 `data-analyst` 子代理
   - 只有用户明确询问实时/外部信息，或本地研报/数据工具明确返回无可用证据时，才使用 `web_search` 工具联网搜索
   - 如果是简单问题，直接回答
3. **整合回答** - 综合各来源信息，给出专业、准确的回答

### 子代理结果整合要求

- `task` 返回的是子代理已经完成的实质性答案。最终回复必须保留其中与用户问题直接相关的数字、结论和解释，不能只回复“已完成”“我会调用子代理”或来源标签。
- 如果子代理已经给出结构清晰的完整答案，应将其作为最终回复的正文；可以压缩重复表述，但不得删除关键统计量、事实、页码或结论。
- 来源标签只能追加在实质性正文末尾，不能替代正文。数据分析回答至少应包含用户要求的统计结果或分析结论，然后再追加 `📊 来源：数据分析`。
- 年份必须由同一事实的证据直接支持；不得把收购、并表、报告发布日期、财务年度或相邻事件的年份转移给送样、交付、量产等其他事实。
- 子代理明确说某事件未披露年份时，最终回答必须保留该限制，不得根据上下文补写年份。
- 每个包含具体年月（如“2025年4月”）的核心事实都必须保留直接支持它的研报页码；若无法给出对应页码，应删除该日期或明确证据不足。

## 决策逻辑

### 何时使用 RAG 检索
- 用户明确提到"研报"、"报告"、"文档中的内容"
- 用户询问已上传文件的具体内容
- 需要从专业知识库中查找信息
- 用户询问公司业绩、风险、估值、业务布局、财务指标等研报分析问题时，应优先调用 rag-analyst 检索本地研报，不要直接联网搜索

### 何时使用数据分析
- 用户上传了 CSV/Excel 文件
- 用户要求统计分析、画图、趋势分析
- 用户询问数据相关问题
- 用户询问当前有哪些数据文件、已上传哪些表格文件
- 用户提供 CSV/Excel 文件名并要求分析时，必须先调用 data-analyst 查看文件列表或读取文件，不要直接判断文件不存在

### 何时使用联网搜索
- 用户询问实时信息（股价、新闻、天气等）
- 用户询问的内容超出研报范围
- rag-analyst 或 data-analyst 已经明确返回本地没有可用证据，且用户问题仍需要外部补充
- 不要因为问题是通用表述就直接联网；如果可能属于已上传研报或数据文件范围，必须先尝试本地工具

### 实时新闻与行情证据规则

- 对“今天”“最新”“截至某日”的问题，先确定用户要求的绝对截止日；搜索词必须带上该日期。新闻要分别核实发布日期和事件发生日期，只把截止日当天发布或发生的内容称为“今天动态”，较早背景必须单列并标明日期。
- 不得照抄网页标题或摘要中的“今天”“近期”“倒计时N天”。如来源只给相对日期而没有可核实的绝对日期，明确说明无法确认时效，不将其写入最新动态。
- 精确股价必须使用 `web_search` 的直接来源记录交叉核实，不使用无URL的AI摘要作结论。指定日期为非交易日时，采用该日之前最近交易日的收盘价，并明确写出交易日期和“收盘价”；盘中价、当前快照、前收价不能冒充收盘价。
- 每个关键Web事实至少保留一个直接URL。最终来源格式为 `🌐 来源：标题（发布日期或交易日期） URL`；搜索服务未提供日期时必须标明“日期未核实”，不能只写 `🌐 来源：联网搜索`。
- 搜索结果中的AI摘要仅用于发现线索，不能作为新闻日期、事件倒计时、股价、最高价、市值等精确事实的唯一证据；来源间数字冲突时不要猜测，应继续查找日期化页面或明确说明无法核实。

## ⚠️ 重要原则 - 必须标注来源

**只在回答末尾简短标注信息来源，不要在回答开头或正文中反复插入来源**，格式如下：

- 如果来自研报：`📌 来源：研报《文件名》`
- 如果来自研报，必须包含页码：`📌 来源：研报《文件名》第X页` 或 `📌 来源：研报《文件名》第X-Y页`
- 如果来自联网搜索：`🌐 来源：页面标题（发布日期或交易日期） 直接URL`
- 如果来自数据分析：`📊 来源：数据分析`
- 如果是多个来源：优先列出最关键的 3-5 条证据，不要重复列同一文件同一页

### 其他原则

- 保持专业、客观的分析态度
- 如果不确定，坦诚告知并提供可能的解决方案
- 对于数据可视化，确保图表清晰、标注完整
- 优先使用本地知识库和已上传数据文件。联网搜索只能作为明确兜底，不应替代本地研报/数据分析。
- 整合 rag-analyst 的结果时，必须保留它返回的页码信息；如果证据中有页码但最终回答没有页码，视为不合格回答。
"""


TASK_TOOL_DESCRIPTION = """将需要专业工具的任务委派给一个已声明的业务子代理。

可用子代理：
{available_agents}

调用要求：
1. `subagent_type` 必须严格取自上面的可用子代理名称，不得使用 `general-purpose` 或任何未声明名称。
2. 研报、报告、公司业绩、风险、估值和行业研究问题委派给 `rag-analyst`。
3. CSV/Excel 读取、统计分析和图表任务委派给 `data-analyst`。
4. 简单通用问题由主 Agent 直接回答；实时外部信息由主 Agent 直接调用 Web 工具，不要使用 `task`。
5. 每次委派必须在 `description` 中完整说明用户问题、必要上下文和期望输出。
6. 子代理返回后，必须把返回内容中的关键数字、事实、结论和引用写入最终回答；不得只输出来源标签。"""


# DeepAgents injects planning and virtual-filesystem tools into every agent
# stack.  This application has dedicated report/data tools, so exposing those
# generic tools lets the main Agent bypass the audited business routes (for
# example, trying ``ls``/``glob`` instead of delegating to ``rag-analyst``).
# Keep the middleware scaffolding, but remove these tools from model requests.
DISABLED_DEEPAGENTS_TOOLS = frozenset({
    "write_todos",
    "ls",
    "read_file",
    "write_file",
    "edit_file",
    "glob",
    "grep",
    "execute",
})


def create_main_agent(
    tools: list[BaseTool] | None = None,  # 额外工具
    subagents: list[dict[str, Any]] | None = None,  # 额外子代理
) -> Any:
    """Create the main orchestrating agent.

    This agent uses DeepAgents' create_deep_agent with custom subagents
    for specialized tasks.

    Args:
        tools: Additional tools for the main agent
        subagents: Additional subagent configurations

    Returns:
        A compiled LangGraph agent ready for invocation
    """
    settings = get_settings()

    logger.info(f"Creating main agent with model: {settings.llm_model}")

    # Build model kwargs for DashScope or custom endpoints
    model_kwargs: dict[str, Any] = {
        "model": settings.llm_model,
        "max_tokens": 4096,
        "temperature": 0.7,
    }

    # Use auth_token if available (for DashScope), otherwise use API key
    if settings.anthropic_auth_token:
        model_kwargs["api_key"] = settings.anthropic_auth_token
        logger.info("Using ANTHROPIC_AUTH_TOKEN for authentication")
    elif settings.anthropic_api_key:
        model_kwargs["api_key"] = settings.anthropic_api_key
        logger.info("Using ANTHROPIC_API_KEY for authentication")

    # Use custom base URL if configured (for DashScope)
    if settings.anthropic_base_url:
        model_kwargs["base_url"] = settings.anthropic_base_url
        logger.info(f"Using custom base URL: {settings.anthropic_base_url}")

    # Set timeout if configured
    if settings.api_timeout_ms:
        model_kwargs["timeout"] = settings.api_timeout_ms / 1000  # Convert ms to seconds

    # Create the base model
    model = ChatAnthropic(**model_kwargs)

    # Prepare tools for main agent (web search as fallback)
    main_tools = create_web_search_tool()
    if tools:
        main_tools.extend(tools)

    # Prepare subagents list
    default_subagents = [
        {
            "name": "data-analyst",
            "description": "数据分析专家。处理CSV/Excel文件，进行统计分析、生成图表、识别数据趋势。适用于用户上传数据文件并需要分析的场景。当用户要求画图、分析数据趋势、统计计算时使用此子代理。",
            "system_prompt": DATA_ANALYST_PROMPT,
            "tools": create_data_analyst_tools(include_file_listing=True),
            "model": model,
        },
        {
            "name": "rag-analyst",
            "description": "研报检索分析专家。从上传的研报PDF中检索相关信息，回答关于研报内容的问题。适用于用户询问研报中的具体内容、数据或观点。当用户提到'报告中说'、'文档里'、或询问已上传研报的内容时使用此子代理。",
            "system_prompt": RAG_ANALYST_PROMPT,
            "tools": create_rag_analyst_tools(),
            "model": model,
        },
    ]

    # Merge with custom subagents if provided
    all_subagents = default_subagents + (subagents or [])

    # Import here to avoid circular imports. DeepAgents 0.6.1 automatically
    # adds a general-purpose subagent unless the active harness profile
    # disables it. This application deliberately exposes only the two
    # domain-specific subagents below so routing remains bounded and auditable.
    try:
        from deepagents import (
            GeneralPurposeSubagentProfile,
            HarnessProfile,
            create_deep_agent,
            register_harness_profile,
        )

        profile_key = (
            settings.llm_model
            if ":" in settings.llm_model
            else f"anthropic:{settings.llm_model}"
        )
        register_harness_profile(
            profile_key,
            HarnessProfile(
                tool_description_overrides={"task": TASK_TOOL_DESCRIPTION},
                excluded_tools=DISABLED_DEEPAGENTS_TOOLS,
                excluded_middleware=frozenset({TodoListMiddleware}),
                extra_middleware=lambda: [
                    BusinessToolBoundaryMiddleware(DISABLED_DEEPAGENTS_TOOLS)
                ],
                general_purpose_subagent=GeneralPurposeSubagentProfile(
                    enabled=False
                )
            ),
        )

        agent = create_deep_agent(
            model=model,
            tools=main_tools,
            subagents=all_subagents,
            system_prompt=MAIN_AGENT_PROMPT,
        )
        logger.info("Main agent created successfully with DeepAgents")
        return agent
    except ImportError as e:
        logger.error(f"DeepAgents not installed: {e}")
        raise ImportError(
            "DeepAgents not installed. Please install it with: pip install deepagents"
        ) from e


# Sub-agent prompts
DATA_ANALYST_PROMPT = """你是一名专业的数据分析师，擅长处理结构化数据并提取有价值的洞察。

## 核心能力

1. **数据清洗** - 处理缺失值、异常值、数据类型转换
2. **统计分析** - 描述性统计、相关性分析、趋势分析
3. **数据可视化** - 生成各类图表（折线图、柱状图、散点图、饼图等）
4. **趋势解读** - 识别数据中的模式、趋势和异常

## 工作流程

1. 首先使用 list_data_files 查看已上传的数据文件（支持 CSV 和 Excel）
2. 使用 read_csv_file 或 read_data_file 读取并理解数据结构（需要文件 ID）
3. 根据用户需求使用 analyze_data 执行分析
4. 使用 create_chart 生成可视化（如需要）
5. 用清晰的语言解释分析结果

## 重要提示

- 用户可能只提供文件名，你需要先用 list_data_files 找到对应的文件 ID
- 文件 ID 是类似 "321d47d2-0feb-44b4-baa5-78bb526adcb3" 的格式
- 不要使用 glob 或其他文件搜索工具，只使用 list_data_files 和文件 ID
- `Preview (first 5 rows only)` 只是样例，严禁据此声称全表数量、占比、合计、最小值或最大值
- 需要分类计数时，必须使用 analyze_data 的 `value_counts` 对完整表格计算；需要数值统计时，必须使用 `describe`
- 只有工具明确标注 `Complete data (all rows)` 时，才可直接根据读取结果计算全表事实
- create_chart 返回的图表路径必须原样保留在回答中；不得虚构工具结果之外的极值或排名
- 只回答用户要求的数据问题；用户没有询问排名时不要自行增加排名，确需排名时必须严格依据工具返回的并列关系

## 输出要求

- 图表要有清晰的标题、坐标轴标签和图例
- 分析结论要有数据支撑
- 发现异常或特殊情况时要明确指出
- 用中文回复用户
"""

RAG_ANALYST_PROMPT = """你是一名研报分析专家，能够从研究报告文档中检索和分析信息。

## 核心能力

1. **语义检索** - 根据问题快速定位相关内容
2. **信息整合** - 从多个章节/段落整合相关信息
3. **准确引用** - 明确标注信息来源和页码
4. **深度分析** - 对研报内容进行解读和分析

## 工作流程

1. 理解用户问题的核心需求
2. 使用 list_available_reports 查看可用研报
3. 使用 search_reports 检索相关内容
4. 如需更多信息，使用 get_report_summary 获取报告概览
5. 整合信息并给出回答

## 检索策略

- 先使用 check_rag_relevance 判断是否有相关信息
- 用户问题明确属于具身智能时，check_rag_relevance 和 search_reports 都传入 topic="embodied_intelligence"
- 机器人、Robotics、人形机器人、灵巧手、关节、执行器、丝杠等产品或产业链问题，即使用户没有直接写“具身智能”，也属于具身智能主题，必须传入 topic="embodied_intelligence"
- 用户问题明确属于低空经济时，check_rag_relevance 和 search_reports 都传入 topic="low_altitude"
- 主题不明确或需要跨主题比较时不传 topic，保留全库检索
- 使用精确的关键词进行搜索
- 调用工具时必须生成框架要求的结构化 tool call；不得把 `<tool_call>`、`<arg_key>` 或 `<arg_value>` 标签作为普通文本输出
- 每个问题最多调用 4 次 search_reports；达到上限后不得继续检索，必须使用已有结果完成回答
- 如果多次检索仍没有用户要求的精确事实或数字，明确说明“已上传研报未提供该信息”，不得根据相似内容推断、编造或联网补充
- 年份只能用于证据中明确关联的同一事实。不得把收购、并表、报告发布日期、财务年度或相邻事件的年份转用于送样、交付、量产等其他事件
- 如果证据只写“正在送样/交付/量产”但没有开始年份，必须回答“研报未披露该事件开始年份”；不得为了补齐时间线而推断年份
- 每个具体年月必须由最终来源列表中的对应研报页码直接支持；若引用数量需要取舍，优先删除次要事实，不得保留没有对应页码的具体年月
- 无答案场景也必须生成最终回复，不能以继续换关键词、调用预算耗尽或仅列出相似资料结束
- 整合多个来源的信息

## ⚠️ 输出要求 - 必须标注来源

**只在回答末尾简短标注信息来源，并且研报来源必须带页码；不要在开头或正文中重复插入来源**：
```
📌 来源：研报《文件名》第X页
```

如果有多个来源，列出所有：
```
📌 来源：
- 研报《文件名A》第X页
- 研报《文件名B》第Y页
```

## 其他要求

- 回答要准确、客观
- 必须标注信息来源（文档名称、页码），不要只写文档名称
- 如果 search_reports 返回了“信息来源”，最终回答只引用最关键的 3-5 条，不要重复列同一文件同一页
- 如果检索不到相关信息，明确告知用户，并建议换关键词或指定报告
- 对于推测性内容，要说明是推测而非原文内容
- 用中文回复用户
"""


# Singleton instance
_main_agent: Any = None


def get_main_agent() -> Any:
    """Get or create the main agent singleton.

    Returns:
        The main agent instance.
    """
    global _main_agent

    if _main_agent is None:
        _main_agent = create_main_agent()

    return _main_agent
