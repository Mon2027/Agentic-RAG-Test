# 项目规划：基于 DeepAgents 架构与 Agentic RAG 的多智能体协同研报分析系统

## 一、项目概述

### 1.1 项目目标
构建一个智能研报分析系统，能够：
- 接收并分析 PDF 格式的研报文件
- 接收 CSV/Excel 数据文件进行数据分析
- 智能回答用户问题，自动判断是否需要检索研报内容
- 兜底机制：RAG 检索不到答案时进行联网搜索

### 1.2 技术栈
| 层级 | 技术选型 |
|------|----------|
| 前端 | Vue 3 + TypeScript + TailwindCSS |
| 后端框架 | FastAPI |
| Agent 框架 | DeepAgents (基于 LangGraph) |
| LLM | Claude (Anthropic) |
| Embedding | BAAI/bge-m3 |
| 向量数据库 | ChromaDB |
| PDF 解析 | pypdf / pdfplumber |

---

## 二、系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Vue 3)                         │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │   Chat UI    │  │ File Upload  │  │   Chart Display      │   │
│  │   对话界面    │  │  文件上传     │  │    图表展示          │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP / SSE (Server-Sent Events)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Backend (FastAPI)                           │
│                                                                   │
│  /api/chat          主对话入口，支持流式响应                       │
│  /api/chat/stream   SSE 流式对话                                  │
│  /api/upload/report 上传研报 PDF                                  │
│  /api/upload/data   上传数据文件 (CSV/Excel)                      │
│  /api/documents     文档管理 CRUD                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Main Agent (协调器)                           │
│              基于 DeepAgents create_deep_agent                   │
│                                                                   │
│  决策逻辑:                                                        │
│  1. 理解用户意图                                                  │
│  2. 判断是否需要调用子代理                                         │
│  3. 整合结果并返回                                                │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                      SubAgents                              │ │
│  │                                                            │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │ │
│  │  │ data-analyst │  │ rag-analyst  │  │  web-search  │     │ │
│  │  │  数据分析     │  │  研报检索     │  │  联网搜索    │     │ │
│  │  │              │  │              │  │   (兜底)     │     │ │
│  │  │ Tools:       │  │ Tools:       │  │ Tools:       │     │ │
│  │  │ - read_csv   │  │ - search     │  │ - tavily     │     │ │
│  │  │ - analyze    │  │ - get_summary│  │   _search    │     │ │
│  │  │ - create_    │  │ - list_      │  │              │     │ │
│  │  │   chart      │  │   reports    │  │              │     │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘     │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        RAG System                                │
│                                                                   │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐     │
│  │ PDF 解析  │ → │ Chunking │ → │ Embedding│ → │ ChromaDB │     │
│  │          │   │   分块    │   │   向量化  │   │  存储    │     │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘     │
│                                                                   │
│  检索流程: 查询 → Embedding → 相似度搜索 → 返回相关文档块         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 三、DeepAgents 架构理解

### 3.1 核心概念
```python
from deepagents import create_deep_agent, SubAgent, SubAgentMiddleware

# create_deep_agent 返回一个 CompiledStateGraph (LangGraph)
agent = create_deep_agent(
    model="claude-sonnet-4-5-20250929",
    system_prompt="你的系统提示词",
    tools=[...],  # 主 Agent 的工具
    subagents=[   # 子代理配置
        {
            "name": "subagent-name",
            "description": "何时使用此子代理的描述",
            "system_prompt": "子代理的系统提示词",
            "tools": [...],  # 子代理的工具
            "model": "claude-sonnet-4-5-20250929",
        }
    ],
)
```

### 3.2 Middleware Stack
DeepAgents 自动注入以下中间件：
1. `TodoListMiddleware` - 任务列表管理
2. `FilesystemMiddleware` - 文件系统操作
3. `SubAgentMiddleware` - 子代理调度
4. `SummarizationMiddleware` - 长对话摘要
5. `AnthropicPromptCachingMiddleware` - Prompt 缓存

### 3.3 子代理调用
主 Agent 通过 `task` 工具调用子代理：
```python
# 自动生成的 task 工具
task(
    subagent_type="rag-analyst",  # 子代理名称
    description="搜索关于XXX的内容并返回结果"  # 任务描述
)
```

---

## 四、模块详细设计

### 4.1 RAG 系统模块

#### 文件结构
```
backend/app/rag/
├── __init__.py
├── document_processor.py  # PDF 解析、文本分块
├── embeddings.py          # Embedding 模型封装
├── vectorstore.py         # ChromaDB 向量存储
└── retriever.py           # 检索器封装
```

#### 核心类设计
```python
# document_processor.py
class DocumentProcessor:
    """PDF 文档解析与分块"""
    def parse_pdf(file_path: Path) -> list[Document]
    def chunk_documents(docs: list[Document]) -> list[Chunk]

# embeddings.py
class EmbeddingModel:
    """Embedding 模型封装"""
    def embed_texts(texts: list[str]) -> list[list[float]]
    def embed_query(query: str) -> list[float]

# vectorstore.py
class VectorStore:
    """ChromaDB 向量存储"""
    def add_documents(chunks: list[Chunk])
    def similarity_search(query: str, k: int) -> list[Chunk]
    def delete_by_file_id(file_id: str)

# retriever.py
class Retriever:
    """检索器"""
    def retrieve(query: str, top_k: int = 5) -> list[RetrievalResult]
```

### 4.2 Agent 模块

#### Main Agent (main_agent.py)
```python
MAIN_AGENT_PROMPT = """你是一个专业的研报分析助手...

## 决策逻辑
1. 分析用户问题类型
2. 如果涉及研报内容 → 调用 rag-analyst
3. 如果涉及数据分析 → 调用 data-analyst
4. 如果需要实时信息 → 使用 web-search
5. 整合结果并回复
"""

def create_main_agent():
    return create_deep_agent(
        model=settings.llm_model,
        system_prompt=MAIN_AGENT_PROMPT,
        subagents=[
            DATA_ANALYST_SUBAGENT,
            RAG_ANALYST_SUBAGENT,
        ],
        tools=[web_search_tool],  # 联网搜索作为主 Agent 工具
    )
```

#### Data Analyst Sub-Agent (data_analyst.py)
```python
@tool
def read_csv_file(file_id: str) -> str:
    """读取 CSV 文件并返回预览"""

@tool
def analyze_data(file_id: str, analysis_type: str) -> str:
    """执行统计分析"""

@tool
def create_chart(file_id: str, chart_type: str, x_col: str, y_col: str) -> str:
    """生成图表"""
```

#### RAG Analyst Sub-Agent (rag_analyst.py)
```python
@tool
def search_reports(query: str, top_k: int = 5) -> str:
    """在研报中搜索相关内容"""

@tool
def get_report_summary(file_id: str) -> str:
    """获取特定研报的摘要"""

@tool
def list_available_reports() -> str:
    """列出所有可用研报"""
```

### 4.3 API 模块

#### routes.py
```python
@router.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """同步对话"""

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式对话 (SSE)"""

@router.post("/upload/report")
async def upload_report(file: UploadFile) -> UploadResponse:
    """上传研报 PDF"""

@router.post("/upload/data")
async def upload_data(file: UploadFile) -> UploadResponse:
    """上传数据文件"""
```

---

## 五、开发阶段规划

### Phase 1: 后端基础架构 ✅ (已完成)
- [x] 项目结构初始化
- [x] FastAPI 应用框架
- [x] 配置管理 (pydantic-settings)
- [x] API 路由框架
- [x] Agent 框架代码
- [x] pyproject.toml 依赖配置

### Phase 2: RAG 系统实现
- [ ] PDF 解析器 (document_processor.py)
- [ ] 文本分块策略
- [ ] Embedding 模型集成 (embeddings.py)
- [ ] ChromaDB 向量存储 (vectorstore.py)
- [ ] 检索器实现 (retriever.py)
- [ ] 异步处理上传的 PDF

### Phase 3: Agent 集成
- [ ] 完善 main_agent 与 DeepAgents 集成
- [ ] 完善 data-analyst 子代理工具
- [ ] 完善 rag-analyst 子代理工具
- [ ] 实现 web-search 联网搜索工具
- [ ] Chat API 与 Agent 连接
- [ ] 实现流式响应 (SSE)

### Phase 4: 前端开发
- [ ] Vue 3 + Vite 项目初始化
- [ ] 聊天 UI 组件
- [ ] 文件上传组件
- [ ] 图表展示组件 (ECharts / Chart.js)
- [ ] 响应式布局

### Phase 5: 测试与优化
- [ ] 后端单元测试
- [ ] 集成测试
- [ ] 性能优化
- [ ] 部署配置

---

## 六、数据流设计

### 6.1 研报上传流程
```
用户上传 PDF → FastAPI 接收 → 保存文件
                                    ↓
                          触发异步处理任务
                                    ↓
                          PDF 解析 → 分块 → Embedding
                                    ↓
                          存入 ChromaDB
                                    ↓
                          返回处理状态
```

### 6.2 对话流程
```
用户提问 → Main Agent 接收
                    ↓
           分析问题类型 (LLM 判断)
                    ↓
    ┌───────────────┼───────────────┐
    ↓               ↓               ↓
研报相关        数据分析        通用问题
    ↓               ↓               ↓
rag-analyst   data-analyst   web-search/直接回答
    ↓               ↓               ↓
    └───────────────┼───────────────┘
                    ↓
              整合结果 → 返回用户
```

---

## 七、关键配置

### 7.1 环境变量 (.env)
```env
# Anthropic API
ANTHROPIC_API_KEY=your_key

# Tavily API (联网搜索)
TAVILY_API_KEY=your_key

# 模型配置
LLM_MODEL=claude-sonnet-4-5-20250929
EMBEDDING_MODEL=BAAI/bge-m3

# 数据路径
VECTOR_STORE_PATH=./data/vectorstore
REPORTS_PATH=./data/reports
UPLOADS_PATH=./data/uploads
```

### 7.2 分块策略
- 块大小: 1000 字符
- 重叠: 200 字符
- 保留元数据: 文件名、页码、章节

---

## 八、后续扩展方向

1. **多模态支持**: 支持研报中的图表、图片解析
2. **多语言支持**: 支持英文研报
3. **用户系统**: 多用户、会话管理
4. **知识图谱**: 构建研报知识图谱
5. **API 开放**: 提供 REST API 供第三方调用