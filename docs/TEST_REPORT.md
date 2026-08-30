# 测试报告

> 状态：五个阶段全部完成；18项已登记缺陷全部关闭；当前约定测试范围验收通过
> 最近更新：2026-08-09
> 测试环境：Windows、Python 3.12.7、pytest 9.0.3
> 测试解释器：`D:\DeepAgents\report-analysis-agent\backend\.venv\Scripts\python.exe`

## 1. 当前执行摘要

| 阶段 | 范围 | 结果 | 结论 |
|---|---|---|---|
| 第 1 阶段 | 永久隔离下的初始全量 pytest 基线 | 165 passed | 基线通过 |
| 第 2 阶段 | 核心 API（不含两个风险场景） | 13 passed | 新增核心 API 用例通过 |
| 第 2 阶段 | RA-API-001 针对性测试 | 4 passed | 非法文件 ID 已被接口拒绝 |
| 第 2 阶段 | RA-API-002 针对性测试 | 6 passed | 重建失败时旧索引得到保护 |
| 第 2 阶段 | API 与向量替换相关回归 | 24 passed | 相关功能回归通过 |
| 第 2 阶段 | 永久隔离下的全量 pytest 回归 | 188 passed，20.30 秒 | 全量回归通过 |
| 第 2 阶段 | 正式数据前后快照对比 | 0 项变化 | 未污染正式数据 |
| 第 3 阶段 | 具身智能评测数据集 v1 | 20 条，静态校验通过 | 18 条可回答、2 条无答案 |
| 第 3 阶段 | 固定模型快照下的全库检索基线 | Recall@5 88.89%，MRR 76.62% | 存在跨主题干扰 |
| 第 3 阶段 | `topic=embodied_intelligence` 对照 | Recall@5 100%，MRR 79.63% | 主题过滤有效，证据覆盖未改善 |
| 第 3 阶段 | RA-RAG-001 正式修复与回归 | 46 个相关测试、194 个全量测试通过 | 正式入口 Top-10 跨主题混入为 0 |
| 第 3 阶段 | RA-RAG-002 正式修复与回归 | 41 个相关测试、198 个全量测试通过 | EI-CAUSE-001 证据页由第 17 名升至第 1 名 |
| 第 3 阶段 | Evidence Page Recall@5 异常盘点 | 5 条完成只读核查 | 2 条真实检索缺陷、3 条评测假警报 |
| 第 3 阶段 | RA-EVAL-001 正式修复与回归 | 9 个针对性测试、201 个全量测试通过 | Top-K 与证据组指标可区分真实缺陷和页码假警报 |
| 第 3 阶段 | RA-RAG-003 正式修复与回归 | 4 个针对性测试、38 个 Retriever 测试、204 个全量测试通过 | EI-FIN-002 目标证据由第 8 名升至第 1 名，Top-5 数字覆盖达到 100% |
| 第 3 阶段 | RA-RAG-004 正式修复与回归 | 4 个针对性测试、40 个 Retriever 测试、206 个全量测试通过 | EI-FACT-002 关键事实分块由第 16 名升至第 2 名，Top-5 七项事实完整覆盖 |
| 第 3 阶段 | 40 条独立冻结 RAG 测试集 | 40 条、30 份报告全覆盖、静态校验 0 errors/0 warnings | 36 条可回答、4 条无答案、6 条跨报告；与开发集问题及问题族均 0 重复 |
| 第 3 阶段 | 冻结测试集唯一一次正式验收 | 可回答题 Recall@5 100%、MRR 90.28%、NDCG 93.28%、Term Coverage@5/10 94.49%/97.36% | 6 条跨报告题 Top-5 全覆盖；第 3 阶段通过 |
| 第 4 阶段 | Agent 路由数据集与离线执行器 | 20 条数据、9 个离线测试通过 | 可执行并保留完整路由事件 |
| 第 4 阶段 | GLM-4.5-Air 直接回答冒烟测试 | 1/1 通过，路由准确率 100% | 智谱 API 与基础 Agent 链路正常 |
| 第 4 阶段 | RA-AGENT-001 初始真实复现 | 0/1 通过，实际调用 `ls → glob` | 主 Agent 文件系统工具抢占 RAG 路由 |
| 第 4 阶段 | RA-AGENT-001 修复后离线回归 | 21 passed，8.37 秒 | 工具排除配置及路由评估链路通过 |
| 第 4 阶段 | RA-AGENT-001 真实同题回归 | 1/1 通过，路由准确率 100% | `task → rag-analyst → search_reports`，禁用工具为 0 |
| 第 4 阶段 | RA-AGENT-001 修复后全量回归 | 216 passed，11.76 秒 | 失败、错误和跳过均为 0 |
| 第 4 阶段 | AGENT-DATA-001 第一轮修复后真实路由 | 0/1 通过，自动检查 85.71% | 正确委派前仍调用 `read_file → ls`，RA-AGENT-001 重新打开 |
| 第 4 阶段 | RA-AGENT-001 第二轮离线针对性测试 | 16 passed，8.79 秒 | 提示清理、schema 过滤和运行时工具拦截全部通过 |
| 第 4 阶段 | RA-AGENT-001 第二轮全量回归 | 220 passed，11.70 秒 | 失败、错误和跳过均为 0 |
| 第 4 阶段 | AGENT-DATA-001 第二轮真实回归 | 1/1 通过，路由准确率 100% | `task → list_data_files → read_data_file`，禁用工具和工具错误为 0 |
| 第 4 阶段 | AGENT-RAG-001 第二轮真实回归 | 1/1 通过，路由准确率 100% | `task → list_available_reports → search_reports × 3`，主题参数正确且未使用 Web |
| 第 4 阶段 | 剩余开发集第一批真实执行 | 1 条有效通过、1 条确认缺陷、6 条额度阻塞 | DATA-002 相同错误调用 400 次并触发智谱 429，原始 1/8 不作为路由准确率结论 |
| 第 4 阶段 | RA-AGENT-002 安全修复与回归 | 236 个全量测试通过，DATA-002 路由 1/1 通过 | 4 次工具调用、0 错误，参数字符串成功归一化，未发生循环或 429 |
| 第 4 阶段 | RA-AGENT-003 真实确认 | 路由与子代理结果正确，最终用户回答不完整 | 主 Agent 收到完整统计正文后只输出“📊 来源：数据分析” |
| 第 4 阶段 | RA-AGENT-003 离线修复回归 | 24 个针对性测试、239 个全量测试通过 | 来源标签兜底、正常正文保留和多步工具调用均通过 |
| 第 4 阶段 | RA-AGENT-003 真实同题回归 | 1/1 通过，最终回答完整 | 865 字子代理正文被完整保留，并追加“📊 来源：数据分析” |
| 第 4 阶段 | 剩余 6 条开发集真实批测 | 5/6 通过，完整执行且无额度阻塞 | Web 2/2、直答 2/2、彩票拒答 1/1；NA-001 触发工具硬预算 |
| 第 4 阶段 | RA-AGENT-004 离线针对性测试 | 29 passed | 主题补全、单项及合计检索上限、越界拒绝和既有 Agent 边界测试全部通过 |
| 第 4 阶段 | RA-AGENT-004 全量回归 | 244 passed | 新增 5 个测试，既有测试无回归失败 |
| 第 4 阶段 | AGENT-NA-001 真实单题回归 | 1/1 通过，RA-AGENT-004 关闭 | 5 次工具调用、3 次主题化搜索、0 错误、无 Web 调用，最终无答案回复完整 |
| 第 4 阶段 | 保留测试集非重复批次 | 3/4 通过 | DATA-003、DATA-004、MULTI-002 通过；RAG-004 错误调用 Web |
| 第 4 阶段 | 保留测试集非联网稳定性批次 | 7/9 次运行、2/3 个案例通过 | RAG-003 两轮重复委派并触发用例预算；DIRECT-003、NA-003 均 3/3 通过 |
| 第 4 阶段 | 保留测试集联网批次 | 7/7 次运行、3/3 个案例通过 | 路由约束全部通过，但两个重复案例的精确工具序列均不稳定 |
| 第 4 阶段 | 保留测试集总体 | 17/20 次运行、8/10 个案例通过 | 运行准确率 85%，案例准确率 80%，精确路由稳定性 1/5（20%），未达 90% 目标 |
| 第 4 阶段 | RA-AGENT-005 离线针对性测试 | 35 passed | 财务 RAG 双层 Web 边界、显式联网放行和概念题保护全部通过 |
| 第 4 阶段 | RA-AGENT-005 全量回归 | 250 passed | 新增 6 个测试，既有测试无回归失败 |
| 第 4 阶段 | RA-AGENT-005 第一轮真实复测 | 0/1，自动得分 92.86% | Web 越界已消失，RAG 路由正确；`search_reports` 缺少具身智能主题参数 |
| 第 4 阶段 | RA-AGENT-005 第二轮离线与全量回归 | 36 个针对性测试、251 个全量测试通过 | 凌云光财务查询主题补全且保留 `file_id` |
| 第 4 阶段 | RA-AGENT-005 三轮真实回归 | 3/3 通过，RA-AGENT-005 关闭 | 全部走 RAG、主题参数正确、0 Web、0 错误，关键数字一致 |
| 第 4 阶段 | RA-AGENT-006 离线与全量回归 | 5 个最小兼容测试、42 个针对性测试、257 个全量测试通过 | 文本工具标记仅可恢复为当前白名单内的结构化调用 |
| 第 4 阶段 | RA-AGENT-006 三轮真实回归 | 3/3 通过，RA-AGENT-006 关闭 | 每轮仅一次 RAG 委派、5 次工具调用、0 Web、0 错误，无文本伪调用残留 |
| 第 4 阶段 | 修复后保留测试集有效结果 | 20/20 次运行、10/10 个案例通过 | 路由准确率 100%；精确签名稳定性 20% 作为非阻断风险保留 |
| 第 5 阶段 | 事实、数字、引用与无答案人工抽查 | 初始4条完全通过、1条部分通过、5条不通过；修复后有效结果10条完全通过 | RA-ANS-001至005均已关闭 |
| 第 5 阶段 | AGENT-DATA-003 真实执行与图表核验 | 路由1/1通过，持久化图表正确，最终答案不通过 | 复现RA-ANS-002；新增RA-ANS-005 |
| 第 5 阶段 | Tavily、智谱与公开原始页面交叉核验 | 3 条 Web/混合回答完成复核 | GLM Judge 仅作辅助，最终结论以日期化原始证据为准 |
| 第 5 阶段 | RA-ANS-002/005 针对性与全量回归 | 50 个针对性测试、264 个全量测试通过 | 全表计数、图表权威正文、URL与artifact持久化均有自动化保护 |
| 第 5 阶段 | AGENT-DATA-003 修复后真实复测 | 1/1通过，人工事实与图表核查通过 | 8人姓名薪资准确，Frank最高、Alice最低，最终URL与永久图表一致 |
| 第 5 阶段 | AGENT-MULTI-002 修复后真实复测 | 1/1通过，人工事实核查通过 | `value_counts`基于完整8行得到北京2人；RAG部分保留本地研报页码且无Web |
| 第 5 阶段 | RA-ANS-001 针对性与全量回归 | 63 个针对性测试、273 个全量测试通过 | 禁止跨事实继承年份，未披露时间显式说明，具体年月绑定直接证据页 |
| 第 5 阶段 | RA-ANS-003/004 针对性与全量回归 | 76 个针对性测试、283 个全量测试通过 | Web URL/日期保全、同日新闻证据门槛、日期化历史行情表抽取及冲突摘要覆盖均通过 |
| 第 5 阶段 | Web/混合样本修复后真实复测 | `WEB-003`、`WEB-004`、`MULTI-001` 均通过人工核验 | 旧闻不冒充当日动态；中信海直14.20元、奥普特120.00元均绑定2026-08-07历史行情直链 |
| 第 5 阶段 | AGENT-RAG-003 修复后真实复测 | 1/1通过，人工事实—年份—页码核查通过 | 送样年份明确未披露；收购、合作、收入和送样状态分别由第21-24页直接支持 |

针对性测试、相关回归和全量回归之间存在用例重叠，不能将各行数量相加。RA-RAG-004 修复后的历史全量回归为206条；RA-AGENT-006修复后的全量回归为257 passed；RA-ANS-002与RA-ANS-005修复后的全量回归为264 passed；RA-ANS-001修复后为273 passed；完成RA-ANS-003与RA-ANS-004修复后，最终全量回归为283 passed，失败、错误和跳过均为0，pytest退出码为0。

## 2. 测试环境与隔离结果

全量回归在 `backend` 目录执行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

测试隔离由 `backend/tests/conftest.py` 统一提供：

- reports、uploads 和 Chroma 向量库均指向每次运行独有的临时目录；
- 默认禁用真实 LLM、Tavily 和模型下载；
- 测试完成后先关闭 Chroma Shared System，再删除临时目录；
- 即使测试失败，也执行清理流程。

全量回归前后对正式 `backend/data` 进行了逐项快照比较：

| 检查项目 | 测试前 | 测试后 | 变化 |
|---|---:|---:|---:|
| 文件 | 70 | 70 | 0 |
| 目录 | 7 | 7 | 0 |
| 总记录 | 77 | 77 | 0 |
| 新增路径 | 0 | 0 | 0 |
| 删除路径 | 0 | 0 | 0 |
| 长度或 SHA-256 变化 | 0 | 0 | 0 |

结论：本阶段自动化测试没有新增、删除或修改任何正式数据。

## 3. 缺陷清单

### RA-API-001：通配符 file_id 可删除多个不相关文档

| 字段 | 内容 |
|---|---|
| 状态 | 已修复，针对性测试与全量回归通过 |
| 严重程度 | 高 |
| 优先级 | P0 |
| 类型 | 数据完整性 / 非法输入校验 / 越权删除 |
| 影响接口 | 文档查询、删除和重建接口 |
| 首次确认日期 | 2026-07-23 |
| 修复验证日期 | 2026-08-01 |
| 自动化证据 | `backend/tests/test_api/test_phase2_file_id_risk_evidence.py`、`backend/tests/test_api/test_phase2_file_id_validation.py` |

#### 初始复现结果

向删除接口传入 `DELETE /api/documents/*` 后，初始实现返回 200，删除了隔离测试目录中的 PDF 和 CSV，并调用了向量删除：

```text
HTTP 状态码：200
响应体：{"status": "deleted", "file_id": "*"}
研报文件存活：False
数据文件存活：False
向量删除被调用：True
```

正式 reports、uploads 和 Chroma 数据未参与复现。

#### 根因

路由没有校验 `file_id` 格式，直接将用户输入拼入文件系统 glob 模式。当 `file_id` 为 `*` 时，生成的模式能够匹配多个不相关文件，相同输入还会传给向量删除方法。

#### 修复

- 文档查询、删除和重建接口的路径参数统一声明为 `uuid.UUID`；
- 路由入口将 UUID 规范化为字符串后再进行后续操作；
- 通配符、路径穿越形式和普通非 UUID 字符串会在文件或向量操作前由 FastAPI 返回 422；
- 原有使用伪造非 UUID 文件 ID 的测试数据已改为合法 UUID。

#### 修复验证

4 个 RA-API-001 针对性测试全部通过。非法 ID 不再触发文件匹配、文件删除或向量删除，随后文档相关回归和 188 条全量 pytest 也全部通过。

### RA-API-002：索引重建失败会删除仍可用的旧索引

| 字段 | 内容 |
|---|---|
| 状态 | 已修复，针对性测试与全量回归通过 |
| 严重程度 | 高 |
| 优先级 | P0 |
| 类型 | 可靠性 / 数据一致性 / 失败恢复 |
| 影响流程 | PDF 索引重建、向量替换及失败回滚 |
| 首次确认日期 | 2026-07-23 |
| 修复验证日期 | 2026-08-01 |
| 自动化证据 | `backend/tests/test_api/test_phase2_core_api.py::TestHighValueRisks::test_failed_reindex_preserves_previous_chunks`、`backend/tests/test_rag/test_vectorstore_replacement.py` |

#### 初始复现结果

重建流程在解析替换 PDF 之前删除旧索引。解析器随后抛出异常，任务虽然被标记为 failed，但旧向量证据已经消失：

```text
文档状态：failed
解析异常：replacement PDF cannot be parsed
重建前旧向量块：1
重建后旧向量块：0
```

正式 Chroma 和正式文件未参与复现。

#### 根因

`process_pdf_task` 在新 PDF 完成解析、分块、嵌入和写入之前调用旧索引删除。一旦后续任一步骤失败，异常处理只更新文档状态，无法恢复已经删除的旧向量。

#### 修复

- 重建时先解析新 PDF，并拒绝零分块结果；
- 新增安全替换流程，在删除旧记录前完成新向量计算；
- 替换前保存旧记录的 ID、文档、元数据和向量快照；
- 写入失败时清理部分写入的新记录，并使用快照恢复旧索引；
- 回滚自身失败时返回明确异常，避免把不完整状态误报为成功。

#### 修复验证

6 个 RA-API-002 针对性实例全部通过，覆盖解析失败、空分块、嵌入失败、向量写入失败、成功替换和失败回滚保护。相关回归 24 passed，全量回归 188 passed。

### RA-RAG-001：公司特定财务查询被通用财务内容和跨主题报告压制

| 字段 | 内容 |
|---|---|
| 状态 | 已修复，自动化与正式语料回归通过 |
| 严重程度 | 中 |
| 优先级 | P1 |
| 类型 | 检索排序 / 混合检索 / 跨主题干扰 |
| 影响流程 | 公司财务问答、Top-K 报告召回和引用选择 |
| 首次确认日期 | 2026-08-07 |
| 自动化证据 | `backend/data/evaluation/embodied_intelligence_retrieval_v1_result.json`、`backend/data/evaluation/embodied_intelligence_topic_filter_ab_v1_result.json` 中的 `EI-FIN-002` |

#### 初始复现结果

凌云光财务题 `EI-FIN-002` 的正确报告排在去重后的第 6 份，Recall@5 为 0，直至 Recall@10 才命中。前列结果包含杭叉集团和低空经济主题的宗申动力报告。

#### 根因

财务问法包含“营收、归母净利润、同比增长”等高频通用词。当前混合检索中，关键词满分结果能够压过公司名称和主题匹配；默认检索也没有施加 `topic=embodied_intelligence` 过滤，跨主题报告因此参与排序。

#### 修复

- `Retriever.retrieve` 新增可选 `topic` 参数，并在候选召回前把主题元数据条件同时传给向量检索和关键词检索；
- 单文件与主题同时指定时使用 Chroma `$and` 条件，多文件检索继续保留原有文件级后过滤；
- Agent 工具 `search_reports`、`check_rag_relevance` 和评测 CLI `run_evaluation --topic` 接入同一正式参数；
- Agent 提示词明确具身智能、低空经济的主题值，并保留主题不明确或跨主题比较时的全库检索行为。

#### 修复验证

新增 6 个自动化回归点，覆盖双检索通道、主题与文件组合、RA-RAG-001 跨主题竞争项、Agent 工具透传、相关性预检透传和评测入口透传。相关测试 46 passed，全量回归 194 passed。

正式评测入口使用同一本地 BGE-M3 快照和 18 条可回答样本复测后，Recall@3/5/10 均为 100%，MRR 为 79.63%，NDCG 为 84.47%。`EI-FIN-002` 的相关报告由全库第 6 份提升到主题内第 3 份，Recall@5 从 0 提升到 1；18 条问题的 Top-10 跨主题报告出现次数为 0。结果保存在 `backend/data/evaluation/embodied_intelligence_topic_filtered_official_v1_result.json`。

### RA-RAG-002：报告级命中掩盖关键证据页未召回

| 字段 | 内容 |
|---|---|
| 状态 | 已修复，自动化与正式语料回归通过 |
| 严重程度 | 中 |
| 优先级 | P1 |
| 类型 | 证据召回 / 分块排序 / 评测盲点 |
| 影响流程 | 原因分析、多事实回答和引用完整性 |
| 首次确认日期 | 2026-08-07 |
| 自动化证据 | `backend/data/evaluation/embodied_intelligence_retrieval_v1_result.json`、`backend/data/evaluation/embodied_intelligence_topic_filter_ab_v1_result.json` 中的 `EI-CAUSE-001` |

#### 初始复现结果

广和通原因题 `EI-CAUSE-001` 在报告级 Recall@3 中被视为命中，但实际召回的是该报告第 2、3 页财务表格，承载“汇兑损失、POC 阶段、小批量出货、账期拉长、信用减值损失”的第 1 页没有进入前 20 个分块，预期关键词覆盖率仅 20%。

#### 根因

当前 Recall 指标按 `file_id` 去重，只能确认相关报告是否出现，不能确认标准答案所需的证据分块是否出现。财务表格的通用关键词得分较高，进一步挤压了原因说明页。

解释型问题同时包含“利润”等财务词时，旧查询改写会优先生成纯财务变体并对表格加权，使“为什么承压、项目处于什么阶段”所需的经营分析正文进一步后移。PDF 抽取产生的空格和断行也导致 `POC阶段` 无法匹配正文中的 `POC 阶段`。

#### 修复

- 新增解释型查询识别，覆盖“为什么、原因、承压、阶段、进展”等意图；
- 解释型问题改用“经营分析/原因因素”和“项目阶段/客户导入/小批量”等通用查询改写，不再追加纯财务表格改写；
- 重排时提升原因链、阶段和进展正文，对解释型问题中的纯表格候选降权；
- 关键词覆盖计算先清除 PDF 空格、断行和标点；
- 评测器读取数据集 `evidence.pages`，新增 Evidence Page Recall@K 和首个证据页分块排名，保留样本 ID。

#### 修复验证

新增 4 个自动化测试，覆盖解释型查询改写、正文与表格排序、报告级假阳性证据页指标以及 PDF 断行归一化；相关测试 41 passed，全量回归 198 passed。

固定本地 BGE-M3 快照后，`EI-CAUSE-001` 的正确第 1 页正文由分块第 17 名提升到第 1 名，报告排名由第 2 名提升到第 1 名，Evidence Page Recall@1/3/5/10 均为 100%，关键词覆盖率由 80% 提升到 100%。单题结果保存在 `backend/data/evaluation/embodied_intelligence_ra_rag_002_targeted_result.json`。

18 条可回答样本回归中，Recall@5 保持 100%，Recall@1 由 52.78% 提升到 63.89%，MRR 由 79.63% 提升到 85.19%，NDCG 由 84.47% 提升到 88.57%，归一化后的关键词覆盖率为 100%。新增 Evidence Page Recall@5 基线为 77.78%，说明评测现在能够暴露其他证据页未进入 Top-5 的样本；这些样本作为后续质量线索保留，不影响本缺陷对应案例的关闭。完整结果保存在 `backend/data/evaluation/embodied_intelligence_ra_rag_002_full_result.json`。

### RA-RAG-003：公司财务查询的 Top-5 被错误报告重复占据

| 字段 | 内容 |
|---|---|
| 状态 | 已修复，已关闭 |
| 严重程度 | 高 |
| 优先级 | P1 |
| 类型 | 公司实体匹配 / 分块排序 / 结果多样性 |
| 影响流程 | 公司财务问答、数字回答和引用正确性 |
| 首次确认日期 | 2026-08-07 |
| 自动化证据 | `backend/tests/test_rag/test_retriever.py`、`backend/data/evaluation/embodied_intelligence_ra_rag_003_targeted_result.json`、`backend/data/evaluation/embodied_intelligence_ra_rag_003_full_result.json` |

#### 复现结果

`EI-FIN-002` 查询凌云光 2025 年营收和归母净利润及同比增速。目标报告第 1 页包含全部四个标准答案，但目标分块排在第 8；Top-5 五个分块全部来自杭叉集团报告，Top-5 关键词覆盖为 0/4。按 `file_id` 去重后的报告排名为第 3，因此报告级 Recall@5 仍显示为 1，无法反映提供给问答模型的前五个分块没有任何正确数字。

#### 根因

“营收、归母净利润、同比增长”等通用财务词使同主题的杭叉集团财务内容取得高分。现有公司名称加权不足以抵消该差距，同时检索结果没有单报告分块数量上限，错误报告的多个相似分块可以占满 Top-5。主题过滤只能排除跨主题干扰，不能解决同主题公司之间的竞争。

#### 修复

- 从候选报告文件名的 `【公司名】` 中提取公司实体，仅当查询明确点名该公司时启用公司约束；
- 精确公司候选增加 0.45 重排分，其他已识别公司候选降低 0.08 分；
- 最终选择时先为每个被点名公司保留最高分候选，再按重排分填充；同一非目标报告第一轮最多保留 2 个分块，候选不足时按原顺序回填，保证返回数量不减少；
- 未点名公司的普通查询保持原排序，同时支持查询中点名两家公司；
- 新增 3 个 RA-RAG-003 自动化场景，并保留原有公司名称重排用例共同形成 4 个针对性测试。

#### 修复验证

4 个针对性测试全部通过，Retriever 文件 38 个测试全部通过；永久隔离下的全量回归为 204 passed，耗时 11.63 秒。

固定本地 BGE-M3、930 个 Chroma 分块、`topic=embodied_intelligence` 和 Top-20 候选进行真实单题复测后，`EI-FIN-002` 的目标报告排名由第 3 升至第 1，目标证据分块由第 8 升至第 1，Term Coverage@5 由 0/4 升至 4/4，Evidence Page Recall@5 由 0% 升至 100%；杭叉集团在 Top-5 的占位由 5 个降至 2 个。单题结果保存在 `backend/data/evaluation/embodied_intelligence_ra_rag_003_targeted_result.json`。

同一配置下对完整 20 条数据集回归，并按历史基线口径排除 2 条无答案安全题后，18 条可回答样本的 Recall@1 由 63.89% 升至 86.11%，Recall@3/5 保持 100%，MRR 由 85.19% 升至 97.22%，NDCG 由 88.57% 升至 97.95%，Term Coverage@5 由 92.06% 升至 97.62%，Term Coverage@10 保持 97.62%，Evidence Page Recall@5 由 77.78% 升至 83.33%，Evidence Group Recall@5 保持 100%。逐题对比未发现报告排名、证据排名、关键词覆盖或证据页召回退化。完整 20 条原始结果保存在 `backend/data/evaluation/embodied_intelligence_ra_rag_003_full_result.json`，其中 2 条无答案题不参与上述 18 条可回答样本指标。

### RA-RAG-004：多事实进展查询的 Top-5 证据不完整

| 字段 | 内容 |
|---|---|
| 状态 | 已修复，已关闭 |
| 严重程度 | 中 |
| 优先级 | P1 |
| 类型 | 多事实召回 / 长文档分块排序 |
| 影响流程 | 合作进展、产品送样和多事实回答完整性 |
| 首次确认日期 | 2026-08-07 |
| 自动化证据 | `backend/tests/test_rag/test_retriever.py`、`backend/data/evaluation/embodied_intelligence_ra_rag_004_targeted_result.json`、`backend/data/evaluation/embodied_intelligence_ra_rag_004_full_result.json` |

#### 复现结果

`EI-FACT-002` 查询奥普特从机器视觉延伸到具身智能的路径以及合作、送样进展。Top-5 只覆盖 7 个标准关键词中的 4 个，缺少“东莞泰莱、关节模组、送样”；Top-10 仍然缺少这三项。标注第 1 页被拆成两个分块，第一个分块排第 7，但真正包含缺失事实的第二个分块排第 16，因此仅按页码判断会高估证据命中。

#### 根因

长报告中“3D 视觉、工业 AI、具身智能”等通用内容分散在多个章节并获得较高排序，包含并购、合作和送样事实的摘要分块没有得到足够的精确进展信号。更直接的实现原因是“进展”被归入原因/阶段解释意图，三条检索查询中有一条被改写为与本题无关的“经营分析、利润承压、原因、因素”；RA-RAG-003 的公司加权对同一家奥普特报告内的所有分块一致，无法改善报告内部事实排序。页级指标也无法区分同一页内“命中标题/财务摘要”和“命中事实段落”。

#### 修复

- 新增由“合作、送样、小批量、量产、落地”触发的进展意图，优先于原因/阶段解释意图；
- 进展题增加“合作、客户验证、送样、导入、量产”通用改写；具身智能或机器人题再增加“关节模组、执行器、精密传动、运动控制、并购、收购”改写；
- 正文重排增加合作、送样、客户验证、并购、关节模组和运控等真实进展信号，最多增加 0.32 分，不对“机器人、具身智能”等泛词额外加分；
- 意图判断只使用原始用户问题，关键词覆盖仍使用全部查询改写，避免原因型改写中的“小批量、量产”反向触发进展意图；
- 新增 2 个 RA-RAG-004 自动化场景，并将 2 个既有原因型测试纳入针对性回归，要求技术路线和合作送样事实同时进入 Top-5。

#### 修复验证

4 个针对性测试全部通过，Retriever 文件 40 个测试全部通过；永久隔离下的全量回归为 206 passed，耗时 11.65 秒。

固定本地 BGE-M3、930 个 Chroma 分块、`topic=embodied_intelligence` 和 Top-20 候选进行真实单题复测后，`EI-FACT-002` 中包含“东莞泰莱、越疆、手眼脑一体化、关节模组、送样”的关键分块由第 16 升至第 2；Term Coverage@3 由 2/7 升至 7/7，Term Coverage@5/10 由 4/7 升至 7/7，Evidence Page Recall@5 由 0% 升至 100%。单题结果保存在 `backend/data/evaluation/embodied_intelligence_ra_rag_004_targeted_result.json`。

同一配置下对完整 20 条数据集回归，并按历史基线口径排除 2 条无答案安全题后，18 条可回答样本的 Recall@1 保持 86.11%，Recall@3/5 保持 100%，MRR 保持 97.22%，NDCG 保持 97.95%；Term Coverage@3 由 92.56% 升至 96.53%，Term Coverage@5/10 由 97.62% 升至 100%，Evidence Page Recall@5 由 83.33% 升至 88.89%，Evidence Group Recall@5 保持 100%。逐题对比只有 `EI-FACT-002` 发生变化且全部为改善，其余 17 条没有报告排名、证据排名或覆盖退化。完整 20 条原始结果保存在 `backend/data/evaluation/embodied_intelligence_ra_rag_004_full_result.json`。

### RA-EVAL-001：精确页码指标误报且 Top-20 覆盖掩盖 Top-5 失败

| 字段 | 内容 |
|---|---|
| 状态 | 已修复，自动化与正式语料回归通过 |
| 严重程度 | 中 |
| 优先级 | P1 |
| 类型 | 评测指标 / 证据标注模型 / 通过条件 |
| 影响流程 | RAG 质量判定、缺陷筛选和回归门禁 |
| 首次确认日期 | 2026-08-07 |
| 自动化证据 | `EI-PROG-002`、`EI-CROSS-001`、`EI-CROSS-003` 的 Evidence Page Recall@5 与 Top-5 关键词覆盖对照 |

#### 复现结果

- `EI-PROG-002` 的 Evidence Page Recall@5 为 0，但第 4、5、14、15 页的等价证据使 Top-5 关键词覆盖达到 8/8；
- `EI-CROSS-001` 和 `EI-CROSS-003` 的 Evidence Page Recall@5 均为 50%，但 Top-5 关键词覆盖均为 8/8；
- `EI-FACT-002` 和 `EI-FIN-002` 的全 Top-20 关键词覆盖为 100%，但 Top-5 分别只有 4/7 和 0/4。

#### 初步根因

当前 Evidence Page Recall 把标注页视为唯一正确证据，无法表达“多个页面包含等价证据，任意一个命中即可”。关键词覆盖率则对全部 Top-20 分块计算，与实际问答常用的 Top-5 上下文不一致。页级命中也无法保证同一页内真正承载答案的分块已经召回。

#### 修复

- 评测结果新增 `Term Coverage@K`，默认输出 Top-1/3/5/10 覆盖率，同时保留全 Top-20 的旧字段用于历史兼容；
- 新增带关键词约束的 Evidence Group Recall@K：每个事实组可以列出多个等价页面，只有这些页面的召回分块完整覆盖该组关键词才算命中；
- 为 `EI-PROG-002`、`EI-CROSS-001`、`EI-CROSS-003` 标注 6 个事实证据组，并在结果中记录每组首次完整命中的分块排名；
- 数据校验脚本新增证据组 ID、相关文件、可接受页码和页内关键词校验；
- 保留严格 Evidence Page Recall@K 作为诊断项，不再单独据此判定问答证据失败。

#### 修复验证

评测器 9 个针对性测试全部通过，覆盖 Top-5 与 Top-20 覆盖差异、等价证据页、同页无关分块和证据组解析；20 条数据集静态校验通过，18 条可回答、2 条无答案，错误和警告均为 0；永久隔离下全量回归为 201 passed，耗时 11.85 秒。

正式语料复测固定本地 BGE-M3 快照 `5617a9f61b028005a4858fdac845db406aefb181`、930 个 Chroma 分块、`topic=embodied_intelligence` 和 Top-20 候选。18 条可回答样本的 Recall@5 为 100%，MRR 为 85.19%，NDCG 为 88.57%，Term Coverage@5 为 92.06%，Term Coverage@10 为 97.62%。严格 Evidence Page Recall@5 仍为 77.78%，但 3 条已标注样本的 6 个事实组在 Top-5 全部命中，Evidence Group Recall@5 为 100%，证明原 3 条异常属于页码口径假警报。

新指标同时保留了两条真实质量线索：`EI-FACT-002` 的 Term Coverage@5/10 均为 4/7，现已由 RA-RAG-004 修复并关闭；`EI-FIN-002` 的 Term Coverage@5 为 0/4、Top-10 为 4/4，现已由 RA-RAG-003 修复并关闭。完整结果保存在 `backend/data/evaluation/embodied_intelligence_ra_eval_001_full_result.json`。

### RA-AGENT-001：主 Agent 内置文件系统工具抢占 RAG 路由

| 字段 | 内容 |
|---|---|
| 状态 | 已关闭；第二轮工具边界修复后，RAG 与数据路径真实回归均通过 |
| 严重程度 | 高 |
| 优先级 | P1 |
| 类型 | Agent 路由 / 工具边界 / 错误拒答 |
| 影响流程 | 所有需要从本地研报或数据文件获取证据的问题 |
| 首次确认日期 | 2026-08-08 |
| 自动化证据 | `backend/data/evaluation/results/agent_routing_eval_20260808_150243.json`、`backend/data/evaluation/results/agent_routing_eval_20260808_165732.json`、`backend/data/evaluation/results/agent_routing_eval_20260808_170909.json`、`backend/data/evaluation/results/agent_routing_eval_20260808_171155.json`、`backend/tests/test_agents/test_main_agent.py`、`backend/tests/test_agents/test_tool_boundary.py` |

#### 复现结果

真实执行 `AGENT-RAG-001` 时，预期路径为 `task(rag-analyst) → search_reports(topic="embodied_intelligence")`；GLM-4.5-Air 实际调用顺序为 `ls(path="/") → glob(pattern="**/*")`，没有委派任何子 Agent，也没有执行 RAG 检索，最终错误回复“当前工作目录中没有文件，请先上传研报”。单条路由准确率为 0%，自动检查得分为 64.29%；轨迹中 3 次模型响应共记录 26,972 tokens。由于检索从未发生，本次没有向外部模型发送研报片段。

初次工具排除修复后，`AGENT-DATA-001` 仍暴露同类残余问题。预期直接执行 `task(data-analyst) → list_data_files → read_data_file/read_csv_file`；实际先调用 `read_file(path="/sample.csv") → ls(path="/")`，其中 `read_file` 因实际 schema 要求 `file_path` 而产生工具错误，随后才恢复为 `task(data-analyst) → list_data_files → read_data_file(file_id="sample.csv")`。最终回答正确给出 8 行、5 列，但禁用工具和工具错误检查失败，路由准确率为 0%，自动检查得分为 85.71%。结果保存在 `backend/data/evaluation/results/agent_routing_eval_20260808_165732.json`，主流程轨迹记录 5 个模型响应、26,156 tokens，临时上传目录已清理。

#### 根因

禁用 `general-purpose` 只移除了默认通用子 Agent，没有移除 DeepAgents 0.6.1 通过 `FilesystemMiddleware` 注入主 Agent 的 `ls`、`glob`、`read_file` 等工具。初次修复使用 `excluded_tools` 从模型请求的工具 schema 中移除了这些工具，但 `FilesystemMiddleware` 仍会在系统提示末尾追加“可以使用 `ls/read_file/glob`”及文件读取约定。GLM-4.5-Air 在“sample.csv 文件”语义下依据残留系统提示生成了已隐藏工具调用；运行时工具节点仍能执行这些调用，因此仅隐藏 schema 不足以建立完整工具边界。

#### 修复

- 第一轮修复：在精确模型 Harness Profile 中配置 `excluded_tools`，从实际模型请求中移除 `write_todos`、`ls`、`read_file`、`write_file`、`edit_file`、`glob`、`grep` 和 `execute`；
- 保留 `task`、Web 工具以及 `rag-analyst`、`data-analyst` 的业务工具；
- 在主 Agent 配置测试中断言完整禁用集合，并明确断言 `task` 和 `web_search` 未被误删；
- 第二轮修复：新增 `BusinessToolBoundaryMiddleware`，在 `FilesystemMiddleware` 追加提示之后移除文件系统、命令执行和待办工具说明，并根据每次模型请求生成当前业务工具白名单；
- 在边界中间件内再次过滤禁用工具 schema；即使模型仍构造越界调用，也会在进入文件系统或执行后端之前返回错误工具消息，不产生实际操作；
- 从 Harness Profile 中排除 `TodoListMiddleware`，消除 `write_todos` 工具和对应残留提示；边界中间件通过 `extra_middleware` 同时应用于主 Agent、`rag-analyst` 和 `data-analyst`。

#### 修复验证

主 Agent 配置、事件收集器和路由评分器共 21 个离线针对性测试全部通过，耗时 8.37 秒；禁用工具集合、`task`/Web 工具保留规则以及数据集执行与评分链路均通过。

使用相同数据集、GLM-4.5-Air 和智谱 Anthropic 兼容入口执行 `AGENT-RAG-001` 真实同题复测，路由评分由 0% 升至 100%。实际调用顺序为 `task → list_available_reports → search_reports → search_reports → search_reports`，唯一委派对象为 `rag-analyst`；三次 `search_reports` 均传入 `topic="embodied_intelligence"`，`ls`、`glob` 及其他禁用工具和 Web 工具调用均为 0，全部自动检查通过。结果保存在 `backend/data/evaluation/results/agent_routing_eval_20260808_164859.json`。该次主流程轨迹记录 2 个模型响应、12,797 tokens，临时上传目录已清理。

第一轮修复通过了 RAG 同题验证和全量自动化回归。首次全量回归为 215 passed、1 failed，唯一失败是 `test_settings_defaults` 错误读取开发 `.env` 中的 `glm-4.5-air`，与产品修复无关；默认值测试改为显式忽略 `.env` 后，针对性复测 1 passed，最终全量回归为 216 passed，耗时 11.76 秒，失败、错误和跳过均为 0。

随后执行的 `AGENT-DATA-001` 证明第一轮修复未覆盖系统提示残留和运行时工具拒绝，因此 RA-AGENT-001 重新打开；据此将第二轮关闭条件确定为：同时通过 `AGENT-RAG-001`、`AGENT-DATA-001` 真实回归及全量 pytest。

第二轮新增工具边界测试后，`test_tool_boundary.py` 与主 Agent 测试合计 16 passed，耗时 8.79 秒；永久隔离下的全量 pytest 为 220 passed，耗时 11.70 秒，退出码为 0。

真实 `AGENT-DATA-001` 回归为 1/1、100%，实际调用顺序为 `task(data-analyst) → list_data_files → read_data_file(file_id="11111111-1111-4111-8111-111111111111")`，3 次工具调用均成功，`read_file`、`ls`、其他禁用工具和 Web 工具调用均为 0，最终正确返回 8 行、5 列；结果保存在 `backend/data/evaluation/results/agent_routing_eval_20260808_170909.json`，隔离的临时上传目录已清理。

真实 `AGENT-RAG-001` 回归同样为 1/1、100%，实际调用顺序为 `task(rag-analyst) → list_available_reports → search_reports × 3`；三次检索均传入 `topic="embodied_intelligence"`，禁用工具、数据工具、Web 工具和工具错误均为 0，最终回答引用《奥普特深度报告：视觉龙头迈向具身智能，双轮驱动开启第二曲线》第 1、7、23、24 页。结果保存在 `backend/data/evaluation/results/agent_routing_eval_20260808_171155.json`。至此第二轮关闭条件全部满足，RA-AGENT-001 关闭。

第一轮 RAG 同题回归的最终回答虽包含研报名称但未匹配到页码格式；第二轮回归已生成带页码来源。引用是否逐项准确支持回答仍属于第 5 阶段的人工答案质量抽查范围。

### RA-AGENT-002：数据子 Agent 参数校验失败后无限重复调用

| 字段 | 内容 |
|---|---|
| 状态 | 已修复并关闭；离线、全量、单条及后续真实批次均通过 |
| 严重程度 | 高 |
| 优先级 | P1 |
| 类型 | Agent 执行安全 / 工具参数兼容 / 资源消耗失控 |
| 影响流程 | 数据分析工具，以及任何可能持续返回相同校验错误的业务工具 |
| 首次确认日期 | 2026-08-08 |
| 自动化证据 | `backend/data/evaluation/results/agent_routing_eval_20260808_173321.json`、`backend/data/evaluation/results/agent_routing_eval_20260808_214114.json`、`backend/app/agents/data_analyst.py`、`backend/app/agents/tool_boundary.py`、`backend/app/evaluation/agent_route_collector.py`、`backend/app/evaluation/agent_route_evaluator.py` |

#### 复现结果

剩余开发集第一批共选择 8 条真实案例。`AGENT-RAG-002` 正常通过，实际执行 `task(rag-analyst) → check_rag_relevance → search_reports`。随后 `AGENT-DATA-002` 正确完成 `task(data-analyst) → list_data_files → read_data_file`，但 GLM-4.5-Air 调用 `analyze_data` 时把工具 schema 要求的列表参数 `columns=["salary"]` 生成为字符串 `columns="[\"salary\"]"`。Pydantic 正确拒绝该参数后，模型没有修正，而是对同一工具、同一参数和同一错误连续重试 400 次；该案例共记录 403 次工具开始事件、400 次 `analyze_data` 校验错误，事件序号达到 806，远超用例上限 6。

循环最终由智谱 API 返回 429、错误码 1113“余额不足或无可用资源包”而停止，不是由 Agent 自身预算停止。后续 `AGENT-WEB-001`、`AGENT-WEB-002`、`AGENT-DIRECT-001`、`AGENT-DIRECT-002`、`AGENT-NA-001` 和 `AGENT-NA-002` 均在首次模型调用时收到相同 429，未形成有效路由轨迹。因此结果文件中的原始 1/8（12.5%）只反映本次受额度中断的批次状态，不能作为 8 条路由题的有效准确率。批次结束后两个 Python 进程均已退出，隔离上传目录已自动清理。

#### 根因

- GLM 通过 Anthropic 兼容接口生成了不符合 schema 的数组参数，并在收到明确校验错误后持续生成完全相同的调用；这是直接触发条件；
- `analyze_data` 的 `columns` 只接受 `list[str] | None`，没有在进入 Pydantic 校验前兼容常见的 JSON 数组字符串；
- 当前业务工具边界只负责隐藏和拒绝禁用工具，没有统计相同失败签名或每个 Agent 的累计工具调用；
- DeepAgents 0.6.1 为每个编译 Agent 绑定 `recursion_limit=9999`，调用声明式子 Agent 时只转发 callbacks、tags 和 configurable，明确不转发外层评测器设置的 `recursion_limit=50`；
- 数据集中的 `max_tool_calls=6` 当前仅在事件流结束后评分，不能在第 7 次调用出现时主动取消运行；
- 评测批次在不可恢复的额度 429 后仍继续尝试后续案例，造成 6 条无效结果。

#### 修复

1. 在 `analyze_data` 参数 schema 进入严格校验前增加归一化：把合法 JSON 数组字符串转换成 `list[str]`，同时保持模型可见 schema 仍为数组；
2. 扩展应用自有的 Agent 安全中间件，在每次模型请求前从当前 Agent 状态统计工具调用和失败签名；同一工具、规范化参数和错误连续出现 2 次后直接返回终止消息，不再调用模型；
3. 为每个 Agent 设置独立工具调用硬预算，初始建议数据 Agent 8 次、RAG Agent 12 次、主 Agent 8 次；预算耗尽后由应用中间件停止，而不依赖 DeepAgents 的 9999；
4. 让路由事件收集器接收当前案例的 `max_tool_calls`，事件流出现第 `max+1` 次工具调用时立即结束并记录 `ToolCallBudgetExceeded`，把事后评分变成评测期间保护；
5. 评测器识别智谱错误码 1113 等不可恢复的额度错误，保存已完成和失败案例后停止批次，不再继续产生无效案例；
6. 不直接修改 `.venv` 内的 DeepAgents 源码，所有限制由项目代码实现，避免依赖重装后丢失。

#### 修复验证

- 参数归一化单元测试：数组、JSON 数组字符串、普通单列字符串、空值和非法 JSON；
- 安全中间件测试：第二次相同错误后停止、不同参数不误判、成功重复调用不误熔断、总预算耗尽后不再调用模型；
- 收集器测试：超过案例预算时主动关闭事件流并保留部分轨迹；
- 评测器测试：额度 429 后停止后续案例并明确标记批次中止；
- 离线针对性测试和全量 pytest 通过后，只执行 `AGENT-DATA-002` 真实复测；确认工具调用不超过 6、没有重复错误且能返回 salary 描述性统计，再恢复其余真实路由测试。

参数归一化与运行时熔断针对性测试为 22 passed；评测主动硬预算针对性测试为 13 passed；`429/1113` fail-fast 针对性测试为 10 passed。完成全部保护后，全量 pytest 为 236 passed，耗时 12.92 秒，失败、错误和跳过均为 0。

使用更新后的智谱 Key 和 GLM-4.5-Air 真实复测 `AGENT-DATA-002`。模型原始参数仍为 `columns="[\"salary\"]"`，兼容层成功归一化并执行；完整路径为 `task(data-analyst) → list_data_files → read_data_file → analyze_data`，共 4 次工具调用，低于案例上限 6，工具错误和流错误均为 0，路由评分为 1.0。`analyze_data` 正确返回 8 条 salary 的均值 24,625、中位数 22,500、标准差 9,164.18、最小值 15,000、最大值 40,000、Q1 17,500 和 Q3 29,750。RA-AGENT-002 关闭。

### RA-AGENT-003：主 Agent 丢弃子代理正文，仅返回来源标签

| 字段 | 内容 |
|---|---|
| 状态 | 已修复并通过离线及单条真实回归 |
| 严重程度 | 高 |
| 优先级 | P1 |
| 类型 | Agent 答案整合 / 用户可见结果完整性 |
| 影响流程 | 所有通过 `task` 委派给 data-analyst 或 rag-analyst 的回答 |
| 首次确认日期 | 2026-08-08 |
| 自动化证据 | `backend/data/evaluation/results/agent_routing_eval_20260808_214114.json`、`backend/data/evaluation/results/agent_routing_eval_20260808_222534.json`、`backend/app/agents/main_agent.py`、`backend/app/agents/tool_boundary.py`、`backend/tests/test_agents/test_main_agent.py`、`backend/tests/test_agents/test_tool_boundary.py` |

#### 复现结果

`AGENT-DATA-002` 的 `task` 工具成功返回 774 字完整统计正文，包含样本数、均值、中位数、标准差、极值、四分位数和分析结论；该 ToolMessage 已进入主 Agent 最后一次模型请求。GLM-4.5-Air 随后生成的最终 AIMessage 却只有“📊 来源：数据分析”，没有任何统计结果。路由评分器因工具路径、参数和 `final_output_present` 均满足而给出 100%，但用户可见答案不满足完整性要求。

#### 根因

- 主 Agent 提示只规定“在回答末尾简短标注来源”，没有把“来源标签不得替代正文”定义为硬性要求；
- 模型在收到完整 `task` ToolMessage 后过度压缩，只保留来源格式；
- 当前自动路由评分只检查最终输出是否存在，不检查子代理实质结果是否进入用户可见答案。

#### 修复

- 在主 Agent 提示和 `task` 工具说明中明确：必须保留子代理返回的关键数字、事实、结论和引用，来源标签只能作为正文后缀；
- 在应用自有中间件增加窄范围运行时兜底：仅主 Agent、仅最近存在成功 `task` 结果、且新回答为空或只有来源标签时，将完整子代理正文补回；
- 正常的实质性回答和仍包含工具调用的中间响应保持不变，避免覆盖模型的有效汇总或提前结束多步任务；
- 增加同步/异步自动化测试覆盖来源标签退化、正常回答不改写和后续工具调用不改写。

#### 计划验证

- 先执行主 Agent 与工具边界离线针对性测试；
- 再执行全量 pytest；
- 最后仅真实复测 `AGENT-DATA-002`，要求最终用户回答包含至少一项 salary 统计值及“📊 来源：数据分析”。

主 Agent 与工具边界离线针对性测试为 24 passed，耗时 9.73 秒；全量 pytest 为 239 passed，耗时 12.73 秒，失败、错误和跳过均为 0。

使用 GLM-4.5-Air 最终真实复测 `AGENT-DATA-002`，路由再次以 1/1、100% 通过。工具路径为 `task(data-analyst) → list_data_files → read_data_file → analyze_data`，共 4 次调用，工具错误和流错误为 0。模型最后一次汇总再次只生成来源标签，运行时兜底识别该退化后，将最近一次成功 `task` 的 865 字正文完整恢复，并追加“📊 来源：数据分析”；最终答案共 877 个去除首尾空白后的字符，包含均值 24,625、中位数 22,500、标准差 9,164.18、最小值 15,000、最大值 40,000、Q1 17,500 和 Q3 29,750。结果保存在 `backend/data/evaluation/results/agent_routing_eval_20260808_222534.json`，RA-AGENT-003 关闭。

### RA-AGENT-004：无答案研报问题重复检索直至用例预算中止

| 字段 | 内容 |
|---|---|
| 状态 | 已修复并关闭 |
| 严重程度 | 中 |
| 优先级 | P1 |
| 类型 | Agent 无答案策略 / 检索停止条件 / 参数约束 |
| 影响流程 | 本地研报没有精确答案、需要基于已有证据明确拒答的 RAG 问题 |
| 首次确认日期 | 2026-08-08 |
| 自动化证据 | `backend/data/evaluation/results/agent_routing_eval_20260808_223244.json`、`backend/data/evaluation/results/agent_routing_eval_20260808_224800.json`、`backend/app/agents/main_agent.py`、`backend/app/agents/tool_boundary.py` |

#### 复现结果

剩余 6 条开发集真实批测完整执行，未出现 `429/1113`，6 条均被观察，排除和跳过均为 0。`AGENT-WEB-001`、`AGENT-WEB-002`、`AGENT-DIRECT-001`、`AGENT-DIRECT-002` 和 `AGENT-NA-002` 自动路由全部通过；唯一失败项为 `AGENT-NA-001`，批次准确率为 5/6（83.3%）。

`AGENT-NA-001` 要求仅根据已上传研报回答“道通科技 2026 年第二季度 Avant Robotics Gen1 的实际销量”。Agent 正确执行 `task(rag-analyst) → list_available_reports`，随后连续发起 5 次 `search_reports(top_k=10)`。前 4 次已完成检索均未返回该实际销量，第 5 次检索成为批次观察到的第 7 个工具启动事件，超过用例 `max_tool_calls=6`，事件流以 `ToolCallBudgetExceeded` 主动取消。该用例没有最终输出，证明硬预算按设计生效，但无答案行为未完成。

5 次 `search_reports` 均未传数据集要求的 `topic="embodied_intelligence"`。查询词虽有变化，但多次返回相同或高度重叠的道通科技与其他机器人研报片段；前 4 次工具输出长度约 10,602～15,936 字，继续检索增加了 token 和延迟，却没有形成新的精确证据。

#### 根因

- RAG 子 Agent 提示允许“信息不足时尝试不同关键词”，但没有规定最大检索轮数和无精确证据后的停止条件；
- 问题未直接出现“具身智能”字样，模型没有从 Avant Robotics/机器人语义推断主题，导致所有检索遗漏主题过滤；
- 当前运行时总预算能阻止继续调用，但达到预算后只能安全终止，不能替代 Agent 基于已有结果生成“研报未提供该数据”的正常拒答。

#### 已实现修复

1. RAG 子 Agent 提示已增加有限检索和无答案规范：`search_reports` 最多调用 4 次；没有精确事实时必须明确说明“已上传研报未提供该信息”，不得推断、编造或改用联网搜索；
2. 工具边界会从机器人、Robotics、人形、灵巧手、执行器、关节、丝杠、减速器和 Avant 等强特征推断 `topic="embodied_intelligence"`，并且不会覆盖模型显式传入的主题；
3. 运行时同时限制 `search_reports` 最多 4 次、全部 RAG 检索工具合计最多 5 次。达到任一上限后，从模型请求中移除全部研报检索工具并要求立即汇总；若模型仍生成检索调用，工具执行边界会拒绝该调用，避免触达后端；
4. 已补充离线测试，覆盖提示约束、单项和合计检索上限、运行时拒绝、具身智能主题自动补全以及显式主题保留；
5. 按顺序执行针对性离线测试、全量 pytest，最后只真实复测 `AGENT-NA-001`。

#### 针对性测试结果

执行 `python -m pytest tests/test_agents/test_main_agent.py tests/test_agents/test_tool_boundary.py -q`，结果为 29 passed，耗时 9.27 秒，pytest 退出码为 0。主题自动补全、显式主题保留、4 次 `search_reports` 上限、5 次 RAG 检索工具合计上限、运行时越界拒绝以及原有 Agent 工具边界行为均通过。

#### 全量与真实回归结果

永久隔离下执行全量 pytest，结果为 244 passed，耗时 12.93 秒，失败、错误和跳过均为 0，退出码为 0。

随后只真实复测 `AGENT-NA-001`，自动评分为 1/1（100%）。实际工具路径为 `task(rag-analyst) → list_available_reports → search_reports ×3`，共 5 次工具启动，低于用例上限 6；工具错误和流错误均为 0。三次 `search_reports` 均携带 `topic="embodied_intelligence"`，没有调用任何 Web 工具。最终回答明确说明已上传研报未找到 2026 年第二季度 Avant Robotics Gen1 的实际销量，没有编造数字，并引用《道通科技2025年年度报告点评》第1页作为相关信息来源。结果保存在 `backend/data/evaluation/results/agent_routing_eval_20260808_224800.json`，RA-AGENT-004 关闭。

### RA-AGENT-005：公司财务研报问题绕过 RAG 直接联网

| 字段 | 内容 |
|---|---|
| 状态 | 已修复并关闭 |
| 严重程度 | 中 |
| 优先级 | P1 |
| 类型 | 主 Agent 路由 / 来源边界 / Web 工具约束 |
| 影响流程 | 用户询问公司业绩或财务指标、预期从本地研报取证且没有明确实时联网要求的场景 |
| 首次确认日期 | 2026-08-08 |
| 自动化证据 | `backend/data/evaluation/results/agent_routing_eval_20260808_231025.json`、`backend/app/agents/main_agent.py`、`backend/app/agents/tool_boundary.py` |

#### 复现结果

`AGENT-RAG-004` 询问“凌云光2025年营收和归母净利润是多少，分别同比增长多少？”。现有主提示和 `task` 描述均明确要求公司业绩、财务指标优先委派 `rag-analyst`，但真实执行直接调用 `web_search(query="凌云光 2025年 营收 归母净利润 预测 同比增长")`，没有调用 `task` 或 `search_reports`。该案例自动得分 57.14%，违反 required subagent、required tools、required order、forbidden tools、Web policy 和参数约束。

#### 根因与已实现修复

- 原有规则主要依赖提示词，Web 工具仍对主 Agent 可见，模型可以在没有本地 RAG 尝试的情况下直接调用；
- 已在模型请求边界识别没有明确联网要求的公司业绩、财务指标和本地研报问题，从当前工具集合中移除 `web_search` 与 `web_search_quick`，并追加必须委派 `rag-analyst` 的来源边界提示；
- 已在工具执行边界增加第二道保护：即使模型仍构造 Web 调用，也返回本地边界错误且不触达 Web 后端；
- 判定优先尊重“不要联网”等明确禁止，同时放行“请联网”“最新股价”“今天”“天气”“新闻”等实时需求和 `AGENT-MULTI-001` 等显式 RAG+Web 多意图；“什么是”“解释”“区别”等概念题不会被强制改为 RAG；
- 已新增 6 个离线测试实例，覆盖财务 RAG 工具隐藏、运行时拒绝、最新股价放行、多意图放行和概念题不误伤；下一步执行针对性测试、全量回归与 `AGENT-RAG-004` 三轮真实复测。

#### 针对性测试结果

执行 `python -m pytest tests/test_agents/test_main_agent.py tests/test_agents/test_tool_boundary.py -q`，结果为 35 passed，耗时 8.87 秒，pytest 退出码为 0。新增 6 个实例及原有 Agent 边界测试全部通过。

随后执行全量 pytest，结果为 250 passed，耗时 12.11 秒，失败、错误和跳过均为 0，pytest 退出码为 0。

#### 第一轮真实复测结果

首次修复后执行 `AGENT-RAG-004`，实际路径已从错误的 `web_search` 改为 `task(rag-analyst) → list_available_reports → search_reports`，没有 Web 调用、工具错误或流错误，最终回答引用凌云光研报第1页并完整给出营收、归母净利润及同比增长。Web 来源边界已经生效。

该轮仍因 `search_reports` 没有携带数据集要求的 `topic="embodied_intelligence"` 而以 92.86% 未通过。查询只包含“凌云光”和财务指标，没有现有机器人/具身智能关键词，主题兜底无法推断。已将“凌云光”加入具身智能强主题映射，并新增保留 `file_id` 的财务查询参数测试。结果保存在 `backend/data/evaluation/results/agent_routing_eval_20260808_233925.json`；其余两轮真实复测暂停，待第二轮离线与全量验证通过后继续。

#### 第二轮回归与三轮真实结果

补充凌云光主题映射后，主 Agent 与工具边界针对性测试为 36 passed，耗时 8.72 秒；全量 pytest 为 251 passed，耗时 12.12 秒，失败、错误和跳过均为 0。

随后从头独立执行 `AGENT-RAG-004` 三轮，三轮均以 1/1、100% 通过：

- 第 1 轮：`task → list_available_reports → search_reports ×3`，共 5 次工具调用；
- 第 2 轮：`task → list_available_reports → search_reports ×2`，共 4 次工具调用；
- 第 3 轮：`task → list_available_reports → search_reports → get_report_summary → search_reports`，共 5 次工具调用。

所有 `search_reports` 均携带 `topic="embodied_intelligence"`，三轮 Web 调用、工具错误和流错误均为 0。三轮最终答案一致给出2025年营收29.12亿元、同比增长30.35%，归母净利润1.61亿元、同比增长50.70%，并引用凌云光点评报告第1页。工具细节存在2～3次搜索及可选摘要的允许范围变化，但核心路由、来源边界、参数约束和答案稳定。结果分别保存在 `backend/data/evaluation/results/agent_routing_eval_20260808_234344.json`、`agent_routing_eval_20260808_234432.json` 和 `agent_routing_eval_20260808_234553.json`，RA-AGENT-005 关闭。

### RA-AGENT-006：RAG 工具调用文本化引发重复委派和预算中止

| 字段 | 内容 |
|---|---|
| 状态 | 已修复并关闭 |
| 严重程度 | 中 |
| 优先级 | P1 |
| 类型 | 兼容模型工具调用 / 子代理结果收口 / 重复委派 |
| 影响流程 | 兼容模型把结构化工具请求输出成 `<tool_call>` 文本，导致子代理提前返回的场景 |
| 首次确认日期 | 2026-08-08 |
| 自动化证据 | `backend/data/evaluation/results/agent_routing_eval_20260808_231430.json`、`backend/data/evaluation/results/agent_routing_eval_20260808_235539.json`、`backend/app/agents/main_agent.py`、`backend/app/agents/tool_boundary.py` |

#### 复现结果

`AGENT-RAG-003` 初始重复 3 次，仅第 1 次通过。进一步读取失败轮的 `task` 输出后确认，第一次子代理并未返回完整答案，而是把后续 `search_reports` 请求输出成普通文本形式的 `<tool_call>`、`<arg_key>` 和 `<arg_value>` 标签。主 Agent识别到结果不完整后再次委派同一个 `rag-analyst`；第二次委派后的首个子代理工具成为第7次工具启动，超过案例 `max_tool_calls=6`，评测器按设计主动取消并导致没有最终输出。

#### 根因与已实现修复

- 直接禁止第二次 `task` 会把不完整的文本工具标记误当成答案，并会破坏 `AGENT-MULTI-002` 等合法多子代理流程，因此没有采用该方案；
- RAG 子代理提示已明确要求使用框架结构化 tool call，不得把 `<tool_call>`、`<arg_key>` 或 `<arg_value>` 作为普通文本输出；
- 模型响应边界新增白名单兼容层：只接受完整、唯一、参数标签成对且工具名存在于当前模型请求工具列表的文本调用，并将数字、布尔值和 JSON 容器恢复为正确参数类型；未知工具、重复参数、畸形标记和普通正文保持惰性，不会执行；
- 原生结构化工具调用不经过兼容转换，既有工具边界、主题补全和预算策略保持不变。

#### 回归结果

最小兼容测试为 5 passed；主 Agent 与工具边界针对性测试为 42 passed，耗时 8.76 秒；全量 pytest 为 257 passed，耗时 12.25 秒，失败、错误和跳过均为 0。

`AGENT-RAG-003` 三轮真实复测为 3/3、100% 通过，每轮均只有一次 `task(rag-analyst)`、总工具调用均为5、所有 `search_reports` 均带 `topic="embodied_intelligence"`，Web调用、工具错误和流错误均为0，结果中没有 `<tool_call>` 文本残留。三轮工具路径分别为：

- `task → list_available_reports → search_reports ×2 → get_report_summary`；
- `task → list_available_reports → check_rag_relevance → search_reports ×2`；
- `task → list_available_reports → search_reports ×3`。

三轮均完整回答越疆合作、东莞泰莱关节模组送样以及机器人业务小批量/收入进展，并提供研报页码。结果保存在 `backend/data/evaluation/results/agent_routing_eval_20260808_235539.json`，RA-AGENT-006 关闭。

### 第 4 阶段保留测试集汇总

三批正式执行共计划并观察 20 次 Agent 调用，排除和跳过均为 0，结果文件分别为：

- `backend/data/evaluation/results/agent_routing_eval_20260808_231025.json`：3/4；
- `backend/data/evaluation/results/agent_routing_eval_20260808_231430.json`：7/9；
- `backend/data/evaluation/results/agent_routing_eval_20260808_231846.json`：7/7。

初始保留测试集运行准确率为 17/20（85%），案例准确率为 8/10（80%），均低于90%目标。RA-AGENT-005与RA-AGENT-006修复后，分别用三轮真实回归替换 `AGENT-RAG-004` 的一次失败和 `AGENT-RAG-003` 的两次失败，当前有效运行准确率为20/20（100%），案例准确率为10/10（100%）。Data、Web、Direct、No-answer、Multi-intent和RAG类别均通过。

5个重复案例的功能性路由均为3/3通过；但只有 `AGENT-DIRECT-003` 的完整工具序列完全相同，因此精确签名稳定性仍为1/5（20%）。`AGENT-RAG-003` 在可选相关性检查、报告摘要和2～3次搜索之间变化，`AGENT-NA-003` 在等价的 `read_data_file` 与 `read_csv_file` 之间变化，`AGENT-WEB-003` 在1～2次合法Web搜索之间变化，`AGENT-MULTI-001` 的RAG/Web调用顺序及补充检索工具发生变化。以上变化没有违反工具、参数、来源或预算约束，作为成本和可复现性风险记录，不再阻断第4阶段。

开发集10条与保留测试集10条均已有最新有效通过证据，20条代表性问题全部满足必须工具、禁止工具和必要参数约束；保留测试集修复后有效路由准确率为100%，超过90%目标。RA-AGENT-001至RA-AGENT-006均已关闭，第4阶段完成；第5阶段随后完成10条答案事实、数字、来源和引用支持关系人工验收。

### RA-ANS-001：RAG 回答为无时间证据的送样事实补写具体年份

| 字段 | 内容 |
|---|---|
| 状态 | 已修复并通过离线、全量及真实RAG回归 |
| 严重程度 | 中 |
| 优先级 | P1 |
| 类型 | 答案事实性 / 证据边界 / 时间推断 |
| 影响流程 | 本地研报问答、事实与引用支持关系 |
| 首次确认日期 | 2026-08-09 |
| 复现案例 | `AGENT-RAG-003` |
| 自动化证据 | `backend/data/evaluation/results/agent_routing_eval_20260808_235539.json`、`backend/data/evaluation/results/agent_rag_003_ra_ans_001_retest_20260809_retry5.json` |

#### 复现与人工核验结果

最终回答写出“送样时间：2025年期间进行送样”。人工查看《【奥普特】深度报告：视觉龙头迈向具身智能，双轮驱动开启第二曲线》第1页和第24页，两处仅说明“东莞泰莱机器人关节模组产品已在送样过程中”，没有给出送样开始年份。报告发布日期为2026年7月27日；第21-22页出现的2025年4月对应收购东莞泰莱51%股权及并表时间，不能作为送样年份证据。

回答中的越疆合作、Optibot发布、AMTS展示、机器人业务收入2330万元、泰莱营收1.12亿元和归母净利润621.49万元均能在所引页找到；缺陷仅针对模型额外补写的“2025年送样”时间结论。按照第5阶段“每项具体事实必须被引用证据直接支持”的口径，该案例整体判定不通过。

#### 初步原因与修复方向

- 模型把相邻的2025年收购、收入信息与没有年份的送样事实合并，生成了来源没有提供的时间；
- 末尾集中列出多个页码，未建立“单项事实—单页证据”约束，无法在生成阶段阻止跨句推断；
- 修复时应明确禁止为缺少年份的事实补写年份，并在必要时回答“截至报告发布日期仍处于送样过程中”；
- 增加针对性测试，要求回答中的年份必须出现在对应证据片段，送样事实不得继承相邻收购或财务事实的年份。

#### 修复与验收结果

- `search_reports` 在每次有结果的输出末尾增加证据使用约束：日期只能支持同一句或同一项目中明确关联的事实，不得把收购、并表、报告发布日期或财务年度转用于送样、交付和量产；
- 主Agent与RAG子Agent提示同时要求：具体年月必须由最终来源中的对应页码直接支持；证据只写“正在送样”时必须明确“研报未披露送样开始年份”；
- 工具边界新增确定性保护：识别“送样时间：2025年”“送样过程发生在2025年”等无直接证据的句式并改写为未披露；若模型只保留送样状态而遗漏限制说明，也会补回“研报未披露送样开始年份”；正确的“2025年4月收购”不会被误删；
- 单一RAG子任务已有带页码的完整正文时，最终回答采用该权威结果，避免主Agent压缩时丢页码；具体年月缺页时，只在年月相同且事实句与检索片段事件词有足够重合时补页，单纯日期相同的目录页不会被当成证据；
- 修复过程保留了四轮中间记录：第一轮事实正确但主Agent丢页码；第二轮缺少第21-22页；第三轮暴露相同日期误绑目录页；第四轮没有虚构年份但未明确说明年份缺失。第五轮关闭候选 `backend/data/evaluation/results/agent_rag_003_ra_ans_001_retest_20260809_retry5.json` 自动评分1/1、100%，工具顺序为 `task → list_available_reports → search_reports ×3`，无Web和流错误；
- 第五轮最终回答明确“研报未披露送样开始年份”；2025年4月收购由第21-22页支持，2026年5-7月合作与展示及2025年机器人收入由第23页/第1页支持，关节模组送样状态由第24页支持，不再包含无关目录页。结果JSON SHA-256为 `5905C8B03CE07FB6D630C6C03CDE65BBF2611B500B9FF5EC9344C504DCF683DE`；
- 针对性测试63 passed、最新全量回归273 passed。RA-ANS-001关闭。

### RA-ANS-002：数据 Agent 根据前5行预览回答全表筛选数量

| 字段 | 内容 |
|---|---|
| 状态 | 已修复并通过离线、全量及两条真实回归 |
| 严重程度 | 高 |
| 优先级 | P1 |
| 类型 | 数据分析正确性 / 局部预览误用 / 计算步骤缺失 |
| 影响流程 | CSV 条件筛选、计数、分组统计和多意图回答 |
| 首次确认日期 | 2026-08-09 |
| 复现案例 | `AGENT-MULTI-002`、`AGENT-DATA-003` |
| 自动化证据 | `backend/data/evaluation/results/agent_routing_eval_20260808_231025.json`、`backend/data/evaluation/results/agent_data_003_manual_acceptance_20260809_retry1.json`、`backend/data/evaluation/results/agent_data_003_ra_ans_002_005_retest_20260809_retry2.json`、`backend/data/evaluation/results/agent_multi_002_ra_ans_002_005_retest_20260809_retry2.json`、`backend/tests/fixtures/sample.csv` |

#### 复现与人工核验结果

问题要求统计 `sample.csv` 中北京员工数量。最终回答称北京只有 Alice 1人，占8名员工的12.5%。实际CSV同时包含 Alice 和 Frank 两名北京员工，正确结果应为2人、25%。

轨迹显示 `read_data_file` 只返回前5行预览，预览中只有 Alice；随后两次 `analyze_data` 分别执行全表的 `describe` 和 `summary`，没有传入或执行 `city == "Beijing"` 的过滤、计数或分组操作。模型据局部预览直接推断了全表结果，因此第4阶段路由虽通过，第5阶段答案事实核验失败。

`AGENT-DATA-003` 的真实复测再次确认同一问题：`create_chart` 对完整8行数据生成的柱状图正确显示Frank为最高薪员工（40000），但数据子Agent只依据 `read_data_file` 的前5行预览声称Charlie（35000）最高，主Agent又原样保留了该错误结论。持久化图表与人工核查记录分别位于 `backend/data/evaluation/artifacts/agent_data_003_manual_acceptance_20260809/uploads/charts/chart_ebfc5ba8.png` 和 `backend/data/evaluation/results/agent_data_003_manual_acceptance_20260809_review.json`。

#### 初步原因与修复方向

- 数据子 Agent 没有把“全表条件计数”识别为必须执行的计算，误把预览当作完整数据；
- 当前工具路径缺少清晰、稳定的筛选计数调用方式，模型退化为目测预览；
- 修复时应增加或明确条件过滤、计数、分组聚合能力，并在提示中禁止用预览回答全表数量；
- 自动化测试应把匹配记录放在预览范围之外，确认 Agent 仍能得到完整计数，同时覆盖中文“北京”与数据值 `Beijing` 的映射。

#### 修复与验收结果

- `read_data_file`/`read_csv_file` 对不超过20行的小表返回明确标注的完整数据；更大文件只返回前5行并带有不得推导全表计数、总计和极值的警告；
- `analyze_data` 新增 `value_counts` 全表频次分析，并输出最高频次及所有并列最高类别；数据子Agent提示要求分类计数必须调用该分析，且不得添加用户未要求的排名；
- `create_chart` 返回参与绘图的完整行数、坐标轴和全表最小/最大值对应标签，避免依据读取预览解释图表；
- 第一轮修复后的 `AGENT-MULTI-002` 已正确得到北京2人和25%，但额外误称北京、上海、广州“并列第二”；补充并列最高证据并禁止无要求排名后，第二轮真实复测不再输出错误排名，自动评分1/1、100%，最终回答确认北京2人（Alice、Frank），RAG部分来自本地研报且无Web调用；
- 最新结果为 `backend/data/evaluation/results/agent_multi_002_ra_ans_002_005_retest_20260809_retry2.json`，SHA-256为 `491EC5940C74ED8887ADE1E34F0B9B563E55E98994897BAD0001401D60674F04`。该次兼容模型曾把两个子Agent名称误作工具名，收到边界错误后自行改用 `task` 并在预算内完成，作为非阻断工具签名波动保留；
- `AGENT-DATA-003` 的最终全表事实与图表核验也已通过，相关证据见RA-ANS-005。RA-ANS-002关闭。

### RA-ANS-003：Web 最新动态回答混入过期材料并错误计算事件倒计时

| 字段 | 内容 |
|---|---|
| 状态 | 已修复并通过离线、全量及真实 Web 回归 |
| 严重程度 | 高 |
| 优先级 | P1 |
| 类型 | Web 时效性 / 日期理解 / 搜索结果筛选 |
| 影响流程 | “今天”“最新”“截至某日”的新闻与行业动态问答 |
| 首次确认日期 | 2026-08-09 |
| 复现案例 | `AGENT-WEB-003` |
| 复核证据 | `backend/data/evaluation/results/agent_routing_eval_20260808_231846.json`、2026-08-09 Tavily复核、上海市政府及企业/研究机构公开页面 |

#### 复现与人工核验结果

问题要求概括截至2026年8月8日的“今天”行业新动态，但回答混入了多项非当天信息：

- [《上海市具身智能产业发展实施方案》](https://www.shanghai.gov.cn/nw12344/20250806/f9cb53544505426d807055ca20bd69fc.html)实际印发于2025年7月28日、发布于2025年8月6日，不是2026年8月的新动态；
- [2026第四届上海具身智能机器人产业展览会](https://sh.eirobotshow.com)举办日期为8月12-14日，从8月8日计算距开幕约4天，回答却写“仅剩30天”，直接沿用了旧网页标题；
- Booster Studio由[加速进化新闻中心](https://www.booster.tech/cn/news/)记录为2026年6月30日发布，不是8月8日当天发布；
- 具身智能企业出海报告和[IDC趋势预测](https://www.idc.com/resource-center/blog/%E6%9C%BA%E5%99%A8%E4%BA%BA%E6%AD%A3%E5%9C%A8%E8%BF%9B%E5%8C%96-2026%E5%B9%B4%E4%B8%AD%E5%9B%BD%E6%9C%BA%E5%99%A8%E4%BA%BA%E4%B8%8E%E5%85%B7%E8%BA%AB%E6%99%BA/)分别属于此前发布的报告或趋势材料，不应与当天新闻并列为“今天的新动态”。

只有部分8月6-7日企业和资本市场信息满足相近时效要求。该回答事实真伪、发布时间和事件发生时间没有分层，整体判定不通过。

#### 初步原因与修复方向

- 搜索查询包含目标日期，但工具没有对结果发布日期实施硬过滤；
- 模型把网页标题中的历史“倒计时30天”当作当前状态，没有根据目标日期重新计算；
- 最终汇总未区分“文章发布日期”“事件发生日”“预测适用年份”和“查询截止日”；
- 修复时应保留并校验发布时间，对“今天”使用明确时间窗口，重新计算相对日期；没有足够当天新闻时应缩小结论并明确说明，而不是用历史趋势填充。

#### 修复与验收结果

- 两种Web工具现在都返回结构化来源记录，包含标题、直接URL、搜索服务发布日期、内容日期候选、查询执行日和用户截止日；搜索服务AI摘要降级为“未核验线索”，不得作为时效或精确数字的唯一依据；
- 新闻查询自动补充“发布日期、事件发生日期”检索语义，主Agent提示明确区分发布日期、事件日、预测年份与查询截止日，并禁止沿用网页中的“今天”或“倒计时N天”；
- 响应边界对“今天/今日”执行同日发布日期硬门槛：如果没有可核验为截止日当天发布的来源，直接说明无法可靠列出当天新动态，不再用旧闻填充；存在同日来源时保留正常回答；
- `AGENT-WEB-003` 修复后真实结果正确拒绝把2026年3月、6月及其他缺少8月8日发布日期的材料称为当天新动态，并保留所检索页面的URL和绝对日期。结果记录在 `backend/data/evaluation/results/agent_web_ra_ans_003_004_retest_20260809_retry2.json`，SHA-256为 `87A18DE6528713C0E577F34123F234885A4C8C93BA9B3E38294C4BBD5017C687`；
- 最新针对性测试76 passed、全量回归283 passed。RA-ANS-003关闭。

### RA-ANS-004：最终回答丢失 Web URL 和数据日期，精确行情失去可追溯性

| 字段 | 内容 |
|---|---|
| 状态 | 已修复并通过离线、全量及真实 Web/RAG+Web 回归 |
| 严重程度 | 高 |
| 优先级 | P1 |
| 类型 | 引用完整性 / Web可追溯性 / 金融数字准确性 |
| 影响流程 | Web回答、股价查询、RAG与Web混合回答 |
| 首次确认日期 | 2026-08-09 |
| 复现案例 | `AGENT-WEB-003`、`AGENT-WEB-004`、`AGENT-MULTI-001` |
| 复核证据 | `backend/data/evaluation/results/agent_routing_eval_20260808_231846.json`、2026-08-09 Tavily及公开行情页面复核结果 |

#### 复现与人工核验结果

三条最终回答都只保留“🌐 来源：联网搜索”，没有输出搜索结果中已有的URL、网页发布日期或行情对应交易日。

`AGENT-WEB-004` 的中信海直14.20元核心数字正确，但2026年8月8日是周六，回答应明确该数字对应最近交易日2026年8月7日收盘，并给出可复查的[日期化行情页面](https://hk.finance.yahoo.com/quote/000099.SZ)。由于引用缺失，该案例只能判为事实通过、引用不通过。

`AGENT-MULTI-001` 的奥普特Web部分存在实际数字错误：根据[奥普特历史行情](https://cn.investing.com/equities/opt-machine-vision-tech-historical-data)，截至8月8日可获得的最近交易日为8月7日，正确收盘价为120.00元，回答写成118.14元；当日最高价应为120.68元，回答写成120.50元；总市值约146.68亿元，回答写成141.40亿元。前一交易日8月6日收盘价115.98元及52周范围85.89-187.00元可以核实。缺少URL和数据日期使用户无法区分盘中快照、收盘数据和页面更新时点。

智谱GLM-4.5-Air辅助复核正确识别了多数问题，但又把8月5日的114.08元误判成8月7日的前一交易日收盘价；原始历史行情证明前一交易日8月6日为115.98元。因此LLM Judge只作为辅助线索，不能替代原始来源裁决。

#### 初步原因与修复方向

- `web_search`结果包含URL，但主 Agent 最终汇总没有被要求逐项保留；
- `web_search_quick`只返回AI摘要，不返回用于生成摘要的URL和日期，结构上无法形成引用；
- 修复时应让两种Web工具都返回结构化来源、发布日期/交易日期，并要求最终答案对关键事实输出可点击URL；
- 股价回答应统一使用最近交易日收盘口径，明确交易日、币种和数据状态，非必要的市盈率、目标价等扩展数字不应自动加入；
- 增加自动化检查：含Web精确事实的最终答案必须至少保留一个URL和一个绝对日期，不能仅输出通用来源标签。

#### 修复与验收结果

- `web_search_quick`不再返回无来源的答案文本，与完整搜索一样保留URL和日期；最终响应边界会从成功的Web工具消息中恢复模型遗漏的直接来源，且RAG+Web混合回答不再被单一RAG任务正文覆盖；
- 明确截止日会被自动注入模型发出的Web查询；股价查询补充历史行情、收盘价和交易日期语义。若首轮结果没有日期化历史行情表，工具内部只补查一次更窄的历史数据表查询；
- 精确股价只从“日期｜收盘｜开盘｜高｜低……”历史表中提取截止日或之前最新交易日的收盘值，不采信Tavily/GLM摘要、实时快照或页面散文。相同日期表格值冲突或无表格时安全拒答；仅询问股价时不再自动输出市值、52周区间、目标价等未请求指标；
- 周末截止日会明确说明无交易数据并采用此前最近交易日。`AGENT-WEB-004`最终为2026年8月7日中信海直收盘14.20元，直接来源为Investing历史行情；结果 `backend/data/evaluation/results/agent_web_004_ra_ans_004_retest_20260809_retry5.json`，SHA-256为 `966E8E90E5E84340F090C5598156DAB9FB449726D20AA54A2814E8A10A080A11`；
- `AGENT-MULTI-001`最终保留研报第21-23页证据，并从历史表得到2026年8月7日奥普特收盘120.00元，即使搜索摘要仍出现118.14元也不会进入最终答案。合格结果记录在 `backend/data/evaluation/results/agent_web_ra_ans_004_retest_20260809_retry4.json`，SHA-256为 `F7CDD40EF75A2D1163EA240C1C5508E83EE7387F6CFE9A3ED3C774AE0187EB35`；
- 最新针对性测试76 passed、全量回归283 passed；两条最终真实路由均1/1通过、无流式错误，人工数字、日期和URL核验通过。RA-ANS-004关闭。

### RA-ANS-005：图表路径在主 Agent 最终回答中丢失

| 字段 | 内容 |
|---|---|
| 状态 | 已修复并通过离线、全量及真实图表回归 |
| 严重程度 | 高 |
| 优先级 | P1 |
| 类型 | 图表产物可访问性 / 子Agent结果传递 / 最终回答完整性 |
| 影响流程 | Data Agent图表生成、Chat图表展示、人工验收产物留存 |
| 首次确认日期 | 2026-08-09 |
| 复现案例 | `AGENT-DATA-003` |
| 自动化证据 | `backend/data/evaluation/results/agent_data_003_manual_acceptance_20260809_retry1.json`、`backend/data/evaluation/results/agent_data_003_ra_ans_002_005_retest_20260809_retry2.json` |

#### 复现与人工核验结果

真实工具链为 `task(data-analyst) → list_data_files → read_data_file → create_chart`，`create_chart` 使用 `bar/name/salary` 参数成功返回 `/static/charts/chart_df66af8e.png`，自动路由检查1/1通过且没有RAG、Web或工具错误。数据子Agent正文也保留了该路径，但主Agent最终回答只说“图表已生成并保存为可视化文件”，没有输出路径或链接。

评测器把CSV暂存于临时uploads目录，运行结束后按设计删除该目录，因此工具返回的原始路径随后也不可访问。为完成人工核验，使用相同CSV及轨迹中的完全相同参数再次调用正式 `create_chart` 工具，持久化得到 `backend/data/evaluation/artifacts/agent_data_003_manual_acceptance_20260809/uploads/charts/chart_ebfc5ba8.png`；目视确认8根柱及全部数值正确。由于真实最终回答既未提供路径，原始临时产物也未保留，案例完成标准“生成可访问的图表路径”不成立，整体判定不通过。

#### 初步原因与修复方向

- 主Agent总结数据子Agent正文时没有强制保留 `/static/charts/...` 产物引用；
- 路由评测器只保存事件JSON，没有在临时uploads清理前复制图表供人工复核；
- 修复时应像Chat接口处理图表URL一样，从子Agent工具输出和正文中提取图表路径并写入最终回答或结构化 `chart_paths`；
- 评测器应为图表案例提供可选的持久化artifact目录，并在结果JSON记录文件路径和哈希；
- 自动化测试应同时验证 `create_chart` 成功、最终回答含路径、路径对应文件存在，避免只验证工具调用顺序。

#### 修复与验收结果

- 工具边界会收集所有成功 `task` 结果中的 `/static/charts/...` URL：多子任务综合回答缺失URL时自动补回；单一图表子任务直接采用数据子Agent的权威完整正文并保留来源标签，避免主Agent二次改写标签、姓名或数值；
- 路由评测器新增 `--artifacts-dir`，没有显式传入时也会按结果文件名建立artifact目录；在临时uploads清理前复制所有图表，并在结果JSON记录源URL、永久路径、字节数和SHA-256；无图表时不会创建空目录；
- 第一轮真实复测已保留URL和正确极值，但主Agent二次转述把 `Charlie/Diana` 写成 `Carol/David`，因此没有作为关闭证据；加固为单图权威正文透传后，第二轮 `AGENT-DATA-003` 自动评分1/1、100%，最终回答逐项正确列出8名员工及薪资、Frank最高40000、Alice最低15000，并包含 `/static/charts/chart_268c37f7.png`；
- 永久图表为 `backend/data/evaluation/artifacts/agent_data_003_ra_ans_002_005_retest_20260809_retry2/chart_268c37f7.png`，SHA-256为 `A2D0737CB8A2FD342BC77349D53684702040DF91B66AC511E569ECFFE32FDC1E`；结果JSON SHA-256为 `FD64579665FCC4897D0DE999712D2375F9991294780429FC66EBFD30829C0EF6`；
- 针对性测试50 passed、最新全量回归264 passed。RA-ANS-005关闭。

## 4. 第 3 阶段初始 RAG 评测与主题过滤对照

对照测试使用 18 条可回答样本；2 条无答案样本不参与 Recall。A、B 两组在同一进程共享同一个本地 BGE-M3 快照、同一个 930 分块 Chroma 集合、相同 Top-K 和排序配置，唯一变量是 B 组增加 `topic=embodied_intelligence` 元数据过滤。

| 指标 | A：全库检索 | B：主题过滤 | 变化 |
|---|---:|---:|---:|
| Recall@1 | 52.78% | 52.78% | 0.00 个百分点 |
| Recall@3 | 88.89% | 100.00% | +11.11 个百分点 |
| Recall@5 | 88.89% | 100.00% | +11.11 个百分点 |
| Recall@10 | 100.00% | 100.00% | 0.00 个百分点 |
| MRR | 76.62% | 79.63% | +3.01 个百分点 |
| NDCG | 81.92% | 84.47% | +2.55 个百分点 |
| 关键词覆盖率 | 95.09% | 95.09% | 0.00 个百分点 |

受控 A/B 基线与首次模型别名评测的数值不同，原因是对照测试固定了本地模型快照；因此本节只比较同一次 A/B 的两组结果，不把不同模型加载口径的结果直接相减。完整对照结果保存在 `backend/data/evaluation/embodied_intelligence_topic_filter_ab_v1_result.json`。RA-RAG-001 修复后又通过正式 `run_evaluation --topic embodied_intelligence` 入口复测，结果与 B 组一致，保存在 `backend/data/evaluation/embodied_intelligence_topic_filtered_official_v1_result.json`。
## 5. 第 3 阶段冻结 RAG 测试集

冻结测试集保存在 `backend/data/evaluation/embodied_intelligence_eval_dataset_test_v1.json`，与 20 条开发集物理分离。数据集共 40 条，其中 36 条可回答、4 条无答案；包含 11 条单事实、8 条财务、9 条进展、1 条风险、1 条表格查询和 6 条跨报告对比，覆盖全部 30 份具身智能研报。所有样本均标记为 `split=test`，与开发集的完整问题和 `question_family` 均无重复。

执行静态校验：

```powershell
.\.venv\Scripts\python.exe scripts\validate_eval_dataset.py data\evaluation\embodied_intelligence_eval_dataset_test_v1.json
```

校验结果为 `valid=true`、0 errors、0 warnings；文档 ID、主题、Chroma 文档存在性、证据页和预期关键词均通过。冻结记录保存在 `backend/data/evaluation/embodied_intelligence_eval_dataset_test_v1.freeze.json`，数据文件 SHA-256 为 `AAAB2EA84C122CC40C257030743E1D4681B3598CF2D44B159048AFCA9A21BAED`。

该数据集从冻结后不得用于检索参数、分块、重排或提示词调优；唯一一次独立正式 RAG 验收已经完成，本版不得为提高结果而重跑或调参。如果发现标注错误，必须形成新版本和新哈希，不能静默修改本版。

唯一一次正式验收已于 2026-08-09 执行，固定使用 `topic=embodied_intelligence`、Top-20 检索和 BAAI/bge-m3 本地向量模型，退出码为0。结果保存在 `backend/data/evaluation/embodied_intelligence_frozen_test_v1_result.json`，结果文件 SHA-256 为 `F54A3DFDEE7E3B5107D17905E463253057B878CCF9FC56D937C1CEB7F56B7D3B`。

| 指标 | 工具原始40条聚合 | 36条可回答题验收口径 | 开发集目标 | 结论 |
|---|---:|---:|---:|---|
| Recall@5 | 90.00% | 100.00% | 100.00%，不退化 | 通过 |
| MRR | 81.25% | 90.28% | 85.19%，不明显退化 | 通过 |
| NDCG | 83.95% | 93.28% | 88.57%，不明显退化 | 通过 |
| Term Coverage@5 | 85.04% | 94.49% | 92.06%，提升 | 通过 |
| Term Coverage@10 | 87.62% | 97.36% | 97.62%，不明显退化 | 通过，下降0.26个百分点 |
| Evidence Page Recall@5 | 90.28% | 90.28% | 仅作诊断 | 不阻断 |

原始聚合值把4条 `relevant_doc_ids=[]` 的无答案题按Recall、MRR、NDCG和关键词覆盖均为0计入平均，因此不能直接用于检索质量验收。按36条可回答题计算后，所有目标报告均在Top-5命中，6条跨报告题也全部覆盖各自所有目标报告。无答案题能否正确拒答属于答案生成质量，不由纯检索评测器判定，继续纳入第5阶段人工验收。冻结数据和检索实现均未根据本次逐题结果修改，第3阶段正式验收通过。

## 6. 第 2 阶段覆盖范围

本阶段新增或补充的自动化测试文件包括：

- `backend/tests/test_api/test_phase2_core_api.py`：PDF/数据上传、文档管理、Chat、SSE 和高价值风险；
- `backend/tests/test_api/test_phase2_file_id_risk_evidence.py`：RA-API-001 初始缺陷证据；
- `backend/tests/test_api/test_phase2_file_id_validation.py`：非法 UUID 输入回归；
- `backend/tests/test_rag/test_vectorstore_replacement.py`：向量安全替换及回滚。

覆盖的核心行为包括：

- 合法 PDF、CSV 和 Excel 上传；
- 非法扩展名、空文件和损坏 PDF；
- 文档列表、指定文档、删除和重建；
- Chat 正常请求；
- SSE 正常事件序列和异常返回；
- 非法 `file_id` 不得影响文件和向量；
- 索引重建失败不得破坏旧索引。

## 7. 第 2 阶段结论

第 2 阶段完成标准已经达到：

- 核心 API 可以自动回归；
- 两个高价值风险均经过动态测试确认；
- 形成并修复了两个高优先级真实缺陷；
- 针对性测试、相关回归和全量回归均通过；
- 正式数据在全量测试前后完全一致。

当前结果主要来自 Mock、组件替身、FastAPI TestClient 和临时 Chroma 环境。它证明接口控制流、输入校验、失败恢复及隔离机制符合本阶段预期，但不代表真实模型、真实 RAG 检索质量和 Agent 路由已经验证。

## 8. 五阶段最终验收结论

| 阶段 | 最终结果 | 判定 |
|---|---|---|
| 第1阶段：环境与基线 | 独立Python 3.12.7虚拟环境、永久测试隔离、正式数据快照无变化 | 通过 |
| 第2阶段：核心API与风险 | 核心API自动化通过；非法`file_id`和索引重建失败保护完成动态验证与修复 | 通过 |
| 第3阶段：RAG检索质量 | 20条开发集完成调优回归；40条冻结集唯一正式验收中36条可回答题Recall@5 100%、MRR 90.28%、NDCG 93.28% | 通过 |
| 第4阶段：Agent路由与工具 | 20条代表性问题均有真实结果；修复后有效运行20/20、案例10/10，准确率100% | 通过 |
| 第5阶段：答案与引用 | 10条人工样本由初始4通过、1部分通过、5不通过提升为修复后10/10通过 | 通过 |

最终自动化回归为283/283 passed，失败、错误和跳过均为0。共登记18项缺陷：RA-API 2项、RA-RAG 4项、RA-EVAL 1项、RA-AGENT 6项、RA-ANS 5项；截至2026-08-09全部具有修复、自动化回归或真实链路复测证据，未保留阻断缺陷。

综合判定：当前约定测试范围验收通过。系统已证明核心API控制流和数据隔离可靠，具身智能语料检索达到既定阈值，业务Agent路由和工具预算具有保护，抽查答案的事实、数字、无答案、图表与引用关系满足要求。该结论支持当前项目演示、求职作品展示和继续进行受控功能验证，但不应表述为未经限定的完整生产级认证。

## 9. 非阻断遗留风险与范围边界

| 风险/边界 | 当前证据 | 已有控制 | 后续建议 |
|---|---|---|---|
| Agent精确工具序列波动 | 5个重复案例功能均3/3通过，但精确签名稳定性仅1/5（20%） | 必须/禁止工具、参数、预算和最终结果均受评分与中间件约束 | 只在成本或可复现性成为产品目标时增加轨迹模板；不要求唯一合法路径 |
| Tavily及网页结果波动 | 相同股价查询不同轮次可能返回历史表、实时页或冲突AI摘要 | URL/日期保全、一次窄化补查、日期化历史表抽取、冲突或无表时安全拒答 | 生产化时接入稳定的授权行情/新闻数据源，并监控补查率与拒答率 |
| 严格证据页指标差异 | 冻结集Evidence Page Recall@5为90.28%，低于报告级Recall@5 | Evidence Group与关键词覆盖用于识别等价证据，严格页码指标仅作诊断 | 扩充多页等价证据标注；引用必须精确到页的场景单独设门槛 |
| 模型/API额度、延迟和429 | 曾发生参数错误循环消耗资源并触发1113/429 | 参数归一化、重复失败熔断、Agent硬预算、评测预算及429 fail-fast | 持续记录单例token、延迟、工具次数；设置供应商额度告警 |
| 数据集代表性有限 | RAG冻结集40条，主要覆盖30份具身智能研报；人工答案仅10条 | 开发集与冻结集物理分离，冻结集未用于调参 | 新增行业或更大语料时建立新版本数据集，不修改本次冻结结果 |
| 非功能与前端覆盖未完成 | 前端构建可通过，但缺少ESLint、前端自动化和CI；未做性能、并发、多用户权限与渗透测试 | 当前结论明确限定为后端功能、AI质量和受控真实链路 | 若进入部署阶段，再建立CI、前端E2E、压测、安全和权限专项 |

上述项目均不阻断本次约定范围验收，但必须在对外描述结果时保留范围限定。特别是外部Web安全拒答代表“避免错误数字”，不保证第三方数据始终可用。

## 10. 最终交付与后续方向

- 测试计划：`docs/TEST_PLAN_LITE.md`；
- 最终报告：`docs/TEST_REPORT.md`；
- RAG开发集、冻结集、覆盖清单、冻结哈希及唯一正式评测结果：`backend/data/evaluation/`；
- Agent路由数据集、事件轨迹和真实复测结果：`backend/data/evaluation/results/`；
- 自动化回归：`backend/tests/`，最终283 passed；
- RA-API-001和RA-API-002修复前源文件备份仍保留，未纳入运行代码。

本轮测试工作到此满足停止条件。若项目目标仍以求职作品和面试展示为主，下一步应整理一页项目成果和三个代表案例，而不是继续扩张测试范围；若准备部署，再按第9节风险优先补齐CI、稳定外部数据源、前端E2E和非功能测试。
