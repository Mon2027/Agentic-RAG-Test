# Report Analysis Agent

基于 DeepAgents 架构与 Agentic RAG 的多智能体协同研报分析系统。

## 功能

- PDF 研报上传与智能问答
- CSV/Excel 数据分析与可视化
- 联网搜索兜底

## 技术栈

- 后端: Python 3.11+, FastAPI, LangChain, ChromaDB
- 前端: Vue 3, TypeScript, TailwindCSS
- LLM: Claude / DashScope

## RAG 检索评测

默认评测集位于 `data/evaluation/eval_dataset.json`，可用于回归 Recall@K、Precision@K、MRR、NDCG 和命中证据的关键词覆盖率。

```bash
python -m app.evaluation.run_evaluation eval data/evaluation/eval_dataset.json --top-k 10 -k 1 3 5 10 -o data/evaluation/latest_eval_result.json
```

也可以根据本地研报文件名快速生成一份弱标注评测集：

```bash
python -m app.evaluation.run_evaluation create-from-reports data/reports -o data/evaluation/report_eval_dataset.json
```
