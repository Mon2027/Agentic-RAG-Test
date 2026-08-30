# Agentic RAG Test

[![CI](https://github.com/Mon2027/Agentic-RAG-Test/actions/workflows/ci.yml/badge.svg)](https://github.com/Mon2027/Agentic-RAG-Test/actions/workflows/ci.yml)

基于 DeepAgents、FastAPI 和 Vue 3 的多智能体研报分析系统。

## 本地 CI 等价命令

后端：

```powershell
cd backend
uv sync --locked --extra dev --python 3.12
uv run --no-sync ruff check .
uv run --no-sync pytest -q
```

前端：

```powershell
cd frontend
npm ci --no-audit --no-fund
npm run lint:check
npm run build
```

`lint` 脚本供本地自动修复使用；CI 只调用不会修改源代码的 `lint:check`。

详细建设记录见 [`docs/CI_PLAN.md`](docs/CI_PLAN.md)。
