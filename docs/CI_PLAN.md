# Agentic RAG 多智能体研报分析系统 CI 建设计划

## 1. 文档信息

| 项目 | 内容 |
|---|---|
| 项目名称 | Agentic RAG 多智能体研报分析系统 |
| 文档名称 | 持续集成（CI）建设计划 |
| 文档版本 | v1.2 |
| 制定日期 | 2026-08-22 |
| 最近更新 | 2026-08-30 |
| 当前状态 | CI v1 的 Push、Pull Request 和手动触发均已实跑通过；分支保护待用户确认 |
| 目标平台 | GitHub Actions |
| 预计投入 | 4～8 小时 |
| 本期范围 | 仅建设 CI，不实施 CD、自动部署或生产环境变更 |

## 2. 建设结论

项目已经具备接入 CI 的良好基础：后端自动化测试完整，测试数据与正式数据已经隔离，前端可以完成 TypeScript 检查和生产构建。本次建设采用“两步上线”策略：先建立稳定、可重复的基础 CI，再启用 Ruff 和前端 ESLint 等质量门禁。

CI 建成后，每次向默认分支推送代码或提交 Pull Request 时，应自动并行执行后端与前端检查。任何必需检查失败时，代码不得合并。

### 2.1 实施记录

截至 2026-08-30，CI v1 已完成以下实施与验证：

- [x] 新增 `.github/workflows/ci.yml`，包含后端测试和前端构建两个并行 Job；
- [x] 生成 `backend/uv.lock`，锁定 187 个后端项目及传递依赖；
- [x] 将项目 Python 约束由不真实的 `>=3.10` 修正为与 `deepagents==0.6.1` 一致的 `>=3.11,<4.0`；
- [x] 在独立干净环境中按锁文件安装依赖并执行全量测试，结果为 `283 passed`；
- [x] 使用 `npm ci` 重新安装前端依赖并完成生产构建；
- [x] 工作流采用只读权限、并发取消、超时、依赖缓存和固定提交版本的 Action；
- [x] 初始化有效 Git 元数据，并配置空仓库 `https://github.com/Mon2027/Agentic-RAG-Test.git` 为 origin；
- [x] 完成首次提交和推送，并在 GitHub Linux Runner 上验证 Push 触发；
- [x] 补充测试所需的最小 CSV 夹具和 Agent 路由评测数据集，继续排除其他本地业务数据及历史评测结果；
- [x] 修复上传文件名清理的跨平台路径问题；
- [x] [GitHub Actions 第 5 次运行](https://github.com/Mon2027/Agentic-RAG-Test/actions/runs/33313489406)首次全绿：后端 `283 passed`，前端构建成功；
- [x] [GitHub Actions 第 6 次运行](https://github.com/Mon2027/Agentic-RAG-Test/actions/runs/33313621268)继续全绿，确认文档提交后的 CI 基线稳定；
- [x] 通过临时 [PR #1](https://github.com/Mon2027/Agentic-RAG-Test/pull/1) 验证 `pull_request` 触发，[第 8 次运行](https://github.com/Mon2027/Agentic-RAG-Test/actions/runs/33315941702)全绿；
- [x] 从 `main` 调用 `workflow_dispatch`，[第 9 次运行](https://github.com/Mon2027/Agentic-RAG-Test/actions/runs/33316034669)全绿；
- [x] 通过正常修复与文档提交连续获得 3 次绿色流水线；
- [ ] 用户确认后启用分支保护；
- [ ] CI v2 的 Ruff 和 ESLint 门禁。

## 3. 当前基线

### 3.1 已具备的条件

- 后端使用 Python 3.11+、FastAPI 和 pytest；CI 基准环境为 Python 3.12；
- 后端全量回归实测为 `283 passed`，耗时约 20 秒；
- `backend/tests/conftest.py` 已将测试目录、Chroma 数据和外部服务配置与正式环境隔离；
- 前端使用 Vue 3、TypeScript 和 Vite；
- 前端执行 `npm run build` 成功，耗时约 5 秒；
- 前端已有 `package-lock.json`，可以使用 `npm ci` 进行确定性安装；
- 后端已有 `/health` 健康检查，但该接口属于未来 CD 范围，本次 CI 不使用它作为部署验证。

### 3.2 当前剩余缺口

- Push、Pull Request 和手动触发均已在真实 GitHub Runner 上完成行为验证；
- Push 流水线已完成连续稳定性验证，分支保护仍需用户确认后配置；
- Ruff 当前检查出 50 个问题，其中 46 个可自动修复，其余需要人工处理；
- 前端 `lint` 脚本带有 `--fix`，不适合作为只读 CI 检查；
- 前端尚未安装和配置 ESLint，也没有前端自动化测试；
- CI v1 已修复目前发现的 Windows/Linux 路径分隔符差异，仍需通过后续运行观察稳定性。

## 4. 建设目标

### 4.1 必须完成

1. 在干净的 GitHub Runner 中从零安装依赖；
2. 后端和前端 Job 并行执行；
3. 后端离线执行全量 pytest，不调用真实 LLM、Tavily 或其他付费服务；
4. 前端完成 TypeScript 检查和 Vite 生产构建；
5. 第二阶段启用后端 Ruff 和前端 ESLint 门禁；
6. 使用锁文件和缓存降低依赖漂移与重复安装成本；
7. 流水线使用最小权限，不读取本地 `.env`，不要求配置业务 API Key；
8. 为默认分支建立可配置为“合并前必需通过”的检查项；
9. 在项目文档中记录本地等价命令和故障定位方式。

### 4.2 本期不做

- Dockerfile、Docker Compose、Kubernetes 或云平台部署；
- 自动发布、自动回滚和环境晋级；
- 真实模型、真实联网搜索和付费 API 评测；
- 大规模 RAG 黄金集评测；
- 前端单元测试、E2E 测试和浏览器兼容性矩阵；
- 性能压测、安全扫描、依赖自动升级和代码覆盖率硬阈值；
- 多 Python 版本、多 Node.js 版本或多操作系统矩阵。

以上项目可在基础 CI 稳定后按风险和收益另行规划。

## 5. CI 总体设计

### 5.1 触发规则

CI 工作流计划支持以下触发方式：

- 向 `main` 分支推送代码；
- 创建或更新目标为 `main` 的 Pull Request；
- 通过 `workflow_dispatch` 手动触发；
- 同一分支出现新提交时，自动取消该分支尚未完成的旧流水线。

第一版不配置路径过滤，避免文档、配置或依赖文件变更意外绕过必要检查。

### 5.2 Job 设计

| Job | 运行环境 | 核心步骤 | 第一阶段是否阻断合并 |
|---|---|---|---|
| `backend-test` | Ubuntu + Python 3.12 | 冻结依赖安装、pytest | 是 |
| `frontend-build` | Ubuntu + Node.js 24 | `npm ci`、TypeScript 检查、Vite 构建 | 是 |
| `backend-lint` | Ubuntu + Python 3.12 | Ruff 只读检查 | 第二阶段启用 |
| `frontend-lint` | Ubuntu + Node.js 24 | ESLint 只读检查 | 第二阶段启用 |

后端和前端 Job 不互相依赖，应并行执行并分别显示失败原因。

### 5.3 安全与稳定性约束

- 工作流默认仅授予 `contents: read` 权限；
- 不向 Pull Request 注入生产密钥；
- 不配置 `ANTHROPIC_API_KEY`、`ANTHROPIC_AUTH_TOKEN` 或 `TAVILY_API_KEY`；
- 除离线测试所需的确定性评测数据集外，不上传 `backend/data` 中的历史结果、正式数据、本地 `.env`、模型缓存或用户文件；
- CI 中禁止使用会修改源代码的 `ruff --fix`、`eslint --fix`；
- 为每个 Job 设置合理超时，避免依赖安装或测试异常挂起；
- 第三方 Action 在实施时选用当前稳定版本，并优先固定到经过验证的提交版本；
- 依赖缓存键必须绑定对应锁文件，锁文件变化后自动失效。

## 6. 分阶段执行计划

### 阶段 0：恢复 Git 仓库基础

#### 执行动作

1. 确认项目是否已经存在对应的 GitHub 远端仓库；
2. 若已有远端，优先重新 clone 或恢复正确的 Git 元数据，不直接覆盖现有仓库历史；
3. 若从未建立仓库，再初始化 Git、创建基线提交并添加远端；
4. 确认默认分支名称为 `main`；
5. 检查 `.gitignore`，确保 `.env`、虚拟环境、Node 依赖、构建产物、日志和业务数据不会提交。

#### 验证命令

```powershell
git rev-parse --show-toplevel
git remote -v
git status
```

#### 完成标准

- 三条命令均正常执行；
- 可以向 GitHub 推送测试分支；
- 项目历史和远端归属得到确认；
- `.env`、`backend/data`、`.venv`、`node_modules` 和 `dist` 未进入待提交列表。

### 阶段 1：固化依赖安装

#### 后端

推荐采用 `uv` 管理锁文件，并继续以 `backend/pyproject.toml` 作为依赖声明来源。

计划命令：

```powershell
cd backend
uv lock --python 3.12
uv sync --locked --extra dev --python 3.12
uv run --no-sync pytest -q
```

需要新增并提交 `backend/uv.lock`。CI 必须使用 `--locked` 检查锁文件状态，锁文件与依赖声明不一致时直接失败，不允许在流水线中隐式更新依赖。

#### 前端

沿用现有 `frontend/package-lock.json`：

```powershell
cd frontend
npm ci
npm run build
```

#### 完成标准

- 删除本地虚拟环境和 `node_modules` 后仍可从零安装；
- 后端使用锁文件安装成功，pytest 全量通过；
- 前端使用 `npm ci` 安装成功，生产构建通过；
- 安装过程不依赖开发者机器中的全局包或本地 `.env`。

### 阶段 2：上线最小可用 CI（CI v1）

#### 新增文件

```text
.github/workflows/ci.yml
```

#### `backend-test` 执行顺序

1. Checkout 源代码；
2. 安装 Python 3.12；
3. 安装并缓存 `uv` 依赖；
4. 在 `backend` 目录执行冻结依赖同步；
5. 执行 `uv run pytest -q`。

#### `frontend-build` 执行顺序

1. Checkout 源代码；
2. 安装 Node.js 24；
3. 根据 `frontend/package-lock.json` 恢复 npm 缓存；
4. 在 `frontend` 目录执行 `npm ci`；
5. 执行 `npm run build`。

现有 `build` 脚本已经先执行 `vue-tsc`，因此 CI v1 同时覆盖 TypeScript 类型检查和 Vite 生产构建。

#### 完成标准

- Push、Pull Request 和手动触发均可以启动流水线；
- 两个 Job 并行执行；
- 后端 283 条测试全部通过；
- 前端生产构建通过；
- CI 不配置业务 Secrets 也能成功；
- 测试只读取已提交的确定性评测数据集，不读取、修改或上传正式数据和历史评测结果；
- 连续出现 3 次稳定的绿色流水线后，再进入质量门禁阶段。

### 阶段 3：修复后端 Ruff 基线

#### 执行动作

先在本地开发分支修复，不在 CI 中自动修改代码：

```powershell
cd backend
uv run ruff check app tests --fix
uv run ruff check app tests
uv run pytest -q
```

具体要求：

1. 审查 Ruff 自动修改产生的全部差异；
2. 手工处理未使用变量、未使用导入等剩余问题；
3. 不使用未审查的 unsafe fix；
4. 修复后执行全量 pytest；
5. 在 CI 中新增只读命令 `uv run ruff check app tests`。

#### 完成标准

- Ruff 返回零问题、退出码为 0；
- pytest 仍为 283 条全部通过；
- CI 中 Ruff 失败会阻止合并；
- 流水线不会修改或提交源代码。

### 阶段 4：建立前端 ESLint 门禁

#### 执行动作

1. 安装与当前 Vue 3、TypeScript 版本兼容的 ESLint 及 Vue/TypeScript 插件；
2. 增加 ESLint 配置文件；
3. 排除 `dist`、`node_modules` 等生成目录；
4. 将开发时修复命令与 CI 只读检查命令分开；
5. 更新 `package-lock.json`；
6. 处理现有 lint 问题并重新执行前端构建。

计划脚本：

```json
{
  "scripts": {
    "typecheck": "vue-tsc --noEmit",
    "lint": "eslint . --fix",
    "lint:check": "eslint ."
  }
}
```

计划验证命令：

```powershell
cd frontend
npm ci
npm run typecheck
npm run lint:check
npm run build
```

#### 完成标准

- `typecheck`、`lint:check` 和 `build` 全部通过；
- CI 只调用不修改文件的 `lint:check`；
- 前端 lint 失败会阻止合并；
- 前端仍使用锁文件进行安装。

### 阶段 5：流水线行为验证

在专用测试分支验证以下场景：

| 场景 | 操作 | 预期结果 |
|---|---|---|
| 正常提交 | 推送当前绿色基线 | 后端、前端均通过 |
| 后端失败 | 临时制造一个断言失败 | 仅后端相关 Job 失败，并显示明确日志 |
| 前端失败 | 临时制造 TypeScript 错误 | 前端 Job 失败，构建不通过 |
| Lint 失败 | 临时加入可识别的违规代码 | 对应 lint Job 失败 |
| 并发取消 | 对同一分支连续推送两次 | 旧流水线被取消，新流水线继续 |
| 无密钥运行 | 不配置任何业务 Secret | 离线 CI 正常通过 |
| 数据隔离 | 对比测试前后的正式数据 | 文件数、长度和内容不发生变化 |

临时错误只能存在于验证分支，完成验证后必须撤销，不得合并到 `main`。

### 阶段 6：启用分支保护和补充文档

在 CI 连续稳定通过后，再配置 GitHub 默认分支保护：

1. 禁止直接向 `main` 推送；
2. 要求 Pull Request 合并前通过后端测试和前端构建；
3. 第二阶段完成后，将后端 Ruff 和前端 ESLint 加入必需检查；
4. 根据协作人数决定是否要求至少一次人工审查；
5. 要求分支在合并前与 `main` 保持最新；
6. 在项目 README 中增加 CI 状态徽章和本地等价命令。

分支保护属于 GitHub 远端配置。实施时应在用户确认后执行，不通过本地代码变更代替。

## 7. 计划文件变更

| 文件 | 操作 | 用途 |
|---|---|---|
| `.gitignore` | 修改 | 排除 Node、pytest、Ruff 和覆盖率缓存 |
| `.github/workflows/ci.yml` | 新增 | GitHub Actions 主工作流 |
| `backend/uv.lock` | 新增 | 固定后端完整依赖版本 |
| `backend/pyproject.toml` | 必要时修改 | 调整开发依赖或 Ruff 配置 |
| `frontend/package.json` | 修改 | 增加只读 lint 和独立类型检查命令 |
| `frontend/package-lock.json` | 更新 | 锁定新增的前端检查依赖 |
| `frontend/eslint.config.js` | 新增 | 配置 Vue 与 TypeScript 静态检查 |
| `docs/CI_PLAN.md` | 新增 | 记录本建设计划 |
| 项目 README | 修改 | 记录 CI 命令、状态徽章和贡献约定 |

除上述文件和 Ruff/ESLint 必要的格式修复外，本次不修改业务功能，不增加部署文件。

## 8. 验收标准

### 8.1 CI v1 验收

- [x] Git 仓库和 GitHub 远端可正常使用；
- [x] 后端依赖可以从锁文件全新安装；
- [x] 前端依赖可以通过 `npm ci` 全新安装；
- [x] GitHub Push 触发有效；
- [x] GitHub Pull Request 触发有效；
- [x] GitHub 手动触发有效；
- [x] 后端 pytest 全量通过；
- [x] 前端 TypeScript 检查和生产构建通过；
- [x] 后端与前端 Job 并行执行；
- [x] 流水线不需要业务 API Key；
- [x] CI 只使用已提交的确定性测试数据，未修改正式数据；
- [x] 至少连续 3 次正常提交获得绿色结果。

### 8.2 CI v2 验收

- [ ] 后端 Ruff 零问题并成为必需检查；
- [ ] 前端 ESLint 零问题并成为必需检查；
- [ ] CI 中不存在自动修复或自动提交操作；
- [ ] 并发取消和超时设置有效；
- [ ] 分支保护配置完成；
- [ ] README 已记录本地等价命令和 CI 状态；
- [ ] 缓存命中时，正常 CI 总耗时目标不超过 10 分钟；
- [ ] 冷缓存情况下，正常 CI 总耗时目标不超过 20 分钟。

## 9. 风险与应对

| 风险 | 影响 | 应对措施 |
|---|---|---|
| 首次推送或 GitHub 认证失败 | 无法触发 CI | 使用已配置的 origin 完成认证并重新推送 |
| Python 锁文件尚未提交远端 | GitHub Runner 无法使用已验证依赖 | 恢复 Git 后提交 `uv.lock`，CI 使用 `--locked` |
| `sentence-transformers` 等依赖较大 | 冷安装时间长 | 使用依赖缓存，设置合理超时 |
| 测试意外下载嵌入模型 | CI 变慢或依赖外网 | 保持 Mock/隔离；真实模型评测改为手动任务 |
| Windows 通过但 Linux 失败 | 首次 CI 不稳定 | 重点检查路径、大小写、编码和系统依赖 |
| Ruff 自动修复改变代码 | 引入行为回归 | 单独提交、人工审查、全量 pytest 回归 |
| 前端 lint 一次性问题较多 | CI v2 延期 | CI v1 先以类型检查和构建作为门禁 |
| 第三方 Action 供应链风险 | 工作流执行不受信任代码 | 使用官方或可信 Action，并固定验证版本 |
| 分支保护过早启用 | 修复 CI 时无法合并 | 连续 3 次绿色后再启用必需检查 |

## 10. 回滚策略

CI 本身不部署应用，也不修改生产数据，因此回滚风险较低。

1. CI v1 失败时，保留本地测试能力，修正 `.github/workflows/ci.yml` 后重新推送；
2. 锁文件导致安装问题时，回退对应锁文件提交，不删除 `pyproject.toml` 中的原始依赖声明；
3. Ruff 或 ESLint 修复引起回归时，只回退独立的质量修复提交；
4. 新增门禁不稳定时，先从分支保护中取消“必需检查”，但保留工作流继续收集结果；
5. 不使用 `git reset --hard` 或覆盖远端历史进行回滚，优先通过普通 revert 提交恢复。

## 11. 建议提交拆分

为便于审查和回滚，建议至少拆分为以下提交：

1. `chore(ci): lock backend dependencies`；
2. `ci: add backend test and frontend build workflow`；
3. `style(backend): establish clean ruff baseline`；
4. `chore(frontend): add eslint and typecheck gates`；
5. `docs: document local CI workflow and branch policy`。

每个提交完成后都应执行当时已建立的全部本地门禁。

## 12. 建议执行顺序与工时

| 顺序 | 工作项 | 预计耗时 |
|---:|---|---:|
| 1 | 恢复 Git 仓库与远端 | 0.5～1 小时 |
| 2 | 生成后端锁文件并验证全新安装 | 1～2 小时 |
| 3 | 创建并验证 CI v1 | 1～2 小时 |
| 4 | 修复 Ruff 基线 | 0.5～1.5 小时 |
| 5 | 配置前端 ESLint | 1～2 小时 |
| 6 | 验证失败场景、配置分支保护和补文档 | 1 小时 |

预计总投入为 4～8 小时。若 Git 元数据恢复、Linux 兼容或大型依赖下载出现问题，应单独记录问题，不通过降低测试隔离或引入生产密钥绕过。

## 13. 完成定义

满足以下条件后，本轮 CI 建设视为完成：

1. `main` 的每次 Push 和 Pull Request 都会触发 CI；
2. 后端 pytest、后端 Ruff、前端类型检查、前端 ESLint 和前端生产构建全部成为稳定门禁；
3. CI 在无业务密钥、无本地缓存的干净环境中可以完成；
4. 测试不会修改正式数据，也不会调用真实付费服务；
5. 必需检查与分支保护已启用；
6. 本地开发者可以根据文档复现全部 CI 命令；
7. CI 连续 3 次正常提交运行成功，且失败场景反馈清晰；
8. 未加入任何 CD、部署或生产环境变更。
