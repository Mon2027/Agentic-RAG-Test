# RA-API-002：索引重建失败会删除仍可用的旧索引

| 字段 | 内容 |
|---|---|
| 状态 | 已确认，待修复 |
| 严重程度 | 高 |
| 优先级 | P0 |
| 类型 | 可靠性 / 数据一致性 / 失败恢复 |
| 首次确认日期 | 2026-07-23 |
| 自动化证据 | `backend/tests/test_api/test_phase2_core_api.py::TestHighValueRisks::test_failed_reindex_preserves_previous_chunks` |

## 复现条件与步骤

1. 为 `existing-report` 预置一条可检索的旧向量证据；
2. 使用实际 `process_pdf_task` 控制流，以 `clear_existing=True` 执行重建；
3. 让替换 PDF 的解析过程抛出 `ValueError`；
4. 检查失败状态及旧向量块是否仍然存在。

## 预期结果

- 文档状态标记为 `failed`；
- 原有一条旧向量证据仍可查询。

## 实际结果

```text
文档状态：failed
解析异常：replacement PDF cannot be parsed
重建前旧向量块：1
重建后旧向量块：0
pytest：1 failed in 13.83s
```

## 根因

`backend/app/api/routes.py` 的 `process_pdf_task` 在解析新 PDF 之前执行：

```python
vector_store.delete_by_file_id(file_id)
```

随后 `processor.process_pdf(...)` 失败，异常处理只写入失败状态，没有恢复已经删除的旧向量。

## 建议修复

1. 至少先完成 PDF 解析和分块，再删除旧索引；
2. 对向量写入失败增加事务、临时索引切换或备份恢复机制；
3. 补充解析失败、嵌入失败和向量写入失败三类回归测试；
4. 修复后执行风险用例及全量 pytest 回归。

## 隔离说明

测试执行真实重建控制流，向量库使用内存状态替身，解析器使用可控异常替身。正式 Chroma 和正式文件均未参与。
