"""文档向量索引“失败安全替换”与回滚机制测试。

重建索引不能简单地“先删旧数据，再计算并写入新数据”：一旦嵌入服务或 Chroma
写入失败，文档会立刻失去原有可检索证据。安全替换应遵循以下事务式顺序：

1. 拒绝空分块；
2. 在触碰旧索引之前完成所有新向量计算；
3. 读取并保存旧文档、metadata 和 embedding 的完整快照；
4. 删除旧记录并写入新记录；
5. 如果写入失败，清理可能存在的半成品，再 upsert 恢复旧快照。

这四个测试依次覆盖：输入不合法、计算新向量失败、写入新索引失败、替换成功。
观察重点是“失败发生在哪一步”以及“旧索引是否仍可恢复”。

这些是纯单元测试：通过 ``__new__`` 绕过真实 VectorStore 初始化，并用 MagicMock
模拟嵌入模型和 Chroma collection，因此可以检查关键调用及回滚参数。
"""

# MagicMock 代替外部组件；call 用于比较同一 mock 多次调用的参数与先后顺序。
from unittest.mock import MagicMock, call

import pytest

from app.rag.document_processor import DocumentChunk
from app.rag.vectorstore import VectorStore


def _make_vector_store() -> VectorStore:
    """仅保留真实业务方法，把它依赖的模型和集合替换为测试替身。"""
    # __new__ 只分配实例，不执行 VectorStore.__init__，从而避免创建持久化客户端、
    # collection 和真实 embedding model。
    vector_store = VectorStore.__new__(VectorStore)
    # 实例仍是 VectorStore，所以调用的 replace_document_for_file 是真实方法；
    # 只需手动补齐该方法会访问的两个实例属性，就能单独检验替换逻辑。
    vector_store._embedding_model = MagicMock()
    vector_store._collection = MagicMock()
    return vector_store


def _chunk(file_id: str, content: str = "new evidence") -> DocumentChunk:
    """构造最小合法的新索引分块，减少各测试重复的准备代码。"""
    # metadata 中的 file_id 将随新分块写入 collection，用来关联所属报告。
    return DocumentChunk(
        content=content,
        metadata={"file_id": file_id, "file_name": "report.pdf"},
    )


def test_replace_document_rejects_empty_chunks_before_touching_store():
    """空替换集必须立即报错，且嵌入模型和旧索引都不能被访问。"""
    vector_store = _make_vector_store()

    # Arrange：不提供新分块；Act 与异常断言合在 pytest.raises 中。
    # pytest.raises 同时断言异常类型和错误消息，避免“因其他原因报错”也误通过。
    with pytest.raises(ValueError, match="empty document"):
        vector_store.replace_document_for_file([], "report-id")

    # Assert：这些零调用断言证明校验先于向量计算、快照读取和删除。
    vector_store._embedding_model.embed_documents.assert_not_called()
    vector_store._collection.get.assert_not_called()
    vector_store._collection.delete.assert_not_called()


def test_replace_document_embedding_failure_keeps_old_index_untouched():
    """新向量计算失败时应原样抛错，并且完全不读取或修改旧索引。"""
    vector_store = _make_vector_store()
    # 与 return_value 规定正常返回值不同，side_effect 规定调用时抛出异常，
    # 从而稳定模拟外部嵌入服务故障。
    vector_store._embedding_model.embed_documents.side_effect = RuntimeError(
        "embedding service unavailable"
    )

    with pytest.raises(RuntimeError, match="embedding service unavailable"):
        vector_store.replace_document_for_file([_chunk("report-id")], "report-id")

    # 嵌入发生在读取快照和删除之前；失败后 collection 的任何方法都不应被调用。
    # 这里既检查旧数据没有被读取，也检查没有触发写入和回滚。
    vector_store._collection.get.assert_not_called()
    vector_store._collection.delete.assert_not_called()
    vector_store._collection.add.assert_not_called()
    vector_store._collection.upsert.assert_not_called()


def test_replace_document_write_failure_restores_old_snapshot():
    """删除旧索引后若新记录写入失败，应清理半成品并恢复完整旧快照。"""
    vector_store = _make_vector_store()
    # 新文档嵌入预先成功，测试才能继续进入“旧索引已删除但写新索引失败”的分支。
    vector_store._embedding_model.embed_documents.return_value = [[0.9, 0.8]]
    # get 返回旧索引快照：ids 用于定位旧分块，documents 是旧正文，
    # metadatas 保留来源信息，embeddings 保留旧向量；回滚时四项都要原样使用。
    vector_store._collection.get.return_value = {
        "ids": ["report-id_0"],
        "documents": ["old evidence"],
        "metadatas": [{"file_id": "report-id", "file_name": "report.pdf"}],
        "embeddings": [[0.1, 0.2]],
    }
    # 模拟 Chroma 在写入新索引时失败：真实方法此时已删除旧记录，
    # 必须进入异常分支执行补偿操作，才能避免文档失去可检索证据。
    vector_store._collection.add.side_effect = RuntimeError("Chroma write failed")

    with pytest.raises(RuntimeError, match="Chroma write failed"):
        vector_store.replace_document_for_file([_chunk("report-id")], "report-id")

    # 回滚成功后仍把原始写入异常抛给调用方，因此不能把这次操作当成成功。
    # 第一次 delete 删除旧记录以准备切换；第二次 delete 清理可能部分写入的新记录。
    # 当前新旧 ID 相同，所以调用参数相同，但调用阶段和目的不同。
    assert vector_store._collection.delete.call_args_list == [
        call(ids=["report-id_0"]),
        call(ids=["report-id_0"]),
    ]
    # upsert 比 add 更适合恢复：无论半成品 ID 是否存在，都能把旧快照覆盖回去。
    vector_store._collection.upsert.assert_called_once_with(
        ids=["report-id_0"],
        documents=["old evidence"],
        embeddings=[[0.1, 0.2]],
        metadatas=[{"file_id": "report-id", "file_name": "report.pdf"}],
    )


def test_replace_document_success_activates_new_index_without_rollback():
    """新索引写入成功时应激活新内容，并且绝不执行回滚 upsert。"""
    vector_store = _make_vector_store()
    chunk = _chunk("report-id")
    vector_store._embedding_model.embed_documents.return_value = [[0.9, 0.8]]
    vector_store._collection.get.return_value = {
        "ids": ["report-id_0"],
        "documents": ["old evidence"],
        "metadatas": [{"file_id": "report-id", "file_name": "report.pdf"}],
        "embeddings": [[0.1, 0.2]],
    }

    # 与上一个测试使用相同的旧快照，但不设置 add.side_effect，模拟正常写入。
    # Act：调用真实替换方法；返回值代表成功写入的新分块数量。
    replaced = vector_store.replace_document_for_file([chunk], "report-id")

    # 正常路径只需删除一次旧记录，再用新正文、新向量和新 metadata 写回同一组 ID。
    assert replaced == 1
    vector_store._collection.delete.assert_called_once_with(ids=["report-id_0"])
    vector_store._collection.add.assert_called_once_with(
        ids=["report-id_0"],
        documents=["new evidence"],
        embeddings=[[0.9, 0.8]],
        metadatas=[chunk.metadata],
    )
    # upsert 是异常回滚专用操作，成功路径调用它反而说明实现发生了意外补偿。
    vector_store._collection.upsert.assert_not_called()
