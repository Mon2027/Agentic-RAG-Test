"""Chroma 向量库包装层的单元测试。

本文件用 patch 替换持久化 Chroma 客户端和嵌入模型工厂，测试无需加载真实模型，
也不会建立真实向量索引。重点观察 VectorStore 如何把输入交给 collection，
以及如何整理 collection 返回的检索、删除和文件列表结果。
"""

from unittest.mock import MagicMock, patch


class TestVectorStore:
    """验证向量库初始化、写入、检索、过滤和删除的基本契约。"""

    # 装饰器从下往上生效，所以 PersistentClient 的 mock 是第一个函数参数，
    # get_embedding_model 的 mock 是第二个函数参数；下方同类用例均如此。
    @patch("app.rag.vectorstore.get_embedding_model")
    @patch("app.rag.vectorstore.chromadb.PersistentClient")
    def test_vectorstore_init(self, mock_client, mock_get_embedding):
        """创建向量库时应使用指定集合名，并能读取集合条目数。"""
        # 模拟 Chroma 的调用链：PersistentClient() → 客户端 → collection。
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0

        mock_chroma_client = MagicMock()
        mock_chroma_client.get_or_create_collection.return_value = mock_collection
        # 被 patch 的构造函数返回模拟客户端，而非真实持久化客户端。
        mock_client.return_value = mock_chroma_client

        # 初始化时还会获取嵌入模型；这里提供替身，避免加载真实模型。
        mock_embedding = MagicMock()
        mock_embedding.dimension = 1024
        mock_get_embedding.return_value = mock_embedding

        from app.rag.vectorstore import VectorStore

        vs = VectorStore(collection_name="test_collection")

        # count 属性最终从 collection.count() 读取，值由上面的替身设定。
        assert vs.collection_name == "test_collection"
        assert vs.count == 0

    @patch("app.rag.vectorstore.get_embedding_model")
    @patch("app.rag.vectorstore.chromadb.PersistentClient")
    def test_add_documents(self, mock_client, mock_get_embedding):
        """两个文档分块写入后，应返回写入数量并调用集合 add。"""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 2

        mock_chroma_client = MagicMock()
        mock_chroma_client.get_or_create_collection.return_value = mock_collection
        mock_client.return_value = mock_chroma_client

        # 为两个分块预设两条向量；add_documents 会调用 embed_documents。
        mock_embedding = MagicMock()
        mock_embedding.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_get_embedding.return_value = mock_embedding

        from app.rag.document_processor import DocumentChunk
        from app.rag.vectorstore import VectorStore

        vs = VectorStore()
        # 每个 DocumentChunk 包含待索引正文及可回溯文件的 file_id。
        chunks = [
            DocumentChunk(content="test 1", metadata={"file_id": "f1"}),
            DocumentChunk(content="test 2", metadata={"file_id": "f2"}),
        ]

        result = vs.add_documents(chunks)

        # 当前用例只核对返回数量和 add 次数，没有检查 add 的具体参数。
        assert result == 2
        mock_collection.add.assert_called_once()

    @patch("app.rag.vectorstore.get_embedding_model")
    @patch("app.rag.vectorstore.chromadb.PersistentClient")
    def test_similarity_search(self, mock_client, mock_get_embedding):
        """相似度检索应把 Chroma 返回值整理为逐条结果。"""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        # Chroma.query 的外层列表代表查询批次：这里模拟一次查询、两个命中。
        mock_collection.query.return_value = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"file_id": "f1"}, {"file_id": "f2"}]],
            "distances": [[0.1, 0.2]],
        }

        mock_chroma_client = MagicMock()
        mock_chroma_client.get_or_create_collection.return_value = mock_collection
        mock_client.return_value = mock_chroma_client

        # 检索词先编码成查询向量，再传入 Chroma 的 query 方法。
        mock_embedding = MagicMock()
        mock_embedding.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_get_embedding.return_value = mock_embedding

        from app.rag.vectorstore import VectorStore

        vs = VectorStore()
        results = vs.similarity_search("test query", k=2)

        # 检查命中数量，以及第一条结果的正文和距离被正确取出。
        assert len(results) == 2
        assert results[0]["content"] == "doc1"
        assert results[0]["distance"] == 0.1

    @patch("app.rag.vectorstore.get_embedding_model")
    @patch("app.rag.vectorstore.chromadb.PersistentClient")
    def test_keyword_search_with_scores(self, mock_client, mock_get_embedding):
        """关键词检索应保留匹配项并给出正的相关性分数。"""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        # Chroma.get 提供两条候选记录：第一条包含查询词，第二条与查询无关。
        mock_collection.get.return_value = {
            "ids": ["id1", "id2"],
            "documents": [
                "营收增长 120 毛利率提升",
                "这是一段无关内容",
            ],
            "metadatas": [
                {"file_id": "f1", "file_name": "report.pdf"},
                {"file_id": "f2", "file_name": "other.pdf"},
            ],
        }

        mock_chroma_client = MagicMock()
        mock_chroma_client.get_or_create_collection.return_value = mock_collection
        mock_client.return_value = mock_chroma_client

        mock_embedding = MagicMock()
        mock_get_embedding.return_value = mock_embedding

        from app.rag.vectorstore import VectorStore

        vs = VectorStore()
        results = vs.keyword_search_with_scores("营收 毛利率", k=5)

        # 返回值中的每项是 (文档字典, 分数)；无匹配词的候选不会进入结果。
        assert len(results) == 1
        assert results[0][0]["id"] == "id1"
        assert results[0][0]["content"] == "营收增长 120 毛利率提升"
        assert results[0][1] > 0

    @patch("app.rag.vectorstore.get_embedding_model")
    @patch("app.rag.vectorstore.chromadb.PersistentClient")
    def test_keyword_search_expands_chinese_title_terms(self, mock_client, mock_get_embedding):
        """连续中文查询可匹配文件名和正文中的公司、业务词。"""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        # 两份报告都涉及低空经济，但第一份的文件名还包含公司和连接器关键词。
        mock_collection.get.return_value = {
            "ids": ["id1", "id2"],
            "documents": [
                "连接器业务持续拓展，低空经济打开新增量。",
                "其他公司也布局低空经济。",
            ],
            "metadatas": [
                {"file_id": "f1", "file_name": "【创益通】连接器小巨人，布局低空经济.pdf"},
                {"file_id": "f2", "file_name": "【其他公司】低空经济行业报告.pdf"},
            ],
        }

        mock_chroma_client = MagicMock()
        mock_chroma_client.get_or_create_collection.return_value = mock_collection
        mock_client.return_value = mock_chroma_client
        mock_get_embedding.return_value = MagicMock()

        from app.rag.vectorstore import VectorStore

        vs = VectorStore()
        results = vs.keyword_search_with_scores("创益通连接器业务低空经济打开想象空间", k=2)

        # 检查排序：更贴合公司和业务主题的报告应排在另一份之前。
        assert len(results) == 2
        assert results[0][0]["id"] == "id1"
        assert results[0][1] > results[1][1]

    @patch("app.rag.vectorstore.get_embedding_model")
    @patch("app.rag.vectorstore.chromadb.PersistentClient")
    def test_keyword_search_passes_filter(self, mock_client, mock_get_embedding):
        """关键词检索应把 file_id 等元数据过滤条件传给 Chroma。"""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        # 返回空集合即可；此用例关注调用参数，不关注排名结果。
        mock_collection.get.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": [],
        }

        mock_chroma_client = MagicMock()
        mock_chroma_client.get_or_create_collection.return_value = mock_collection
        mock_client.return_value = mock_chroma_client
        mock_get_embedding.return_value = MagicMock()

        from app.rag.vectorstore import VectorStore

        vs = VectorStore()
        vs.keyword_search_with_scores("营收", k=5, filter_={"file_id": "f1"})

        # 精确断言 where 和 include，确保过滤在 Chroma 查询阶段就生效。
        mock_collection.get.assert_called_once_with(
            where={"file_id": "f1"},
            include=["documents", "metadatas"],
        )

    @patch("app.rag.vectorstore.get_embedding_model")
    @patch("app.rag.vectorstore.chromadb.PersistentClient")
    def test_delete_by_file_id(self, mock_client, mock_get_embedding):
        """按 file_id 删除分块时，应返回找到的分块数量并调用集合删除。"""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        # 模拟按 file_id 查询后找到两个待删除的分块 ID。
        mock_collection.get.return_value = {"ids": ["id1", "id2"]}

        mock_chroma_client = MagicMock()
        mock_chroma_client.get_or_create_collection.return_value = mock_collection
        mock_client.return_value = mock_chroma_client

        mock_embedding = MagicMock()
        mock_get_embedding.return_value = mock_embedding

        from app.rag.vectorstore import VectorStore

        vs = VectorStore()
        result = vs.delete_by_file_id("test_file_id")

        # 返回值代表删除的分块数；此用例仅检查 delete 被调用一次。
        assert result == 2
        mock_collection.delete.assert_called_once()

    @patch("app.rag.vectorstore.get_embedding_model")
    @patch("app.rag.vectorstore.chromadb.PersistentClient")
    def test_list_files(self, mock_client, mock_get_embedding):
        """文件列表应将属于同一 file_id 的多个分块合并为一项。"""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        # f1 的两条元数据代表同一文件的两个分块，f2 只有一个分块。
        mock_collection.get.return_value = {
            "metadatas": [
                {"file_id": "f1", "file_name": "test1.pdf"},
                {"file_id": "f1", "file_name": "test1.pdf"},
                {"file_id": "f2", "file_name": "test2.pdf"},
            ]
        }

        mock_chroma_client = MagicMock()
        mock_chroma_client.get_or_create_collection.return_value = mock_collection
        mock_client.return_value = mock_chroma_client

        mock_embedding = MagicMock()
        mock_get_embedding.return_value = mock_embedding

        from app.rag.vectorstore import VectorStore

        vs = VectorStore()
        files = vs.list_files()

        # 按首次出现顺序列出 f1、f2，而不是返回三条分块记录。
        assert len(files) == 2  # 三条分块元数据只对应两个不同的 file_id。
        assert files[0]["file_id"] == "f1"
        assert files[1]["file_id"] == "f2"


class TestGetVectorStore:
    """验证 get_vector_store 返回的进程内单例。"""

    @patch("app.rag.vectorstore.get_embedding_model")
    @patch("app.rag.vectorstore.chromadb.PersistentClient")
    def test_get_vector_store_singleton(self, mock_client, mock_get_embedding):
        """连续调用工厂函数应取得同一个 VectorStore 实例。"""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0

        mock_chroma_client = MagicMock()
        mock_chroma_client.get_or_create_collection.return_value = mock_collection
        mock_client.return_value = mock_chroma_client

        mock_embedding = MagicMock()
        mock_get_embedding.return_value = mock_embedding

        # 清除模块中先前缓存的实例，确保第一次调用会重新创建测试对象。
        import app.rag.vectorstore as vs_module
        from app.rag.vectorstore import get_vector_store
        vs_module._vector_store = None

        vs1 = get_vector_store()
        vs2 = get_vector_store()

        # 第二次调用应复用模块变量 _vector_store 指向的对象。
        # is 检查对象身份；两个变量必须指向同一个实例，而不只是值相等。
        assert vs1 is vs2
