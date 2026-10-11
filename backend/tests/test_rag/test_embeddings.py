"""向量嵌入模型的单元测试。

EmbeddingModel 把文本交给 SentenceTransformer 编码，再将 NumPy 数组转换为
Python 列表，供向量库使用。这里用 patch 和 MagicMock 替代真实模型，测试时
无需下载模型、占用大量内存或依赖网络。
"""

from unittest.mock import MagicMock, patch


class TestEmbeddingModel:
    """验证模型初始化、单条查询编码和批量文档编码。"""

    def test_embedding_model_init(self):
        """自定义模型名应被保存，向量维度应来自底层模型。"""
        # patch 替换 embeddings 模块实际使用的 SentenceTransformer 符号。
        # with 代码块结束后会自动恢复，避免真正加载或下载模型。
        with patch("app.rag.embeddings.SentenceTransformer") as mock_st:
            # 构造底层模型替身，规定查询向量维度的方法返回 1024。
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 1024
            # EmbeddingModel 调用 SentenceTransformer(...) 时将得到该替身。
            mock_st.return_value = mock_model

            from app.rag.embeddings import EmbeddingModel

            model = EmbeddingModel(model_name="test-model")

            # 分别验证显式传入的名称，以及从底层模型读取到的维度。
            assert model.model_name == "test-model"
            assert model.dimension == 1024

    def test_embed_query(self):
        """单条查询应被编码为一维的 Python 浮点数列表。"""
        with patch("app.rag.embeddings.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            import numpy as np
            # 模拟 SentenceTransformer.encode 返回 NumPy 向量；生产代码会调用 tolist()。
            mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
            mock_model.get_sentence_embedding_dimension.return_value = 3
            mock_st.return_value = mock_model

            from app.rag.embeddings import EmbeddingModel

            model = EmbeddingModel()
            result = model.embed_query("test query")

            # 查询向量应由三个数构成；同时确认底层编码方法仅被调用一次。
            assert isinstance(result, list)
            assert len(result) == 3
            mock_model.encode.assert_called_once()

    def test_embed_documents(self):
        """多条文档应被编码为二维列表，每条文档对应一个向量。"""
        with patch("app.rag.embeddings.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            import numpy as np
            # 两行分别模拟 doc1、doc2 的向量；每个向量有两个维度。
            mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
            mock_model.get_sentence_embedding_dimension.return_value = 2
            mock_st.return_value = mock_model

            from app.rag.embeddings import EmbeddingModel

            model = EmbeddingModel()
            result = model.embed_documents(["doc1", "doc2"])

            # 外层列表长度对应文档数，内层列表长度对应向量维度。
            assert isinstance(result, list)
            assert len(result) == 2
            assert len(result[0]) == 2

    def test_embed_empty_documents(self):
        """空文档列表应得到空结果，而不是产生伪造向量。"""
        with patch("app.rag.embeddings.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 1024
            mock_st.return_value = mock_model

            from app.rag.embeddings import EmbeddingModel

            model = EmbeddingModel()
            result = model.embed_documents([])

            # 当前断言只检查返回值；不检查 encode 是否被调用。
            assert result == []


class TestGetEmbeddingModel:
    """验证 get_embedding_model 工厂函数的缓存与参数传递。"""

    def test_get_embedding_model_cached(self):
        """相同参数重复调用时，应复用同一个 EmbeddingModel 实例。"""
        from app.rag.embeddings import get_embedding_model

        # 先清空 lru_cache，避免其他测试创建的实例影响本用例。
        get_embedding_model.cache_clear()

        # 即使缓存未命中而需要创建模型，仍用替身隔离真实模型加载。
        with patch("app.rag.embeddings.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 1024
            mock_st.return_value = mock_model

            model1 = get_embedding_model()
            model2 = get_embedding_model()

            # is 检查对象身份；值相等不足以证明发生了实例复用。
            assert model1 is model2

    def test_get_embedding_model_with_custom_params(self):
        """工厂函数应把自定义模型名传给创建的 EmbeddingModel。"""
        from app.rag.embeddings import get_embedding_model

        # 清除已有缓存，确保此次调用确实会按新参数创建实例。
        get_embedding_model.cache_clear()

        with patch("app.rag.embeddings.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 1024
            mock_st.return_value = mock_model

            model = get_embedding_model(model_name="custom-model")

            # 这里只验证模型名的传递，不比较底层编码结果。
            assert model.model_name == "custom-model"
