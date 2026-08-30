"""各 API 路由的基础接口测试。

本文件以“一个接口、一个主要行为”为粒度，快速验证健康检查、文档管理、文件上传、
Chat、PDF 后台处理和 SSE 流式响应。与更完整的核心测试相比，这里的断言更聚焦于
状态码和关键响应字段，适合作为接口测试入门以及日常快速回归测试。

外部大模型、配置对象、PDF 处理器和向量库通过 ``unittest.mock.patch`` 替换，
从而让用例运行快速、稳定且不访问真实服务。
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


# 路由的 file_id 类型是 UUID，因此测试常量也必须是语法合法的 UUID。
# 两个不同常量分别表达“测试中会创建对应文件”和“确认不存在对应文件”的语义。
EXISTING_FILE_ID = "11111111-1111-4111-8111-111111111111"
MISSING_FILE_ID = "22222222-2222-4222-8222-222222222222"


@pytest.fixture
def client():
    """创建可直接调用 FastAPI 应用的同步测试客户端。

    测试请求仍会经过路由匹配、参数校验、依赖调用和响应序列化，但不会真正监听
    网络端口，因此比启动服务器后再发 HTTP 请求更轻量。
    """
    return TestClient(app)


class TestHealthEndpoint:
    """健康检查接口测试。"""

    def test_health_check(self, client):
        """GET /health 应返回 200、健康状态和版本号。"""
        # Act：通过 fixture 注入的客户端发起 GET 请求。
        response = client.get("/health")

        # Assert：先检查 HTTP 状态，再反序列化 JSON 检查业务字段。
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestDocumentsEndpoint:
    """文档列表、详情、删除和重建索引接口测试。"""

    def test_list_documents_empty(self, client):
        """没有文档时，列表接口仍应成功并返回 JSON 数组。"""
        response = client.get("/api/documents")

        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_get_nonexistent_document(self, client):
        """查询不存在但格式合法的 UUID 应返回 404。"""
        response = client.get(f"/api/documents/{MISSING_FILE_ID}")

        assert response.status_code == 404

    def test_delete_nonexistent_document(self, client):
        """删除不存在但格式合法的 UUID 应返回 404。"""
        response = client.delete(f"/api/documents/{MISSING_FILE_ID}")

        assert response.status_code == 404

    # patch 装饰器从下往上应用，所以注入测试函数的 mock 参数顺序也是：
    # 下方 get_settings 对应第一个参数，上方 process_pdf_task 对应第二个参数。
    @patch("app.api.routes.process_pdf_task")
    @patch("app.api.routes.get_settings")
    def test_reindex_pdf_document(self, mock_get_settings, mock_process_pdf, client, tmp_path):
        """已上传 PDF 发起重建索引时，应安排一次安全的后台处理。"""
        # Arrange：在 pytest 临时目录中创建符合“file_id_原文件名”规则的 PDF。
        reports_path = tmp_path / "reports"
        reports_path.mkdir()
        pdf_path = reports_path / f"{EXISTING_FILE_ID}_test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        # 让路由从临时 reports 目录查找文件，避免使用应用真实配置。
        mock_settings = MagicMock()
        mock_settings.reports_path = reports_path
        mock_get_settings.return_value = mock_settings

        # Act：file_id 放在 URL 路径参数中，FastAPI 会先完成 UUID 校验。
        response = client.post(f"/api/documents/{EXISTING_FILE_ID}/reindex")

        # Assert：检查响应契约，并精确验证后台任务的四个位置参数。
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["file_id"] == EXISTING_FILE_ID
        assert data["file_name"] == "test.pdf"
        mock_process_pdf.assert_called_once_with(
            pdf_path,
            EXISTING_FILE_ID,
            "test.pdf",
            True,  # clear_existing=True 表示重建现有索引，而不是首次新增。
        )

    @patch("app.api.routes.get_settings")
    def test_reindex_nonexistent_document(self, mock_get_settings, client, tmp_path):
        """UUID 合法但磁盘中没有对应 PDF 时，重建索引应返回 404。"""
        reports_path = tmp_path / "reports"
        reports_path.mkdir()
        mock_settings = MagicMock()
        mock_settings.reports_path = reports_path
        mock_get_settings.return_value = mock_settings

        response = client.post(f"/api/documents/{MISSING_FILE_ID}/reindex")

        assert response.status_code == 404


class TestChatEndpoint:
    """普通 Chat JSON 接口测试。"""

    @patch("app.agents.main_agent.get_main_agent")
    def test_chat_simple_message(self, mock_get_agent, client):
        """发送一条用户消息后，应返回助手消息和会话 ID。"""
        # Arrange：模拟智能体返回 LangChain 风格的 messages 列表，隔离真实模型调用。
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [MagicMock(content="Hello! How can I help you?")]
        }
        mock_get_agent.return_value = mock_agent

        # json 参数会自动序列化请求体并设置 application/json Content-Type。
        response = client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )

        # 基础测试只断言关键字段存在；具体内容和会话历史由核心测试深入验证。
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "session_id" in data

    def test_chat_empty_messages(self, client):
        """消息数组为空时，业务层应返回 400 和明确错误详情。"""
        response = client.post(
            "/api/chat",
            json={"messages": []}
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "No messages provided"

    def test_chat_invalid_message_format(self, client):
        """消息对象缺少 role/content 字段时，应在模型校验层返回 422。"""
        response = client.post(
            "/api/chat",
            json={"messages": [{"invalid": "format"}]}
        )

        # 422 表示请求 JSON 可解析，但不满足 Pydantic 请求模型约束。
        assert response.status_code == 422  # Validation error


class TestUploadEndpoint:
    """multipart/form-data 文件上传接口测试。"""

    def test_upload_report_invalid_type(self, client):
        """报告上传接口应拒绝扩展名不是 .pdf 的文件。"""
        # files 中的三元组依次是文件名、字节内容和 MIME 类型。
        response = client.post(
            "/api/upload/report",
            files={"file": ("test.txt", b"test content", "text/plain")}
        )

        assert response.status_code == 400
        assert "PDF" in response.json()["detail"]

    def test_upload_data_invalid_type(self, client):
        """数据上传接口应拒绝不在 CSV/Excel 白名单中的 PDF。"""
        response = client.post(
            "/api/upload/data",
            files={"file": ("test.pdf", b"test content", "application/pdf")}
        )

        assert response.status_code == 400

    def test_upload_data_csv(self, client):
        """合法 CSV 应上传成功，并返回文件类型和服务器生成的 file_id。"""
        # 用最小 CSV 字节串即可覆盖扩展名校验、读取、保存和响应序列化。
        csv_content = b"name,age\nAlice,25\nBob,30"

        response = client.post(
            "/api/upload/data",
            files={"file": ("test.csv", csv_content, "text/csv")}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["file_type"] == "csv"
        assert "file_id" in data

    def test_upload_data_sanitizes_filename(self, client):
        """服务端必须清理客户端文件名中的路径片段，防止路径穿越。"""
        csv_content = b"name,age\nAlice,25\nBob,30"

        # ``..\unsafe.csv`` 模拟恶意或异常客户端提交的 Windows 风格相对路径。
        response = client.post(
            "/api/upload/data",
            files={"file": ("..\\unsafe.csv", csv_content, "text/csv")}
        )

        assert response.status_code == 200
        data = response.json()
        # 响应中只能保留安全文件名，不能回显任何目录分隔符。
        assert data["file_name"] == "unsafe.csv"
        assert "\\" not in data["file_name"]
        assert "/" not in data["file_name"]

    def test_upload_data_excel(self, client):
        """扩展名合法的 XLSX 应通过数据上传接口的基础校验。"""
        # 这里只构造 XLSX/ZIP 文件头，并非完整工作簿；本用例验证上传层扩展名逻辑，
        # 真实 Excel 二进制保真验证位于 test_phase2_core_api.py。
        xlsx_content = b"PK\x03\x04" + b"\x00" * 100  # ZIP header

        response = client.post(
            "/api/upload/data",
            files={"file": ("test.xlsx", xlsx_content, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["file_type"] == "xlsx"


class TestPdfProcessingTask:
    """PDF 后台处理函数的索引替换策略测试。"""

    @patch("app.api.routes.get_vector_store")
    @patch("app.api.routes.DocumentProcessor")
    def test_process_pdf_task_uses_safe_index_replacement(self, mock_processor_cls, mock_get_vs):
        """重建索引应调用原子替换方法，而不是先删除旧索引再追加。"""
        # 延迟导入被测函数和真实分块数据类型，使测试准备过程更贴近实际调用。
        from app.api.routes import process_pdf_task
        from app.rag.document_processor import DocumentChunk

        # Arrange：处理器返回一个有效分块，向量库本身则用 mock 观察交互。
        chunk = DocumentChunk(content="content", metadata={"file_id": "file123"})
        mock_processor = MagicMock()
        mock_processor.process_pdf.return_value = [chunk]
        mock_processor_cls.return_value = mock_processor
        mock_vector_store = MagicMock()
        mock_get_vs.return_value = mock_vector_store

        # Act：clear_existing=True 进入“重建已有索引”分支。
        process_pdf_task(
            file_path=MagicMock(),
            file_id="file123",
            file_name="test.pdf",
            clear_existing=True,
        )

        # Assert：必须走具备回滚保护的 replace；旧的 delete+add 组合绝不能再被调用。
        mock_vector_store.replace_document_for_file.assert_called_once_with(
            [chunk],
            "file123",
        )
        mock_vector_store.delete_by_file_id.assert_not_called()
        mock_vector_store.add_document_for_file.assert_not_called()


class TestStreamEndpoint:
    """基于 Server-Sent Events 的流式 Chat 接口基础测试。"""

    @patch("app.agents.main_agent.get_main_agent")
    def test_stream_simple_message(self, mock_get_agent, client):
        """正常流式消息应建立成功，并返回 SSE 媒体类型。"""
        # astream_events 返回异步生成器；yield 模拟模型逐步产出一个 token。
        async def mock_stream(*args, **kwargs):
            yield {"event": "on_chat_model_stream", "data": {"chunk": MagicMock(content="Hello")}}

        mock_agent = MagicMock()
        mock_agent.astream_events.return_value = mock_stream()
        mock_get_agent.return_value = mock_agent

        response = client.post(
            "/api/chat/stream",
            json={"messages": [{"role": "user", "content": "Hello"}]}
        )

        # SSE 是否真正按事件顺序输出由核心测试验证；这里先检查传输层契约。
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

    def test_stream_empty_messages(self, client):
        """空消息的 SSE 请求仍建立 HTTP 200 响应，由流内 error 事件报告错误。"""
        response = client.post(
            "/api/chat/stream",
            json={"messages": []}
        )

        # 与普通 Chat 的 400 不同，SSE 响应头已发出后要通过事件表达业务错误。
        assert response.status_code == 200
