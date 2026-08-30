"""API 路由的应用级端到端（End-to-End）集成测试。

覆盖的完整功能包括：

* 健康检查；
* PDF、CSV 和 Excel 文件上传；
* 文档查询与删除；
* 普通 Chat、带会话 Chat 和 SSE 流式 Chat；
* 非法 JSON、字段缺失等异常输入；
* 跨多个接口的 CSV 生命周期和连续对话流程。

这里的“端到端”是指请求会从 FastAPI 入口经过请求校验、路由业务逻辑、文件系统
和响应序列化，多个用例还会串联不同接口。它仍然使用进程内 ``TestClient``，并把
真实大模型替换为 mock，因此不包含真实网络部署和外部模型服务，属于应用级 E2E。

全局 ``conftest.py`` 已把上传与报告目录重定向到测试临时目录，避免污染开发数据。
"""

import os
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# 必须在导入 app 之前声明测试环境，因为应用配置可能在模块导入阶段就被读取。
# setdefault 只在变量不存在时设置；更强的强制隔离由上级 conftest.py 统一完成。
os.environ.setdefault("APP_ENV", "testing")


# 查询和删除“不存在文档”的测试仍需使用格式合法的 UUID，否则会先得到 422，
# 无法真正覆盖路由业务层的 404 分支。
MISSING_FILE_ID = "22222222-2222-4222-8222-222222222222"


class TestAPIHealthCheck:
    """健康检查端到端测试。"""

    def test_health_endpoint(self):
        """应用启动后，健康检查应返回 healthy 状态和版本信息。"""
        # 在测试函数内延迟导入 app，确保上面的测试环境变量已先设置。
        from app.main import app

        # TestClient 无需启动 uvicorn，即可完整执行 ASGI 请求链路。
        client = TestClient(app)
        response = client.get("/health")

        # 端到端断言既检查传输层状态码，也检查调用方真正依赖的业务字段。
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestAPIUploadFlow:
    """文件上传接口的成功与拒绝流程。"""

    @pytest.fixture
    def client(self):
        """为本测试类中的每个用例创建 FastAPI 测试客户端。"""
        from app.main import app

        return TestClient(app)

    def test_upload_csv_file(self, client: TestClient):
        """CSV 经 multipart 上传后应返回成功、类型和唯一 file_id。"""
        # BytesIO 模拟用户选择并上传的二进制文件流。
        csv_content = b"name,age\nAlice,28\nBob,35"
        # files 三元组依次为文件名、文件对象/字节和 MIME 类型。
        response = client.post(
            "/api/upload/data",
            files={"file": ("test.csv", BytesIO(csv_content), "text/csv")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["file_type"] == "csv"
        assert "file_id" in data

    def test_upload_excel_file(self, client: TestClient):
        """具有 XLSX 文件头和扩展名的文件应通过上传层基础校验。"""
        # XLSX 本质是 ZIP 容器，PK\x03\x04 是 ZIP 的常见文件头。
        # 这不是完整工作簿；本用例关注上传路由，不负责验证 Excel 解析能力。
        xlsx_content = b"PK\x03\x04" + b"\x00" * 100
        response = client.post(
            "/api/upload/data",
            files={
                "file": (
                    "test.xlsx",
                    BytesIO(xlsx_content),
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["file_type"] == "xlsx"

    def test_upload_pdf_file(self, client: TestClient):
        """PDF 报告应通过专用上传端点并返回 pdf 类型。"""
        # 最小 PDF 风格字节用于上传流程；后台解析依赖在测试环境中受控。
        pdf_content = b"%PDF-1.4\n%test pdf content"
        response = client.post(
            "/api/upload/report",
            files={"file": ("test.pdf", BytesIO(pdf_content), "application/pdf")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["file_type"] == "pdf"

    def test_upload_invalid_file_type(self, client: TestClient):
        """报告端点收到 TXT 时应返回 400，并提示只支持 PDF。"""
        response = client.post(
            "/api/upload/report",
            files={"file": ("test.txt", b"test content", "text/plain")},
        )

        assert response.status_code == 400
        assert "PDF" in response.json()["detail"]

    def test_upload_data_invalid_type(self, client: TestClient):
        """数据端点收到 PDF 时应返回 400，不允许混用上传入口。"""
        response = client.post(
            "/api/upload/data",
            files={"file": ("test.pdf", b"test content", "application/pdf")},
        )

        assert response.status_code == 400


class TestAPIDocumentManagement:
    """文档列表、详情与删除接口测试。"""

    @pytest.fixture
    def client(self):
        """为文档管理用例创建测试客户端。"""
        from app.main import app

        return TestClient(app)

    def test_list_documents_empty(self, client: TestClient):
        """文档为空时列表接口仍应返回 200 和 JSON 数组。"""
        response = client.get("/api/documents")

        assert response.status_code == 200
        # 不强制必须为空，避免与同一测试会话中其他上传用例的文件状态耦合。
        assert isinstance(response.json(), list)

    def test_get_nonexistent_document(self, client: TestClient):
        """查询不存在但格式合法的 file_id 应返回 404。"""
        response = client.get(f"/api/documents/{MISSING_FILE_ID}")

        assert response.status_code == 404

    def test_delete_nonexistent_document(self, client: TestClient):
        """删除不存在但格式合法的 file_id 应返回 404。"""
        response = client.delete(f"/api/documents/{MISSING_FILE_ID}")

        assert response.status_code == 404


class TestAPIChatFlow:
    """普通、带会话以及 SSE 流式 Chat 流程测试。"""

    @pytest.fixture
    def client(self):
        """为 Chat 用例创建测试客户端。"""
        from app.main import app

        return TestClient(app)

    def test_chat_endpoint_simple(self, client: TestClient):
        """单轮 Chat 应返回助手消息和服务器生成的 session_id。"""
        # patch 只在 with 代码块内生效，退出后自动恢复真实 get_main_agent。
        with patch("app.agents.main_agent.get_main_agent") as mock_get_agent:
            # 模拟 LangChain 智能体的 invoke 返回结构，避免访问真实大模型。
            mock_agent = MagicMock()
            mock_agent.invoke.return_value = {
                "messages": [MagicMock(content="Hello! How can I help you?")]
            }
            mock_get_agent.return_value = mock_agent

            # 请求体会经过 ChatRequest 的 Pydantic 校验后再进入路由。
            response = client.post(
                "/api/chat",
                json={"messages": [{"role": "user", "content": "Hello"}]},
            )

            assert response.status_code == 200
            data = response.json()
            assert "message" in data
            assert "session_id" in data

    def test_chat_endpoint_with_session(self, client: TestClient):
        """客户端显式传入 session_id 时，响应应沿用同一个 ID。"""
        with patch("app.agents.main_agent.get_main_agent") as mock_get_agent:
            mock_agent = MagicMock()
            mock_agent.invoke.return_value = {
                "messages": [MagicMock(content="Response")]
            }
            mock_get_agent.return_value = mock_agent

            response = client.post(
                "/api/chat",
                json={
                    "messages": [{"role": "user", "content": "Test"}],
                    "session_id": "test-session-123",
                },
            )

            assert response.status_code == 200
            data = response.json()
            # 这是会话连续性的基础契约，服务端不能无故生成新的 ID。
            assert data["session_id"] == "test-session-123"

    def test_chat_endpoint_empty_messages(self, client: TestClient):
        """普通 Chat 的空消息数组应返回 400 业务错误。"""
        response = client.post("/api/chat", json={"messages": []})

        assert response.status_code == 400
        assert response.json()["detail"] == "No messages provided"

    def test_chat_stream_endpoint(self, client: TestClient):
        """流式 Chat 应成功建立 SSE 响应并声明正确媒体类型。"""
        with patch("app.agents.main_agent.get_main_agent") as mock_get_agent:
            # astream_events 是异步迭代接口，因此 mock 也必须返回异步生成器。
            async def mock_stream(*args, **kwargs):
                yield {"event": "on_chat_model_stream", "data": {"chunk": MagicMock(content="Hello")}}

            mock_agent = MagicMock()
            mock_agent.astream_events.return_value = mock_stream()
            mock_get_agent.return_value = mock_agent

            response = client.post(
                "/api/chat/stream",
                json={"messages": [{"role": "user", "content": "Hello"}]},
            )

            # 本用例验证流传输层；具体 start/token/end 序列在核心 API 测试中验证。
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]


class TestAPIErrorHandling:
    """请求解析与 Pydantic 模型校验的异常场景测试。"""

    @pytest.fixture
    def client(self):
        """为异常输入用例创建测试客户端。"""
        from app.main import app

        return TestClient(app)

    def test_invalid_json(self, client: TestClient):
        """声明 JSON Content-Type 却发送非法文本时，应返回 422。"""
        # content 发送原始正文；与 json 参数不同，它不会自动做 JSON 序列化。
        response = client.post(
            "/api/chat",
            content="invalid json",
            headers={"Content-Type": "application/json"},
        )

        # 请求在 JSON 解析/校验阶段失败，路由业务函数不会执行。
        assert response.status_code == 422

    def test_missing_required_field(self, client: TestClient):
        """请求体缺少必填 messages 字段时，应返回 422。"""
        response = client.post("/api/chat", json={})

        assert response.status_code == 422

    def test_invalid_message_role(self, client: TestClient):
        """消息 role 不在允许枚举中时，应返回 422。"""
        response = client.post(
            "/api/chat",
            json={"messages": [{"role": "invalid", "content": "test"}]},
        )

        assert response.status_code == 422


class TestAPISessionManagement:
    """会话历史查询与删除接口测试。"""

    @pytest.fixture
    def client(self):
        """为会话管理用例创建测试客户端。"""
        from app.main import app

        return TestClient(app)

    def test_get_nonexistent_session(self, client: TestClient):
        """查询不存在的会话应返回 404。"""
        response = client.get("/api/sessions/nonexistent-session")

        assert response.status_code == 404

    def test_delete_session(self, client: TestClient):
        """删除会话采用幂等语义：会话不存在也返回成功。"""
        response = client.delete("/api/sessions/any-session-id")

        # 幂等删除便于客户端安全重试，不必先查询会话是否存在。
        assert response.status_code == 200
        assert response.json()["status"] == "deleted"


class TestEndToEndWorkflow:
    """跨多个 API 的完整业务工作流测试。"""

    @pytest.fixture
    def client(self):
        """为跨接口工作流创建同一个测试客户端。"""
        from app.main import app

        return TestClient(app)

    def test_complete_csv_workflow(self, client: TestClient):
        """验证 CSV 的完整生命周期：上传 → 列表 → 删除。"""
        # 第一步：上传 CSV，并从响应中取得后续接口所需的动态 file_id。
        csv_content = b"name,value\nA,100\nB,200"
        upload_response = client.post(
            "/api/upload/data",
            files={"file": ("test.csv", BytesIO(csv_content), "text/csv")},
        )

        assert upload_response.status_code == 200
        file_id = upload_response.json()["file_id"]

        # 第二步：调用文档列表。这里聚焦接口串联，只要求列表请求成功。
        list_response = client.get("/api/documents")
        assert list_response.status_code == 200

        # 第三步：使用第一步返回的 file_id 删除刚上传的同一份文档。
        delete_response = client.delete(f"/api/documents/{file_id}")
        assert delete_response.status_code == 200

    def test_complete_chat_workflow(self, client: TestClient):
        """验证同一 session_id 下的两轮连续对话。"""
        with patch("app.agents.main_agent.get_main_agent") as mock_get_agent:
            mock_agent = MagicMock()
            mock_agent.invoke.return_value = {
                "messages": [MagicMock(content="I can help with report analysis.")]
            }
            mock_get_agent.return_value = mock_agent

            # 第一步：发送初始消息，让服务端创建新会话并返回 session_id。
            response1 = client.post(
                "/api/chat",
                json={"messages": [{"role": "user", "content": "Hello"}]},
            )

            assert response1.status_code == 200
            session_id = response1.json()["session_id"]

            # 第二步：调整 mock 的下一次回答，并携带已有历史和同一个 session_id。
            mock_agent.invoke.return_value = {
                "messages": [MagicMock(content="I can analyze your reports.")]
            }

            response2 = client.post(
                "/api/chat",
                json={
                    "messages": [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "I can help with report analysis."},
                        {"role": "user", "content": "What can you do?"},
                    ],
                    "session_id": session_id,
                },
            )

            # 响应仍使用原 session_id，证明两次请求被串成同一条业务会话。
            assert response2.status_code == 200
            assert response2.json()["session_id"] == session_id
