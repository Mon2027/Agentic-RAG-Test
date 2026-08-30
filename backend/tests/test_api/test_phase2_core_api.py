"""第二阶段核心 API 与高价值风险场景的自动化测试。

本文件很适合用来学习一条完整的接口自动化测试链路：

* 使用 FastAPI ``TestClient`` 发起真实的 HTTP 请求；
* 用 ``tmp_path`` 隔离文件系统，避免测试污染正式数据；
* 用 ``monkeypatch`` 和 ``MagicMock`` 替换大模型、向量库等外部依赖；
* 对状态码、响应体、落盘文件、内存状态以及 mock 调用同时进行断言；
* 验证普通 JSON 接口、文件上传接口和 SSE 流式接口；
* 用失败场景验证接口不会留下不完整状态或破坏旧数据。

所有文件写入都发生在 pytest 创建的临时目录中，外部服务也都被测试替身替换。
最后两个风险用例描述的是系统必须满足的安全结果：非法通配符 file_id 不能触发
批量删除，重建索引失败时必须保留旧的可检索分块。
"""

import json
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api import routes
from app.main import app


@dataclass
class APIContext:
    """集中保存单个 API 用例所需的隔离资源。

    ``@dataclass`` 会自动生成初始化方法，因此 fixture 可以把两个临时目录和一个
    向量库 mock 打包返回；测试函数通过字段名访问，比返回匿名元组更清楚。
    """

    # PDF 研究报告的测试专用保存目录。
    reports_path: Path
    # CSV/Excel 数据文件的测试专用保存目录。
    uploads_path: Path
    # 向量库测试替身，可用于断言某个方法是否被调用以及调用参数。
    vector_store: MagicMock


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """动态生成一个结构合法的单页 PDF，并以字节串返回。

    使用 PDF 库生成真实文件结构，比只伪造 ``%PDF`` 文件头更能覆盖解析流程。
    fixture 默认是 function 作用域，每个使用它的测试函数都会重新生成一份数据。
    """
    # 放在函数内延迟导入，只有使用此 fixture 的用例才需要加载 pypdf。
    from pypdf import PdfWriter

    # BytesIO 是内存中的二进制文件对象，不需要在磁盘上创建中间文件。
    buffer = BytesIO()
    writer = PdfWriter()
    # 612×792 point 对应常见的 Letter 页面尺寸；页面内容留空即可验证文件结构。
    writer.add_blank_page(width=612, height=792)
    writer.write(buffer)
    # getvalue() 取出完整 PDF 二进制内容，供 multipart/form-data 上传。
    return buffer.getvalue()


@pytest.fixture
def sample_xlsx_bytes() -> bytes:
    """动态生成包含代表性表格数据的真实 Excel 工作簿字节串。"""
    # 与 PDF fixture 一样采用延迟导入，降低不相关测试的加载成本。
    from openpyxl import Workbook

    buffer = BytesIO()
    workbook = Workbook()
    # 默认工作表足以模拟一个最小但真实可打开的业务数据文件。
    worksheet = workbook.active
    worksheet.title = "metrics"
    # 第一行是字段名，后两行包含正、负增长率，用于代表常见财务数据形态。
    worksheet.append(["company", "revenue", "growth_rate"])
    worksheet.append(["Alpha", 1200, 0.12])
    worksheet.append(["Beta", 980, -0.03])
    # 直接保存到内存缓冲区，关闭 workbook 后返回最终的 ZIP/XLSX 二进制内容。
    workbook.save(buffer)
    workbook.close()
    return buffer.getvalue()


@pytest.fixture
def api_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> APIContext:
    """把每个 API 用例的副作用全部重定向到独立临时环境。

    ``tmp_path`` 和 ``monkeypatch`` 都是 pytest 内置 fixture：前者为当前用例提供
    唯一临时目录，后者会在用例结束后自动恢复被替换的属性。

    这是一个 yield fixture：yield 之前是 Arrange/准备阶段，yield 返回测试资源，
    yield 之后是 teardown/清理阶段，即使用例失败也会执行。
    """
    # 模拟应用配置中的报告目录与普通上传目录。
    reports_path = tmp_path / "reports"
    uploads_path = tmp_path / "uploads"
    reports_path.mkdir()
    uploads_path.mkdir()

    # SimpleNamespace 可快速构造“只包含被测代码真正会读取字段”的轻量配置对象。
    settings = SimpleNamespace(
        reports_path=reports_path,
        uploads_path=uploads_path,
        chunk_size=1000,
        chunk_overlap=200,
    )
    # MagicMock 代替真实向量数据库，避免打开 Chroma/SQLite 或加载嵌入模型。
    vector_store = MagicMock()
    # 文档列表接口会遍历 list_files()，显式返回空列表可使行为稳定且符合类型预期。
    vector_store.list_files.return_value = []

    # 注意 patch 的位置是 routes 模块中的“使用点”，而不是函数最初的定义位置。
    monkeypatch.setattr(routes, "get_settings", lambda: settings)
    monkeypatch.setattr(routes, "get_vector_store", lambda: vector_store)
    # document_status 和 sessions 是模块级可变字典，测试前清空以避免用例相互影响。
    routes.document_status.clear()
    routes.sessions.clear()

    yield APIContext(reports_path, uploads_path, vector_store)

    # 测试结束后再次清空全局内存状态，确保后续测试从干净环境开始。
    routes.document_status.clear()
    routes.sessions.clear()


@pytest.fixture
def client(api_context: APIContext):
    """在路由依赖完成隔离后创建 FastAPI 测试客户端。

    参数 ``api_context`` 即使没有在函数体中直接使用也不能删除：pytest 会先执行
    依赖 fixture，保证 monkeypatch 和临时目录已准备好，然后才启动 TestClient。
    使用上下文管理器还能正确执行 FastAPI 的 lifespan 启动与关闭逻辑。
    """
    with TestClient(app) as test_client:
        yield test_client


def _single_saved_file(directory: Path) -> Path:
    """断言目录中恰好有一个普通文件，并返回该文件路径。

    这个辅助函数不仅方便取文件，还同时验证接口没有多写或漏写文件。
    """
    files = [path for path in directory.iterdir() if path.is_file()]
    assert len(files) == 1
    return files[0]


def _sse_events(response) -> list[dict]:
    """从已完成的 SSE 响应中提取并反序列化所有 JSON 事件。

    SSE 每条数据行的格式为 ``data: <JSON>``，空行用于分隔事件。TestClient 会等
    流式响应完成并把文本放到 ``response.text``，因此这里可以按行解析。
    """
    events = []
    for line in response.text.splitlines():
        # 忽略空行、event/id/retry 等其他 SSE 字段，只处理 data 行。
        if line.startswith("data: "):
            events.append(json.loads(line.removeprefix("data: ")))
    return events


class TestCoreUploadAndDocuments:
    """PDF/CSV/Excel 上传与文档管理的核心流程测试。"""

    def test_valid_pdf_is_saved_and_processing_is_scheduled(
        self,
        client: TestClient,
        api_context: APIContext,
        sample_pdf_bytes: bytes,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """合法 PDF 应原样落盘、生成 UUID，并安排后台解析任务。"""
        # Arrange：替换真实 PDF 处理任务，避免测试执行解析和向量化。
        process_pdf = MagicMock()
        monkeypatch.setattr(routes, "process_pdf_task", process_pdf)

        # Act：files 参数会由 TestClient 编码为 multipart/form-data。
        # 元组依次表示客户端文件名、二进制内容和 MIME 类型。
        response = client.post(
            "/api/upload/report",
            files={"file": ("quarterly-report.pdf", sample_pdf_bytes, "application/pdf")},
        )

        # Assert（协议层）：请求成功，响应包含格式合法的 UUID 和准确文件大小。
        assert response.status_code == 200
        body = response.json()
        file_id = body["file_id"]
        # 构造 UUID 对象本身就是格式校验；格式非法时这里会直接抛异常使测试失败。
        uuid.UUID(file_id)
        assert body["metadata"]["file_size"] == len(sample_pdf_bytes)

        # Assert（副作用层）：文件名、内容、处理状态以及后台任务参数都必须正确。
        saved_file = _single_saved_file(api_context.reports_path)
        assert saved_file.name == f"{file_id}_quarterly-report.pdf"
        assert saved_file.read_bytes() == sample_pdf_bytes
        assert routes.document_status[file_id]["status"] == "pending"
        process_pdf.assert_called_once_with(
            saved_file,
            file_id,
            "quarterly-report.pdf",
            False,
        )

    def test_csv_upload_list_get_and_delete_workflow(
        self,
        client: TestClient,
        api_context: APIContext,
        sample_csv_path: Path,
    ):
        """覆盖 CSV 从上传、列表、详情到删除的完整文档生命周期。"""
        # 复用 conftest.py 提供的真实 CSV 样例，而不是在本用例中重复造数据。
        csv_content = sample_csv_path.read_bytes()

        # 第一步：上传 CSV 并确认服务端原样保存其二进制内容。
        upload_response = client.post(
            "/api/upload/data",
            files={"file": ("companies.csv", csv_content, "text/csv")},
        )

        assert upload_response.status_code == 200
        file_id = upload_response.json()["file_id"]
        saved_file = _single_saved_file(api_context.uploads_path)
        assert saved_file.read_bytes() == csv_content

        # 第二步：列表接口应能从上传目录发现文件并还原原始文件名及类型。
        list_response = client.get("/api/documents")
        assert list_response.status_code == 200
        # 转为以 file_id 为键的字典，避免测试依赖接口返回顺序。
        listed = {item["file_id"]: item for item in list_response.json()}
        assert listed[file_id]["file_name"] == "companies.csv"
        assert listed[file_id]["file_type"] == "csv"
        assert listed[file_id]["status"] == "completed"

        # 第三步：详情接口应返回同一个文件，并报告真实字节大小。
        get_response = client.get(f"/api/documents/{file_id}")
        assert get_response.status_code == 200
        assert get_response.json()["file_size"] == len(csv_content)

        # 第四步：删除接口要同时删除磁盘文件和相同 file_id 的向量数据。
        delete_response = client.delete(f"/api/documents/{file_id}")
        assert delete_response.status_code == 200
        assert not saved_file.exists()
        api_context.vector_store.delete_by_file_id.assert_called_once_with(file_id)
        # 再次查询得到 404，验证删除后的外部可观察状态，而不只相信删除响应。
        assert client.get(f"/api/documents/{file_id}").status_code == 404

    def test_valid_excel_upload_preserves_workbook_bytes(
        self,
        client: TestClient,
        api_context: APIContext,
        sample_xlsx_bytes: bytes,
    ):
        """合法 XLSX 上传后，文件类型、大小和落盘字节均应保持一致。"""
        response = client.post(
            "/api/upload/data",
            files={
                "file": (
                    "metrics.xlsx",
                    sample_xlsx_bytes,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["file_type"] == "xlsx"
        assert body["metadata"]["file_size"] == len(sample_xlsx_bytes)
        assert _single_saved_file(api_context.uploads_path).read_bytes() == sample_xlsx_bytes

    # 参数化让同一套断言覆盖“报告接口拒绝 TXT”和“数据接口拒绝 JSON”。
    # pytest 会把参数表中的每一行生成一个独立测试用例，失败报告也会分别展示。
    @pytest.mark.parametrize(
        ("endpoint", "filename", "content_type"),
        [
            ("/api/upload/report", "notes.txt", "text/plain"),
            ("/api/upload/data", "payload.json", "application/json"),
        ],
    )
    def test_invalid_extension_is_rejected_without_saving_file(
        self,
        client: TestClient,
        api_context: APIContext,
        endpoint: str,
        filename: str,
        content_type: str,
    ):
        """不支持的扩展名必须返回 400，且不能在任何目录留下文件。"""
        response = client.post(
            endpoint,
            files={"file": (filename, b"must not be stored", content_type)},
        )

        assert response.status_code == 400
        assert list(api_context.reports_path.iterdir()) == []
        assert list(api_context.uploads_path.iterdir()) == []

    # 覆盖空文件和“有 PDF 文件头但结构损坏”两类典型坏文件。
    @pytest.mark.parametrize("invalid_pdf", [b"", b"%PDF-1.7\ncorrupt-and-incomplete"])
    def test_invalid_pdf_processing_records_failure_without_adding_vectors(
        self,
        client: TestClient,
        api_context: APIContext,
        invalid_pdf: bytes,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """PDF 解析失败应记录 failed 状态，且绝不能写入向量库。"""
        # 让处理器稳定抛出解析异常，精确控制要验证的失败分支。
        processor = MagicMock()
        processor.process_pdf.side_effect = ValueError("invalid PDF structure")
        monkeypatch.setattr(routes, "DocumentProcessor", MagicMock(return_value=processor))

        response = client.post(
            "/api/upload/report",
            files={"file": ("broken.pdf", invalid_pdf, "application/pdf")},
        )

        # 上传动作本身成功，所以接口先返回 200；TestClient 随后执行后台任务，
        # 最终处理结果则记录在 document_status 中。
        assert response.status_code == 200
        file_id = response.json()["file_id"]
        status = routes.document_status[file_id]
        assert status["status"] == "failed"
        assert "invalid PDF structure" in status["error"]
        api_context.vector_store.add_document_for_file.assert_not_called()


class TestCoreChatAndSSE:
    """Chat 普通响应与 SSE 流式响应的正常、异常行为测试。"""

    def test_chat_success_persists_complete_session(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Chat 成功时应返回助手消息，并完整保存本轮会话历史。"""
        # 模拟智能体 invoke 的 LangChain 风格返回值，避免调用真实大模型。
        agent = MagicMock()
        agent.invoke.return_value = {"messages": [MagicMock(content="analysis complete")]}
        monkeypatch.setattr("app.agents.main_agent.get_main_agent", lambda: agent)

        response = client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "Analyze revenue"}]},
        )

        # 同时验证 HTTP 响应契约和服务端会话持久化结果。
        assert response.status_code == 200
        body = response.json()
        assert body["message"]["role"] == "assistant"
        assert body["message"]["content"] == "analysis complete"
        assert "timestamp" in body["message"]
        stored_messages = routes.sessions[body["session_id"]]["messages"]
        assert [message.content for message in stored_messages] == [
            "Analyze revenue",
            "analysis complete",
        ]

    # 同一失败契约需要覆盖普通运行错误和超时错误。
    @pytest.mark.parametrize("agent_error", [RuntimeError("model failed"), TimeoutError("model timed out")])
    def test_chat_agent_error_returns_500_without_creating_session(
        self,
        client: TestClient,
        agent_error: Exception,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """智能体调用失败应返回 500，且不能保存半成品会话。"""
        agent = MagicMock()
        # side_effect 让 mock 在被调用时抛出指定异常，而不是返回正常值。
        agent.invoke.side_effect = agent_error
        monkeypatch.setattr("app.agents.main_agent.get_main_agent", lambda: agent)

        response = client.post(
            "/api/chat",
            json={
                "session_id": "failed-session",
                "messages": [{"role": "user", "content": "Analyze revenue"}],
            },
        )

        assert response.status_code == 500
        assert "failed-session" not in routes.sessions

    def test_sse_success_emits_start_tokens_and_matching_end(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """SSE 成功时应按 start → token* → end 的顺序发送完整内容。"""
        # 异步生成器模拟 LangChain astream_events() 逐块产出的模型事件。
        async def stream_events(*args, **kwargs):
            yield {"event": "on_chat_model_stream", "data": {"chunk": MagicMock(content="Hello ")}}
            yield {"event": "on_chat_model_stream", "data": {"chunk": MagicMock(content="world")}}

        agent = MagicMock()
        agent.astream_events.return_value = stream_events()
        monkeypatch.setattr("app.agents.main_agent.get_main_agent", lambda: agent)

        response = client.post(
            "/api/chat/stream",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )

        # SSE 使用长连接传输，但其 HTTP Content-Type 必须是 text/event-stream。
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        events = _sse_events(response)
        # 事件顺序本身就是流式接口的重要协议契约。
        assert [event["type"] for event in events] == ["start", "token", "token", "end"]
        # 拼接所有 token 应等于 end 事件中的最终完整文本。
        assert "".join(event["content"] for event in events if event["type"] == "token") == "Hello world"
        assert events[-1]["content"] == "Hello world"
        assert events[0]["session_id"] == events[-1]["session_id"]
        assert events[-1]["session_id"] in routes.sessions

    def test_sse_empty_messages_emits_only_error(self, client: TestClient):
        """空消息通过 SSE error 事件表达，不创建会话。"""
        response = client.post("/api/chat/stream", json={"messages": []})

        # 流式响应已经成功建立，所以 HTTP 层仍为 200；业务错误位于事件数据内。
        assert response.status_code == 200
        events = _sse_events(response)
        assert events == [{"type": "error", "message": "No messages provided"}]
        assert routes.sessions == {}

    def test_sse_agent_error_has_no_end_event_or_session(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """流处理中断时只发送 start/error，不发送误导性的 end 或保存会话。"""
        async def failing_stream(*args, **kwargs):
            raise RuntimeError("stream interrupted")
            # Python 只有看到 yield 才会把函数编译为异步生成器；本行实际不可达。
            yield  # pragma: no cover - keeps this function an async generator

        agent = MagicMock()
        agent.astream_events.return_value = failing_stream()
        monkeypatch.setattr("app.agents.main_agent.get_main_agent", lambda: agent)

        response = client.post(
            "/api/chat/stream",
            json={
                "session_id": "failed-stream",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        events = _sse_events(response)
        assert [event["type"] for event in events] == ["start", "error"]
        assert events[-1]["message"] == "stream interrupted"
        assert "failed-stream" not in routes.sessions


class TestHighValueRisks:
    """第二阶段两个高价值风险候选项必须满足的安全结果。"""

    def test_wildcard_file_id_cannot_delete_unrelated_documents(
        self,
        client: TestClient,
        api_context: APIContext,
    ):
        """通配符 file_id 必须被拒绝，不能删除任何无关文档或向量。"""
        # Arrange：创建两个“哨兵文件”，内容用于证明它们请求后仍是原文件。
        report = api_context.reports_path / "report-a_company.pdf"
        upload = api_context.uploads_path / "dataset-b_metrics.csv"
        report.write_bytes(b"report must survive")
        upload.write_bytes(b"dataset must survive")

        # Act：历史缺陷把 * 拼进 glob，可能匹配并删除目录中的全部文件。
        response = client.delete("/api/documents/*")

        # 400/404/422 都表示请求被安全拒绝；当前 UUID 路径校验通常返回 422。
        assert response.status_code in {400, 404, 422}
        assert report.read_bytes() == b"report must survive"
        assert upload.read_bytes() == b"dataset must survive"
        api_context.vector_store.delete_by_file_id.assert_not_called()

    def test_failed_reindex_preserves_previous_chunks(
        self,
        api_context: APIContext,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """重建 PDF 索引失败时，旧的可检索分块必须原封不动保留。"""
        file_id = "existing-report"
        # 这份旧数据是测试哨兵，用来观察失败后是否发生了破坏性删除。
        old_chunks = [
            {"content": "old searchable evidence", "metadata": {"file_id": file_id}}
        ]

        class StatefulVectorStore:
            """最小有状态向量库替身，比 MagicMock 更适合观察数据前后变化。"""

            def __init__(self):
                # 复制列表，避免后续修改直接影响外部 old_chunks 期望值。
                self.chunks = list(old_chunks)

            def delete_by_file_id(self, target_file_id: str) -> int:
                """模拟按 file_id 删除已有向量，并返回删除数量。"""
                deleted = len(self.chunks)
                self.chunks = [
                    chunk
                    for chunk in self.chunks
                    if chunk["metadata"]["file_id"] != target_file_id
                ]
                return deleted

            def add_document_for_file(self, chunks, target_file_id: str) -> None:
                """模拟追加新向量；target_file_id 由分块 metadata 表示。"""
                self.chunks.extend(chunks)

            def get_file_chunks(self, target_file_id: str) -> list[dict]:
                """返回指定文件的所有分块，供最终状态断言。"""
                return [
                    chunk
                    for chunk in self.chunks
                    if chunk["metadata"]["file_id"] == target_file_id
                ]

        vector_store = StatefulVectorStore()
        processor = MagicMock()
        # 模拟新 PDF 无法解析；安全实现应在拿到有效新分块前完全不碰旧索引。
        processor.process_pdf.side_effect = ValueError("replacement PDF cannot be parsed")
        monkeypatch.setattr(routes, "get_vector_store", lambda: vector_store)
        monkeypatch.setattr(routes, "DocumentProcessor", MagicMock(return_value=processor))

        # 直接调用后台任务函数以聚焦索引替换逻辑；clear_existing=True 表示重建索引。
        routes.process_pdf_task(
            file_path=api_context.reports_path / f"{file_id}_report.pdf",
            file_id=file_id,
            file_name="report.pdf",
            clear_existing=True,
        )

        # 任务状态应明确失败，但旧的搜索证据仍必须存在。
        assert routes.document_status[file_id]["status"] == "failed"
        assert vector_store.get_file_chunks(file_id) == old_chunks
