"""用于复现并记录“通配符文档 ID”缺陷证据的诊断测试。

历史风险是：删除接口若直接把用户提供的 ``file_id`` 拼进 ``Path.glob()``，那么
``*`` 会被当作文件通配符，可能匹配并删除目录中的所有报告和数据文件。

本测试采用“缺陷证据”写法：请求前放置两个哨兵文件，请求后把状态码、文件是否
存活以及向量删除方法是否被调用汇总成 observed，再与安全期望 expected 比较。
一旦回归，失败信息会完整展示所有观察项，便于快速定位破坏发生在哪一层。
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api import routes
from app.main import app


@pytest.fixture
def isolated_risk_client(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """提供隔离的文档目录、测试客户端和可观察调用的向量库 spy。

    ``tmp_path`` 保证即使缺陷真的存在，被删除的也只是本用例临时文件；
    ``monkeypatch`` 在用例结束后自动恢复 routes 中原有依赖。
    """
    # Arrange：为两类上传文件创建当前用例独享的空目录。
    reports_path = tmp_path / "reports"
    uploads_path = tmp_path / "uploads"
    reports_path.mkdir()
    uploads_path.mkdir()

    # 被测删除路由只需要这两个配置字段，因此无需构造完整 Settings 对象。
    settings = SimpleNamespace(
        reports_path=reports_path,
        uploads_path=uploads_path,
    )
    # MagicMock 在这里兼具 stub 和 spy 的作用：阻断真实向量库，并记录调用行为。
    vector_store = MagicMock()
    # patch 必须发生在 TestClient 发请求之前，路由执行时才会读取测试配置。
    monkeypatch.setattr(routes, "get_settings", lambda: settings)
    monkeypatch.setattr(routes, "get_vector_store", lambda: vector_store)

    # yield 元组把客户端、路径和 spy 一起交给测试函数；上下文结束会关闭客户端。
    with TestClient(app) as client:
        yield client, reports_path, uploads_path, vector_store


def test_wildcard_file_id_reports_complete_deletion_evidence(isolated_risk_client):
    """通配符 ID 必须在进入文件或向量删除操作之前被拒绝。"""
    # pytest 按 fixture 名自动注入返回值，这里再解包成有意义的局部变量。
    client, reports_path, uploads_path, vector_store = isolated_risk_client
    # 两个文件是“哨兵”：固定名称和内容可证明请求是否造成误删或篡改。
    report = reports_path / "report-a_company.pdf"
    upload = uploads_path / "dataset-b_metrics.csv"
    report.write_bytes(b"report must survive")
    upload.write_bytes(b"dataset must survive")

    # Act：URL 中直接传入非法 *。修复后 file_id 的 UUID 校验会在路由执行前拦截它。
    response = client.delete("/api/documents/*")

    # Collect evidence：把多个副作用压缩成结构化证据，失败时比连续 assert 信息更全。
    observed = {
        # 400/404/422 都可代表安全拒绝；UUID 路径参数验证通常返回 422。
        "request_rejected": response.status_code in {400, 404, 422},
        "report_survived": (
            report.exists() and report.read_bytes() == b"report must survive"
        ),
        "upload_survived": (
            upload.exists() and upload.read_bytes() == b"dataset must survive"
        ),
        # 即使磁盘文件幸存，向量删除方法也绝不能收到这个非法 ID。
        "vector_delete_called": vector_store.delete_by_file_id.called,
    }
    # 安全契约：拒绝请求、两个哨兵均存活、向量库删除未被调用。
    expected = {
        "request_rejected": True,
        "report_survived": True,
        "upload_survived": True,
        "vector_delete_called": False,
    }
    # 自定义失败消息把 HTTP 响应与全部观察值一并输出，形成可复现的缺陷证据。
    assert observed == expected, (
        f"status_code={response.status_code}, "
        f"response_body={response.json()}, observed={observed}"
    )
