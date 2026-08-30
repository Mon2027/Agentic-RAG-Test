"""文档 API 的 file_id 输入校验回归测试。

这个文件验证通配符误删缺陷修复后的“第一道防线”：三个文档详情类路由都把
``file_id`` 声明为 UUID。FastAPI/Pydantic 应在请求进入路由函数之前拒绝 ``*``，
返回 422，并且任何配置读取、文件扫描或向量库操作都不能发生。

它与 ``test_phase2_file_id_risk_evidence.py`` 的区别是：前者保存完整缺陷证据，
本文件则用参数化测试精确验证修复契约，适合作为长期回归用例。
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api import routes
from app.main import app


@pytest.fixture
def client():
    """创建用于路径参数校验测试的 FastAPI 客户端。

    使用上下文管理器可以正确触发应用 lifespan；yield 后 pytest 会自动退出上下文。
    """
    with TestClient(app) as test_client:
        yield test_client


# 用一份参数表覆盖 GET 详情、DELETE 删除和 POST 重建索引三个入口。
# pytest 会为每组 method/path 生成独立用例，因此能明确报告哪个路由发生回归。
@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/api/documents/*"),
        ("delete", "/api/documents/*"),
        ("post", "/api/documents/*/reindex"),
    ],
)
def test_wildcard_file_id_is_rejected_before_side_effects(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    path: str,
):
    """所有文档路由都必须在业务逻辑运行前拒绝通配符 file_id。"""
    # 把关键依赖替换为 spy。若路径校验未能提前拦截请求，路由读取这些依赖时，
    # MagicMock 就会记录调用，后面的 assert_not_called 会立即暴露问题。
    get_settings = MagicMock()
    get_vector_store = MagicMock()
    monkeypatch.setattr(routes, "get_settings", get_settings)
    monkeypatch.setattr(routes, "get_vector_store", get_vector_store)

    # getattr 根据参数动态取得 client.get/delete/post，避免复制三份相同测试逻辑。
    response = getattr(client, method)(path)

    # 422 Unprocessable Entity 表示请求结构可解析，但路径参数不符合 UUID 类型约束。
    assert response.status_code == 422
    # 这两个断言证明请求停在框架校验层，尚未进入会产生副作用的路由业务逻辑。
    get_settings.assert_not_called()
    get_vector_store.assert_not_called()
