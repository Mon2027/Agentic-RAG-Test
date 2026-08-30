"""Agent 路由批量评测命令入口的产物持久化测试。

批量评测会把 CSV 等输入暂存到临时 uploads 目录，数据 Agent 生成的图表也位于
``uploads/charts``。临时目录清理前必须把图表复制到永久 artifacts 目录，并记录
来源 URL、绝对路径、SHA-256 和文件大小，评测结果才具有可复核性。
"""

from pathlib import Path

from app.evaluation.run_agent_route_evaluation import _persist_chart_artifacts


def test_persist_chart_artifacts_before_temporary_upload_cleanup(tmp_path: Path):
    """临时图表应被复制到永久目录，并返回完整的可校验产物清单。"""
    # Arrange：模拟一次评测在临时 uploads/charts 下生成 PNG。
    uploads_path = tmp_path / "temporary-uploads"
    charts_path = uploads_path / "charts"
    charts_path.mkdir(parents=True)
    source = charts_path / "chart_abc12345.png"
    source.write_bytes(b"test-chart-content")
    artifacts_path = tmp_path / "permanent-artifacts"

    # Act：持久化必须发生在上层上下文清理 temporary-uploads 之前。
    persisted = _persist_chart_artifacts(uploads_path, artifacts_path)

    # Assert：校验文件内容，以及报告中用于追踪和防篡改的全部元数据。
    destination = artifacts_path / source.name
    assert destination.read_bytes() == b"test-chart-content"
    assert persisted == [{
        "source_url": "/static/charts/chart_abc12345.png",
        "path": str(destination.resolve()),
        "sha256": "E11DE38610913F1F3A691F76A63A75D5A80AFDAC5992F6AAD3C4C9B1D065459D",
        "size_bytes": 18,
    }]


def test_persist_chart_artifacts_does_not_create_empty_directory(tmp_path: Path):
    """没有图表时应返回空清单，也不应制造无意义的 artifacts 空目录。"""
    uploads_path = tmp_path / "temporary-uploads"
    uploads_path.mkdir()
    artifacts_path = tmp_path / "permanent-artifacts"

    # 延迟创建目标目录可以让调用方通过目录是否存在判断本轮是否真的有产物。
    assert _persist_chart_artifacts(uploads_path, artifacts_path) == []
    assert not artifacts_path.exists()
