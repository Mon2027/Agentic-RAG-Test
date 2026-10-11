"""Agent 路由批量评测命令入口的产物持久化测试。

批量评测会把 CSV 等输入暂存到临时 uploads 目录，数据 Agent 生成的图表也位于
``uploads/charts``。临时目录清理前必须把图表复制到永久 artifacts 目录，并记录
来源 URL、绝对路径、SHA-256 和文件大小，评测结果才具有可复核性。

本文件直接调用 _persist_chart_artifacts，不运行命令行 main() 或创建真实 Agent。
两个用例分别覆盖“存在图表”和“没有 charts 目录”两种情况，均使用 pytest 临时目录。
测试中的 permanent-artifacts 仅模拟生产中的持久化位置，仍位于 tmp_path 下。
第一个用例检查复制和清单生成，没有实际删除源目录，也不验证上层清理调用顺序。
"""

# Path 支持路径拼接、目录创建和文件读写；参数类型标注便于编辑器提示相关方法。
from pathlib import Path

# 导入辅助函数即可测试文件保存逻辑，无需调用大模型或生成真正的图表。
from app.evaluation.run_agent_route_evaluation import _persist_chart_artifacts


def test_persist_chart_artifacts_before_temporary_upload_cleanup(tmp_path: Path):
    """临时图表应被复制到永久目录，并返回完整的可校验产物清单。"""
    # tmp_path 是 pytest 内置 fixture：测试声明同名参数后，pytest 自动注入
    # 当前用例独享的临时目录 Path，不需要手动创建 fixture 或传参调用测试。
    # Arrange：模拟一次评测在临时 uploads/charts 下生成 PNG。
    # Path 的 / 运算符用于拼接目录名，不是数值除法。
    uploads_path = tmp_path / "temporary-uploads"
    charts_path = uploads_path / "charts"
    # parents=True 同时创建缺失的上级 temporary-uploads 目录。
    charts_path.mkdir(parents=True)
    source = charts_path / "chart_abc12345.png"
    # b 前缀表示 bytes；这里写入18个 ASCII 字节，无须构造有效 PNG 图像，
    # 因为被测函数只复制文件并计算校验信息，不会解码图片。
    source.write_bytes(b"test-chart-content")
    # 此时只是构造目标路径，并未 mkdir；创建目标目录应由被测函数完成。
    artifacts_path = tmp_path / "permanent-artifacts"

    # Act：持久化必须发生在上层上下文清理 temporary-uploads 之前。
    persisted = _persist_chart_artifacts(uploads_path, artifacts_path)

    # Assert：校验文件内容，以及报告中用于追踪和内容校验的全部元数据。
    # source.name 只取文件名，让复制后的文件位于 artifacts_path 下且名称不变。
    destination = artifacts_path / source.name
    # 读取目标文件的原始字节，检查复制内容一致；文件未生成时也会直接导致测试失败。
    assert destination.read_bytes() == b"test-chart-content"
    # 整个清单与期望相等，同时检查只有一个产物、字段和值均正确。
    # source_url 是原图表的应用 URL 标识，这里不会发 HTTP 请求验证其可访问性。
    # resolve() 得到绝对路径；str() 转为可写入 JSON 的字符串。
    # SHA-256 是内容摘要，下面固定期望值来自测试字节，用于检查实际计算是否正确。
    # 摘要可用于发现内容变化；size_bytes 是文件字节数，不是图片的宽度或高度。
    assert persisted == [{
        "source_url": "/static/charts/chart_abc12345.png",
        "path": str(destination.resolve()),
        "sha256": "E11DE38610913F1F3A691F76A63A75D5A80AFDAC5992F6AAD3C4C9B1D065459D",
        "size_bytes": 18,
    }]


def test_persist_chart_artifacts_does_not_create_empty_directory(tmp_path: Path):
    """没有图表时应返回空清单，也不应制造无意义的 artifacts 空目录。"""
    uploads_path = tmp_path / "temporary-uploads"
    # 仅创建 uploads，不创建 uploads/charts，模拟没有生成图表的评测。
    uploads_path.mkdir()
    artifacts_path = tmp_path / "permanent-artifacts"

    # 延迟创建目标目录可以让调用方通过目录是否存在判断本轮是否真的有产物。
    # 同时检查返回值和文件系统状态，防止返回空清单却仍创建了目标目录。
    assert _persist_chart_artifacts(uploads_path, artifacts_path) == []
    assert not artifacts_path.exists()
