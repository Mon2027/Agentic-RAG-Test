"""数据分析 Agent 工具的 CSV 读取、统计分析和图表生成测试。

这些函数使用 LangChain ``@tool`` 包装，因此测试通过 ``tool.invoke({...})`` 传入
结构化参数，而不是直接调用原始 Python 函数。测试重点包括：

* 根据 file_id 在上传目录中定位真实 CSV；
* 预览数据与全表统计的边界，防止从前 5 行推断整张表；
* describe、summary、correlation、value_counts 等分析类型；
* 模型偶尔传入字符串 columns 时的兼容修复与 Pydantic schema 约束；
* 折线图、柱状图的生成结果和全量数据覆盖说明。

每个涉及文件的用例都使用临时目录并 patch ``get_settings``，不会读取用户上传数据。
阅读时先看临时 CSV 如何建立，再看 ``invoke`` 的输入和返回文本断言；
其中部分用例只检查输出包含关键字，不等同于验证每个统计数值都正确。
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pydantic import ValidationError

from app.agents.data_analyst import (
    analyze_data,
    create_chart,
    create_data_analyst_tools,
    read_csv_file,
)


class TestReadCSVFile:
    """read_csv_file 文件定位、读取和预览边界测试。"""

    @patch("app.agents.data_analyst.get_settings")
    def test_read_nonexistent_file(self, mock_get_settings):
        """file_id 无法匹配上传文件时，应返回可读错误而不是抛出未处理异常。"""
        # 把上传目录指向明确不存在的路径，稳定触发找不到文件分支。
        mock_settings = MagicMock()
        mock_settings.uploads_path = Path("/nonexistent")
        mock_get_settings.return_value = mock_settings

        # LangChain tool 的标准调用方式是 invoke，并用字典匹配参数 schema。
        result = read_csv_file.invoke({"file_id": "nonexistent"})

        # 工具把找不到文件转换为文本结果；两种措辞都代表同一失败分支。
        assert "Error" in result or "未找到" in result

    @patch("app.agents.data_analyst.get_settings")
    def test_read_csv_file_success(self, mock_get_settings):
        """合法 CSV 应返回字段、行数和完整数据标识。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Arrange：用 pandas 构造包含文本、整数和浮点列的代表性 CSV。
            df = pd.DataFrame({
                "name": ["Alice", "Bob", "Charlie"],
                "age": [25, 30, 35],
                "score": [85.5, 90.0, 78.5]
            })
            csv_path = Path(tmpdir) / "test123_data.csv"
            df.to_csv(csv_path, index=False)

            mock_settings = MagicMock()
            mock_settings.uploads_path = Path(tmpdir)
            mock_get_settings.return_value = mock_settings

            # 文件名遵循“file_id_原文件名”规则，工具通过 test123 定位它。
            result = read_csv_file.invoke({"file_id": "test123"})

            # 同时检查预览可见、总行数以及“已展示全部行”的标识。
            assert "Alice" in result or "name" in result
            assert "3" in result  # 验证总行数，而不只验证预览中出现了姓名。
            assert "Complete data (all rows)" in result

    @patch("app.agents.data_analyst.get_settings")
    def test_large_file_preview_warns_against_full_table_inference(self, mock_get_settings):
        """大表截断预览必须明确声明不能用它推断全表统计。"""
        # 21 行超过完整展示阈值，工具应只显示前 5 行并附加限制说明。
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({"value": range(21)})
            (Path(tmpdir) / "test123_data.csv").write_text(
                df.to_csv(index=False), encoding="utf-8"
            )
            mock_get_settings.return_value.uploads_path = Path(tmpdir)

            result = read_csv_file.invoke({"file_id": "test123"})

            # 大表只展示前 5 行；提示文字要阻止上层 Agent 把预览当作全量数据。
            assert "Preview (first 5 rows only" in result
            assert "do not use this preview to calculate full-table" in result
            # 值 20 位于最后一行，不应泄漏到“前 5 行”预览片段中。
            assert "20" not in result.split("Preview", maxsplit=1)[1]


class TestAnalyzeData:
    """analyze_data 统计类型、列参数修复和 schema 测试。"""

    @patch("app.agents.data_analyst.get_settings")
    def test_analyze_describe(self, mock_get_settings):
        """describe 应对数值列输出计数、均值、标准差和分位数等描述统计。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                "value": [10, 20, 30, 40, 50]
            })
            csv_path = Path(tmpdir) / "test123_data.csv"
            df.to_csv(csv_path, index=False)

            mock_settings = MagicMock()
            mock_settings.uploads_path = Path(tmpdir)
            mock_get_settings.return_value = mock_settings

            result = analyze_data.invoke({
                "file_id": "test123",
                "analysis_type": "describe"
            })

            # 当前断言仅检查描述统计的标识文字，未逐一核对均值等具体数值。
            assert "Statistics" in result or "mean" in result.lower() or "统计" in result

    @patch("app.agents.data_analyst.get_settings")
    def test_analyze_summary(self, mock_get_settings):
        """summary 应概括数据表的总行数和总列数。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                "a": [1, 2, 3],
                "b": [4, 5, 6]
            })
            csv_path = Path(tmpdir) / "test123_data.csv"
            df.to_csv(csv_path, index=False)

            mock_settings = MagicMock()
            mock_settings.uploads_path = Path(tmpdir)
            mock_get_settings.return_value = mock_settings

            result = analyze_data.invoke({
                "file_id": "test123",
                "analysis_type": "summary"
            })

            # 输入是 3 行 2 列；这里检查输出包含这些数值，未限定它们所在字段。
            assert "3" in result  # 行数。
            assert "2" in result or "column" in result.lower()

    @patch("app.agents.data_analyst.get_settings")
    def test_analyze_correlation(self, mock_get_settings):
        """correlation 应对数值列计算相关系数矩阵。"""
        # y=2x 构造完全线性相关的数据，便于稳定验证相关分析分支。
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                "x": [1, 2, 3, 4, 5],
                "y": [2, 4, 6, 8, 10]
            })
            csv_path = Path(tmpdir) / "test123_data.csv"
            df.to_csv(csv_path, index=False)

            mock_settings = MagicMock()
            mock_settings.uploads_path = Path(tmpdir)
            mock_get_settings.return_value = mock_settings

            result = analyze_data.invoke({
                "file_id": "test123",
                "analysis_type": "correlation"
            })

            # y=2x 让相关系数可预测，但当前断言只确认进入了相关分析分支。
            assert "Correlation" in result or "相关" in result

    def test_analyze_unknown_type(self):
        """未知 analysis_type 应返回“不支持”提示，不能静默执行错误分析。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({"a": [1, 2, 3]})
            csv_path = Path(tmpdir) / "test123_data.csv"
            df.to_csv(csv_path, index=False)

            with patch("app.agents.data_analyst.get_settings") as mock_settings:
                # 这里使用 with 形式的 patch；离开代码块后会自动恢复原工厂函数。
                mock_settings.return_value.uploads_path = Path(tmpdir)

                result = analyze_data.invoke({
                    "file_id": "test123",
                    "analysis_type": "unknown_type"
                })

                assert "Unknown" in result or "不支持" in result

    @patch("app.agents.data_analyst.get_settings")
    def test_value_counts_uses_all_rows(self, mock_get_settings):
        """分类频数必须基于全部 8 行计算，不能只统计预览行。"""
        # Beijing、Shanghai、Guangzhou 都出现两次，且部分重复值位于第 5 行之后。
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                "name": [
                    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"
                ],
                "city": [
                    "Beijing", "Shanghai", "Guangzhou", "Shenzhen",
                    "Hangzhou", "Beijing", "Shanghai", "Guangzhou",
                ],
            })
            (Path(tmpdir) / "test123_data.csv").write_text(
                df.to_csv(index=False), encoding="utf-8"
            )
            mock_get_settings.return_value.uploads_path = Path(tmpdir)

            result = analyze_data.invoke({
                "file_id": "test123",
                "analysis_type": "value_counts",
                "columns": ["city"],
            })

            # 第 6～8 行分别补上三个城市的第二次出现；若只统计前 5 行，
            # 最高频数会误变成 1，且并列最高类别的文字也会不同。
            # 同时断言数据覆盖声明、最高频数和并列最高频类别。
            assert "all 8 rows" in result
            assert "Beijing" in result
            assert "2" in result
            assert "Highest frequency: 2" in result
            assert "Beijing, Shanghai, Guangzhou" in result

    @pytest.mark.parametrize("columns", ['["salary"]', "salary"])
    @patch("app.agents.data_analyst.get_settings")
    def test_analyze_normalizes_string_columns(self, mock_get_settings, columns):
        """JSON 数组字符串和单列字符串都应修复成列数组后再分析。"""
        # 参数化分别模拟模型输出 '["salary"]' 与 'salary' 两种常见偏差。
        # pytest 会为每个 columns 值生成一次独立测试；mock_get_settings 由 patch 注入。
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                "name": ["Alice", "Bob"],
                "salary": [15000, 25000],
            })
            csv_path = Path(tmpdir) / "test123_data.csv"
            df.to_csv(csv_path, index=False)
            mock_get_settings.return_value.uploads_path = Path(tmpdir)

            result = analyze_data.invoke({
                "file_id": "test123",
                "analysis_type": "describe",
                "columns": columns,
            })

            # 只分析 salary，未选择的 name 不应出现在统计结果中。
            assert "salary" in result
            assert "name" not in result

    def test_analyze_rejects_malformed_json_array_string(self):
        """无法解析的残缺数组字符串必须保留为 ValidationError。"""
        # 兼容修复不能无限宽松，否则模型的坏参数可能被错误解释并执行。
        # 此处在工具参数校验阶段就应报错，因此无需准备 CSV 或替换 get_settings。
        with pytest.raises(ValidationError):
            analyze_data.invoke({
                "file_id": "test123",
                "analysis_type": "describe",
                "columns": '["salary"',
            })

    def test_analyze_columns_schema_remains_array(self):
        """运行时虽兼容字符串，但向模型发布的 columns schema 仍必须是数组。"""
        # schema 是给模型生成工具参数的正式契约；不能因容错逻辑鼓励模型输出字符串。
        schema = analyze_data.args_schema.model_json_schema()
        columns_schema = schema["properties"]["columns"]
        variants = columns_schema.get("anyOf", [columns_schema])

        # 可选字段在 Pydantic schema 中可能是 anyOf(数组、null)，
        # 所以逐个查看候选类型：允许数组，不向模型声明字符串类型。
        assert any(variant.get("type") == "array" for variant in variants)
        assert not any(variant.get("type") == "string" for variant in variants)


class TestCreateChart:
    """create_chart 折线图和柱状图生成测试。"""

    @patch("app.agents.data_analyst.get_settings")
    def test_create_line_chart(self, mock_get_settings):
        """指定 x/y 列创建折线图时，应返回成功信息。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                "x": [1, 2, 3, 4, 5],
                "y": [10, 20, 15, 25, 30]
            })
            csv_path = Path(tmpdir) / "test123_data.csv"
            df.to_csv(csv_path, index=False)

            mock_settings = MagicMock()
            mock_settings.uploads_path = Path(tmpdir)
            mock_get_settings.return_value = mock_settings

            # invoke 参数与工具 schema 一致，图表文件也会写入临时上传目录下。
            result = create_chart.invoke({
                "file_id": "test123",
                "chart_type": "line",
                "x_column": "x",
                "y_column": "y"
            })

            # 当前测试只检查成功提示，没有打开或检查生成的 PNG 文件。
            assert "success" in result.lower() or "created" in result.lower() or "成功" in result

    @patch("app.agents.data_analyst.get_settings")
    def test_create_bar_chart(self, mock_get_settings):
        """柱状图应使用全部行，并报告绘图数据的最小值和最大值。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                "category": ["A", "B", "C"],
                "value": [10, 20, 15]
            })
            csv_path = Path(tmpdir) / "test123_data.csv"
            df.to_csv(csv_path, index=False)

            mock_settings = MagicMock()
            mock_settings.uploads_path = Path(tmpdir)
            mock_get_settings.return_value = mock_settings

            result = create_chart.invoke({
                "file_id": "test123",
                "chart_type": "bar",
                "x_column": "category",
                "y_column": "value"
            })

            # 除了图表提示，还检查全量数据覆盖与极值类别，避免只用部分行绘图。
            assert "chart" in result.lower() or "图" in result
            # 覆盖说明用于防止上层 Agent 把抽样图表误称为全量结论。
            assert "Data coverage: all 3 rows" in result
            assert "Minimum plotted value: A = 10" in result
            assert "Maximum plotted value: B = 20" in result


class TestCreateDataAnalystTools:
    """数据分析工具工厂的注册清单测试。"""

    def test_creates_all_tools(self):
        """默认工厂应注册读取、分析和绘图三个核心工具。"""
        tools = create_data_analyst_tools()

        assert len(tools) == 3
        # 按工具 name 而不是函数对象身份检查，符合 Agent 实际看到的工具 schema。
        tool_names = [t.name for t in tools]
        assert "read_csv_file" in tool_names
        assert "analyze_data" in tool_names
        assert "create_chart" in tool_names
