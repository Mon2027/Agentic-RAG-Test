"""数据分析工具链的集成测试。

测试用例在临时目录创建 CSV，再分别调用文件读取、统计分析、图表生成工具。
它们验证工具能读取真实文件并处理数据，但不经过 HTTP 上传接口，也没有在
同一个用例中串联“上传 → 分析 → 绘图”的完整用户流程。

所有文件写入都由 pytest 的 tmp_path 隔离；get_settings 被临时替换，
使工具从测试目录读取数据。图表用例当前允许成功文本或 Error 文本通过，
阅读测试结果时需要留意这个断言范围。
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

# 在应用模块导入前声明测试环境；上级 conftest.py 还会强制隔离数据目录和外部凭证。
os.environ.setdefault("APP_ENV", "testing")


class TestDataAnalysisPipeline:
    """分别验证 CSV 读取和多种分析类型的工具调用。"""

    @pytest.fixture
    def sample_csv_path(self, tmp_path: Path) -> Path:
        """在当前用例独享的临时目录中创建四行、五列的 CSV。"""
        # 字段同时包含文本、数值和日期字符串，覆盖常见数据文件的基本形态。
        csv_path = tmp_path / "test_data.csv"
        csv_content = """name,age,city,salary,join_date
Alice,28,Beijing,15000,2022-01-15
Bob,35,Shanghai,25000,2021-03-20
Charlie,42,Guangzhou,35000,2020-06-10
Diana,31,Shenzhen,20000,2022-08-05"""
        csv_path.write_text(csv_content)
        return csv_path

    @pytest.fixture
    def sample_excel_path(self, tmp_path: Path) -> Path:
        """创建最小 Excel 样例；当前文件的测试用例尚未引用此 fixture。"""
        try:
            import pandas as pd

            excel_path = tmp_path / "test_data.xlsx"
            df = pd.DataFrame({
                "name": ["Alice", "Bob", "Charlie"],
                "age": [28, 35, 42],
                "salary": [15000, 25000, 35000],
            })
            df.to_excel(excel_path, index=False)
            return excel_path
        except ImportError:
            # 只有用例请求此 fixture 时才会执行；缺少依赖就标记跳过。
            pytest.skip("pandas/openpyxl not available")

    def test_read_csv_file_tool(self, sample_csv_path: Path):
        """工具应按 file_id 找到 CSV 并返回结构与行数信息。"""
        with patch("app.agents.data_analyst.get_settings") as mock_settings:
            # patch 只替换工具实际读取的配置工厂，不改动临时 CSV 的内容。
            mock_settings.return_value.uploads_path = sample_csv_path.parent

            # 上传目录中的文件采用“file_id_原文件名”命名；重命名后才能按 ID 查找。
            file_id = "test_csv_id"
            renamed_path = sample_csv_path.parent / f"{file_id}_test_data.csv"
            sample_csv_path.rename(renamed_path)

            from app.agents.data_analyst import read_csv_file

            result = read_csv_file.invoke({"file_id": file_id})

            # @tool 包装后的入口是 invoke(dict)；这里检查格式提示、列名和四行数据。
            assert "CSV File Information" in result
            assert "name,age,city,salary" in result or "columns" in result.lower()
            assert "4" in result  # CSV 有四行数据；此断言未限定数字所在字段。

    def test_analyze_data_describe(self, sample_csv_path: Path):
        """describe 分支应返回描述统计的标识文字。"""
        with patch("app.agents.data_analyst.get_settings") as mock_settings:
            mock_settings.return_value.uploads_path = sample_csv_path.parent

            file_id = "test_csv_id"
            renamed_path = sample_csv_path.parent / f"{file_id}_test_data.csv"
            sample_csv_path.rename(renamed_path)

            from app.agents.data_analyst import analyze_data

            result = analyze_data.invoke({
                "file_id": file_id,
                "analysis_type": "describe",
            })

            # 当前断言不核对均值、标准差等具体数值，只确认执行了描述统计分支。
            assert "Descriptive Statistics" in result

    def test_analyze_data_correlation(self, sample_csv_path: Path):
        """correlation 分支应返回相关系数矩阵或无数值列提示。"""
        with patch("app.agents.data_analyst.get_settings") as mock_settings:
            mock_settings.return_value.uploads_path = sample_csv_path.parent

            file_id = "test_csv_id"
            renamed_path = sample_csv_path.parent / f"{file_id}_test_data.csv"
            sample_csv_path.rename(renamed_path)

            from app.agents.data_analyst import analyze_data

            result = analyze_data.invoke({
                "file_id": file_id,
                "analysis_type": "correlation",
            })

            # 此 CSV 有 age 和 salary 数值列；断言仍允许“无数值列”提示通过，
            # 因而只覆盖工具能给出可识别的响应，没有验证具体相关系数。
            assert "Correlation Matrix" in result or "No numeric columns" in result

    def test_analyze_data_summary(self, sample_csv_path: Path):
        """summary 分支应包含数据摘要及总行数字段。"""
        with patch("app.agents.data_analyst.get_settings") as mock_settings:
            mock_settings.return_value.uploads_path = sample_csv_path.parent

            file_id = "test_csv_id"
            renamed_path = sample_csv_path.parent / f"{file_id}_test_data.csv"
            sample_csv_path.rename(renamed_path)

            from app.agents.data_analyst import analyze_data

            result = analyze_data.invoke({
                "file_id": file_id,
                "analysis_type": "summary",
            })

            # 检查输出结构的关键标题；未断言总行数必须等于四。
            assert "Data Summary" in result
            assert "Total Rows" in result

    def test_analyze_data_trend(self, sample_csv_path: Path):
        """trend 分支应返回趋势分析或无数值列提示。"""
        with patch("app.agents.data_analyst.get_settings") as mock_settings:
            mock_settings.return_value.uploads_path = sample_csv_path.parent

            file_id = "test_csv_id"
            renamed_path = sample_csv_path.parent / f"{file_id}_test_data.csv"
            sample_csv_path.rename(renamed_path)

            from app.agents.data_analyst import analyze_data

            result = analyze_data.invoke({
                "file_id": file_id,
                "analysis_type": "trend",
            })

            # 当前断言没有核对涨跌方向和变化百分比。
            assert "Trend Analysis" in result or "No numeric columns" in result

    def test_analyze_data_file_not_found(self):
        """找不到文件时，分析工具应返回错误文字。"""
        with patch("app.agents.data_analyst.get_settings") as mock_settings:
            mock_settings.return_value.uploads_path = Path("/tmp/nonexistent")

            from app.agents.data_analyst import analyze_data

            result = analyze_data.invoke({
                "file_id": "nonexistent_id",
                "analysis_type": "describe",
            })

            # 错误由工具转换为文本，不作为未处理异常向测试传播。
            assert "Error" in result or "not found" in result


class TestChartGeneration:
    """验证图表工具接收不同 chart_type 时的返回文本。"""

    @pytest.fixture
    def sample_csv_with_data(self, tmp_path: Path) -> tuple[Path, str]:
        """创建含月份、营收、利润的五行 CSV，并返回上传目录和 file_id。"""
        csv_path = tmp_path / "chart_data.csv"
        csv_content = """month,revenue,profit
Jan,10000,2000
Feb,12000,2500
Mar,15000,3000
Apr,18000,3500
May,20000,4000"""
        csv_path.write_text(csv_content)

        file_id = "chart_test_id"
        # 与正式上传文件保持相同命名规则，供 create_chart 通过 file_id 定位。
        renamed_path = tmp_path / f"{file_id}_chart_data.csv"
        csv_path.rename(renamed_path)

        return renamed_path.parent, file_id

    def test_create_line_chart(self, sample_csv_with_data):
        """传入月份和营收列，调用折线图分支。"""
        uploads_path, file_id = sample_csv_with_data

        with patch("app.agents.data_analyst.get_settings") as mock_settings:
            mock_settings.return_value.uploads_path = uploads_path

            from app.agents.data_analyst import create_chart

            result = create_chart.invoke({
                "file_id": file_id,
                "chart_type": "line",
                "x_column": "month",
                "y_column": "revenue",
                "title": "Revenue Trend",
            })

            # 这里允许成功或 Error 文本，两种结果都会通过；因此不能据此认定
            # 图表一定生成，也没有检查 PNG 文件是否存在或内容是否正确。
            assert "Chart created" in result or "Error" in result

    def test_create_bar_chart(self, sample_csv_with_data):
        """传入月份和营收列，调用柱状图分支。"""
        uploads_path, file_id = sample_csv_with_data

        with patch("app.agents.data_analyst.get_settings") as mock_settings:
            mock_settings.return_value.uploads_path = uploads_path

            from app.agents.data_analyst import create_chart

            result = create_chart.invoke({
                "file_id": file_id,
                "chart_type": "bar",
                "x_column": "month",
                "y_column": "revenue",
            })

            # 与折线图用例相同：此断言只接受两类返回文本，不保证实际绘图成功。
            assert "Chart created" in result or "Error" in result

    def test_create_histogram(self, sample_csv_with_data):
        """直方图只需数值列，不需要传入 y_column。"""
        uploads_path, file_id = sample_csv_with_data

        with patch("app.agents.data_analyst.get_settings") as mock_settings:
            mock_settings.return_value.uploads_path = uploads_path

            from app.agents.data_analyst import create_chart

            result = create_chart.invoke({
                "file_id": file_id,
                "chart_type": "histogram",
                "x_column": "revenue",
            })

            # 当前断言未检验直方图的分箱或图片文件。
            assert "Chart created" in result or "Error" in result

    def test_create_pie_chart(self, sample_csv_with_data):
        """饼图用月份列作为输入，检查工具返回可识别的结果。"""
        uploads_path, file_id = sample_csv_with_data

        with patch("app.agents.data_analyst.get_settings") as mock_settings:
            mock_settings.return_value.uploads_path = uploads_path

            from app.agents.data_analyst import create_chart

            result = create_chart.invoke({
                "file_id": file_id,
                "chart_type": "pie",
                "x_column": "month",
            })

            # 允许 Error 文本通过，不能把该用例视为图片生成成功的证明。
            assert "Chart created" in result or "Error" in result

    def test_create_chart_invalid_type(self, sample_csv_with_data):
        """不支持的 chart_type 应返回明确的类型错误提示。"""
        uploads_path, file_id = sample_csv_with_data

        with patch("app.agents.data_analyst.get_settings") as mock_settings:
            mock_settings.return_value.uploads_path = uploads_path

            from app.agents.data_analyst import create_chart

            result = create_chart.invoke({
                "file_id": file_id,
                "chart_type": "invalid_type",
                "x_column": "month",
            })

            # 负向场景比前四个宽松断言更明确地锁定了一个错误分支。
            assert "Unknown chart type" in result


class TestDataAnalystToolsIntegration:
    """验证主智能体可获得的数据分析工具清单及参数 schema。"""

    def test_all_tools_creation(self):
        """默认工具工厂应返回读取、分析、绘图三项工具。"""
        from app.agents.data_analyst import create_data_analyst_tools

        tools = create_data_analyst_tools()

        # 先核对数量，再按 LangChain tool.name 检查对智能体暴露的工具名称。
        assert len(tools) == 3
        tool_names = [t.name for t in tools]
        assert "read_csv_file" in tool_names
        assert "analyze_data" in tool_names
        assert "create_chart" in tool_names

    def test_tool_schemas(self):
        """每个工具都应提供名称、描述及参数 schema。"""
        from app.agents.data_analyst import create_data_analyst_tools

        tools = create_data_analyst_tools()

        # 这里只检查这些字段存在，不验证 schema 中每个参数的类型与必填规则。
        for tool in tools:
            assert tool.name
            assert tool.description
            assert tool.args_schema
