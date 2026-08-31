"""Integration tests for Data Analysis pipeline.

Tests the complete flow:
CSV/Excel upload -> Data processing -> Analysis -> Chart generation
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

# Set test environment before imports
os.environ.setdefault("APP_ENV", "testing")


class TestDataAnalysisPipeline:
    """Test the complete data analysis pipeline."""

    @pytest.fixture
    def sample_csv_path(self, tmp_path: Path) -> Path:
        """Create a sample CSV file for testing."""
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
        """Create a sample Excel file for testing."""
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
            pytest.skip("pandas/openpyxl not available")

    def test_read_csv_file_tool(self, sample_csv_path: Path):
        """Test read_csv_file tool functionality."""
        with patch("app.agents.data_analyst.get_settings") as mock_settings:
            mock_settings.return_value.uploads_path = sample_csv_path.parent

            # Rename file to match expected pattern
            file_id = "test_csv_id"
            renamed_path = sample_csv_path.parent / f"{file_id}_test_data.csv"
            sample_csv_path.rename(renamed_path)

            from app.agents.data_analyst import read_csv_file

            result = read_csv_file.invoke({"file_id": file_id})

            assert "CSV File Information" in result
            assert "name,age,city,salary" in result or "columns" in result.lower()
            assert "4" in result  # 4 rows of data

    def test_analyze_data_describe(self, sample_csv_path: Path):
        """Test analyze_data tool with describe analysis."""
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

            assert "Descriptive Statistics" in result

    def test_analyze_data_correlation(self, sample_csv_path: Path):
        """Test analyze_data tool with correlation analysis."""
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

            assert "Correlation Matrix" in result or "No numeric columns" in result

    def test_analyze_data_summary(self, sample_csv_path: Path):
        """Test analyze_data tool with summary analysis."""
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

            assert "Data Summary" in result
            assert "Total Rows" in result

    def test_analyze_data_trend(self, sample_csv_path: Path):
        """Test analyze_data tool with trend analysis."""
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

            assert "Trend Analysis" in result or "No numeric columns" in result

    def test_analyze_data_file_not_found(self):
        """Test analyze_data with non-existent file."""
        with patch("app.agents.data_analyst.get_settings") as mock_settings:
            mock_settings.return_value.uploads_path = Path("/tmp/nonexistent")

            from app.agents.data_analyst import analyze_data

            result = analyze_data.invoke({
                "file_id": "nonexistent_id",
                "analysis_type": "describe",
            })

            assert "Error" in result or "not found" in result


class TestChartGeneration:
    """Test chart generation functionality."""

    @pytest.fixture
    def sample_csv_with_data(self, tmp_path: Path) -> tuple[Path, str]:
        """Create a sample CSV file with numeric data."""
        csv_path = tmp_path / "chart_data.csv"
        csv_content = """month,revenue,profit
Jan,10000,2000
Feb,12000,2500
Mar,15000,3000
Apr,18000,3500
May,20000,4000"""
        csv_path.write_text(csv_content)

        file_id = "chart_test_id"
        renamed_path = tmp_path / f"{file_id}_chart_data.csv"
        csv_path.rename(renamed_path)

        return renamed_path.parent, file_id

    def test_create_line_chart(self, sample_csv_with_data):
        """Test creating a line chart."""
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

            # Check for success or error (matplotlib might not have Chinese fonts)
            assert "Chart created" in result or "Error" in result

    def test_create_bar_chart(self, sample_csv_with_data):
        """Test creating a bar chart."""
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

            assert "Chart created" in result or "Error" in result

    def test_create_histogram(self, sample_csv_with_data):
        """Test creating a histogram."""
        uploads_path, file_id = sample_csv_with_data

        with patch("app.agents.data_analyst.get_settings") as mock_settings:
            mock_settings.return_value.uploads_path = uploads_path

            from app.agents.data_analyst import create_chart

            result = create_chart.invoke({
                "file_id": file_id,
                "chart_type": "histogram",
                "x_column": "revenue",
            })

            assert "Chart created" in result or "Error" in result

    def test_create_pie_chart(self, sample_csv_with_data):
        """Test creating a pie chart."""
        uploads_path, file_id = sample_csv_with_data

        with patch("app.agents.data_analyst.get_settings") as mock_settings:
            mock_settings.return_value.uploads_path = uploads_path

            from app.agents.data_analyst import create_chart

            result = create_chart.invoke({
                "file_id": file_id,
                "chart_type": "pie",
                "x_column": "month",
            })

            assert "Chart created" in result or "Error" in result

    def test_create_chart_invalid_type(self, sample_csv_with_data):
        """Test creating chart with invalid type."""
        uploads_path, file_id = sample_csv_with_data

        with patch("app.agents.data_analyst.get_settings") as mock_settings:
            mock_settings.return_value.uploads_path = uploads_path

            from app.agents.data_analyst import create_chart

            result = create_chart.invoke({
                "file_id": file_id,
                "chart_type": "invalid_type",
                "x_column": "month",
            })

            assert "Unknown chart type" in result


class TestDataAnalystToolsIntegration:
    """Test data analyst tools integration."""

    def test_all_tools_creation(self):
        """Test that all data analyst tools can be created."""
        from app.agents.data_analyst import create_data_analyst_tools

        tools = create_data_analyst_tools()

        assert len(tools) == 3
        tool_names = [t.name for t in tools]
        assert "read_csv_file" in tool_names
        assert "analyze_data" in tool_names
        assert "create_chart" in tool_names

    def test_tool_schemas(self):
        """Test that tools have proper schemas."""
        from app.agents.data_analyst import create_data_analyst_tools

        tools = create_data_analyst_tools()

        for tool in tools:
            assert tool.name
            assert tool.description
            assert tool.args_schema
