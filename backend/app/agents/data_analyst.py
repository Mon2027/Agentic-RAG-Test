"""Data Analyst Sub-Agent tools."""

import json
from pathlib import Path
from typing import Annotated

import pandas as pd
from langchain_core.tools import tool
from pydantic import BeforeValidator

from app.core import get_settings

SUPPORTED_DATA_EXTENSIONS = {".csv", ".xlsx", ".xls"}
COMPLETE_DATA_ROW_LIMIT = 20


def _normalize_analysis_columns(value: object) -> object:
    """Normalize common model encodings without weakening the array schema.

    Some Anthropic-compatible model endpoints serialize a JSON array tool
    argument as a string (for example, ``'["salary"]'``).  Pydantic runs this
    function before strict list validation so valid array strings and plain
    single-column strings can be repaired while malformed JSON is still
    rejected.
    """
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if not stripped:
        return None

    if stripped.startswith("["):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value

    return [stripped]


@tool
def list_data_files() -> str:
    """List all uploaded data files (CSV and Excel) available for analysis.

    Returns a list of files with their IDs, names, and types.
    Use this tool first to see what data files are available before analysis.
    """
    settings = get_settings()

    try:
        files = []
        for f in settings.uploads_path.iterdir():
            if f.is_file() and f.suffix.lower() in SUPPORTED_DATA_EXTENSIONS:
                # Extract file_id and original name
                parts = f.name.split("_", 1)
                file_id = parts[0]
                file_name = parts[1] if len(parts) > 1 else f.name

                files.append({
                    "file_id": file_id,
                    "file_name": file_name,
                    "file_type": f.suffix[1:].upper(),
                    "file_size": f.stat().st_size,
                })

        if not files:
            return """当前没有已上传的数据文件。

上传数据文件:
1. 使用界面上传 CSV 或 Excel 文件
2. 文件将自动存储并可用于分析"""

        # Format output
        lines = [f"已上传数据文件列表 (共 {len(files)} 个):\n"]
        for f in files:
            size_kb = f["file_size"] / 1024
            lines.append(f"- ID: {f['file_id']}")
            lines.append(f"  名称: {f['file_name']}")
            lines.append(f"  类型: {f['file_type']}")
            lines.append(f"  大小: {size_kb:.1f} KB\n")

        lines.append("使用文件 ID 配合 read_csv_file/read_data_file 工具读取文件内容。")

        return "\n".join(lines)

    except Exception as e:
        return f"获取数据文件列表出错: {str(e)}"


def _find_data_file(identifier: str) -> Path | str:
    """Find an uploaded data file by file ID or original filename."""
    settings = get_settings()
    normalized = identifier.strip().lower()

    try:
        files = [
            f for f in settings.uploads_path.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_DATA_EXTENSIONS
        ]
    except Exception as e:
        return f"Error searching for file: {str(e)}"

    exact_id_matches = [f for f in files if f.name.startswith(f"{identifier}_")]
    if exact_id_matches:
        return exact_id_matches[0]

    exact_name_matches = []
    fuzzy_name_matches = []
    for file_path in files:
        parts = file_path.name.split("_", 1)
        original_name = parts[1] if len(parts) > 1 else file_path.name
        original_lower = original_name.lower()
        full_lower = file_path.name.lower()

        if normalized in {original_lower, full_lower}:
            exact_name_matches.append(file_path)
        elif normalized and normalized in original_lower:
            fuzzy_name_matches.append(file_path)

    if len(exact_name_matches) == 1:
        return exact_name_matches[0]
    if len(exact_name_matches) > 1:
        return "Error: Multiple files matched this filename. Please use the file ID."
    if len(fuzzy_name_matches) == 1:
        return fuzzy_name_matches[0]
    if len(fuzzy_name_matches) > 1:
        return "Error: Multiple files partially matched this filename. Please use the file ID."

    return f"Error: File '{identifier}' not found. Please use list_data_files to check available files and IDs."


def _resolve_column(df: pd.DataFrame, column: str | None) -> str | None:
    """Resolve a user/model-provided column name against a dataframe."""
    if not column:
        return None

    normalized = column.strip().lower()
    for existing_column in df.columns:
        if str(existing_column).strip().lower() == normalized:
            return str(existing_column)

    for existing_column in df.columns:
        if normalized and normalized in str(existing_column).strip().lower():
            return str(existing_column)

    return None


def _default_x_column(df: pd.DataFrame) -> str:
    """Choose a readable default x-axis column."""
    preferred_terms = ("date", "time", "quarter", "year", "月份", "季度", "年份", "日期", "时间")
    for column in df.columns:
        column_text = str(column).lower()
        if any(term in column_text for term in preferred_terms):
            return str(column)
    return str(df.columns[0])


def _default_y_column(df: pd.DataFrame, x_column: str | None = None) -> str | None:
    """Choose the first numeric column that is not the x-axis."""
    numeric_columns = [
        str(column)
        for column in df.select_dtypes(include=["number"]).columns
        if str(column) != x_column
    ]
    return numeric_columns[0] if numeric_columns else None


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series | None:
    """Return a numeric series for plotting, coercing where possible."""
    series = pd.to_numeric(df[column], errors="coerce")
    if series.notna().sum() == 0:
        return None
    return series


def _read_data_file_impl(file_id: str) -> str:
    """Read a data file by file ID and return structure and preview."""
    file_path = _find_data_file(file_id)
    if isinstance(file_path, str):
        return file_path

    file_ext = file_path.suffix.lower()

    try:
        # Read file based on extension
        if file_ext == ".csv":
            df = pd.read_csv(file_path)
        elif file_ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
        else:
            return f"Error: Unsupported file type '{file_ext}'. Supported: .csv, .xlsx, .xls"

        complete_data = len(df) <= COMPLETE_DATA_ROW_LIMIT
        displayed_data = df.to_string() if complete_data else df.head(5).to_string()
        info = {
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "row_count": len(df),
            "column_count": len(df.columns),
            "displayed_data": displayed_data,
            "missing_values": df.isnull().sum().to_dict(),
        }

        return f"""Data File Information ({file_ext.upper()}):
Columns: {info['columns']}
Total Rows: {info['row_count']}
Total Columns: {info['column_count']}

Data Types:
{info['dtypes']}

Missing Values:
{info['missing_values']}

{"Complete data (all rows):" if complete_data else "Preview (first 5 rows only; do not use this preview to calculate full-table counts, totals, minima, or maxima):"}
{info['displayed_data']}"""

    except Exception as e:
        return f"Error reading data file: {str(e)}"


@tool
def read_csv_file(
    file_id: Annotated[str, "The file ID or original filename of the uploaded data file (CSV or Excel)"],
) -> str:
    """Read a data file (CSV or Excel) and return its basic information and preview.

    Kept for compatibility with older prompts/tests that used the CSV-specific name.
    Supports CSV (.csv) and Excel (.xlsx, .xls) files.
    """
    return _read_data_file_impl(file_id).replace("Data File Information", "CSV File Information", 1)


@tool
def read_data_file(
    file_id: Annotated[str, "The file ID or original filename of the uploaded data file (CSV or Excel)"],
) -> str:
    """Read a data file (CSV or Excel) and return its basic information and preview.

    Use this tool to understand the structure of an uploaded data file.
    Supports CSV (.csv) and Excel (.xlsx, .xls) files.
    Returns column names, data types, row count, and a preview of the data.
    """
    return _read_data_file_impl(file_id)


@tool
def analyze_data(
    file_id: Annotated[str, "The file ID or original filename of the uploaded data file"],
    analysis_type: Annotated[
        str, "Type of analysis: 'describe', 'correlation', 'trend', 'summary', or 'value_counts'"
    ],
    columns: Annotated[
        list[str] | None,
        BeforeValidator(_normalize_analysis_columns),
        "Specific columns to analyze (optional; pass a JSON array of column names)",
    ] = None,
) -> str:
    """Perform statistical analysis on a data file.

    Supported analysis types:
    - 'describe': Basic descriptive statistics (mean, std, min, max, etc.)
    - 'correlation': Correlation matrix between numeric columns
    - 'trend': Trend analysis for time series data
    - 'summary': Overall summary of the data
    - 'value_counts': Full-table frequency counts for one or more columns
    """
    file_path = _find_data_file(file_id)
    if isinstance(file_path, str):
        return file_path

    try:
        # Read file based on extension
        if file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        if columns:
            resolved_columns = [_resolve_column(df, column) for column in columns]
            missing_columns = [
                column for column, resolved in zip(columns, resolved_columns, strict=True)
                if resolved is None
            ]
            if missing_columns:
                return f"Error: Columns not found: {missing_columns}. Available: {list(df.columns)}"
            df = df[[column for column in resolved_columns if column is not None]]

        if analysis_type == "describe":
            result = df.describe(include="all").to_string()
            return f"Descriptive Statistics:\n\n{result}"

        elif analysis_type == "correlation":
            numeric_df = df.select_dtypes(include=["number"])
            if numeric_df.empty:
                return "No numeric columns found for correlation analysis"
            corr = numeric_df.corr()
            return f"Correlation Matrix:\n\n{corr.to_string()}"

        elif analysis_type == "trend":
            # Basic trend analysis
            numeric_cols = df.select_dtypes(include=["number"]).columns
            if len(numeric_cols) == 0:
                return "No numeric columns found for trend analysis"

            trends = []
            for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
                values = df[col].dropna()
                if len(values) > 1:
                    trend = "increasing" if values.iloc[-1] > values.iloc[0] else "decreasing"
                    change_pct = ((values.iloc[-1] - values.iloc[0]) / values.iloc[0] * 100) if values.iloc[0] != 0 else 0
                    trends.append(f"  - {col}: {trend} ({change_pct:.2f}% change)")

            return "Trend Analysis:\n" + "\n".join(trends)

        elif analysis_type == "summary":
            return f"""Data Summary:
- Total Rows: {len(df)}
- Total Columns: {len(df.columns)}
- Column Names: {list(df.columns)}
- Numeric Columns: {list(df.select_dtypes(include=['number']).columns)}
- Missing Values Total: {df.isnull().sum().sum()}
- Duplicate Rows: {df.duplicated().sum()}"""

        elif analysis_type == "value_counts":
            if not columns:
                return "Error: 'value_counts' requires at least one column"
            sections = []
            for column in df.columns:
                counts = df[column].value_counts(dropna=False)
                highest_frequency = int(counts.max()) if not counts.empty else 0
                highest_values = [
                    str(value)
                    for value, count in counts.items()
                    if int(count) == highest_frequency
                ]
                sections.append(
                    f"Value Counts ({column}, all {len(df)} rows):\n{counts.to_string()}\n"
                    f"Highest frequency: {highest_frequency}. Values tied for highest "
                    f"frequency: {', '.join(highest_values)}."
                )
            return "\n\n".join(sections)

        else:
            return (
                f"Unknown analysis type: {analysis_type}. Supported: "
                "describe, correlation, trend, summary, value_counts"
            )

    except Exception as e:
        return f"Error during analysis: {str(e)}"


@tool
def create_chart(
    file_id: Annotated[str, "The file ID or original filename of the uploaded data file"],
    chart_type: Annotated[
        str, "Type of chart: 'line', 'bar', 'scatter', 'pie', 'histogram'"
    ],
    x_column: Annotated[str, "Column name for X-axis"],
    y_column: Annotated[str | None, "Column name for Y-axis (optional for histogram/pie)"] = None,
    title: Annotated[str | None, "Chart title"] = None,
) -> str:
    """Create a chart from the data file and save it.

    Returns the path to the generated chart image.
    Chart types supported: line, bar, scatter, pie, histogram
    """
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")  # Use non-interactive backend

    settings = get_settings()
    file_path = _find_data_file(file_id)
    if isinstance(file_path, str):
        return file_path

    try:
        # Read file
        if file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        if df.empty:
            return "Error: Data file is empty"

        x_column = _resolve_column(df, x_column) or _default_x_column(df)
        y_column = _resolve_column(df, y_column)

        if chart_type in {"line", "bar", "scatter"}:
            if not y_column:
                y_column = _default_y_column(df, x_column)
            if not y_column:
                return "Error: No numeric column found for Y-axis"
            y_values = _numeric_series(df, y_column)
            if y_values is None:
                fallback_y_column = _default_y_column(df, x_column)
                if not fallback_y_column or fallback_y_column == y_column:
                    return f"Error: Y-axis column '{y_column}' is not numeric"
                y_column = fallback_y_column
                y_values = _numeric_series(df, y_column)
            if y_values is None:
                return f"Error: Y-axis column '{y_column}' is not numeric"

        # Create charts directory
        charts_dir = settings.uploads_path / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique filename
        import uuid

        chart_id = str(uuid.uuid4())[:8]
        chart_path = charts_dir / f"chart_{chart_id}.png"

        # Set Chinese font support
        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, ax = plt.subplots(figsize=(10, 6))

        if chart_type == "line":
            ax.plot(df[x_column], y_values, marker="o")
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)

        elif chart_type == "bar":
            ax.bar(df[x_column], y_values)
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)

        elif chart_type == "scatter":
            ax.scatter(df[x_column], y_values)
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)

        elif chart_type == "pie":
            x_column = _resolve_column(df, x_column) or _default_x_column(df)
            counts = df[x_column].value_counts()
            ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%")

        elif chart_type == "histogram":
            x_column = _resolve_column(df, x_column) or _default_y_column(df)
            if not x_column:
                return "Error: No numeric column found for histogram"
            x_values = _numeric_series(df, x_column)
            if x_values is None:
                return f"Error: Histogram column '{x_column}' is not numeric"
            ax.hist(x_values, bins=20)
            ax.set_xlabel(x_column)
            ax.set_ylabel("Frequency")

        else:
            return f"Unknown chart type: {chart_type}"

        if title:
            ax.set_title(title)

        if chart_type in {"line", "bar", "scatter"}:
            ax.grid(True, alpha=0.25)
            ax.tick_params(axis="x", rotation=30)

        plt.tight_layout()
        plt.savefig(chart_path, dpi=150)
        plt.close()

        evidence_lines = [f"Data coverage: all {len(df)} rows."]
        if chart_type in {"line", "bar", "scatter"} and y_values is not None:
            valid_values = y_values.dropna()
            min_index = valid_values.idxmin()
            max_index = valid_values.idxmax()
            evidence_lines.extend([
                f"Axes: X={x_column}; Y={y_column}.",
                f"Minimum plotted value: {df.loc[min_index, x_column]} = {valid_values.loc[min_index]}.",
                f"Maximum plotted value: {df.loc[max_index, x_column]} = {valid_values.loc[max_index]}.",
            ])

        return (
            f"Chart created successfully. Path: /static/charts/chart_{chart_id}.png\n"
            + "\n".join(evidence_lines)
        )

    except Exception as e:
        return f"Error creating chart: {str(e)}"


def create_data_analyst_tools(include_file_listing: bool = False):
    """Create all data analyst tools."""
    tools = [read_csv_file, analyze_data, create_chart]
    if include_file_listing:
        return [list_data_files, read_data_file, *tools]
    return tools
