"""pytest 的全局测试配置、安全保护措施和公共 fixture。

学习提示：
1. pytest 会在收集测试用例时自动发现并加载名为 ``conftest.py`` 的文件，
   因此同一目录及其子目录中的测试可以直接使用这里定义的 fixture，
   不需要再显式 ``import conftest``。
2. 本文件最重要的职责是“测试隔离”：测试产生的报告、上传文件和向量数据
   全部写入临时目录，避免误删或污染开发环境中的正式数据。
3. 本文件在应用模块导入前设置测试环境变量，并在 pytest 退出时统一清理资源。
"""

# Python 自带的垃圾回收模块；清理 Chroma 后用它促使残留对象及时释放文件句柄。
import gc
# 用于设置进程级环境变量，让应用在测试期间读取到测试专用配置。
import os
# 用于修改模块搜索路径，以及查询当前已经导入的模块。
import sys
# 用于创建本次测试进程独享、退出后可自动删除的临时目录。
import tempfile
# 面向对象的路径工具，比直接拼接字符串更安全，也更易跨平台使用。
from pathlib import Path

# pytest 本体：本文件会使用 fixture 装饰器、配置类型和退出钩子。
import pytest


# ``__file__`` 是当前 conftest.py 的路径：
#   Path(__file__).parent        -> backend/tests
#   Path(__file__).parent.parent -> backend
# ``resolve()`` 将其转换为规范的绝对路径，便于后续进行可靠的路径比较。
backend_path = Path(__file__).parent.parent.resolve()

# 把 backend 放到模块搜索路径的最前面。
# 这样测试代码中的 ``from app...`` 会优先导入当前项目 backend/app 下的代码，
# 即使运行 pytest 时的工作目录不是 backend，也不会出现找不到 app 包的问题。
sys.path.insert(0, str(backend_path))

# 开发/正式环境默认使用的 backend/data 目录。
# 这里只记录并规范化该路径，后面会用它进行安全检查，防止测试误用真实数据目录。
formal_data_path = (backend_path / "data").resolve()

# 必须在 pytest 导入应用模块之前完成环境隔离。
# 原因是应用的 settings 可能在“模块导入阶段”就读取环境变量并创建数据目录；
# 如果把这些配置放进 fixture，fixture 直到执行测试时才运行，那时已经太晚了。
#
# TemporaryDirectory 会立即创建一个随机临时目录，并持有它的生命周期。
# prefix 只是让系统临时目录中的名称更容易辨认和排查。
_test_data_directory = tempfile.TemporaryDirectory(
    prefix="report-analysis-agent-tests-"
)

# ``.name`` 是刚创建的临时目录字符串路径；转为 Path 并规范化后，
# 它将作为本次 pytest 进程所有测试数据的总根目录。
isolated_data_path = Path(_test_data_directory.name).resolve()

# 双重安全检查：测试临时目录既不能等于正式数据目录，也不能位于正式数据目录内部。
# ``isolated_data_path.parents`` 包含该路径的所有上级目录；若 formal_data_path 在其中，
# 就说明临时目录被错误地建到了 backend/data 下面，后续清理可能危及真实数据。
if isolated_data_path == formal_data_path or formal_data_path in isolated_data_path.parents:
    raise RuntimeError(
        f"Refusing to use formal application data for tests: {isolated_data_path}"
    )

# 这里必须使用直接赋值，而不是 ``os.environ.setdefault(...)``。
# 直接赋值会强制覆盖终端环境变量和 .env 中可能已有的值，确保任何开发者机器、
# CI 环境或运行方式都不能把测试重新指向开发数据或真实的外部服务。

# 告诉应用当前运行环境是 testing；应用可据此启用测试专用行为或默认值。
os.environ["APP_ENV"] = "testing"

# 将三类持久化数据分别放到隔离根目录的不同子目录中。
# 应用后续读取 settings 时会使用这些路径，而不会写入 backend/data。
os.environ["REPORTS_PATH"] = str(isolated_data_path / "reports")
os.environ["UPLOADS_PATH"] = str(isolated_data_path / "uploads")
os.environ["VECTOR_STORE_PATH"] = str(isolated_data_path / "vectorstore")

# 清空真实 Anthropic API Key，防止测试意外向真实模型服务发出计费请求。
os.environ["ANTHROPIC_API_KEY"] = ""

# 提供一个明显的测试凭证，使只检查“是否存在认证令牌”的代码仍可正常初始化。
os.environ["ANTHROPIC_AUTH_TOKEN"] = "test_token"

# 将模型服务地址指向本机 9 号端口（通常不会有服务监听）。
# 即使某个测试遗漏了 mock，请求也会快速失败，而不会访问公网真实服务。
os.environ["ANTHROPIC_BASE_URL"] = "http://127.0.0.1:9"

# 禁用 Tavily 搜索服务的真实凭证，避免测试误调用外部搜索 API。
os.environ["TAVILY_API_KEY"] = ""

# 强制 Hugging Face 及 Transformers/Datasets 使用离线模式。
# 测试运行时不会临时下载模型或数据集，因此结果更稳定，也不依赖网络。
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# 关闭匿名遥测，避免测试过程向外发送使用信息。
os.environ["ANONYMIZED_TELEMETRY"] = "False"


def _close_chroma_systems() -> None:
    """关闭 Chroma 资源，确保 Windows 能删除其 SQLite 文件。

    Chroma 的持久化客户端内部可能一直持有 SQLite 文件句柄。Windows 不允许删除
    仍被进程占用的文件，所以在清理临时目录前要先停止 Chroma 的共享系统、清空
    缓存，并解除项目自身保存的向量库单例引用。

    函数名以下划线开头，表示它是本模块内部使用的辅助函数，不是给测试用例直接
    调用的公共 fixture。
    """
    try:
        # 延迟导入：只有执行清理时才需要 Chroma。
        # 某些只运行轻量测试的环境可能没有安装 chromadb，此时无需清理，直接返回。
        from chromadb.api.shared_system_client import SharedSystemClient
    except ImportError:
        return

    # ``_identifier_to_system`` 保存当前进程内 Chroma 创建过的共享 System 对象。
    # 先复制成 list 再遍历，避免 stop() 间接修改原字典时触发“遍历期间改变大小”。
    for system in list(SharedSystemClient._identifier_to_system.values()):
        # 停止每个 Chroma System，释放数据库连接、线程以及底层文件句柄。
        system.stop()

    # 清除 Chroma 类级别的共享系统缓存，避免缓存继续持有已停止对象的引用。
    SharedSystemClient.clear_system_cache()

    # 项目的 vectorstore 模块还维护了 ``_vector_store`` 单例。
    # 通过 sys.modules 查询“模块是否已经导入”，而不是在此处主动 import；
    # 主动导入可能反而创建新的向量库客户端，抵消清理工作的效果。
    vectorstore_module = sys.modules.get("app.rag.vectorstore")
    if vectorstore_module is not None:
        # 解除模块对 VectorStore 实例的全局引用，使其具备被垃圾回收的条件。
        vectorstore_module._vector_store = None

    # 主动触发一次垃圾回收，让失去引用的客户端对象尽快执行析构并释放文件句柄。
    gc.collect()


def pytest_unconfigure(config: pytest.Config) -> None:
    """pytest 退出钩子：关闭 Chroma，然后删除本次测试的隔离数据。

    ``pytest_unconfigure`` 是 pytest 规定的钩子函数名，无需手动调用。无论测试全部
    通过、出现失败，还是没有收集到用例，pytest 在撤销配置阶段都会调用它。

    Args:
        config: 当前 pytest 会话的配置对象。本函数不需要读取它，但钩子签名必须
            接收该参数，类型标注也方便编辑器识别。
    """
    # 清理顺序很重要：先释放数据库文件句柄，再删除包含数据库的临时目录。
    _close_chroma_systems()

    # 删除 TemporaryDirectory 创建的整个隔离目录及其中的测试产物。
    _test_data_directory.cleanup()


# 注册一个 session 级 fixture。
# scope="session" 表示整个 pytest 进程只创建/获取一次返回值，所有用例共享同一路径。
@pytest.fixture(scope="session")
def isolated_data_root() -> Path:
    """返回当前 pytest 进程使用的隔离数据根目录。

    测试函数只需声明同名参数即可让 pytest 自动注入，例如：
    ``def test_upload(isolated_data_root): ...``。

    Returns:
        本次测试会话专属的临时目录绝对路径。
    """
    return isolated_data_path


# 未指定 scope 时默认是 function：每个使用它的测试函数都会单独执行一次 fixture。
# 这里返回的是不可变路径，因此即便用 session scope 也可以；保留默认 scope 能直观
# 展示 pytest fixture 的默认行为。
@pytest.fixture
def sample_pdf_path() -> Path:
    """返回接口上传测试所用示例 PDF 的路径。

    Returns:
        backend/tests/fixtures/sample.pdf 的 Path 对象。
    """
    # 从当前文件所在的 tests 目录出发构造路径，不依赖 pytest 的启动目录。
    return Path(__file__).parent / "fixtures" / "sample.pdf"


@pytest.fixture
def sample_csv_path() -> Path:
    """返回接口上传或数据分析测试所用示例 CSV 的路径。

    Returns:
        backend/tests/fixtures/sample.csv 的 Path 对象。
    """
    return Path(__file__).parent / "fixtures" / "sample.csv"


@pytest.fixture
def sample_text() -> str:
    """提供一段可复用的中文财报文本测试数据。

    Returns:
        包含标题、业绩概述、业务亮点和投资建议的多行字符串。

    多行字符串保留了源码中的换行和缩进。消费它的测试若关心精确文本格式，通常
    可以再调用 ``textwrap.dedent`` 或 ``str.strip`` 进行规范化。
    """
    return """
    中信海直2025年半年报点评

    一、业绩概述
    2025年上半年，公司实现营业收入10.38亿元，同比增长7.9%。

    二、业务亮点
    1. 通航业务稳健增长
    2. 低空经济布局加速

    三、投资建议
    给予"谨慎推荐"评级。
    """
