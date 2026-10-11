"""应用的 API 路由。

这里负责接收聊天、上传、文档和会话请求，协调智能体、文件系统与向量库，
并把处理结果转换为普通 JSON 响应或 SSE 流式事件。
"""

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentStatus,
    Message,
    MessageType,
    UploadResponse,
)
from app.core import get_settings
from app.rag.document_processor import DocumentProcessor
from app.rag.vectorstore import get_vector_store

logger = logging.getLogger(__name__)
router = APIRouter()

# 进程内的会话历史；进程重启后会丢失，多个服务进程之间也不共享。
sessions: dict[str, dict[str, Any]] = {}

# PDF 后台处理状态同样只保存在当前进程内。
document_status: dict[str, dict[str, Any]] = {}

# 从工具输出中识别已生成图表的静态资源地址和研报来源行。
CHART_URL_PATTERN = re.compile(r"/static/charts/chart_[\w-]+\.png")
REPORT_SOURCE_PATTERN = re.compile(r"-\s+研报《[^》]+》第\d+(?:-\d+)?页(?:，[^-\n]+)?")


def _safe_upload_filename(filename: str) -> str:
    """提取上传文件名的最后一段，避免客户端路径参与服务器端路径拼接。"""
    # 客户端可能传入 Windows 或 POSIX 路径分隔符；先统一为 /，再提取文件名，
    # 使不同服务器平台得到相同结果。
    return Path(filename.replace("\\", "/")).name


def process_pdf_task(
    file_path: Path,
    file_id: str,
    file_name: str,
    clear_existing: bool = False,
) -> None:
    """在后台解析 PDF 并写入向量索引。

    参数：
        file_path: 已保存的 PDF 路径。
        file_id: 上传时生成的文档唯一 ID。
        file_name: 返回给用户的原始文件名。
        clear_existing: 为 True 时重建已有索引，否则新增文档。

    异常会记录为失败状态，供文档查询接口读取；后台任务本身不向客户端抛错。
    """
    import traceback

    try:
        logger.info(f"Processing PDF: {file_name} (ID: {file_id})")
        document_status[file_id] = {
            "status": DocumentStatus.PROCESSING,
            "message": "Processing PDF...",
        }

        settings = get_settings()

        # 使用应用配置创建解析器；向量库通过工厂函数获取，便于测试时替换。
        processor = DocumentProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        vector_store = get_vector_store()

        # 先完成解析和分块，再操作旧索引；空结果视为处理失败。
        chunks = processor.process_pdf(file_path, file_id)
        if not chunks:
            raise ValueError("PDF processing produced no document chunks")

        # 重建时由向量库的替换方法保护旧分块；首次上传则直接新增。
        if clear_existing:
            vector_store.replace_document_for_file(chunks, file_id)
        else:
            vector_store.add_document_for_file(chunks, file_id)

        document_status[file_id] = {
            "status": DocumentStatus.COMPLETED,
            "message": f"Successfully processed {len(chunks)} chunks",
            "chunk_count": len(chunks),
        }

        logger.info(f"PDF processing completed: {file_name} ({len(chunks)} chunks)")

    except Exception as e:
        # 后台异常无法通过上传请求的 HTTP 状态返回，因此保存到状态表并记日志。
        error_msg = f"Error processing PDF: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        document_status[file_id] = {
            "status": DocumentStatus.FAILED,
            "message": error_msg,
            "error": str(e),
        }


def _content_to_text(content: Any) -> str:
    """将 LangChain 事件中的字符串、内容块或消息对象统一转为文本。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # 内容块可能是 text/content 字典，也可能是可直接转成字符串的对象。
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(str(block.get("text", "")))
                elif "content" in block:
                    text_parts.append(str(block.get("content", "")))
            else:
                text_parts.append(str(block))
        return "\n".join(part for part in text_parts if part)
    if hasattr(content, "content"):
        return _content_to_text(content.content)
    return str(content)


def _extract_tool_output_text(event_data: dict[str, Any]) -> str:
    """从工具结束事件读取 output 或 result 字段，并统一转为文本。"""
    output = event_data.get("output")
    if output is None:
        output = event_data.get("result")
    return _content_to_text(output)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """执行一次非流式对话，并返回智能体的最终回复。

    请求消息会转为 LangChain 消息；处理完成后，历史记录写入进程内会话表。
    """
    try:
        # 1. 延迟导入智能体；未提供 session_id 时生成一个新会话 ID。
        from app.agents.main_agent import get_main_agent

        session_id = request.session_id or str(uuid.uuid4())
        agent = get_main_agent()

        # 2. 仅把用户和助手消息转为 LangChain 格式；SYSTEM 消息不会加入本次调用。
        messages = []
        for msg in request.messages:
            if msg.role == MessageType.USER:
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == MessageType.ASSISTANT:
                messages.append(AIMessage(content=msg.content))

        # 转换后为空时返回 400，避免调用没有有效消息的智能体。
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")

        # 3. 同步执行智能体；递归上限约束工具和子智能体的调用深度。
        logger.info(f"Invoking agent for session {session_id}")
        result = agent.invoke(
            {"messages": messages},
            config={"recursion_limit": 50}
        )

        # 4. 从结果消息列表中提取最终消息。
        last_message = result["messages"][-1] if result.get("messages") else None

        if last_message:
            raw_content = last_message.content if hasattr(last_message, "content") else str(last_message)
            # 模型内容可能同时包含思考块和文本块；仅将文本块暴露给客户端。
            if isinstance(raw_content, list):
                text_parts = []
                for block in raw_content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "thinking":
                            # 思考块属于内部过程，不拼入用户可见回复。
                            pass
                    else:
                        text_parts.append(str(block))
                response_content = "\n".join(text_parts) if text_parts else str(raw_content)
            else:
                response_content = str(raw_content)
        else:
            response_content = "抱歉，我无法处理您的请求。请稍后重试。"

        response_message = Message(
            role=MessageType.ASSISTANT,
            content=response_content,
        )

        # 保存本次请求消息和助手回复，供会话查询接口读取。
        sessions[session_id] = {
            "messages": request.messages + [response_message],
            "last_updated": None,
        }

        return ChatResponse(
            message=response_message,
            session_id=session_id,
        )

    except HTTPException:
        # 保留主动抛出的 400 等业务错误，不转成 500。
        raise
    except ImportError as e:
        logger.error(f"Agent import error: {e}")
        raise HTTPException(status_code=500, detail=f"Agent not available: {str(e)}")
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """以 SSE 事件流逐步返回智能体回复及工具调用状态。

    每个事件都以 ``data: <JSON>\n\n`` 格式发送；最终的 end 事件包含完整文本。
    流式生成器内部捕获异常并发送 error 事件，而不是中途改写 HTTP 状态码。
    """
    import traceback

    session_id = request.session_id or str(uuid.uuid4())

    async def generate():
        """按智能体事件产生 SSE 数据块，并在结束时记录会话。"""
        try:
            from app.agents.main_agent import get_main_agent
            # 获取智能体，并准备与普通聊天接口相同的消息格式。
            agent = get_main_agent()

            # SYSTEM 消息不在这里转换；无有效消息时只发送 error 事件。
            messages = []
            for msg in request.messages:
                if msg.role == MessageType.USER:
                    messages.append(HumanMessage(content=msg.content))
                elif msg.role == MessageType.ASSISTANT:
                    messages.append(AIMessage(content=msg.content))

            if not messages:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No messages provided'})}\n\n"
                return

            # start 事件先告诉客户端会话 ID，随后才输出模型和工具事件。
            yield f"data: {json.dumps({'type': 'start', 'session_id': session_id})}\n\n"

            # 累积内容用于最终 end 事件和会话历史；来源行先暂存，最后统一附加。
            accumulated_content = []
            pending_source_lines: list[str] = []
            # astream_events(v2) 会逐个产生模型 token 和工具生命周期事件。
            async for event in agent.astream_events(
                {"messages": messages},
                version="v2",
                config={"recursion_limit": 50}
            ):
                if event["event"] == "on_chat_model_stream":
                    # 从模型片段中提取用户可见文本，逐段发送 token 事件。
                    raw_content = event["data"]["chunk"].content if hasattr(event["data"]["chunk"], "content") else ""
                    if raw_content:
                        # 列表格式只取 text 内容块，跳过思考等其他类型。
                        if isinstance(raw_content, list):
                            for block in raw_content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    text_content = block.get("text", "")
                                    if text_content:
                                        accumulated_content.append(text_content)
                                        yield f"data: {json.dumps({'type': 'token', 'content': text_content})}\n\n"
                        else:
                            # 字符串格式可以直接作为一个 token 事件发出。
                            accumulated_content.append(str(raw_content))
                            yield f"data: {json.dumps({'type': 'token', 'content': str(raw_content)})}\n\n"

                elif event["event"] == "on_tool_start":
                    # 通知客户端工具开始执行，便于界面展示处理进度。
                    tool_name = event["name"]
                    yield f"data: {json.dumps({'type': 'tool_start', 'tool': tool_name})}\n\n"

                elif event["event"] == "on_tool_end":
                    # 通知客户端工具已结束，再从特定工具的输出补充可见内容。
                    tool_name = event["name"]
                    yield f"data: {json.dumps({'type': 'tool_end', 'tool': tool_name})}\n\n"

                    if tool_name == "create_chart":
                        # 提取图表 URL 并去重，避免同一图表在消息中重复出现。
                        tool_output = _extract_tool_output_text(event.get("data", {}))
                        chart_urls = list(dict.fromkeys(CHART_URL_PATTERN.findall(tool_output)))
                        for chart_url in chart_urls:
                            chart_text = f"\n\n图表已生成：{chart_url}\n\n"
                            accumulated_content.append(chart_text)
                            yield f"data: {json.dumps({'type': 'token', 'content': chart_text})}\n\n"

                    if tool_name == "search_reports":
                        # 收集最多 8 条不重复的研报来源，在回答末尾集中展示。
                        tool_output = _extract_tool_output_text(event.get("data", {}))
                        source_lines = list(dict.fromkeys(REPORT_SOURCE_PATTERN.findall(tool_output)))
                        for source_line in source_lines:
                            if source_line not in pending_source_lines:
                                pending_source_lines.append(source_line)
                            if len(pending_source_lines) >= 8:
                                break

            # 先把检索来源作为 token 补给客户端，再发送包含完整内容的 end 事件。
            if pending_source_lines:
                source_text = "\n\n📌 检索来源：\n" + "\n".join(pending_source_lines) + "\n\n"
                accumulated_content.append(source_text)
                yield f"data: {json.dumps({'type': 'token', 'content': source_text})}\n\n"

            full_response = "".join(accumulated_content)
            yield f"data: {json.dumps({'type': 'end', 'session_id': session_id, 'content': full_response})}\n\n"

            # 只有正常流式结束后才把完整回复写入进程内会话表。
            sessions[session_id] = {
                "messages": request.messages + [Message(role=MessageType.ASSISTANT, content=full_response)],
            }

        except ImportError as e:
            logger.error(f"Agent import error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': f'Agent not available: {str(e)}'})}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            logger.error(traceback.format_exc())
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    # StreamingResponse 负责将异步生成器产生的数据持续写入 HTTP 响应。
    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/upload/report", response_model=UploadResponse)
async def upload_report(
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(...)],
):
    """上传 PDF 研报，并安排后台解析与向量索引。

    接口先保存原文件并返回 file_id；后台任务完成前，文档状态为等待处理。
    """
    settings = get_settings()

    # 先检查文件名及扩展名；这里不解析 PDF 内容，实际解析由后台任务完成。
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        # UUID 用于区分同名上传文件；安全文件名用于避免客户端路径影响保存位置。
        file_id = str(uuid.uuid4())
        safe_filename = _safe_upload_filename(file.filename)
        file_path = settings.reports_path / f"{file_id}_{safe_filename}"

        # 将上传内容保存到配置指定的报告目录，文件名以 file_id 开头。
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        logger.info(f"Saved report: {safe_filename} -> {file_path}")

        # 先登记等待状态，后续可通过文档接口查看后台处理结果。
        document_status[file_id] = {
            "status": DocumentStatus.PENDING,
            "message": "File uploaded, waiting to process",
        }

        # 响应发出后运行后台任务；False 表示新增索引而非替换旧索引。
        background_tasks.add_task(
            process_pdf_task,
            file_path,
            file_id,
            safe_filename,
            False,
        )

        return UploadResponse(
            success=True,
            file_id=file_id,
            file_name=safe_filename,
            file_type="pdf",
            message="Report uploaded successfully. Processing has started in the background.",
            metadata={"file_size": len(content)},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload/data", response_model=UploadResponse)
async def upload_data(file: Annotated[UploadFile, File(...)]):
    """上传 CSV 或 Excel 文件，供数据分析智能体后续读取。

    与 PDF 上传不同，此接口只保存文件，不安排向量索引任务。
    """
    settings = get_settings()

    # 依据安全文件名的扩展名判断类型，不读取文件内容来验证格式。
    allowed_extensions = [".csv", ".xlsx", ".xls"]
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    safe_filename = _safe_upload_filename(file.filename)
    file_ext = Path(safe_filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}",
        )

    try:
        # 与 PDF 上传一样，用 UUID 前缀避免同名文件互相覆盖。
        file_id = str(uuid.uuid4())
        file_path = settings.uploads_path / f"{file_id}_{safe_filename}"

        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        logger.info(f"Saved data file: {safe_filename} -> {file_path}")

        return UploadResponse(
            success=True,
            file_id=file_id,
            file_name=safe_filename,
            file_type=file_ext[1:],  # 去掉扩展名前的点，返回 csv、xlsx 或 xls。
            message="Data file uploaded successfully.",
            metadata={"file_size": len(content)},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents")
async def list_documents():
    """合并磁盘上的 PDF 和数据文件，返回文档列表及已知状态。"""
    settings = get_settings()
    documents: list[dict[str, Any]] = []

    # 向量库提供已建立索引的 PDF 信息；不可用时仍可列出磁盘文件。
    try:
        vector_store = get_vector_store()
        indexed_files = {f["file_id"]: f for f in vector_store.list_files()}
    except Exception:
        indexed_files = {}

    # 报告文件采用 ``file_id_原文件名.pdf`` 命名，按第一个下划线拆出 ID。
    for pdf_file in settings.reports_path.glob("*.pdf"):
        parts = pdf_file.name.split("_", 1)
        file_id = parts[0]
        file_name = parts[1] if len(parts) > 1 else pdf_file.name

        # 状态优先级：当前进程的后台状态、向量库索引、默认等待处理。
        if file_id in document_status:
            status_info = document_status[file_id]
        elif file_id in indexed_files:
            # 向量库中存在记录，说明当前进程之外也可能已完成索引。
            status_info = {
                "status": DocumentStatus.COMPLETED,
                "chunk_count": indexed_files[file_id].get("chunk_count", 0),
            }
        else:
            status_info = {"status": DocumentStatus.PENDING}

        documents.append({
            "file_id": file_id,
            "file_name": file_name,
            "file_type": "pdf",
            "file_size": pdf_file.stat().st_size,
            "status": status_info.get("status", DocumentStatus.PENDING),
            "chunk_count": status_info.get("chunk_count"),
            "message": status_info.get("message"),
        })

    # CSV/Excel 不经过 PDF 后台处理；存在于上传目录即视为可用。
    for data_file in settings.uploads_path.glob("*"):
        if data_file.is_file() and not data_file.name.startswith("."):
            parts = data_file.name.split("_", 1)
            file_id = parts[0]
            file_name = parts[1] if len(parts) > 1 else data_file.name

            documents.append({
                "file_id": file_id,
                "file_name": file_name,
                "file_type": data_file.suffix[1:] if data_file.suffix else "unknown",
                "file_size": data_file.stat().st_size,
                "status": DocumentStatus.COMPLETED,
            })

    return documents


@router.post("/documents/{file_id}/reindex", response_model=UploadResponse)
async def reindex_document(file_id: uuid.UUID, background_tasks: BackgroundTasks):
    """为已有 PDF 重新安排后台解析和向量索引任务。

    UUID 类型让 FastAPI 在进入函数前拒绝非法路径参数；找不到对应 PDF 时返回 404。
    """
    # 转成规范 UUID 字符串后再参与文件匹配，避免原始路径参数含有通配符。
    document_id = str(file_id)
    settings = get_settings()
    reports = list(settings.reports_path.glob(f"{document_id}_*.pdf"))
    if not reports:
        raise HTTPException(status_code=404, detail="PDF document not found")

    file_path = reports[0]
    file_name = file_path.name.replace(f"{document_id}_", "", 1)

    # 立即标记为等待处理；后台任务使用 True 表示替换已有索引。
    document_status[document_id] = {
        "status": DocumentStatus.PENDING,
        "message": "Reindex requested, waiting to process",
    }

    background_tasks.add_task(
        process_pdf_task,
        file_path,
        document_id,
        file_name,
        True,
    )

    return UploadResponse(
        success=True,
        file_id=document_id,
        file_name=file_name,
        file_type="pdf",
        message="Reindex started in the background.",
        metadata={"file_size": file_path.stat().st_size},
    )


@router.get("/documents/{file_id}")
async def get_document(file_id: uuid.UUID):
    """按 UUID 查询单个 PDF 或数据文件的磁盘信息与处理状态。"""
    document_id = str(file_id)
    settings = get_settings()

    # 优先在研报目录查找；PDF 状态来自当前进程的后台状态表。
    reports = list(settings.reports_path.glob(f"{document_id}_*"))
    if reports:
        file_path = reports[0]
        status_info = document_status.get(
            document_id,
            {"status": DocumentStatus.PENDING},
        )

        return {
            "file_id": document_id,
            "file_name": file_path.name.replace(f"{document_id}_", ""),
            "file_type": "pdf",
            "file_size": file_path.stat().st_size,
            "status": status_info.get("status", DocumentStatus.PENDING),
            "chunk_count": status_info.get("chunk_count"),
            "message": status_info.get("message"),
        }

    # 如果不是 PDF，再查找数据上传目录；此类文件无需等待向量索引。
    uploads = list(settings.uploads_path.glob(f"{document_id}_*"))
    if uploads:
        file_path = uploads[0]
        return {
            "file_id": document_id,
            "file_name": file_path.name.replace(f"{document_id}_", ""),
            "file_type": file_path.suffix[1:] if file_path.suffix else "unknown",
            "file_size": file_path.stat().st_size,
            "status": DocumentStatus.COMPLETED,
        }

    raise HTTPException(status_code=404, detail="Document not found")


@router.delete("/documents/{file_id}")
async def delete_document(file_id: uuid.UUID):
    """按 UUID 删除对应磁盘文件、向量分块和内存状态。

    路径参数先由 FastAPI 校验为 UUID，再拼入 glob 模式，防止通配符匹配其他文件。
    向量库删除失败只记警告；最终是否返回 404 取决于磁盘文件是否被删除。
    """
    document_id = str(file_id)
    settings = get_settings()
    deleted = False

    # 删除研报目录中该 ID 对应的文件，并记录是否确实删除了磁盘文件。
    reports = list(settings.reports_path.glob(f"{document_id}_*"))
    for report in reports:
        report.unlink()
        deleted = True

    # 尝试清理同一 ID 的检索分块；失败时继续执行其余清理步骤。
    try:
        vector_store = get_vector_store()
        vector_store.delete_by_file_id(document_id)
    except Exception as e:
        logger.warning(f"Error deleting from vector store: {e}")

    # 移除本进程保存的后台处理状态。
    if document_id in document_status:
        del document_status[document_id]

    # 同一 ID 也可能对应普通数据文件，因此继续检查上传目录。
    uploads = list(settings.uploads_path.glob(f"{document_id}_*"))
    for upload in uploads:
        upload.unlink()
        deleted = True

    if deleted:
        return {"status": "deleted", "file_id": document_id}

    raise HTTPException(status_code=404, detail="Document not found")


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """读取当前进程保存的会话历史；不存在时返回 404。"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除当前进程的会话记录；重复删除也返回成功。"""
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "deleted"}
