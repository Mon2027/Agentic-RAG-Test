"""API routes for the application."""

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

# In-memory session storage
sessions: dict[str, dict[str, Any]] = {}

# Document processing status tracker
document_status: dict[str, dict[str, Any]] = {}

CHART_URL_PATTERN = re.compile(r"/static/charts/chart_[\w-]+\.png")
REPORT_SOURCE_PATTERN = re.compile(r"-\s+研报《[^》]+》第\d+(?:-\d+)?页(?:，[^-\n]+)?")


def _safe_upload_filename(filename: str) -> str:
    """Return the basename of a client filename on every server platform."""
    # Browsers may submit either Windows or POSIX separators.  Normalize them
    # before asking pathlib for the final component so Linux and Windows agree.
    return Path(filename.replace("\\", "/")).name


def process_pdf_task(
    file_path: Path,
    file_id: str,
    file_name: str,
    clear_existing: bool = False,
) -> None:
    """Background task to process uploaded PDF.

    Args:
        file_path: Path to the uploaded PDF file.
        file_id: Unique file identifier.
        file_name: Original file name.
        clear_existing: Whether to delete existing vector chunks before indexing.
    """
    import traceback

    try:
        logger.info(f"Processing PDF: {file_name} (ID: {file_id})")
        document_status[file_id] = {
            "status": DocumentStatus.PROCESSING,
            "message": "Processing PDF...",
        }

        settings = get_settings()

        # Initialize processor and vector store
        processor = DocumentProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        vector_store = get_vector_store()

        # Process the PDF before touching any existing index.
        chunks = processor.process_pdf(file_path, file_id)
        if not chunks:
            raise ValueError("PDF processing produced no document chunks")

        # Replace an existing index with rollback protection when reindexing.
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
        error_msg = f"Error processing PDF: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        document_status[file_id] = {
            "status": DocumentStatus.FAILED,
            "message": error_msg,
            "error": str(e),
        }


def _content_to_text(content: Any) -> str:
    """Convert LangChain event content into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
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
    """Read tool output text from a LangChain stream event."""
    output = event_data.get("output")
    if output is None:
        output = event_data.get("result")
    return _content_to_text(output)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message and return the agent's response.

    This is the main endpoint for interacting with the multi-agent system.
    """
    try:
        # 1. 导入并获取智能体实例
        from app.agents.main_agent import get_main_agent

        session_id = request.session_id or str(uuid.uuid4())
        agent = get_main_agent()

        # 2. 将前端消息转换为 LangChain 格式
        messages = []
        for msg in request.messages:
            if msg.role == MessageType.USER:
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == MessageType.ASSISTANT:
                messages.append(AIMessage(content=msg.content))

        # If no messages, return error
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")

        # 3. 调用智能体
        logger.info(f"Invoking agent for session {session_id}")
        result = agent.invoke(
            {"messages": messages},
            config={"recursion_limit": 50}
        )

        # 4. 提取并返回响应
        last_message = result["messages"][-1] if result.get("messages") else None

        if last_message:
            raw_content = last_message.content if hasattr(last_message, "content") else str(last_message)
            # Handle list content format (thinking + text blocks)
            if isinstance(raw_content, list):
                text_parts = []
                for block in raw_content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "thinking":
                            # Skip thinking blocks in response
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

        # Store session
        sessions[session_id] = {
            "messages": request.messages + [response_message],
            "last_updated": None,
        }

        return ChatResponse(
            message=response_message,
            session_id=session_id,
        )

    except HTTPException:
        raise
    except ImportError as e:
        logger.error(f"Agent import error: {e}")
        raise HTTPException(status_code=500, detail=f"Agent not available: {str(e)}")
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Process a chat message and stream the response.

    Uses Server-Sent Events (SSE) for real-time streaming.
    """
    import traceback

    session_id = request.session_id or str(uuid.uuid4())

    async def generate():
        try:
            from app.agents.main_agent import get_main_agent
            # 流式调用智能体
            agent = get_main_agent()

            # Convert messages
            messages = []
            for msg in request.messages:
                if msg.role == MessageType.USER:
                    messages.append(HumanMessage(content=msg.content))
                elif msg.role == MessageType.ASSISTANT:
                    messages.append(AIMessage(content=msg.content))

            if not messages:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No messages provided'})}\n\n"
                return

            # Send start event
            yield f"data: {json.dumps({'type': 'start', 'session_id': session_id})}\n\n"

            # Stream agent response
            accumulated_content = []
            pending_source_lines: list[str] = []
            # 流式调用智能体
            async for event in agent.astream_events(
                {"messages": messages},
                version="v2",
                config={"recursion_limit": 50}
            ):
                if event["event"] == "on_chat_model_stream":
                    # 返回 LLM 生成的 token
                    raw_content = event["data"]["chunk"].content if hasattr(event["data"]["chunk"], "content") else ""
                    if raw_content:
                        # Handle list content format (thinking + text blocks)
                        if isinstance(raw_content, list):
                            for block in raw_content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    text_content = block.get("text", "")
                                    if text_content:
                                        accumulated_content.append(text_content)
                                        yield f"data: {json.dumps({'type': 'token', 'content': text_content})}\n\n"
                        else:
                            # Simple string content
                            accumulated_content.append(str(raw_content))
                            yield f"data: {json.dumps({'type': 'token', 'content': str(raw_content)})}\n\n"

                elif event["event"] == "on_tool_start":
                    # 工具开始调用
                    tool_name = event["name"]
                    yield f"data: {json.dumps({'type': 'tool_start', 'tool': tool_name})}\n\n"

                elif event["event"] == "on_tool_end":
                    # Tool call finished
                    tool_name = event["name"]
                    yield f"data: {json.dumps({'type': 'tool_end', 'tool': tool_name})}\n\n"

                    if tool_name == "create_chart":
                        tool_output = _extract_tool_output_text(event.get("data", {}))
                        chart_urls = list(dict.fromkeys(CHART_URL_PATTERN.findall(tool_output)))
                        for chart_url in chart_urls:
                            chart_text = f"\n\n图表已生成：{chart_url}\n\n"
                            accumulated_content.append(chart_text)
                            yield f"data: {json.dumps({'type': 'token', 'content': chart_text})}\n\n"

                    if tool_name == "search_reports":
                        tool_output = _extract_tool_output_text(event.get("data", {}))
                        source_lines = list(dict.fromkeys(REPORT_SOURCE_PATTERN.findall(tool_output)))
                        for source_line in source_lines:
                            if source_line not in pending_source_lines:
                                pending_source_lines.append(source_line)
                            if len(pending_source_lines) >= 8:
                                break

            # Send end event
            if pending_source_lines:
                source_text = "\n\n📌 检索来源：\n" + "\n".join(pending_source_lines) + "\n\n"
                accumulated_content.append(source_text)
                yield f"data: {json.dumps({'type': 'token', 'content': source_text})}\n\n"

            full_response = "".join(accumulated_content)
            yield f"data: {json.dumps({'type': 'end', 'session_id': session_id, 'content': full_response})}\n\n"

            # Store session
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

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/upload/report", response_model=UploadResponse)
async def upload_report(
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(...)],
):
    """Upload a research report PDF file.

    The file will be processed, chunked, and stored in the vector database.
    Processing happens asynchronously in the background.
    """
    settings = get_settings()

    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        safe_filename = _safe_upload_filename(file.filename)
        file_path = settings.reports_path / f"{file_id}_{safe_filename}"

        # Save file
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        logger.info(f"Saved report: {safe_filename} -> {file_path}")

        # Initialize status
        document_status[file_id] = {
            "status": DocumentStatus.PENDING,
            "message": "File uploaded, waiting to process",
        }

        # Schedule background processing
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
    """Upload a data file (CSV or Excel) for analysis.

    The file will be stored and available for the data analysis agent.
    """
    settings = get_settings()

    # Validate file type
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
            file_type=file_ext[1:],  # Remove the dot
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
    """List all uploaded documents with their status."""
    settings = get_settings()
    documents: list[dict[str, Any]] = []

    # Get indexed files from vector store
    try:
        vector_store = get_vector_store()
        indexed_files = {f["file_id"]: f for f in vector_store.list_files()}
    except Exception:
        indexed_files = {}

    # Get reports
    for pdf_file in settings.reports_path.glob("*.pdf"):
        parts = pdf_file.name.split("_", 1)
        file_id = parts[0]
        file_name = parts[1] if len(parts) > 1 else pdf_file.name

        # Check status: first memory, then vector store
        if file_id in document_status:
            status_info = document_status[file_id]
        elif file_id in indexed_files:
            # File is indexed in vector store
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

    # Get data files
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
    """Rebuild vector index for an uploaded PDF document."""
    document_id = str(file_id)
    settings = get_settings()
    reports = list(settings.reports_path.glob(f"{document_id}_*.pdf"))
    if not reports:
        raise HTTPException(status_code=404, detail="PDF document not found")

    file_path = reports[0]
    file_name = file_path.name.replace(f"{document_id}_", "", 1)

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
    """Get information about a specific document."""
    document_id = str(file_id)
    settings = get_settings()

    # Check in reports
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

    # Check in uploads
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
    """Delete a document and its associated data."""
    document_id = str(file_id)
    settings = get_settings()
    deleted = False

    # Check and delete from reports
    reports = list(settings.reports_path.glob(f"{document_id}_*"))
    for report in reports:
        report.unlink()
        deleted = True

    # Delete from vector store
    try:
        vector_store = get_vector_store()
        vector_store.delete_by_file_id(document_id)
    except Exception as e:
        logger.warning(f"Error deleting from vector store: {e}")

    # Clean up status
    if document_id in document_status:
        del document_status[document_id]

    # Check and delete from uploads
    uploads = list(settings.uploads_path.glob(f"{document_id}_*"))
    for upload in uploads:
        upload.unlink()
        deleted = True

    if deleted:
        return {"status": "deleted", "file_id": document_id}

    raise HTTPException(status_code=404, detail="Document not found")


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session information and history."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its history."""
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "deleted"}
