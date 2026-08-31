"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.core import get_settings

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if get_settings().debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    settings = get_settings()

    # Startup
    logger.info(f"Starting {settings.app_name} in {settings.app_env} mode")

    # Ensure data directories exist
    for path in [settings.vector_store_path, settings.reports_path, settings.uploads_path]:
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Data directory ready: {path}")

    yield

    # Shutdown
    logger.info("Shutting down application")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="基于 DeepAgents 架构与 Agentic RAG 的多智能体协同研报分析系统",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes, 挂载路由，监听 /api/* 请求
    app.include_router(api_router, prefix="/api")

    # Mount static files for generated charts
    # 确保 charts 目录存在并挂载为静态文件路径
    charts_path = settings.uploads_path / "charts"
    charts_path.mkdir(parents=True, exist_ok=True)
    app.mount("/static/charts", StaticFiles(directory=str(charts_path)), name="charts")

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "version": "0.1.0"}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
