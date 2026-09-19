import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.domain_endpoints import (
    analytics_router,
    compliance_router,
    facility_router,
    genealogy_router,
    production_router,
    tools_router,
)
from app.api.summary import router as summary_router
from app.core.config import settings
from app.core.database import close_mongo, init_mongo
from app.clients.base_client import close_http_client, init_http_client
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def validate_production_environment(settings_obj: Any = None) -> None:
    """
    Ensure required environment variables are present in production.
    Refuse to start if any required variable is empty.
    Skip check in development.
    """
    cfg = settings_obj or settings
    env = (getattr(cfg, "environment", None) or os.getenv("ENVIRONMENT", "production")).strip().lower()
    if env == "development":
        return

    required_variables = [
        ("PERMISSIONS_API", getattr(cfg, "permissions_api", "") or os.getenv("PERMISSIONS_API", "")),
        ("FACILITY_API", getattr(cfg, "facility_api", "") or os.getenv("FACILITY_API", "")),
        ("PRODUCTION_API", getattr(cfg, "production_api", "") or os.getenv("PRODUCTION_API", "")),
        ("COMPLIANCE_API", getattr(cfg, "compliance_api", "") or os.getenv("COMPLIANCE_API", "")),
        ("ORDER_API", getattr(cfg, "order_api", "") or os.getenv("ORDER_API", "")),
        ("NOTIFICATION_HUB_API", getattr(cfg, "notification_hub_api", "") or os.getenv("NOTIFICATION_HUB_API", "")),
        ("FRONTEND_ORIGIN", getattr(cfg, "frontend_origin", "") or os.getenv("FRONTEND_ORIGIN", "")),
        ("MONGO_CONNECTION_STRING", getattr(cfg, "mongo_connection_string", "") or os.getenv("MONGO_CONNECTION_STRING", "")),
        ("MONGO_DATABASE_NAME", getattr(cfg, "mongo_database_name", "") or os.getenv("MONGO_DATABASE_NAME", "")),
    ]

    for var_name, var_value in required_variables:
        if not var_value or not str(var_value).strip():
            raise RuntimeError(f"Missing required environment variable: {var_name}")


def get_cors_origins(settings_obj: Any = None) -> list[str]:
    """
    Lock CORS to FRONTEND_ORIGIN only when ENVIRONMENT="production";
    allow localhost in development.
    """
    cfg = settings_obj or settings
    env = (getattr(cfg, "environment", None) or os.getenv("ENVIRONMENT", "production")).strip().lower()
    if env == "production":
        frontend = (getattr(cfg, "frontend_origin", "") or os.getenv("FRONTEND_ORIGIN", "")).strip()
        return [frontend] if frontend else []
    
    dev_origins = [
        "http://localhost:4200",
        "http://127.0.0.1:4200",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "null",
    ]
    frontend = (getattr(cfg, "frontend_origin", "") or os.getenv("FRONTEND_ORIGIN", "")).strip()
    if frontend and frontend not in dev_origins:
        dev_origins.append(frontend)
    return dev_origins


@asynccontextmanager
async def lifespan(_: FastAPI):
    validate_production_environment()
    env = (getattr(settings, "environment", None) or os.getenv("ENVIRONMENT", "production")).strip().lower()
    if env == "production":
        logger.info(
            "OpenAI compatibility routes (/v1/chat/completions, /v1/models, /models) are disabled in production."
        )
    await init_mongo()
    await init_http_client()
    try:
        yield
    finally:
        await close_http_client()
        await close_mongo()


def create_app() -> FastAPI:
    app_instance = FastAPI(
        title="CPG AI Backend",
        description="AI backend for CPG AI Facility chatbot",
        version="1.0.0",
        lifespan=lifespan,
    )

    env = (getattr(settings, "environment", None) or os.getenv("ENVIRONMENT", "production")).strip().lower()
    cors_origins = get_cors_origins()
    if env == "production":
        if not cors_origins:
            raise RuntimeError(
                "FRONTEND_ORIGIN must be set for CORS in production; wildcard fallback is prohibited."
            )
    else:
        cors_origins = cors_origins or ["*"]

    app_instance.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$" if env != "production" else None,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app_instance.middleware("http")
    async def extract_cpg_headers_middleware(request: Request, call_next):
        col_id = (
            request.headers.get("collectionid")
            or request.headers.get("x-collection-id")
            or request.headers.get("collection_id")
            or request.headers.get("x-tenant-id")
            or request.headers.get("tenantid")
            or request.headers.get("x-facility-id")
            or request.headers.get("facilityid")
            or ""
        ).strip()
        if col_id:
            from app.clients.base_client import _request_collection_id
            _request_collection_id.set(col_id)
        return await call_next(request)

    @app_instance.get("/health")
    async def health():
        return {
            "status": "ok",
            "service": "cpg-ai-backend",
        }

    # Documented industry-standard API prefix: /v1
    app_instance.include_router(chat_router, prefix="/v1")
    app_instance.include_router(summary_router, prefix="/v1")
    app_instance.include_router(production_router)
    app_instance.include_router(compliance_router)
    app_instance.include_router(genealogy_router)
    app_instance.include_router(analytics_router)
    app_instance.include_router(tools_router)
    app_instance.include_router(facility_router)

    # Backward-compatibility alias for unversioned clients (hidden from Swagger docs)
    app_instance.include_router(chat_router, include_in_schema=False)
    app_instance.include_router(summary_router, include_in_schema=False)

    # ── OPENAI / OLLAMA COMPATIBILITY ROUTES (DEV-ONLY) ───────────────────────────
    # These routes exist specifically to support unauthenticated model discovery and
    # chat probing from local development tools (e.g. OpenWebUI, Chatbot UI, LibreChat)
    # without requiring CPG AI JWT tokens or facility session credentials.
    # In production, they are strictly excluded to prevent unauthenticated LLM cost/abuse
    # exposure, ensuring all production traffic routes through authenticated /chat endpoints.
    if env != "production":
        @app_instance.get("/test-ui", include_in_schema=False)
        async def test_ui():
            from pathlib import Path
            from fastapi.responses import FileResponse
            ui_path = Path(__file__).parent.parent / "test_chat_ui.html"
            if ui_path.exists():
                return FileResponse(ui_path)
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="test_chat_ui.html not found")

        @app_instance.get("/v1/models")
        @app_instance.get("/models")
        async def list_models():
            model_name = getattr(settings, "ai_model", "gpt-4o-mini")
            return {
                "object": "list",
                "data": [
                    {
                        "id": model_name,
                        "object": "model",
                        "created": 1700000000,
                        "owned_by": "cpg-ai-backend",
                    }
                ],
                "models": [{"name": model_name, "model": model_name}],
            }

        @app_instance.post("/v1/chat/completions")
        async def chat_completions(payload: dict):
            messages = payload.get("messages") or []
            last_msg = messages[-1].get("content", "Hello") if messages else "Hello"
            from app.services.ai_service import generate_response
            res = await generate_response(last_msg)
            return {
                "id": "chatcmpl-cpg",
                "object": "chat.completion",
                "created": 1700000000,
                "model": getattr(settings, "ai_model", "gpt-4o-mini"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": str(res),
                        },
                        "finish_reason": "stop",
                    }
                ],
            }

    return app_instance


app = create_app()


