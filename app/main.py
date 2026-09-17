import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.core.config import settings
from app.core.database import close_mongo, init_mongo
from app.services.facility_api_service import close_http_client, init_http_client

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
    
    dev_origins = ["http://localhost:4200", "http://127.0.0.1:4200", "http://localhost:3000"]
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
        description="AI backend for CareOpsRx Facility chatbot",
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
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app_instance.get("/health")
    async def health():
        return {
            "status": "ok",
            "service": "cpg-ai-backend",
        }

    # Documented industry-standard API prefix: /v1
    app_instance.include_router(chat_router, prefix="/v1")

    # Backward-compatibility alias for unversioned clients (hidden from Swagger docs)
    app_instance.include_router(chat_router, include_in_schema=False)

    # ── OPENAI / OLLAMA COMPATIBILITY ROUTES (DEV-ONLY) ───────────────────────────
    # These routes exist specifically to support unauthenticated model discovery and
    # chat probing from local development tools (e.g. OpenWebUI, Chatbot UI, LibreChat)
    # without requiring CareOpsRx JWT tokens or facility session credentials.
    # In production, they are strictly excluded to prevent unauthenticated LLM cost/abuse
    # exposure, ensuring all production traffic routes through authenticated /chat endpoints.
    if env != "production":
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


