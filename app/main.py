import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.core.config import settings


def validate_production_environment() -> None:
    """
    Ensure required environment variables are present in production.
    Refuse to start if any required variable is empty.
    Skip check in development.
    """
    env = (getattr(settings, "environment", None) or os.getenv("ENVIRONMENT", "production")).strip().lower()
    if env == "development":
        return

    required_variables = [
        ("PERMISSIONS_API", getattr(settings, "permissions_api", "") or os.getenv("PERMISSIONS_API", "")),
        ("FACILITY_API", getattr(settings, "facility_api", "") or os.getenv("FACILITY_API", "")),
        ("PRODUCTION_API", getattr(settings, "production_api", "") or os.getenv("PRODUCTION_API", "")),
        ("COMPLIANCE_API", getattr(settings, "compliance_api", "") or os.getenv("COMPLIANCE_API", "")),
        ("ORDER_API", getattr(settings, "order_api", "") or os.getenv("ORDER_API", "")),
        ("NOTIFICATION_HUB_API", getattr(settings, "notification_hub_api", "") or os.getenv("NOTIFICATION_HUB_API", "")),
    ]

    for var_name, var_value in required_variables:
        if not var_value or not str(var_value).strip():
            raise RuntimeError(f"Missing required environment variable: {var_name}")


def get_cors_origins() -> list[str]:
    """
    Lock CORS to FRONTEND_ORIGIN only when ENVIRONMENT="production";
    allow localhost in development.
    """
    env = (getattr(settings, "environment", None) or os.getenv("ENVIRONMENT", "production")).strip().lower()
    if env == "production":
        frontend = (getattr(settings, "frontend_origin", "") or os.getenv("FRONTEND_ORIGIN", "")).strip()
        return [frontend] if frontend else []
    
    dev_origins = ["http://localhost:4200", "http://127.0.0.1:4200", "http://localhost:3000"]
    frontend = (getattr(settings, "frontend_origin", "") or os.getenv("FRONTEND_ORIGIN", "")).strip()
    if frontend and frontend not in dev_origins:
        dev_origins.append(frontend)
    return dev_origins


@asynccontextmanager
async def lifespan(_: FastAPI):
    validate_production_environment()
    yield


app = FastAPI(
    title="CPG AI Backend",
    description="AI backend for CareOpsRx Facility chatbot",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins() or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "cpg-ai-backend",
    }


# Documented industry-standard API prefix: /v1
app.include_router(chat_router, prefix="/v1")

# Backward-compatibility alias for unversioned clients (hidden from Swagger docs)
app.include_router(chat_router, include_in_schema=False)


# ── OPENAI / OLLAMA COMPATIBILITY ROUTES ──────────────────────────────────────
# Fixes 404s when frontend or chat clients (e.g. OpenWebUI, Chatbot UI, LibreChat)
# probe the backend for available models.
@app.get("/v1/models")
@app.get("/models")
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


@app.post("/v1/chat/completions")
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


