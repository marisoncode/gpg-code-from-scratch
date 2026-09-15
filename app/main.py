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
        ("JWT_SECRET", getattr(settings, "jwt_secret", "") or os.getenv("JWT_SECRET", "")),
        ("PERMISSIONS_API", getattr(settings, "permissions_api", "") or os.getenv("PERMISSIONS_API", "")),
        ("COSMOS_DB_READONLY_KEY", getattr(settings, "cosmos_db_readonly_key", "") or os.getenv("COSMOS_DB_READONLY_KEY", "")),
        ("COSMOS_DB_WRITE_KEY", getattr(settings, "cosmos_db_write_key", "") or os.getenv("COSMOS_DB_WRITE_KEY", "")),
        ("MONGODB_URI", os.getenv("MONGODB_URI", "") or getattr(settings, "mongodb_uri", "")),
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


app.include_router(chat_router)

