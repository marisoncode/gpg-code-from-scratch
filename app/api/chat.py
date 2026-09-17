from __future__ import annotations

from datetime import datetime
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from app.core.auth import AuthenticatedUser, require_facility_user
from app.core.config import settings
from app.services import audit_service, chat_store
from app.services.ai_service import (
    AiAuthError,
    AiConfigError,
    AiProviderError,
    AiRateLimitError,
    generate_response,
)
from app.services.capabilities import (
    ChatPermissions,
    allowed_scopes,
    normalize_lens,
    refuse_message,
    requested_scopes,
    wants_dashboard_data,
)
from app.services.dashboard_service import fetch_dashboard_snapshot, format_snapshot_for_prompt
from app.services.facility_api_service import set_current_token, set_current_user
from app.services.permissions_service import PermissionDeniedError, resolve_permissions
from app.services.welcome import build_welcome_message

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)
    username: str | None = None
    thread_id: str | None = None
    permissions: ChatPermissions | None = None
    # Login user id/name — required by facility APIs (UserId / UserName headers)
    facility_user_id: str | None = None
    facility_user_name: str | None = None


class StoredMessage(BaseModel):
    id: str
    thread_id: str
    role: str
    content: str
    created_at: datetime | None = None


class ChatTurn(BaseModel):
    id: str
    request: str
    response: str
    created_at: datetime | None = None


class ChatResponse(BaseModel):
    message: str
    thread_id: str
    messages: list[StoredMessage] = Field(default_factory=list)


class WelcomeResponse(BaseModel):
    message: str
    username: str
    authenticated: bool = True
    # Empty until the user sends the first message (thread created lazily).
    thread_id: str | None = None
    messages: list[StoredMessage] = Field(default_factory=list)


class ThreadSummary(BaseModel):
    id: str
    user_id: str
    username: str
    title: str
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ThreadDetail(ThreadSummary):
    greeting: str = ""
    turns: list[ChatTurn] = Field(default_factory=list)
    messages: list[StoredMessage] = Field(default_factory=list)


class CreateThreadRequest(BaseModel):
    username: str | None = None
    title: str | None = None


class CreateThreadResponse(BaseModel):
    id: str
    title: str


class DashboardSummaryRequest(BaseModel):
    message: str | None = None
    permissions: ChatPermissions | None = None
    facility_user_id: str | None = None
    facility_user_name: str | None = None


class DashboardSummaryResponse(BaseModel):
    allowed: list[str]
    refused: str = ""
    snapshot: dict = Field(default_factory=dict)


def _facility_identity(
    user: AuthenticatedUser,
    *,
    facility_user_id: str | None = None,
    facility_user_name: str | None = None,
) -> tuple[str, str]:
    """
    Resolve identity for request execution.

    SECURITY BOUNDARY:
    Authorization decisions and permission resolution MUST ALWAYS use the verified,
    token-derived identity (user.user_id).
    Client-supplied `facility_user_id` MUST NEVER be used for authorization, permission
    checks, or resource gating to prevent privilege escalation / IDOR spoofing attacks.
    If `facility_user_id` is supplied and differs from `user.user_id`, we log a security
    warning and ignore it for authorization decisions.

    `facility_user_name` is only allowed as a display name override for non-auth UI greetings.
    """
    token_user_id = (user.user_id or "").strip()
    client_user_id = (facility_user_id or "").strip()

    if client_user_id and token_user_id and client_user_id != token_user_id:
        logger.warning(
            "Security notice: client-supplied facility_user_id '%s' differs from token user_id '%s'. "
            "Ignoring client-supplied ID for authorization decisions.",
            client_user_id,
            token_user_id,
        )

    # Strictly use verified token identity for authorization/user identity
    authoritative_user_id = token_user_id
    display_name = (facility_user_name or "").strip() or (user.name or "").strip()
    return authoritative_user_id, display_name


def _user_key(user: AuthenticatedUser) -> str:
    return (user.user_id or "").strip() or f"name:{(user.name or 'anonymous').strip().lower()}"


def _to_stored(messages: list[dict]) -> list[StoredMessage]:
    return [StoredMessage.model_validate(m) for m in messages]


@router.get("/welcome", response_model=WelcomeResponse)
async def welcome(
    username: str | None = None,
    dashboard_assign: str | None = None,
    user: AuthenticatedUser = Depends(require_facility_user),
):
    """
    Fresh chat page greeting only — does NOT create a Mongo thread.
    Thread is created on the first user message via POST /chat.
    """
    display = (username or "").strip() or user.name
    _ = _user_key(user)
    greeting = build_welcome_message(display, lens=dashboard_assign)
    return WelcomeResponse(
        message=greeting,
        username=display,
        authenticated=True,
        thread_id=None,
        messages=[],
    )


@router.get("/threads", response_model=list[ThreadSummary])
async def list_threads(
    limit: int = Query(20, ge=1, le=100),
    user: AuthenticatedUser = Depends(require_facility_user),
):
    rows = await chat_store.list_threads(user_id=_user_key(user), limit=limit)
    # Only conversations that have at least one user request
    active = [
        r
        for r in rows
        if (r.get("title") or "").strip() and (r.get("title") or "").strip() != "Support chat"
    ]
    return [ThreadSummary.model_validate(r) for r in active]


@router.post("/threads", response_model=CreateThreadResponse, status_code=status.HTTP_201_CREATED)
async def create_thread(
    body: CreateThreadRequest | None = None,
    user: AuthenticatedUser = Depends(require_facility_user),
):
    display = ((body.username if body else None) or "").strip() or user.name
    title = (body.title if body else None) or None
    thread = await chat_store.create_thread(
        user_id=_user_key(user),
        username=display,
        title=title,
    )
    return CreateThreadResponse(id=thread["id"], title=thread["title"])


@router.get("/threads/{thread_id}", response_model=ThreadDetail)
async def get_thread(
    thread_id: str,
    user: AuthenticatedUser = Depends(require_facility_user),
):
    thread = await chat_store.get_thread(thread_id=thread_id, user_id=_user_key(user))
    return ThreadDetail.model_validate(thread)


@router.delete("/threads/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_thread(
    thread_id: str,
    user: AuthenticatedUser = Depends(require_facility_user),
):
    await chat_store.delete_thread(thread_id=thread_id, user_id=_user_key(user))


async def _load_dashboard_context(
    *,
    message: str,
    permissions: ChatPermissions | None,
    user: AuthenticatedUser,
    facility_user_id: str | None = None,
    facility_user_name: str | None = None,
) -> tuple[str | None, str]:
    """Returns (direct_reply, extra_context). direct_reply skips the LLM."""
    if not wants_dashboard_data(message):
        return None, ""

    requested = requested_scopes(message)
    allowed = allowed_scopes(permissions)
    scopes = requested & allowed
    lens = normalize_lens(permissions.Dashboard_assign if permissions else None)
    denied_note = refuse_message(requested, allowed, lens)

    if not scopes:
        return (
            denied_note
            or "Your role cannot access Production or Compliance dashboard data.",
            "",
        )

    actor_id, actor_name = _facility_identity(
        user,
        facility_user_id=facility_user_id,
        facility_user_name=facility_user_name,
    )
    try:
        snapshot = await fetch_dashboard_snapshot(
            token=user.token,
            user_id=user.user_id,
            username=actor_name,
            scopes=scopes,
            permissions=permissions,
        )
    except (HTTPException, PermissionDeniedError) as exc:
        denial = exc.detail if isinstance(exc, HTTPException) else getattr(exc, "message", str(exc))
        return (
            denial
            or denied_note
            or "Your role cannot access Production or Compliance dashboard data.",
            "",
        )
    context = format_snapshot_for_prompt(snapshot)
    if denied_note:
        context = f"{denied_note}\n\n{context}"
    return None, context


@router.post("/dashboard-summary", response_model=DashboardSummaryResponse)
async def dashboard_summary(
    request: DashboardSummaryRequest,
    user: AuthenticatedUser = Depends(require_facility_user),
):
    set_current_token(user.token)
    actor_id, actor_name = _facility_identity(
        user,
        facility_user_id=request.facility_user_id,
        facility_user_name=request.facility_user_name,
    )
    # Context identity and downstream permissions MUST strictly use token-derived identity
    set_current_user(user_id=user.user_id, username=actor_name or user.name)
    # Resolve authoritative permissions directly from CPG ManageUser API using verified token identity
    effective_permissions = await resolve_permissions(user.token, user_id=user.user_id)
    message = (request.message or "dashboard summary").strip()
    requested = requested_scopes(message)
    allowed = allowed_scopes(effective_permissions)
    scopes = requested & allowed
    lens = normalize_lens(effective_permissions.Dashboard_assign)
    refused = refuse_message(requested, allowed, lens)
    snapshot: dict = {}
    if scopes:
        snapshot = await fetch_dashboard_snapshot(
            token=user.token,
            user_id=user.user_id,
            username=actor_name,
            scopes=scopes,
            permissions=effective_permissions,
        )
    return DashboardSummaryResponse(
        allowed=sorted(scopes),
        refused=refused,
        snapshot=snapshot,
    )


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user: AuthenticatedUser = Depends(require_facility_user),
):
    set_current_token(user.token)
    display = (request.username or "").strip() or user.name
    user_id = _user_key(user)
    actor_id, actor_name = _facility_identity(
        user,
        facility_user_id=request.facility_user_id,
        facility_user_name=request.facility_user_name or display,
    )
    # Context identity: use token-derived user_id for security
    set_current_user(user_id=user.user_id, username=actor_name or display)

    # SECURITY BOUNDARY:
    # Resolve authoritative permissions from CPG ManageUser / Permissions API using raw forwarded token.
    # Never trust client-supplied permissions, facility_user_id, or unverified claims for authorization.
    effective_permissions = await resolve_permissions(user.token, user_id=user.user_id)

    thread_id = (request.thread_id or "").strip() or None
    history: list[dict[str, str]] = []

    if thread_id:
        await chat_store.get_thread(thread_id=thread_id, user_id=user_id)
        history = await chat_store.history_for_model(thread_id=thread_id, user_id=user_id)

    if not history and request.history:
        history = [{"role": m.role, "content": m.content} for m in request.history]

    extra_context = ""
    functions_called: list[dict] = []
    record_ids: list[str] = []
    permission_denied = False
    denial_reason = None
    entity_type = ""
    traceability_chain = ""
    risk_config_version = ""
    data_source = ""
    role = effective_permissions.Dashboard_assign or "User"

    try:
        direct, extra_context = await _load_dashboard_context(
            message=request.message,
            permissions=effective_permissions,
            user=user,
            facility_user_id=request.facility_user_id,
            facility_user_name=request.facility_user_name or display,
        )
        if direct:
            response_text = direct
            if "cannot access" in direct or "not available" in direct:
                permission_denied = True
                denial_reason = direct
        else:
            gen_result = await generate_response(
                request.message,
                username=display,
                history=history,
                extra_context=extra_context or None,
                permissions=effective_permissions,
                token=user.token,
            )
            response_text = str(gen_result)
            if hasattr(gen_result, "functions_called"):
                functions_called = gen_result.functions_called
            if hasattr(gen_result, "retrieved_record_ids"):
                record_ids = gen_result.retrieved_record_ids
            if getattr(gen_result, "permission_denied", False):
                permission_denied = True
                denial_reason = getattr(gen_result, "denial_reason", None)
            entity_type = getattr(gen_result, "entity_type", "") or ""
            traceability_chain = getattr(gen_result, "traceability_chain", "") or ""
            risk_config_version = getattr(gen_result, "risk_config_version", "") or ""
            data_source = getattr(gen_result, "data_source", "") or ""

        # Audit log for successful / completed response
        await audit_service.log_audit_entry(
            user_id=user_id,
            role=str(role),
            raw_query=request.message,
            functions_called=functions_called,
            retrieved_record_ids=record_ids,
            model_version=settings.ai_model,
            final_response=response_text,
            permission_denied=permission_denied,
            denial_reason=denial_reason,
            entity_type=entity_type,
            traceability_chain=traceability_chain,
            risk_config_version=risk_config_version,
            data_source=data_source,
        )

    except (HTTPException, PermissionDeniedError) as exc:
        if isinstance(exc, HTTPException) and exc.status_code != status.HTTP_403_FORBIDDEN:
            raise
        denial_text = exc.detail if isinstance(exc, HTTPException) else getattr(exc, "message", str(exc))
        if not denial_text or not str(denial_text).strip():
            denial_text = "Your role cannot access this resource. Ask an admin for permission."
        response_text = str(denial_text)
        permission_denied = True
        denial_reason = response_text
        await audit_service.log_audit_entry(
            user_id=user_id,
            role=str(role),
            raw_query=request.message,
            functions_called=functions_called,
            retrieved_record_ids=record_ids,
            model_version=settings.ai_model,
            final_response=response_text,
            permission_denied=permission_denied,
            denial_reason=denial_reason,
            entity_type=entity_type,
            traceability_chain=traceability_chain,
            risk_config_version=risk_config_version,
        )
    except (AiAuthError, AiConfigError) as exc:
        await audit_service.log_audit_entry(
            user_id=user_id,
            role=str(role),
            raw_query=request.message,
            functions_called=functions_called,
            retrieved_record_ids=record_ids,
            model_version=settings.ai_model,
            final_response=f"Failed: {exc}",
            permission_denied=permission_denied,
            denial_reason=str(exc),
            entity_type=entity_type,
            traceability_chain=traceability_chain,
            risk_config_version=risk_config_version,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc)
            or "Invalid AI API key. Check OPENAI_API_KEY / AI_PROVIDER in cpg-ai-backend/.env and restart uvicorn.",
        ) from exc
    except AiRateLimitError as exc:
        await audit_service.log_audit_entry(
            user_id=user_id,
            role=str(role),
            raw_query=request.message,
            functions_called=functions_called,
            retrieved_record_ids=record_ids,
            model_version=settings.ai_model,
            final_response=f"Rate limited: {exc}",
            permission_denied=permission_denied,
            entity_type=entity_type,
            traceability_chain=traceability_chain,
            risk_config_version=risk_config_version,
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc),
        ) from exc
    except AiProviderError as exc:
        await audit_service.log_audit_entry(
            user_id=user_id,
            role=str(role),
            raw_query=request.message,
            functions_called=functions_called,
            retrieved_record_ids=record_ids,
            model_version=settings.ai_model,
            final_response=f"Provider error: {exc}",
            permission_denied=permission_denied,
            entity_type=entity_type,
            traceability_chain=traceability_chain,
            risk_config_version=risk_config_version,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        await audit_service.log_audit_entry(
            user_id=user_id,
            role=str(role),
            raw_query=request.message,
            functions_called=functions_called,
            retrieved_record_ids=record_ids,
            model_version=settings.ai_model,
            final_response=f"Internal error: {exc}",
            permission_denied=permission_denied,
            entity_type=entity_type,
            traceability_chain=traceability_chain,
            risk_config_version=risk_config_version,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat failed: {exc}",
        ) from exc

    # Create thread in Cosmos DB only after a successful AI reply (first user request).
    if not thread_id:
        thread = await chat_store.create_thread(
            user_id=user_id,
            username=display,
            title=request.message.strip()[:80] or "Support chat",
        )
        thread_id = thread["id"]

    updated = await chat_store.append_turn(
        thread_id=thread_id,
        user_id=user_id,
        request=request.message,
        response=response_text,
    )

    return ChatResponse(
        message=response_text,
        thread_id=thread_id,
        messages=_to_stored(updated.get("messages") or []),
    )

