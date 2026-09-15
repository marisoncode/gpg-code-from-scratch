"""Facility JWT auth for chat routes.

Angular sends the same Bearer access token used by CPG Facility APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings

_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str
    name: str
    raw_claims: dict[str, Any]
    token: str = ""


def _pick_name(claims: dict[str, Any]) -> str:
    for key in ("name", "userName", "UserName", "unique_name", "preferred_username", "email"):
        value = claims.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "there"


def _pick_id(claims: dict[str, Any]) -> str:
    for key in (
        "userId",
        "UserId",
        "User_id",
        "user_id",
        "nameid",
        "sub",
        "id",
        "oid",
        "uid",
    ):
        value = claims.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def decode_facility_token(token: str) -> AuthenticatedUser:
    try:
        if settings.jwt_secret:
            claims = jwt.decode(
                token,
                settings.jwt_secret,
                algorithms=[a.strip() for a in settings.jwt_algorithms.split(",") if a.strip()],
            )
        else:
            # Dev mode: accept structurally valid JWTs without signature verify.
            claims = jwt.decode(
                token,
                options={"verify_signature": False, "verify_exp": False},
            )
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid facility access token.",
        ) from exc

    if not isinstance(claims, dict):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid facility access token claims.",
        )

    return AuthenticatedUser(
        user_id=_pick_id(claims),
        name=_pick_name(claims),
        raw_claims=claims,
        token=token,
    )


async def require_facility_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> AuthenticatedUser:
    if credentials is None or credentials.scheme.lower() != "bearer" or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Sign in to CareOpsRx Facility first.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return decode_facility_token(credentials.credentials)
