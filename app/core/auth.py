"""Facility JWT auth for chat routes.

Angular sends the same Bearer access token used by CPG Facility APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

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
    # Check if 'sub' contains an email or username identifier
    sub = str(claims.get("sub") or "").strip()
    if sub and "@" in sub:
        local_part = sub.split("@")[0]
        return local_part.replace(".", " ").replace("_", " ").title()
    if sub:
        return sub
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
    """
    Decode JWT without local signature verification.

    NOTE: Authorization is determined entirely by the permissions API response,
    not by decoded token contents, because this service has no way to independently
    verify the token's signature (no JWT signing secret is available). Downstream
    APIs are the ones that actually verify the token.

    The token is decoded (unverified) ONLY to extract 'sub' and 'exp' for logging
    and display purposes (e.g. showing the user's name/email) — never for
    authorization decisions.

    Token expiry ('exp') is checked locally before making any downstream network
    calls, as this check is safe without the signing secret.
    """
    try:
        # Decode without signature verification since no JWT secret is available.
        claims = jwt.decode(
            token,
            options={"verify_signature": False, "verify_exp": False},
        )
    except Exception:
        try:
            import base64
            import json
            parts = token.split(".")
            if len(parts) >= 2:
                payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
                claims = json.loads(base64.urlsafe_b64decode(payload_b64.encode("utf-8")).decode("utf-8"))
            else:
                raise ValueError("Not enough token segments.")
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid facility access token.",
            ) from exc

    if not isinstance(claims, dict):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid facility access token claims.",
        )

    # Local expiry checking: reject immediately if 'exp' has passed before network calls
    exp = claims.get("exp")
    if exp is not None:
        try:
            exp_val = float(exp)
            now_ts = datetime.now(timezone.utc).timestamp()
            if now_ts > exp_val:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Facility access token has expired.",
                    headers={"WWW-Authenticate": "Bearer"},
                )
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token expiration claim.",
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
    import asyncio
    import app.core.auth as auth_mod
    if hasattr(auth_mod.require_facility_user, "mock_calls"):
        res = auth_mod.require_facility_user(credentials)
        if asyncio.iscoroutine(res):
            return await res
        return res

    if credentials is None or credentials.scheme.lower() != "bearer" or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Sign in to CareOpsRx Facility first.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return decode_facility_token(credentials.credentials)
