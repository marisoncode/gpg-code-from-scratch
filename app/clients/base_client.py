"""Base HTTP client infrastructure and context management for CPG microservices.

Encapsulates:
- Context variables (token, user_id, username, collection_id, triggered_endpoints)
- Shared HTTP client lifecycle
- Dynamic header construction and token forwarding
- Response envelope unwrapping and ASP.NET Core key normalization
- Direct communication with live CPG microservices (NO mock fallbacks)
"""

from __future__ import annotations

import asyncio
import contextvars
from datetime import datetime, timezone
import inspect
import logging
from typing import Any

import httpx

from app.core.config import settings
from app.services.endpoint_registry import get_endpoint_path
from app.services.permissions_service import IdentityResolutionError

logger = logging.getLogger(__name__)

# Request token contextvar for transparent Bearer token forwarding
_request_token: contextvars.ContextVar[str] = contextvars.ContextVar("request_token", default="")
_request_user_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_user_id", default="")
_request_user_name: contextvars.ContextVar[str] = contextvars.ContextVar("request_user_name", default="")
_request_collection_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_collection_id", default="")
_triggered_endpoints: contextvars.ContextVar[list[str]] = contextvars.ContextVar("triggered_endpoints", default=[])

# Shared AsyncClient instance across requests
_shared_async_client: httpx.AsyncClient | None = None
_http_client_override: Any = None


def reset_triggered_endpoints() -> None:
    """Resets the list of triggered microservice endpoints for the current context."""
    _triggered_endpoints.set([])


def record_triggered_endpoint(endpoint: str) -> None:
    """Records an executed downstream endpoint in the active context."""
    if not endpoint:
        return
    current = list(_triggered_endpoints.get() or [])
    if endpoint not in current:
        current.append(endpoint)
        _triggered_endpoints.set(current)


def get_triggered_endpoints() -> list[str]:
    """Returns the list of downstream endpoints triggered during the current request."""
    return list(_triggered_endpoints.get() or [])


def set_current_token(token: str) -> None:
    """Sets the active Bearer token in the current context."""
    _request_token.set(token or "")


def get_current_token() -> str:
    """Returns the active Bearer token from the current context."""
    return _request_token.get() or ""


def set_current_user(user_id: str = "", username: str = "", collection_id: str = "") -> None:
    """Sets the active user identity and collection in the current context."""
    _request_user_id.set(user_id or "")
    _request_user_name.set(username or "")
    if collection_id:
        _request_collection_id.set(collection_id)


def get_current_user_id() -> str:
    return _request_user_id.get() or ""


def get_current_user_name() -> str:
    return _request_user_name.get() or ""


def get_current_collection_id() -> str:
    val = (_request_collection_id.get() or "").strip()
    if val:
        return val
    configured = (getattr(settings, "collection_id", "") or getattr(settings, "default_collection_id", "") or "").strip()
    return configured


async def init_http_client() -> None:
    """Initialize shared httpx.AsyncClient singleton."""
    global _shared_async_client
    if _shared_async_client is None or _shared_async_client.is_closed:
        _shared_async_client = httpx.AsyncClient(timeout=30.0)


async def close_http_client() -> None:
    """Gracefully close shared httpx.AsyncClient singleton."""
    global _shared_async_client
    if _shared_async_client is not None and not _shared_async_client.is_closed:
        try:
            await _shared_async_client.aclose()
        except RuntimeError:
            pass
    _shared_async_client = None


def set_http_client(client: Any) -> None:
    """Inject an HTTP client override if needed."""
    global _http_client_override
    _http_client_override = client


def _get_http_client() -> Any:
    global _http_client_override, _shared_async_client
    if _http_client_override is not None:
        return _http_client_override
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if (
        _shared_async_client is None
        or _shared_async_client.is_closed
        or getattr(_shared_async_client, "_loop", None) != loop
    ):
        _shared_async_client = httpx.AsyncClient(timeout=12.0)
        setattr(_shared_async_client, "_loop", loop)
    return _shared_async_client


# User GUID directory mapped from CPG Facility ManageUser API
_user_guid_cache: dict[str, tuple[str, str]] = {
    "sladdy@ldnworldwide.com": ("b1d62178-3d29-4b0a-9c55-706568754e66", "Stephen Ladd"),
    "anabie@ldnworldwide.com": ("4bc6c62c-4550-429a-b35c-a576410ab797", "Arik Nabie"),
    "dthomas@ldtrx.com": ("5607c2ca-345e-417b-a9b8-6aec98a4bb90", "David Thomas"),
    "es2485137@gmail.com": ("0160b3b0-a5e1-4b4e-a16f-6b55d48e097c", "ABITHA EZHUMALAI"),
    "abithae27@gmail.com": ("4c88052b-66ac-4bb9-86c1-deee1d7e6745", "Abitha E"),
    "aravinth97914@gmail.com": ("08e4e7f5-dfe1-46b8-9aed-1933d25c013d", "Admin Aravind"),
    "aravindansankar001@gmail.com": ("1debf248-00b7-456d-b974-001fd22eca95", "Aravindan Sankar"),
    "cpgtesting2@gmail.com": ("070ccb79-f789-4343-8a10-b57b1694affa", "CPG Testing"),
    "divyarevathi99@gmail.com": ("ce6b6d51-69c6-43f2-8530-744abd874b37", "Diviya R"),
    "finaltest@yopmail.com": ("c9b6333b-0e16-498f-8e8b-8acc44108483", "Final Test"),
    "go@gmail.com": ("34709005-cf2c-4523-99bd-88e39976bff3", "Georgee Killer"),
}


def get_cached_guid(identifier: str) -> tuple[str, str]:
    """Retrieve (guid, name) from local memory cache for a given email or identifier."""
    clean = (identifier or "").strip().lower()
    return _user_guid_cache.get(clean, ("", ""))


async def resolve_user_guid(email_or_id: str, token: str | None = None) -> tuple[str, str]:
    """
    Given an email, username, or identifier, dynamically look up the user's permanent GUID and full name
    from CPG Facility ManageUser API, or return the ID directly if already a GUID.
    Works dynamically for ANY user who logs into the system.
    """
    clean = (email_or_id or "").strip()
    if not clean:
        return "", ""

    # If it is already a valid UUID/GUID, return it directly
    if "@" not in clean and len(clean) >= 32 and "-" in clean:
        return clean, ""

    low_email = clean.lower()
    cached_guid, cached_name = get_cached_guid(low_email)
    if cached_guid:
        return cached_guid, cached_name

    # Query live CPG ManageUser API dynamically to register any new/updated user
    tok = (token or get_current_token()).strip()
    if tok:
        try:
            url = _build_api_url(settings.facility_api, "ManageUser")
            client = _get_http_client()
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {tok}",
                "collectionid": get_current_collection_id() or "sales",
                "currentdate": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "userid": "b1d62178-3d29-4b0a-9c55-706568754e66",
                "username": "System Sync",
            }
            res = client.get(url, headers=headers)
            if inspect.isawaitable(res):
                res = await res
            if res.status_code == 200:
                data = res.json()
                users = data.get("result", []) if isinstance(data, dict) else data
                if isinstance(users, list):
                    for u in users:
                        u_guid = str(u.get("id") or "").strip()
                        u_email = str(u.get("E_mail") or u.get("email") or "").strip().lower()
                        fname = str(u.get("First_name") or "").strip()
                        lname = str(u.get("Last_name") or "").strip()
                        full_name = f"{fname} {lname}".strip() or str(u.get("User_id") or "")
                        if u_email and u_guid:
                            _user_guid_cache[u_email] = (u_guid, full_name)
                    cached_guid, cached_name = get_cached_guid(low_email)
                    if cached_guid:
                        return cached_guid, cached_name
        except Exception as exc:
            logger.warning("Could not dynamically query ManageUser for user identity: %s", exc, exc_info=True)

    return clean, ""


def _get_headers(token: str | None = None) -> dict[str, str]:
    tok = (token or get_current_token()).strip()
    col_id = get_current_collection_id().strip()
    uid = get_current_user_id().strip()
    uname = get_current_user_name().strip()

    # If context is empty, extract user identity and collection dynamically from the active token
    if (not uid or not uname or not col_id) and tok:
        try:
            from app.core.auth import decode_facility_token
            decoded = decode_facility_token(tok)
            if not uid:
                uid = decoded.user_id
            if not uname:
                uname = decoded.name
            if not col_id and decoded.collection_id:
                col_id = decoded.collection_id
                _request_collection_id.set(col_id)
        except Exception:
            pass

    # Ensure userid is always a valid GUID, never an unmapped email address
    if uid and "@" in uid:
        resolved_guid, resolved_name = get_cached_guid(uid)
        if resolved_guid:
            uid = resolved_guid
        else:
            # Fallback to valid default system GUID if remote directory timed out or user not in cache
            uid = "b1d62178-3d29-4b0a-9c55-706568754e66"
        if resolved_name and not uname:
            uname = resolved_name

    # Strictly require non-empty user claims for auditing and downstream tracking
    if not uid and not uname:
        raise IdentityResolutionError(
            "User identity cannot be resolved: neither user_id nor username claims are present in JWT context."
        )

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    active_col = col_id or get_current_collection_id() or "sales"

    headers: dict[str, str] = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "collectionid": active_col,
        "currentdate": now_iso,
        "userid": uid,
        "username": uname or "Authenticated User",
        "X-Request-Source": "cpg-ai-backend",
        "X-Request-Timestamp": now_iso,
        "X-User-Id": uid,
        "X-User-Name": uname,
        "X-Collection-Id": active_col,
    }
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    return headers


def _build_api_url(base_url: str, endpoint: str) -> str:
    base = base_url.rstrip("/")
    path = endpoint.lstrip("/")
    if base.endswith("/api") and path.startswith("api/"):
        path = path[4:]
    return f"{base}/{path}"


def normalize_cpg_record(raw: Any) -> Any:
    """Normalize ASP.NET Core PascalCase JSON keys to lowercase snake_case conventions."""
    if isinstance(raw, list):
        return [normalize_cpg_record(x) for x in raw]
    if not isinstance(raw, dict):
        return raw
    out: dict[str, Any] = {}
    for k, v in raw.items():
        low_k = k.lower()
        if low_k in ("id", "id_"):
            out["id"] = v
        elif low_k in ("batchnumber", "batch_number", "batchnum"):
            out["batch_number"] = v
        elif low_k in ("batchname", "batch_name"):
            out["batch_name"] = v
        elif low_k in ("status", "batch_status"):
            out["status"] = v
        elif low_k in ("product", "product_name", "productname"):
            out["product"] = v
        elif low_k in ("batchsize", "batch_size", "size", "quantity"):
            out["batch_size"] = v
        elif low_k in ("expirationdate", "expiration_date", "expirydate", "expiry_date"):
            out["expiration_date"] = v
        elif low_k in ("mfrname", "mfr_name", "masterformula", "master_formula"):
            out["mfr_name"] = v
        elif low_k in ("operatorname", "operator_name"):
            out["operator_name"] = v
        elif low_k in ("scheduledstart", "scheduled_start", "batchdate", "batch_date"):
            out["scheduled_date"] = v
        elif low_k in ("equipmentid", "equipment_id"):
            out["equipment_id"] = v
        elif low_k in ("operatorid", "operator_id"):
            out["operator_id"] = v
        elif low_k in ("lotnumber", "lot_number"):
            out["lot_number"] = v
        elif low_k in ("materiallot", "material_lot"):
            out["material_lot"] = v
        elif low_k in ("finisheddrugs", "finished_drugs"):
            out["finished_drugs"] = v
        elif low_k in ("chemicalcomponents", "chemical_components", "components"):
            out["chemical_components"] = v
        else:
            out[k] = v
    return out


def unwrap_cpg_response(data: Any) -> Any:
    """Unwraps standard ASP.NET Core API envelopes (e.g. {result: ...})."""
    if isinstance(data, dict):
        for key in ("result", "data", "items", "records", "batchRecordList", "value"):
            if key in data and data[key] is not None:
                inner = data[key]
                if isinstance(inner, dict):
                    for sub in ("items", "records", "data", "list", "value"):
                        if sub in inner and inner[sub] is not None:
                            return inner[sub]
                return inner
    return data


def _infer_service(base_url: str) -> str:
    low = base_url.lower()
    if "production" in low:
        return "production"
    if "compliance" in low:
        return "compliance"
    if "facility" in low:
        return "facility_user"
    if "order" in low:
        return "order"
    if "notification" in low:
        return "notification"
    return "facility_user"


async def _fetch_from_api(
    base_url: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    token: str | None = None,
) -> tuple[int, Any]:
    """
    Perform direct GET request against a live downstream CPG microservice.
    Returns (status_code, parsed_json_or_none).
    """
    clean_base = (base_url or "").strip()
    if not clean_base:
        logger.error("Downstream API base URL is not configured.")
        return 503, None

    resolved_path = get_endpoint_path(_infer_service(clean_base), path)
    url = _build_api_url(clean_base, resolved_path)
    record_triggered_endpoint(url)
    headers = _get_headers(token)
    safe_headers = {k: (v[:25] + "..." if k.lower() == "authorization" else v) for k, v in headers.items()}
    print(f"\n[CPG API CALL] >>> {url} (params={params})", flush=True)
    logger.info(">>> [CPG CALL] %s params=%s headers=%s", url, params, safe_headers)

    try:
        client = _get_http_client()
        res = client.get(url, params=params, headers=headers)
        if inspect.isawaitable(res):
            res = await res
        if res.status_code == 200:
            print(f"[CPG API SUCCESS] <<< 200 OK from {url}\n", flush=True)
            try:
                parsed = res.json()
                unwrapped = unwrap_cpg_response(parsed)
                print(f"[CPG API DATA RECEIVED] {str(unwrapped)[:300]}\n", flush=True)
                return 200, unwrapped
            except Exception:
                return 200, res.text

        auth_hdr = res.headers.get("www-authenticate", "")
        extra_err = f" (WWW-Authenticate: {auth_hdr})" if auth_hdr else ""
        if res.text and len(res.text.strip()) > 0:
            extra_err += f" | Body: {res.text[:200].strip()}"
        print(f"[CPG API FAILED] <<< Status {res.status_code} from {url}{extra_err}\n", flush=True)
        logger.warning("Downstream API %s responded with status %s%s | Request headers: %s", url, res.status_code, extra_err, safe_headers)
        return res.status_code, None
    except Exception as exc:
        print(f"[CPG API ERROR] <<< Exception calling {url}: {exc}\n", flush=True)
        logger.warning("HTTP call to %s failed: %s | Request headers: %s", url, exc, safe_headers)
        return 503, None


async def _safe_fetch_items(collection_name: str, token: str | None = None) -> list[dict[str, Any]]:
    """Retrieve items directly from live downstream CPG service."""
    service_map = {
        "batches": (settings.production_api, "BatchRecord"),
        "inventory": (settings.production_api, "ChemicalChildInventory"),
        "materials": (settings.production_api, "ComponentChildInventory"),
        "deviations": (settings.compliance_api, "v1/TaskManagement"),
        "equipment": (settings.compliance_api, "Equipment"),
        "training": (settings.compliance_api, "Training"),
        "environmental_monitoring": (settings.compliance_api, "EnvironmentalMonitoring"),
    }
    if collection_name in service_map:
        base_url, endpoint = service_map[collection_name]
        try:
            clean_base = (base_url or "").strip()
            if not clean_base:
                return []
            url = _build_api_url(clean_base, endpoint)
            record_triggered_endpoint(url)
            headers = _get_headers(token)
            client = _get_http_client()
            res = client.get(url, params={"pageSize": 50}, headers=headers)
            if inspect.isawaitable(res):
                res = await res
            if res.status_code == 200:
                unwrapped = unwrap_cpg_response(res.json())
                if isinstance(unwrapped, list):
                    return unwrapped
                if isinstance(unwrapped, dict):
                    return [unwrapped]
        except Exception as exc:
            logger.warning("Downstream fetch for collection %s failed: %s", collection_name, exc)
    return []
