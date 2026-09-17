"""Contract tests verifying tool mappings against real CPG Swagger specifications.

Validates:
1. Every tool's downstream route exists in the downloaded Swagger OpenAPI definition.
2. URL builder guarantees the /api/ prefix without duplicates.
3. C# PascalCase response models are transparently normalized to pythonic fields.
4. Response envelope unwrapping (e.g. {"result": ...}) functions properly.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from app.services.facility_api_service import (
    _build_api_url,
    normalize_cpg_record,
    unwrap_cpg_response,
)


def load_swagger(filename: str) -> dict:
    path = Path("d:/cpg") / filename
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"paths": {}}


@pytest.fixture(scope="module")
def production_swagger() -> dict:
    return load_swagger("production_api_swagger.json")


@pytest.fixture(scope="module")
def compliance_swagger() -> dict:
    return load_swagger("compliance_api_swagger.json")


@pytest.fixture(scope="module")
def facility_swagger() -> dict:
    return load_swagger("facility_user_api_swagger.json")


@pytest.fixture(scope="module")
def order_swagger() -> dict:
    return load_swagger("order_api_swagger.json")


def test_build_api_url_normalization() -> None:
    """_build_api_url must ensure /api/ is cleanly formed regardless of trailing/leading slashes."""
    assert _build_api_url("https://prod.cpg.com", "BatchRecord") == "https://prod.cpg.com/api/BatchRecord"
    assert _build_api_url("https://prod.cpg.com/", "/BatchRecord") == "https://prod.cpg.com/api/BatchRecord"
    assert _build_api_url("https://facility.cpg.com/api/", "DashboardProduction") == "https://facility.cpg.com/api/DashboardProduction"
    assert _build_api_url("https://compliance.cpg.com/api", "/api/Training") == "https://compliance.cpg.com/api/Training"


def test_production_api_routes_exist_in_swagger(production_swagger: dict) -> None:
    """Production tools map to verified paths in Production API Swagger."""
    paths = production_swagger.get("paths", {})
    if not paths:
        pytest.skip("Swagger file not found on disk")

    assert "/api/BatchRecord" in paths
    assert "/api/BatchRecord/{id}" in paths
    assert "/api/ComponentChildInventory" in paths
    assert "/api/ChemicalChildInventory" in paths


def test_compliance_api_routes_exist_in_swagger(compliance_swagger: dict) -> None:
    """Compliance tools map to verified paths in Compliance API Swagger."""
    paths = compliance_swagger.get("paths", {})
    if not paths:
        pytest.skip("Swagger file not found on disk")

    assert "/api/Training" in paths
    assert "/api/Equipment" in paths
    assert "/api/EnvironmentalMonitoring" in paths
    assert "/api/v1/TaskManagement" in paths
    assert "/api/v1/TaskManagement/{id}" in paths


def test_facility_user_api_routes_exist_in_swagger(facility_swagger: dict) -> None:
    """Facility and Dashboard tools map to verified paths in Facility User API Swagger."""
    paths = facility_swagger.get("paths", {})
    if not paths:
        pytest.skip("Swagger file not found on disk")

    assert "/api/DashboardProduction/GetProductionDetails" in paths
    assert "/api/DashboardMonitor/GetOutOfComplianceDetails/{kind}" in paths


def test_cpg_record_normalization() -> None:
    """normalize_cpg_record preserves PascalCase and maps standard lowercase aliases."""
    raw = {
        "Lot_number": "B-1021",
        "MFR_name": "Aspirin 500mg",
        "Status": "IN_PROGRESS",
        "Id": "guid-1234",
    }
    norm = normalize_cpg_record(raw)

    assert norm["Lot_number"] == "B-1021"
    assert norm["batch_number"] == "B-1021"
    assert norm["lot_number"] == "B-1021"
    assert norm["product_name"] == "Aspirin 500mg"
    assert norm["status"] == "IN_PROGRESS"
    assert norm["id"] == "guid-1234"


def test_unwrap_cpg_response_envelope() -> None:
    """unwrap_cpg_response extracts and normalizes the 'result' key."""
    envelope = {
        "result": [
            {"Lot_number": "LOT-99", "Status": "Released"}
        ],
        "count": 1
    }
    unwrapped = unwrap_cpg_response(envelope)
    assert isinstance(unwrapped, list)
    assert len(unwrapped) == 1
    assert unwrapped[0]["batch_number"] == "LOT-99"
    assert unwrapped[0]["status"] == "Released"

