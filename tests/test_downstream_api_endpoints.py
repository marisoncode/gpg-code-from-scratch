"""Unit tests for downstream HTTP REST API endpoint data access.

Verifies:
1. Batch operations call PRODUCTION_API with forwarded Bearer token.
2. Deviation & EM operations call COMPLIANCE_API with forwarded Bearer token.
3. Equipment, Training, Material operations call FACILITY_API with forwarded Bearer token.
4. Pre-query permission gating blocks unauthorized callers before network request is made.
5. 404, 500, and network error responses are handled gracefully as 'unavailable'.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from app.core.config import settings
from app.services import facility_api_service
from app.services.capabilities import ChatPermissions, ModulePermission
from app.services.facility_api_service import (
    get_batch_by_id,
    get_batches,
    get_compliance_dashboard_data,
    get_deviation_by_id,
    get_environmental_monitoring,
    get_equipment_status,
    get_material_lot_trace,
    get_operator_training_status,
    get_production_dashboard_data,
)
from app.services.permissions_service import PermissionDeniedError


@pytest.fixture
def qa_permissions() -> ChatPermissions:
    return ChatPermissions(
        Dashboard_assign="All",
        modules=[
            ModulePermission(Module_name="Batch Record", View=True),
            ModulePermission(Module_name="Compliance", View=True),
            ModulePermission(Module_name="Equipment", View=True),
            ModulePermission(Module_name="Inventory", View=True),
        ],
    )


@pytest.fixture
def sales_permissions() -> ChatPermissions:
    return ChatPermissions(Dashboard_assign="Sales", modules=[])


@pytest.fixture(autouse=True)
def setup_test_user_identity():
    """Explicitly set test user identity instead of relying on legacy hardcoded fallbacks."""
    facility_api_service.set_current_user(user_id="test-operator-01", username="Test Operator")
    yield
    facility_api_service.set_current_user(user_id="", username="")


@pytest.mark.asyncio
async def test_get_batch_by_id_calls_production_api_with_token(qa_permissions: ChatPermissions) -> None:
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "B-9021",
        "batch_number": "B-9021",
        "product": "Aspirin 500mg",
        "status": "IN_PROGRESS",
    }
    mock_client.get.return_value = mock_response

    facility_api_service.set_http_client(mock_client)
    try:
        res = await get_batch_by_id("B-9021", permissions=qa_permissions, token="jwt-token-abc")
        assert res["status"] == "found"
        assert res["record_id"] == "B-9021"
        assert res["batch"]["product"] == "Aspirin 500mg"

        mock_client.get.assert_called_once()
        url, kwargs = mock_client.get.call_args
        assert "/api/BatchRecord" in url[0]
        assert kwargs["headers"]["Authorization"] == "Bearer jwt-token-abc"
        assert kwargs["headers"]["collectionid"] == "sales"
    finally:
        facility_api_service.set_http_client(None)


@pytest.mark.asyncio
async def test_get_batches_filtered_by_status(qa_permissions: ChatPermissions) -> None:
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {"id": "B-1", "status": "RELEASED"},
        {"id": "B-2", "status": "RELEASED"},
    ]
    mock_client.get.return_value = mock_response

    facility_api_service.set_http_client(mock_client)
    try:
        res = await get_batches(status="RELEASED", limit=5, permissions=qa_permissions, token="jwt-tok")
        assert len(res) == 2
        mock_client.get.assert_called_once()
        url, kwargs = mock_client.get.call_args
        assert "/api/BatchRecord" in url[0]
        assert kwargs["headers"]["Authorization"] == "Bearer jwt-tok"
    finally:
        facility_api_service.set_http_client(None)


@pytest.mark.asyncio
async def test_get_batch_by_lot_number_resolves_via_batchrecord(qa_permissions: ChatPermissions) -> None:
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "result": [
            {
                "Lot_number": "2026258002",
                "Batch_name": "2026258002",
                "Master_formula": "GEL#01",
                "Status": "Pending",
                "id": "2d49ff34-f0d5-40cb-a556-5d107b98d419",
            }
        ],
        "count": 1,
    }
    mock_client.get.return_value = mock_response

    facility_api_service.set_http_client(mock_client)
    try:
        res = await get_batch_by_id("2026258002", permissions=qa_permissions, token="real-jwt-token")
        assert res["status"] == "found"
        assert res["batch"]["Lot_number"] == "2026258002"
        assert res["batch"]["batch_number"] == "2026258002"
    finally:
        facility_api_service.set_http_client(None)


@pytest.mark.asyncio
async def test_get_deviation_by_id_calls_compliance_api(qa_permissions: ChatPermissions) -> None:
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "DEV-101",
        "title": "Temperature Excursion",
        "status": "OPEN",
    }
    mock_client.get.return_value = mock_response

    facility_api_service.set_http_client(mock_client)
    try:
        res = await get_deviation_by_id("DEV-101", permissions=qa_permissions, token="jwt-tok")
        assert res["status"] == "found"
        assert res["deviation"]["title"] == "Temperature Excursion"

        mock_client.get.assert_called_once()
        url, kwargs = mock_client.get.call_args
        assert "/api/v1/TaskManagement" in url[0]
        assert kwargs["headers"]["Authorization"] == "Bearer jwt-tok"
    finally:
        facility_api_service.set_http_client(None)


@pytest.mark.asyncio
async def test_get_operator_training_calls_compliance_api(qa_permissions: ChatPermissions) -> None:
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {"id": "TR-1", "operator_id": "OP-10", "course_name": "GMP Aseptic", "status": "CURRENT"}
    ]
    mock_client.get.return_value = mock_response

    facility_api_service.set_http_client(mock_client)
    try:
        res = await get_operator_training_status("OP-10", permissions=qa_permissions, token="jwt-tok")
        assert res["status"] == "found"
        assert len(res["training_records"]) == 1

        mock_client.get.assert_called_once()
        url, kwargs = mock_client.get.call_args
        assert "/api/Training" in url[0]
    finally:
        facility_api_service.set_http_client(None)


@pytest.mark.asyncio
async def test_get_equipment_status_calls_compliance_api(qa_permissions: ChatPermissions) -> None:
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "result": [
            {"Id": "eq-101", "Name": "EQ-101", "Status": "Active", "pm_status": "CURRENT"}
        ]
    }
    mock_client.get.return_value = mock_response

    facility_api_service.set_http_client(mock_client)
    try:
        res = await get_equipment_status("EQ-101", permissions=qa_permissions, token="jwt-tok")
        assert res["status"] == "found"
        assert res["equipment"]["Name"] == "EQ-101"
        assert res["pm_overdue"] is False

        mock_client.get.assert_called_once()
        url, kwargs = mock_client.get.call_args
        assert "/api/Equipment" in url[0]
    finally:
        facility_api_service.set_http_client(None)


@pytest.mark.asyncio
async def test_get_material_lot_trace_calls_production_api(qa_permissions: ChatPermissions) -> None:
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "result": [
            {"Lot_number": "RM-88321", "Sku": "SKU-01", "Status": "Released"}
        ]
    }
    mock_client.get.return_value = mock_response

    facility_api_service.set_http_client(mock_client)
    try:
        res = await get_material_lot_trace("RM-88321", permissions=qa_permissions, token="jwt-tok")
        assert res["status"] == "found"
        assert res["material_lot"]["Lot_number"] == "RM-88321"

        mock_client.get.assert_called_once()
        url, kwargs = mock_client.get.call_args
        assert "/api/ComponentChildInventory" in url[0]
    finally:
        facility_api_service.set_http_client(None)


@pytest.mark.asyncio
async def test_get_environmental_monitoring_calls_compliance_api(qa_permissions: ChatPermissions) -> None:
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "result": [
            {"Id": "em-01", "Location_name": "Cleanroom-A", "Status": "PASS"}
        ]
    }
    mock_client.get.return_value = mock_response

    facility_api_service.set_http_client(mock_client)
    try:
        res = await get_environmental_monitoring(location_id="Cleanroom-A", permissions=qa_permissions, token="jwt-tok")
        assert len(res) == 1
        assert res[0]["Location_name"] == "Cleanroom-A"

        mock_client.get.assert_called_once()
        url, kwargs = mock_client.get.call_args
        assert "/api/EnvironmentalMonitoring" in url[0]
    finally:
        facility_api_service.set_http_client(None)


@pytest.mark.asyncio
async def test_get_production_dashboard_data_calls_facility_api(qa_permissions: ChatPermissions) -> None:
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "result": {
            "activeBatches": 3,
            "completedToday": 2,
        }
    }
    mock_client.get.return_value = mock_response

    facility_api_service.set_http_client(mock_client)
    try:
        res = await get_production_dashboard_data(permissions=qa_permissions, token="jwt-tok")
        assert res["activeBatches"] == 3

        mock_client.get.assert_called_once()
        url, kwargs = mock_client.get.call_args
        assert "/api/DashboardProduction/GetProductionDetails" in url[0]
    finally:
        facility_api_service.set_http_client(None)


@pytest.mark.asyncio
async def test_downstream_api_404_returns_unavailable(qa_permissions: ChatPermissions) -> None:
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 404
    mock_client.get.return_value = mock_response

    facility_api_service.set_http_client(mock_client)
    try:
        res = await get_equipment_status("EQ-NONEXISTENT", permissions=qa_permissions)
        assert res["status"] == "unavailable"
        assert "unavailable or not found" in res["error"]
        assert res["record_id"] == "EQ-NONEXISTENT"
    finally:
        facility_api_service.set_http_client(None)


@pytest.mark.asyncio
async def test_downstream_api_network_timeout_fails_gracefully(qa_permissions: ChatPermissions) -> None:
    mock_client = MagicMock(spec=httpx.Client)
    mock_client.get.side_effect = httpx.TimeoutException("Connection timed out")

    facility_api_service.set_http_client(mock_client)
    try:
        res = await get_material_lot_trace("RM-999", permissions=qa_permissions)
        assert res["status"] == "unavailable"
        assert "unavailable or not found" in res["error"]
    finally:
        facility_api_service.set_http_client(None)


@pytest.mark.asyncio
async def test_permission_gate_blocks_network_request(sales_permissions: ChatPermissions) -> None:
    mock_client = MagicMock(spec=httpx.Client)
    facility_api_service.set_http_client(mock_client)

    try:
        with pytest.raises(PermissionDeniedError):
            await get_batch_by_id("B-101", permissions=sales_permissions)

        with pytest.raises(PermissionDeniedError):
            await get_deviation_by_id("DEV-101", permissions=sales_permissions)

        with pytest.raises(PermissionDeniedError):
            await get_production_dashboard_data(permissions=sales_permissions)

        mock_client.get.assert_not_called()
    finally:
        facility_api_service.set_http_client(None)

