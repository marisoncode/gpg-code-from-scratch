"""AI Capability Registry for CPG AI Assistant.

Provides an authoritative, declarative registry of all business tools and capabilities.
Every tool must be registered with:
- Required resource permission (enforced in Python at application level)
- Target downstream endpoint pattern
- Allowed HTTP method (strictly GET for read operations)
- Execution handler function
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from app.ai import tools


@dataclass(frozen=True)
class AICapability:
    """Metadata and execution definition for an authorized AI capability."""

    name: str
    description: str
    required_resource: str
    permission_label: str
    target_endpoint: str
    http_method: str
    handler: Callable[..., Any]


CAPABILITY_REGISTRY: dict[str, AICapability] = {
    "get_batch_by_id": AICapability(
        name="get_batch_by_id",
        description="Fetch a single batch record by ID or Lot Number.",
        required_resource="batch",
        permission_label="batch.read",
        target_endpoint="BatchRecord/{id}",
        http_method="GET",
        handler=tools.get_batch_by_id,
    ),
    "get_batches": AICapability(
        name="get_batches",
        description="Retrieve batch records, optionally filtered by status.",
        required_resource="batch",
        permission_label="batch.read",
        target_endpoint="BatchRecord",
        http_method="GET",
        handler=tools.get_batches,
    ),
    "get_operator_training_status": AICapability(
        name="get_operator_training_status",
        description="Retrieve operator qualifications from COMPLIANCE_API.",
        required_resource="training",
        permission_label="training.read",
        target_endpoint="v1/TaskManagement",
        http_method="GET",
        handler=tools.get_operator_training_status,
    ),
    "get_equipment_status": AICapability(
        name="get_equipment_status",
        description="Retrieve equipment asset details and PM calibration from COMPLIANCE_API.",
        required_resource="equipment",
        permission_label="equipment.read",
        target_endpoint="Equipment/{id}",
        http_method="GET",
        handler=tools.get_equipment_status,
    ),
    "get_material_lot_trace": AICapability(
        name="get_material_lot_trace",
        description="Trace a chemical or component lot through batches via PRODUCTION_API.",
        required_resource="material",
        permission_label="material.read",
        target_endpoint="ChemicalChildInventory/{lot}",
        http_method="GET",
        handler=tools.get_material_lot_trace,
    ),
    "get_deviation_by_id": AICapability(
        name="get_deviation_by_id",
        description="Fetch deviation records and OOC details.",
        required_resource="deviation",
        permission_label="deviation.read",
        target_endpoint="DashboardMonitor/GetOOCDetails",
        http_method="GET",
        handler=tools.get_deviation_by_id,
    ),
    "get_environmental_monitoring": AICapability(
        name="get_environmental_monitoring",
        description="Retrieve environmental monitoring (EM) excursions.",
        required_resource="environmental_monitoring",
        permission_label="environmental.read",
        target_endpoint="DashboardMonitor/GetOOCDetails",
        http_method="GET",
        handler=tools.get_environmental_monitoring,
    ),
    "get_production_dashboard_data": AICapability(
        name="get_production_dashboard_data",
        description="Retrieve live production KPIs and active batch metrics.",
        required_resource="production",
        permission_label="dashboard.production.read",
        target_endpoint="DashboardProduction/GetProductionDetails",
        http_method="GET",
        handler=tools.get_production_dashboard_data,
    ),
    "get_compliance_dashboard_data": AICapability(
        name="get_compliance_dashboard_data",
        description="Retrieve live compliance KPIs, deviations, and PM calibration status.",
        required_resource="compliance",
        permission_label="dashboard.compliance.read",
        target_endpoint="DashboardMonitor/GetOOCDetails",
        http_method="GET",
        handler=tools.get_compliance_dashboard_data,
    ),
    "get_related_batches": AICapability(
        name="get_related_batches",
        description="Find all batches related to any entity type (operator, equipment, material, deviation).",
        required_resource="batch",
        permission_label="batch.read",
        target_endpoint="BatchRecord",
        http_method="GET",
        handler=tools.get_related_batches,
    ),
    "get_material_lot_genealogy": AICapability(
        name="get_material_lot_genealogy",
        description="Traverse forward genealogy from chemical/material lot -> batches -> finished drugs.",
        required_resource="material",
        permission_label="genealogy.material.read",
        target_endpoint="BatchRecord",
        http_method="GET",
        handler=tools.get_material_lot_genealogy,
    ),
    "get_component_lot_genealogy": AICapability(
        name="get_component_lot_genealogy",
        description="Traverse packaging component genealogy: component_lot -> batches -> finished drugs.",
        required_resource="material",
        permission_label="genealogy.component.read",
        target_endpoint="BatchRecord",
        http_method="GET",
        handler=tools.get_component_lot_genealogy,
    ),
    "get_operator_batch_history": AICapability(
        name="get_operator_batch_history",
        description="Retrieve operator history: batches executed, equipment handled, qualifications.",
        required_resource="training",
        permission_label="operator.history.read",
        target_endpoint="BatchRecord",
        http_method="GET",
        handler=tools.get_operator_batch_history,
    ),
    "get_equipment_batch_history": AICapability(
        name="get_equipment_batch_history",
        description="Retrieve equipment history: batches processed on asset, PM status, deviations.",
        required_resource="equipment",
        permission_label="equipment.history.read",
        target_endpoint="BatchRecord",
        http_method="GET",
        handler=tools.get_equipment_batch_history,
    ),
    "get_finished_drug_genealogy": AICapability(
        name="get_finished_drug_genealogy",
        description="Reverse traceability chain: finished_drug -> batch -> input_lots.",
        required_resource="batch",
        permission_label="genealogy.drug.read",
        target_endpoint="BatchRecord",
        http_method="GET",
        handler=tools.get_finished_drug_genealogy,
    ),
    "get_deviation_impact": AICapability(
        name="get_deviation_impact",
        description="Forward impact analysis: deviation -> affected batches.",
        required_resource="deviation",
        permission_label="deviation.impact.read",
        target_endpoint="DashboardMonitor/GetOOCDetails",
        http_method="GET",
        handler=tools.get_deviation_impact,
    ),
    "investigate_entity": AICapability(
        name="investigate_entity",
        description="Universal multi-entity investigation entry point.",
        required_resource="batch",
        permission_label="investigation.read",
        target_endpoint="BatchRecord",
        http_method="GET",
        handler=tools.investigate_entity,
    ),
    "find_similar_batches": AICapability(
        name="find_similar_batches",
        description="Multi-dimensional batch similarity comparison (SRS FR-006).",
        required_resource="batch",
        permission_label="analytics.similarity.read",
        target_endpoint="BatchRecord",
        http_method="GET",
        handler=tools.find_similar_batches,
    ),
    "calculate_risk_score": AICapability(
        name="calculate_risk_score",
        description="Configurable risk scoring based on versioned weights (SRS Section 19).",
        required_resource="batch",
        permission_label="analytics.risk.read",
        target_endpoint="BatchRecord",
        http_method="GET",
        handler=tools.calculate_risk_score,
    ),
    "detect_trends": AICapability(
        name="detect_trends",
        description="Trend & anomaly detection with non-causal language (SRS Section 13).",
        required_resource="batch",
        permission_label="analytics.trends.read",
        target_endpoint="BatchRecord",
        http_method="GET",
        handler=tools.detect_trends,
    ),
    "recommend_batch_configuration": AICapability(
        name="recommend_batch_configuration",
        description="GAMP-5 batch configuration recommendation with hard precedence (SRS Section 7).",
        required_resource="batch",
        permission_label="recommendation.batch.read",
        target_endpoint="BatchRecord",
        http_method="GET",
        handler=tools.recommend_batch_configuration,
    ),
}


def get_capability(name: str) -> AICapability | None:
    """Retrieve capability definition by name."""
    return CAPABILITY_REGISTRY.get((name or "").strip())


def is_capability_registered(name: str) -> bool:
    """Check if capability name is strictly registered."""
    return (name or "").strip() in CAPABILITY_REGISTRY


def list_capabilities() -> list[dict[str, Any]]:
    """Return serialized summary of all registered capabilities for discovery."""
    return [
        {
            "name": cap.name,
            "description": cap.description,
            "required_resource": cap.required_resource,
            "permission": cap.permission_label,
            "endpoint": cap.target_endpoint,
            "method": cap.http_method,
        }
        for cap in CAPABILITY_REGISTRY.values()
    ]

