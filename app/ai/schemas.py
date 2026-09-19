"""Predefined OpenAI function tool schemas for the CPG AI Assistant.

Only business-level capabilities are exposed to the LLM (18 explicit tools).
Raw database queries and arbitrary HTTP execution are strictly prohibited.
"""

from __future__ import annotations

PREDEFINED_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_batch_by_id",
            "description": "Fetch a single batch record by its batch ID (e.g. B-1021).",
            "parameters": {
                "type": "object",
                "properties": {
                    "batch_id": {"type": "string", "description": "The unique batch identifier (e.g. B-1021)"}
                },
                "required": ["batch_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_batches",
            "description": "Retrieve recent batch records, optionally filtered by status (e.g. IN_PROGRESS, RELEASED).",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "description": "Batch status to filter by"},
                    "limit": {"type": "integer", "description": "Maximum records to return (max 50)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_operator_training_status",
            "description": "Retrieve training and qualification records for an operator (e.g. OP-017).",
            "parameters": {
                "type": "object",
                "properties": {
                    "operator_id": {"type": "string", "description": "The operator ID (e.g. OP-017)"}
                },
                "required": ["operator_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_equipment_status",
            "description": "Retrieve PM, calibration, and status for equipment (e.g. EQ-102).",
            "parameters": {
                "type": "object",
                "properties": {
                    "equipment_id": {"type": "string", "description": "The equipment asset ID (e.g. EQ-102)"}
                },
                "required": ["equipment_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_material_lot_trace",
            "description": "Trace a chemical or component lot through batches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lot_number": {"type": "string", "description": "The lot number or chemical code"}
                },
                "required": ["lot_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_deviation_by_id",
            "description": "Retrieve deviation and CAPA details by deviation ID (e.g. DEV-445).",
            "parameters": {
                "type": "object",
                "properties": {
                    "deviation_id": {"type": "string", "description": "The deviation ID"}
                },
                "required": ["deviation_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_environmental_monitoring",
            "description": "Retrieve environmental monitoring excursions and readings for a location or room.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location_id": {"type": "string", "description": "Location or room identifier (e.g. R-204)"}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_production_dashboard_data",
            "description": "Retrieve current production dashboard data (KPIs, active batches, low inventory).",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "Optional user identifier"}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_compliance_dashboard_data",
            "description": "Retrieve current compliance dashboard data (out-of-compliance counts, overdue PMs, expired qualifications).",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "Optional user identifier"}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "investigate_entity",
            "description": "Universal entry point to investigate ANY entity (operator, equipment, material, component, deviation, finished_drug, location, product) and obtain full traceability chain.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_type": {
                        "type": "string",
                        "description": "Type of starting entity: 'batch', 'product', 'material_lot', 'component', 'operator', 'equipment', 'location', 'em', 'pm', 'deviation', 'oos'",
                    },
                    "entity_id": {"type": "string", "description": "The identifier of the starting entity"},
                    "date_range": {"type": "string", "description": "Optional date range filter (e.g. '2026-01-01 to 2026-08-31')"},
                },
                "required": ["entity_type", "entity_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_related_batches",
            "description": "Find all batches related to a specific entity (operator, equipment, material, component, location, deviation, product).",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_type": {"type": "string", "description": "The entity type (e.g. operator, equipment, material, location)"},
                    "entity_id": {"type": "string", "description": "The entity identifier"},
                },
                "required": ["entity_type", "entity_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_material_lot_genealogy",
            "description": "Traverse forward genealogy from chemical/material lot -> batches -> finished drugs, labeling confirmed vs potential impact.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lot_id": {"type": "string", "description": "The material lot number or identifier"}
                },
                "required": ["lot_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_component_lot_genealogy",
            "description": "Traverse packaging component genealogy: component_lot -> batches -> finished drugs, labeling confirmed vs potential impact.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lot_id": {"type": "string", "description": "The component lot number (e.g. COMP-501)"}
                },
                "required": ["lot_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_operator_batch_history",
            "description": "Retrieve batches executed by an operator, equipment handled, date-aware qualifications, and deviations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operator_id": {"type": "string", "description": "The operator ID (e.g. OP-017)"}
                },
                "required": ["operator_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_equipment_batch_history",
            "description": "Retrieve batches processed on equipment, PM/calibration history, and deviations linked to asset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "equipment_id": {"type": "string", "description": "The equipment ID (e.g. EQ-102)"}
                },
                "required": ["equipment_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_finished_drug_genealogy",
            "description": "Reverse traceability from finished drug ID/lot -> batch -> input raw material and component lots.",
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_id_or_lot": {"type": "string", "description": "The finished drug identifier or lot number"}
                },
                "required": ["drug_id_or_lot"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_deviation_impact",
            "description": "Evaluate deviation/OOS impact: deviation -> directly affected batches (confirmed) + co-manufactured batches (potential).",
            "parameters": {
                "type": "object",
                "properties": {
                    "deviation_id": {"type": "string", "description": "The deviation ID (e.g. DEV-445)"}
                },
                "required": ["deviation_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_similar_batches",
            "description": "Find historically similar batches comparing product, formulation, size, materials, operators, equipment, and location, returning similarity percentage and top explaining factors (SRS FR-006).",
            "parameters": {
                "type": "object",
                "properties": {
                    "batch_id": {"type": "string", "description": "Target batch identifier to compare against historical batches"},
                    "top_n": {"type": "integer", "description": "Number of top similar batches to return (default 5)"},
                },
                "required": ["batch_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_risk_score",
            "description": "Calculate a batch risk score (0-100) and risk level using approved, configurable risk weights (SRS Section 19).",
            "parameters": {
                "type": "object",
                "properties": {
                    "batch_id": {"type": "string", "description": "Batch ID to evaluate for risk"},
                    "proposed_config": {"type": "object", "description": "Optional proposed batch run configuration parameters"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_trends",
            "description": "Detect statistical trends and anomalies (yield drift, recurring EM events, equipment deviations). Enforces non-causal reporting per SRS Section 13.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string", "description": "The product identifier to analyze"},
                    "metric": {"type": "string", "description": "Metric to analyze (yield, em_events, equipment_deviations)"},
                    "window": {"type": "string", "description": "Time window (e.g. '90d', '6m', '1y')"},
                },
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_batch_configuration",
            "description": "Generate an advisory-only batch creation recommendation (never creates or approves a batch record). Applies hard eligibility constraints before historical ranking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string", "description": "The product identifier or name to formulate (e.g. Aspirin 500mg)"},
                    "target_quantity": {"type": "number", "description": "Target quantity/size in kg or units (default 1000.0)"},
                    "target_date": {"type": "string", "description": "Optional target production date"},
                    "target_location": {"type": "string", "description": "Optional target suite or cleanroom"},
                },
                "required": ["product_id"],
            },
        },
    },
]

