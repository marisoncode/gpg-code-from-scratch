# CPG Guardian: End-to-End System Architecture, Data Flow & Ownership Model

---

## 1. Executive Summary & Core Paradigm

CPG Guardian is a multi-tenant, microservice-driven ERP/MES platform designed for regulated consumer packaged goods (cosmetics, pharmaceuticals, nutraceuticals, food & beverage) complying with **21 CFR Part 11**.

The system revolves around three core primitives:
1. **Identity & Tenant Partitioning**: Managed via Cosmos DB collection routing (`collectionid`), JWT authentication, and user GUID tracking.
2. **Master Recipes to Real-World Execution**: Master Formulation Records (MFRs) are scheduled and instantiated into executed **Batch Records**.
3. **Traceability & Attribution**: Every physical action (chemical addition, mixing, quality testing, packaging) is cryptographically or identity-linked to an operator, location, equipment, and lot number.

```mermaid
flowchart TD
    User([User / Operator / Admin]) -->|Login| Identity[Facility & Identity API]
    Identity -->|JWT + User GUID| ClientApp[Web UI / AI Assistant]
    ClientApp -->|HTTP with CPG Headers| Microservices[CPG Microservices Gateway]
    Microservices --> BatchAPI[Production API: /api/BatchRecord]
    Microservices --> InventoryAPI[Inventory API: /api/ChemicalChildInventory]
    Microservices --> ComplianceAPI[Compliance API: /api/Deviation]
    Microservices --> FacilityAPI[Facility API: /api/UserRole]
    
    BatchAPI --> CosmosDB[(Cosmos DB: Tenant Collection)]
    InventoryAPI --> CosmosDB
    ComplianceAPI --> CosmosDB
    FacilityAPI --> CosmosDB
```

---

## 2. Authentication, Identity & Request Headers

### 2.1 The Two Identifiers: Email vs. GUID
In CPG Guardian:
- **`sub` (Email)**: Used in JWT authentication token for login credentials (e.g., `sladdy@ldnworldwide.com`).
- **`userid` (Database GUID)**: The unique permanent database key for the user record (e.g., `b1d62178-3d29-4b0a-9c55-706568754e66`).

> [!IMPORTANT]
> CPG microservices (**Production API**, **Inventory API**, etc.) validate access against the **GUID**, not the email. If the GUID header is missing or incorrect, queries return empty sets or unauthorized errors.

### 2.2 Mandatory Microservice Request Headers
Every downstream HTTP request from the Web Frontend or AI Backend must include:

| Header Name | Type | Example Value | Description |
| :--- | :--- | :--- | :--- |
| `authorization` | String | `Bearer eyJhbGciOi...` | Standard Bearer JWT token |
| `collectionid` | String | `sales` | Active tenant identifier (lowercased Cosmos DB container) |
| `userid` | UUID | `b1d62178-3d29-4b0a-9c55-706568754e66` | User's permanent database GUID |
| `username` | String | `Stephen Ladd` | Display name of the active user |
| `currentdate` | ISO Date | `2026-09-19T07:22:24.986Z` | Client timestamp for audit trail alignment |

---

## 3. Role-Based Access Control (RBAC) & Dashboard Lenses

Permissions are defined in the `UserRole` service (`https://facility-user-api.cpguardian.com/api/UserRole`).
Each user is assigned a primary role with a **`Dashboard_assign`** lens and module permission flags (`View`, `Add`, `Edit`, `Delete`).

### 3.1 The 4 Dashboard Lenses

```mermaid
graph TD
    AdminLens["All / Admin Lens"] -->|Full Access| ProductionMod[Batch Records & MFR]
    AdminLens -->|Full Access| InventoryMod[Raw Chemicals & Finished Goods]
    AdminLens -->|Full Access| ComplianceMod[Deviations & CAPA]
    AdminLens -->|Full Access| SalesMod[Orders & Shipping]

    ProdLens["Production Lens"] -->|Full Access| ProductionMod
    ProdLens -->|Dispense & View| InventoryMod
    ProdLens -.->|Read Only / Flag| ComplianceMod
    ProdLens x-.-x|Blocked| SalesMod

    CompLens["Compliance / QA Lens"] -->|Release / Quarantine| ProductionMod
    CompLens -->|Audit Quality| InventoryMod
    CompLens -->|Full Access| ComplianceMod
    CompLens x-.-x|Blocked| SalesMod

    SalesLens["Sales Lens"] x-.-x|Blocked| ProductionMod
    SalesLens -->|Finished Goods Only| InventoryMod
    SalesLens x-.-x|Blocked| ComplianceMod
    SalesLens -->|Full Access| SalesMod
```

### 3.2 Detailed Lens Access Matrix

| Module / Asset | All / Admin | Production Lens | Compliance / QA Lens | Sales Lens |
| :--- | :---: | :---: | :---: | :---: |
| **MFR (Master Recipes)** | Full (`CRUD`) | Full (`CRUD`) | Read / Approve (`R`) | ❌ No Access |
| **Batch Records** | Full (`CRUD`) | Full (`CRU`) | Review / Release / Hold (`RU`) | ❌ No Access |
| **Raw Chemicals** | Full (`CRUD`) | Dispense / Weigh (`RU`) | Audit / Inspect (`R`) | ❌ No Access |
| **Equipment & Rooms** | Full (`CRUD`) | Use / Log (`RU`) | Calibration / Cleanliness (`RU`) | ❌ No Access |
| **Deviations & CAPA** | Full (`CRUD`) | Create Initial Report (`C`) | Investigate / Close (`CRUD`) | ❌ No Access |
| **Finished Goods** | Full (`CRUD`) | Package / Label (`CR`) | Release / Hold (`RU`) | Read / Reserve (`RU`) |
| **Customer Orders** | Full (`CRUD`) | ❌ No Access | ❌ No Access | Full (`CRUD`) |

---

## 4. Batch Record Architecture & Lifecycle

### 4.1 What is a Batch Record?
A **Batch Record** (`/api/BatchRecord`) is the digital record of executing a Master Formulation Record (MFR) to produce a specific quantity of finished or bulk product under GMP conditions.

### 4.2 Batch Record Schema Structure
```json
{
  "id": "7fa12b84-4e91-4c12-8901-ab87321f9812",
  "batch_number": "LOT-2026-09-001",
  "mfr_id": "mfr-hand-cream-v2",
  "product_name": "Hydrating Hand Cream 50ml",
  "status": "InProcess",
  "created_info": {
    "created_by": "b1d62178-3d29-4b0a-9c55-706568754e66",
    "created_name": "Stephen Ladd",
    "created_date": "2026-09-19T06:30:00Z"
  },
  "operator_id": "b1d62178-3d29-4b0a-9c55-706568754e66",
  "operator_name": "Stephen Ladd",
  "equipment_ids": [
    "TANK-MIXER-03",
    "FILLER-LINE-B"
  ],
  "location_id": "CLEANROOM-SUITE-2",
  "scheduled_start": "2026-09-19T07:00:00Z",
  "completed_date": null,
  "chemical_components": [
    {
      "inventory_child_id": "chem-lot-glyc-9921",
      "chemical_name": "Glycerin USP",
      "lot_number": "GLYC-2026-441",
      "quantity_dispensed": 25.4,
      "unit": "kg",
      "dispensed_by": "b1d62178-3d29-4b0a-9c55-706568754e66",
      "dispensed_date": "2026-09-19T07:15:22Z"
    }
  ]
}
```

### 4.3 Lifecycle State Machine

```mermaid
stateDiagram-v2
    [*] --> Scheduled: Created from MFR by Scheduler
    Scheduled --> InProcess: Operator starts run & dispenses lots
    InProcess --> Quarantine: Run complete; samples sent to QC
    Quarantine --> Released: QA approves & digitally signs
    Quarantine --> Rejected: QA rejects; Deviation CAPA triggered
    Released --> [*]
    Rejected --> [*]
```

---

## 5. Inventory & Chemical Lineage (Traceability)

Every batch connects upstream raw materials to downstream finished goods:

```mermaid
flowchart LR
    subgraph Upstream["Raw Materials"]
        C1["Chemical Lot: Glycerin #GLYC-441"]
        C2["Chemical Lot: Purified Water #PW-102"]
        C3["Packaging: 50ml Jars #JAR-88"]
    end

    subgraph Batch["Batch Manufacturing"]
        B["Batch Record #LOT-2026-09-001\n(Creator: Stephen Ladd)\n(Operator: Stephen Ladd)\n(Mixer-03 @ Suite-2)"]
    end

    subgraph Downstream["Finished Goods & Distribution"]
        FG["Finished Product Lot #LOT-2026-09-001\n(5,000 units Hand Cream)"]
        O1["Customer Order #PO-9001"]
        O2["Customer Order #PO-9002"]
    end

    C1 -->|Weighed & Added| B
    C2 -->|Dispensed| B
    C3 -->|Packaged| B
    B -->|Transferred| FG
    FG -->|Shipped| O1
    FG -->|Shipped| O2
```

### 5.1 Bi-Directional Traceability
- **Backward Geneaology (Root Cause Analysis)**:
  Given finished lot `#LOT-2026-09-001`, inspect the batch record to identify every raw chemical lot, machine used, and technician who logged in.
- **Forward Geneaology (Recall Prevention)**:
  Given contaminated raw chemical lot `#GLYC-441`, query all batch records containing that child inventory ID to locate all finished goods before shipment.

---

## 6. User Attribution: "Who Owns What?"

A critical question is: **How does the system know how many batches belong to a particular user?**

There are two primary attribution relationships:

```mermaid
classDiagram
    class User {
        +UUID id ("b1d62178...")
        +string email ("sladdy@ldn...")
        +string full_name ("Stephen Ladd")
        +string role ("Admin")
    }

    class BatchRecord {
        +UUID id
        +string lot_number
        +string status
        +UUID created_by
        +UUID operator_id
    }

    User "1" --> "0..*" BatchRecord : Schedules / Creates (created_by)
    User "1" --> "0..*" BatchRecord : Runs / Operates (operator_id)
```

### 6.1 Batches Created by a User
- **Field**: `created_info.created_by`
- **Definition**: The planner, manager, or lead who created or scheduled the batch record.
- **Query Filter**:
  ```python
  user_created_batches = [
      b for b in all_batches 
      if b.get("created_info", {}).get("created_by") == user_guid
  ]
  ```

### 6.2 Batches Operated by a User
- **Field**: `operator_id` or step digital signature logs
- **Definition**: The technician who performed compounding, mixing, or filling.
- **Query Endpoint**:
  ```http
  GET /v1/genealogy/operator/{user_guid}
  ```
- **Returns**: Complete historical list of all batch runs and step-level actions performed by that technician.

---

## 7. How the AI Agent Handles User Queries

When a user interacts with the AI Assistant, here is the exact sequence of how questions are processed:

```mermaid
sequenceDiagram
    autonumber
    actor User as User (Stephen Ladd)
    participant UI as Web / Chat UI
    participant Agent as AI Agent (cpg-ai-backend)
    participant Auth as Session & Token Store
    participant ProdAPI as Production Microservice

    User->>UI: "How many batches belong to me?"
    UI->>Agent: POST /api/chat { message, token }
    Agent->>Auth: Decode Token (JWT)
    Auth-->>Agent: user_id: "b1d62178...", collectionid: "sales", role: "Admin"
    
    Agent->>Agent: Determine Intent -> Get User Batches
    Agent->>ProdAPI: GET /api/BatchRecord?page=1&pageSize=50<br/>Headers: { collectionid: "sales", userid: "b1d62178...", username: "Stephen Ladd" }
    ProdAPI-->>Agent: 200 OK (List of Batch Records)
    
    Agent->>Agent: Filter batches where created_by == "b1d62178..." OR operator_id == "b1d62178..."
    Agent->>Agent: Group by status (Released, InProcess, Quarantine)
    
    Agent-->>UI: "You have 10 batches associated with your account:<br/>- 4 InProcess<br/>- 4 Released<br/>- 2 Quarantine<br/>Here are the latest 3..."
    UI-->>User: Rendered Response
```

---

## 8. CPG Microservice Endpoints Reference

| Service | Base URL | Relevant Endpoints | Key Functionality |
| :--- | :--- | :--- | :--- |
| **Production API** | `https://production-api.cpguardian.com/api` | `GET /BatchRecord`<br/>`GET /BatchRecord/{id}`<br/>`POST /BatchRecord`<br/>`GET /MasterFormulationRecord` | Batch scheduling, step tracking, formula execution |
| **Facility API** | `https://facility-user-api.cpguardian.com/api` | `GET /UserRole`<br/>`GET /UserRole/{id}`<br/>`GET /ManageUser`<br/>`GET /ManageUser/{id}` | Users, GUIDs, roles, dashboard lenses, permissions |
| **Inventory API** | `https://inventory-api.cpguardian.com/api` | `GET /ChemicalChildInventory`<br/>`GET /InventoryLots`<br/>`POST /Dispense` | Raw chemical lot tracking, BOM dispensing, quantities |
| **Compliance API**| `https://compliance-api.cpguardian.com/api` | `GET /Deviation`<br/>`POST /Deviation`<br/>`GET /AuditTrail` | Quality incidents, CAPAs, 21 CFR Part 11 audit records |
| **AI Backend** | `http://localhost:8000` / `/v1` | `POST /v1/chat/message`<br/>`GET /v1/genealogy/operator/{id}`<br/>`GET /v1/recommendations/batch-combo` | AI agents, lot recall, MFR combo suggestions, analytics |

---

## 9. Key Takeaways

1. **You are never blind to user identity**: Every request carries the exact tenant (`collectionid`) and user GUID (`userid`).
2. **Every batch has clear attribution**: You can always differentiate between who scheduled a batch (`created_by`) and who compounded it (`operator_id`).
3. **Traceability is end-to-end**: From incoming chemical vendor lots $\rightarrow$ batch mixing $\rightarrow$ finished good distribution.
4. **Permissions are deterministic**: Defined by `UserRole` $\rightarrow$ `Dashboard_assign` (All, Production, Compliance, Sales).

