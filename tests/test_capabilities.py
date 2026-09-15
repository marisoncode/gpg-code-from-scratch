from app.services.capabilities import (
    ChatPermissions,
    ModulePermission,
    allowed_scopes,
    requested_scopes,
    resolve_scopes,
    wants_dashboard_data,
)


def test_wants_dashboard_data() -> None:
    assert wants_dashboard_data("Give me a production dashboard summary.")
    assert wants_dashboard_data("What production alerts should I look at?")
    assert wants_dashboard_data("compliance dashboard summary")
    assert not wants_dashboard_data("How do I create a user role?")


def test_requested_scopes() -> None:
    assert requested_scopes("production dashboard summary") == {"production"}
    assert requested_scopes("compliance dashboard summary") == {"compliance"}
    assert requested_scopes("both production and compliance dashboards") == {
        "production",
        "compliance",
    }


def test_sales_lens_blocks_both() -> None:
    perms = ChatPermissions(Dashboard_assign="Sales")
    assert allowed_scopes(perms) == set()
    assert resolve_scopes("dashboard summary", perms) == set()


def test_all_lens_allows_both_when_modules_empty() -> None:
    perms = ChatPermissions(Dashboard_assign="All")
    assert allowed_scopes(perms) == {"production", "compliance"}


def test_module_view_gate() -> None:
    perms = ChatPermissions(
        Dashboard_assign="All",
        modules=[
            ModulePermission(Module_name="Batch Record", View=True),
            ModulePermission(Module_name="Compliance", View=False),
        ],
    )
    assert resolve_scopes("dashboard summary", perms) == {"production"}
    assert resolve_scopes("compliance dashboard summary", perms) == set()
