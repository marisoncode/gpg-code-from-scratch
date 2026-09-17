from app.services.capabilities import (
    ChatPermissions,
    ModulePermission,
    allowed_scopes,
    has_module_view,
    normalize_lens,
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


def test_normalize_lens_fail_closed() -> None:
    # Invalid or missing inputs must fail closed to "none"
    assert normalize_lens(None) == "none"
    assert normalize_lens("") == "none"
    assert normalize_lens("   ") == "none"
    assert normalize_lens("bogus-value") == "none"
    assert normalize_lens("admin") == "none"
    assert normalize_lens("superuser") == "none"

    # Recognized values (trimmed, case-insensitive)
    assert normalize_lens("  All  ") == "all"
    assert normalize_lens("PRODUCTION") == "production"
    assert normalize_lens("  compliance  ") == "compliance"
    assert normalize_lens("sales") == "sales"
    assert normalize_lens("none") == "none"


def test_has_module_view_fail_closed() -> None:
    # Empty or missing modules list must return False (fail closed)
    perms_empty_list = ChatPermissions(Dashboard_assign="All", modules=[])
    assert has_module_view(perms_empty_list, "Compliance") is False

    perms_no_modules_field = ChatPermissions(Dashboard_assign="All")
    assert has_module_view(perms_no_modules_field, "Batch Record") is False

    assert has_module_view(None, "Batch Record") is False

    # Module present with View=True -> True (unchanged)
    perms_view_true = ChatPermissions(
        Dashboard_assign="All",
        modules=[ModulePermission(Module_name="Batch Record", View=True)],
    )
    assert has_module_view(perms_view_true, "Batch Record") is True

    # Module present with View=False -> False (unchanged)
    perms_view_false = ChatPermissions(
        Dashboard_assign="All",
        modules=[ModulePermission(Module_name="Batch Record", View=False)],
    )
    assert has_module_view(perms_view_false, "Batch Record") is False


def test_all_lens_fails_closed_when_modules_empty() -> None:
    # Under fail-closed rules, empty modules grants no scopes even if lens is "All"
    perms_empty = ChatPermissions(Dashboard_assign="All", modules=[])
    assert allowed_scopes(perms_empty) == set()

    # With explicit View permissions, "All" grants both scopes
    perms_with_views = ChatPermissions(
        Dashboard_assign="All",
        modules=[
            ModulePermission(Module_name="Batch Record", View=True),
            ModulePermission(Module_name="Compliance", View=True),
        ],
    )
    assert allowed_scopes(perms_with_views) == {"production", "compliance"}


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
