from datetime import datetime

from app.services.capabilities import DashboardLens, normalize_lens

_LENS_LINE = {
    "production": "I can summarize your Production dashboard.",
    "compliance": "I can summarize your Compliance dashboard.",
    "all": "I can summarize Production and Compliance dashboards.",
    "sales": "Your role is assigned to Sales, so Production/Compliance dashboard summaries are not available.",
}


def build_welcome_message(username: str, lens: str | None = None) -> str:
    """Time-of-day welcome, e.g. 'Good evening John, welcome!'"""
    name = (username or "there").strip() or "there"
    hour = datetime.now().hour

    if 5 <= hour < 12:
        period = "morning"
    elif 12 <= hour < 17:
        period = "afternoon"
    elif 17 <= hour < 22:
        period = "evening"
    else:
        period = "evening"

    greeting = f"Good {period} {name}, welcome!"
    normalized: DashboardLens | None = None
    if lens and lens.strip():
        normalized = normalize_lens(lens)
    extra = _LENS_LINE.get(normalized or "all", _LENS_LINE["all"])
    return f"{greeting} {extra}"
