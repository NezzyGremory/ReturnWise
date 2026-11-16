"""Business-calendar helpers for timestamps stored in UTC."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import os
from zoneinfo import ZoneInfo


TIMEZONE_NAME = os.environ.get("SMART_RETAIL_TIMEZONE", "Asia/Jakarta")
BUSINESS_TIMEZONE = ZoneInfo(TIMEZONE_NAME)


def business_today() -> date:
    return datetime.now(BUSINESS_TIMEZONE).date()


def business_date(value: str | datetime) -> date:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(BUSINESS_TIMEZONE).date()


def utc_date_bounds(start: date | str | None, end: date | str | None) -> tuple[str | None, str | None]:
    start_day = date.fromisoformat(start) if isinstance(start, str) else start
    end_day = date.fromisoformat(end) if isinstance(end, str) else end
    lower = (
        datetime.combine(start_day, time.min, BUSINESS_TIMEZONE)
        .astimezone(timezone.utc).isoformat(timespec="seconds")
        if start_day else None
    )
    upper = (
        datetime.combine(end_day + timedelta(days=1), time.min, BUSINESS_TIMEZONE)
        .astimezone(timezone.utc).isoformat(timespec="seconds")
        if end_day else None
    )
    return lower, upper
