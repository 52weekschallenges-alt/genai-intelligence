"""Canonical article/event model shared by all collectors."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_datetime(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(value, tz=timezone.utc)
    else:
        raw = str(value).strip()
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            try:
                from email.utils import parsedate_to_datetime
                dt = parsedate_to_datetime(raw)
            except (TypeError, ValueError, OverflowError):
                return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass(slots=True)
class Article:
    """A normalized source item. IDs are stable hashes of source identity."""
    source_id: str
    source_name: str
    title: str
    url: str
    item_id: str | None = None
    summary: str | None = None
    content: str | None = None
    author: str | None = None
    published_at: datetime | None = None
    updated_at: datetime | None = None
    categories: list[str] = field(default_factory=list)
    trust_score: float = 0.5
    language: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    collected_at: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("published_at", "updated_at", "collected_at"):
            if data[key] is not None:
                data[key] = data[key].isoformat()
        return data
