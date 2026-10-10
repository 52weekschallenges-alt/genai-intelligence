"""Deterministic, explainable selection for the daily briefing."""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from typing import Any


BRIEFING_SIZE = 20
MAX_PER_SOURCE = 3
LOOKBACK_DAYS = 7

_SIGNALS = {
    "release": 1.0,
    "launch": 1.0,
    "announces": 0.8,
    "announced": 0.8,
    "open weights": 1.0,
    "open source": 0.9,
    "benchmark": 0.8,
    "state of the art": 1.0,
    "reasoning": 0.6,
    "agent": 0.5,
    "agents": 0.5,
    "multimodal": 0.5,
    "dataset": 0.5,
    "safety": 0.5,
    "research": 0.4,
    "paper": 0.4,
}
_WORD_RE = re.compile(r"[a-z0-9]+")


def _as_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _json_object(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}


def _normalized_title(value: str) -> str:
    return " ".join(_WORD_RE.findall(value.casefold()))


def _score(item: dict, now: datetime) -> tuple[float, list[str], datetime] | None:
    source_timestamps = [dt for dt in (_as_datetime(item.get("published_at")),
                                       _as_datetime(item.get("updated_at"))) if dt]
    # Collection time is only a fallback; using it for every re-fetched item would
    # make old stories look new on each daily ingestion.
    published = max(source_timestamps) if source_timestamps else _as_datetime(item.get("collected_at"))
    if published is None:
        return None
    age_hours = max(0.0, (now - published).total_seconds() / 3600)
    if age_hours > LOOKBACK_DAYS * 24:
        return None

    source_trust = min(1.0, max(0.0, float(item.get("trust_score") or 0)))
    freshness = 4.0 * math.exp(-age_hours / 72.0)
    score = freshness + source_trust * 2.0
    reasons: list[str] = []
    if age_hours <= 24:
        reasons.append("Published in the last day")
    elif age_hours <= 72:
        reasons.append("Published in the last three days")
    if source_trust >= 0.9:
        reasons.append("High-trust source")

    headline = str(item.get("title") or "").casefold()
    matched = [(phrase, weight) for phrase, weight in _SIGNALS.items() if phrase in headline]
    if matched:
        score += min(2.4, sum(weight for _, weight in matched))
        reasons.append("High-signal topic: " + matched[0][0])

    metadata = _json_object(item.get("metadata_json", item.get("metadata")))
    engagement = max((metadata.get(k, 0) or 0 for k in ("stars", "likes", "downloads")
                      if isinstance(metadata.get(k, 0), (int, float))), default=0)
    if engagement > 0:
        score += min(0.8, math.log1p(engagement) / 20.0)
        if engagement >= 100:
            reasons.append("Strong community uptake")
    if not reasons:
        reasons.append("Recent source update")
    return score, reasons[:3], published


def curate_articles(
    articles: list[dict], *, now: datetime | None = None, limit: int = BRIEFING_SIZE,
    per_source: int = MAX_PER_SOURCE,
) -> list[dict]:
    """Choose a recent, diverse, deduplicated set without generating text."""
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    ranked = []
    for item in articles:
        result = _score(item, now)
        if result is None:
            continue
        score, reasons, published = result
        normalized = _normalized_title(str(item.get("title") or ""))
        ranked.append((score, published, normalized, item, reasons))
    ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)

    selected: list[dict] = []
    source_counts: dict[str, int] = {}
    seen_urls: set[str] = set()
    seen_titles: list[str] = []
    for score, _published, title, item, reasons in ranked:
        source_id = str(item.get("source_id") or "unknown")
        canonical_url = str(item.get("canonical_url") or item.get("url") or "").casefold().rstrip("/")
        if source_counts.get(source_id, 0) >= per_source:
            continue
        if canonical_url and canonical_url in seen_urls:
            continue
        if title and any(
            title == previous
            or (len(set(title.split()) | set(previous.split())) >= 5
                and len(set(title.split()) & set(previous.split())) / len(set(title.split()) | set(previous.split())) >= 0.82)
            for previous in seen_titles
        ):
            continue

        selected_item = dict(item)
        selected_item["editorial_score"] = round(score, 3)
        selected_item["selection_reasons"] = reasons
        selected.append(selected_item)
        source_counts[source_id] = source_counts.get(source_id, 0) + 1
        if canonical_url:
            seen_urls.add(canonical_url)
        if title:
            seen_titles.append(title)
        if len(selected) >= max(1, limit):
            break
    return selected

