"""Source registry loading and validation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import yaml


@dataclass(slots=True)
class Source:
    id: str
    name: str
    type: str
    url: str
    categories: list[str] = field(default_factory=list)
    trust_score: float = 0.5
    enabled: bool = True
    options: dict[str, Any] = field(default_factory=dict)


def normalize_url(url: str) -> str:
    """Normalize common insecure feed URLs and remove fragments."""
    p = urlsplit(url.strip())
    scheme = "https" if p.scheme.lower() == "http" and (p.hostname or "").lower() == "export.arxiv.org" else p.scheme.lower()
    return urlunsplit((scheme, p.netloc.lower(), p.path, p.query, ""))


def load_sources(path: str | Path) -> list[Source]:
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise ValueError(f"Cannot read source registry {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("sources"), list):
        raise ValueError("Source registry must contain a 'sources' list")
    sources: list[Source] = []
    seen: set[str] = set()
    for row in data["sources"]:
        if not isinstance(row, dict):
            raise ValueError("Each source must be a mapping")
        for key in ("id", "name", "type", "url"):
            if not row.get(key):
                raise ValueError(f"Source is missing required field {key!r}")
        sid = str(row["id"])
        if sid in seen:
            raise ValueError(f"Duplicate source id: {sid}")
        seen.add(sid)
        score = float(row.get("trust_score", 0.5))
        if not 0 <= score <= 1:
            raise ValueError(f"trust_score for {sid} must be between 0 and 1")
        known = {"id", "name", "type", "url", "categories", "trust_score", "enabled"}
        sources.append(Source(sid, str(row["name"]), str(row["type"]).lower(), normalize_url(str(row["url"])),
                              list(row.get("categories") or []), score, bool(row.get("enabled", True)),
                              {k: v for k, v in row.items() if k not in known}))
    return sources
