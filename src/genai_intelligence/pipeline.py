"""Source ingestion orchestration."""
from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import Callable

from .collectors import collect
from .db import Repository
from .models import Article
from .sources import Source


def ingest_source(repo: Repository, source: Source, collector: Callable[[Source], list[Article]] = collect) -> dict:
    repo.upsert_source(source)
    started = datetime.now(timezone.utc).isoformat()
    timer = time.monotonic()
    try:
        items = collector(source)
        inserted = 0
        for article in items:
            # Stable IDs get refreshed in place. A different ID for an already-seen
            # canonical URL is treated as a duplicate and is not inserted.
            same_id = bool(article.item_id and repo.conn.execute(
                "SELECT 1 FROM articles WHERE source_id=? AND item_id=?",
                (article.source_id, article.item_id)).fetchone())
            same_url = repo.has_article(article.source_id, None, article.url)
            if same_url and not same_id:
                continue
            # Commit the source batch once in record_run() instead of forcing a
            # database round trip for every item in large RSS feeds.
            repo.save_article(article, commit=False)
            if not same_id:
                inserted += 1
        finished = datetime.now(timezone.utc).isoformat()
        latency_ms = round((time.monotonic() - timer) * 1000)
        repo.record_run(source.id, started, finished, "success", len(items), inserted, latency_ms=latency_ms)
        return {"source_id": source.id, "status": "success", "fetched": len(items), "inserted": inserted, "latency_ms": latency_ms}
    except Exception as exc:
        finished = datetime.now(timezone.utc).isoformat()
        latency_ms = round((time.monotonic() - timer) * 1000)
        repo.record_run(source.id, started, finished, "error", 0, 0,
                        f"{type(exc).__name__}: {exc}", latency_ms=latency_ms)
        return {"source_id": source.id, "status": "error", "fetched": 0, "inserted": 0,
                "latency_ms": latency_ms, "error": f"{type(exc).__name__}: {exc}"}


def ingest_all(repo: Repository, sources: list[Source], collector: Callable[[Source], list[Article]] = collect) -> list[dict]:
    return [ingest_source(repo, source, collector) for source in sources if source.enabled]
