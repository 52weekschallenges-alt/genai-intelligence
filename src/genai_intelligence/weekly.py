"""Build an immutable, cached edition for the current ISO week."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from .curation import curate_articles
from .db import Repository
from .study import generate_study

logger = logging.getLogger("genai_intelligence.weekly")
WEEKLY_STORY_COUNT = 6
MAX_NEW_STUDIES_PER_RUN = 2


def build_weekly_edition(repo: Repository, *, now: datetime | None = None) -> dict:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    week_id = now.strftime("%G-W%V")
    existing = repo.get_weekly_edition(week_id)
    if existing and existing["status"] == "published":
        return {"status": "already_published", "week_id": week_id,
                "stories": existing["story_count"]}

    rows = repo.conn.execute("SELECT * FROM articles ORDER BY COALESCE(published_at,collected_at) DESC LIMIT 10000").fetchall()
    articles = []
    for row in rows:
        item = dict(row)
        item["categories"] = json.loads(item.pop("categories_json") or "[]")
        item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
        articles.append(item)
    by_id = {str(item["id"]): item for item in articles}
    if existing:
        article_ids = json.loads(existing["article_ids_json"])
    else:
        # Model catalog entries often have only a model ID and tags; they are
        # useful in the archive but too thin to become a synthesized newspaper story.
        readable = [item for item in articles
                    if len(str(item.get("summary") or item.get("content") or "").strip()) >= 80]
        selected = curate_articles(readable, now=now, limit=WEEKLY_STORY_COUNT)
        article_ids = [str(item["id"]) for item in selected]
        if not article_ids:
            return {"status": "waiting_for_stories", "week_id": week_id, "stories": 0}
        repo.save_weekly_draft(week_id, article_ids)

    generated_this_run = 0
    for article_id in article_ids:
        saved = repo.get_study(article_id)
        if not saved:
            if generated_this_run >= MAX_NEW_STUDIES_PER_RUN:
                prepared = sum(1 for saved_id in article_ids if repo.get_study(saved_id))
                return {"status": "in_progress", "week_id": week_id,
                        "prepared_stories": prepared,
                        "total_stories": len(article_ids)}
            try:
                study, provider, model = generate_study(by_id[article_id])
            except Exception as exc:
                prepared = sum(1 for saved_id in article_ids if repo.get_study(saved_id))
                logger.warning("Weekly issue %s paused at article %s (%s)",
                               week_id, article_id[:10], type(exc).__name__)
                return {"status": "waiting_for_provider", "week_id": week_id,
                        "prepared_stories": prepared, "remaining": len(article_ids) - prepared}
            primary = by_id[article_id]
            study["evidence_sources"] = [{"source_id": primary.get("source_id"), "title": primary.get("title"), "url": primary.get("url")}] + [
                {"source_id": source.get("source_id"), "title": source.get("title"), "url": source.get("url")}
                for source in primary.get("related_sources", [])[:3]
            ]
            repo.save_study(article_id, study, provider, model)
            generated_this_run += 1

    repo.publish_weekly_edition(week_id)
    return {"status": "published", "week_id": week_id, "stories": len(article_ids)}
