"""Build an immutable, cached edition for the current ISO week."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

from .curation import curate_articles
from .db import Repository
from .analysis import ANALYSIS_VERSION, generate_analysis

logger = logging.getLogger("genai_intelligence.weekly")
WEEKLY_STORY_COUNT = 6
MAX_NEW_STUDIES_PER_RUN = 3
CATALOG_SOURCE_IDS = {"huggingface_models", "github_trending_ai"}


def build_weekly_edition(repo: Repository, *, now: datetime | None = None) -> dict:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    week_id = now.strftime("%G-W%V")
    existing = repo.get_weekly_edition(week_id)
    cutoff = (now - timedelta(days=7)).isoformat()
    rows = repo.conn.execute("""SELECT a.*,d.body AS indexed_document,d.content_hash AS document_hash,
      d.content_scope,d.extraction_method FROM articles a JOIN article_documents d ON d.article_id=a.id
      WHERE d.status='ready' AND d.content_scope='full_text'
      AND COALESCE(a.published_at,a.collected_at)>=?
      ORDER BY COALESCE(a.published_at,a.collected_at) DESC LIMIT 1000""", (cutoff,)).fetchall()
    articles = []
    for row in rows:
        item = dict(row)
        item["categories"] = json.loads(item.pop("categories_json") or "[]")
        item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
        articles.append(item)
    by_id = {str(item["id"]): item for item in articles}
    selected = curate_articles(articles, now=now, limit=WEEKLY_STORY_COUNT)
    candidate_ids = [str(item["id"]) for item in selected]
    if existing and existing["status"] == "published":
        current_ids = json.loads(existing["article_ids_json"])
        up_to_date = all(
            (saved := repo.get_article_analysis(article_id))
            and saved["analysis_version"] == ANALYSIS_VERSION
            and article_id in by_id and saved["document_hash"] == by_id[article_id]["document_hash"]
            for article_id in current_ids
        )
        if up_to_date:
            return {"status": "already_published", "week_id": week_id,
                    "stories": existing["story_count"]}
    if existing and existing["status"] == "draft":
        old_ids = json.loads(existing["article_ids_json"])
        article_ids = [article_id for article_id in old_ids if article_id in by_id]
        article_ids.extend(article_id for article_id in candidate_ids if article_id not in article_ids)
        article_ids = article_ids[:WEEKLY_STORY_COUNT]
    else:
        article_ids = candidate_ids
    if existing:
        repo.update_weekly_draft(week_id, article_ids)
    else:
        repo.save_weekly_draft(week_id, article_ids)
    if not article_ids:
        return {"status": "waiting_for_index", "week_id": week_id, "stories": 0}

    generated_this_run = 0
    for article_id in article_ids:
        saved = repo.get_article_analysis(article_id)
        primary = by_id[article_id]
        if not saved or saved["document_hash"] != primary["document_hash"] or saved["analysis_version"] != ANALYSIS_VERSION:
            if generated_this_run >= MAX_NEW_STUDIES_PER_RUN:
                prepared = sum(1 for saved_id in article_ids
                               if (cached := repo.get_article_analysis(saved_id))
                               and cached["analysis_version"] == ANALYSIS_VERSION
                               and saved_id in by_id
                               and cached["document_hash"] == by_id[saved_id]["document_hash"])
                return {"status": "in_progress", "week_id": week_id,
                        "prepared_stories": prepared,
                        "total_stories": len(article_ids)}
            try:
                related = repo.related_indexed_chunks(article_id, limit=8)
                analysis, provider, model = generate_analysis(primary, related)
            except Exception as exc:
                prepared = sum(1 for saved_id in article_ids
                               if (cached := repo.get_article_analysis(saved_id))
                               and cached["analysis_version"] == ANALYSIS_VERSION
                               and saved_id in by_id
                               and cached["document_hash"] == by_id[saved_id]["document_hash"])
                logger.warning("Weekly issue %s paused at article %s (%s)",
                               week_id, article_id[:10], type(exc).__name__)
                return {"status": "waiting_for_provider", "week_id": week_id,
                        "prepared_stories": prepared, "remaining": len(article_ids) - prepared}
            repo.save_article_analysis(article_id, analysis, provider, model,
                                       primary["document_hash"], ANALYSIS_VERSION)
            generated_this_run += 1

    repo.publish_weekly_edition(week_id)
    return {"status": "published", "week_id": week_id, "stories": len(article_ids)}
