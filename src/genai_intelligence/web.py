"""Small public reader UI and JSON API for a hosted GenAI newspaper."""
from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import hmac
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from urllib.parse import urlencode

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi import HTTPException

from .curation import BRIEFING_SIZE, curate_articles
from .classification import CATEGORY_LABELS
from .db import Repository
from .pipeline import ingest_all
from .sources import load_sources
from .study import generate_study
from .weekly import build_weekly_edition


def _database_path() -> str:
    return os.getenv("GENAI_INTEL_DB", "data/genai_intelligence.sqlite3")


def _config_path() -> str:
    return os.getenv("GENAI_INTEL_CONFIG", "config/sources.yaml")


def _run_ingestion() -> list[dict]:
    sources = load_sources(_config_path())
    repo = Repository(_database_path())
    try:
        results = ingest_all(repo, sources)
        for result in results:
            if result["status"] == "error":
                logger.error("Source %s failed: %s", result["source_id"], result["error"])
            else:
                logger.info("Source %s: fetched=%s inserted=%s", result["source_id"], result["fetched"], result["inserted"])
        return results
    finally:
        repo.close()


def _run_weekly_edition() -> dict:
    repo = Repository(_database_path())
    try:
        return build_weekly_edition(repo)
    finally:
        repo.close()


async def _scheduled_ingestion() -> None:
    if os.getenv("INGEST_ON_STARTUP", "true").lower() not in {"0", "false", "no"}:
        try:
            await asyncio.to_thread(_run_ingestion)
        except Exception:
            logger.exception("Initial source ingestion failed")
    interval = max(1, int(os.getenv("INGEST_INTERVAL_HOURS", "24"))) * 3600
    while True:
        await asyncio.sleep(interval)
        try:
            await asyncio.to_thread(_run_ingestion)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Scheduled source ingestion failed")


@asynccontextmanager
async def lifespan(_: FastAPI):
    task = None
    default_ingestion = "false" if os.getenv("VERCEL") else "true"
    if os.getenv("ENABLE_INGESTION", default_ingestion).lower() not in {"0", "false", "no"}:
        task = asyncio.create_task(_scheduled_ingestion())
    yield
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="GenAI Intelligence", description="A weekly GenAI notebook with saved learning tools.", version="0.4.0", lifespan=lifespan)
logger = logging.getLogger("genai_intelligence.web")


@app.get("/api/cron/collect")
async def scheduled_collection(request: Request):
    secret = os.getenv("CRON_SECRET", "")
    supplied = request.headers.get("authorization", "")
    if not secret or not hmac.compare_digest(supplied, f"Bearer {secret}"):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    results = await asyncio.to_thread(_run_ingestion)
    failures = sum(1 for row in results if row["status"] == "error")
    return {"ok": failures == 0, "sources": len(results), "failures": failures, "results": results}


@app.get("/api/cron/edition")
async def scheduled_edition(request: Request):
    secret = os.getenv("CRON_SECRET", "")
    supplied = request.headers.get("authorization", "")
    if not secret or not hmac.compare_digest(supplied, f"Bearer {secret}"):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return await asyncio.to_thread(_run_weekly_edition)


def _require_notebook_access(request: Request) -> None:
    expected = os.getenv("NOTEBOOK_ACCESS_TOKEN", "").strip()
    supplied = request.headers.get("authorization", "")
    if not expected:
        raise HTTPException(status_code=503, detail="Set NOTEBOOK_ACCESS_TOKEN in the hosting environment to enable your private notebook.")
    if not hmac.compare_digest(supplied, f"Bearer {expected}"):
        raise HTTPException(status_code=401, detail="Enter your notebook access token to continue.")


@app.get("/api/notebook/{article_id}")
def notebook_item(article_id: str, request: Request) -> dict:
    _require_notebook_access(request)
    repo = Repository(_database_path())
    try:
        progress = repo.conn.execute("SELECT completed_at FROM reading_progress WHERE article_id=?", (article_id,)).fetchone()
        study = repo.get_study(article_id)
        return {"completed": bool(progress), "completed_at": progress["completed_at"] if progress else None,
                "study": json.loads(study["study_json"]) if study else None,
                "provider": study["provider"] if study else None}
    finally:
        repo.close()


@app.post("/api/notebook/{article_id}/complete")
def complete_notebook_item(article_id: str, request: Request) -> dict:
    _require_notebook_access(request)
    repo = Repository(_database_path())
    try:
        row = repo.get_article(article_id)
        if not row:
            raise HTTPException(status_code=404, detail="Article not found")
        saved = repo.get_study(article_id)
        if saved:
            study = json.loads(saved["study_json"])
            provider = saved["provider"]
        else:
            article = dict(row)
            generated, provider, model = generate_study(article)
            repo.save_study(article_id, generated, provider, model)
            saved = repo.get_study(article_id)
            study = json.loads(saved["study_json"])
            provider = saved["provider"]
        repo.mark_completed(article_id)
        progress = repo.conn.execute("SELECT completed_at FROM reading_progress WHERE article_id=?", (article_id,)).fetchone()
        return {"completed": True, "completed_at": progress["completed_at"], "study": study, "provider": provider}
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        repo.close()


def _articles(q: str, source: str, category: str, limit: int, offset: int,
              edition: str = "briefing") -> tuple[list[dict], int, dict[str, int]]:
    repo = Repository(_database_path())
    try:
        clauses: list[str] = []
        params: list[object] = []
        if q:
            clauses.append("(title LIKE ? OR summary LIKE ? OR author LIKE ?)")
            needle = f"%{q[:200]}%"
            params.extend([needle, needle, needle])
        if source:
            clauses.append("source_id=?")
            params.append(source)
        if category:
            if repo.backend == "postgres":
                clauses.append("EXISTS (SELECT 1 FROM jsonb_array_elements_text(articles.categories_json::jsonb) AS cat(value) WHERE cat.value=?)")
            else:
                clauses.append("EXISTS (SELECT 1 FROM json_each(articles.categories_json) WHERE value=?)")
            params.append(category)
        weekly_issue = None
        if edition == "briefing":
            week_id = datetime.now(timezone.utc).strftime("%G-W%V")
            weekly_issue = repo.get_weekly_edition(week_id)
            if weekly_issue and weekly_issue["status"] != "published":
                weekly_issue = None
        issue_ids = json.loads(weekly_issue["article_ids_json"]) if weekly_issue else []
        if issue_ids:
            clauses.append("id IN (" + ",".join("?" for _ in issue_ids) + ")")
            params.extend(issue_ids)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        total = int(repo.conn.execute(f"SELECT COUNT(*) AS total FROM articles{where}", params).fetchone()["total"])
        if edition == "all":
            rows = repo.conn.execute(f"SELECT * FROM articles{where} ORDER BY COALESCE(published_at,collected_at) DESC LIMIT ? OFFSET ?",
                                     [*params, limit, offset]).fetchall()
        elif issue_ids:
            ordering = "CASE id " + " ".join(f"WHEN ? THEN {index}" for index in range(len(issue_ids))) + " END"
            rows = repo.conn.execute(f"SELECT * FROM articles{where} ORDER BY {ordering}",
                                     [*params, *issue_ids]).fetchall()
        else:
            rows = repo.conn.execute(f"SELECT * FROM articles{where} ORDER BY COALESCE(published_at,collected_at) DESC LIMIT 10000",
                                     params).fetchall()
        data = []
        for row in rows:
            item = dict(row)
            item["categories"] = json.loads(item.pop("categories_json"))
            item["metadata"] = json.loads(item.pop("metadata_json"))
            if issue_ids:
                study_row = repo.get_study(item["id"])
                if study_row:
                    item["weekly_analysis"] = json.loads(study_row["study_json"])
            data.append(item)
        if edition != "all" and not issue_ids:
            data = curate_articles(data, limit=max(BRIEFING_SIZE, limit + offset))
            total = len(data)
            data = data[offset:offset + limit]
        sources = [dict(row) for row in repo.conn.execute("SELECT id,name FROM sources WHERE enabled=1 ORDER BY name")]
        categories = list(CATEGORY_LABELS)
        pool_total = int(repo.conn.execute("SELECT COUNT(*) AS total FROM articles").fetchone()["total"])
        return data, total, {"articles": total, "pool_total": pool_total, "sources": len(sources),
                             "categories": len(categories)} | {"_sources": sources, "_categories": categories}
    finally:
        repo.close()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/notebook.js")
def notebook_script():
    return FileResponse(os.path.join(os.path.dirname(__file__), "notebook.js"), media_type="text/javascript")


@app.get("/api/articles")
def articles_api(q: str = "", source: str = "", category: str = "", edition: str = "briefing",
                 limit: int = Query(BRIEFING_SIZE, ge=1, le=100), offset: int = Query(0, ge=0)) -> dict:
    if edition not in {"briefing", "all"}:
        edition = "briefing"
    data, total, counts = _articles(q, source, category, limit, offset, edition)
    return {"items": data, "total": total, "pool_total": counts["pool_total"], "edition": edition,
            "limit": limit, "offset": offset}


@app.get("/api/sources")
def sources_api() -> list[dict]:
    repo = Repository(_database_path())
    try:
        return [dict(r) for r in repo.conn.execute("SELECT id,name,type,url FROM sources WHERE enabled=1 ORDER BY name")]
    finally:
        repo.close()


@app.get("/api/editions")
def editions_api() -> list[dict]:
    repo = Repository(_database_path())
    try:
        rows = repo.conn.execute("SELECT week_id,generated_at,story_count FROM weekly_editions WHERE status='published' ORDER BY week_id DESC LIMIT 52").fetchall()
        return [dict(row) for row in rows]
    finally:
        repo.close()


@app.get("/issues", response_class=HTMLResponse)
def issues_page() -> str:
    repo = Repository(_database_path())
    try:
        issues = repo.conn.execute("SELECT week_id,generated_at,story_count FROM weekly_editions WHERE status='published' ORDER BY week_id DESC LIMIT 52").fetchall()
        links = "".join(f'<li><a href="/issues/{html.escape(row["week_id"], quote=True)}">Week {html.escape(row["week_id"])}</a> · {row["story_count"]} stories · published {html.escape(str(row["generated_at"])[:10])}</li>' for row in issues)
    finally:
        repo.close()
    body = f"<ul>{links}</ul>" if links else '<p>No saved weekly editions yet. The scheduled publisher will create one when a free-tier model is available.</p>'
    return f'<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>Saved editions · GenAI Intelligence</title></head><body><main><h1>Saved weekly editions</h1>{body}<p><a href="/">Back to this week</a></p></main></body></html>'


@app.get("/issues/{week_id}", response_class=HTMLResponse)
def issue_page(week_id: str) -> str:
    repo = Repository(_database_path())
    try:
        issue = repo.get_weekly_edition(week_id)
        if not issue or issue["status"] != "published":
            raise HTTPException(status_code=404, detail="Weekly edition not found")
        article_ids = json.loads(issue["article_ids_json"])
        articles = []
        for article_id in article_ids:
            row = repo.get_article(article_id)
            study = repo.get_study(article_id)
            if row and study:
                articles.append((dict(row), json.loads(study["study_json"])))
    finally:
        repo.close()
    cards = []
    for article, study in articles:
        safe_id = html.escape(article["id"], quote=True)
        evidence = study.get("evidence_sources") or [{"source_id": article["source_id"], "url": article["url"]}]
        links = " · ".join(f'<a href="{html.escape(str(source.get("url") or "#"), quote=True)}" target="_blank" rel="noopener noreferrer">{html.escape(str(source.get("source_id") or "Source"))}</a>' for source in evidence)
        cards.append(f'<article><h2>{html.escape(article["title"])}</h2><p>{html.escape(study["briefing"])}</p><h3>Why it matters</h3><p>{html.escape(study["why_it_matters"])}</p><p>Evidence: {links}</p><button class="complete" data-id="{safe_id}" type="button">Open saved quiz + flashcards</button><div class="study" id="study-{safe_id}"></div></article>')
    return f'<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>Week {html.escape(week_id)} · GenAI Intelligence</title><style>body{{font:16px/1.6 system-ui;background:#f5f7fa;color:#172536;margin:0}}main{{max-width:850px;margin:32px auto;padding:0 18px}}article{{background:white;border:1px solid #e2e8f0;border-radius:10px;padding:20px;margin:14px 0}}a{{color:#2859c5}}.study fieldset{{margin:10px 0;padding:12px}}.answer{{color:#176b43}}.error{{color:#a22}}</style></head><body><main><p><a href="/issues">← All saved editions</a></p><h1>Week {html.escape(week_id)}</h1><p>{len(articles)} developments · saved {html.escape(str(issue["generated_at"])[:10])}</p>{"".join(cards)}</main><script src="/notebook.js" defer></script></body></html>'


@app.get("/", response_class=HTMLResponse)
def home(request: Request, q: str = "", source: str = "", category: str = "",
         edition: str = "briefing", page: int = Query(1, ge=1)) -> str:
    if edition not in {"briefing", "all"}:
        edition = "briefing"
    limit = 20
    items, total, counts = _articles(q, source, category, limit,
                                     (page - 1) * limit if edition == "all" else 0, edition)
    sources = counts.pop("_sources")
    categories = counts.pop("_categories")
    def option(value: str, label: str, current: str) -> str:
        selected = " selected" if value == current else ""
        return f'<option value="{html.escape(value, quote=True)}"{selected}>{html.escape(label)}</option>'
    source_options = option("", "All sources", source) + "".join(option(s["id"], s["name"], source) for s in sources)
    category_options = option("", "All topics", category) + "".join(option(c, CATEGORY_LABELS.get(c, c.replace("_", " ").title()), category) for c in categories)
    cards = []
    for item in items:
        published = item.get("published_at") or item.get("collected_at")
        date_label = ""
        if published:
            try:
                date_label = datetime.fromisoformat(published.replace("Z", "+00:00")).astimezone(timezone.utc).strftime("%b %d, %Y")
            except (ValueError, AttributeError):
                date_label = published[:10]
        weekly_analysis = item.get("weekly_analysis") or {}
        summary = weekly_analysis.get("briefing") or item.get("summary") or item.get("content") or "No summary was supplied by the source. Open the source for the full story; mark it read afterward to save a study kit."
        chips = "".join(f'<span class="tag">{html.escape(CATEGORY_LABELS.get(c, c.replace("_", " ").title()))}</span>' for c in item["categories"][:4])
        why = " · ".join(item.get("selection_reasons", []))
        editorial_note = f'<div class="why">Why it made the briefing: {html.escape(why)}</div>' if why and edition == "briefing" else ""
        if weekly_analysis.get("why_it_matters"):
            editorial_note += f'<div class="why"><strong>Why it matters:</strong> {html.escape(weekly_analysis["why_it_matters"])}</div>'
        evidence_sources = weekly_analysis.get("evidence_sources") or ([{"source_id": item["source_id"], "url": item["url"]}] + item.get("related_sources", []))
        if len(evidence_sources) > 1 or weekly_analysis.get("evidence_sources"):
            evidence_links = " · ".join(f'<a href="{html.escape(str(source.get("url") or "#"), quote=True)}" target="_blank" rel="noopener noreferrer">{html.escape(str(source.get("source_id") or "Source"))}</a>' for source in evidence_sources)
            editorial_note += f'<div class="why">Coverage: {evidence_links}</div>'
        safe_id = html.escape(item["id"], quote=True)
        cards.append(f'''<article class="card" data-article-id="{safe_id}"><div class="meta"><span>{html.escape(item["source_id"].replace("_", " ").title())}</span><span>{html.escape(date_label)}</span></div>
<h2><a href="{html.escape(item["url"], quote=True)}" target="_blank" rel="noopener noreferrer">{html.escape(item["title"])}</a></h2>
<p>{html.escape(summary[:600])}{"…" if len(summary) > 600 else ""}</p><div class="tags">{chips}</div>{editorial_note}<button class="complete" type="button" data-id="{safe_id}">Mark read &amp; create study kit</button><div class="study" id="study-{safe_id}"></div></article>''')
    pages = max(1, (total + limit - 1) // limit)
    def page_link(n: int, label: str) -> str:
        params = {k: v for k, v in {"q": q, "source": source, "category": category, "edition": edition, "page": str(n)}.items() if v}
        return f'<a class="page-link" href="/?{urlencode(params)}">{label}</a>'
    pagination = ((page_link(page - 1, "← Newer") if page > 1 else "") + (page_link(page + 1, "Older →") if page < pages else "")) if edition == "all" else ""
    briefing_url = "/?" + urlencode({k: v for k, v in {"q": q, "source": source, "category": category}.items() if v})
    archive_url = "/?" + urlencode({k: v for k, v in {"q": q, "source": source, "category": category, "edition": "all"}.items() if v})
    topic_base = {k: v for k, v in {"q": q, "source": source, "edition": edition}.items() if v}
    all_topics_url = "/?" + urlencode(topic_base)
    topic_links = []
    for topic in categories:
        topic_params = topic_base | {"category": topic}
        active = "active" if category == topic else ""
        label = html.escape(CATEGORY_LABELS.get(topic, topic.replace("_", " ").title()))
        topic_links.append(f'<a class="{active}" href="/?{urlencode(topic_params)}">{label}</a>')
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="A weekly AI notebook with categorized developments, saved learning notes, quizzes, and flashcards."><title>GenAI Intelligence</title>
<style>
:root{{--ink:#172536;--muted:#64748b;--paper:#f5f7fa;--line:#e2e8f0;--accent:#2859c5}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}header{{background:#10253d;color:white;padding:34px max(calc((100vw - 1000px)/2),24px) 30px}}header h1{{font-family:Georgia,serif;font-size:clamp(2rem,5vw,3.25rem);line-height:1.05;margin:0 0 10px;letter-spacing:-.04em}}header p{{margin:0;color:#c3d2e5;max-width:620px}}main{{max-width:1000px;margin:25px auto;padding:0 20px}}.stats{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:18px}}.stat{{background:white;border:1px solid var(--line);border-radius:8px;padding:9px 13px;color:var(--muted);font-size:.9rem}}.stat strong{{color:var(--ink)}}form{{display:grid;grid-template-columns:1fr 180px 180px auto;gap:9px;background:white;border:1px solid var(--line);border-radius:10px;padding:12px;margin-bottom:18px}}input,select,button{{font:inherit;border:1px solid var(--line);border-radius:7px;padding:10px 12px;background:white;color:var(--ink)}}button{{background:var(--accent);color:white;border-color:var(--accent);font-weight:600;cursor:pointer}}.card{{background:white;border:1px solid var(--line);border-radius:10px;padding:20px 22px;margin:12px 0;box-shadow:0 2px 8px #16263b08}}.meta{{display:flex;justify-content:space-between;gap:10px;color:var(--muted);font-size:.82rem;text-transform:uppercase;letter-spacing:.05em}}h2{{font-family:Georgia,serif;font-size:1.4rem;line-height:1.25;margin:10px 0}}h2 a{{color:var(--ink);text-decoration:none}}h2 a:hover{{color:var(--accent)}}.card p{{color:#475569;margin:0 0 12px}}.tags{{display:flex;gap:6px;flex-wrap:wrap}}.tag{{font-size:.76rem;text-transform:capitalize;background:#eef3ff;color:#34569c;border-radius:99px;padding:3px 9px}}.empty{{padding:50px 15px;text-align:center;color:var(--muted);background:white;border:1px solid var(--line);border-radius:10px}}.pagination{{display:flex;justify-content:space-between;margin:20px 0 40px}}.page-link{{color:var(--accent);text-decoration:none;font-weight:600}}footer{{border-top:1px solid var(--line);padding:18px;text-align:center;color:var(--muted);font-size:.84rem}}footer a{{color:var(--accent)}}@media(max-width:720px){{form{{grid-template-columns:1fr 1fr}}form input{{grid-column:1/-1}}header{{padding:28px 22px}}main{{margin-top:18px}}}}
.editions{{display:flex;gap:18px;margin:0 0 12px;padding:0 4px}}.editions a{{color:var(--accent);text-decoration:none;font-weight:600}}.editions a.active{{color:var(--ink);border-bottom:2px solid var(--accent)}}.why{{margin-top:12px;color:var(--muted);font-size:.82rem}}.topics{{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 16px}}.topics a{{background:#fff;border:1px solid var(--line);border-radius:999px;padding:7px 12px;text-decoration:none;color:#34569c;font-size:.88rem}}.topics a.active{{background:#2859c5;color:white;border-color:#2859c5}}.complete{{margin-top:14px}}.study{{margin-top:14px}}.study h3{{margin-bottom:4px}}.study details{{border-top:1px solid var(--line);padding:8px 0}}.study summary{{cursor:pointer;font-weight:600}}.study .answer{{color:#176b43;font-weight:600}}.study .error{{color:#a22}}.study fieldset{{border:1px solid var(--line);border-radius:8px;margin:10px 0;padding:12px}}.study fieldset label{{display:inline-block;padding:4px}}
</style></head><body><header><h1>GenAI Intelligence</h1><p>Your weekly AI notebook: important developments, saved learning notes, quizzes, and flashcards.</p></header>
<main><div class="stats"><span class="stat"><strong>{total:,}</strong> {"stories from the last 7 days" if edition == "briefing" else "matching articles"}</span><span class="stat"><strong>{counts["pool_total"]:,}</strong> in library</span><span class="stat"><strong>{counts["sources"]}</strong> sources</span></div>
<nav class="editions"><a class="{"active" if edition == "briefing" else ""}" href="{briefing_url}">This week</a><a class="{"active" if edition == "all" else ""}" href="{archive_url}">All articles ({counts["pool_total"]:,})</a><a href="/issues">Saved editions</a></nav>
<nav class="topics" aria-label="Browse by topic"><a class="{"active" if not category else ""}" href="{all_topics_url}">All topics</a>{"".join(topic_links)}</nav>
<form action="/" method="get"><input type="hidden" name="edition" value="{edition}"><input type="search" name="q" value="{html.escape(q, quote=True)}" placeholder="Search headlines and summaries" aria-label="Search articles"><select name="source" aria-label="Filter by source">{source_options}</select><select name="category" aria-label="Filter by topic">{category_options}</select><button type="submit">Filter</button></form>
{"".join(cards) if cards else '<div class="empty">No recent stories match this view. Try the full article archive or adjust the filters.</div>'}
<nav class="pagination">{pagination}</nav></main><footer>Stories link to original sources. Your completed study kits are saved in this notebook.</footer>
<script src="/notebook.js" defer></script></body></html>'''
