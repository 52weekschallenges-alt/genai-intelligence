"""Small public reader UI and JSON API for a hosted GenAI newspaper."""
from __future__ import annotations

import asyncio
import html
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from urllib.parse import urlencode

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse

from .db import Repository
from .pipeline import ingest_all
from .sources import load_sources


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


app = FastAPI(title="GenAI Intelligence", description="A curated, source-linked GenAI technology newspaper.", version="0.1.0", lifespan=lifespan)
logger = logging.getLogger("genai_intelligence.web")


@app.get("/api/cron/collect")
async def scheduled_collection(request: Request):
    import hmac
    secret = os.getenv("CRON_SECRET", "")
    supplied = request.headers.get("authorization", "")
    if not secret or not hmac.compare_digest(supplied, f"Bearer {secret}"):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    results = await asyncio.to_thread(_run_ingestion)
    failures = sum(1 for row in results if row["status"] == "error")
    return {"ok": failures == 0, "sources": len(results), "failures": failures, "results": results}


def _articles(q: str, source: str, category: str, limit: int, offset: int) -> tuple[list[dict], int, dict[str, int]]:
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
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        total = int(repo.conn.execute(f"SELECT COUNT(*) AS total FROM articles{where}", params).fetchone()["total"])
        rows = repo.conn.execute(f"SELECT * FROM articles{where} ORDER BY COALESCE(published_at,collected_at) DESC LIMIT ? OFFSET ?",
                                 [*params, limit, offset]).fetchall()
        data = []
        for row in rows:
            item = dict(row)
            import json
            item["categories"] = json.loads(item.pop("categories_json"))
            item["metadata"] = json.loads(item.pop("metadata_json"))
            data.append(item)
        sources = [dict(row) for row in repo.conn.execute("SELECT id,name FROM sources WHERE enabled=1 ORDER BY name")]
        if repo.backend == "postgres":
            categories = [r["value"] for r in repo.conn.execute("""SELECT DISTINCT cat.value AS value FROM articles a
                CROSS JOIN LATERAL jsonb_array_elements_text(a.categories_json::jsonb) AS cat(value)
                WHERE cat.value != '' ORDER BY cat.value""")]
        else:
            categories = [r["value"] for r in repo.conn.execute("""SELECT DISTINCT j.value AS value FROM articles a, json_each(a.categories_json) j
                WHERE j.value != '' ORDER BY j.value""")]
        return data, total, {"articles": total, "sources": len(sources), "categories": len(categories)} | {"_sources": sources, "_categories": categories}
    finally:
        repo.close()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/articles")
def articles_api(q: str = "", source: str = "", category: str = "", limit: int = Query(30, ge=1, le=100), offset: int = Query(0, ge=0)) -> dict:
    data, total, _ = _articles(q, source, category, limit, offset)
    return {"items": data, "total": total, "limit": limit, "offset": offset}


@app.get("/api/sources")
def sources_api() -> list[dict]:
    repo = Repository(_database_path())
    try:
        return [dict(r) for r in repo.conn.execute("SELECT id,name,type,url FROM sources WHERE enabled=1 ORDER BY name")]
    finally:
        repo.close()


@app.get("/", response_class=HTMLResponse)
def home(request: Request, q: str = "", source: str = "", category: str = "", page: int = Query(1, ge=1)) -> str:
    limit = 20
    items, total, counts = _articles(q, source, category, limit, (page - 1) * limit)
    sources = counts.pop("_sources")
    categories = counts.pop("_categories")
    def option(value: str, label: str, current: str) -> str:
        selected = " selected" if value == current else ""
        return f'<option value="{html.escape(value, quote=True)}"{selected}>{html.escape(label)}</option>'
    source_options = option("", "All sources", source) + "".join(option(s["id"], s["name"], source) for s in sources)
    category_options = option("", "All topics", category) + "".join(option(c, c.replace("_", " ").title(), category) for c in categories)
    cards = []
    for item in items:
        published = item.get("published_at") or item.get("collected_at")
        date_label = ""
        if published:
            try:
                date_label = datetime.fromisoformat(published.replace("Z", "+00:00")).astimezone(timezone.utc).strftime("%b %d, %Y")
            except (ValueError, AttributeError):
                date_label = published[:10]
        summary = item.get("summary") or item.get("content") or "Open the original source for the full article."
        chips = "".join(f'<span class="tag">{html.escape(c.replace("_", " "))}</span>' for c in item["categories"][:4])
        cards.append(f'''<article class="card"><div class="meta"><span>{html.escape(item["source_id"].replace("_", " ").title())}</span><span>{html.escape(date_label)}</span></div>
<h2><a href="{html.escape(item["url"], quote=True)}" target="_blank" rel="noopener noreferrer">{html.escape(item["title"])}</a></h2>
<p>{html.escape(summary[:600])}{"…" if len(summary) > 600 else ""}</p><div class="tags">{chips}</div></article>''')
    pages = max(1, (total + limit - 1) // limit)
    def page_link(n: int, label: str) -> str:
        params = {k: v for k, v in {"q": q, "source": source, "category": category, "page": str(n)}.items() if v}
        return f'<a class="page-link" href="/?{urlencode(params)}">{label}</a>'
    pagination = (page_link(page - 1, "← Newer") if page > 1 else "") + (page_link(page + 1, "Older →") if page < pages else "")
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="A source-linked daily briefing on important developments across generative AI."><title>GenAI Intelligence</title>
<style>
:root{{--ink:#172536;--muted:#64748b;--paper:#f5f7fa;--line:#e2e8f0;--accent:#2859c5}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}header{{background:#10253d;color:white;padding:34px max(calc((100vw - 1000px)/2),24px) 30px}}header h1{{font-family:Georgia,serif;font-size:clamp(2rem,5vw,3.25rem);line-height:1.05;margin:0 0 10px;letter-spacing:-.04em}}header p{{margin:0;color:#c3d2e5;max-width:620px}}main{{max-width:1000px;margin:25px auto;padding:0 20px}}.stats{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:18px}}.stat{{background:white;border:1px solid var(--line);border-radius:8px;padding:9px 13px;color:var(--muted);font-size:.9rem}}.stat strong{{color:var(--ink)}}form{{display:grid;grid-template-columns:1fr 180px 180px auto;gap:9px;background:white;border:1px solid var(--line);border-radius:10px;padding:12px;margin-bottom:18px}}input,select,button{{font:inherit;border:1px solid var(--line);border-radius:7px;padding:10px 12px;background:white;color:var(--ink)}}button{{background:var(--accent);color:white;border-color:var(--accent);font-weight:600;cursor:pointer}}.card{{background:white;border:1px solid var(--line);border-radius:10px;padding:20px 22px;margin:12px 0;box-shadow:0 2px 8px #16263b08}}.meta{{display:flex;justify-content:space-between;gap:10px;color:var(--muted);font-size:.82rem;text-transform:uppercase;letter-spacing:.05em}}h2{{font-family:Georgia,serif;font-size:1.4rem;line-height:1.25;margin:10px 0}}h2 a{{color:var(--ink);text-decoration:none}}h2 a:hover{{color:var(--accent)}}.card p{{color:#475569;margin:0 0 12px}}.tags{{display:flex;gap:6px;flex-wrap:wrap}}.tag{{font-size:.76rem;text-transform:capitalize;background:#eef3ff;color:#34569c;border-radius:99px;padding:3px 9px}}.empty{{padding:50px 15px;text-align:center;color:var(--muted);background:white;border:1px solid var(--line);border-radius:10px}}.pagination{{display:flex;justify-content:space-between;margin:20px 0 40px}}.page-link{{color:var(--accent);text-decoration:none;font-weight:600}}footer{{border-top:1px solid var(--line);padding:18px;text-align:center;color:var(--muted);font-size:.84rem}}footer a{{color:var(--accent)}}@media(max-width:720px){{form{{grid-template-columns:1fr 1fr}}form input{{grid-column:1/-1}}header{{padding:28px 22px}}main{{margin-top:18px}}}}
</style></head><body><header><h1>GenAI Intelligence</h1><p>A source-linked briefing on what’s changing across generative AI, open models, research, and developer tools.</p></header>
<main><div class="stats"><span class="stat"><strong>{total:,}</strong> matching articles</span><span class="stat"><strong>{counts["sources"]}</strong> sources</span><span class="stat">Updated as feeds publish</span></div>
<form action="/" method="get"><input type="search" name="q" value="{html.escape(q, quote=True)}" placeholder="Search headlines and summaries" aria-label="Search articles"><select name="source" aria-label="Filter by source">{source_options}</select><select name="category" aria-label="Filter by topic">{category_options}</select><button type="submit">Filter</button></form>
{"".join(cards) if cards else '<div class="empty">No articles found yet. Collection runs automatically after deployment. Try again in a few minutes or adjust the filters.</div>'}
<nav class="pagination">{pagination}</nav></main><footer>Summaries and links are provided for discovery. Read each story at its <a href="/api/sources">original source</a>.</footer></body></html>'''
