"""Small public reader UI and JSON API for a hosted GenAI newspaper."""
from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import hmac
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from urllib.parse import urlencode, urlsplit

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
from .analysis import ANALYSIS_VERSION
from .indexing import index_recent_documents

CATALOG_SOURCE_IDS = {"huggingface_models", "github_trending_ai"}
MIN_STORY_TEXT_CHARS = 80


def _database_path() -> str:
    return os.getenv("GENAI_INTEL_DB", "data/genai_intelligence.sqlite3")


def _config_path() -> str:
    return os.getenv("GENAI_INTEL_CONFIG", "config/sources.yaml")


def _short_brief(generated: str, source_text: str, *, word_limit: int = 72) -> tuple[str, str]:
    """Return a compact, labeled card brief without implying source text is AI-written."""
    generated = str(generated or "").strip()
    value = generated if generated else str(source_text or "")
    label = "AI brief" if generated else "From the source"
    value = html.unescape(re.sub(r"<[^>]+>", " ", value or ""))
    value = re.sub(r"\s+", " ", value).strip()
    if not value:
        return "The feed did not include a summary for this story.", "Story details"

    sentences = re.split(r"(?<=[.!?])\s+", value)
    compact = ""
    for sentence in sentences:
        candidate = f"{compact} {sentence}".strip()
        if len(candidate.split()) > word_limit:
            break
        compact = candidate
        if len(compact.split()) >= 36:
            break
    if not compact:
        compact = " ".join(value.split()[:word_limit])
    if len(compact.split()) < len(value.split()):
        compact = compact.rstrip(" ,;:") + "…"
    return compact, label


def _source_label(source_id: object) -> str:
    return str(source_id or "Source").replace("_", " ").replace("-", " ").title()


def _analysis_source_label(source: dict) -> str:
    label = str(source.get("label") or "").strip()
    name = _source_label(source.get("source_id"))
    return f"{label} · {name}" if label else name


def _external_url(value: object) -> str:
    url = str(value or "").strip()
    try:
        parsed = urlsplit(url)
    except ValueError:
        return ""
    return url if parsed.scheme in {"http", "https"} and parsed.netloc else ""


def _is_newspaper_story(item: dict) -> bool:
    """Keep catalog records and empty feed entries out of the newspaper view."""
    if str(item.get("source_id") or "") in CATALOG_SOURCE_IDS:
        return False
    detail = str(item.get("summary") or item.get("content") or "").strip()
    return len(detail) >= MIN_STORY_TEXT_CHARS


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


def _run_document_index() -> dict:
    repo = Repository(_database_path())
    try:
        return index_recent_documents(repo)
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


app = FastAPI(title="GenAI Intelligence", description="A weekly GenAI notebook with saved learning tools.", version="0.5.0", lifespan=lifespan)
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


@app.get("/api/cron/index")
async def scheduled_index(request: Request):
    secret = os.getenv("CRON_SECRET", "")
    supplied = request.headers.get("authorization", "")
    if not secret or not hmac.compare_digest(supplied, f"Bearer {secret}"):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return await asyncio.to_thread(_run_document_index)


@app.get("/api/cron/edition")
async def scheduled_edition(request: Request):
    secret = os.getenv("CRON_SECRET", "")
    supplied = request.headers.get("authorization", "")
    if not secret or not hmac.compare_digest(supplied, f"Bearer {secret}"):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return await asyncio.to_thread(_run_weekly_edition)


@app.get("/api/study/{article_id}")
def study_item(article_id: str) -> dict:
    repo = Repository(_database_path())
    try:
        study = repo.get_study(article_id)
        return {"study": json.loads(study["study_json"]) if study else None,
                "provider": study["provider"] if study else None}
    finally:
        repo.close()


@app.post("/api/study/{article_id}")
def create_study_item(article_id: str) -> dict:
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
            day = datetime.now(timezone.utc).date().isoformat()
            if not repo.reserve_study_generation(day, daily_limit=20):
                raise HTTPException(status_code=429, detail="The daily free-tier study-kit limit has been reached. Please try again tomorrow; saved kits are still available.")
            article = dict(row)
            document = repo.get_article_document(article_id)
            if document and document["status"] == "ready":
                article["content"] = document["body"]
            analysis_row = repo.get_article_analysis(article_id)
            if analysis_row:
                article["indexed_analysis"] = json.loads(analysis_row["analysis_json"])
            generated, provider, model = generate_study(article)
            repo.save_study(article_id, generated, provider, model)
            saved = repo.get_study(article_id)
            study = json.loads(saved["study_json"])
            provider = saved["provider"]
        return {"completed": True, "study": study, "provider": provider}
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
            clauses.append("(title LIKE ? OR summary LIKE ? OR author LIKE ? OR EXISTS (SELECT 1 FROM article_documents d WHERE d.article_id=articles.id AND d.body LIKE ?))")
            needle = f"%{q[:200]}%"
            params.extend([needle, needle, needle, needle])
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
            if edition == "briefing":
                analysis_row = repo.get_article_analysis(item["id"])
                document = repo.get_article_document(item["id"])
                if (not analysis_row or not document
                        or analysis_row["analysis_version"] != ANALYSIS_VERSION
                        or analysis_row["document_hash"] != document["content_hash"]
                        or document["content_scope"] != "full_text"):
                    continue
                item["weekly_analysis"] = json.loads(analysis_row["analysis_json"])
            data.append(item)
        if edition == "briefing":
            total = len(data)
        if edition != "all" and not issue_ids:
            data = curate_articles(data, limit=max(BRIEFING_SIZE, limit + offset))
            total = len(data)
            data = data[offset:offset + limit]
        sources = [dict(row) for row in repo.conn.execute("SELECT id,name FROM sources WHERE enabled=1 ORDER BY name")]
        categories = list(CATEGORY_LABELS)
        pool_total = int(repo.conn.execute("SELECT COUNT(*) AS total FROM articles").fetchone()["total"])
        indexed_total = int(repo.conn.execute("SELECT COUNT(*) AS total FROM article_documents WHERE status='ready'").fetchone()["total"])
        return data, total, {"articles": total, "pool_total": pool_total, "indexed_total": indexed_total, "sources": len(sources),
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
            analysis_row = repo.get_article_analysis(article_id)
            document = repo.get_article_document(article_id)
            if (row and analysis_row and document
                    and analysis_row["analysis_version"] == ANALYSIS_VERSION
                    and analysis_row["document_hash"] == document["content_hash"]
                    and document["content_scope"] == "full_text"):
                articles.append((dict(row), json.loads(analysis_row["analysis_json"])))
    finally:
        repo.close()
    cards = []
    for article, study in articles:
        safe_id = html.escape(article["id"], quote=True)
        summary, summary_label = str(study.get("executive_takeaway") or ""), "Indexed, source-grounded analysis"
        evidence = study.get("sources") or [{"source_id": article["source_id"], "url": article["url"]}]
        links = " · ".join(f'<a href="{html.escape(_external_url(source.get("url")), quote=True)}" target="_blank" rel="noopener noreferrer">{html.escape(_analysis_source_label(source))}</a>' for source in evidence if _external_url(source.get("url")))
        original_url = _external_url(article.get("url"))
        categories = json.loads(article.get("categories_json") or "[]")
        chips = "".join(f'<span class="tag">{html.escape(CATEGORY_LABELS.get(c, c.replace("_", " ").title()))}</span>' for c in categories[:3])
        original_link = f'<a class="original" href="{html.escape(original_url, quote=True)}" target="_blank" rel="noopener noreferrer">Read original <span aria-hidden="true">↗</span></a>' if original_url else ""
        evidence_note = f'<p class="evidence">Coverage: {links}</p>' if links else ""
        sections_html = "".join(f'<section class="analysis-section"><h3>{html.escape(section.get("heading", "Analysis"))}</h3>' + "".join(f'<p>{html.escape(paragraph)}</p>' for paragraph in section.get("paragraphs", [])) + '</section>' for section in study.get("sections", []))
        points_html = "".join(f'<li>{html.escape(point)}</li>' for point in study.get("key_points", []))
        cards.append(f'<article class="card"><div class="tags">{chips}</div><h2>{html.escape(article["title"])}</h2><p class="brief-label">{html.escape(summary_label)}</p><p class="brief">{html.escape(summary)}</p><div class="why"><strong>Key points</strong><ul>{points_html}</ul></div>{sections_html}{evidence_note}<div class="actions">{original_link}<button class="complete" data-id="{safe_id}" type="button">Open quiz + flashcards</button></div><div class="study" id="study-{safe_id}"></div></article>')
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Week {html.escape(week_id)} · GenAI Intelligence</title><style>body{{font:16px/1.6 system-ui;background:#f4f6f8;color:#172536;margin:0}}main{{max-width:760px;margin:32px auto;padding:0 18px}}.card{{background:white;border:1px solid #e2e8f0;border-radius:14px;padding:22px;margin:18px 0;box-shadow:0 5px 18px #1328420a}}h2{{font:600 1.45rem/1.28 Georgia,serif;margin:14px 0 8px}}h3{{font-size:.98rem;margin:18px 0 4px}}.brief-label,.evidence{{font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;color:#64748b}}.brief{{font-size:1.08rem;color:#26384d}}.tags{{display:flex;gap:6px;flex-wrap:wrap}}.tag{{font-size:.76rem;background:#eef3ff;color:#34569c;border-radius:99px;padding:3px 9px}}.actions{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:16px}}a{{color:#2859c5}}.original{{display:inline-block;background:#2859c5;color:white;text-decoration:none;border-radius:7px;padding:9px 13px;font-weight:650}}button{{font:inherit;border:1px solid #d7deea;border-radius:7px;padding:9px 13px;background:white;color:#23416b;font-weight:600;cursor:pointer}}.study fieldset{{margin:10px 0;padding:12px}}.answer{{color:#176b43}}.error{{color:#a22}}</style></head><body><main><p><a href="/issues">← All saved editions</a></p><h1>Week {html.escape(week_id)}</h1><p>{len(articles)} developments · saved {html.escape(str(issue["generated_at"])[:10])}</p>{"".join(cards)}</main><script src="/notebook.js" defer></script></body></html>'


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
        summary = str(weekly_analysis.get("executive_takeaway") or "") if weekly_analysis else _short_brief("", item.get("summary") or item.get("content") or "")[0]
        summary_label = "Indexed, source-grounded analysis" if weekly_analysis else "Source feed summary"
        chips = "".join(f'<span class="tag">{html.escape(CATEGORY_LABELS.get(c, c.replace("_", " ").title()))}</span>' for c in item["categories"][:4])
        editorial_note = ""
        if weekly_analysis:
            sections_html = "".join(f'<section class="analysis-section"><h3>{html.escape(section.get("heading", "Analysis"))}</h3>' + "".join(f'<p>{html.escape(paragraph)}</p>' for paragraph in section.get("paragraphs", [])) + '</section>' for section in weekly_analysis.get("sections", []))
            points_html = "".join(f'<li>{html.escape(point)}</li>' for point in weekly_analysis.get("key_points", []))
            editorial_note = f'<div class="why"><strong>Key points</strong><ul>{points_html}</ul><p>{html.escape(str(weekly_analysis.get("scope_note") or ""))}</p></div>{sections_html}'
        evidence_sources = weekly_analysis.get("sources") or ([{"source_id": item["source_id"], "url": item["url"]}])
        evidence_links = " · ".join(f'<a href="{html.escape(_external_url(source.get("url")), quote=True)}" target="_blank" rel="noopener noreferrer">{html.escape(_analysis_source_label(source))}</a>' for source in evidence_sources if _external_url(source.get("url")))
        if len(evidence_sources) > 1 and evidence_links:
            editorial_note += f'<div class="evidence">Coverage: {evidence_links}</div>'
        safe_id = html.escape(item["id"], quote=True)
        original_url = _external_url(item.get("url"))
        original_link = f'<a class="original" href="{html.escape(original_url, quote=True)}" target="_blank" rel="noopener noreferrer">Read original <span aria-hidden="true">↗</span></a>' if original_url else ""
        cards.append(f'''<article class="card" data-article-id="{safe_id}"><div class="meta"><span>{html.escape(_source_label(item["source_id"]))}</span><span>{html.escape(date_label)}</span></div>
<div class="tags">{chips}</div><h2>{html.escape(item["title"])}</h2><p class="brief-label">{html.escape(summary_label)}</p><p class="brief">{html.escape(summary)}</p>{editorial_note}<div class="actions">{original_link}<button class="complete" type="button" data-id="{safe_id}">Mark read &amp; create study kit</button></div><div class="study" id="study-{safe_id}"></div></article>''')
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
<meta name="description" content="A searchable AI document index with source-grounded long-form analysis and a personal learning notebook."><title>GenAI Intelligence</title>
<style>
:root{{--ink:#172536;--muted:#64748b;--paper:#f5f7fa;--line:#e2e8f0;--accent:#2859c5}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}header{{background:#10253d;color:white;padding:34px max(calc((100vw - 1000px)/2),24px) 30px}}header h1{{font-family:Georgia,serif;font-size:clamp(2rem,5vw,3.25rem);line-height:1.05;margin:0 0 10px;letter-spacing:-.04em}}header p{{margin:0;color:#c3d2e5;max-width:620px}}main{{max-width:1000px;margin:25px auto;padding:0 20px}}.stats{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:18px}}.stat{{background:white;border:1px solid var(--line);border-radius:8px;padding:9px 13px;color:var(--muted);font-size:.9rem}}.stat strong{{color:var(--ink)}}form{{display:grid;grid-template-columns:1fr 180px 180px auto;gap:9px;background:white;border:1px solid var(--line);border-radius:10px;padding:12px;margin-bottom:18px}}input,select,button{{font:inherit;border:1px solid var(--line);border-radius:7px;padding:10px 12px;background:white;color:var(--ink)}}button{{background:var(--accent);color:white;border-color:var(--accent);font-weight:600;cursor:pointer}}.card{{background:white;border:1px solid var(--line);border-radius:10px;padding:20px 22px;margin:12px 0;box-shadow:0 2px 8px #16263b08}}.meta{{display:flex;justify-content:space-between;gap:10px;color:var(--muted);font-size:.82rem;text-transform:uppercase;letter-spacing:.05em}}h2{{font-family:Georgia,serif;font-size:1.4rem;line-height:1.25;margin:10px 0}}h2 a{{color:var(--ink);text-decoration:none}}h2 a:hover{{color:var(--accent)}}.card p{{color:#475569;margin:0 0 12px}}.tags{{display:flex;gap:6px;flex-wrap:wrap}}.tag{{font-size:.76rem;text-transform:capitalize;background:#eef3ff;color:#34569c;border-radius:99px;padding:3px 9px}}.empty{{padding:50px 15px;text-align:center;color:var(--muted);background:white;border:1px solid var(--line);border-radius:10px}}.pagination{{display:flex;justify-content:space-between;margin:20px 0 40px}}.page-link{{color:var(--accent);text-decoration:none;font-weight:600}}footer{{border-top:1px solid var(--line);padding:18px;text-align:center;color:var(--muted);font-size:.84rem}}footer a{{color:var(--accent)}}@media(max-width:720px){{form{{grid-template-columns:1fr 1fr}}form input{{grid-column:1/-1}}header{{padding:28px 22px}}main{{margin-top:18px}}}}
.editions{{display:flex;gap:18px;margin:0 0 12px;padding:0 4px}}.editions a{{color:var(--accent);text-decoration:none;font-weight:600}}.editions a.active{{color:var(--ink);border-bottom:2px solid var(--accent)}}.why{{margin:15px 0 0;padding:13px 15px;border-left:3px solid #a8bce9;background:#f7f9fd;border-radius:0 8px 8px 0;color:#33445a;font-size:.93rem}}.why p{{margin:4px 0 0;color:#475569}}.analysis-section{{max-width:780px;margin:22px auto}}.analysis-section h3{{font-family:Georgia,serif;font-size:1.2rem;margin:0 0 8px}}.analysis-section p{{font-size:1.04rem;line-height:1.75;color:#334155!important;margin:0 0 12px!important}}.topics{{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 16px}}.topics a{{background:#fff;border:1px solid var(--line);border-radius:999px;padding:7px 12px;text-decoration:none;color:#34569c;font-size:.88rem}}.topics a.active{{background:#2859c5;color:white;border-color:#2859c5}}.complete{{margin-top:14px}}.study{{margin-top:14px}}.study h3{{margin-bottom:4px}}.study details{{border-top:1px solid var(--line);padding:8px 0}}.study summary{{cursor:pointer;font-weight:600}}.study .answer{{color:#176b43;font-weight:600}}.study .error{{color:#a22}}.study fieldset{{border:1px solid var(--line);border-radius:8px;margin:10px 0;padding:12px}}.study fieldset label{{display:inline-block;padding:4px}}.card{{border-radius:14px;padding:22px 24px;margin:16px auto;max-width:820px;box-shadow:0 5px 18px #16263b0a}}.meta{{text-transform:none;letter-spacing:0}}.card h2{{font:600 clamp(1.35rem,3vw,1.7rem)/1.25 Georgia,serif;letter-spacing:-.015em;margin:13px 0 8px}}.brief-label{{font-size:.75rem!important;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)!important;margin:12px 0 3px!important}}.card p.brief{{font-size:1.08rem;line-height:1.58;color:#26384d;margin:0 0 14px}}.evidence{{margin-top:13px;color:var(--muted);font-size:.82rem}}.evidence a{{color:#34569c;text-decoration:none}}.actions{{display:flex;align-items:center;gap:9px;flex-wrap:wrap;margin-top:16px}}.original{{display:inline-flex;align-items:center;gap:6px;background:var(--accent);color:white;text-decoration:none;border-radius:7px;padding:9px 13px;font-weight:650}}.original:hover{{background:#1d459f}}.actions button.complete{{margin:0;background:white;color:#23416b;border-color:#cbd5e1;font-weight:600}}.actions button.complete:hover{{background:#f4f7fb}}.study{{border-top:1px solid var(--line);padding-top:12px}}@media(max-width:600px){{.card{{padding:18px}}.actions{{align-items:stretch}}.actions>*{{text-align:center;justify-content:center;flex:1 1 100%}}}}
</style></head><body><header><h1>GenAI Intelligence</h1><p>A maintained, searchable index of AI documents, with substantial source-grounded analysis and a personal learning notebook.</p></header>
<main><div class="stats"><span class="stat"><strong>{total:,}</strong> {"stories from the last 7 days" if edition == "briefing" else "matching articles"}</span><span class="stat"><strong>{counts["pool_total"]:,}</strong> in library</span><span class="stat"><strong>{counts["indexed_total"]:,}</strong> documents indexed</span><span class="stat"><strong>{counts["sources"]}</strong> sources</span></div>
<nav class="editions"><a class="{"active" if edition == "briefing" else ""}" href="{briefing_url}">This week</a><a class="{"active" if edition == "all" else ""}" href="{archive_url}">All articles ({counts["pool_total"]:,})</a><a href="/issues">Saved editions</a></nav>
<nav class="topics" aria-label="Browse by topic"><a class="{"active" if not category else ""}" href="{all_topics_url}">All topics</a>{"".join(topic_links)}</nav>
<form action="/" method="get"><input type="hidden" name="edition" value="{edition}"><input type="search" name="q" value="{html.escape(q, quote=True)}" placeholder="Search headlines and indexed analyses" aria-label="Search articles"><select name="source" aria-label="Filter by source">{source_options}</select><select name="category" aria-label="Filter by topic">{category_options}</select><button type="submit">Filter</button></form>
{"".join(cards) if cards else '<div class="empty">No complete deep reads are ready yet. The scheduled pipeline collects source updates, indexes full documents, and then writes the long-form analyses shown here. Only completed analyses appear in this edition. <a href="/issues">View saved editions</a>.</div>'}
<nav class="pagination">{pagination}</nav></main><footer>Every analysis is generated from indexed source documents; original sources are linked for verification. Completed study kits are saved in your notebook.</footer>
<script src="/notebook.js" defer></script></body></html>'''
