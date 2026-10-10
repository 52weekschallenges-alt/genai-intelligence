"""Small public reader UI and JSON API for a hosted GenAI newspaper."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import html
import json
import logging
import os
import hmac
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from urllib.parse import urlencode, urlsplit, quote

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


def _track_pipeline_phase(phase: str, operation) -> dict:
    run_id = str(uuid.uuid4())
    started = datetime.now(timezone.utc).isoformat()
    repo = Repository(_database_path())
    try:
        repo.start_pipeline_run(run_id, phase, started)
    finally:
        repo.close()
    try:
        result = operation()
        if phase == "collect":
            status = "success" if not any(row.get("status") == "error" for row in result) else "partial"
        elif result.get("status") in {"complete", "already_published", "published"} and not result.get("failed", 0):
            status = "success"
        elif result.get("status") in {"waiting_for_embedding_key", "waiting_for_index", "waiting_for_provider", "in_progress"}:
            status = "waiting"
        else:
            status = "partial"
        repo = Repository(_database_path())
        try:
            repo.finish_pipeline_run(run_id, datetime.now(timezone.utc).isoformat(), status,
                                     {"result": result})
        finally:
            repo.close()
        return result
    except Exception as exc:
        repo = Repository(_database_path())
        try:
            repo.finish_pipeline_run(run_id, datetime.now(timezone.utc).isoformat(), "error",
                                     error=f"{type(exc).__name__}: {str(exc)[:300]}")
        finally:
            repo.close()
        raise


def _probe_provider(service: str, model: str, key: str) -> tuple[str, int, str]:
    """Check provider/model reachability without generating text or embeddings."""
    import time
    import httpx

    started = time.monotonic()
    from .telemetry import track_external_call
    endpoint = ("api.groq.com/openai/v1/models" if service == "groq_chat"
                else "generativelanguage.googleapis.com/v1beta/models/" + quote(model, safe=""))
    with track_external_call(service=service, operation="metadata_health_check", model=model,
                             endpoint=endpoint) as trace:
        if service == "groq_chat":
            response = httpx.get("https://api.groq.com/openai/v1/models", timeout=5.0,
                                 headers={"Authorization": f"Bearer {key}"})
            trace["http_status"] = response.status_code
            response.raise_for_status()
            available = any(item.get("id") == model and item.get("active", True)
                            for item in response.json().get("data", []))
            if not available:
                return "model_unavailable", round((time.monotonic() - started) * 1000), f"API reachable; configured model {model} is not listed"
        else:
            response = httpx.get("https://generativelanguage.googleapis.com/v1beta/models/" + quote(model, safe=""),
                                 timeout=5.0, headers={"x-goog-api-key": key})
            trace["http_status"] = response.status_code
            response.raise_for_status()
    return "reachable", round((time.monotonic() - started) * 1000), f"Configured model {model} is available"


def _refresh_provider_health(repo: Repository) -> list[dict]:
    """Refresh provider metadata checks at most every 15 minutes across instances."""
    now = datetime.now(timezone.utc)
    checked_at = now.isoformat()
    stale_before = (now.timestamp() - 15 * 60)
    stale_before_iso = datetime.fromtimestamp(stale_before, timezone.utc).isoformat()
    specs = [
        ("groq_chat", "Groq · text generation", os.getenv("GROQ_API_KEY", "").strip(), os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")),
        ("gemini_generation", "Gemini · text generation", os.getenv("GEMINI_API_KEY", "").strip(), os.getenv("GEMINI_MODEL", "gemini-3.8-flash")),
        ("gemini_embeddings", "Gemini · embeddings", os.getenv("GEMINI_API_KEY", "").strip(), os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")),
    ]
    reserved = []
    for service, label, key, model in specs:
        if key and repo.reserve_service_health_check(service, checked_at, stale_before_iso):
            reserved.append((service, label, key, model))
    if reserved:
        with ThreadPoolExecutor(max_workers=len(reserved)) as pool:
            pending = {item[0]: pool.submit(_probe_provider, item[0], item[3], item[2]) for item in reserved}
            for service, label, key, model in reserved:
                try:
                    status, latency, detail = pending[service].result(timeout=6.0)
                except FutureTimeout:
                    status, latency, detail = "timeout", None, "Provider metadata request timed out"
                except Exception as exc:
                    response = getattr(exc, "response", None)
                    code = response.status_code if response is not None else None
                    if code == 401:
                        status, detail = "unauthorized", "Provider rejected the configured API key (HTTP 401)"
                    elif code == 403:
                        status, detail = "forbidden", "Provider denied access to the configured model (HTTP 403)"
                    elif code == 404:
                        status, detail = "model_unavailable", f"Configured model {model} was not found (HTTP 404)"
                    elif code == 429:
                        status, detail = "rate_limited", "Provider health endpoint returned HTTP 429"
                    elif code:
                        status, detail = "http_error", f"Provider health endpoint returned HTTP {code}"
                    else:
                        status, detail = "unreachable", type(exc).__name__
                    latency = None
                repo.save_service_health(service, checked_at, status, latency, detail)
    latest = {row["service"]: dict(row) for row in repo.conn.execute("SELECT * FROM external_service_health").fetchall()}
    output = []
    for service, label, key, model in specs:
        if not key:
            output.append({"service": service, "name": label, "model": model, "status": "not_configured",
                           "checked_at": None, "latency_ms": None, "detail": "API key is not configured"})
        else:
            item = latest.get(service, {})
            output.append({"service": service, "name": label, "model": model,
                           "status": item.get("status", "checking"), "checked_at": item.get("checked_at"),
                           "latency_ms": item.get("latency_ms"), "detail": item.get("detail", "Health check in progress")})
    return output


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


@app.middleware("http")
async def request_telemetry(request: Request, call_next):
    """Persist route outcome/latency without recording query strings or user data."""
    import time

    request_id = str(uuid.uuid4())
    started = time.monotonic()
    status_code = 500
    error_type = None
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as exc:
        error_type = type(exc).__name__
        raise
    finally:
        try:
            repo = Repository(_database_path())
            try:
                repo.record_internal_http_request(
                    request_id, created_at=datetime.now(timezone.utc).isoformat(),
                    method=request.method, path=request.url.path, status_code=status_code,
                    latency_ms=round((time.monotonic() - started) * 1000), error_type=error_type,
                )
            finally:
                repo.close()
        except Exception as exc:
            logger.warning("Could not persist route telemetry (%s)", type(exc).__name__)
logger = logging.getLogger("genai_intelligence.web")


@app.get("/api/cron/collect")
async def scheduled_collection(request: Request):
    secret = os.getenv("CRON_SECRET", "")
    supplied = request.headers.get("authorization", "")
    if not secret or not hmac.compare_digest(supplied, f"Bearer {secret}"):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    results = await asyncio.to_thread(_track_pipeline_phase, "collect", _run_ingestion)
    failures = sum(1 for row in results if row["status"] == "error")
    return {"ok": failures == 0, "sources": len(results), "failures": failures, "results": results}


@app.get("/api/cron/index")
async def scheduled_index(request: Request):
    secret = os.getenv("CRON_SECRET", "")
    supplied = request.headers.get("authorization", "")
    if not secret or not hmac.compare_digest(supplied, f"Bearer {secret}"):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return await asyncio.to_thread(_track_pipeline_phase, "index", _run_document_index)


@app.get("/api/cron/edition")
async def scheduled_edition(request: Request):
    secret = os.getenv("CRON_SECRET", "")
    supplied = request.headers.get("authorization", "")
    if not secret or not hmac.compare_digest(supplied, f"Bearer {secret}"):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return await asyncio.to_thread(_track_pipeline_phase, "edition", _run_weekly_edition)


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
            if not weekly_issue or weekly_issue["status"] != "published":
                weekly_issue = repo.conn.execute("SELECT * FROM weekly_editions WHERE status='published' ORDER BY week_id DESC LIMIT 1").fetchone()
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
        if edition == "briefing" and not issue_ids:
            data = []
            total = 0
        elif edition != "all" and not issue_ids:
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


@app.get("/api/status")
def status_api() -> dict:
    """Read-only operations snapshot; never returns configured secret values."""
    repo = Repository(_database_path())
    try:
        provider_health = _refresh_provider_health(repo)
        counts = {
            "articles": int(repo.conn.execute("SELECT COUNT(*) AS total FROM articles").fetchone()["total"]),
            "sources": int(repo.conn.execute("SELECT COUNT(*) AS total FROM sources WHERE enabled=1").fetchone()["total"]),
            "indexed_ready": int(repo.conn.execute("SELECT COUNT(*) AS total FROM article_documents WHERE status='ready'").fetchone()["total"]),
            "indexed_full_text": int(repo.conn.execute("SELECT COUNT(*) AS total FROM article_documents WHERE status='ready' AND content_scope='full_text'").fetchone()["total"]),
            "indexed_failed": int(repo.conn.execute("SELECT COUNT(*) AS total FROM article_documents WHERE status='failed'").fetchone()["total"]),
            "analyses": int(repo.conn.execute("SELECT COUNT(*) AS total FROM article_analyses").fetchone()["total"]),
            "published_editions": int(repo.conn.execute("SELECT COUNT(*) AS total FROM weekly_editions WHERE status='published'").fetchone()["total"]),
        }
        source_rows = repo.conn.execute("""SELECT s.id,s.name,s.type,s.url,r.started_at AS checked_at,
          r.status AS last_status,r.fetched_count,r.inserted_count,r.error,r.latency_ms
          FROM sources s LEFT JOIN ingestion_runs r ON r.id=(
            SELECT MAX(id) FROM ingestion_runs WHERE source_id=s.id
          ) WHERE s.enabled=1 ORDER BY s.name""").fetchall()
        pipeline_rows = repo.conn.execute("""SELECT p.* FROM pipeline_runs p JOIN (
          SELECT phase,MAX(started_at) AS latest FROM pipeline_runs GROUP BY phase
        ) latest ON latest.phase=p.phase AND latest.latest=p.started_at ORDER BY p.phase""").fetchall()
        pipeline_by_phase = {row["phase"]: dict(row) for row in pipeline_rows}
        # Older deployments have source/document/edition records but no phase-level
        # telemetry yet. Surface that evidence honestly until the first tracked cron run.
        if "collect" not in pipeline_by_phase:
            last_collect = repo.conn.execute("SELECT MAX(started_at) AS at FROM ingestion_runs").fetchone()["at"]
            if last_collect:
                pipeline_by_phase["collect"] = {"phase": "collect", "started_at": last_collect,
                    "finished_at": None, "status": "observed", "error": None,
                    "detail_json": json.dumps({"result": {"note": "Existing source history; tracked run history begins after this deployment."}})}
        if "index" not in pipeline_by_phase:
            last_index = repo.conn.execute("SELECT MAX(indexed_at) AS at FROM article_documents").fetchone()["at"]
            if last_index:
                pipeline_by_phase["index"] = {"phase": "index", "started_at": last_index,
                    "finished_at": None, "status": "observed", "error": None,
                    "detail_json": json.dumps({"result": {"note": "Last saved document index; tracked run history begins after this deployment."}})}
        if "edition" not in pipeline_by_phase:
            last_edition = repo.conn.execute("SELECT MAX(generated_at) AS at FROM weekly_editions").fetchone()["at"]
            if last_edition:
                pipeline_by_phase["edition"] = {"phase": "edition", "started_at": last_edition,
                    "finished_at": None, "status": "observed", "error": None,
                    "detail_json": json.dumps({"result": {"note": "Last saved edition activity; tracked run history begins after this deployment."}})}
        current_week = datetime.now(timezone.utc).strftime("%G-W%V")
        edition = repo.get_weekly_edition(current_week)
        documents = repo.conn.execute("SELECT status,COUNT(*) AS total FROM article_documents GROUP BY status").fetchall()
        last_analysis = repo.conn.execute("SELECT provider,model,created_at FROM article_analyses ORDER BY created_at DESC LIMIT 1").fetchone()
        last_study = repo.conn.execute("SELECT provider,model,created_at FROM article_studies ORDER BY created_at DESC LIMIT 1").fetchone()
        last_embedding = repo.conn.execute("SELECT status,indexed_at,embedding_model,error FROM article_documents ORDER BY indexed_at DESC LIMIT 1").fetchone()
        internal_requests = [dict(row) for row in repo.recent_internal_requests(100)]
        external_requests = [dict(row) for row in repo.recent_external_requests(100)
                             if row["operation"] != "metadata_health_check"]
        source_requests = [dict(row) for row in repo.conn.execute("""SELECT r.started_at AS created_at,
          s.name AS source,r.status,r.fetched_count AS fetched,r.inserted_count AS inserted,
          r.latency_ms,r.error FROM ingestion_runs r JOIN sources s ON s.id=r.source_id
          ORDER BY r.started_at DESC LIMIT 100""").fetchall()]
        return {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "database": {"status": "connected", "backend": repo.backend},
            "configuration": {
                "embedding_key_configured": bool(os.getenv("GEMINI_API_KEY", "").strip()),
                "study_and_analysis_provider_configured": bool(os.getenv("GROQ_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()),
            },
            "providers": provider_health,
            "last_successful_analysis": dict(last_analysis) if last_analysis else None,
            "last_successful_study_kit": dict(last_study) if last_study else None,
            "last_embedding_attempt": dict(last_embedding) if last_embedding else None,
            "internal_requests": internal_requests,
            "external_requests": external_requests,
            "source_requests": source_requests,
            "counts": counts,
            "documents_by_status": {row["status"]: int(row["total"]) for row in documents},
            "current_edition": dict(edition) if edition else None,
            "pipeline": [{
                "phase": row["phase"], "started_at": row["started_at"], "finished_at": row["finished_at"],
                "status": row["status"], "error": row["error"], "detail": json.loads(row["detail_json"] or "{}"),
            } for row in pipeline_by_phase.values()],
            "sources": [{
                "id": row["id"], "name": row["name"], "type": row["type"], "url": row["url"],
                "checked_at": row["checked_at"], "last_status": row["last_status"],
                "fetched": row["fetched_count"], "inserted": row["inserted_count"], "error": row["error"],
                "latency_ms": row["latency_ms"],
            } for row in source_rows],
            "internal_endpoints": [
                {"path": "/", "purpose": "Reader"}, {"path": "/health", "purpose": "App health"},
                {"path": "/api/status", "purpose": "This dashboard and database status"},
                {"path": "/api/articles", "purpose": "Article API"}, {"path": "/api/sources", "purpose": "Source registry API"},
                {"path": "/api/cron/collect", "purpose": "Manual collector API · protected"},
                {"path": "/api/cron/index", "purpose": "Manual indexer API · protected"},
                {"path": "/api/cron/edition", "purpose": "Manual edition API · protected"},
            ],
        }
    finally:
        repo.close()


@app.get("/status", response_class=HTMLResponse)
def status_page() -> str:
    try:
        snapshot = status_api()
    except Exception as exc:
        logger.exception("Operations dashboard could not read status")
        return HTMLResponse("<h1>Status dashboard unavailable</h1><p>The app could not read its database. Check the deployment logs.</p>", status_code=503)

    def esc(value: object) -> str:
        return html.escape(str(value if value is not None else "—"), quote=True)

    def stamp(value: object) -> str:
        if not value:
            return "Never"
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc).strftime("%b %d, %H:%M UTC")
        except (ValueError, AttributeError):
            return esc(value)

    def is_fresh(value: object, hours: int = 36) -> bool:
        if not value:
            return False
        try:
            checked = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if checked.tzinfo is None:
                checked = checked.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - checked.astimezone(timezone.utc)).total_seconds() < hours * 3600
        except (ValueError, AttributeError):
            return False

    phases = {row["phase"]: row for row in snapshot["pipeline"]}
    phase_names = {"collect": "Collect sources", "index": "Index full documents", "edition": "Write weekly reads"}
    phase_cards = []
    for phase in ("collect", "index", "edition"):
        run = phases.get(phase)
        status = run["status"] if run else "not run yet"
        stale = bool(run and status != "running" and not is_fresh(run["started_at"]))
        stuck = bool(run and status == "running" and not is_fresh(run["started_at"], hours=1))
        if stale:
            status = f"stale · {status}"
        if stuck:
            status = "possibly stuck"
        css = "good" if status in {"success", "already_published", "complete"} else "warn" if status in {"waiting", "partial", "running"} else "bad" if status in {"error", "possibly stuck"} else "muted"
        if stale:
            css = "warn"
        detail = run.get("detail", {}).get("result", {}) if run else {}
        if isinstance(detail, list):
            failed_sources = sum(1 for item in detail if item.get("status") == "error")
            summary = f"{len(detail)} sources checked · {failed_sources} failed"
        elif isinstance(detail, dict):
            summary = ", ".join(f"{esc(k.replace('_', ' '))}: {esc(v)}" for k, v in detail.items() if not isinstance(v, (dict, list)))
        else:
            summary = ""
        error = f'<p class="error">{esc(run["error"])}</p>' if run and run.get("error") else ""
        phase_cards.append(f'<section class="card"><div class="row"><h2>{phase_names[phase]}</h2><span class="pill {css}">{esc(status)}</span></div><p>Last run: {stamp(run["started_at"]) if run else "Never"}</p><p>{summary or "Waiting for its first scheduled run."}</p>{error}</section>')

    source_rows = []
    for source in snapshot["sources"]:
        state = source["last_status"] or "not checked"
        stale = bool(source["checked_at"] and not is_fresh(source["checked_at"]))
        if stale:
            state = f"stale · {state}"
        css = "warn" if stale else "good" if state == "success" else "bad" if state == "error" else "muted"
        checked = stamp(source["checked_at"])
        counts = f'{source["fetched"]} fetched · {source["inserted"]} new · {source["latency_ms"] if source["latency_ms"] is not None else "—"} ms' if source["fetched"] is not None else "—"
        error = f'<br><span class="error">{esc(source["error"])}</span>' if source.get("error") else ""
        source_rows.append(f'<tr><td><a href="{esc(source["url"])}" target="_blank" rel="noopener noreferrer">{esc(source["name"])}</a><small>{esc(source["type"])} · {esc(source["url"])}</small></td><td><span class="pill {css}">{esc(state)}</span>{error}</td><td>{checked}</td><td>{counts}</td></tr>')

    latest_route = {}
    route_counts = {}
    for request_log in snapshot["internal_requests"]:
        latest_route.setdefault(request_log["path"], request_log)
        route_counts[request_log["path"]] = route_counts.get(request_log["path"], 0) + 1
    endpoint_rows = "".join(
        f'<tr><td><code>{esc(endpoint["path"])}</code></td><td>{esc(endpoint["purpose"])}</td>' +
        (f'<td><span class="pill {"good" if latest_route[endpoint["path"]]["status_code"] < 400 else "bad"}">{latest_route[endpoint["path"]]["status_code"]} · {latest_route[endpoint["path"]]["latency_ms"]} ms</span><small>{route_counts.get(endpoint["path"], 0)} requests in last 30 days</small></td>'
         if endpoint["path"] in latest_route else '<td><span class="pill muted">not called yet</span></td>') + '</tr>'
        for endpoint in snapshot["internal_endpoints"])
    external_rows = "".join(
        f'<tr><td>{esc(item["service"])}<small>{esc(item["operation"])} · {esc(item["model"])}</small></td>'
        f'<td><span class="pill {"good" if item["outcome"] == "success" else "bad"}">{esc(item["outcome"])} · {esc(item["http_status"] or "no response")}</span>'
        f'<small>{esc(item["error_detail"] or item["error_type"] or "")}</small></td><td>{esc(item["latency_ms"])} ms</td><td>{stamp(item["created_at"])}</td></tr>'
        for item in snapshot["external_requests"])
    def source_request_row(item: dict) -> str:
        error = f'<small class="error">{esc(item["error"])}</small>' if item.get("error") else ""
        css = "good" if item["status"] == "success" else "bad"
        latency = item["latency_ms"] if item["latency_ms"] is not None else "—"
        return (f'<tr><td>{esc(item["source"])}</td><td><span class="pill {css}">{esc(item["status"])}</span>'
                f'<small>{esc(item["fetched"])} fetched · {esc(item["inserted"])} new</small></td>'
                f'<td>{esc(latency)} ms</td><td>{stamp(item["created_at"])}{error}</td></tr>')
    source_request_rows = "".join(source_request_row(item) for item in snapshot["source_requests"])
    sources_html = "".join(source_rows) or '<tr><td colspan="4">No sources are registered yet.</td></tr>'
    provider_rows = []
    for provider in snapshot["providers"]:
        status = provider["status"]
        css = "good" if status == "reachable" else "muted" if status == "not_configured" else "warn" if status in {"checking", "rate_limited", "timeout"} else "bad"
        latency = f'{provider["latency_ms"]} ms' if provider["latency_ms"] is not None else "—"
        provider_rows.append(f'<tr><td>{esc(provider["name"])}<small>Model: {esc(provider["model"])}</small></td><td><span class="pill {css}">{esc(status.replace("_", " "))}</span><small>{esc(provider["detail"])}</small></td><td>{stamp(provider["checked_at"])}</td><td>{latency}</td></tr>')
    providers_html = "".join(provider_rows)
    count = snapshot["counts"]
    current = snapshot["current_edition"]
    edition_status = current["status"] if current else "not started"
    embedding = snapshot["last_embedding_attempt"]
    embedding_status = f'{embedding["status"]} · {stamp(embedding["indexed_at"])} · {embedding["embedding_model"]}' if embedding else "No embedding attempt recorded"
    embedding_error = f' · {embedding["error"]}' if embedding and embedding.get("error") else ""
    analysis = snapshot["last_successful_analysis"]
    analysis_status = f'{analysis["provider"]} / {analysis["model"]} · {stamp(analysis["created_at"])}' if analysis else "No successful long-form generation recorded"
    study = snapshot["last_successful_study_kit"]
    study_status = f'{study["provider"]} / {study["model"]} · {stamp(study["created_at"])}' if study else "No successful study-kit generation recorded"
    reachable = sum(1 for provider in snapshot["providers"] if provider["status"] == "reachable")
    configured = sum(1 for provider in snapshot["providers"] if provider["status"] != "not_configured")
    source_failures = [source for source in snapshot["sources"] if source["last_status"] == "error"]
    index_run = phases.get("index")
    if not count["indexed_ready"]:
        situation_title = "Collection has run, but indexing has produced no documents."
        situation_body = (f'{count["articles"]:,} articles are in the library, but 0 are indexed. Without indexed full text, the LLM cannot write deep reads. '
                          f'Provider metadata checks: {reachable}/{configured} reachable; these do not test content generation or embeddings.')
        if not snapshot["configuration"]["embedding_key_configured"]:
            next_step = "Next: check that GEMINI_API_KEY is set in GitHub Actions secrets; the independent backend worker needs it for embeddings."
        elif not index_run or index_run.get("status") in {"observed", "not run yet"}:
            next_step = "Next: start the GenAI newspaper backend workflow from the GitHub Actions tab. This database has no recorded index run yet."
        else:
            next_step = "Next: open the latest Index full documents result above and the matching GitHub Actions run for detailed worker output."
        callout_class = "danger"
    elif not analysis:
        situation_title = "Documents are indexed, but no long-form analysis has succeeded yet."
        situation_body = f'{count["indexed_full_text"]:,} full-text documents are ready. Provider metadata checks: {reachable}/{configured} reachable; check the edition run for actual generation or quota errors.'
        next_step = "Next: inspect the Write weekly reads result above and the matching GitHub Actions run for provider errors."
        callout_class = "warn"
    else:
        situation_title = "The pipeline has produced long-form reads."
        situation_body = f'{count["indexed_full_text"]:,} full-text documents and {count["analyses"]:,} analyses are saved. Provider metadata checks: {reachable}/{configured} reachable.'
        next_step = "Review the latest source, indexing, and edition results below for any partial failures."
        callout_class = "ok"
    if source_failures:
        next_step += f' Latest collection also has {len(source_failures)} failing source(s); see the source table below.'
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="60"><title>System status · GenAI Intelligence</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f4f6f8;color:#172536;font:15px/1.5 system-ui,sans-serif}}main{{max-width:1100px;margin:30px auto;padding:0 18px}}header,.card,.panel{{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:20px;margin:14px 0}}header h1{{margin:0}}header p,small,.muted{{color:#64748b}}.top{{display:flex;justify-content:space-between;gap:12px;align-items:center;flex-wrap:wrap}}a{{color:#2859c5}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(245px,1fr));gap:12px}}.card{{margin:0}}.card h2{{margin:0;font-size:1.05rem}}.card p{{margin:10px 0 0;color:#64748b}}.row{{display:flex;justify-content:space-between;gap:8px;align-items:center}}.pill{{display:inline-block;border-radius:99px;padding:3px 9px;font-size:.8rem;font-weight:650;text-transform:capitalize;background:#eef2f7;color:#475569}}.good{{background:#e7f7ef;color:#167448}}.warn{{background:#fff4d8;color:#885b00}}.bad,.error{{color:#b42318}}.bad{{background:#fdebea}}.callout{{border:1px solid #e2e8f0;border-radius:12px;padding:18px 20px;margin:14px 0}}.callout h2{{margin:0 0 6px;font-size:1.15rem}}.callout p{{margin:6px 0}}.callout.danger{{background:#fff2f0;border-color:#f2b8b5}}.callout.warn{{background:#fff8e7;border-color:#f0d28a}}.callout.ok{{background:#eef9f1;border-color:#b6dfc2}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px 8px;text-align:left;border-top:1px solid #e2e8f0;vertical-align:top}}th{{font-size:.8rem;color:#64748b}}td small{{display:block;overflow-wrap:anywhere}}.scroll{{overflow:auto}}.metrics{{display:flex;gap:10px;flex-wrap:wrap}}.metric{{padding:10px 13px;background:#f7f9fc;border-radius:8px}}.metric strong{{font-size:1.2rem}}code{{font-size:.9em}}@media(max-width:650px){{main{{margin-top:14px}}th:nth-child(3),td:nth-child(3){{display:none}}}}
</style></head><body><main><header><div class="top"><div><h1>System status</h1><p>Pipeline progress and last known source checks. Refreshes every 60 seconds.</p></div><a href="/">← Back to newspaper</a></div><p>Updated {stamp(snapshot["checked_at"])} · Database: <span class="pill good">{esc(snapshot["database"]["status"])} ({esc(snapshot["database"]["backend"])})</span> · Current edition: <span class="pill">{esc(edition_status)}</span></p></header>
<section class="callout {callout_class}"><h2>What this means</h2><p><strong>{esc(situation_title)}</strong></p><p>{esc(situation_body)}</p><p><strong>{esc(next_step)}</strong></p></section>
<section class="panel"><h2>Library progress</h2><div class="metrics"><span class="metric"><strong>{count["sources"]}</strong> enabled sources</span><span class="metric"><strong>{count["articles"]:,}</strong> collected articles</span><span class="metric"><strong>{count["indexed_ready"]:,}</strong> indexed docs</span><span class="metric"><strong>{count["indexed_full_text"]:,}</strong> full text</span><span class="metric"><strong>{count["indexed_failed"]:,}</strong> indexing failures</span><span class="metric"><strong>{count["analyses"]:,}</strong> deep reads</span><span class="metric"><strong>{count["published_editions"]:,}</strong> saved editions</span></div></section>
<h2>Scheduled pipeline</h2><p>The independent GitHub Actions worker runs directly against Postgres. These phase results are saved in the database; the repository's Actions tab has the full runner output.</p><div class="grid">{"".join(phase_cards)}</div>
<section class="panel"><h2>External feeds and APIs</h2><p>These show the outcome of the latest scheduled collection; opening a row visits that source.</p><div class="scroll"><table><thead><tr><th>Source endpoint</th><th>Last result</th><th>Last checked</th><th>Collection</th></tr></thead><tbody>{sources_html}</tbody></table></div></section>
<section class="panel"><h2>Recent source request history</h2><p>Individual feed/API collection attempts with result, item counts, latency, and any error. Showing the latest {len(snapshot["source_requests"])} attempts.</p><div class="scroll"><table><thead><tr><th>Source</th><th>Result</th><th>Time</th><th>When / error</th></tr></thead><tbody>{source_request_rows or '<tr><td colspan="4">No source requests logged yet.</td></tr>'}</tbody></table></div></section>
<section class="panel"><h2>LLM and embedding endpoint health</h2><p>Checks call provider model-metadata endpoints every 15 minutes at most. They verify the key and configured model can be reached; they do not generate text or embeddings or consume model tokens.</p><div class="scroll"><table><thead><tr><th>Provider/model</th><th>Reachability</th><th>Last checked</th><th>Response</th></tr></thead><tbody>{providers_html}</tbody></table></div><h3>Last successful real work</h3><p>Embedding: {esc(embedding_status)}{esc(embedding_error)}</p><p>Long-form analysis: {esc(analysis_status)}</p><p>Study kit: {esc(study_status)}</p></section>
<section class="panel"><h2>Actual model and embedding requests</h2><p>These are real generation and embedding calls, with result and response time. Prompts, credentials, and response bodies are not stored.</p><div class="scroll"><table><thead><tr><th>Provider / operation</th><th>Result</th><th>Time</th><th>When</th></tr></thead><tbody>{external_rows or '<tr><td colspan="4">No real model or embedding request has been made yet.</td></tr>'}</tbody></table></div></section>
<section class="panel"><h2>App and backend request log</h2><p>Last {len(snapshot["internal_requests"])} requests (up to 30 days retained). Route, HTTP result, latency, and error type are logged; query strings, IP addresses, and request contents are excluded.</p><div class="scroll"><table><thead><tr><th>Request</th><th>Result</th><th>Time</th><th>When</th></tr></thead><tbody>{''.join(f'<tr><td><code>{esc(item["method"])} {esc(item["path"])}</code></td><td><span class="pill {"good" if item["status_code"] < 400 else "bad"}">{item["status_code"]}</span><small>{esc(item["error_type"] or "")}</small></td><td>{item["latency_ms"]} ms</td><td>{stamp(item["created_at"])}</td></tr>' for item in snapshot["internal_requests"]) or '<tr><td colspan="4">No requests logged yet.</td></tr>'}</tbody></table></div></section>
<section class="panel"><h2>Configuration checks</h2><p>Database connection: <span class="pill good">Connected</span></p><p>Embedding key for reader-side tasks: <span class="pill {"good" if snapshot["configuration"]["embedding_key_configured"] else "bad"}">{"configured" if snapshot["configuration"]["embedding_key_configured"] else "not configured in the web app"}</span></p><p>Analysis/study provider for reader-side tasks: <span class="pill {"good" if snapshot["configuration"]["study_and_analysis_provider_configured"] else "bad"}">{"configured" if snapshot["configuration"]["study_and_analysis_provider_configured"] else "not configured in the web app"}</span></p><p>The independent backend worker credentials are stored separately as GitHub Actions secrets.</p></section>
<p class="muted">This dashboard reports recorded checks and configuration presence; it does not make extra paid monitoring or model calls.</p></main></body></html>'''


@app.get("/notebook.js")
def notebook_script():
    return FileResponse(os.path.join(os.path.dirname(__file__), "notebook.js"), media_type="text/javascript")


@app.get("/api/articles")
def articles_api(q: str = "", source: str = "", category: str = "", edition: str = "briefing",
                 limit: int = Query(BRIEFING_SIZE, ge=1, le=100), offset: int = Query(0, ge=0)) -> dict:
    if edition not in {"briefing", "all"}:
        edition = "briefing"
    repo = Repository(_database_path())
    try:
        has_published_issue = bool(repo.conn.execute("SELECT 1 FROM weekly_editions WHERE status='published' LIMIT 1").fetchone())
    finally:
        repo.close()
    if not has_published_issue:
        return {"items": [], "total": 0, "pool_total": 0, "edition": "preparing", "ready": False,
                "message": "The first complete weekly edition has not been published yet."}
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
    repo = Repository(_database_path())
    try:
        has_published_issue = bool(repo.conn.execute("SELECT 1 FROM weekly_editions WHERE status='published' LIMIT 1").fetchone())
    finally:
        repo.close()
    if not has_published_issue:
        return '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="60"><title>Preparing first edition · GenAI Intelligence</title><style>body{font:16px/1.6 system-ui,sans-serif;background:#f4f6f8;color:#172536;margin:0}main{max-width:720px;margin:12vh auto;padding:28px}article{background:white;border:1px solid #e2e8f0;border-radius:14px;padding:30px}a{color:#2859c5;font-weight:650}</style></head><body><main><article><h1>Preparing the first complete edition</h1><p>The backend is collecting sources, indexing full documents, and preparing the first source-grounded weekly reads. The newspaper will appear here after the first complete edition is saved.</p><p><a href="/status">View live pipeline status →</a></p><p>This page refreshes automatically.</p></article></main></body></html>'''
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
<nav class="editions"><a class="{"active" if edition == "briefing" else ""}" href="{briefing_url}">This week</a><a class="{"active" if edition == "all" else ""}" href="{archive_url}">All articles ({counts["pool_total"]:,})</a><a href="/issues">Saved editions</a><a href="/status">System status</a></nav>
<nav class="topics" aria-label="Browse by topic"><a class="{"active" if not category else ""}" href="{all_topics_url}">All topics</a>{"".join(topic_links)}</nav>
<form action="/" method="get"><input type="hidden" name="edition" value="{edition}"><input type="search" name="q" value="{html.escape(q, quote=True)}" placeholder="Search headlines and indexed analyses" aria-label="Search articles"><select name="source" aria-label="Filter by source">{source_options}</select><select name="category" aria-label="Filter by topic">{category_options}</select><button type="submit">Filter</button></form>
{"".join(cards) if cards else '<div class="empty">No complete deep reads are ready yet. The scheduled pipeline collects source updates, indexes full documents, and then writes the long-form analyses shown here. Only completed analyses appear in this edition. <a href="/status">Check pipeline progress and source health</a> · <a href="/issues">View saved editions</a>.</div>'}
<nav class="pagination">{pagination}</nav></main><footer>Every analysis is generated from indexed source documents; original sources are linked for verification. Completed study kits are saved in your notebook.</footer>
<script src="/notebook.js" defer></script></body></html>'''
