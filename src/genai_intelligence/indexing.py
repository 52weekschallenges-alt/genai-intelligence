"""Fetch, chunk, embed, and incrementally maintain source documents."""
from __future__ import annotations

import hashlib
import ipaddress
import io
import json
import logging
import os
import re
import socket
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader

from .db import Repository

logger = logging.getLogger("genai_intelligence.indexing")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
EMBEDDING_DIMENSIONS = 768
MAX_DOCUMENTS_PER_RUN = 8
MIN_DOCUMENT_CHARS = 900
MAX_DOCUMENT_CHARS = 60000
MAX_FETCH_BYTES = 26 * 1024 * 1024
CHUNK_CHARS = 3200
CHUNK_OVERLAP = 320
CATALOG_SOURCES = {"huggingface_models", "github_trending_ai"}
INDEXABLE_SOURCE_TYPES = {"rss", "arxiv"}
USER_AGENT = "GenAI-Intelligence/0.5 (+source-indexer; contact via project repository)"


def _fingerprint(article: dict) -> str:
    fields = [article.get(key) or "" for key in ("title", "url", "summary", "content", "updated_at", "published_at")]
    return hashlib.sha256(json.dumps(fields, ensure_ascii=False).encode()).hexdigest()


def _public_http_url(value: str) -> str:
    parsed = urlsplit(str(value or ""))
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        return ""
    host = parsed.hostname.rstrip(".").casefold()
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        return ""
    try:
        addresses = [ipaddress.ip_address(host)] if re.fullmatch(r"[0-9a-fA-F:.]+", host) else [
            ipaddress.ip_address(result[4][0].split("%")[0])
            for result in socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
        ]
    except (ValueError, OSError):
        return ""
    if not addresses or any(not address.is_global for address in addresses):
        return ""
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))


def _extract_page(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")
    for node in soup.select("script,style,noscript,svg,nav,footer,header,aside,form,button,[hidden],[aria-hidden=true]"):
        node.decompose()
    for node in soup.select(".cookie, .cookies, .advert, .advertisement, .social-share, .related-posts"):
        node.decompose()
    root = soup.select_one("article") or soup.select_one("main") or soup.select_one('[role="main"]') or soup.body or soup
    paragraphs = []
    for node in root.select("h1,h2,h3,p,li,blockquote,pre"):
        text = re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()
        if len(text) >= 35 and (not paragraphs or text != paragraphs[-1]):
            paragraphs.append(text)
    return "\n\n".join(paragraphs)[:MAX_DOCUMENT_CHARS]


def _extract_pdf(raw_pdf: bytes) -> str:
    if len(raw_pdf) > 25 * 1024 * 1024 or not raw_pdf.startswith(b"%PDF"):
        return ""
    try:
        reader = PdfReader(io.BytesIO(raw_pdf), strict=False)
        paragraphs = []
        for page in reader.pages[:100]:
            text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
            if text:
                paragraphs.append(text)
            if sum(map(len, paragraphs)) >= MAX_DOCUMENT_CHARS:
                break
        return "\n\n".join(paragraphs)[:MAX_DOCUMENT_CHARS]
    except Exception as exc:
        logger.info("PDF extraction failed (%s)", type(exc).__name__)
        return ""


def _fetch_page(url: str, *, arxiv: bool) -> tuple[str, str]:
    safe_url = _public_http_url(url)
    if not safe_url:
        return "", ""
    parsed = urlsplit(safe_url)
    if parsed.path.lower().endswith(".pdf"):
        targets = [safe_url]
    else:
        targets = [safe_url]
    if arxiv and "/abs/" in parsed.path:
        arxiv_html = urlunsplit((parsed.scheme, parsed.netloc, parsed.path.replace("/abs/", "/html/", 1), "", ""))
        arxiv_pdf_path = parsed.path.replace("/abs/", "/pdf/", 1)
        arxiv_pdf = urlunsplit((parsed.scheme, parsed.netloc, arxiv_pdf_path + ".pdf", "", ""))
        targets = [arxiv_html, arxiv_pdf, safe_url]
    for target in targets:
        if not _public_http_url(target):
            continue
        try:
            for _ in range(5):
                if not _public_http_url(target):
                    break
                headers = {"User-Agent": USER_AGENT,
                           "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8"}
                with httpx.stream("GET", target, timeout=httpx.Timeout(16.0, connect=6.0),
                                  follow_redirects=False, headers=headers) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            break
                        target = urljoin(target, location)
                        continue
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").casefold()
                    declared_size = int(response.headers.get("content-length", "0") or "0")
                    if declared_size > MAX_FETCH_BYTES:
                        raise ValueError("Source document exceeds the download size limit")
                    raw = bytearray()
                    for part in response.iter_bytes():
                        if len(raw) + len(part) > MAX_FETCH_BYTES:
                            raise ValueError("Source document exceeds the download size limit")
                        raw.extend(part)
                raw_content = bytes(raw)
                if "pdf" in content_type or raw_content.startswith(b"%PDF"):
                    extracted = _extract_pdf(raw_content)
                    if len(extracted) >= MIN_DOCUMENT_CHARS:
                        return extracted, "arxiv_pdf" if arxiv else "article_pdf"
                    break
                if "html" not in content_type:
                    break
                extracted = _extract_page(raw_content.decode("utf-8", errors="replace"))
                if len(extracted) >= MIN_DOCUMENT_CHARS:
                    return extracted, "article_html" if "/html/" not in target else "arxiv_html"
                break
        except Exception as exc:
            logger.info("Document fetch failed for %s (%s)", parsed.netloc, type(exc).__name__)
    return "", ""


def _split_chunks(text: str) -> list[str]:
    text = re.sub(r"\r\n?", "\n", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + CHUNK_CHARS)
        if end < len(text):
            boundary = max(text.rfind("\n\n", start + CHUNK_CHARS // 2, end),
                           text.rfind(". ", start + CHUNK_CHARS // 2, end))
            if boundary > start:
                end = boundary + (2 if text[boundary:boundary + 2] == "\n\n" else 1)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(start + 1, end - CHUNK_OVERLAP)
    return chunks


def _embed(chunks: list[str], title: str) -> list[list[float]]:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is required to build the semantic document index")
    model = os.getenv("GEMINI_EMBEDDING_MODEL", EMBEDDING_MODEL)
    requests = [{
        "model": f"models/{model}",
        "content": {"parts": [{"text": f"Title: {title}\n\nDocument passage: {chunk}"}]},
        "outputDimensionality": EMBEDDING_DIMENSIONS,
    } for chunk in chunks]
    response = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents",
        timeout=httpx.Timeout(45.0, connect=8.0), headers={"x-goog-api-key": key}, json={"requests": requests})
    response.raise_for_status()
    values = [item.get("values", []) for item in response.json().get("embeddings", [])]
    if len(values) != len(chunks) or any(len(vector) != EMBEDDING_DIMENSIONS for vector in values):
        raise ValueError("Embedding service returned vectors with an unexpected count or dimension")
    return [[float(value) for value in vector] for vector in values]


def _fresh(existing: dict | None, fingerprint: str, *, now: datetime, model: str) -> bool:
    if not existing:
        return False
    try:
        checked = datetime.fromisoformat(existing["indexed_at"].replace("Z", "+00:00"))
        if checked.tzinfo is None:
            checked = checked.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return False
    age = now - checked.astimezone(timezone.utc)
    if (existing["source_fingerprint"] == fingerprint and existing["embedding_model"] == model
            and existing["status"] == "failed" and age < timedelta(days=7)):
        return True
    return (existing["status"] == "ready" and existing["source_fingerprint"] == fingerprint
            and existing["embedding_model"] == model and age < timedelta(days=30))


def index_recent_documents(repo: Repository, *, now: datetime | None = None,
                           limit: int = MAX_DOCUMENTS_PER_RUN) -> dict:
    """Incrementally fetch and vector-index recent editorial documents."""
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    model = os.getenv("GEMINI_EMBEDDING_MODEL", EMBEDDING_MODEL)
    if not os.getenv("GEMINI_API_KEY", "").strip():
        return {"status": "waiting_for_embedding_key", "indexed": 0, "failed": 0, "skipped": 0}
    rows = repo.conn.execute("""SELECT a.*,s.type AS source_type FROM articles a
      JOIN sources s ON s.id=a.source_id
      WHERE s.type IN ('rss','arxiv') AND a.source_id NOT IN ('huggingface_models','github_trending_ai')
      ORDER BY COALESCE(a.published_at,a.collected_at) DESC""").fetchall()
    indexed = failed = skipped = attempted = 0
    for raw in rows:
        if attempted >= max(1, limit):
            break
        article = dict(raw)
        fingerprint = _fingerprint(article)
        existing = repo.get_article_document(article["id"])
        if _fresh(existing, fingerprint, now=now, model=model):
            skipped += 1
            continue
        attempted += 1
        try:
            feed_text = str(article.get("content") or article.get("summary") or "").strip()
            feed_text = feed_text[:MAX_DOCUMENT_CHARS]
            page_text = ""
            page_method = ""
            extraction_method = "feed_content"
            # RSS descriptions are commonly teasers. Fetch the linked document unless the feed
            # already supplied a full-length body; arXiv has an HTML full-text route when available.
            if len(feed_text) < 8000:
                page_text, page_method = _fetch_page(article.get("url") or "", arxiv=article.get("source_type") == "arxiv")
            if len(page_text) >= max(MIN_DOCUMENT_CHARS, len(feed_text) + 200):
                body, extraction_method = page_text, page_method
            else:
                body = feed_text
            if len(body) < MIN_DOCUMENT_CHARS:
                repo.save_index_failure(article["id"], source_fingerprint=fingerprint,
                    error="No substantial readable text in the feed or linked page", embedding_model=model)
                failed += 1
                continue
            body = body[:MAX_DOCUMENT_CHARS]
            content_scope = ("full_text" if extraction_method in {"article_html", "article_pdf", "arxiv_html", "arxiv_pdf"} or len(body) >= 8000
                             else "abstract_only" if article.get("source_type") == "arxiv"
                             else "feed_provided_text")
            chunks = _split_chunks(body)
            vectors = _embed(chunks, article.get("title") or "")
            digest = hashlib.sha256(body.encode()).hexdigest()
            repo.save_article_document(article["id"], body=body, content_hash=digest,
                source_fingerprint=fingerprint, extraction_method=extraction_method,
                content_scope=content_scope, embedding_model=model,
                chunks=list(zip(chunks, vectors)))
            indexed += 1
        except Exception as exc:
            logger.warning("Document index failed for %s (%s)", article.get("id", "")[:10], type(exc).__name__)
            try:
                repo.save_index_failure(article["id"], source_fingerprint=fingerprint,
                                        error=type(exc).__name__, embedding_model=model)
            except Exception:
                logger.exception("Unable to persist indexing failure")
            failed += 1
        # Keep each embedding batch below the public API's recommended request rate.
        if attempted < max(1, limit):
            time.sleep(1.0)
    return {"status": "complete", "indexed": indexed, "failed": failed, "skipped": skipped,
            "attempted": attempted, "embedding_model": model}
