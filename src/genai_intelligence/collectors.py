"""Free, network-backed collectors with bounded timeouts and per-source errors."""
from __future__ import annotations

import os
import re
from typing import Any
from urllib.parse import urlencode

import feedparser
import httpx

from .models import Article, parse_datetime
from .sources import Source

USER_AGENT = "GenAI-Intelligence/0.1 (+local research reader)"


def _client() -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(25.0, connect=10.0), follow_redirects=True,
                        headers={"User-Agent": USER_AGENT, "Accept": "application/json, application/atom+xml, application/rss+xml, application/xml, text/xml"})


def _clean_html(value: str | None) -> str | None:
    if not value:
        return None
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", value)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text, flags=re.I)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&quot;", '"', text)
    return re.sub(r"\s+", " ", text).strip() or None


def _article(source: Source, *, title: Any, url: Any, item_id: Any = None,
             summary: Any = None, content: Any = None, author: Any = None,
             published: Any = None, updated: Any = None, metadata: dict | None = None,
             categories: list[str] | None = None) -> Article | None:
    title = str(title or "").strip()
    url = str(url or "").strip()
    if not title or not url:
        return None
    return Article(source_id=source.id, source_name=source.name, title=title, url=url,
                   item_id=str(item_id) if item_id else None, summary=_clean_html(str(summary)) if summary else None,
                   content=_clean_html(str(content)) if content else None, author=str(author) if author else None,
                   published_at=parse_datetime(published), updated_at=parse_datetime(updated),
                   categories=list(dict.fromkeys(source.categories + (categories or []))),
                   trust_score=source.trust_score, metadata=metadata or {})


def collect_rss(source: Source, client: httpx.Client | None = None) -> list[Article]:
    owns_client = client is None
    client = client or _client()
    try:
        response = client.get(source.url)
        response.raise_for_status()
        parsed = feedparser.parse(response.content)
        if parsed.bozo and not parsed.entries:
            raise ValueError(f"Invalid feed: {parsed.bozo_exception}")
        articles = []
        for entry in parsed.entries:
            link = entry.get("link")
            content = entry.get("content", [{}])
            articles.append(_article(source, title=entry.get("title"), url=link,
                item_id=entry.get("id") or entry.get("guid"), summary=entry.get("summary"),
                content=content[0].get("value") if content else None, author=entry.get("author"),
                published=entry.get("published") or entry.get("updated"), updated=entry.get("updated"),
                categories=[x.get("term", "") for x in entry.get("tags", []) if x.get("term")],
                metadata={"feed_title": parsed.feed.get("title"), "entry_id": entry.get("id")}))
        return [a for a in articles if a]
    finally:
        if owns_client:
            client.close()


def collect_github(source: Source, client: httpx.Client | None = None) -> list[Article]:
    owns_client = client is None
    client = client or _client()
    try:
        params = {"q": source.options.get("query", "topic:llm"), "sort": source.options.get("sort", "updated"),
                  "order": source.options.get("order", "desc"), "per_page": min(int(source.options.get("per_page", 30)), 100)}
        headers = {}
        token = os.getenv("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        response = client.get(source.url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        return [a for item in data.get("items", []) if (a := _article(source,
            title=item.get("full_name"), url=item.get("html_url"), item_id=item.get("node_id") or str(item.get("id")),
            summary=item.get("description"), author=(item.get("owner") or {}).get("login"),
            published=item.get("created_at"), updated=item.get("updated_at"),
            categories=["github"] + list(item.get("topics") or []), metadata={"stars": item.get("stargazers_count"),
                "forks": item.get("forks_count"), "language": item.get("language"), "full_name": item.get("full_name")}))]
    finally:
        if owns_client:
            client.close()


def collect_huggingface(source: Source, client: httpx.Client | None = None) -> list[Article]:
    owns_client = client is None
    client = client or _client()
    try:
        params = {"limit": min(int(source.options.get("limit", 30)), 100), "sort": source.options.get("sort", "lastModified"), "direction": -1}
        if source.options.get("task"):
            params["pipeline_tag"] = source.options["task"]
        headers = {}
        token = os.getenv("HF_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        response = client.get(source.url, params=params, headers=headers)
        response.raise_for_status()
        items = response.json()
        return [a for item in items if (a := _article(source, title=item.get("id"),
            url=f"https://huggingface.co/{item.get('id')}", item_id=item.get("_id") or item.get("id"),
            author=(item.get("id", "/").split("/")[0] if "/" in item.get("id", "") else None),
            published=item.get("createdAt"), updated=item.get("lastModified"),
            categories=list(item.get("tags") or []), metadata={"downloads": item.get("downloads"),
                "likes": item.get("likes"), "pipeline_tag": item.get("pipeline_tag"), "model_id": item.get("id")}))]
    finally:
        if owns_client:
            client.close()


def collect_arxiv(source: Source, client: httpx.Client | None = None) -> list[Article]:
    owns_client = client is None
    client = client or _client()
    try:
        params = {"search_query": source.options.get("query", "cat:cs.AI"), "start": 0,
                  "max_results": min(int(source.options.get("max_results", 30)), 100),
                  "sortBy": source.options.get("sort_by", "submittedDate"), "sortOrder": "descending"}
        response = client.get(source.url, params=params)
        response.raise_for_status()
        parsed = feedparser.parse(response.content)
        articles = []
        for entry in parsed.entries:
            authors = [a.get("name") for a in entry.get("authors", []) if a.get("name")]
            articles.append(_article(source, title=entry.get("title"), url=entry.get("link"),
                item_id=entry.get("id") or entry.get("link"), summary=entry.get("summary"),
                author=", ".join(authors), published=entry.get("published"), updated=entry.get("updated"),
                categories=[t.get("term", "") for t in entry.get("tags", []) if t.get("term")],
                metadata={"arxiv_id": entry.get("id"), "authors": authors}))
        return [a for a in articles if a]
    finally:
        if owns_client:
            client.close()


COLLECTORS = {"rss": collect_rss, "github": collect_github, "huggingface": collect_huggingface, "arxiv": collect_arxiv}


def collect(source: Source) -> list[Article]:
    try:
        collector = COLLECTORS[source.type]
    except KeyError as exc:
        raise ValueError(f"Unsupported source type {source.type!r} for {source.id}") from exc
    return collector(source)
