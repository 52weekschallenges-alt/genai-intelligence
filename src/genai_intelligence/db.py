"""SQLite persistence and idempotent article upserts."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit, urlunsplit

from .models import Article
from .sources import Source

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS sources (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, type TEXT NOT NULL, url TEXT NOT NULL,
  categories_json TEXT NOT NULL DEFAULT '[]', trust_score REAL NOT NULL DEFAULT 0.5,
  enabled INTEGER NOT NULL DEFAULT 1, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS articles (
  id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id), item_id TEXT,
  title TEXT NOT NULL, url TEXT NOT NULL, canonical_url TEXT NOT NULL,
  summary TEXT, content TEXT, author TEXT, published_at TEXT, updated_at TEXT,
  categories_json TEXT NOT NULL DEFAULT '[]', trust_score REAL NOT NULL,
  language TEXT, metadata_json TEXT NOT NULL DEFAULT '{}', collected_at TEXT NOT NULL,
  UNIQUE(source_id, item_id)
);
CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source_id);
CREATE INDEX IF NOT EXISTS idx_articles_canonical ON articles(canonical_url);
CREATE TABLE IF NOT EXISTS ingestion_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, source_id TEXT NOT NULL REFERENCES sources(id),
  started_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL,
  fetched_count INTEGER NOT NULL DEFAULT 0, inserted_count INTEGER NOT NULL DEFAULT 0,
  error TEXT
);
CREATE TABLE IF NOT EXISTS app_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS article_studies (
  article_id TEXT PRIMARY KEY REFERENCES articles(id) ON DELETE CASCADE,
  study_json TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reading_progress (
  article_id TEXT PRIMARY KEY REFERENCES articles(id) ON DELETE CASCADE,
  completed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS study_generation_limits (
  day TEXT PRIMARY KEY, count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS pipeline_runs (
  id TEXT PRIMARY KEY, phase TEXT NOT NULL, started_at TEXT NOT NULL,
  finished_at TEXT, status TEXT NOT NULL, detail_json TEXT NOT NULL DEFAULT '{}', error TEXT
);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_phase_started ON pipeline_runs(phase, started_at DESC);
CREATE TABLE IF NOT EXISTS external_service_health (
  service TEXT PRIMARY KEY, checked_at TEXT NOT NULL, status TEXT NOT NULL,
  latency_ms INTEGER, detail TEXT
);
CREATE TABLE IF NOT EXISTS weekly_editions (
  week_id TEXT PRIMARY KEY, generated_at TEXT NOT NULL, article_ids_json TEXT NOT NULL,
  story_count INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'published'
);
CREATE TABLE IF NOT EXISTS article_documents (
  article_id TEXT PRIMARY KEY REFERENCES articles(id) ON DELETE CASCADE,
  body TEXT NOT NULL, content_hash TEXT NOT NULL, source_fingerprint TEXT NOT NULL,
  extraction_method TEXT NOT NULL, content_scope TEXT NOT NULL, status TEXT NOT NULL,
  indexed_at TEXT NOT NULL, embedding_model TEXT NOT NULL, chunk_count INTEGER NOT NULL DEFAULT 0,
  error TEXT
);
CREATE TABLE IF NOT EXISTS article_chunks (
  chunk_id TEXT PRIMARY KEY, article_id TEXT NOT NULL REFERENCES article_documents(article_id) ON DELETE CASCADE,
  chunk_index INTEGER NOT NULL, content TEXT NOT NULL, embedding_json TEXT NOT NULL,
  UNIQUE(article_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_article_chunks_article ON article_chunks(article_id, chunk_index);
CREATE TABLE IF NOT EXISTS article_analyses (
  article_id TEXT PRIMARY KEY REFERENCES articles(id) ON DELETE CASCADE,
  analysis_json TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL,
  document_hash TEXT NOT NULL, analysis_version INTEGER NOT NULL, created_at TEXT NOT NULL
);
"""

POSTGRES_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS sources (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, type TEXT NOT NULL, url TEXT NOT NULL,
  categories_json TEXT NOT NULL DEFAULT '[]', trust_score DOUBLE PRECISION NOT NULL DEFAULT 0.5,
  enabled SMALLINT NOT NULL DEFAULT 1, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS articles (
  id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id), item_id TEXT,
  title TEXT NOT NULL, url TEXT NOT NULL, canonical_url TEXT NOT NULL,
  summary TEXT, content TEXT, author TEXT, published_at TEXT, updated_at TEXT,
  categories_json TEXT NOT NULL DEFAULT '[]', trust_score DOUBLE PRECISION NOT NULL,
  language TEXT, metadata_json TEXT NOT NULL DEFAULT '{}', collected_at TEXT NOT NULL,
  UNIQUE(source_id, item_id)
);
CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source_id);
CREATE INDEX IF NOT EXISTS idx_articles_canonical ON articles(canonical_url);
CREATE TABLE IF NOT EXISTS ingestion_runs (
  id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id),
  started_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL,
  fetched_count INTEGER NOT NULL DEFAULT 0, inserted_count INTEGER NOT NULL DEFAULT 0, error TEXT
);
CREATE TABLE IF NOT EXISTS app_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS article_studies (
  article_id TEXT PRIMARY KEY REFERENCES articles(id) ON DELETE CASCADE,
  study_json TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reading_progress (
  article_id TEXT PRIMARY KEY REFERENCES articles(id) ON DELETE CASCADE,
  completed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS study_generation_limits (
  day TEXT PRIMARY KEY, count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS pipeline_runs (
  id TEXT PRIMARY KEY, phase TEXT NOT NULL, started_at TEXT NOT NULL,
  finished_at TEXT, status TEXT NOT NULL, detail_json TEXT NOT NULL DEFAULT '{}', error TEXT
);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_phase_started ON pipeline_runs(phase, started_at DESC);
CREATE TABLE IF NOT EXISTS external_service_health (
  service TEXT PRIMARY KEY, checked_at TEXT NOT NULL, status TEXT NOT NULL,
  latency_ms INTEGER, detail TEXT
);
CREATE TABLE IF NOT EXISTS weekly_editions (
  week_id TEXT PRIMARY KEY, generated_at TEXT NOT NULL, article_ids_json TEXT NOT NULL,
  story_count INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'published'
);
CREATE TABLE IF NOT EXISTS article_documents (
  article_id TEXT PRIMARY KEY REFERENCES articles(id) ON DELETE CASCADE,
  body TEXT NOT NULL, content_hash TEXT NOT NULL, source_fingerprint TEXT NOT NULL,
  extraction_method TEXT NOT NULL, content_scope TEXT NOT NULL, status TEXT NOT NULL,
  indexed_at TEXT NOT NULL, embedding_model TEXT NOT NULL, chunk_count INTEGER NOT NULL DEFAULT 0,
  error TEXT
);
CREATE TABLE IF NOT EXISTS article_chunks (
  chunk_id TEXT PRIMARY KEY, article_id TEXT NOT NULL REFERENCES article_documents(article_id) ON DELETE CASCADE,
  chunk_index INTEGER NOT NULL, content TEXT NOT NULL, embedding vector(768) NOT NULL,
  UNIQUE(article_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_article_chunks_article ON article_chunks(article_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_article_chunks_embedding ON article_chunks USING hnsw (embedding vector_cosine_ops);
CREATE TABLE IF NOT EXISTS article_analyses (
  article_id TEXT PRIMARY KEY REFERENCES articles(id) ON DELETE CASCADE,
  analysis_json TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL,
  document_hash TEXT NOT NULL, analysis_version INTEGER NOT NULL, created_at TEXT NOT NULL
);
"""


class _PostgresConnection:
    """Small adapter so the repository's parameterized SQL works on SQLite and Postgres."""
    def __init__(self, connection):
        self._connection = connection

    def execute(self, sql: str, params=()):
        return self._connection.execute(sql.replace("?", "%s"), params)

    def executescript(self, sql: str):
        cursor = None
        for statement in sql.split(";"):
            if statement.strip():
                cursor = self._connection.execute(statement)
        return cursor

    def commit(self):
        self._connection.commit()

    def close(self):
        self._connection.close()


def canonicalize_url(url: str) -> str:
    p = urlsplit(url.strip())
    host = p.netloc.lower().removeprefix("www.")
    path = p.path.rstrip("/") or "/"
    # Drop common tracking parameters while preserving meaningful query values.
    query = "&".join(q for q in p.query.split("&") if q and not q.lower().startswith(("utm_", "fbclid=", "gclid=")))
    return urlunsplit((p.scheme.lower() or "https", host, path, query, ""))


class Repository:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.database_url = os.getenv("DATABASE_URL")
        self.backend = "postgres" if self.database_url else "sqlite"
        if str(path) != ":memory:" and self.backend == "sqlite":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.backend == "postgres":
            try:
                import psycopg
                from psycopg.rows import dict_row
            except ImportError as exc:
                raise RuntimeError("DATABASE_URL requires psycopg; install the project dependencies") from exc
            raw_connection = psycopg.connect(self.database_url, row_factory=dict_row)
            self.conn = _PostgresConnection(raw_connection)
            self.conn.executescript(POSTGRES_SCHEMA)
        else:
            self.conn = sqlite3.connect(str(path), timeout=30)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.executescript(SCHEMA)
        self.conn.commit()
        self._apply_topic_taxonomy()

    def _apply_topic_taxonomy(self) -> None:
        """Reclassify the existing library once when the stable topic set changes."""
        migration = "topic_taxonomy_v1"
        if self.conn.execute("SELECT 1 FROM app_meta WHERE key=?", (migration,)).fetchone():
            return
        from .classification import classify_article
        rows = self.conn.execute("SELECT id,title,summary,content,categories_json FROM articles").fetchall()
        for row in rows:
            try:
                tags = json.loads(row["categories_json"] or "[]")
            except (TypeError, ValueError):
                tags = []
            categories = classify_article(title=row["title"] or "", summary=row["summary"] or "",
                                          content=row["content"] or "", tags=tags)
            self.conn.execute("UPDATE articles SET categories_json=? WHERE id=?",
                              (json.dumps(categories), row["id"]))
        self.conn.execute("INSERT INTO app_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO NOTHING",
                          (migration, "complete"))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def upsert_source(self, source: Source) -> None:
        from datetime import datetime, timezone
        self.conn.execute("""INSERT INTO sources VALUES (?, ?, ?, ?, ?, ?, ?, ?)
          ON CONFLICT(id) DO UPDATE SET name=excluded.name,type=excluded.type,url=excluded.url,
          categories_json=excluded.categories_json,trust_score=excluded.trust_score,
          enabled=excluded.enabled,updated_at=excluded.updated_at""",
          (source.id, source.name, source.type, source.url, json.dumps(source.categories), source.trust_score,
           int(source.enabled), datetime.now(timezone.utc).isoformat()))
        self.conn.commit()

    def save_article(self, article: Article) -> bool:
        canonical = canonicalize_url(article.url)
        identity = article.item_id or canonical
        article_id = hashlib.sha256(f"{article.source_id}\0{identity}".encode()).hexdigest()
        data = article.to_dict()
        cur = self.conn.execute("""INSERT INTO articles
          (id,source_id,item_id,title,url,canonical_url,summary,content,author,published_at,updated_at,
           categories_json,trust_score,language,metadata_json,collected_at)
          VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
          ON CONFLICT(id) DO UPDATE SET title=excluded.title,url=excluded.url,canonical_url=excluded.canonical_url,
           summary=COALESCE(excluded.summary,articles.summary),content=COALESCE(excluded.content,articles.content),
           author=COALESCE(excluded.author,articles.author),published_at=COALESCE(excluded.published_at,articles.published_at),
           updated_at=COALESCE(excluded.updated_at,articles.updated_at),categories_json=excluded.categories_json,
           metadata_json=excluded.metadata_json,collected_at=excluded.collected_at""",
          (article_id, article.source_id, article.item_id, article.title, article.url, canonical, article.summary,
           article.content, article.author, data["published_at"], data["updated_at"], json.dumps(article.categories),
           article.trust_score, article.language, json.dumps(article.metadata, ensure_ascii=False, default=str), data["collected_at"]))
        self.conn.commit()
        # Rowcount reports 1 for insert and update; use total_changes-independent pre-check.
        return cur.rowcount > 0

    def has_article(self, source_id: str, item_id: str | None, url: str) -> bool:
        if item_id:
            row = self.conn.execute("SELECT 1 FROM articles WHERE source_id=? AND item_id=?", (source_id, item_id)).fetchone()
            if row:
                return True
        return self.conn.execute("SELECT 1 FROM articles WHERE source_id=? AND canonical_url=?", (source_id, canonicalize_url(url))).fetchone() is not None

    def record_run(self, source_id: str, started: str, finished: str, status: str, fetched: int, inserted: int, error: str | None = None) -> None:
        self.conn.execute("INSERT INTO ingestion_runs(source_id,started_at,finished_at,status,fetched_count,inserted_count,error) VALUES(?,?,?,?,?,?,?)",
                          (source_id, started, finished, status, fetched, inserted, error))
        self.conn.commit()

    def list_articles(self, limit: int = 50) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM articles ORDER BY COALESCE(published_at,collected_at) DESC LIMIT ?", (limit,)).fetchall()

    def counts(self) -> dict[str, int]:
        counts = {table: int(self.conn.execute(f"SELECT COUNT(*) AS total FROM {table}").fetchone()["total"])
                  for table in ("sources", "articles", "ingestion_runs")}
        counts["indexed_documents"] = int(self.conn.execute("SELECT COUNT(*) AS total FROM article_documents WHERE status='ready'").fetchone()["total"])
        counts["indexed_chunks"] = int(self.conn.execute("SELECT COUNT(*) AS total FROM article_chunks").fetchone()["total"])
        counts["long_form_analyses"] = int(self.conn.execute("SELECT COUNT(*) AS total FROM article_analyses").fetchone()["total"])
        return counts

    def get_article(self, article_id: str):
        return self.conn.execute("SELECT * FROM articles WHERE id=?", (article_id,)).fetchone()

    def get_study(self, article_id: str):
        return self.conn.execute("SELECT * FROM article_studies WHERE article_id=?", (article_id,)).fetchone()

    def save_study(self, article_id: str, study: dict, provider: str, model: str) -> None:
        from datetime import datetime, timezone
        self.conn.execute("""INSERT INTO article_studies(article_id,study_json,provider,model,created_at)
          VALUES(?,?,?,?,?) ON CONFLICT(article_id) DO NOTHING""",
          (article_id, json.dumps(study, ensure_ascii=False), provider, model, datetime.now(timezone.utc).isoformat()))
        self.conn.commit()

    def reserve_study_generation(self, day: str, daily_limit: int = 20) -> bool:
        """Atomically reserve one public study-kit generation for this UTC day."""
        self.conn.execute(
            "INSERT INTO study_generation_limits(day,count) VALUES(?,0) ON CONFLICT(day) DO NOTHING",
            (day,),
        )
        result = self.conn.execute(
            "UPDATE study_generation_limits SET count=count+1 WHERE day=? AND count<?",
            (day, daily_limit),
        )
        self.conn.commit()
        return result.rowcount == 1

    def start_pipeline_run(self, run_id: str, phase: str, started_at: str) -> None:
        self.conn.execute(
            "INSERT INTO pipeline_runs(id,phase,started_at,status) VALUES(?,?,?,'running')",
            (run_id, phase, started_at),
        )
        self.conn.commit()

    def finish_pipeline_run(self, run_id: str, finished_at: str, status: str,
                            detail: dict | None = None, error: str | None = None) -> None:
        self.conn.execute(
            "UPDATE pipeline_runs SET finished_at=?,status=?,detail_json=?,error=? WHERE id=?",
            (finished_at, status, json.dumps(detail or {}, ensure_ascii=False, default=str), error, run_id),
        )
        self.conn.commit()

    def reserve_service_health_check(self, service: str, checked_at: str, stale_before: str) -> bool:
        result = self.conn.execute("""INSERT INTO external_service_health(service,checked_at,status,detail)
          VALUES(?,?,'checking','Health check in progress')
          ON CONFLICT(service) DO UPDATE SET checked_at=excluded.checked_at,status='checking',
          latency_ms=NULL,detail='Health check in progress'
          WHERE external_service_health.checked_at<?""", (service, checked_at, stale_before))
        self.conn.commit()
        return result.rowcount == 1

    def save_service_health(self, service: str, checked_at: str, status: str,
                            latency_ms: int | None, detail: str) -> None:
        self.conn.execute("""UPDATE external_service_health SET checked_at=?,status=?,latency_ms=?,detail=?
          WHERE service=?""", (checked_at, status, latency_ms, detail[:300], service))
        self.conn.commit()

    def get_article_document(self, article_id: str):
        return self.conn.execute("SELECT * FROM article_documents WHERE article_id=?", (article_id,)).fetchone()

    def save_article_document(self, article_id: str, *, body: str, content_hash: str,
                              source_fingerprint: str, extraction_method: str, content_scope: str,
                              embedding_model: str, chunks: list[tuple[str, list[float]]],
                              status: str = "ready", error: str | None = None) -> None:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute("""INSERT INTO article_documents
          (article_id,body,content_hash,source_fingerprint,extraction_method,content_scope,status,indexed_at,embedding_model,chunk_count,error)
          VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(article_id) DO UPDATE SET
          body=excluded.body,content_hash=excluded.content_hash,source_fingerprint=excluded.source_fingerprint,
          extraction_method=excluded.extraction_method,content_scope=excluded.content_scope,status=excluded.status,
          indexed_at=excluded.indexed_at,embedding_model=excluded.embedding_model,chunk_count=excluded.chunk_count,error=excluded.error""",
          (article_id, body, content_hash, source_fingerprint, extraction_method, content_scope, status,
           now, embedding_model, len(chunks), error))
        self.conn.execute("DELETE FROM article_chunks WHERE article_id=?", (article_id,))
        for index, (content, vector) in enumerate(chunks):
            chunk_id = hashlib.sha256(f"{article_id}\0{content_hash}\0{index}".encode()).hexdigest()
            if self.backend == "postgres":
                vector_text = "[" + ",".join(f"{float(value):.8f}" for value in vector) + "]"
                self.conn.execute("INSERT INTO article_chunks(chunk_id,article_id,chunk_index,content,embedding) VALUES(?,?,?,?,?::vector)",
                                  (chunk_id, article_id, index, content, vector_text))
            else:
                self.conn.execute("INSERT INTO article_chunks(chunk_id,article_id,chunk_index,content,embedding_json) VALUES(?,?,?,?,?)",
                                  (chunk_id, article_id, index, content, json.dumps(vector)))
        self.conn.commit()

    def save_index_failure(self, article_id: str, *, source_fingerprint: str, error: str,
                           embedding_model: str) -> None:
        self.save_article_document(article_id, body="", content_hash="", source_fingerprint=source_fingerprint,
                                   extraction_method="failed", content_scope="unavailable",
                                   embedding_model=embedding_model, chunks=[], status="failed", error=error[:500])

    def get_article_analysis(self, article_id: str):
        return self.conn.execute("SELECT * FROM article_analyses WHERE article_id=?", (article_id,)).fetchone()

    def save_article_analysis(self, article_id: str, analysis: dict, provider: str, model: str,
                              document_hash: str, analysis_version: int) -> None:
        from datetime import datetime, timezone
        self.conn.execute("""INSERT INTO article_analyses
          (article_id,analysis_json,provider,model,document_hash,analysis_version,created_at)
          VALUES(?,?,?,?,?,?,?) ON CONFLICT(article_id) DO UPDATE SET
          analysis_json=excluded.analysis_json,provider=excluded.provider,model=excluded.model,
          document_hash=excluded.document_hash,analysis_version=excluded.analysis_version,created_at=excluded.created_at""",
          (article_id, json.dumps(analysis, ensure_ascii=False), provider, model, document_hash,
           analysis_version, datetime.now(timezone.utc).isoformat()))
        self.conn.commit()

    def related_indexed_chunks(self, article_id: str, *, limit: int = 8, threshold: float = 0.74) -> list[dict]:
        if self.backend == "postgres":
            own = self.conn.execute("SELECT embedding FROM article_chunks WHERE article_id=? ORDER BY chunk_index LIMIT 1", (article_id,)).fetchone()
            if not own:
                return []
            vector = str(own["embedding"])
            rows = self.conn.execute("""SELECT c.article_id,c.content,1-(c.embedding <=> ?::vector) AS similarity,
              a.title,a.url,a.source_id FROM article_chunks c JOIN articles a ON a.id=c.article_id
              WHERE c.article_id<>? ORDER BY c.embedding <=> ?::vector LIMIT ?""",
              (vector, article_id, vector, limit)).fetchall()
            return [dict(row) for row in rows if float(row["similarity"]) >= threshold]
        rows = self.conn.execute("""SELECT c.article_id,c.content,c.embedding_json,a.title,a.url,a.source_id
          FROM article_chunks c JOIN articles a ON a.id=c.article_id WHERE c.article_id<>?""", (article_id,)).fetchall()
        own = self.conn.execute("SELECT embedding_json FROM article_chunks WHERE article_id=? ORDER BY chunk_index LIMIT 1", (article_id,)).fetchone()
        if not own:
            return []
        query = json.loads(own["embedding_json"])
        query_norm = sum(value * value for value in query) ** 0.5 or 1.0
        scored = []
        for row in rows:
            vector = json.loads(row["embedding_json"])
            denom = query_norm * (sum(value * value for value in vector) ** 0.5 or 1.0)
            similarity = sum(left * right for left, right in zip(query, vector)) / denom
            if similarity >= threshold:
                scored.append({"article_id": row["article_id"], "content": row["content"],
                               "title": row["title"], "url": row["url"], "source_id": row["source_id"],
                               "similarity": similarity})
        scored.sort(key=lambda row: row["similarity"], reverse=True)
        return scored[:limit]

    def update_weekly_draft(self, week_id: str, article_ids: list[str]) -> None:
        from datetime import datetime, timezone
        self.conn.execute("""UPDATE weekly_editions SET article_ids_json=?,story_count=?,status='draft',generated_at=?
          WHERE week_id=?""", (json.dumps(article_ids), len(article_ids), datetime.now(timezone.utc).isoformat(), week_id))
        self.conn.commit()

    def mark_completed(self, article_id: str) -> None:
        from datetime import datetime, timezone
        self.conn.execute("""INSERT INTO reading_progress(article_id,completed_at) VALUES(?,?)
          ON CONFLICT(article_id) DO NOTHING""", (article_id, datetime.now(timezone.utc).isoformat()))
        self.conn.commit()

    def get_weekly_edition(self, week_id: str):
        return self.conn.execute("SELECT * FROM weekly_editions WHERE week_id=?", (week_id,)).fetchone()

    def save_weekly_draft(self, week_id: str, article_ids: list[str]) -> None:
        from datetime import datetime, timezone
        self.conn.execute("""INSERT INTO weekly_editions(week_id,generated_at,article_ids_json,story_count,status)
          VALUES(?,?,?,?, 'draft') ON CONFLICT(week_id) DO NOTHING""",
          (week_id, datetime.now(timezone.utc).isoformat(), json.dumps(article_ids), len(article_ids)))
        self.conn.commit()

    def publish_weekly_edition(self, week_id: str) -> None:
        from datetime import datetime, timezone
        self.conn.execute("UPDATE weekly_editions SET status='published',generated_at=? WHERE week_id=?",
                          (datetime.now(timezone.utc).isoformat(), week_id))
        self.conn.commit()
