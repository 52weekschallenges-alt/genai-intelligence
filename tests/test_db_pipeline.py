from datetime import datetime, timezone

from genai_intelligence.db import Repository
from genai_intelligence.models import Article
from genai_intelligence.pipeline import ingest_source
from genai_intelligence.sources import Source


def test_ingestion_is_idempotent_and_updates_existing_article():
    repo = Repository(":memory:")
    src = Source("test", "Test", "rss", "https://example.com/feed", ["ai"], .8)
    articles = [Article("test", "Test", "First", "https://example.com/post?utm_source=x", item_id="guid-1",
                        published_at=datetime(2024, 1, 1, tzinfo=timezone.utc))]
    result1 = ingest_source(repo, src, lambda _: articles)
    articles[0].title = "Updated"
    result2 = ingest_source(repo, src, lambda _: articles)
    assert result1["inserted"] == 1
    assert result2["inserted"] == 0
    assert repo.counts()["articles"] == 1
    assert repo.list_articles()[0]["title"] == "Updated"
    assert repo.counts()["ingestion_runs"] == 2
    repo.close()


def test_collect_failure_is_recorded_and_does_not_crash_pipeline():
    repo = Repository(":memory:")
    src = Source("bad", "Bad", "rss", "https://example.com", enabled=True)
    result = ingest_source(repo, src, lambda _: (_ for _ in ()).throw(RuntimeError("offline")))
    assert result["status"] == "error"
    assert "offline" in repo.conn.execute("SELECT error FROM ingestion_runs").fetchone()[0]
    repo.close()
