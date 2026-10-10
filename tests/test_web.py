from fastapi.testclient import TestClient

from genai_intelligence.db import Repository
from genai_intelligence.models import Article
from genai_intelligence.sources import Source
from genai_intelligence.web import app


def test_reader_page_and_articles_api(monkeypatch, tmp_path):
    db_path = tmp_path / "test.sqlite3"
    monkeypatch.setenv("GENAI_INTEL_DB", str(db_path))
    monkeypatch.setenv("ENABLE_INGESTION", "false")
    repo = Repository(db_path)
    source = Source("demo", "Demo", "rss", "https://example.com/feed", ["open_source"], .8)
    repo.upsert_source(source)
    repo.save_article(Article("demo", "Demo", "A new model & tools", "https://example.com/story",
                              summary="A useful public release.", categories=["open_source"], trust_score=.8))
    repo.close()

    with TestClient(app) as client:
        page = client.get("/")
        api = client.get("/api/articles?q=model")
    assert page.status_code == 200
    assert "A new model &amp; tools" in page.text
    assert api.status_code == 200
    assert api.json()["total"] == 1
    assert api.json()["items"][0]["url"] == "https://example.com/story"


def test_cron_route_rejects_requests_without_secret(monkeypatch):
    monkeypatch.delenv("CRON_SECRET", raising=False)
    monkeypatch.setenv("ENABLE_INGESTION", "false")
    with TestClient(app) as client:
        response = client.get("/api/cron/collect")
    assert response.status_code == 401
