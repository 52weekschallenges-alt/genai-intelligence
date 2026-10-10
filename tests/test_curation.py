from datetime import datetime, timedelta, timezone

from genai_intelligence.curation import curate_articles


def _story(source, number, published_at, title=None):
    return {
        "id": f"{source}-{number}",
        "source_id": source,
        "title": title or f"Unique model update {source} item {number}",
        "url": f"https://{source}.example/{number}",
        "published_at": published_at.isoformat(),
        "collected_at": published_at.isoformat(),
        "trust_score": 0.9,
        "metadata": {},
    }


def test_briefing_is_recent_ranked_diverse_and_bounded():
    now = datetime(2026, 10, 10, 10, tzinfo=timezone.utc)
    items = []
    for source_number in range(12):
        for item_number in range(5):
            title = f"Model update source{source_number} item{item_number}"
            if source_number == 0 and item_number == 0:
                title = "New benchmark release for reasoning models"
            items.append(_story(f"source{source_number}", item_number,
                                now - timedelta(hours=item_number + source_number), title))
    stale = _story("old_source", 0, now - timedelta(days=10))
    stale["collected_at"] = now.isoformat()
    items.append(stale)

    selected = curate_articles(items, now=now)

    assert len(selected) == 20
    assert all((now - datetime.fromisoformat(item["published_at"])).days < 7 for item in selected)
    assert max(sum(item["source_id"] == source for item in selected)
               for source in {item["source_id"] for item in selected}) <= 3
    assert selected[0]["title"] == "New benchmark release for reasoning models"
    assert "High-signal topic: benchmark" in selected[0]["selection_reasons"]


def test_briefing_collapses_duplicate_headlines_across_sources():
    now = datetime(2026, 10, 10, 10, tzinfo=timezone.utc)
    first = _story("primary", 1, now, "Open model release reaches new benchmark")
    duplicate = _story("secondary", 1, now - timedelta(minutes=1), "Open model release reaches new benchmark!")

    selected = curate_articles([first, duplicate], now=now)

    assert len(selected) == 1
    assert selected[0]["source_id"] == "primary"
