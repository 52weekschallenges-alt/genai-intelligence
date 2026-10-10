from pathlib import Path

from genai_intelligence.sources import load_sources, normalize_url


def test_registry_includes_user_feeds_and_normalizes_arxiv():
    sources = load_sources(Path(__file__).parents[1] / "config/sources.yaml")
    by_id = {s.id: s for s in sources}
    assert by_id["huggingface_blog"].url == "https://huggingface.co/blog/feed.xml"
    assert by_id["deeplearning_ai_batch"].url == "https://www.deeplearning.ai/the-batch/feed/"
    assert by_id["interconnects"].url == "https://www.interconnects.ai/feed"
    assert by_id["arxiv_cs_ai"].url == "https://export.arxiv.org/rss/cs.AI"


def test_http_arxiv_is_upgraded_to_https():
    assert normalize_url("http://export.arxiv.org/rss/cs.AI#x") == "https://export.arxiv.org/rss/cs.AI"
