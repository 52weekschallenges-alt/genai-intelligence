"""Command line interface."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .db import Repository
from .pipeline import ingest_all, ingest_source
from .sources import load_sources
from .indexing import index_recent_documents
from .weekly import build_weekly_edition


def _default_registry() -> Path:
    return Path(os.getenv("GENAI_INTEL_CONFIG", Path(__file__).resolve().parents[2] / "config" / "sources.yaml"))


def _default_db() -> str:
    return os.getenv("GENAI_INTEL_DB", "data/genai_intelligence.sqlite3")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="genai-intel", description="Collect and browse GenAI technology sources locally.")
    parser.add_argument("--config", default=str(_default_registry()), help="Source registry YAML path")
    parser.add_argument("--db", default=_default_db(), help="SQLite database path")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("sources", help="List configured sources")
    sub.add_parser("ingest", help="Collect all enabled sources")
    one = sub.add_parser("collect", help="Collect one source by ID")
    one.add_argument("source_id")
    show = sub.add_parser("articles", help="List recent collected articles")
    show.add_argument("--limit", type=int, default=20)
    sub.add_parser("index", help="Fetch, extract, chunk, and embed eligible documents")
    sub.add_parser("edition", help="Generate/resume the current week's long-form notebook edition")
    sub.add_parser("stats", help="Show local database and index counts")
    args = parser.parse_args(argv)

    try:
        sources = load_sources(args.config)
        if args.command == "sources":
            for s in sources:
                print(f"{s.id}\t{s.type}\t{'enabled' if s.enabled else 'disabled'}\t{s.name}")
            return 0
        repo = Repository(args.db)
        try:
            if args.command == "ingest":
                results = ingest_all(repo, sources)
                print(json.dumps(results, indent=2))
                return 1 if any(r["status"] == "error" for r in results) else 0
            if args.command == "collect":
                source = next((s for s in sources if s.id == args.source_id), None)
                if source is None:
                    parser.error(f"Unknown source id {args.source_id!r}")
                result = ingest_source(repo, source)
                print(json.dumps(result, indent=2))
                return 0 if result["status"] == "success" else 1
            if args.command == "articles":
                for row in repo.list_articles(max(1, min(args.limit, 500))):
                    print(json.dumps({"title": row["title"], "url": row["url"], "source_id": row["source_id"],
                                      "published_at": row["published_at"], "summary": row["summary"]}, ensure_ascii=False))
                return 0
            if args.command == "index":
                print(json.dumps(index_recent_documents(repo), indent=2))
                return 0
            if args.command == "edition":
                result = build_weekly_edition(repo)
                print(json.dumps(result, indent=2))
                return 0 if result.get("status") in {"published", "already_published", "in_progress", "waiting_for_index"} else 1
            if args.command == "stats":
                print(json.dumps(repo.counts(), indent=2))
                return 0
        finally:
            repo.close()
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
