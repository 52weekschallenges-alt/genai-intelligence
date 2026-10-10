"""Independent scheduled backend worker for GitHub Actions and other job runners."""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone

from .db import Repository
from .indexing import index_recent_documents
from .pipeline import ingest_all
from .sources import load_sources
from .weekly import build_weekly_edition


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _phase(repo: Repository, name: str, operation) -> dict:
    run_id = str(uuid.uuid4())
    started = _now()
    repo.start_pipeline_run(run_id, name, started)
    try:
        result = operation()
        if name == "collect":
            failures = sum(item.get("status") == "error" for item in result)
            status = "partial" if failures else "success"
        else:
            result_status = result.get("status")
            if result_status in {"published", "already_published", "complete"}:
                status = "partial" if result.get("failed", 0) else "success"
            elif result_status in {"waiting_for_index", "waiting_for_provider", "waiting_for_embedding_key"}:
                status = "waiting"
            else:
                status = "partial"
        repo.finish_pipeline_run(run_id, _now(), status, {"result": result})
        record = {"phase": name, "status": status, "result": result}
        _report(record)
        return record
    except Exception as exc:
        repo.finish_pipeline_run(run_id, _now(), "error", error=f"{type(exc).__name__}: {str(exc)[:300]}")
        record = {"phase": name, "status": "error", "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        _report(record)
        raise


def _report(record: dict) -> None:
    line = json.dumps(record, ensure_ascii=False, default=str)
    print(line, flush=True)
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as summary:
            summary.write(f"\n### {record['phase'].title()} · {record['status']}\n\n```json\n{json.dumps(record, indent=2, ensure_ascii=False, default=str)}\n```\n")


def run(mode: str) -> int:
    if not os.getenv("DATABASE_URL", "").strip():
        print("DATABASE_URL is required; refusing to create an ephemeral SQLite database.", file=sys.stderr)
        return 2
    sources = load_sources(os.getenv("GENAI_INTEL_CONFIG", "config/sources.yaml"))
    repo = Repository(os.getenv("GENAI_INTEL_DB", "data/genai_intelligence.sqlite3"))
    try:
        _phase(repo, "collect", lambda: ingest_all(repo, sources))
        rounds = 6 if mode == "bootstrap" else 1
        for _ in range(rounds):
            index_result = _phase(repo, "index", lambda: index_recent_documents(repo))
            if index_result["result"].get("status") == "waiting_for_embedding_key":
                return 1
            edition = _phase(repo, "edition", lambda: build_weekly_edition(repo))
            current_week = datetime.now(timezone.utc).strftime("%G-W%V")
            saved = repo.get_weekly_edition(current_week)
            if saved and saved["status"] == "published":
                return 0
            if edition["result"].get("status") == "waiting_for_provider":
                return 1
        # Indexing and analysis are incremental; the scheduled run resumes the saved draft.
        saved = repo.get_weekly_edition(datetime.now(timezone.utc).strftime("%G-W%V"))
        if mode == "bootstrap" and not (saved and saved["status"] == "published"):
            print("Bootstrap made progress but the first edition is not published yet; the next run can resume.", file=sys.stderr)
            return 1
        return 0
    finally:
        repo.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the independent GenAI Intelligence backend pipeline.")
    parser.add_argument("--mode", choices=("bootstrap", "daily"), default="daily")
    args = parser.parse_args(argv)
    return run(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
