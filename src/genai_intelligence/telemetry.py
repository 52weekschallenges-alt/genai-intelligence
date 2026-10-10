"""Small persistent telemetry for outbound model API requests."""
from __future__ import annotations

import logging
import os
import json
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def _safe_error_detail(exc: Exception) -> str:
    """Keep provider diagnostics useful without saving response or prompt bodies."""
    response = getattr(exc, "response", None)
    if response is not None:
        detail = ""
        try:
            payload = response.json()
            if isinstance(payload, dict):
                error = payload.get("error", payload)
                if isinstance(error, dict):
                    detail = str(error.get("message") or error.get("status") or error.get("code") or "")
        except Exception:
            pass
        if detail:
            detail = " ".join(detail.split())
            return f"HTTP {response.status_code}: {detail[:400]}"
        return f"HTTP {response.status_code}: {getattr(response, 'reason_phrase', 'request failed')}"
    if isinstance(exc, json.JSONDecodeError):
        return f"Provider returned invalid JSON (line {exc.lineno}, column {exc.colno})"
    message = " ".join(str(exc).split())
    return message[:500] or type(exc).__name__


@contextmanager
def track_external_call(*, service: str, operation: str, model: str, endpoint: str):
    """Record one model/embedding request, excluding credentials and prompt contents."""
    started = time.monotonic()
    trace: dict[str, int | str | None] = {"http_status": None, "error_detail": None}
    outcome = "success"
    error_type = None
    try:
        yield trace
    except Exception as exc:
        outcome = "error"
        error_type = type(exc).__name__
        trace["error_detail"] = _safe_error_detail(exc)
        response = getattr(exc, "response", None)
        if response is not None:
            trace["http_status"] = response.status_code
        raise
    finally:
        try:
            from .db import Repository

            repo = Repository(os.getenv("GENAI_INTEL_DB", "data/genai_intelligence.sqlite3"))
            try:
                repo.record_external_request(
                    str(uuid.uuid4()), created_at=datetime.now(timezone.utc).isoformat(),
                    service=service, operation=operation, model=model, endpoint=endpoint,
                    outcome=outcome, http_status=trace.get("http_status"),
                    latency_ms=round((time.monotonic() - started) * 1000), error_type=error_type,
                    error_detail=trace.get("error_detail"),
                )
            finally:
                repo.close()
        except Exception as exc:
            # Telemetry must not turn a successful provider response into a failed job.
            logger.warning("Could not persist outbound request telemetry (%s)", type(exc).__name__)
