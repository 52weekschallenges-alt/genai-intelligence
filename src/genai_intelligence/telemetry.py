"""Small persistent telemetry for outbound model API requests."""
from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@contextmanager
def track_external_call(*, service: str, operation: str, model: str, endpoint: str):
    """Record one model/embedding request, excluding credentials and prompt contents."""
    started = time.monotonic()
    trace: dict[str, int | None] = {"http_status": None}
    outcome = "success"
    error_type = None
    try:
        yield trace
    except Exception as exc:
        outcome = "error"
        error_type = type(exc).__name__
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
                )
            finally:
                repo.close()
        except Exception as exc:
            # Telemetry must not turn a successful provider response into a failed job.
            logger.warning("Could not persist outbound request telemetry (%s)", type(exc).__name__)
