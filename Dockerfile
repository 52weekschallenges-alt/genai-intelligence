FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GENAI_INTEL_CONFIG=/app/config/sources.yaml \
    GENAI_INTEL_DB=/data/genai_intelligence.sqlite3 \
    PORT=8000

WORKDIR /app
COPY pyproject.toml README.md setup.py ./
COPY src ./src
COPY config ./config
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir . \
    && mkdir -p /data \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /data

USER appuser
EXPOSE 8000
VOLUME ["/data"]
CMD ["sh", "-c", "uvicorn genai_intelligence.web:app --host 0.0.0.0 --port ${PORT:-8000}"]
