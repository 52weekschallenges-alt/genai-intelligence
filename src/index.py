"""Vercel FastAPI entrypoint."""
import sys
from pathlib import Path

# Vercel bundles this entrypoint from the project root; make the src-layout
# package importable even when the project itself has not been installed.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from genai_intelligence.web import app

__all__ = ["app"]
