# GenAI Intelligence V0.5

A free-first, source-linked GenAI weekly notebook. It collects public RSS/Atom feeds and public GitHub, Hugging Face Hub, and arXiv API results, then selects a compact seven-day briefing with explainable rules. A scheduled publisher prepares one story per day into a saved weekly edition using Groq first and Gemini second. There is no paid-provider fallback; free-tier availability and quotas depend on the provider accounts.

## What is included

- A YAML source registry at `config/sources.yaml`, including the four requested feeds, additional official/editorial feeds, and API-backed sources.
- A shared article model with source, identity, title, canonical URL, summary/content, author, timestamps, categories, trust score, and source metadata.
- SQLite tables for sources, articles, and ingestion runs, with indexes and idempotent upserts.
- RSS/Atom, GitHub Search API, Hugging Face Models API, and arXiv Atom API collectors.
- A daily briefing of up to 20 items from the last seven days, ranked by freshness, source trust, headline signals, and available community engagement.
- Duplicate-title and URL suppression plus a three-story-per-source cap to make the briefing more varied. Each selected story shows the rules that helped it qualify.
- A full archive remains available from the **All articles** view; items are not deleted by curation.
- The weekly newspaper excludes Hugging Face model-catalog and GitHub repository-search records, plus any item without at least 80 characters of readable source text. Those directory records remain in the underlying library; they are not presented as news stories.
- Articles are classified individually into the agreed eight radar topics: foundation models, agentic AI, frameworks/tools, RAG/retrieval, benchmarks/evaluation, AI research, inference/infrastructure, and startups/products. The site exposes these as topic navigation; source-wide labels no longer determine every article's topic.
- Topic assignment currently uses transparent keyword rules over each article's title, abstract/summary, body, and feed tags. Existing database rows are reclassified once on first startup after this update. Items without a strong keyword match fall into AI research so every document stays discoverable.
- A rolling seven-day reader labeled **This week**, plus a full library archive.
- A maintained document index: the daily indexer fetches article pages (and arXiv HTML/PDF when available), extracts readable text, stores the full document, chunks it, and creates 768-dimensional Gemini embeddings. PostgreSQL stores vectors in Neon pgvector; local SQLite stores vectors as JSON for development. It works through the eligible archive in recency order and revisits documents periodically for source changes.
- A saved weekly edition containing up to six selected developments with 500–800-word LLM analyses grounded in indexed full text. The daily workflow collects, indexes, then analyzes documents; it saves progress and publishes the issue only when every included analysis is ready. Groq is tried first and Gemini second.
- Similarity search over indexed chunks retrieves related passages from other documents as optional context. The analysis labels sources and distinguishes retrieved context from direct evidence.
- Private read-to-study workflow: marking an item complete saves reading progress and generates a grounded learning note, a short multiple-choice quiz, and flashcards. Study material is cached per article so it is generated once and reused.
- Groq-first, Gemini-second provider fallback. Failed requests and free-quota exhaustion fall through to the next configured provider; when both fail, no paid service is attempted and the item can be retried later.
- A CLI for listing sources, ingesting, browsing recent articles, and checking database counts, plus a searchable web reader and JSON API.
- A Dockerfile and Docker Compose configuration for hosting with persistent storage and daily scheduled collection.
- Existing tests cover source loading, URL normalization, idempotency, and collection failures.

## Run locally

Requires Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
python -m pip install -e '.[dev]'
cp .env.example .env             # optional; credentials are not required
genai-intel sources
genai-intel ingest
genai-intel articles --limit 20
export GEMINI_API_KEY=your_key_here   # required for document embeddings
export GROQ_API_KEY=your_key_here    # optional; Gemini also generates analyses
# Run after collection; each run indexes up to eight eligible documents.
genai-intel index
genai-intel edition
genai-intel stats
uvicorn genai_intelligence.web:app --reload
```

The local sequence is collect (`genai-intel ingest`), build the document/vector index (`genai-intel index`), generate or resume long-form analyses (`genai-intel edition`), then start the reader. The index command processes at most eight eligible documents per run. `genai-intel stats` reports the document, chunk, and analysis counts. The reader is available at `http://127.0.0.1:8000`; its default page shows only completed, indexed deep reads for the current week, while **All articles** opens the raw article archive. The JSON API defaults to the briefing at `/api/articles`; use `/api/articles?edition=all` for the archive. `/health` is the health check. The web service starts collection on startup and repeats every 24 hours. Set `ENABLE_INGESTION=false` to serve only already-collected data, or change `INGEST_INTERVAL_HOURS`. The command defaults to `config/sources.yaml` and `data/genai_intelligence.sqlite3`. Override these with `--config`, `--db`, `GENAI_INTEL_CONFIG`, or `GENAI_INTEL_DB`. A `GITHUB_TOKEN` or `HF_TOKEN` environment variable is optional and only raises public API rate limits. No `.env` loader is required; export values in your shell or use your preferred dotenv tool.

For private study-kit and reading-progress endpoints, set `NOTEBOOK_ACCESS_TOKEN` to a long random secret. The browser asks for this token on the first use and stores it in local browser storage. Add `GROQ_API_KEY` and/or `GEMINI_API_KEY`; the app tries Groq first, then Gemini. For a strict $0 setup, keep both provider accounts on their free tiers and do not attach billing that would allow paid usage. Provider quota exhaustion never triggers a paid provider or paid model fallback.

## Run as a hosted web service

The repository includes a Docker image and Compose setup. Compose keeps the SQLite database in a named persistent volume and runs the web reader and scheduled collector together:

```bash
docker compose up --build
```

Open `http://localhost:8000`. Check `http://localhost:8000/health` and `http://localhost:8000/api/articles`. Stop the service with `Ctrl+C`; the named `newspaper-data` volume keeps collected data across container rebuilds.

To publish externally, deploy this Dockerfile to a container host that supports persistent disks or volumes. Mount persistent storage at `/data`, expose the configured `PORT` (default `8000`), and set `GENAI_INTEL_DB=/data/genai_intelligence.sqlite3`. Keep `GENAI_INTEL_CONFIG=/app/config/sources.yaml`, `ENABLE_INGESTION=true`, and `INGEST_ON_STARTUP=true`. The service serves a public, read-only API and reader page; it has no accounts or private user data. Use one service instance with SQLite; multiple replicas need a shared database such as PostgreSQL and a separate scheduler design.

### Deploy on Vercel (free personal project)

1. Put this repository on GitHub, GitLab, or Bitbucket and import it into Vercel. You can make the app source public/open-source; Vercel itself is a hosted service, not open-source software. You do not need to install Vercel tooling on your computer for Git based deployment.
2. Create a Neon Postgres database using the free plan and copy its pooled connection string into the Vercel project's `DATABASE_URL` environment variable.
3. Add a random secret as `CRON_SECRET` in Vercel project environment variables. Add `NOTEBOOK_ACCESS_TOKEN`, `GROQ_API_KEY`, and `GEMINI_API_KEY` there too. Gemini is required for document embeddings; Groq/Gemini are used for the long-form analysis with Groq first. Do not commit any real secrets to the repository. Leave `ENABLE_INGESTION` unset or set it to `false`; Vercel Cron handles collection.
4. Deploy the project. The Vercel config collects sources at 02:00 UTC, indexes them at 03:00 UTC, and advances long-form edition creation at 04:00 UTC. Each cron runs daily and resumes the saved weekly draft if a provider quota is temporarily exhausted.
5. Visit the deployment URL and `/issues`; a six-story edition is normally completed across two successful daily generation runs (three long analyses per run). The indexer processes up to eight eligible documents per run, then continues through the archive on later days. The current week displays only completed long-form analyses as they are prepared. Stories without readable full text are excluded rather than shown as short feed excerpts. Indexing and analysis run on Vercel Cron using the server-side `CRON_SECRET`; notebook users do not need an operational token to keep the pipeline running.

In Vercel, `DATABASE_URL` switches the repository layer from local SQLite to Postgres so articles persist across serverless invocations. Vercel's free Hobby plan is for personal, non-commercial projects; plan limits and Neon free allowances can change, so check their current terms before public growth. [Vercel Python/FastAPI support](https://vercel.com/docs/functions/runtimes/python), [Vercel Hobby plan](https://vercel.com/docs/plans/hobby), [Vercel Cron limits](https://vercel.com/docs/cron-jobs/manage-cron-jobs), [Neon integration](https://vercel.com/marketplace/neon).


Run the test suite with `python -m pytest`.

## Configure sources

Each `sources` entry has required `id`, `name`, `type`, and `url` fields. `type` is one of `rss`, `github`, `huggingface`, or `arxiv`. `categories`, `trust_score` (0–1), and `enabled` are optional. Other fields are passed to the relevant collector, such as `query`, `per_page`, `task`, or `max_results`. Invalid/duplicate IDs and out-of-range trust scores fail registry loading. HTTP URLs for `export.arxiv.org` are upgraded to HTTPS.

Collection errors are isolated per source, recorded in `ingestion_runs`, reported in the CLI output, and cause `ingest` to return a nonzero status if any source failed. Re-running a collection updates records by source identity and avoids duplicating them. URL canonicalization removes fragments and common tracking parameters. Similar stories arriving from different sources remain distinct records so their provenance is retained.

## Topic classification and curation

The database keeps the complete normalized archive. The seven-day view favors freshness and higher configured source trust, boosts selected high-signal headline terms, uses engagement metadata when available, removes exact and near-duplicate headlines, and caps each source at three stories. Selection rules and topic tags are deterministic and explainable. Topic tagging is article-level and can assign multiple radar topics. Embeddings support semantic retrieval across the indexed document library; they are not used as the topic classifier yet. The weekly notebook analysis is grounded in extracted full-text pages, with source scope and links retained. Mark-as-read quiz/flashcards remain a separate on-demand study kit. `trust_score` is a source default, not a claim that an article has been editorially reviewed.
