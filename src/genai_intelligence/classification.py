"""Explainable article-level topic tagging for the notebook taxonomy."""
from __future__ import annotations

import re
from collections.abc import Iterable


# The stable top-level radar topics agreed for the newspaper. Keep these labels
# provider-independent so changing LLMs never changes the library taxonomy.
CATEGORY_RULES: dict[str, tuple[str, ...]] = {
    "foundation_models": (
        r"\b(llm|large language model|foundation model|language model|open[- ]weights?|model release|gpt[- ]?\d|claude|gemini|llama|qwen|mistral|deepseek|phi[- ]?\d)\b",
    ),
    "agentic_ai": (
        r"\b(agentic|\bagents?\b|multi[- ]agent|tool[- ]use|computer use|function calling|autonomous workflow)\b",
    ),
    "frameworks_tools": (
        r"\b(framework|sdk|library|developer tool|langchain|llamaindex|transformers|pytorch|tensorflow|vllm|ollama|hugging face)\b",
    ),
    "rag_retrieval": (
        r"\b(rag|retrieval[- ]augmented|retrieval|vector database|vector db|embedding search|knowledge graph|rerank(?:er|ing)?)\b",
    ),
    "benchmarks_evaluation": (
        r"\b(benchmark|evaluation|evals?|leaderboard|red[- ]team|safety test|robustness test|arena score)\b",
    ),
    "ai_research": (
        r"\b(research|arxiv|paper|preprint|研究|theorem|training method|post[- ]training|fine[- ]tun(?:e|ing)|reinforcement learning|reasoning)\b",
    ),
    "inference_infrastructure": (
        r"\b(inference|serving|infrastructure|gpu|\btpu\b|accelerator|datacenter|data center|latency|throughput|quantiz(?:ation|ed)|distillation|kernel|chip|hardware)\b",
    ),
    "startups_products": (
        r"\b(startup|product|launch(?:es|ed)?|release(?:s|d)?|available now|api|enterprise|funding|raises? \$|acquisition|partnership|pricing|customer)\b",
    ),
}

CATEGORY_LABELS = {
    "foundation_models": "Foundation models",
    "agentic_ai": "Agentic AI",
    "frameworks_tools": "Frameworks & tools",
    "rag_retrieval": "RAG & retrieval",
    "benchmarks_evaluation": "Benchmarks & evaluation",
    "ai_research": "AI research",
    "inference_infrastructure": "Inference & infrastructure",
    "startups_products": "Startups & products",
}

_COMPILED = {name: tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)
             for name, patterns in CATEGORY_RULES.items()}


def classify_article(*, title: str = "", summary: str = "", content: str = "",
                     tags: Iterable[str] = ()) -> list[str]:
    """Return matching topical categories in a fixed, explainable order.

    Source-wide labels are deliberately excluded by callers. Explicit article
    tags and the item's own text are useful evidence; otherwise AI research is
    the conservative default for this AI-only source library.
    """
    text = " ".join((title, summary, content, " ".join(str(tag) for tag in tags))).casefold()
    matches = [category for category, patterns in _COMPILED.items()
               if any(pattern.search(text) for pattern in patterns)]
    return matches or ["ai_research"]

