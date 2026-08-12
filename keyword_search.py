from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Dict, List, Tuple

from langchain_core.documents import Document


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in re.split(r"[^a-z0-9]+", text.lower()) if token]


def _result_key(document: Document) -> str:
    metadata = document.metadata or {}
    source = str(metadata.get("source") or "")
    content_hash = hashlib.sha256(document.page_content.encode("utf-8")).hexdigest()[:16]
    if source:
        return f"{source}:{content_hash}"
    return content_hash


def keyword_search(query: str, chunks: List[Document], top_k: int = 4) -> List[Tuple[Document, float]]:
    """Rank chunks with deterministic keyword scoring and duplicate removal."""
    if not chunks:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    # Keep lexical retrieval lexical. Broad technical-term expansion made a
    # query about one search method match unrelated documents about another.
    expanded_tokens = set(query_tokens)

    doc_freq: Counter[str] = Counter()
    term_doc_freq: List[Counter[str]] = []
    for chunk in chunks:
        tokens = _tokenize(chunk.page_content)
        counter = Counter(tokens)
        term_doc_freq.append(counter)
        doc_freq.update(counter.keys())

    unique_results: Dict[str, Tuple[Document, float]] = {}
    for chunk, counter in zip(chunks, term_doc_freq):
        if not counter:
            continue

        content_text = " ".join(counter.keys())
        score = 0.0
        total_terms = sum(counter.values())
        phrase = " ".join(query_tokens)
        if phrase and phrase in content_text:
            score += 1.6

        for size in range(2, min(4, len(query_tokens)) + 1):
            for index in range(len(query_tokens) - size + 1):
                phrase_candidate = " ".join(query_tokens[index:index + size])
                if phrase_candidate and phrase_candidate in content_text:
                    score += 0.3 * size

        for token in expanded_tokens:
            if token not in counter:
                continue
            tf = counter[token] / total_terms
            idf = math.log((1 + len(chunks)) / (1 + doc_freq[token])) + 1.0
            weight = 1.0
            if token in query_tokens:
                weight += 0.35
            if token in {"rag", "generation", "augmented", "embedding", "embeddings", "vector", "database", "semantic", "keyword", "hybrid"}:
                weight += 0.3
            elif token == "retrieval":
                weight += 0.05
            score += tf * idf * weight

        if any(token in counter for token in expanded_tokens):
            score += 0.15
        if score > 0:
            key = _result_key(chunk)
            existing = unique_results.get(key)
            if existing is None or score > existing[1]:
                unique_results[key] = (chunk, score)

    results = list(unique_results.values())
    results.sort(
        key=lambda item: (
            -item[1],
            str(item[0].metadata.get("source") or ""),
            item[0].metadata.get("chunk_id", 0),
            item[0].page_content,
        )
    )
    return results[:top_k]
