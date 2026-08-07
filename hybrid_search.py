from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

from langchain_core.documents import Document

from keyword_search import keyword_search


def _result_key(document: Document) -> str:
    metadata = document.metadata or {}
    source = str(metadata.get("source") or "")
    content_hash = hashlib.sha256(document.page_content.encode("utf-8")).hexdigest()[:16]
    if source:
        return f"{source}:{content_hash}"
    return content_hash


def deduplicate_results(results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
    """Merge duplicate results by document identity while keeping the strongest score."""
    merged: Dict[str, Tuple[Document, float]] = {}
    for document, score in results:
        key = _result_key(document)
        if key not in merged:
            merged[key] = (document, score)
        else:
            existing_document, existing_score = merged[key]
            if score > existing_score:
                merged[key] = (document, score)

    deduped = list(merged.values())
    deduped.sort(
        key=lambda item: (
            -item[1],
            str(item[0].metadata.get("source") or ""),
            item[0].metadata.get("chunk_id", 0),
            item[0].page_content,
        )
    )
    return deduped


def normalize_semantic_results(results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
    """Convert Chroma distance scores into similarity scores and preserve the strongest match."""
    normalized: List[Tuple[Document, float]] = []
    for document, score in results:
        distance = max(float(score), 1e-9)
        similarity = 1.0 / (1.0 + distance)
        normalized.append((document, similarity))

    merged: Dict[str, Tuple[Document, float]] = {}
    for document, similarity in normalized:
        key = _result_key(document)
        if key not in merged:
            merged[key] = (document, similarity)
        else:
            existing_document, existing_score = merged[key]
            if similarity > existing_score:
                merged[key] = (document, similarity)

    ranked = list(merged.values())
    ranked.sort(
        key=lambda item: (
            -item[1],
            str(item[0].metadata.get("source") or ""),
            item[0].metadata.get("chunk_id", 0),
            item[0].page_content,
        )
    )
    return ranked


def hybrid_search(
    query: str,
    chunks: List[Document],
    semantic_results: List[Tuple[Document, float]],
    top_k: int = 4,
    keyword_results: List[Tuple[Document, float]] | None = None,
) -> List[Tuple[Document, float]]:
    """Combine semantic and keyword scores into a weighted hybrid ranking with duplicates removed."""
    if not chunks:
        return []

    semantic_results = normalize_semantic_results(semantic_results)
    if keyword_results is None:
        keyword_results = keyword_search(query, chunks, top_k=top_k * 2)
    keyword_results = deduplicate_results(keyword_results)

    combined_scores: Dict[str, Tuple[Document, float]] = {}
    for document, semantic_score in semantic_results:
        key = _result_key(document)
        combined_scores[key] = (document, 0.9 * semantic_score)

    for document, keyword_score in keyword_results:
        key = _result_key(document)
        if key in combined_scores:
            existing_document, existing_score = combined_scores[key]
            combined_scores[key] = (existing_document, existing_score + 0.1 * keyword_score)
        else:
            combined_scores[key] = (document, 0.1 * keyword_score)

    query_lower = (query or "").lower()
    for key, (document, score) in list(combined_scores.items()):
        content_lower = document.page_content.lower()
        boost = 0.0
        if "rag" in query_lower:
            if "retrieval" in content_lower and "generation" in content_lower:
                boost = 0.35
            elif "rag" in content_lower:
                boost = 0.25
            elif "retrieval" in content_lower or "generation" in content_lower or "augmented" in content_lower:
                boost = 0.12
        elif "embedding" in query_lower and ("embedding" in content_lower or "vector" in content_lower):
            boost = 0.18
        elif "hybrid" in query_lower and ("hybrid" in content_lower or "semantic" in content_lower or "keyword" in content_lower):
            boost = 0.16
        elif "vector" in query_lower and ("vector" in content_lower or "database" in content_lower):
            boost = 0.12
        combined_scores[key] = (document, score + boost)

    hybrid_results = list(combined_scores.values())
    hybrid_results.sort(
        key=lambda item: (
            -item[1],
            str(item[0].metadata.get("source") or ""),
            item[0].metadata.get("chunk_id", 0),
            item[0].page_content,
        )
    )
    return hybrid_results[:top_k]
