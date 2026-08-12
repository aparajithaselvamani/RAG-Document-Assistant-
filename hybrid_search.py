from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

from langchain_core.documents import Document

from keyword_search import keyword_search


def _result_key(document: Document) -> str:
    metadata = document.metadata or {}
    source = str(metadata.get("source") or "")
    content_hash = hashlib.sha256(document.page_content.encode("utf-8")).hexdigest()[:16]
    return f"{source}:{content_hash}" if source else content_hash


def deduplicate_results(results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
    """Keep one copy of each chunk, retaining its strongest relevance score."""
    merged: Dict[str, Tuple[Document, float]] = {}
    for document, score in results:
        key = _result_key(document)
        if key not in merged or score > merged[key][1]:
            merged[key] = (document, float(score))
    return _sort_results(list(merged.values()))


def _sort_results(results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
    return sorted(
        results,
        key=lambda item: (-item[1], str(item[0].metadata.get("source") or ""), item[0].metadata.get("chunk_id", 0), item[0].page_content),
    )


def normalize_semantic_results(results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
    """Convert Chroma distances (lower is better) to a 0..1 similarity score."""
    normalized = [(document, 1.0 / (1.0 + max(0.0, float(distance)))) for document, distance in results]
    return deduplicate_results(normalized)


def _normalize_scores(results: List[Tuple[Document, float]]) -> Dict[str, float]:
    """Normalize non-negative relevance scores by the strongest candidate."""
    strongest = max((score for _, score in results), default=0.0)
    if strongest <= 0:
        return {}
    return {_result_key(document): max(0.0, score) / strongest for document, score in results}


def hybrid_search(
    query: str,
    chunks: List[Document],
    semantic_results: List[Tuple[Document, float]],
    top_k: int = 4,
    keyword_results: List[Tuple[Document, float]] | None = None,
) -> List[Tuple[Document, float]]:
    """Combine Chroma distance and lexical relevance without filling results with noise.

    ``semantic_results`` contains raw Chroma distances.  It is intentionally
    normalized exactly once here; lower distances become higher similarities.
    A chunk needs lexical evidence or to be close to the best semantic match to
    qualify, so a merely returned nearest neighbour is not automatically shown.
    """
    if not chunks or top_k <= 0:
        return []

    semantic = normalize_semantic_results(semantic_results)
    keyword = deduplicate_results(keyword_results if keyword_results is not None else keyword_search(query, chunks, top_k=top_k * 2))
    semantic_map = {_result_key(document): (document, score) for document, score in semantic}
    keyword_map = {_result_key(document): (document, score) for document, score in keyword}
    keyword_normalized = _normalize_scores(keyword)
    best_semantic = max((score for _, score in semantic), default=0.0)

    ranked: List[Tuple[Document, float]] = []
    for key in set(semantic_map) | set(keyword_map):
        semantic_entry = semantic_map.get(key)
        keyword_entry = keyword_map.get(key)
        document = semantic_entry[0] if semantic_entry else keyword_entry[0]
        semantic_score = semantic_entry[1] if semantic_entry else 0.0
        keyword_score = keyword_normalized.get(key, 0.0)

        # Lexical matches are explicit evidence. Pure semantic candidates must
        # be near the best match instead of being included just because top-k
        # vector search always returns neighbours.
        has_keyword_evidence = key in keyword_map
        semantically_competitive = best_semantic > 0 and semantic_score >= max(0.35, best_semantic * 0.88)
        if not has_keyword_evidence and not semantically_competitive:
            continue

        ranked.append((document, 0.65 * semantic_score + 0.35 * keyword_score))

    return _sort_results(ranked)[:top_k]
