from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Tuple

from langchain_core.documents import Document

from keyword_search import content_terms, keyword_search


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


# --- Relevance gate -------------------------------------------------------
# Semantic scores are 1 / (1 + squared L2 distance).  For the normalised
# MiniLM embeddings that value equals cosine similarity at 0.5, so the floor
# below reads as "cosine similarity of at least 0.5".
SEMANTIC_FLOOR = 0.5
# A chunk must also be close to the best match: noise chunks that merely share
# a generic word ("search", "retrieval", "and") sat 0.12-0.2 below the best one.
SEMANTIC_MARGIN = 0.1
# Rare exact terms (e.g. "BM25") or full coverage of the query's content words
# can rescue a chunk that embeddings rank low.
RARE_TERM_MAX_DOC_SHARE = 0.05
RARE_TERM_SEMANTIC_FLOOR = 0.35
SEMANTIC_WEIGHT = 0.65
KEYWORD_WEIGHT = 0.35


def _rare_query_terms(query: str, chunks: List[Document]) -> set[str]:
    """Query content terms that occur in very few chunks (at least one)."""
    terms = set(content_terms(query))
    if not terms:
        return set()
    doc_freq = {term: 0 for term in terms}
    for chunk in chunks:
        chunk_terms = set(content_terms(chunk.page_content))
        for term in terms & chunk_terms:
            doc_freq[term] += 1
    limit = max(1, int(len(chunks) * RARE_TERM_MAX_DOC_SHARE))
    return {term for term, frequency in doc_freq.items() if 0 < frequency <= limit}


def hybrid_search(
    query: str,
    chunks: List[Document],
    semantic_results: List[Tuple[Document, float]],
    top_k: int = 4,
    keyword_results: List[Tuple[Document, float]] | None = None,
) -> List[Tuple[Document, float]]:
    """Combine Chroma distance and BM25 relevance, admitting only chunks with real evidence.

    ``semantic_results`` contains raw Chroma distances and is normalised once here.

    Previously *any* shared token counted as keyword evidence and let a chunk in,
    so "What is RAG?" also returned the query-rewriting PDF (shared word
    "retrieval") and a question containing "and" matched unrelated chunks.  Now a
    chunk qualifies when it is
      1. semantically relevant in absolute terms (>= SEMANTIC_FLOOR) and close to
         the best semantic match (within SEMANTIC_MARGIN), or
      2. lexically decisive while no chunk is a confident semantic match (keyword
         rescue): it contains a rare exact query term such as "BM25", or every
         content word of the query, and has at least a weak semantic relation
         (>= RARE_TERM_SEMANTIC_FLOOR).
    Qualifying chunks are ranked by 0.65 * semantic + 0.35 * normalised BM25.
    """
    if not chunks or top_k <= 0:
        return []

    semantic = normalize_semantic_results(semantic_results)
    keyword = deduplicate_results(keyword_results if keyword_results is not None else keyword_search(query, chunks, top_k=top_k * 2))
    semantic_map = {_result_key(document): (document, score) for document, score in semantic}
    keyword_map = {_result_key(document): (document, score) for document, score in keyword}
    keyword_normalized = _normalize_scores(keyword)
    best_semantic = max((score for _, score in semantic), default=0.0)
    rare_terms = _rare_query_terms(query, chunks)
    query_terms = set(content_terms(query))
    # Keyword rescue is a fallback for when embeddings find nothing convincing
    # (acronyms like "BM25", "where"-questions).  With a confident semantic match,
    # extra lexical-only chunks were mostly noise (e.g. the ChromaDB sentence for
    # "What is semantic search?").
    confident_semantic_match = best_semantic >= SEMANTIC_FLOOR

    candidates: Dict[str, Tuple[Document, float, float, bool]] = {}
    for key in set(semantic_map) | set(keyword_map):
        semantic_entry = semantic_map.get(key)
        keyword_entry = keyword_map.get(key)
        document = semantic_entry[0] if semantic_entry else keyword_entry[0]
        semantic_score = semantic_entry[1] if semantic_entry else 0.0
        keyword_score = keyword_normalized.get(key, 0.0)

        semantically_relevant = semantic_score >= SEMANTIC_FLOOR and semantic_score >= best_semantic - SEMANTIC_MARGIN
        chunk_terms = set(content_terms(document.page_content))
        lexical_rescue = (
            not confident_semantic_match
            and key in keyword_map
            and semantic_score >= RARE_TERM_SEMANTIC_FLOOR
            and (bool(rare_terms & chunk_terms) or (bool(query_terms) and query_terms <= chunk_terms))
        )
        candidates[key] = (document, semantic_score, keyword_score, semantically_relevant or lexical_rescue)

    # When only lexical rescue identified the right document, its other chunks
    # that are about as close semantically are admitted too.  Otherwise a
    # paraphrased question ("notice" vs "in advance") gets the sentence with the
    # matching word but not the sentence holding the answer.
    rescued_sources = {str((doc.metadata or {}).get("source") or "") for doc, _, _, ok in candidates.values() if ok}
    ranked: List[Tuple[Document, float]] = []
    for document, semantic_score, keyword_score, qualified in candidates.values():
        if not qualified and not confident_semantic_match:
            same_source = str((document.metadata or {}).get("source") or "") in rescued_sources
            qualified = same_source and semantic_score >= RARE_TERM_SEMANTIC_FLOOR and semantic_score >= best_semantic - SEMANTIC_MARGIN
        if qualified:
            ranked.append((document, SEMANTIC_WEIGHT * semantic_score + KEYWORD_WEIGHT * keyword_score))

    return _sort_results(ranked)[:top_k]


# --- Context expansion ------------------------------------------------------
# Sentence-level chunks are precise but can be incomplete: "Hybrid search
# combines both approaches ..." is meaningless without the two sentences before
# it.  When a selected chunk refers back like this, its preceding chunk(s) from
# the same document are added so the answer model sees the full definition.
TWO_BACK_REFERENCE_RE = re.compile(r"\b(?:both|the two|the former|the latter)\b", re.IGNORECASE)
ONE_BACK_REFERENCE_RE = re.compile(
    r"^(?:these|this|those|it|its|they|their|such)\b|\b(?:these|those|the above|this approach|that approach)\b",
    re.IGNORECASE,
)


def _previous_chunks_needed(text: str) -> int:
    if TWO_BACK_REFERENCE_RE.search(text):
        return 2
    if ONE_BACK_REFERENCE_RE.search(text):
        return 1
    return 0


def expand_with_context(results: List[Tuple[Document, float]], chunks: List[Document]) -> List[Tuple[Document, float]]:
    """Add back-referenced neighbour chunks and merge adjacent chunks into passages.

    Results keep their ranking order by best score; within a passage, text is in
    document order.  Chunks without ``chunk_index`` metadata are passed through.
    """
    if not results:
        return []

    by_position: Dict[Tuple[str, int], Document] = {}
    for chunk in chunks:
        metadata = chunk.metadata or {}
        if metadata.get("chunk_index") is not None:
            by_position[(str(metadata.get("source") or ""), int(metadata["chunk_index"]))] = chunk

    selected: Dict[Tuple[str, int], float] = {}
    passthrough: List[Tuple[Document, float]] = []
    for document, score in results:
        metadata = document.metadata or {}
        if metadata.get("chunk_index") is None:
            passthrough.append((document, score))
            continue
        source, index = str(metadata.get("source") or ""), int(metadata["chunk_index"])
        selected[(source, index)] = max(score, selected.get((source, index), 0.0))
        for step in range(1, _previous_chunks_needed(document.page_content) + 1):
            neighbour = (source, index - step)
            if neighbour in by_position and neighbour not in selected:
                selected[neighbour] = 0.0  # context only; inherits the passage score below
        if (source, index) not in by_position:
            by_position[(source, index)] = document

    passages: List[Tuple[Document, float]] = []
    for source in sorted({source for source, _ in selected}):
        positions = sorted(index for src, index in selected if src == source)
        span: List[int] = []
        for position in positions + [None]:
            if span and (position is None or position != span[-1] + 1):
                parts = [by_position[(source, i)] for i in span]
                metadata = dict(parts[0].metadata or {})
                if len(span) > 1:
                    metadata["chunk_span"] = f"{span[0]}-{span[-1]}"
                text = " ".join(part.page_content for part in parts)
                passages.append((Document(page_content=text, metadata=metadata), max(selected[(source, i)] for i in span)))
                span = []
            if position is not None:
                span.append(position)

    return _sort_results(passages + passthrough)
