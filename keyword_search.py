from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Dict, List, Tuple

from langchain_core.documents import Document

# BM25 parameters (standard Okapi defaults).
BM25_K1 = 1.5
BM25_B = 0.75
PHRASE_BONUS = 0.25  # per matched adjacent query bigram, relative to the BM25 score scale

# Function words plus question scaffolding ("explain", "tell me about").  The
# previous list missed words like "and", so a query containing "and" counted as
# keyword evidence for any chunk that also contained "and".
STOP_WORDS = frozenset(
    """
    a about above after again all also am an and any are as at be because been being between both but by
    can could did do does doing done during each either explain describe define definition difference
    different few for from further give had has have having he her here hers him his how i if in into is it
    its itself just me mean means meaning more most my no nor not of off on once only or other our out over
    own please same say says she should so some such tell than that the their theirs them then there these
    they this those through to too under until up very was we were what when where which while who whom why
    will with would you your
    """.split()
)


def _stem(token: str) -> str:
    """Small suffix stripper applied identically to queries and chunks.

    Folds plural and -ed/-ing forms so 'stored'/'stores', 'passes'/'pass' and
    'embeddings'/'embedding'/'embed' meet.  It is deliberately crude (no
    dependency on nltk); consistency matters more than linguistic accuracy.
    """
    if token.isdigit() or len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        token = token[:-3] + "y"
    elif token.endswith("sses"):
        token = token[:-2]
    elif token.endswith(("xes", "zes", "ches", "shes")):
        token = token[:-2]
    elif token.endswith("s") and not token.endswith(("ss", "us", "is")):
        token = token[:-1]
    for suffix in ("ing", "ed"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)]
            if len(token) > 3 and token[-1] == token[-2] and token[-1] not in "lsz":
                token = token[:-1]  # 'embedd' -> 'embed', 'stopp' -> 'stop'
            break
    if len(token) > 3 and token.endswith("e"):
        token = token[:-1]  # 'store'/'stor(ed)' -> 'stor'
    return token


def _tokenize(text: str) -> List[str]:
    return [token for token in re.split(r"[^a-z0-9]+", (text or "").lower()) if token]


def content_terms(text: str) -> List[str]:
    """Tokens that carry meaning: lower-cased, stop words removed, plurals folded."""
    return [_stem(token) for token in _tokenize(text) if token not in STOP_WORDS]


def _result_key(document: Document) -> str:
    metadata = document.metadata or {}
    source = str(metadata.get("source") or "")
    content_hash = hashlib.sha256(document.page_content.encode("utf-8")).hexdigest()[:16]
    if source:
        return f"{source}:{content_hash}"
    return content_hash


def keyword_search(query: str, chunks: List[Document], top_k: int = 4) -> List[Tuple[Document, float]]:
    """Rank chunks with BM25 over content terms, plus a small bonus for matched query bigrams.

    Only chunks sharing at least one *content* term with the query are returned,
    and duplicate chunk texts are collapsed.
    """
    if not chunks:
        return []

    query_terms = content_terms(query)
    if not query_terms:
        return []
    unique_query_terms = list(dict.fromkeys(query_terms))
    query_bigrams = {f"{a} {b}" for a, b in zip(query_terms, query_terms[1:])}

    chunk_terms = [content_terms(chunk.page_content) for chunk in chunks]
    term_counts = [Counter(terms) for terms in chunk_terms]
    doc_freq: Counter[str] = Counter()
    for counts in term_counts:
        doc_freq.update(counts.keys())
    total_docs = len(chunks)
    avg_length = (sum(len(terms) for terms in chunk_terms) / total_docs) or 1.0

    unique_results: Dict[str, Tuple[Document, float]] = {}
    for chunk, terms, counts in zip(chunks, chunk_terms, term_counts):
        if not counts:
            continue
        length_norm = BM25_K1 * (1 - BM25_B + BM25_B * len(terms) / avg_length)
        score = 0.0
        for term in unique_query_terms:
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            idf = math.log(1 + (total_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            score += idf * frequency * (BM25_K1 + 1) / (frequency + length_norm)
        if score <= 0:
            continue
        chunk_bigrams = {f"{a} {b}" for a, b in zip(terms, terms[1:])}
        score += PHRASE_BONUS * len(query_bigrams & chunk_bigrams)

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
