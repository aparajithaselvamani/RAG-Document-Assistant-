"""Second-stage reranking of retrieved chunks with a cross-encoder.

The first stage (semantic + BM25 hybrid search) scores the question and each
chunk *separately* (embeddings, word statistics).  A cross-encoder reads the
question and a chunk *together* and outputs one relevance score, which is far
more precise but too slow to run over a whole corpus.  So the pipeline is:

    hybrid search -> up to RERANK_CANDIDATES chunks -> cross-encoder -> top_k

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (trained on MS MARCO passage
ranking, ~90 MB, ~20 ms for 8 chunks on CPU).  It is loaded from
models/ms-marco-MiniLM-L-6-v2 when that folder exists, otherwise downloaded
from the Hugging Face hub on first use.  If it cannot be loaded, reranking is
skipped and the hybrid ranking is used unchanged.

Scores are logits: > 0 means "probably relevant", around -11 means "clearly
unrelated".  A candidate is dropped when its score is below RERANK_MIN_SCORE.
Returned scores are the sigmoid of the logit (0..1) so they read like the other
scores in the app.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from langchain_core.documents import Document

BASE_DIR = Path(__file__).resolve().parent
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_LOCAL_PATH = BASE_DIR / "models" / "ms-marco-MiniLM-L-6-v2"

ENABLE_RERANKING = True
# How many first-stage candidates the cross-encoder gets to choose from.
RERANK_CANDIDATES = 8
# Logit cutoff.  Answer-bearing chunks in the evaluation set scored >= -4.0;
# chunks that slipped through the first stage for out-of-scope questions
# ("Explain the Python programming language.") scored about -10.7.  Results were
# the same for cutoffs from -4.5 to -8 (see RERANKING_EVALUATION.md).
RERANK_MIN_SCORE = -6.0


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


class CrossEncoderReranker:
    """Score (query, chunk) pairs with a cross-encoder and keep the relevant ones in score order."""

    def __init__(self, model: object, min_score: float = RERANK_MIN_SCORE) -> None:
        self.model = model
        self.min_score = min_score

    def score(self, query: str, documents: Sequence[Document]) -> List[float]:
        if not documents:
            return []
        pairs = [(query, document.page_content) for document in documents]
        return [float(score) for score in self.model.predict(pairs, show_progress_bar=False)]

    def rerank(
        self,
        query: str,
        results: Sequence[Tuple[Document, float]],
        top_k: int | None = None,
    ) -> List[Tuple[Document, float]]:
        """Reorder first-stage results by cross-encoder relevance and drop weak matches.

        Each kept document is returned as a copy with ``rerank_score`` (logit)
        and ``first_stage_score`` added to its metadata; the paired score is
        sigmoid(logit).
        """
        documents = [document for document, _ in results]
        if not documents:
            return []
        logits = self.score(query, documents)

        scored = []
        for (document, first_stage_score), logit in zip(results, logits):
            if logit < self.min_score:
                continue
            metadata = dict(document.metadata or {})
            metadata["rerank_score"] = round(logit, 4)
            metadata["first_stage_score"] = round(float(first_stage_score), 4)
            scored.append((Document(page_content=document.page_content, metadata=metadata), logit))

        scored.sort(key=lambda item: (-item[1], str(item[0].metadata.get("source") or ""), item[0].metadata.get("chunk_index", 0)))
        if top_k is not None:
            scored = scored[:top_k]
        return [(document, _sigmoid(logit)) for document, logit in scored]


_reranker: Optional[CrossEncoderReranker] = None
_load_failed = False


def get_reranker(force: bool = False) -> Optional[CrossEncoderReranker]:
    """Return the shared reranker, loading the model on first use.

    Returns None when reranking is disabled (unless ``force``) or the model cannot be loaded.
    """
    global _reranker, _load_failed
    if (not ENABLE_RERANKING and not force) or _load_failed:
        return None
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder

            source = str(RERANKER_LOCAL_PATH) if RERANKER_LOCAL_PATH.exists() else RERANKER_MODEL_NAME
            _reranker = CrossEncoderReranker(CrossEncoder(source, device="cpu"))
        except Exception as exc:  # no network on first run, missing package, ...
            print(f"Reranker unavailable, using hybrid ranking only: {exc}")
            _load_failed = True
            return None
    return _reranker
