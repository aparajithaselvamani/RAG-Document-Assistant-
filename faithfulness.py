"""Sentence-level diagnostic for whether generated answers match retrieved evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from ingest import split_into_sentences

SUPPORTED_THRESHOLD = 0.75
WEAKLY_SUPPORTED_THRESHOLD = 0.45

# Calibrated with all-MiniLM-L6-v2 on project document text: close grounded
# paraphrases scored 0.927-0.948, an exact sentence scored 1.000, a grounded
# compressed paraphrase scored 0.577, and unrelated fact pairs scored -0.030
# and 0.007. The gap motivates 0.75 for strong evidence and 0.45 for weak.
SentenceVerdict = Literal["SUPPORTED", "WEAKLY_SUPPORTED", "UNSUPPORTED"]
FaithfulnessVerdict = Literal["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"]


@dataclass(frozen=True)
class SentenceFaithfulness:
    sentence: str
    similarity: float
    verdict: SentenceVerdict


@dataclass(frozen=True)
class FaithfulnessResult:
    verdict: FaithfulnessVerdict
    sentences: list[SentenceFaithfulness]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding dimensions do not match")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def check_faithfulness(
    answer: str,
    context_chunks: list[str],
    embedding_model: Any,
) -> FaithfulnessResult:
    """Compare each answer sentence with retrieved sentence/chunk embeddings.

    ``embedding_model`` is the existing LangChain embeddings object attached to
    the vector store; this function never initializes or downloads a model.
    """
    answer_sentences = split_into_sentences((answer or "").strip())
    context_sentences = [
        sentence
        for chunk in context_chunks
        for sentence in split_into_sentences((chunk or "").strip())
    ]
    if not answer_sentences:
        return FaithfulnessResult("UNSUPPORTED", [])
    if not context_sentences:
        return FaithfulnessResult(
            "UNSUPPORTED",
            [SentenceFaithfulness(sentence, 0.0, "UNSUPPORTED") for sentence in answer_sentences],
        )

    vectors = embedding_model.embed_documents(answer_sentences + context_sentences)
    answer_vectors = vectors[: len(answer_sentences)]
    context_vectors = vectors[len(answer_sentences):]
    results: list[SentenceFaithfulness] = []
    for sentence, answer_vector in zip(answer_sentences, answer_vectors):
        best_score = max(_cosine_similarity(answer_vector, context_vector) for context_vector in context_vectors)
        if best_score >= SUPPORTED_THRESHOLD:
            verdict: SentenceVerdict = "SUPPORTED"
        elif best_score >= WEAKLY_SUPPORTED_THRESHOLD:
            verdict = "WEAKLY_SUPPORTED"
        else:
            verdict = "UNSUPPORTED"
        results.append(SentenceFaithfulness(sentence, best_score, verdict))

    high_support_count = sum(item.verdict == "SUPPORTED" for item in results)
    any_support = any(item.verdict != "UNSUPPORTED" for item in results)
    if high_support_count == len(results):
        overall: FaithfulnessVerdict = "SUPPORTED"
    elif any_support:
        overall = "PARTIALLY_SUPPORTED"
    else:
        overall = "UNSUPPORTED"
    return FaithfulnessResult(overall, results)
