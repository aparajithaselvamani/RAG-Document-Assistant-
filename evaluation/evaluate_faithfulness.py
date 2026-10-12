"""Run diagnostic faithfulness checks on labeled examples and RAG evaluations."""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import app  # noqa: E402
from evaluation.evaluate_rag import load_evaluation_cases  # noqa: E402
from faithfulness import check_faithfulness  # noqa: E402


def main() -> int:
    store = app.load_vector_store(app.VECTOR_DB_DIR)
    cases = load_evaluation_cases()
    selected = [
        next(case for case in cases if case["id"] == case_id)
        for case_id in ("direct_02", "direct_04", "no_info_01")
    ]
    rows = []
    for case in selected:
        history = deque(maxlen=app.MAX_CONVERSATION_TURNS)
        _, _, _, retrieved = app.retrieve_context(store, case["question"], conversation_history=history)
        context = [document.page_content for document, _ in retrieved]
        # Use a verbatim retrieved passage as a deterministic supported-answer
        # fixture; this script evaluates the checker, not Ollama variability.
        answer = context[0] if context else app.NO_INFORMATION_RESPONSE
        faithfulness = check_faithfulness(answer, context, store.embeddings)
        expected = "UNSUPPORTED" if case["type"] == "no_relevant_information" else "SUPPORTED"
        rows.append((case["id"], expected, faithfulness.verdict))

    search_context = Path(ROOT_DIR / "documents" / "search_methods.txt").read_text(encoding="utf-8-sig").strip()
    grounded_sentence = "Semantic search uses vector similarity to find conceptually related passages."
    examples = [
        ("labeled_supported", grounded_sentence, [search_context], "SUPPORTED"),
        ("labeled_partial", grounded_sentence + " Quantum computers use qubits.", [search_context], "PARTIALLY_SUPPORTED"),
        ("labeled_unsupported", "Quantum computers use qubits.", [search_context], "UNSUPPORTED"),
    ]
    for case_id, answer, context, expected in examples:
        result = check_faithfulness(answer, context, store.embeddings)
        rows.append((case_id, expected, result.verdict))

    print("\nFAITHFULNESS EVALUATION (diagnostic only)")
    print(f"{'case':24} {'expected':22} {'actual':22} result")
    print("-" * 78)
    for case_id, expected, actual in rows:
        print(f"{case_id:24} {expected:22} {actual:22} {'PASS' if expected == actual else 'MISMATCH'}")
    return 0 if all(expected == actual for _, expected, actual in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
