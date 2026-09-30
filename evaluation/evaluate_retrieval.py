"""Retrieval-only evaluation: scores what the pipeline hands to the LLM, no Ollama needed.

evaluate_rag.py checks that the expected source appears *somewhere* in the
results, so it cannot see irrelevant extra chunks.  This script runs the same
cases (plus a small probe set) through the real routing + retrieval code and
reports:

  hit_rate            expected source retrieved
  MRR                 1 / rank of the first chunk from the expected source
  precision           share of retrieved passages that come from the expected source
  noisy cases         cases where any other source was retrieved
  answer_in_context   expected answer keywords present in the retrieved text
                      (catches chunks that are relevant but incomplete)
  relevant_char_share share of context characters from the expected source
  no-info rejected    unsupported questions that retrieve nothing, both end to end
                      and with the classifier bypassed (retrieval alone)

Probe questions (retrieval_probe_questions.json) are paraphrases and exact-term
questions outside the main set; they are reported with the classifier bypassed
(pure retrieval quality) and end to end.

Usage:  python evaluation/evaluate_retrieval.py [--details] [--json results.json]
"""
from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import app  # noqa: E402
from evaluation import evaluate_rag as ev  # noqa: E402

PROBE_PATH = Path(__file__).with_name("retrieval_probe_questions.json")


def _source(document: Any) -> str:
    return ev.normalize_source_name((document.metadata or {}).get("source"))


def _no_llm(*_args: Any, **_kwargs: Any) -> Dict[str, str]:
    raise RuntimeError("retrieval evaluation does not call the LLM")


def evaluate_case(vector_store: Any, case: Dict[str, Any], bypass_classifier: bool = False) -> Dict[str, Any]:
    history: deque = deque(maxlen=app.MAX_CONVERSATION_TURNS)
    for prior_question in case.get("history", []):
        app.update_conversation_history(history, prior_question, "")
    routed = app.route_query(case["question"], history, vector_store=vector_store)
    bypass = app.retrieve_context(vector_store, case["question"], conversation_history=history)[3]
    passages = bypass if bypass_classifier else routed["hybrid_results"]

    sources = [_source(document) for document, _ in passages]
    context = "\n".join(document.page_content for document, _ in passages)
    row: Dict[str, Any] = {
        "id": case["id"],
        "question": case["question"],
        "query": routed["rewritten_query"],
        "category": routed["category"],
        "retrieved": [(source, round(score, 3), document.page_content) for (document, score), source in zip(passages, sources)],
        "context_chars": len(context),
    }
    expected = ev.normalize_source_name(case["expected_source"]) if case.get("expected_source") else None
    if expected:
        ranks = [index + 1 for index, source in enumerate(sources) if source == expected]
        relevant_chars = sum(len(document.page_content) for document, _ in passages if _source(document) == expected)
        row.update(
            answerable=True,
            hit=bool(ranks),
            reciprocal_rank=1 / ranks[0] if ranks else 0.0,
            precision=len(ranks) / len(sources) if sources else 0.0,
            noise_sources=sorted({source for source in sources if source != expected}),
            answer_in_context=ev.keyword_coverage(context, case.get("expected_answer_keywords", [])),
            relevant_char_share=relevant_chars / len(context) if context else 0.0,
        )
    else:
        row.update(answerable=False, rejected=not passages, rejected_by_retrieval_alone=not bypass,
                   bypass_sources=sorted({_source(document) for document, _ in bypass}))
    return row


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    answerable = [row for row in rows if row["answerable"]]
    unsupported = [row for row in rows if not row["answerable"]]

    def mean(values: List[float]) -> float:
        return round(sum(values) / len(values), 3) if values else 0.0

    return {
        "answerable_cases": len(answerable),
        "hit_rate": mean([row["hit"] for row in answerable]),
        "MRR": mean([row["reciprocal_rank"] for row in answerable]),
        "mean_precision": mean([row["precision"] for row in answerable]),
        "noisy_cases": sum(bool(row["noise_sources"]) for row in answerable),
        "answer_in_context": mean([row["answer_in_context"] for row in answerable]),
        "relevant_char_share": mean([row["relevant_char_share"] for row in answerable]),
        "mean_context_chars": mean([row["context_chars"] for row in answerable]),
        "no_info_rejected": f"{sum(row['rejected'] for row in unsupported)}/{len(unsupported)}",
        "no_info_rejected_by_retrieval_alone": f"{sum(row['rejected_by_retrieval_alone'] for row in unsupported)}/{len(unsupported)}",
    }


def _print_rows(rows: List[Dict[str, Any]]) -> None:
    for row in rows:
        clean = row["hit"] and not row["noise_sources"] and row["answer_in_context"] if row["answerable"] else row["rejected_by_retrieval_alone"]
        print(f"\n[{'OK ' if clean else 'CHECK'}] {row['id']}: {row['question']}  (query: {row['query']})")
        for source, score, text in row["retrieved"]:
            print(f"    {score:>6}  {source:32} {ev._normalize_text(text)[:90]}")
        if not row["retrieved"]:
            print("    (nothing retrieved)")
        if not row["answerable"] and row["bypass_sources"]:
            print(f"    retrieval alone (classifier bypassed) would return: {', '.join(row['bypass_sources'])}")


def main() -> int:
    app.ollama.generate = _no_llm  # retrieval only; rewrite_query falls back to the original question
    vector_store = app.load_vector_store(app.VECTOR_DB_DIR)
    cases = [case for case in ev.load_evaluation_cases() if case["type"] in {"direct", "follow_up", "no_relevant_information"}]
    results = {"evaluation_cases": [evaluate_case(vector_store, case) for case in cases]}
    if PROBE_PATH.exists():
        probes = json.loads(PROBE_PATH.read_text(encoding="utf-8"))
        # Probes are scored on retrieval alone: many are phrased outside the
        # classifier's topic keywords and would otherwise never reach retrieval.
        results["probe_questions (classifier bypassed)"] = [evaluate_case(vector_store, case, bypass_classifier=True) for case in probes]
        results["probe_questions (end to end)"] = [evaluate_case(vector_store, case) for case in probes]

    print("\n========================================")
    print("RETRIEVAL EVALUATION")
    print("========================================")
    for name, rows in results.items():
        print(f"\n{name}:")
        for key, value in summarize(rows).items():
            print(f"  {key:38} {value}")
    if "--details" in sys.argv:
        for name, rows in results.items():
            print(f"\n---------------- {name} ----------------")
            _print_rows(rows)
    if "--json" in sys.argv:
        output = Path(sys.argv[sys.argv.index("--json") + 1])
        output.write_text(json.dumps({name: {"summary": summarize(rows), "rows": rows} for name, rows in results.items()}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
