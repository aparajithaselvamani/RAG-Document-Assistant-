"""Compare retrieval with and without the cross-encoder reranking step (no Ollama needed).

Runs the existing evaluation cases (direct, follow-up, no-information) and the
probe questions twice: hybrid search only ("before") and hybrid search +
cross-encoder reranking ("after").  Chunk-level relevance labels come from
rerank_relevance_labels.json (grade 2 = answers the question, 1 = supporting).

Ranking metrics are computed on the ranked chunk list *before* context
expansion, which is the list the reranker changes:

  answer@1        the top chunk answers the question (grade 2)
  MRR             1 / rank of the first answering chunk
  nDCG@k          graded ranking quality (1.0 = ideal order)
  precision@k     share of returned chunks with grade >= 1
  irrelevant      returned chunks with grade 0
  answer_in_ctx   expected answer keywords present in the final, expanded context
  no-info         unsupported questions for which retrieval returns nothing
  latency         mean retrieval time per question (model already loaded)

Usage:  python evaluation/evaluate_reranking.py [--details] [--json results.json]
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import app  # noqa: E402
from evaluation import evaluate_rag as ev  # noqa: E402
from reranker import get_reranker  # noqa: E402
from utils import DEFAULT_TOP_K  # noqa: E402

LABELS_PATH = Path(__file__).with_name("rerank_relevance_labels.json")
PROBE_PATH = Path(__file__).with_name("retrieval_probe_questions.json")
MODES = {"before (hybrid only)": False, "after (hybrid + rerank)": True}


def _no_llm(*_args: Any, **_kwargs: Any) -> Dict[str, str]:
    raise RuntimeError("reranking evaluation does not call the LLM")


def _grade(document: Any, labels: List[Dict[str, Any]]) -> int:
    source = ev.normalize_source_name((document.metadata or {}).get("source"))
    text = document.page_content.strip()
    for label in labels:
        if label["source"] == source and text.startswith(label["starts_with"]):
            return int(label["grade"])
    return 0


def _ndcg(grades: List[int], all_grades: List[int], k: int) -> float:
    dcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades[:k]))
    ideal = sorted(all_grades, reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def evaluate_case(store: Any, case: Dict[str, Any], labels: Dict[str, List[Dict[str, Any]]], rerank: bool) -> Dict[str, Any]:
    history: deque = deque(maxlen=app.MAX_CONVERSATION_TURNS)
    for prior_question in case.get("history", []):
        app.update_conversation_history(history, prior_question, "")

    started = time.perf_counter()
    query, _, _, ranked, _ = app.rank_chunks(store, case["question"], conversation_history=history, rerank=rerank)
    elapsed_ms = (time.perf_counter() - started) * 1000
    final = app.retrieve_context(store, case["question"], conversation_history=history, rerank=rerank)[3]

    row: Dict[str, Any] = {
        "id": case["id"],
        "question": case["question"],
        "query": query,
        "latency_ms": elapsed_ms,
        "ranked": [(ev.normalize_source_name(d.metadata.get("source")), d.page_content) for d, _ in ranked],
    }
    case_labels = labels.get(case["id"])
    if case_labels is None:  # no-information question
        row.update(answerable=False, rejected=not ranked)
        return row

    grades = [_grade(document, case_labels) for document, _ in ranked]
    answer_ranks = [i + 1 for i, g in enumerate(grades) if g == 2]
    context = "\n".join(document.page_content for document, _ in final)
    row.update(
        answerable=True,
        grades=grades,
        answer_at_1=bool(grades) and grades[0] == 2,
        reciprocal_rank=1 / answer_ranks[0] if answer_ranks else 0.0,
        ndcg=_ndcg(grades, [int(label["grade"]) for label in case_labels], DEFAULT_TOP_K),
        precision=sum(g >= 1 for g in grades) / len(grades) if grades else 0.0,
        irrelevant=sum(g == 0 for g in grades),
        answer_in_context=ev.keyword_coverage(context, case.get("expected_answer_keywords", [])),
        context_chars=len(context),
    )
    return row


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    answerable = [r for r in rows if r["answerable"]]
    unsupported = [r for r in rows if not r["answerable"]]

    def mean(values: List[float]) -> float:
        return round(sum(values) / len(values), 3) if values else 0.0

    summary = {
        "answerable": len(answerable),
        "answer@1": mean([r["answer_at_1"] for r in answerable]),
        "MRR": mean([r["reciprocal_rank"] for r in answerable]),
        f"nDCG@{DEFAULT_TOP_K}": mean([r["ndcg"] for r in answerable]),
        "precision": mean([r["precision"] for r in answerable]),
        "irrelevant_chunks": sum(r["irrelevant"] for r in answerable),
        "answer_in_ctx": mean([r["answer_in_context"] for r in answerable]),
        "context_chars": round(mean([r["context_chars"] for r in answerable])),
        "latency_ms": round(mean([r["latency_ms"] for r in rows]), 1),
    }
    if unsupported:
        summary["no_info_rejected"] = f"{sum(r['rejected'] for r in unsupported)}/{len(unsupported)}"
    return summary


def main() -> int:
    app.ollama.generate = _no_llm
    if get_reranker(force=True) is None:
        print("The reranker model could not be loaded; see the message above.")
        return 1
    store = app.load_vector_store(app.VECTOR_DB_DIR)
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))["labels"]
    sets = {
        "evaluation cases": [c for c in ev.load_evaluation_cases() if c["type"] in {"direct", "follow_up", "no_relevant_information"}],
        "probe questions": json.loads(PROBE_PATH.read_text(encoding="utf-8")),
    }
    # Warm-up so model loading is not counted as latency.
    app.rank_chunks(store, "warm up", rerank=True)

    results: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for set_name, cases in sets.items():
        results[set_name] = {mode: [evaluate_case(store, case, labels, flag) for case in cases] for mode, flag in MODES.items()}

    print("\n========================================")
    print("RERANKING EVALUATION (retrieval only)")
    print("========================================")
    for set_name, by_mode in results.items():
        summaries = {mode: summarize(rows) for mode, rows in by_mode.items()}
        keys = list(next(iter(summaries.values())))
        print(f"\n{set_name}:")
        print(f"  {'metric':20}" + "".join(f"{mode:>26}" for mode in summaries))
        for key in keys:
            print(f"  {key:20}" + "".join(f"{str(summaries[mode][key]):>26}" for mode in summaries))

    if "--details" in sys.argv:
        for set_name, by_mode in results.items():
            print(f"\n---------------- {set_name}: ranked chunks (grade) ----------------")
            before_rows, after_rows = by_mode.values()
            for before, after in zip(before_rows, after_rows):
                changed = before["ranked"] != after["ranked"]
                print(f"\n{'*' if changed else ' '} {before['id']}: {before['question']}")
                for label, row in (("before", before), ("after ", after)):
                    grades = row.get("grades", [None] * len(row["ranked"]))
                    items = [f"[{g if g is not None else '-'}] {ev._normalize_text(text)[:48]}" for (_, text), g in zip(row["ranked"], grades)]
                    print(f"    {label}: " + (" | ".join(items) if items else "(nothing)"))
    if "--json" in sys.argv:
        output = Path(sys.argv[sys.argv.index("--json") + 1])
        output.write_text(json.dumps({s: {m: {"summary": summarize(r), "rows": r} for m, r in bm.items()} for s, bm in results.items()}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
