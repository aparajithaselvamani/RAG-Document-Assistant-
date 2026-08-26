"""Run the JSON evaluation set through the application's existing RAG pipeline."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import app

DATASET_PATH = Path(__file__).with_name("evaluation_questions.json")
REQUIRED_FIELDS = {"id", "type", "question", "expected_source", "expected_answer_keywords", "expected_behavior"}
NO_INFORMATION_RESPONSE = "i could not find that information in the provided documents."
CLASSIFICATION_TYPES = {"classification", "classification_followup"}
VALID_TYPES = {"direct", "follow_up", "no_relevant_information", *CLASSIFICATION_TYPES}


def load_evaluation_cases(path: Path = DATASET_PATH) -> List[Dict[str, Any]]:
    """Load and validate the independent evaluation data set."""
    with path.open(encoding="utf-8") as dataset_file:
        cases = json.load(dataset_file)
    if not isinstance(cases, list):
        raise ValueError("Evaluation dataset must be a JSON array.")
    for case in cases:
        case_type = case.get("type")
        if case_type in CLASSIFICATION_TYPES:
            if "expected_category" not in case:
                raise ValueError(f"Classification case {case.get('id', '<unknown>')} is missing expected_category.")
            if case_type == "classification":
                if "question" not in case:
                    raise ValueError(f"Classification case {case.get('id', '<unknown>')} is missing question.")
            if case_type == "classification_followup":
                if "turns" not in case or not isinstance(case["turns"], list):
                    raise ValueError(f"Classification follow-up case {case.get('id', '<unknown>')} is missing turns.")
            continue

        missing = REQUIRED_FIELDS - set(case)
        if missing:
            raise ValueError(f"Evaluation case {case.get('id', '<unknown>')} is missing: {sorted(missing)}")
        if case_type not in {"direct", "follow_up", "no_relevant_information"}:
            raise ValueError(f"Unsupported evaluation case type: {case['type']}")
        if not isinstance(case["expected_answer_keywords"], list):
            raise ValueError(f"Evaluation case {case['id']} has invalid expected_answer_keywords.")
    return cases


def source_names(results: Iterable[Tuple[Any, float]]) -> List[str]:
    """Extract unique, filename-normalized sources in retrieval order."""
    names: List[str] = []
    for document, _ in results:
        source = normalize_source_name((document.metadata or {}).get("source"))
        if source not in names:
            names.append(source)
    return names


def normalize_source_name(source: Any) -> str:
    """Normalize a source path to its filename for portable comparisons."""
    raw_source = str(source or "Unknown source")
    # Accept Windows paths even when a test is run on a non-Windows host.
    return Path(raw_source.replace("\\", "/")).name


def _normalize_text(text: Any) -> str:
    return " ".join(str(text or "").lower().split())


def is_no_information_answer(answer: str) -> bool:
    """Identify the application's explicit grounded-answer fallback."""
    normalized = _normalize_text(answer)
    return normalized == NO_INFORMATION_RESPONSE or "could not find that information in the provided documents" in normalized


def _keyword_variants(keyword: Any) -> List[str]:
    """Return simple documented word forms for evaluator-only matching."""
    normalized = _normalize_text(keyword)
    variants = {
        "retrieve": {"retrieve", "retrieval", "retrieving", "retrieved", "retrieves"},
        "retrieval": {"retrieve", "retrieval", "retrieving", "retrieved", "retrieves"},
        "retrieving": {"retrieve", "retrieval", "retrieving", "retrieved", "retrieves"},
    }
    return sorted(variants.get(normalized, {normalized}))


def _contains_keyword(answer: str, keyword: Any) -> bool:
    normalized_answer = _normalize_text(answer)
    return any(re.search(rf"\b{re.escape(variant)}\b", normalized_answer) for variant in _keyword_variants(keyword))


def keyword_coverage(answer: str, keywords: List[Any]) -> bool:
    """Require each expected concept, allowing a group of documented alternatives."""
    for expected in keywords:
        alternatives = expected if isinstance(expected, list) else [expected]
        if not any(_contains_keyword(answer, option) for option in alternatives):
            return False
    return True


def _missing_keyword_groups(answer: str, keywords: List[Any]) -> List[str]:
    missing = []
    for expected in keywords:
        alternatives = expected if isinstance(expected, list) else [expected]
        if not any(_contains_keyword(answer, option) for option in alternatives):
            missing.append(" or ".join(str(option) for option in alternatives))
    return missing


def _run_question(vector_store: Any, question: str, history: Deque[Tuple[str, str]]) -> Dict[str, Any]:
    routed = app.route_query(vector_store=vector_store, question=question, conversation_history=history)
    rewritten = routed["rewritten_query"]
    semantic = routed["semantic_results"]
    keyword = routed["keyword_results"]
    hybrid = routed["hybrid_results"]
    answer = routed["answer"]
    app.update_conversation_history(history, question, answer)
    return {
        "rewritten_query": rewritten,
        "semantic_sources": source_names(semantic),
        "keyword_sources": source_names(keyword),
        "retrieved_sources": source_names(hybrid),
        "answer": answer,
    }


def evaluate_case(vector_store: Any, case: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate one case, first replaying its prior turns when supplied."""
    case_type = case["type"]
    if case_type in CLASSIFICATION_TYPES:
        history: Deque[Tuple[str, str]] = deque(maxlen=app.MAX_CONVERSATION_TURNS)
        if case_type == "classification_followup":
            turns = case.get("turns", [])
            for prior_question in turns[:-1]:
                history.append((prior_question, ""))
            question = turns[-1] if turns else case.get("question", "")
        else:
            question = case.get("question", "")
        predicted_category = app.classify_query(question, history)["category"]
        routing = {
            "DOCUMENT_RELATED": "Document retrieval",
            "FOLLOW_UP": "Conversation history + document retrieval",
            "GENERAL_OR_UNRELATED": "Fallback without retrieval",
        }.get(predicted_category, "Fallback without retrieval")
        passed = predicted_category == case["expected_category"]
        return {
            **case,
            "question": question,
            "predicted_category": predicted_category,
            "routing": routing,
            "passed": passed,
            "failure_reasons": [] if passed else [f"Expected {case['expected_category']}, got {predicted_category}"],
            "history_turns": len(history),
        }

    history: Deque[Tuple[str, str]] = deque(maxlen=app.MAX_CONVERSATION_TURNS)
    for prior_question in case.get("history", []):
        _run_question(vector_store, prior_question, history)

    result = _run_question(vector_store, case["question"], history)
    expected_source = normalize_source_name(case["expected_source"]) if case["expected_source"] else None
    failure_reasons: List[str] = []
    if case_type == "no_relevant_information":
        source_passed = not result["retrieved_sources"]
        answer_passed = is_no_information_answer(result["answer"])
        if not source_passed:
            failure_reasons.append("Unsupported question returned retrieved sources: " + ", ".join(result["retrieved_sources"]))
        if not answer_passed:
            failure_reasons.append("Answer did not use the grounded no-information fallback.")
    else:
        source_passed = expected_source in result["retrieved_sources"]
        answer_passed = keyword_coverage(result["answer"], case["expected_answer_keywords"])
        if not source_passed:
            failure_reasons.append(f"Expected source '{expected_source}' was not retrieved.")
        missing_keywords = _missing_keyword_groups(result["answer"], case["expected_answer_keywords"])
        if missing_keywords:
            failure_reasons.append("Answer is missing expected concept(s): " + "; ".join(missing_keywords))
    passed = source_passed and answer_passed
    checks = {"source": source_passed, "answer": answer_passed}
    return {**case, **result, "expected_source": expected_source, "checks": checks, "passed": passed, "failure_reasons": failure_reasons, "history_turns": len(history)}


def build_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate total and per-query-type pass/fail counts."""
    totals = Counter(result["type"] for result in results)
    passed = Counter(result["type"] for result in results if result["passed"])
    ordered_types = ("direct", "follow_up", "no_relevant_information", "classification", "classification_followup")
    return {
        "total": len(results),
        "passed": sum(result["passed"] for result in results),
        "failed": sum(not result["passed"] for result in results),
        "by_type": {kind: {"passed": passed[kind], "total": totals[kind]} for kind in ordered_types},
    }


def print_summary(results: List[Dict[str, Any]]) -> None:
    summary = build_summary(results)
    pass_rate = 0.0 if not summary["total"] else 100 * summary["passed"] / summary["total"]
    print("\n========================================")
    print("RAG EVALUATION RESULTS")
    print("========================================")
    print(f"Total tests: {summary['total']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Pass rate: {pass_rate:.1f}%")
    print("\n----------------------------------------")
    labels = {
        "direct": "Direct Questions",
        "follow_up": "Follow-up Questions",
        "no_relevant_information": "No-information Questions",
        "classification": "Query Classification",
        "classification_followup": "Follow-up Classification",
    }
    for kind, label in labels.items():
        counts = summary["by_type"].get(kind, {"passed": 0, "total": 0})
        print(f"{label}: {counts['passed']}/{counts['total']}")
    print("----------------------------------------\n\nDetailed Results:")
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n[{status}] {result['id']}")
        if result["type"] in CLASSIFICATION_TYPES:
            question = result.get("question") or result.get("turns", [""])[-1]
            print(f"Question: {question}")
            print(f"Expected category: {result['expected_category']}")
            print(f"Predicted category: {result.get('predicted_category', 'UNKNOWN')}")
            print(f"Routing: {result.get('routing', 'UNKNOWN')}")
            if result["failure_reasons"]:
                print("Failure reason: " + " ".join(result["failure_reasons"]))
            continue
        print(f"Question: {result['question']}")
        print(f"Expected source: {result['expected_source'] or 'None'}")
        print(f"Retrieved sources: {', '.join(result['retrieved_sources']) or 'None'}")
        print(f"Source match: {'PASS' if result['checks']['source'] else 'FAIL'}")
        print(f"Answer keyword match: {'PASS' if result['checks']['answer'] else 'FAIL'}")
        if result["failure_reasons"]:
            print("Failure reason: " + " ".join(result["failure_reasons"]))
        print(f"Answer: {result['answer']}")
    print("\n========================================")


def main() -> int:
    try:
        cases = load_evaluation_cases()
        vector_store = app.load_vector_store(app.VECTOR_DB_DIR)
        results = [evaluate_case(vector_store, case) for case in cases]
    except Exception as exc:
        print(f"Evaluation could not run: {exc}")
        return 1
    print_summary(results)
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
