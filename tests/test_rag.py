import sys
from collections import deque
from pathlib import Path

from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app
from app import (
    NO_INFORMATION_RESPONSE,
    format_conversation_history,
    has_sufficient_grounding,
    resolve_follow_up_question,
    rewrite_query,
    update_conversation_history,
)
from hybrid_search import hybrid_search, normalize_semantic_results
from ingest import copy_uploaded_document, index_uploaded_document, validate_uploaded_file
from keyword_search import keyword_search
from utils import create_sample_documents
from evaluation import evaluate_rag


def test_keyword_search_prefers_relevant_chunks() -> None:
    chunks = [
        Document(page_content="Retrieval augmented generation uses embeddings and vector search to answer questions.", metadata={"source": "rag.txt"}),
        Document(page_content="Hybrid search combines semantic and keyword signals for better ranking.", metadata={"source": "hybrid.txt"}),
    ]

    results = keyword_search("What is retrieval augmented generation?", chunks, top_k=1)

    assert results
    assert results[0][0].metadata["source"] == "rag.txt"


def test_create_sample_documents_writes_documents(tmp_path: Path) -> None:
    docs_dir = tmp_path / "documents"
    created_files = create_sample_documents(docs_dir)

    assert created_files
    assert any(path.suffix == ".txt" for path in created_files)


def test_validate_uploaded_file_accepts_supported_types(tmp_path: Path) -> None:
    text_file = tmp_path / "sample.txt"
    text_file.write_text("hello", encoding="utf-8")

    validated_path = validate_uploaded_file(text_file)

    assert validated_path == text_file


def test_copy_uploaded_document_moves_file_into_documents_folder(tmp_path: Path) -> None:
    source_file = tmp_path / "upload.txt"
    source_file.write_text("hello", encoding="utf-8")
    documents_dir = tmp_path / "documents"

    copied_file = copy_uploaded_document(source_file, documents_dir)

    assert copied_file.exists()
    assert copied_file.parent == documents_dir
    assert copied_file.name == source_file.name


def test_index_uploaded_document_works_with_vector_stores_without_persist(tmp_path: Path) -> None:
    source_file = tmp_path / "upload.txt"
    source_file.write_text("The leave policy allows paid vacation days.", encoding="utf-8")
    documents_dir = tmp_path / "documents"

    class StubVectorStore:
        def __init__(self) -> None:
            self.added_documents = []

        def add_documents(self, documents: list[Document]) -> None:
            self.added_documents.extend(documents)

        def get(self, include: list[str] | None = None) -> dict:
            return {"documents": [], "metadatas": []}

    vector_store = StubVectorStore()

    chunk_count, indexed_count = index_uploaded_document(source_file, vector_store, documents_dir)

    assert chunk_count == 1
    assert indexed_count == 1
    assert len(vector_store.added_documents) == 1


def test_rewrite_query_expands_known_abbreviation() -> None:
    assert rewrite_query("What is RAG?") == "What is Retrieval-Augmented Generation (RAG)?"
    assert rewrite_query("How work?") == "How does it work?"
    assert rewrite_query("vector db?") == "What is a vector database?"


def test_hybrid_search_deduplicates_and_ranks_results() -> None:
    chunks = [
        Document(page_content="Retrieval augmented generation uses embeddings and vector search to answer questions.", metadata={"source": "rag.txt", "chunk_id": 0}),
        Document(page_content="Hybrid search combines semantic and keyword signals for better ranking.", metadata={"source": "hybrid.txt", "chunk_id": 0}),
        Document(page_content="Vector databases store embeddings for semantic search.", metadata={"source": "vector.txt", "chunk_id": 0}),
    ]

    semantic_results = [
        (chunks[0], 0.20),
        (chunks[1], 0.40),
        (chunks[0], 0.25),
    ]

    hybrid_results = hybrid_search("What is RAG?", chunks, semantic_results, top_k=2)

    assert hybrid_results
    assert len(hybrid_results) == 1
    assert hybrid_results[0][0].metadata["source"] == "rag.txt"


def test_hybrid_search_prefers_rag_document_for_rag_queries() -> None:
    chunks = [
        Document(page_content="Retrieval-Augmented Generation helps language models answer questions using external documents.", metadata={"source": "rag_basics.txt", "chunk_id": 0}),
        Document(page_content="Query rewriting improves retrieval quality for the assistant.", metadata={"source": "query_rewriting_and_ollama.pdf", "chunk_id": 0}),
    ]

    semantic_results = [
        (chunks[0], 0.20),
        (chunks[1], 0.40),
    ]

    hybrid_results = hybrid_search("What is RAG?", chunks, semantic_results, top_k=1)

    assert hybrid_results[0][0].metadata["source"] == "rag_basics.txt"


def test_rewrite_query_leaves_clear_queries_unchanged() -> None:
    assert rewrite_query("How do embeddings improve semantic search?") == "How do embeddings improve semantic search?"
    assert rewrite_query("Who is Elon Musk?") == "Who is Elon Musk?"


def test_keyword_search_does_not_return_duplicate_chunks() -> None:
    chunks = [
        Document(page_content="Embeddings represent text as vectors for semantic search.", metadata={"source": "embeddings.txt", "chunk_id": 0}),
        Document(page_content="Embeddings represent text as vectors for semantic search.", metadata={"source": "embeddings.txt", "chunk_id": 1}),
    ]

    results = keyword_search("embeddings", chunks, top_k=5)

    assert len(results) == 1
    assert results[0][0].metadata["chunk_id"] == 0


def test_update_conversation_history_keeps_only_latest_five_turns() -> None:
    history = deque(maxlen=5)

    for index in range(6):
        update_conversation_history(history, f"Question {index}", f"Answer {index}")

    assert len(history) == 5
    assert history[0][0] == "Question 1"
    assert history[-1][0] == "Question 5"


def test_format_conversation_history_renders_turns() -> None:
    history = deque(maxlen=5)
    history.append(("What is RAG?", "RAG is retrieval augmented generation."))
    history.append(("How does it work?", "It uses retrieved context to ground responses."))

    formatted = format_conversation_history(history)

    assert "Conversation History:" in formatted
    assert "User: What is RAG?" in formatted
    assert "Assistant: It uses retrieved context to ground responses." in formatted


def test_generate_answer_includes_conversation_history_in_prompt(monkeypatch: object) -> None:
    history = deque(maxlen=5)
    history.append(("What is RAG?", "RAG is retrieval augmented generation."))

    captured: dict[str, str] = {}

    def fake_generate(*args: object, **kwargs: object) -> dict[str, str]:
        captured["prompt"] = kwargs["prompt"]
        return {"response": "A follow-up answer."}

    monkeypatch.setattr(app.ollama, "generate", fake_generate)

    context = [Document(page_content="RAG combines retrieval with generation.", metadata={"source": "rag.txt"})]
    answer = app.generate_answer("How does it work?", context, conversation_history=history)

    assert answer == "A follow-up answer."
    assert "Conversation History (reference resolution only; not factual evidence):" in captured["prompt"]
    assert "User: What is RAG?" in captured["prompt"]
    assert "Question to Answer:\nHow does RAG work?" in captured["prompt"]


def test_hybrid_search_filters_irrelevant_documents_for_semantic_search_queries() -> None:
    relevant_chunk = Document(
        page_content="Semantic search uses vector similarity to find conceptually related passages.",
        metadata={"source": "search_methods.txt", "chunk_id": 0},
    )
    irrelevant_chunk = Document(
        page_content="The leave policy allows paid vacation days and requires advance notice.",
        metadata={"source": "upload_test.txt", "chunk_id": 0},
    )
    chunks = [relevant_chunk, irrelevant_chunk]
    # Raw Chroma distances: lower means a closer semantic match.
    semantic_results = [(relevant_chunk, 0.21), (irrelevant_chunk, 1.25)]

    hybrid_results = hybrid_search("How is it different from keyword search?", chunks, semantic_results, top_k=4)

    assert hybrid_results
    assert hybrid_results[0][0].metadata["source"] == "search_methods.txt"
    assert all(result[0].metadata["source"] != "upload_test.txt" for result in hybrid_results)


def test_generate_answer_prompt_anchors_on_the_current_question() -> None:
    context = [Document(page_content="Semantic search uses vector similarity while keyword search uses lexical overlap.", metadata={"source": "search_methods.txt"})]

    captured: dict[str, str] = {}

    def fake_generate(*args: object, **kwargs: object) -> dict[str, str]:
        captured["prompt"] = kwargs["prompt"]
        return {"response": "Semantic search ranks by meaning."}

    app.ollama.generate = fake_generate

    answer = app.generate_answer("How is semantic search different from keyword search?", context)

    assert answer == "Semantic search ranks by meaning."
    assert "Answer the Question to Answer" in captured["prompt"]
    assert "Retrieved Context:" in captured["prompt"]
    assert "Question to Answer:" in captured["prompt"]


def test_resolve_follow_up_question_identifies_compared_search_methods() -> None:
    history = deque(maxlen=5)
    history.append(("What is semantic search?", "Semantic search uses vector similarity."))
    history.append(("How is it different from keyword search?", "Keyword search uses lexical overlap."))

    resolved = resolve_follow_up_question("Which one uses embeddings?", history)

    assert resolved == "Which of semantic search and keyword search uses embeddings?"


def test_resolve_follow_up_question_preserves_other_reference_cases() -> None:
    leave_history = deque([("What is the leave policy?", "It allows paid vacation days.")], maxlen=5)
    rag_history = deque([("What is RAG?", "RAG uses external documents.")], maxlen=5)

    assert resolve_follow_up_question("How far in advance should employees request it?", leave_history) == "How far in advance should employees request the leave policy?"
    assert resolve_follow_up_question("How does it use external documents?", rag_history) == "How does RAG use external documents?"


def test_generate_answer_uses_resolved_question_only_for_reference_resolution(monkeypatch: object) -> None:
    history = deque([
        ("What is semantic search?", "Semantic search uses vector similarity."),
        ("How is it different from keyword search?", "Keyword search uses lexical overlap."),
    ], maxlen=5)
    context = [
        Document(page_content="Embeddings convert text into vectors. ChromaDB stores embeddings.", metadata={"source": "embeddings_and_chroma.txt"}),
        Document(page_content="Semantic search uses vector similarity. Keyword search uses lexical overlap.", metadata={"source": "search_methods.txt"}),
    ]
    captured: dict[str, str] = {}

    def fake_generate(*_args: object, **kwargs: object) -> dict[str, str]:
        captured["prompt"] = kwargs["prompt"]
        return {"response": "Semantic search uses embeddings."}

    monkeypatch.setattr(app.ollama, "generate", fake_generate)
    answer = app.generate_answer("Which one uses embeddings?", context, conversation_history=history)

    assert answer == "Semantic search uses embeddings."
    assert "Question to Answer:" in captured["prompt"]
    assert "Which of semantic search and keyword search uses embeddings?" in captured["prompt"]
    assert "not an unrelated term in Retrieved Context" in captured["prompt"]


def test_grounding_check_rejects_empty_and_irrelevant_context() -> None:
    rag_context = [Document(page_content="RAG uses retrieval from external documents.", metadata={"source": "rag.txt"})]

    assert not has_sufficient_grounding("What is the capital of France?", [])
    assert not has_sufficient_grounding("What is the population of Japan?", rag_context)


def test_grounding_check_rejects_keyword_neighbour_when_named_fact_is_missing() -> None:
    rag_context = [Document(page_content="RAG retrieves relevant chunks from a document store.", metadata={"source": "rag.txt"})]

    assert not has_sufficient_grounding("What does RAG say about France?", rag_context)


def test_generate_answer_rejects_unsupported_context_without_calling_ollama(monkeypatch: object) -> None:
    history = deque([("What is semantic search?", "Semantic search uses vector similarity.")], maxlen=5)
    context = [Document(page_content="Semantic search uses vector similarity.", metadata={"source": "search_methods.txt"})]

    def fail_if_called(*_args: object, **_kwargs: object) -> dict[str, str]:
        raise AssertionError("Ollama must not be called for insufficient context")

    monkeypatch.setattr(app.ollama, "generate", fail_if_called)

    assert app.generate_answer("What is the population of Japan?", context, conversation_history=history) == NO_INFORMATION_RESPONSE


def test_normalize_semantic_results_converts_lower_distance_to_higher_similarity() -> None:
    close = Document(page_content="close", metadata={"source": "close.txt"})
    far = Document(page_content="far", metadata={"source": "far.txt"})

    results = normalize_semantic_results([(far, 1.5), (close, 0.2)])

    assert results[0][0] == close
    assert 0 < results[0][1] <= 1


def test_hybrid_search_removes_duplicate_chunks() -> None:
    chunk = Document(page_content="Semantic search uses vector similarity.", metadata={"source": "methods.txt", "chunk_id": 0})
    results = hybrid_search("semantic search", [chunk], [(chunk, 0.2), (chunk, 0.4)], top_k=4)

    assert len(results) == 1


def test_display_sources_only_lists_the_chunks_given_to_answer_generation(capsys: object) -> None:
    relevant = Document(page_content="Semantic search uses vector similarity.", metadata={"source": "search_methods.txt"})

    app.display_sources([relevant])

    output = capsys.readouterr().out
    assert "search_methods.txt" in output
    assert "upload_test.txt" not in output


def test_evaluation_dataset_loads_with_required_fields() -> None:
    cases = evaluate_rag.load_evaluation_cases()

    assert len(cases) == 17
    assert {case["type"] for case in cases} == {"direct", "follow_up", "no_relevant_information"}
    assert all(evaluate_rag.REQUIRED_FIELDS <= set(case) for case in cases)


def test_evaluation_direct_case_checks_source_and_answer(monkeypatch: object) -> None:
    document = Document(page_content="RAG uses retrieval and external documents.", metadata={"source": "rag_basics.txt"})
    monkeypatch.setattr(app, "retrieve_context", lambda *_, **_kwargs: ("What is RAG?", [], [], [(document, 0.9)]))
    monkeypatch.setattr(app, "generate_answer", lambda *_args, **_kwargs: "RAG uses retrieval and external documents.")
    case = {"id": "direct", "type": "direct", "question": "What is RAG?", "expected_source": "rag_basics.txt", "expected_answer_keywords": ["retrieval", "external documents"], "expected_behavior": "answer_from_documents"}

    result = evaluate_rag.evaluate_case(object(), case)

    assert result["passed"]
    assert result["checks"] == {"source": True, "answer": True}


def test_evaluation_follow_up_replays_history(monkeypatch: object) -> None:
    document = Document(page_content="Semantic search uses vectors; keyword search uses lexical overlap.", metadata={"source": "search_methods.txt"})
    asked: list[tuple[str, int]] = []

    def fake_retrieve(_store: object, question: str, **_kwargs: object) -> tuple[str, list[object], list[object], list[tuple[Document, float]]]:
        asked.append((question, 0))
        return question, [], [], [(document, 0.9)]

    def fake_answer(question: str, _context: list[Document], conversation_history: object = None) -> str:
        asked[-1] = (question, len(conversation_history))
        return "Semantic search uses vectors while keyword search uses lexical overlap."

    monkeypatch.setattr(app, "retrieve_context", fake_retrieve)
    monkeypatch.setattr(app, "generate_answer", fake_answer)
    case = {"id": "follow", "type": "follow_up", "question": "How is it different?", "history": ["What is semantic search?"], "expected_source": "search_methods.txt", "expected_answer_keywords": ["semantic", "lexical"], "expected_behavior": "answer_from_documents"}

    result = evaluate_rag.evaluate_case(object(), case)

    assert result["passed"]
    assert asked == [("What is semantic search?", 0), ("How is it different?", 1)]


def test_evaluation_no_information_requires_empty_sources_and_fallback(monkeypatch: object) -> None:
    monkeypatch.setattr(app, "retrieve_context", lambda *_, **_kwargs: ("capital of France", [], [], []))
    monkeypatch.setattr(app, "generate_answer", lambda *_args, **_kwargs: "I could not find that information in the provided documents.")
    case = {"id": "none", "type": "no_relevant_information", "question": "What is the capital of France?", "expected_source": None, "expected_answer_keywords": [], "expected_behavior": "should_not_answer_from_documents"}

    result = evaluate_rag.evaluate_case(object(), case)

    assert result["passed"]


def test_evaluation_summary_counts_passes_and_failures() -> None:
    summary = evaluate_rag.build_summary([
        {"type": "direct", "passed": True},
        {"type": "direct", "passed": False},
        {"type": "follow_up", "passed": True},
        {"type": "no_relevant_information", "passed": True},
    ])

    assert summary["total"] == 4
    assert summary["passed"] == 3
    assert summary["by_type"]["direct"] == {"passed": 1, "total": 2}


def test_evaluation_normalizes_full_path_sources_and_matches_any_rank(monkeypatch: object) -> None:
    first = Document(page_content="Embeddings use vectors.", metadata={"source": "C:\\RAG_ASSISTANT\\documents\\embeddings_and_chroma.txt"})
    expected = Document(page_content="Semantic search uses vector similarity.", metadata={"source": "C:\\RAG_ASSISTANT\\documents\\search_methods.txt"})
    monkeypatch.setattr(app, "retrieve_context", lambda *_, **_kwargs: ("semantic", [], [], [(first, 0.9), (expected, 0.8)]))
    monkeypatch.setattr(app, "generate_answer", lambda *_args, **_kwargs: "Semantic search uses vector similarity.")
    case = {"id": "path", "type": "direct", "question": "What is semantic search?", "expected_source": "search_methods.txt", "expected_answer_keywords": ["vector similarity"], "expected_behavior": "answer_from_documents"}

    result = evaluate_rag.evaluate_case(object(), case)

    assert result["passed"]
    assert result["retrieved_sources"] == ["embeddings_and_chroma.txt", "search_methods.txt"]


def test_evaluation_keyword_matching_is_case_and_whitespace_insensitive() -> None:
    answer = "Embeddings are VECTORS that capture the\n semantic   meaning of text."

    assert evaluate_rag.keyword_coverage(answer, [["vectors", "vector"], "semantic meaning"])


def test_evaluation_accepts_document_store_as_direct_rag_evidence() -> None:
    answer = "RAG retrieves relevant chunks from a document store before answering."

    assert evaluate_rag.keyword_coverage(answer, [["external documents", "document store", "documents"]])


def test_evaluation_follow_up_matches_normalized_source(monkeypatch: object) -> None:
    document = Document(page_content="Semantic search uses embeddings.", metadata={"source": "C:\\docs\\search_methods.txt"})
    monkeypatch.setattr(app, "retrieve_context", lambda *_, **_kwargs: ("semantic", [], [], [(document, 0.9)]))
    monkeypatch.setattr(app, "generate_answer", lambda *_args, **_kwargs: "Semantic search uses embeddings.")
    case = {"id": "follow-path", "type": "follow_up", "question": "Which one uses embeddings?", "history": ["What is semantic search?"], "expected_source": "search_methods.txt", "expected_answer_keywords": ["semantic", "embeddings"], "expected_behavior": "answer_from_documents"}

    assert evaluate_rag.evaluate_case(object(), case)["passed"]
