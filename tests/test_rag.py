import sys
from collections import deque
from pathlib import Path

from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app
from app import format_conversation_history, rewrite_query, update_conversation_history
from hybrid_search import hybrid_search, normalize_semantic_results
from ingest import copy_uploaded_document, index_uploaded_document, validate_uploaded_file
from keyword_search import keyword_search
from utils import create_sample_documents


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
    assert "Conversation History:" in captured["prompt"]
    assert "User: What is RAG?" in captured["prompt"]
    assert "Question:\nHow does it work?" in captured["prompt"]


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
    assert "Answer the user's current question" in captured["prompt"]
    assert "Retrieved Context:" in captured["prompt"]
    assert "Question:" in captured["prompt"]


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
