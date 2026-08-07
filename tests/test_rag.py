from pathlib import Path

from langchain_core.documents import Document

from app import rewrite_query
from hybrid_search import hybrid_search
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
        (chunks[0], 0.40),
        (chunks[1], 0.20),
        (chunks[0], 0.35),
    ]

    hybrid_results = hybrid_search("What is RAG?", chunks, semantic_results, top_k=2)

    assert hybrid_results
    assert len(hybrid_results) == 2
    assert hybrid_results[0][0].metadata["source"] == "rag.txt"


def test_hybrid_search_prefers_rag_document_for_rag_queries() -> None:
    chunks = [
        Document(page_content="Retrieval-Augmented Generation helps language models answer questions using external documents.", metadata={"source": "rag_basics.txt", "chunk_id": 0}),
        Document(page_content="Query rewriting improves retrieval quality for the assistant.", metadata={"source": "query_rewriting_and_ollama.pdf", "chunk_id": 0}),
    ]

    semantic_results = [
        (chunks[0], 0.40),
        (chunks[1], 0.20),
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
