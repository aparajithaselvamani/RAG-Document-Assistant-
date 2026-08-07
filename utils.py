from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from pypdf import PdfWriter

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
VECTOR_DB_DIR = BASE_DIR / "vector_db"
VECTOR_DB_PATH = VECTOR_DB_DIR / "chroma"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL = "qwen2.5:3b"
DEFAULT_TOP_K = 4


def create_sample_documents(documents_dir: Path) -> List[Path]:
    """Create a small set of demonstration documents if the folder is missing or empty."""
    documents_dir.mkdir(parents=True, exist_ok=True)

    if any(documents_dir.iterdir()):
        return sorted([path for path in documents_dir.iterdir() if path.is_file()])

    sample_files: List[Path] = []

    txt_documents = {
        "rag_basics.txt": """Retrieval-Augmented Generation, or RAG, helps language models answer questions using external documents.\nThe system retrieves relevant chunks from a document store and passes that context to the model, which then answers using only the retrieved evidence.\nRAG is useful when the model needs up-to-date or domain-specific knowledge that is not stored in its weights.\n""",
        "embeddings_and_chroma.txt": """Embeddings convert text into vectors that capture semantic meaning.\nThese vectors allow systems to find related content even when the wording differs.\nChromaDB is a vector database that stores embeddings and supports semantic search over document chunks.\n""",
        "search_methods.txt": """Semantic search uses vector similarity to find conceptually related passages.\nKeyword search uses lexical overlap and term statistics such as TF-IDF or BM25 to find relevant words.\nHybrid search combines both approaches by merging and re-ranking their results.\n""",
    }

    for file_name, content in txt_documents.items():
        file_path = documents_dir / file_name
        file_path.write_text(content, encoding="utf-8")
        sample_files.append(file_path)

    pdf_path = documents_dir / "query_rewriting_and_ollama.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n4 0 obj\n<< /Length 44 >>\nstream\nBT /F1 12 Tf 50 70 Td (Query rewriting improves retrieval.) Tj ET\nendstream\nendobj\n5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\nxref\n0 6\n0000000000 65535 f \n0000000010 00000 n \n0000000062 00000 n \n0000000119 00000 n \n0000000203 00000 n \n0000000306 00000 n \ntrailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n0\n%%EOF\n")
    sample_files.append(pdf_path)
    return sample_files


def build_chunk_metadata(chunk_index: int, source: str, page_number: int | None = None) -> dict:
    """Create metadata for each chunk with source, chunk indexing, and optional page information."""
    metadata = {"source": source, "chunk_id": chunk_index}
    if page_number is not None:
        metadata["page_number"] = page_number
    return metadata


def format_snippet(text: str, max_chars: int = 180) -> str:
    """Create a compact display snippet for attribution output."""
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def ensure_documents_exists(documents_dir: Optional[Path] = None) -> Path:
    """Ensure the documents folder exists and contains sample documents when needed."""
    target_dir = documents_dir or DOCUMENTS_DIR
    create_sample_documents(target_dir)
    return target_dir


def chunk_to_document(chunk: Document) -> Document:
    """Return a copy of a chunk with normalized metadata."""
    metadata = dict(chunk.metadata or {})
    source = metadata.get("source") or "unknown"
    metadata["source"] = source
    return Document(page_content=chunk.page_content, metadata=metadata)
