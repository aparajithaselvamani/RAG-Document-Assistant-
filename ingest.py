from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils import DOCUMENTS_DIR, EMBEDDING_MODEL, VECTOR_DB_PATH, build_chunk_metadata, ensure_documents_exists

SUPPORTED_UPLOAD_SUFFIXES = {".pdf", ".txt"}


def _load_document_from_path(file_path: Path) -> List[Document]:
    """Load a single supported document and normalize its metadata."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif suffix == ".txt":
        loader = TextLoader(str(file_path), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix or 'unknown'}")

    loaded_docs = loader.load()
    documents: List[Document] = []
    for document in loaded_docs:
        metadata = dict(document.metadata or {})
        metadata["source"] = file_path.name
        metadata["file_path"] = str(file_path)
        metadata["page_number"] = metadata.get("page")
        document.metadata = metadata
        documents.append(document)
    return documents


def load_documents(documents_dir: Path) -> List[Document]:
    """Load all supported documents from the documents folder."""
    if not documents_dir.exists():
        raise FileNotFoundError(f"Documents folder not found: {documents_dir}")
    if not documents_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory, but found: {documents_dir}")

    files = sorted([path for path in documents_dir.rglob("*") if path.is_file()])
    if not files:
        raise ValueError(f"No documents found in {documents_dir}. Add PDF or TXT files first.")

    documents: List[Document] = []
    for file_path in files:
        try:
            documents.extend(_load_document_from_path(file_path))
        except Exception as exc:
            print(f"Skipping {file_path.name}: {exc}")

    if not documents:
        raise ValueError("No documents could be loaded. Check the files and try again.")

    return documents


def split_documents(documents: List[Document], start_index: int = 0) -> List[Document]:
    """Split documents into smaller chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata or {})
        page_number = metadata.get("page_number") or metadata.get("page")
        chunk.metadata.update(
            build_chunk_metadata(start_index + index, metadata.get("source", "unknown"), page_number)
        )
    return chunks


def build_vector_database(chunks: List[Document], persist_dir: Path) -> None:
    """Create and persist a Chroma vector database from the document chunks."""
    if persist_dir.exists():
        if persist_dir.is_dir():
            shutil.rmtree(persist_dir, ignore_errors=True)
        else:
            persist_dir.unlink(missing_ok=True)
    persist_dir.mkdir(parents=True, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )


def validate_uploaded_file(file_path: Path | str) -> Path:
    """Validate that a file exists and is a supported upload type."""
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a file, but found: {path}")
    if path.suffix.lower() not in SUPPORTED_UPLOAD_SUFFIXES:
        raise ValueError(f"Unsupported file type: {path.suffix or 'unknown'}")
    return path


def copy_uploaded_document(source_file: Path | str, documents_dir: Path) -> Path:
    """Copy an uploaded file into the documents folder for indexing."""
    source_path = validate_uploaded_file(source_file)
    destination_dir = documents_dir.expanduser().resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / source_path.name
    shutil.copy2(source_path, destination_path)
    return destination_path


def index_uploaded_document(
    file_path: Path | str,
    vector_store: Chroma,
    documents_dir: Path = DOCUMENTS_DIR,
) -> Tuple[int, int]:
    """Load one new document, split it, and add its embeddings to the existing Chroma index."""
    validated_path = validate_uploaded_file(file_path)
    copied_path = copy_uploaded_document(validated_path, documents_dir)

    documents = _load_document_from_path(copied_path)
    existing_chunks = get_all_chunks(vector_store)
    chunks = split_documents(documents, start_index=len(existing_chunks))

    vector_store.add_documents(chunks)
    if hasattr(vector_store, "persist"):
        vector_store.persist()

    return len(chunks), len(chunks)


def get_all_chunks(vector_store: Chroma) -> List[Document]:
    """Return all indexed chunks from the vector store."""
    try:
        result = vector_store.get(include=["documents", "metadatas"])
        documents: List[Document] = []
        raw_documents = result.get("documents", []) or []
        raw_metadatas = result.get("metadatas", []) or []
        for index, content in enumerate(raw_documents):
            metadata = raw_metadatas[index] if index < len(raw_metadatas) else {}
            documents.append(Document(page_content=content, metadata=metadata or {}))
        return documents
    except Exception:
        return []


def main() -> None:
    """Run the ingestion pipeline end to end."""
    print("Starting ingestion...")
    try:
        ensure_documents_exists(DOCUMENTS_DIR)
        documents = load_documents(DOCUMENTS_DIR)
        print(f"Loaded {len(documents)} document(s).")

        chunks = split_documents(documents)
        print(f"Created {len(chunks)} chunk(s).")

        build_vector_database(chunks, VECTOR_DB_PATH)
        print("Indexing completed successfully.")
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
    except NotADirectoryError as exc:
        print(f"Error: {exc}")
    except ValueError as exc:
        print(f"Error: {exc}")
    except KeyboardInterrupt:
        print("\nIngestion cancelled by user.")
    except Exception as exc:
        print(f"Unexpected error during ingestion: {exc}")


if __name__ == "__main__":
    main()
