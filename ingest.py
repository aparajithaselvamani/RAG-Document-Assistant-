from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Tuple

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils import DOCUMENTS_DIR, EMBEDDING_MODEL, VECTOR_DB_PATH, build_chunk_metadata, ensure_documents_exists

SUPPORTED_UPLOAD_SUFFIXES = {".pdf", ".txt"}


@dataclass(frozen=True)
class DocumentOperationResult:
    """Outcome of an incremental document indexing operation."""

    status: Literal["added", "updated", "duplicate"]
    source_name: str
    chunks_created: int = 0
    duplicate_source: str | None = None

    @property
    def embeddings_generated(self) -> int:
        return self.chunks_created


# Chunking parameters (characters). Units are normally single sentences.
MAX_CHUNK_CHARS = 400
MIN_CHUNK_CHARS = 40
# Sentence end followed by whitespace and a capital/digit/quote; avoids splitting "e.g. bm25".
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
# A sentence opening with one of these depends on the previous sentence for meaning.
LEADING_REFERENCE_RE = re.compile(
    r"^(?:these|this|those|that|it|its|they|their|such|both|however|also|additionally|"
    r"furthermore|moreover|therefore|thus|as a result|in addition)\b",
    flags=re.IGNORECASE,
)


def _load_document_from_path(file_path: Path) -> List[Document]:
    """Load a single supported document and normalize its metadata."""
    digest = content_hash(file_path)
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
        metadata["content_hash"] = digest
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


def clean_text(text: str) -> str:
    """Remove byte-order marks and collapse layout whitespace before chunking."""
    text = (text or "").replace("\ufeff", "")
    return re.sub(r"\s+", " ", text).strip()


def split_into_sentences(text: str) -> List[str]:
    """Split cleaned text into sentences without breaking abbreviations like 'e.g.'."""
    sentences = [sentence.strip() for sentence in SENTENCE_BOUNDARY_RE.split(text) if sentence.strip()]
    return sentences


def chunk_text(text: str) -> List[str]:
    """Create sentence-aligned retrieval units.

    The old 500-character splitter turned every document into one chunk, so a
    question about one concept retrieved every other concept in the same file.
    Each unit is now one sentence (so it is about one idea), except that a
    sentence which opens with a back-reference ("These vectors ...") is kept
    with the sentence it refers to, and tiny fragments are merged forward.
    Sentences longer than MAX_CHUNK_CHARS fall back to the recursive splitter.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    units: List[str] = []
    for sentence in split_into_sentences(cleaned):
        if (
            units
            and (LEADING_REFERENCE_RE.match(sentence) or len(units[-1]) < MIN_CHUNK_CHARS)
            and len(units[-1]) + 1 + len(sentence) <= MAX_CHUNK_CHARS
        ):
            units[-1] = f"{units[-1]} {sentence}"
        else:
            units.append(sentence)

    fallback = RecursiveCharacterTextSplitter(chunk_size=MAX_CHUNK_CHARS, chunk_overlap=MAX_CHUNK_CHARS // 5)
    chunks: List[str] = []
    for unit in units:
        chunks.extend(fallback.split_text(unit) if len(unit) > MAX_CHUNK_CHARS else [unit])
    return chunks


def split_documents(documents: List[Document], start_index: int = 0) -> List[Document]:
    """Split documents into sentence-aligned chunks with position metadata.

    ``chunk_index`` records the chunk's position inside its source document so
    retrieval can pull in the neighbouring chunk when a chunk refers back to it.
    """
    chunks: List[Document] = []
    position_in_source: dict[str, int] = {}
    for document in documents:
        metadata = dict(document.metadata or {})
        source = metadata.get("source", "unknown")
        page_number = metadata.get("page_number") if metadata.get("page_number") is not None else metadata.get("page")
        for text in chunk_text(document.page_content):
            chunk_index = position_in_source.get(source, 0)
            position_in_source[source] = chunk_index + 1
            chunk_metadata = {key: value for key, value in metadata.items() if isinstance(value, (str, int, float, bool))}
            chunk_metadata.update(build_chunk_metadata(start_index + len(chunks), source, page_number))
            chunk_metadata["chunk_index"] = chunk_index
            chunks.append(Document(page_content=text, metadata=chunk_metadata))
    return chunks


def chunk_ids(chunks: List[Document]) -> List[str]:
    """Stable ids (source + position) so re-indexing a file replaces instead of duplicating it."""
    return [f"{chunk.metadata.get('source', 'unknown')}::{chunk.metadata.get('chunk_index', index)}" for index, chunk in enumerate(chunks)]


def content_hash(file_path: Path | str) -> str:
    """Return a SHA-256 digest of a document's exact file contents."""
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as document_file:
        for block in iter(lambda: document_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        ids=chunk_ids(chunks),
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
    if source_path != destination_path:
        shutil.copy2(source_path, destination_path)
    return destination_path


def _get_matching_rows(vector_store: Chroma, where: dict) -> dict:
    """Fetch only IDs and metadata for matching chunks, with compatibility for simple stores."""
    try:
        result = vector_store.get(where=where, include=["metadatas"])
    except TypeError:
        result = vector_store.get(include=["metadatas"])

    matched_ids: List[str] = []
    matched_metadatas: List[dict] = []
    ids = result.get("ids", []) or []
    metadatas = result.get("metadatas", []) or []
    for index, metadata in enumerate(metadatas):
        metadata = metadata or {}
        if all(metadata.get(key) == value for key, value in where.items()) and index < len(ids):
            matched_ids.append(ids[index])
            matched_metadatas.append(metadata)
    return {"ids": matched_ids, "metadatas": matched_metadatas}


def _backfill_legacy_hashes(vector_store: Chroma, documents_dir: Path, target_digest: str) -> str | None:
    """Hash legacy sources and add metadata without changing their embeddings."""
    result = vector_store.get(include=["metadatas"])
    ids = result.get("ids", []) or []
    metadatas = result.get("metadatas", []) or []
    legacy_rows: dict[str, list[tuple[str, dict]]] = {}
    for index, metadata in enumerate(metadatas):
        metadata = metadata or {}
        source = metadata.get("source")
        if source and "content_hash" not in metadata and index < len(ids):
            legacy_rows.setdefault(source, []).append((ids[index], metadata))

    collection = getattr(vector_store, "_collection", None)
    duplicate_source = None
    for source_name, rows in legacy_rows.items():
        if Path(source_name).name != source_name:
            continue
        source_path = documents_dir / source_name
        if not source_path.is_file():
            continue
        digest = content_hash(source_path)
        if collection is not None:
            collection.update(
                ids=[chunk_id for chunk_id, _ in rows],
                metadatas=[{**metadata, "content_hash": digest} for _, metadata in rows],
            )
        if digest == target_digest:
            duplicate_source = source_name
    return duplicate_source


def _find_duplicate_source(vector_store: Chroma, digest: str, documents_dir: Path) -> str | None:
    matches = _get_matching_rows(vector_store, {"content_hash": digest})
    for metadata in matches["metadatas"]:
        source = metadata.get("source")
        if source:
            return str(source)

    legacy_duplicate = _backfill_legacy_hashes(vector_store, documents_dir, digest)
    if legacy_duplicate:
        return legacy_duplicate
    matches = _get_matching_rows(vector_store, {"content_hash": digest})
    for metadata in matches["metadatas"]:
        source = metadata.get("source")
        if source:
            return str(source)
    return None


def remove_source_chunks(vector_store: Chroma, source_name: str) -> int:
    """Delete previously indexed chunks of one source file and return the removed count."""
    stale_ids = _get_matching_rows(vector_store, {"source": source_name})["ids"]
    if stale_ids:
        vector_store.delete(ids=stale_ids)
    return len(stale_ids)


def _index_document(
    validated_path: Path,
    digest: str,
    vector_store: Chroma,
    documents_dir: Path,
    status: Literal["added", "updated"],
) -> DocumentOperationResult:
    copied_path = copy_uploaded_document(validated_path, documents_dir)
    documents = _load_document_from_path(copied_path)
    chunks = split_documents(documents)
    for chunk in chunks:
        chunk.metadata["content_hash"] = digest

    remove_source_chunks(vector_store, copied_path.name)
    try:
        vector_store.add_documents(chunks, ids=chunk_ids(chunks))
    except TypeError:
        vector_store.add_documents(chunks)
    if hasattr(vector_store, "persist"):
        vector_store.persist()

    return DocumentOperationResult(status=status, source_name=copied_path.name, chunks_created=len(chunks))


def add_document(
    file_path: Path | str,
    vector_store: Chroma,
    documents_dir: Path = DOCUMENTS_DIR,
) -> DocumentOperationResult:
    """Add a document incrementally, or update/skip it when its name/content already exists."""
    validated_path = validate_uploaded_file(file_path)
    digest = content_hash(validated_path)
    destination_dir = documents_dir.expanduser().resolve()

    duplicate_source = _find_duplicate_source(vector_store, digest, destination_dir)
    if duplicate_source:
        return DocumentOperationResult(
            status="duplicate",
            source_name=validated_path.name,
            duplicate_source=duplicate_source,
        )

    existing_source = _get_matching_rows(vector_store, {"source": validated_path.name})["ids"]
    status = "updated" if existing_source else "added"
    return _index_document(validated_path, digest, vector_store, destination_dir, status)


def update_document(
    file_path: Path | str,
    vector_store: Chroma,
    documents_dir: Path = DOCUMENTS_DIR,
) -> DocumentOperationResult:
    """Replace the indexed version of a source file without changing other sources."""
    validated_path = validate_uploaded_file(file_path)
    source_ids = _get_matching_rows(vector_store, {"source": validated_path.name})["ids"]
    if not source_ids:
        raise FileNotFoundError(f"No indexed document named {validated_path.name!r}.")

    digest = content_hash(validated_path)
    destination_dir = documents_dir.expanduser().resolve()
    duplicate_source = _find_duplicate_source(vector_store, digest, destination_dir)
    if duplicate_source:
        return DocumentOperationResult(
            status="duplicate",
            source_name=validated_path.name,
            duplicate_source=duplicate_source,
        )
    return _index_document(validated_path, digest, vector_store, destination_dir, "updated")


def delete_document(
    source_name: str,
    vector_store: Chroma,
    documents_dir: Path = DOCUMENTS_DIR,
) -> int:
    """Delete a source's chunks and stored file, returning the number of removed chunks."""
    if not source_name or Path(source_name).name != source_name:
        raise ValueError(f"Expected a document filename, got {source_name!r}.")

    chunks_removed = remove_source_chunks(vector_store, source_name)
    if chunks_removed:
        stored_path = documents_dir.expanduser().resolve() / source_name
        if stored_path.is_file():
            stored_path.unlink()
        if hasattr(vector_store, "persist"):
            vector_store.persist()
    return chunks_removed


def index_uploaded_document(
    file_path: Path | str,
    vector_store: Chroma,
    documents_dir: Path = DOCUMENTS_DIR,
) -> Tuple[int, int]:
    """Backward-compatible wrapper for the incremental add/update operation."""
    result = add_document(file_path, vector_store, documents_dir)
    return result.chunks_created, result.embeddings_generated


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
