from __future__ import annotations

import re
from collections import deque
from pathlib import Path
from typing import Deque, List, Tuple

import ollama
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from hybrid_search import deduplicate_results, hybrid_search, normalize_semantic_results
from ingest import index_uploaded_document
from keyword_search import keyword_search
from utils import (
    DEFAULT_MODEL,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    ensure_documents_exists,
    format_snippet,
)

BASE_DIR = Path(__file__).resolve().parent
VECTOR_DB_DIR = BASE_DIR / "vector_db" / "chroma"
MAX_CONVERSATION_TURNS = 5


def update_conversation_history(history: Deque[Tuple[str, str]], user_question: str, assistant_answer: str) -> None:
    """Append a new interaction to the conversation history while keeping it bounded."""
    history.append((user_question.strip(), assistant_answer.strip()))


def format_conversation_history(history: Deque[Tuple[str, str]]) -> str:
    """Render the recent conversation history for prompt injection."""
    if not history:
        return ""

    formatted_turns = []
    for index, (user_question, assistant_answer) in enumerate(history, start=1):
        formatted_turns.append(
            f"{index}. User: {user_question}\n   Assistant: {assistant_answer}"
        )

    return "Conversation History:\n" + "\n\n".join(formatted_turns)


def load_vector_store(vector_db_dir: Path) -> Chroma:
    """Load a persistent Chroma vector store from disk."""
    if not vector_db_dir.exists():
        raise FileNotFoundError(
            f"Vector database not found at {vector_db_dir}. Run 'python ingest.py' first."
        )

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
    return Chroma(persist_directory=str(vector_db_dir), embedding_function=embeddings)


def get_all_chunks(vector_store: Chroma) -> List[Document]:
    """Return all indexed chunks from the vector store for keyword-based ranking."""
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


def rewrite_query(question: str) -> str:
    """Rewrite a user question to be easier to retrieve without inventing new meaning."""
    cleaned_question = (question or "").strip()
    if not cleaned_question:
        return ""

    normalized_question = re.sub(r"\s+", " ", cleaned_question).strip()
    normalized_lower = normalized_question.lower()

    if normalized_lower in {"what is rag?", "what is rag", "what is retrieval augmented generation?", "what is retrieval augmented generation"}:
        return "What is Retrieval-Augmented Generation (RAG)?"
    if normalized_lower in {"how work?", "how work", "how it work?", "how it work", "how it works?", "how it works"}:
        return "How does it work?"
    if normalized_lower in {"vector db?", "vector db", "vector database?", "vector database"}:
        return "What is a vector database?"
    if normalized_lower in {"hybrid search?", "hybrid search"}:
        return "hybrid search"
    if normalized_lower in {"what is query rewriting?", "what is query rewriting"}:
        return "What is query rewriting?"
    if re.fullmatch(r"what\s+is\s+retrieval(?:-|\s+)augmented(?:-|\s+)generation\s*\??", normalized_lower):
        return "What is Retrieval-Augmented Generation (RAG)?"
    if re.fullmatch(r"explain\s+retrieval(?:-|\s+)augmented(?:-|\s+)generation\s*\??", normalized_lower):
        return "Explain Retrieval-Augmented Generation."
    if re.fullmatch(r"how\s+does\s+it\s+work\s*\??", normalized_lower):
        return "How does it work?"
    if re.fullmatch(r"(?:what|how|why|when|where|who|which|explain|tell me about|can|do|does|is|are)\b.*", normalized_lower):
        return cleaned_question

    prompt = f"""You are rewriting search queries for a document retrieval system.

Rewrite the user's question so it is easier to match against the provided documents.

Rules:
- Preserve the user's original intent.
- Fix spelling and grammar when needed.
- Expand abbreviations only when the meaning is known with high confidence.
- Never invent or guess meanings.
- Keep already-clear queries unchanged.
- Do not answer the question.
- Return ONLY the rewritten query.

User question:
{cleaned_question}
"""

    try:
        response = ollama.generate(model=DEFAULT_MODEL, prompt=prompt)
        rewritten_query = re.sub(r"\s+", " ", response.get("response", "").strip())
        if rewritten_query:
            return rewritten_query
    except Exception:
        pass

    return cleaned_question


def retrieve_context(
    vector_store: Chroma,
    question: str,
    top_k: int = DEFAULT_TOP_K,
) -> Tuple[str, List[Tuple[Document, float]], List[Tuple[Document, float]], List[Tuple[Document, float]]]:
    """Retrieve semantic, keyword, and hybrid results for a question."""
    rewritten_query = rewrite_query(question)
    semantic_distances = vector_store.similarity_search_with_score(rewritten_query, k=top_k * 2)
    all_chunks = get_all_chunks(vector_store)
    keyword_results = keyword_search(rewritten_query, all_chunks, top_k=top_k * 2)
    semantic_results = normalize_semantic_results(semantic_distances)
    keyword_results = deduplicate_results(keyword_results)
    hybrid_results = hybrid_search(
        rewritten_query,
        all_chunks,
        semantic_distances,
        top_k=top_k,
        keyword_results=keyword_results,
    )

    return rewritten_query, semantic_results[:top_k], keyword_results[:top_k], hybrid_results[:top_k]


def generate_answer(
    question: str,
    context: List[Document],
    conversation_history: Deque[Tuple[str, str]] | None = None,
) -> str:
    """Generate an answer from retrieved context using Ollama, with a safe fallback."""
    if not context:
        return "I could not find that information in the provided documents."

    context_text = "\n\n".join(document.page_content for document in context)
    history_text = format_conversation_history(conversation_history or deque(maxlen=MAX_CONVERSATION_TURNS))
    prompt_sections = [
        "You are a helpful assistant.",
        "",
        "Answer ONLY using the Retrieved Context as factual evidence.",
        "",
        "Guidelines:",
        "- Answer the user's current question directly.",
        "- Do not change the subject or answer a different question.",
        "- Synthesize information from multiple relevant chunks when needed.",
        "- Avoid repeating the same point.",
        "- Be concise but complete.",
        "- Do not invent information.",
        "- Use Conversation History only to resolve references such as 'it' or 'which one'; it is not factual evidence.",
        "- If the Retrieved Context does not contain enough evidence, say exactly:",
        "  'I could not find that information in the provided documents.'",
        "",
        "Retrieved Context:",
        context_text,
        "",
    ]
    if history_text:
        prompt_sections.extend([history_text, ""])
    prompt_sections.extend(["Question:", question])
    prompt = "\n".join(prompt_sections)

    try:
        response = ollama.generate(model=DEFAULT_MODEL, prompt=prompt)
        answer = response.get("response", "").strip()
        if answer:
            return answer
    except Exception:
        pass

    return "I could not find that information in the provided documents."


def display_search_results(title: str, results: List[Tuple[Document, float]], score_label: str = "Score") -> None:
    """Display a search result section clearly in the terminal."""
    print(f"\n----------------------------------")
    print(title)
    print("----------------------------------")
    if not results:
        print("No results found.")
        return

    for index, (chunk, score) in enumerate(results, start=1):
        source = chunk.metadata.get("source") if chunk.metadata else None
        document_name = Path(str(source)).name if source else "Unknown source"
        print(f"\n{index}.")
        print(f"Document: {document_name}")
        print(f"{score_label}: {score:.2f}")
        print("Snippet:")
        print(format_snippet(chunk.page_content))


def display_sources(retrieved_chunks: List[Document]) -> None:
    """Display a clear source attribution section for the retrieved chunks."""
    print("\n----------------------------------")
    print("Sources")
    print("----------------------------------")

    for index, chunk in enumerate(retrieved_chunks, start=1):
        source = chunk.metadata.get("source") if chunk.metadata else None
        document_name = Path(str(source)).name if source else "Unknown source"
        snippet = format_snippet(chunk.page_content)
        print(f"\nChunk {index}")
        print(f"Document: {document_name}")
        print("Snippet:")
        print(snippet)


def display_menu() -> None:
    """Show the main application menu."""
    print("\n----------------------------------")
    print("RAG Document Assistant")
    print("----------------------------------")
    print("1. Ask questions")
    print("2. Upload a document")
    print("3. Exit")


def handle_upload(vector_store: Chroma) -> None:
    """Prompt for a file path, validate it, copy it into the documents folder, and index it."""
    print("\nEnter file path:")
    file_path = input().strip()
    if not file_path:
        print("No file path provided.")
        return

    try:
        chunks_created, embeddings_generated = index_uploaded_document(file_path, vector_store)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return
    except ValueError as exc:
        print(f"Error: {exc}")
        return
    except Exception as exc:
        print(f"Error indexing uploaded document: {exc}")
        return

    print("\n----------------------------------")
    print("Upload Successful")
    print("----------------------------------")
    print(f"Filename: {Path(file_path).name}")
    print(f"Chunks Created: {chunks_created}")
    print(f"Embeddings Generated: {embeddings_generated}")
    print("Vector Database Updated Successfully.")


def main() -> None:
    """Start the interactive question-answering loop."""
    print("Loading the RAG assistant...")
    ensure_documents_exists()
    try:
        vector_store = load_vector_store(VECTOR_DB_DIR)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return
    except Exception as exc:
        print(f"Error loading vector database: {exc}")
        return

    conversation_history: Deque[Tuple[str, str]] = deque(maxlen=MAX_CONVERSATION_TURNS)

    while True:
        display_menu()
        try:
            choice = input("\nChoose an option: ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break

        if choice == "1":
            try:
                question = input("\nQuestion: ").strip()
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except EOFError:
                print("\nGoodbye!")
                break

            if not question:
                continue
            if question.lower() == "exit":
                print("Goodbye!")
                break

            try:
                rewritten_query, semantic_results, keyword_results, hybrid_results = retrieve_context(
                    vector_store,
                    question,
                )
            except Exception as exc:
                print(f"Error retrieving context: {exc}")
                continue

            print("\n----------------------------------")
            print("Question")
            print("----------------------------------")
            print(question)

            print("\n----------------------------------")
            print("Rewritten Query")
            print("----------------------------------")
            print(rewritten_query)

            print("\n----------------------------------")
            print("Semantic Search")
            print("----------------------------------")
            display_search_results("Document", semantic_results, score_label="Similarity")

            print("\n----------------------------------")
            print("Keyword Search")
            print("----------------------------------")
            display_search_results("Document", keyword_results, score_label="Score")

            print("\n----------------------------------")
            print("Hybrid Ranking")
            print("----------------------------------")
            display_search_results("Document", hybrid_results, score_label="Combined Score")

            print("\n----------------------------------")
            print("Answer")
            print("----------------------------------")
            # Only hybrid-qualified chunks are supplied to the model and cited.
            # Falling back to unfiltered vector neighbours would reintroduce
            # irrelevant source attribution.
            answer_chunks = [chunk for chunk, _ in hybrid_results]
            answer = generate_answer(question, answer_chunks, conversation_history=conversation_history)
            update_conversation_history(conversation_history, question, answer)
            print(answer)
            display_sources(answer_chunks)
        elif choice == "2":
            handle_upload(vector_store)
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid option. Please choose 1, 2, or 3.")


if __name__ == "__main__":
    main()
