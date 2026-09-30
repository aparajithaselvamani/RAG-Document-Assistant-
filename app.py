from __future__ import annotations

import re
from collections import deque
from pathlib import Path
from typing import Deque, List, Tuple

import ollama
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from hybrid_search import deduplicate_results, expand_with_context, hybrid_search, normalize_semantic_results
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
VALID_CLASSIFICATION_LABELS = {"DOCUMENT_RELATED", "FOLLOW_UP", "GENERAL_OR_UNRELATED"}
NO_INFORMATION_RESPONSE = "I could not find that information in the provided documents."


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


def _normalize_question(question: str) -> str:
    """Normalize question text for deterministic classification."""
    return " ".join((question or "").strip().lower().split())


def _question_subject(question: str) -> str:
    """Extract the subject of a simple prior user question for reference resolution."""
    cleaned = re.sub(r"\s+", " ", (question or "").strip()).rstrip("?.! ")
    match = re.match(r"(?:what|who)\s+(?:is|are)\s+(.+)", cleaned, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.match(r"(?:explain|define|tell me about)\s+(.+)", cleaned, flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _recent_subject(history: Deque[Tuple[str, str]], before_index: int | None = None) -> str:
    """Find the latest explicit user-question subject, never using assistant answers."""
    questions = [user_question for user_question, _ in history]
    if before_index is not None:
        questions = questions[:before_index]
    for prior_question in reversed(questions):
        subject = _question_subject(prior_question)
        if subject:
            return subject
    return ""


def resolve_follow_up_question(question: str, history: Deque[Tuple[str, str]] | None = None) -> str:
    """Resolve a small set of ambiguous follow-up references from prior user questions.

    This does not add facts or alter retrieval.  It only restates the user's
    current intent so answer generation can select the right retrieved evidence.
    """
    current_question = (question or "").strip()
    if not current_question or not history:
        return current_question

    normalized = current_question.lower()
    if not re.search(r"\b(it|they|that|which one|which of them|the first one|the second one)\b", normalized):
        return current_question

    prior_questions = [user_question for user_question, _ in history]
    latest_question = prior_questions[-1] if prior_questions else ""
    comparison = re.search(r"\bdifferent\s+from\s+(.+?)[?.!]*$", latest_question, flags=re.IGNORECASE)
    if re.search(r"\b(which one|which of them)\b", normalized) and comparison:
        second_subject = comparison.group(1).strip().rstrip("?.! ")
        first_subject = _recent_subject(history, before_index=len(prior_questions) - 1)
        if first_subject and second_subject:
            return re.sub(
                r"\b(which one|which of them)\b",
                f"Which of {first_subject} and {second_subject}",
                current_question,
                count=1,
                flags=re.IGNORECASE,
            )

    subject = _recent_subject(history)
    if subject:
        return re.sub(r"\bit\b", subject, current_question, flags=re.IGNORECASE)
    return current_question


def classify_query(question: str, conversation_history: Deque[Tuple[str, str]] | None = None) -> dict[str, str]:
    """Classify incoming questions before retrieval.

    The classifier is intentionally small and deterministic: it prefers explicit
    follow-up references when history is available, otherwise it uses document
    topic keywords to decide whether retrieval is appropriate.
    """
    normalized = _normalize_question(question)
    history = conversation_history or deque(maxlen=MAX_CONVERSATION_TURNS)

    if not normalized:
        return {"category": "GENERAL_OR_UNRELATED"}

    reference_terms = (
        "it",
        "they",
        "that",
        "this",
        "which one",
        "which of them",
        "how far in advance",
        "how does it",
        "how is it",
        "what is it",
    )
    history_topics = []
    for prior_question, _ in history:
        history_topics.extend(_normalize_question(prior_question).split())

    if history and any(term in normalized for term in reference_terms):
        return {"category": "FOLLOW_UP"}

    document_topic_terms = (
        "rag",
        "retrieval augmented generation",
        "semantic search",
        "keyword search",
        "hybrid search",
        "embedding",
        "embeddings",
        "vector database",
        "query rewriting",
        "ollama",
        "chroma",
        "leave policy",
        "vacation",
        "retrieval",
        "document store",
        "documents",
    )
    about_match = re.search(r"\babout\s+(.+?)[?.!]*$", normalized)
    if about_match:
        about_subject = about_match.group(1)
        if not any(term in about_subject for term in document_topic_terms):
            return {"category": "GENERAL_OR_UNRELATED"}
    if any(term in normalized for term in document_topic_terms):
        return {"category": "DOCUMENT_RELATED"}

    history_has_topic = any(term in " ".join(history_topics) for term in ("rag", "semantic", "keyword", "embedding", "retrieval", "leave", "document"))
    if history and history_has_topic and any(word in normalized for word in ("it", "they", "that", "this", "which one")):
        return {"category": "FOLLOW_UP"}

    return {"category": "GENERAL_OR_UNRELATED"}


def has_sufficient_grounding(question: str, context: List[Document]) -> bool:
    """Reject unsupported questions when the retrieved evidence does not cover the current fact."""
    if not context:
        return False

    normalized_question = _normalize_question(question)
    question_tokens = [
        token for token in re.split(r"[^a-z0-9]+", normalized_question)
        if token and token not in {
            "what", "how", "why", "when", "where", "which", "who", "is", "are",
            "the", "a", "an", "of", "to", "it", "does", "do", "did", "should",
            "in", "on", "for", "from", "about", "say", "says", "use", "uses",
            "different", "one", "work", "works", "doesnt", "doesn", "shouldn",
        }
    ]
    if not question_tokens:
        return False

    significant_terms = set(question_tokens)
    for document in context:
        content = _normalize_question(document.page_content)
        if not content:
            continue
        content_tokens = {token for token in re.split(r"[^a-z0-9]+", content) if token}
        if significant_terms.issubset(content_tokens):
            return True
        if any(term in content for term in significant_terms):
            # Named entities and specific fact terms must be present in the evidence;
            # otherwise a nearby keyword like 'RAG' should not satisfy a different fact.
            if all(term in content for term in significant_terms if term not in {"rag", "semantic", "keyword", "retrieval", "embedding", "embeddings", "vector", "database", "search"}):
                return True
    return False


def route_query(
    question: str,
    conversation_history: Deque[Tuple[str, str]] | None = None,
    vector_store: Chroma | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """Route a question by category before any retrieval happens for unrelated requests."""
    history = conversation_history or deque(maxlen=MAX_CONVERSATION_TURNS)
    category = classify_query(question, history)["category"]
    resolved_question = resolve_follow_up_question(question, history)

    result = {
        "category": category,
        "resolved_question": resolved_question,
        "rewritten_query": resolved_question,
        "semantic_results": [],
        "keyword_results": [],
        "hybrid_results": [],
        "answer": NO_INFORMATION_RESPONSE,
    }

    if category == "GENERAL_OR_UNRELATED":
        return result

    if vector_store is None:
        return result

    rewritten_query, semantic_results, keyword_results, hybrid_results = retrieve_context(
        vector_store,
        question,
        top_k=top_k,
        conversation_history=history,
    )
    result.update(
        {
            "rewritten_query": rewritten_query,
            "semantic_results": semantic_results,
            "keyword_results": keyword_results,
            "hybrid_results": hybrid_results,
        }
    )

    answer_context = [document for document, _ in hybrid_results]
    result["answer"] = generate_answer(question, answer_context, conversation_history=history)
    return result


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
    conversation_history: Deque[Tuple[str, str]] | None = None,
) -> Tuple[str, List[Tuple[Document, float]], List[Tuple[Document, float]], List[Tuple[Document, float]]]:
    """Retrieve semantic, keyword, and hybrid results using conversation-aware question resolution."""
    history = conversation_history or deque(maxlen=MAX_CONVERSATION_TURNS)
    resolved_question = resolve_follow_up_question(question, history)
    rewritten_query = rewrite_query(resolved_question)

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
    # Chunks are sentence-sized; pull in the preceding chunk(s) when a selected
    # chunk refers back to them ("both approaches", "these vectors").
    hybrid_results = expand_with_context(hybrid_results, all_chunks)

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
    history = conversation_history or deque(maxlen=MAX_CONVERSATION_TURNS)
    history_text = format_conversation_history(history)
    resolved_question = resolve_follow_up_question(question, history)
    prompt_sections = [
        "You are a helpful assistant for a document-grounded RAG system.",
        "",
        "Answer the Question to Answer.",
        "Use ONLY the Retrieved Context as factual evidence.",
        "",
        "Guidelines:",
        "- Conversation History is ONLY for resolving references and understanding what the user means; it is NOT factual evidence.",
        "- A Resolved Current Question is ONLY a clarification of the user's reference; verify every factual claim in Retrieved Context.",
        "- If the user says 'which one' after comparing two subjects, answer which of those compared subjects satisfies the question.",
        "- Do not substitute an unrelated entity from Retrieved Context for one of the compared subjects; the chosen answer must be the same subject and not an unrelated term in Retrieved Context.",
        "- For example, if the conversation compares semantic search and keyword search and the user asks 'Which one uses embeddings?', answer about semantic search and keyword search, not ChromaDB.",
        "- Answer the user's current intent directly and concisely.",
        "- Synthesize multiple relevant chunks when needed.",
        "- Do not invent information or use outside knowledge.",
        "- If the Retrieved Context does not contain enough evidence, say exactly:",
        "  'I could not find that information in the provided documents.'",
        "",
        "Retrieved Context:",
        context_text,
        "",
    ]
    if history_text:
        prompt_sections.extend(["Conversation History (reference resolution only; not factual evidence):", history_text.removeprefix("Conversation History:\n"), ""])
    prompt_sections.extend(["Current Question:", question])
    if resolved_question != question:
        prompt_sections.extend(["", "Question to Answer:", resolved_question])
    else:
        prompt_sections.extend(["", "Question to Answer:", question])
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

            classification = classify_query(question, conversation_history)
            if classification["category"] == "GENERAL_OR_UNRELATED":
                answer = NO_INFORMATION_RESPONSE
                update_conversation_history(conversation_history, question, answer)
                print("\n----------------------------------")
                print("Question")
                print("----------------------------------")
                print(question)
                print("\n----------------------------------")
                print("Classification")
                print("----------------------------------")
                print(classification["category"])
                print("\nAnswer")
                print(answer)
                continue

            try:
                rewritten_query, semantic_results, keyword_results, hybrid_results = retrieve_context(
                    vector_store,
                    question,
                    conversation_history=conversation_history,
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