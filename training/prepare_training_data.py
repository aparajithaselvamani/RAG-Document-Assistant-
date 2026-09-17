"""Build and validate a behavior-focused SFT dataset from documents/ only."""

from __future__ import annotations

import argparse
import difflib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

TRAINING_DIR = Path(__file__).resolve().parent
ROOT_DIR = TRAINING_DIR.parent
DOCUMENTS_DIR = ROOT_DIR / "documents"
EVALUATION_PATH = ROOT_DIR / "evaluation" / "evaluation_questions.json"
REFUSAL = "I could not find that information in the provided documents."
INSTRUCTION = "Answer using only the retrieved context. Preserve terminology from the context, be concise, and do not infer facts that are not stated. If the answer is not supported, say exactly: I could not find that information in the provided documents."
REQUIRED_FIELDS = {"instruction", "input", "context", "source", "output", "category"}
VALID_CATEGORIES = {"direct_grounded", "multi_context", "refusal", "follow_up", "reference_resolution", "comparison", "clarification", "terminology", "concise_technical", "adversarial"}
SOURCE_NAMES = {"rag_basics.txt", "embeddings_and_chroma.txt", "search_methods.txt", "upload_test.txt", "query_rewriting_and_ollama.pdf"}


def read_source(name: str) -> str:
    path = DOCUMENTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        text = "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages).strip()
    else:
        text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"Source document is empty: {name}")
    return text


def source_sentences() -> dict[str, list[str]]:
    sources = {name: read_source(name) for name in sorted(SOURCE_NAMES)}
    return {name: [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()] for name, text in sources.items()}


def make(category: str, question: str, context: str, source: str, output: str) -> dict[str, str]:
    if category not in VALID_CATEGORIES:
        raise ValueError(f"Unknown category: {category}")
    return {"instruction": INSTRUCTION, "input": question, "context": context, "source": source, "output": output, "category": category}


def build_examples() -> list[dict[str, str]]:
    s = source_sentences()
    rag, emb, search, leave, rewrite = (s[name] for name in ("rag_basics.txt", "embeddings_and_chroma.txt", "search_methods.txt", "upload_test.txt", "query_rewriting_and_ollama.pdf"))
    examples: list[dict[str, str]] = []

    def add(category: str, question: str, context: str, source: str, output: str) -> None:
        examples.append(make(category, question, context, source, output))

    direct = [
        ("Which capability lets a language model use information outside its weights?", rag[0], "rag_basics.txt", "Retrieval-Augmented Generation lets a language model use external documents."),
        ("What does the retrieval stage provide to the answering model?", rag[1], "rag_basics.txt", "It provides relevant document chunks as context."),
        ("When is external retrieval especially useful?", rag[2], "rag_basics.txt", "It is useful when current or domain-specific knowledge is not stored in the model's weights."),
        ("What form do embeddings give to text?", emb[0], "embeddings_and_chroma.txt", "Embeddings convert text into vectors that capture semantic meaning."),
        ("What can vector representations help a search system locate?", emb[1], "embeddings_and_chroma.txt", "They help locate related content even when the wording differs."),
        ("What does the vector database store?", emb[2], "embeddings_and_chroma.txt", "ChromaDB stores embeddings and supports semantic search over document chunks."),
        ("What signal does semantic retrieval use?", search[0], "search_methods.txt", "Semantic search uses vector similarity."),
        ("What kind of overlap supports keyword retrieval?", search[1], "search_methods.txt", "Keyword search uses lexical overlap and term statistics such as TF-IDF or BM25."),
        ("What does the hybrid method do with its results?", search[2], "search_methods.txt", "It merges and re-ranks semantic and keyword search results."),
        ("What benefit is stated for the leave policy?", leave[0], "upload_test.txt", "The policy allows employees to take paid vacation days during the year."),
        ("What advance notice does the handbook specify for leave?", leave[1], "upload_test.txt", "Employees should request leave at least two weeks in advance."),
        ("What does the query-rewriting source say about retrieval?", rewrite[0], "query_rewriting_and_ollama.pdf", "It says that query rewriting improves retrieval."),
        ("Which document capability supplies evidence before generation?", rag[0] + " " + rag[1], "rag_basics.txt", "RAG retrieves relevant chunks from external documents and passes them to the model."),
        ("What is the stated relationship between meaning and vector form?", emb[0] + " " + emb[1], "embeddings_and_chroma.txt", "Embeddings represent semantic meaning as vectors that help find related content."),
        ("What does the search corpus say about lexical methods?", search[1], "search_methods.txt", "Lexical methods use word overlap and statistics such as TF-IDF or BM25."),
        ("What is the documented role of the handbook?", leave[0] + " " + leave[1], "upload_test.txt", "It documents paid vacation and a two-week advance request guideline."),
        ("What does the source explicitly claim query rewriting improves?", rewrite[0], "query_rewriting_and_ollama.pdf", "The source explicitly claims that query rewriting improves retrieval."),
        ("What type of knowledge motivates the use of document retrieval?", rag[2], "rag_basics.txt", "Up-to-date or domain-specific knowledge motivates document retrieval."),
        ("What does ChromaDB support according to the source?", emb[2], "embeddings_and_chroma.txt", "ChromaDB supports semantic search over document chunks."),
        ("What does the keyword passage name besides lexical overlap?", search[1], "search_methods.txt", "It names term statistics such as TF-IDF and BM25."),
        ("What does the policy say employees may take?", leave[0], "upload_test.txt", "Employees may take paid vacation days during the year."),
        ("What is the timing rule in the handbook?", leave[1], "upload_test.txt", "Leave should be requested at least two weeks in advance."),
        ("What does the retrieval passage say is passed onward?", rag[1], "rag_basics.txt", "Relevant chunks and their context are passed to the model."),
        ("What wording difference can embeddings tolerate?", emb[1], "embeddings_and_chroma.txt", "They can find related content even when the wording differs."),
        ("What is the search behavior based on vector similarity?", search[0], "search_methods.txt", "It is semantic search."),
        ("What is the search behavior based on term statistics?", search[1], "search_methods.txt", "It is keyword search."),
        ("What operation combines both search approaches?", search[2], "search_methods.txt", "Hybrid search combines them by merging and re-ranking results."),
        ("What single improvement is attributed to query rewriting?", rewrite[0], "query_rewriting_and_ollama.pdf", "Improved retrieval is attributed to query rewriting."),
    ]
    for item in direct: add("direct_grounded", *item)

    synthesis = [
        ("How do external documents and vector representations work together in retrieval?", rag[1] + " " + emb[0], "rag_basics.txt;embeddings_and_chroma.txt", "Documents provide chunks as context, while embeddings represent text as vectors for finding related content."),
        ("How does semantic search connect to the storage layer?", emb[0] + " " + emb[2] + " " + search[0], "embeddings_and_chroma.txt;search_methods.txt", "Embeddings represent text as vectors, ChromaDB stores them, and semantic search uses vector similarity."),
        ("How do keyword and semantic evidence differ in a hybrid search?", search[0] + " " + search[1] + " " + search[2], "search_methods.txt", "Semantic evidence uses vector similarity; keyword evidence uses lexical overlap and term statistics; hybrid search merges and re-ranks both."),
        ("How could retrieved chunks support a concise answer?", rag[1] + " " + rag[2], "rag_basics.txt", "Retrieved chunks provide evidence for the model, especially when the needed knowledge is current or domain-specific."),
        ("What two facts define the leave guidance?", leave[0] + " " + leave[1], "upload_test.txt", "The policy allows paid vacation, and leave should be requested at least two weeks in advance."),
        ("How are query rewriting and retrieval related in the available sources?", rewrite[0] + " " + rag[1], "query_rewriting_and_ollama.pdf;rag_basics.txt", "Query rewriting improves retrieval, which supplies context to the answering model."),
        ("How do semantic meaning and lexical overlap represent different retrieval signals?", emb[0] + " " + search[0] + " " + search[1], "embeddings_and_chroma.txt;search_methods.txt", "Semantic meaning is represented through embeddings and vector similarity, while lexical overlap and term statistics support keyword search."),
        ("What pipeline relationship is supported by the RAG and search passages?", rag[0] + " " + search[2], "rag_basics.txt;search_methods.txt", "RAG uses retrieval to obtain external evidence, and hybrid search can merge semantic and keyword results."),
        ("How does the vector representation support document search?", emb[0] + " " + emb[1] + " " + emb[2], "embeddings_and_chroma.txt", "Vectors capture semantic meaning, help find related wording, and can be stored in ChromaDB for semantic search."),
        ("What facts would answer a question about requesting paid leave?", leave[0] + " " + leave[1], "upload_test.txt", "The policy allows paid vacation and says requests should be made at least two weeks in advance."),
        ("How does the source chain move from retrieval to answering?", rag[0] + " " + rag[1] + " " + rag[2], "rag_basics.txt", "RAG retrieves document chunks, passes them to the model, and is useful for knowledge absent from the model's weights."),
        ("What are the two parts of the documented hybrid method?", search[0] + " " + search[1] + " " + search[2], "search_methods.txt", "The method combines semantic search based on vector similarity with keyword search based on lexical statistics."),
    ]
    for item in synthesis: add("multi_context", *item)

    refusals = [
        ("Which programming language is described in the retrieved passage?", rag[0], "rag_basics.txt"),
        ("What date was the vector database released?", emb[2], "embeddings_and_chroma.txt"),
        ("Who authored the search-methods passage?", search[0], "search_methods.txt"),
        ("What is the maximum number of vacation days?", leave[0], "upload_test.txt"),
        ("What does the PDF say about model quantization?", rewrite[0], "query_rewriting_and_ollama.pdf"),
        ("Which city is named in the retrieved RAG passage?", rag[1], "rag_basics.txt"),
        ("What is the embedding vector dimension?", emb[0], "embeddings_and_chroma.txt"),
        ("What ranking weights does hybrid search use?", search[2], "search_methods.txt"),
        ("Does the handbook state whether remote work is allowed?", leave[1], "upload_test.txt"),
        ("What is the query rewriting algorithm's accuracy?", rewrite[0], "query_rewriting_and_ollama.pdf"),
        ("Which employee must approve a leave request?", leave[0] + " " + leave[1], "upload_test.txt"),
        ("What hardware runs the retrieval system?", rag[0] + " " + emb[2], "rag_basics.txt;embeddings_and_chroma.txt"),
        ("What is the storage capacity of ChromaDB?", emb[2], "embeddings_and_chroma.txt"),
        ("Which external website does the search passage cite?", search[1], "search_methods.txt"),
    ]
    for question, context, source in refusals: add("refusal", question, context, source, REFUSAL)

    followups = [
        ("Earlier topic: document retrieval. Current: Why is that useful?", rag[2], "rag_basics.txt", "It is useful for up-to-date or domain-specific knowledge that is not stored in the model's weights."),
        ("Earlier topic: vector representations. Current: What can they find despite wording changes?", emb[1], "embeddings_and_chroma.txt", "They can find related content even when the wording differs."),
        ("Earlier topic: lexical retrieval. Current: Which statistics does it use?", search[1], "search_methods.txt", "It uses term statistics such as TF-IDF or BM25."),
        ("Earlier topic: paid vacation. Current: How early should that be requested?", leave[1], "upload_test.txt", "It should be requested at least two weeks in advance."),
        ("Earlier topic: the combined search method. Current: What does it merge?", search[2], "search_methods.txt", "It merges semantic and keyword search results."),
        ("Earlier topic: query rewriting. Current: What does that improve?", rewrite[0], "query_rewriting_and_ollama.pdf", "It improves retrieval."),
        ("Earlier topic: embeddings stored in a vector database. Current: What does the database support?", emb[2], "embeddings_and_chroma.txt", "ChromaDB supports semantic search over document chunks."),
        ("Earlier topic: retrieved evidence. Current: Where is that passed?", rag[1], "rag_basics.txt", "It is passed to the language model as context."),
        ("Earlier topic: semantic and lexical signals. Current: How are they combined?", search[0] + " " + search[1] + " " + search[2], "search_methods.txt", "Hybrid search merges and re-ranks the semantic and keyword results."),
        ("Earlier topic: the handbook's timing rule. Current: What does the rule require?", leave[1], "upload_test.txt", "It requires employees to request leave at least two weeks in advance."),
    ]
    for item in followups: add("follow_up", *item)

    references = [
        ("Earlier: vectors capture meaning. Current: What do these representations enable?", emb[1], "embeddings_and_chroma.txt", "They enable finding related content even when wording differs."),
        ("Earlier: keyword evidence uses term statistics. Current: Which approach is being referred to?", search[1], "search_methods.txt", "The referenced approach is keyword search."),
        ("Earlier: vector similarity finds related passages. Current: Name that method.", search[0], "search_methods.txt", "The method is semantic search."),
        ("Earlier: the policy includes a benefit. Current: What benefit is it?", leave[0], "upload_test.txt", "The benefit is paid vacation days during the year."),
        ("Earlier: a combined ranking process. Current: What does it combine?", search[2], "search_methods.txt", "It combines semantic and keyword search."),
        ("Earlier: external evidence is retrieved. Current: What is passed onward?", rag[1], "rag_basics.txt", "The retrieved context is passed to the model."),
        ("Earlier: a retrieval improvement was mentioned. Current: Identify it.", rewrite[0], "query_rewriting_and_ollama.pdf", "The improvement is query rewriting improving retrieval."),
        ("Earlier: a vector store was named. Current: What does it store?", emb[2], "embeddings_and_chroma.txt", "ChromaDB stores embeddings."),
    ]
    for item in references: add("reference_resolution", *item)

    comparisons = [
        ("Contrast semantic retrieval with lexical retrieval.", search[0] + " " + search[1], "search_methods.txt", "Semantic retrieval uses vector similarity, while lexical retrieval uses word overlap and term statistics."),
        ("How do vectors differ from term statistics as search evidence?", emb[0] + " " + search[1], "embeddings_and_chroma.txt;search_methods.txt", "Vectors capture semantic meaning, while term statistics support lexical keyword matching."),
        ("Compare external-document knowledge with knowledge in model weights.", rag[0] + " " + rag[2], "rag_basics.txt", "RAG supplies external document knowledge when the needed current or domain-specific knowledge is not in the model's weights."),
        ("How is a combined search different from either individual signal?", search[0] + " " + search[1] + " " + search[2], "search_methods.txt", "It uses both semantic and keyword evidence by merging and re-ranking their results."),
        ("Compare the two leave facts documented in the handbook.", leave[0] + " " + leave[1], "upload_test.txt", "One fact concerns paid vacation; the other requires at least two weeks' advance notice."),
        ("How do retrieval and query rewriting differ in their roles?", rewrite[0] + " " + rag[1], "query_rewriting_and_ollama.pdf;rag_basics.txt", "Query rewriting improves retrieval, while retrieval supplies context to the language model."),
        ("Compare storing embeddings with using them for semantic matching.", emb[0] + " " + emb[2], "embeddings_and_chroma.txt", "Embeddings represent text as vectors; ChromaDB stores them for semantic search."),
        ("How does keyword search differ from hybrid search?", search[1] + " " + search[2], "search_methods.txt", "Keyword search relies on lexical statistics, while hybrid search combines that signal with semantic search."),
        ("Compare the two reasons RAG is useful according to the source.", rag[0] + " " + rag[2], "rag_basics.txt", "RAG uses external documents and is useful for knowledge that is current or domain-specific and absent from model weights."),
        ("How do the document-store and vector-database roles differ?", rag[1] + " " + emb[2], "rag_basics.txt;embeddings_and_chroma.txt", "The document store supplies retrieved chunks, while ChromaDB stores embeddings for semantic search."),
    ]
    for item in comparisons: add("comparison", *item)

    clarifications = [
        ("In the context, does 'related' mean identical wording?", emb[1], "embeddings_and_chroma.txt", "No. The context says related content can be found even when wording differs."),
        ("When the passage says 'both approaches,' which approaches are meant?", search[2] + " " + search[0] + " " + search[1], "search_methods.txt", "It means semantic search and keyword search."),
        ("In the RAG passage, what does 'that context' refer to?", rag[1], "rag_basics.txt", "It refers to the relevant chunks retrieved from the document store."),
        ("Does 'up-to-date' describe the documents or the model weights?", rag[2], "rag_basics.txt", "It describes knowledge that may be needed but is not stored in the model's weights."),
        ("In the leave passage, what does 'in advance' modify?", leave[1], "upload_test.txt", "It modifies the timing of the leave request: at least two weeks before leave."),
        ("What does 'improves' refer to in the query-rewriting source?", rewrite[0], "query_rewriting_and_ollama.pdf", "It refers to improving retrieval."),
    ]
    for item in clarifications: add("clarification", *item)

    terminology = [
        ("Use the source's term for vector-based meaning representation.", emb[0], "embeddings_and_chroma.txt", "The source calls these representations embeddings."),
        ("Use the source's term for word-overlap retrieval.", search[1], "search_methods.txt", "The source calls it keyword search."),
        ("Use the source's term for conceptually related passage retrieval.", search[0], "search_methods.txt", "The source calls it semantic search."),
        ("Use the source's term for combining the two search signals.", search[2], "search_methods.txt", "The source calls it hybrid search."),
        ("What exact system name is used for the vector database?", emb[2], "embeddings_and_chroma.txt", "The source names ChromaDB."),
        ("Preserve the acronym used for retrieval-augmented generation.", rag[0], "rag_basics.txt", "The acronym is RAG."),
    ]
    for item in terminology: add("terminology", *item)

    concise = [
        ("In one sentence, explain why embeddings help retrieval.", emb[0] + " " + emb[1], "embeddings_and_chroma.txt", "Embeddings represent semantic meaning as vectors, helping retrieval find related content despite wording differences."),
        ("In one sentence, explain hybrid search.", search[0] + " " + search[1] + " " + search[2], "search_methods.txt", "Hybrid search merges and re-ranks semantic vector evidence with keyword evidence."),
        ("In one sentence, explain the RAG handoff.", rag[0] + " " + rag[1], "rag_basics.txt", "RAG retrieves relevant document chunks and passes them as context to the language model."),
        ("In one sentence, explain the leave timing rule.", leave[1], "upload_test.txt", "Employees should request leave at least two weeks in advance."),
    ]
    for item in concise: add("concise_technical", *item)

    adversarial = [
        ("The context mentions a vector database; does that prove it stores employee leave records?", emb[2] + " " + leave[0], "embeddings_and_chroma.txt;upload_test.txt", REFUSAL),
        ("The context mentions external documents; does it state that every answer is correct?", rag[0] + " " + rag[1], "rag_basics.txt", REFUSAL),
        ("The context mentions BM25; does it say BM25 creates embeddings?", search[1] + " " + emb[0], "search_methods.txt;embeddings_and_chroma.txt", REFUSAL),
        ("The context mentions paid vacation; does it specify unlimited vacation?", leave[0], "upload_test.txt", REFUSAL),
        ("The context says query rewriting improves retrieval; does it give a numerical improvement?", rewrite[0], "query_rewriting_and_ollama.pdf", REFUSAL),
        ("The context describes semantic search; does it state that semantic search always beats keyword search?", search[0] + " " + search[1] + " " + search[2], "search_methods.txt", REFUSAL),
    ]
    for item in adversarial: add("adversarial", *item)
    return examples


def normalize_question(question: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", question.lower())).strip()


def source_files(source_field: str) -> list[str]:
    return [part.strip() for part in source_field.split(";") if part.strip()]


def evaluation_questions() -> list[tuple[str, str]]:
    if not EVALUATION_PATH.exists():
        return []
    cases = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
    return [(str(case.get("id", "")), str(case.get("question", ""))) for case in cases if case.get("question")]


def validate_dataset(train: list[dict[str, str]], validation: list[dict[str, str]], report: bool = True) -> dict[str, object]:
    all_examples = train + validation
    errors: list[str] = []
    for split_name, examples in (("train", train), ("validation", validation)):
        for index, example in enumerate(examples, 1):
            missing = REQUIRED_FIELDS - set(example)
            if missing:
                errors.append(f"{split_name}:{index} missing {sorted(missing)}")
            if example.get("category") not in VALID_CATEGORIES:
                errors.append(f"{split_name}:{index} invalid category")
            if not all(str(example.get(field, "")).strip() for field in REQUIRED_FIELDS):
                errors.append(f"{split_name}:{index} contains an empty field")
            for source in source_files(str(example.get("source", ""))):
                if source not in SOURCE_NAMES or not (DOCUMENTS_DIR / source).exists():
                    errors.append(f"{split_name}:{index} has invalid source {source!r}")
    train_questions = [normalize_question(item["input"]) for item in train]
    validation_questions = [normalize_question(item["input"]) for item in validation]
    if len(set(train_questions)) != len(train_questions):
        errors.append("duplicate questions inside training")
    if len(set(validation_questions)) != len(validation_questions):
        errors.append("duplicate questions inside validation")
    if set(train_questions) & set(validation_questions):
        errors.append("train/validation question overlap")

    near_duplicates: list[tuple[str, str, float]] = []
    for left_index, left in enumerate(all_examples):
        for right in all_examples[left_index + 1:]:
            score = difflib.SequenceMatcher(None, normalize_question(left["input"]), normalize_question(right["input"])).ratio()
            if score >= 0.82:
                near_duplicates.append((left["input"], right["input"], score))
    if near_duplicates:
        errors.append(f"{len(near_duplicates)} suspicious near-duplicate questions")

    embedded_evaluation: list[tuple[str, str, str]] = []
    evaluation_overlap: list[tuple[str, str, str]] = []
    for example in all_examples:
        normalized = normalize_question(example["input"])
        for evaluation_id, question in evaluation_questions():
            evaluation_normalized = normalize_question(question)
            score = difflib.SequenceMatcher(None, normalized, evaluation_normalized).ratio()
            if evaluation_normalized and evaluation_normalized in normalized:
                embedded_evaluation.append((example["input"], evaluation_id, question))
            elif score >= 0.78:
                evaluation_overlap.append((example["input"], evaluation_id, f"{score:.3f}"))
    if embedded_evaluation:
        errors.append(f"{len(embedded_evaluation)} embedded evaluation questions")
    if evaluation_overlap:
        errors.append(f"{len(evaluation_overlap)} suspicious evaluation near-overlaps")

    report_data = {"training_examples": len(train), "validation_examples": len(validation), "categories": {"training": dict(Counter(item["category"] for item in train)), "validation": dict(Counter(item["category"] for item in validation))}, "sources": dict(Counter(source for item in all_examples for source in source_files(item["source"]))), "evaluation_overlap": len(embedded_evaluation) + len(evaluation_overlap), "embedded_evaluation_questions": embedded_evaluation, "near_duplicates": near_duplicates, "errors": errors}
    if report:
        print_report(report_data)
    if errors:
        raise ValueError("Dataset validation failed:\n- " + "\n- ".join(errors))
    return report_data


def print_report(report: dict[str, object]) -> None:
    print(f"Training examples: {report['training_examples']}")
    print(f"Validation examples: {report['validation_examples']}")
    print("Training categories:", report["categories"]["training"])
    print("Validation categories:", report["categories"]["validation"])
    print("Sources:", report["sources"])
    print(f"Evaluation overlap: {report['evaluation_overlap']}")
    print("Train/validation overlap: 0")
    print(f"Suspicious near-duplicates: {len(report['near_duplicates'])}")
    print("Validation errors:", len(report["errors"]))


def load_jsonl(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as data_file:
        return [json.loads(line) for line in data_file if line.strip()]


def save_jsonl(examples: Iterable[dict[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as data_file:
        for example in examples:
            data_file.write(json.dumps(example, ensure_ascii=True) + "\n")


def split_examples(examples: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {category: [] for category in VALID_CATEGORIES}
    for example in examples:
        grouped[example["category"]].append(example)
    train: list[dict[str, str]] = []
    validation: list[dict[str, str]] = []
    for category in sorted(grouped):
        items = grouped[category]
        validation_count = max(1, round(len(items) * 0.2))
        validation_items = items[:: max(1, len(items) // validation_count)][:validation_count]
        validation_ids = {id(item) for item in validation_items}
        validation.extend(validation_items)
        train.extend(item for item in items if id(item) not in validation_ids)
    return train, validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true", help="validate existing JSONL files")
    args = parser.parse_args()
    if args.validate:
        validate_dataset(load_jsonl(TRAINING_DIR / "train.jsonl"), load_jsonl(TRAINING_DIR / "validation.jsonl"))
        return
    examples = build_examples()
    train, validation = split_examples(examples)
    validate_dataset(train, validation, report=False)
    save_jsonl(train, TRAINING_DIR / "train.jsonl")
    save_jsonl(validation, TRAINING_DIR / "validation.jsonl")
    validate_dataset(train, validation)


if __name__ == "__main__":
    main()
