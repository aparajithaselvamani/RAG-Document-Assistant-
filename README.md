# RAG Document Assistant

A complete Retrieval-Augmented Generation (RAG) document assistant built in Python. The application ingests PDF and TXT files, creates embeddings, stores them in a persistent Chroma vector database, rewrites user questions, performs semantic and keyword retrieval, combines them with hybrid ranking, and answers questions using the retrieved evidence.

## Project Overview

This project demonstrates a full document Q&A pipeline for local development and experimentation. It is designed to be easy to run on Windows, modular, and ready to extend with more advanced retrieval methods.

## Architecture Diagram

```text
Documents (PDF/TXT)
        |
        v
Document Loader -> Chunking -> Embeddings -> Chroma Vector DB
        |                                      |
        v                                      v
Query Rewriting -> Hybrid Retrieval -> LLM Answering
```

## Features

- Reads PDF and TXT files from the documents folder
- Splits documents into chunks using RecursiveCharacterTextSplitter
- Generates embeddings with sentence-transformers/all-MiniLM-L6-v2
- Persists embeddings in Chroma
- Rewrites user questions before retrieval
- Performs semantic, keyword, and hybrid retrieval
- Displays source attribution for every answer
- Falls back gracefully if Ollama is unavailable

## Installation

1. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure Ollama is installed and running locally if you want the best LLM-backed responses.

## How to Run

1. Build the vector database:

```bash
python ingest.py
```

2. Launch the assistant:

```bash
python app.py
```

3. Ask questions such as:

- What is RAG?
- What is hybrid search?
- What is query rewriting?
- What documents are indexed?

## Technologies Used

- Python 3.11
- LangChain
- ChromaDB
- sentence-transformers
- pypdf
- Ollama

## Example Queries

- What is retrieval augmented generation?
- Explain semantic search.
- Explain keyword search.
- What is hybrid search?
- What is query rewriting?

## Conversation Memory

The assistant now keeps a short-term memory of the most recent user/assistant exchanges while the app is running. It uses a bounded deque of up to five turns, so follow-up questions can be answered with context from the recent conversation without changing the retrieval pipeline.

Example:

- User: What is RAG?
- Assistant: RAG combines retrieval with generation.
- User: How does it work?

The second answer can use the recent exchange to interpret "it" more accurately while still relying on the same retrieved documents.

## Evaluation

The project includes a small repeatable evaluation suite to check grounded retrieval and answers against the documents currently indexed. It contains 12 cases: direct document questions, multi-turn follow-up questions that replay conversation history, and no-information questions that must be rejected rather than answered from general knowledge.

Run the evaluation after ingesting the documents and starting Ollama:

```bash
python evaluation/evaluate_rag.py
```

The runner uses the normal retrieval and answer-generation functions, then reports source retrieval, required answer-keyword coverage, and whether unsupported questions received the application's grounded fallback. Its summary is designed for demonstrations:

```text
RAG EVALUATION RESULTS
Total tests: 12
Passed: 10
Failed: 2
Pass rate: 83.3%
```

## Out-of-Scope Question Handling

Vector search can return nearby chunks even when the documents do not contain an answer. The assistant therefore treats hybrid retrieval as a candidate set, checks that the selected chunks cover the question's material terms (including named entities), and only then calls Ollama. If the evidence is insufficient, it returns:

```text
I could not find that information in the provided documents.
```

Conversation history can resolve references such as `it` or `which one`, but it is never used as factual evidence. The evaluation suite includes unsupported questions with no matches as well as questions with lexical or semantic neighbours, ensuring the assistant rejects unsupported answers instead of attributing unrelated sources.
