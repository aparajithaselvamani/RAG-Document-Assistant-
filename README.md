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
