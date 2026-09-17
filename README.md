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

## Query Classification

Before retrieval, the assistant classifies each question into one of three categories:

1. DOCUMENT_RELATED
2. FOLLOW_UP
3. GENERAL_OR_UNRELATED

The routing is intentionally explicit:

- DOCUMENT_RELATED → document retrieval → hybrid search → grounded answer
- FOLLOW_UP → conversation history + resolved question → document retrieval → grounded answer
- GENERAL_OR_UNRELATED → no document retrieval → grounded fallback response

This happens before any embedding or vector search for unrelated requests, so off-topic questions do not trigger unnecessary retrieval.

Example:

- User: What is RAG?
- Classification: DOCUMENT_RELATED
- User: How is it different from keyword search?
- Classification: FOLLOW_UP, using recent conversation history to resolve "it"
- User: What is the capital of France?
- Classification: GENERAL_OR_UNRELATED, returning the grounded fallback without retrieval

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

## Fine-Tuning

The project includes a complete fine-tuning pipeline to specialize the Qwen 2.5 3B language model for RAG tasks using LoRA (Low-Rank Adaptation).

### Overview

**Why separate fine-tuning from RAG?**

- **Training Data** teaches the model how to behave (from project documents)
- **Evaluation Data** tests the model's generalization (kept strictly separate)
- **RAG Documents** provide external knowledge at inference time

These three are intentionally kept distinct to prevent data leakage and test true performance.

**What the fine-tuning teaches:**

- How to answer questions grounded in provided context
- How to refuse unsupported questions correctly
- How to handle follow-up questions with conversation history
- How to be concise and factual
- When to use the "I could not find that information" fallback

**Method:**

- **LoRA (Low-Rank Adaptation):** Only ~2-3M trainable parameters instead of 3B
- **QLoRA:** Optional 4-bit quantization for low-VRAM systems
- **Data:** Generated from project documents (`documents/` folder)
- **Evaluation:** Separate dataset remains unused for training (`evaluation/evaluation_questions.json`)

### Quick Start

**1. Install fine-tuning dependencies:**

```bash
pip install -r requirements-training.txt
```

**2. Prepare training data from project documents:**

```bash
python training/prepare_training_data.py
```

This generates:
- `training/train.jsonl` (~80 examples)
- `training/validation.jsonl` (~20 examples)

**3. Run fine-tuning:**

```bash
python training/train.py
```

Outputs to: `models/qwen2.5-3b-lora/`

Estimated time:
- GPU (RTX 3090): ~2 minutes
- GPU (RTX 4080): ~1 minute
- CPU: 60-120 minutes

**4. Test the fine-tuned model:**

```bash
python training/test_finetuned_model.py
```

Loads the adapter and provides an interactive prompt.

### Architecture

```text
Original RAG Pipeline:
User → Classification → Routing → Retrieval → (Base Qwen 2.5 3B) → Answer + Sources

With Fine-Tuning:
User → Classification → Routing → Retrieval → (Fine-tuned Qwen 2.5 3B) → Answer + Sources
                                                      ↓
                                              Base Model (3B, frozen)
                                              + LoRA Adapter (2-3M, trained)
```

The fine-tuned model receives the same inputs as the base model but produces more RAG-optimized responses.

### Training Data Format

Training examples are in Qwen instruction/chat format:

```json
{
  "instruction": "You are a helpful RAG assistant. Answer using only provided context.",
  "input": "What is RAG?",
  "output": "Retrieval-Augmented Generation helps language models answer questions using external documents..."
}
```

Generated from:
- `documents/rag_basics.txt`
- `documents/embeddings_and_chroma.txt`
- `documents/search_methods.txt`
- `documents/upload_test.txt`
- `documents/query_rewriting_and_ollama.pdf`

Does NOT include evaluation data from `evaluation/evaluation_questions.json`.

### Configuration

Edit `training/train.py` to modify:

```python
EPOCHS = 3                  # Number of training passes
BATCH_SIZE = 8             # Gradient accumulation batch
LEARNING_RATE = 3e-4       # LoRA learning rate
MAX_SEQ_LENGTH = 512       # Context window
LORA_R = 8                 # LoRA rank (8 or 16)
USE_INT4_QUANTIZATION = False  # Set True for <8GB VRAM
```

### Hardware Requirements

- **GPU (Recommended):** 8GB+ VRAM (e.g., RTX 3090, RTX 4080)
  - With INT4 quantization: 4GB+
- **CPU:** 16GB+ RAM (very slow, 60-120 minutes)
- **Disk:** 10GB for model + adapter

The script automatically:
- Detects available CUDA
- Falls back to CPU with warnings
- Offers 4-bit/8-bit quantization options

### Integration with RAG

The fine-tuned model can be used as a drop-in replacement:

**Option 1: Keep using Ollama (base model)**

Keep current setup:
```bash
python app.py  # Uses base Qwen 2.5 3B via Ollama
```

Test fine-tuned model separately:
```bash
python training/test_finetuned_model.py  # Uses LoRA adapter
```

**Option 2: Merge adapter and create Ollama model**

(Requires additional tools for GGUF conversion)

```bash
# Merge LoRA into base model
python -c "
from peft import AutoPeftModelForCausalLM
model = AutoPeftModelForCausalLM.from_pretrained('models/qwen2.5-3b-lora')
model.save_pretrained('models/qwen2.5-3b-merged')
"

# Then create Ollama model from merged weights
# ollama create qwen-finetuned -f Modelfile
```

**Option 3: Replace inference in app.py**

Load fine-tuned model directly:
```python
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained("models/qwen2.5-3b-lora")
# Then use model.generate() instead of ollama.generate()
```

See `training/README.md` for detailed integration instructions.

### Evaluation

**Important Distinction:**

- `python evaluation/evaluate_rag.py` → Tests base model on evaluation questions
- `python training/test_finetuned_model.py` → Tests fine-tuned model interactively

The evaluation dataset is intentionally kept separate from training to test true generalization.

### Expected Results

**Before fine-tuning (base model):**
- May produce verbose answers
- May not understand RAG constraints fully
- Some variance in grounding

**After fine-tuning:**
- More concise, focused answers
- Better understanding of "answer from context only" constraint
- More consistent refusal of unsupported questions
- Better preservation of document terminology

### Troubleshooting

**"CUDA out of memory"**
- Set `USE_INT4_QUANTIZATION = True` in `train.py`
- Reduce `BATCH_SIZE` to 4 or 2
- Reduce `MAX_SEQ_LENGTH` to 256

**"No GPU detected"**
- Install CUDA PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
- Verify: `python -c "import torch; print(torch.cuda.is_available())"`

**Training loss not decreasing**
- Increase learning rate (3e-4 → 5e-4)
- Increase epochs (3 → 5)
- Check training data quality

### Further Reading

- See `training/README.md` for comprehensive documentation
- LoRA Paper: https://arxiv.org/abs/2106.09685
- QLoRA Paper: https://arxiv.org/abs/2305.14314
