# Reranking: Before and After

## What was added

A second retrieval stage, implemented in `reranker.py` and wired into `app.rank_chunks` and `retrieve_context`:

```
question -> hybrid search (semantic + BM25, relevance gate) -> up to 8 candidates
         -> cross-encoder reranking (ms-marco-MiniLM-L-6-v2) -> drop score < -6 -> top 4
         -> context expansion -> LLM
```

Hybrid search scores the question and each chunk separately, using embedding similarity and word statistics. A cross-encoder reads the question and the chunk together and outputs one relevance score. That is more precise, but too slow to run over a whole corpus, which is why it only reranks the shortlist. It adds about 20 to 30 ms per question on CPU.

## How it was evaluated

`evaluation/evaluate_reranking.py` runs each question twice, with reranking off and on. It scores the ranked chunk list against chunk-level labels in `evaluation/rerank_relevance_labels.json`: grade 2 means the chunk answers the question, grade 1 means it is useful supporting context, and anything else counts as irrelevant. Chunk-level labels were needed because the source-level metrics from the previous round were already at their ceiling: the right file was ranked first for every evaluation question.

Two question sets were used:

- **Evaluation cases** from `evaluation_questions.json`: 9 answerable (direct and follow-up) and 8 no-information.
- **Probe questions** from `retrieval_probe_questions.json`: 11 answerable (paraphrases and exact terms) and 3 out-of-scope.

Both sets are scored on retrieval directly, with the query classifier not involved. No LLM is called.

## Results

| Metric | Eval cases: before | Eval cases: after | Probes: before | Probes: after |
|---|---|---|---|---|
| Answer chunk ranked first | 9 / 9 | 9 / 9 | 10 / 11 | **11 / 11** |
| MRR (first answering chunk) | 1.00 | 1.00 | 0.955 | **1.00** |
| nDCG@4 | 0.913 | 0.913 | 0.925 | 0.928 |
| Precision (chunks with grade ≥ 1) | 1.00 | 1.00 | 0.71 | **0.94** |
| Irrelevant chunks returned | 0 | 0 | 9 | **2** |
| Answer keywords in final context | 100% | 100% | 100% | 100% |
| Mean context size (chars) | 178 | 178 | 209 | **146** |
| Unsupported questions rejected by retrieval | 7 / 8 | **8 / 8** | 1 / 3 | **2 / 3** |
| Mean retrieval latency (ms) | ~6 | ~23 | ~5 | ~33 |

The evaluation cases were already close to the ceiling, so reranking changes little there beyond ordering and one rejection. The gains show on the harder probe questions: rephrasings and exact terms where the first stage has to fall back on keyword matching and lets extra chunks through.

## Questions where the result changed

| Question | Before (ranked chunks) | After |
|---|---|---|
| How much notice do I need to give before a vacation? | "The leave policy allows… paid vacation", then "…request leave at least two weeks in advance" | The two-weeks sentence first and alone. It holds the answer. |
| What is BM25? / What is TF-IDF used for? / What is BM25 in Elasticsearch? | The keyword-search sentence plus the semantic-search and hybrid-search sentences | Only the keyword-search sentence |
| What does the system pass to the model? | The answer sentence plus two unrelated RAG sentences | Only the answer sentence |
| Explain the Python programming language. (out of scope) | 3 RAG chunks (matched the word "language") | Nothing retrieved, scored about −10.7 |
| How is semantic search different from keyword search? | Keyword sentence, then semantic sentence | Semantic sentence first, matching the question's subject |
| Which one uses embeddings? | Semantic-search sentence first | "ChromaDB … stores embeddings and supports semantic search" first. It is the most direct evidence. |

On the source-level metric in `evaluate_retrieval.py`, the last row lowers MRR from 1.0 to 0.944. That metric counts only `search_methods.txt` as correct for this question. The chunk-level labels grade the ChromaDB sentence as an answer chunk too.

## Threshold choice and sensitivity

Answer-bearing chunks scored −4.0 or higher. The chunks that slipped through for the out-of-scope Python question scored −10.7. Every cutoff from −4.5 to −8 gave the same answer@1 (100%) and the same rejection count. The cutoff was set to −6, in the middle of that range.

An extra rule was also tried: drop any chunk scoring more than N points below the best one. It removed useful supporting chunks, such as "request leave two weeks in advance" for "What is the leave policy?", and lowered nDCG to 0.905. It was left out.

| Cutoff | Max gap | Answer@1 | nDCG@4 | Precision | Irrelevant | Rejected |
|---|---|---|---|---|---|---|
| before (no reranking) | | 0.95 | 0.919 | 0.84 | 9 | 8/11 |
| −4.5 | none | 1.00 | 0.921 | 0.98 | 1 | 10/11 |
| **−6 (chosen)** | **none** | **1.00** | **0.921** | **0.97** | **2** | **10/11** |
| −8 | none | 1.00 | 0.921 | 0.93 | 4 | 10/11 |
| −6 | 8 | 1.00 | 0.905 | 0.97 | 2 | 10/11 |

(Both question sets combined.)

## Limitations

- **"What is the sick leave policy?"** is still answered from the leave-policy chunk (score +1.2). The cross-encoder sees "leave policy" as relevant and does not notice that "sick" is unsupported. This still needs an answer-side grounding check.
- **The labels were written by hand after looking at the corpus** (20 labelled questions over 11 chunks). With a corpus this small, treat the numbers as a sanity check rather than a benchmark. Re-run `evaluate_reranking.py --details` as documents are added.
- **Not tested on generated answers.** End-to-end answer quality with Ollama (`evaluate_rag.py`) was not run here. Reranking also changes the order of the context the LLM sees.
- **The classifier still blocks paraphrased questions before retrieval**, as noted in `RETRIEVAL_EVALUATION.md`.

## Reproducing

```bash
python evaluation/evaluate_reranking.py --details   # before/after comparison
python evaluation/evaluate_retrieval.py             # source-level metrics (reranking on)
python -m pytest tests
```

The numbers were produced with the same ms-marco-MiniLM-L-6-v2 model saved from your machine, and the same all-MiniLM-L6-v2 embeddings as your index. Latency was measured on a cloud CPU and will differ on your laptop.
