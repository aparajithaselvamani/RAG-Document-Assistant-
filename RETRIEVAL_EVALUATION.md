# Retrieval Quality: Before and After

## Summary

Every expected source was already retrieved at rank 1. The problem was what came with it: 6 of the 9 answerable evaluation questions also sent an irrelevant chunk to the LLM, and only 65% of the context text came from the right document. After the changes, 1 question has an extra source, and in that case the extra chunk is useful supporting evidence. 93% of the context is relevant, the context is 58% shorter, and no answer became incomplete.

`evaluate_rag.py` could not see this, because it only checks that the expected source appears somewhere in the results. `evaluation/evaluate_retrieval.py` measures precision and completeness directly.

## Failing questions and why they failed

| Question (as retrieved) | Irrelevant chunk returned | Why it got through |
|---|---|---|
| What is Retrieval-Augmented Generation (RAG)? | `query_rewriting_and_ollama.pdf` ("Query rewriting improves retrieval.") | Shared the single word "retrieval" |
| What is semantic search? | `embeddings_and_chroma.txt` (whole file) | Its ChromaDB sentence mentions "semantic search" |
| What is hybrid search? | `embeddings_and_chroma.txt` | Shared the word "search" |
| How does query rewriting improve retrieval? | `rag_basics.txt` | Shared "retrieval" (inside "Retrieval-Augmented") |
| How is semantic search different from keyword search? | `embeddings_and_chroma.txt` | Shared "semantic", "search" |
| Which of semantic search and keyword search uses embeddings? | `rag_basics.txt` | Shared the word "and", which was missing from the stop-word list |

Root causes:

1. **Any shared word counted as keyword evidence.** In `hybrid_search`, a chunk with even one overlapping token skipped the semantic threshold completely. That included stop words ("and") and generic words ("search", "retrieval", "language"). The noisy chunks had cosine similarity around 0.1 to 0.2, far below the correct ones at around 0.8.
2. **Chunks were whole documents.** With `chunk_size=500`, every file (all under 400 characters) became a single chunk. `search_methods.txt` defines three different methods in one chunk, so any match on one of them sent all three definitions, and a single mention of "semantic search" inside the ChromaDB sentence pulled in the whole embeddings file.
3. **Index hygiene.** `upload_test.txt` was indexed twice, because uploading appended chunks without removing the old ones. Its text also started with a byte-order mark, and older chunks stored full Windows paths as their source.

## What changed

| File | Change |
|---|---|
| `ingest.py` | Sentence-aligned chunking: one idea per chunk, with back-reference sentences kept together. Text cleaning, `chunk_index` metadata, stable chunk ids, and re-uploads replace old chunks. |
| `keyword_search.py` | BM25 with a proper stop-word list and a light stemmer. This replaces the custom TF-IDF, which had hand-tuned per-word boosts. |
| `hybrid_search.py` | New gate: semantic floor plus a margin to the best match. Keyword rescue applies only when embeddings are unsure. Adds `expand_with_context`, which pulls in the preceding sentence(s) when a chunk says "both", "these", and so on. |
| `app.py` | `retrieve_context` calls `expand_with_context`. |
| `evaluation/evaluate_retrieval.py`, `retrieval_probe_questions.json` | Retrieval-only evaluation, plus 14 probe questions covering paraphrases, exact terms and out-of-scope questions. |
| `tests/test_rag.py` | 8 new tests. All 40 existing tests pass unchanged. |

## Results on the existing evaluation cases

The 9 answerable questions are the 6 direct and 3 follow-up cases. The 8 no-information cases are scored separately.

| Metric | Before | After |
|---|---|---|
| Expected source retrieved (hit rate) | 100% | 100% |
| MRR | 1.00 | 1.00 |
| Precision (passages from expected source) | 0.65 | **0.94** |
| Questions with an irrelevant chunk | 6 / 9 | **1 / 9** |
| Answer keywords present in retrieved text | 100% | 100% |
| Relevant share of context characters | 65% | **93%** |
| Mean context size (characters) | 424 | **178** |
| No-info questions rejected end to end | 8 / 8 | 8 / 8 |
| No-info questions rejected by retrieval alone (classifier bypassed) | 6 / 8 | 7 / 8 |

The remaining extra source is in `followup_02` ("Which one uses embeddings?"). The chunk "Embeddings convert text into vectors …" is supporting evidence here: `search_methods.txt` never mentions embeddings, so this chunk is what connects "embeddings" to "vector similarity".

## Probe questions (classifier bypassed)

| Metric | Before | After |
|---|---|---|
| Precision | 0.85 | **1.00** |
| Questions with an irrelevant chunk | 3 / 11 | **0 / 11** |
| Answer keywords present in retrieved text | 100% | 100% |
| Out-of-scope questions rejected | 1 / 3 | 1 / 3 |

## What each change contributes (ablation, evaluation cases)

| Variant | Precision | Noisy cases | Answer in context |
|---|---|---|---|
| Baseline | 0.65 | 6 | 100% |
| Sentence chunks only | 0.77 | 6 | 100% |
| Sentence chunks + BM25 and new gate, no expansion | 0.94 | 1 | **89%** |
| Full (with context expansion) | 0.94 | 1 | 100% |

Smaller chunks alone do not remove the noise, because the old gate still admitted any single-word overlap. The new gate removes the noise, but without expansion "What is hybrid search?" retrieves only "Hybrid search combines both approaches …", which never says which approaches. Context expansion fixes that incompleteness.

## Known limitations

- **The classifier blocks paraphrased questions before retrieval.** Probes such as "What is BM25?", "Can staff take paid time off?" and "Which search method matches exact words?" are classified GENERAL_OR_UNRELATED, because `classify_query` uses a fixed list of topic keywords. End to end, only 4 of the 11 answerable probes reach retrieval, before or after these changes. This is now the biggest recall bottleneck.
- **"What is the sick leave policy?"** retrieves the leave-policy chunk. It is semantically close, and retrieval cannot tell that "sick" is unsupported. That needs an answer-side grounding check. `has_sufficient_grounding` exists in `app.py` but is not called anywhere. Wiring it in as it stands would reject valid questions (for example, "far" in "How far in advance…" is not in the document).
- **"Explain the Python programming language."** reaches `rag_basics.txt` through the rare word "language" when the classifier is bypassed. End to end, the classifier rejects it.
- The thresholds (0.5 floor, 0.1 margin) were set on a very small corpus. The margins were clear (out-of-scope questions topped out at a cosine similarity of 0.45, relevant chunks admitted on semantics in the evaluation set started at 0.52), but re-check them with `evaluate_retrieval.py --details` as more documents are added.

## Reproducing

```bash
python ingest.py                                # rebuild the index with the new chunking
python evaluation/evaluate_retrieval.py --details
python evaluation/evaluate_rag.py               # end-to-end, needs Ollama running
python -m pytest tests
```

The numbers above were produced with the same all-MiniLM-L6-v2 weights (ONNX export). Its embeddings matched the vectors in the existing Chroma index exactly (cosine 1.000). End-to-end answer scores from `evaluate_rag.py` require Ollama and were not re-run.
