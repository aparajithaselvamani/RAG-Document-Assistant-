"""Evaluate the trained adapter through the unchanged RAG evaluator.

The evaluation JSON is read only by evaluation/evaluate_rag.py after answers are
 generated. It is never used to build training examples.
"""

from __future__ import annotations

from collections import deque

import app
from evaluation import evaluate_rag
from training.test_finetuned_model import generate_response, load_model_and_tokenizer


def main() -> int:
    model, tokenizer = load_model_and_tokenizer()

    def fine_tuned_answer(question, context, conversation_history=None):
        history = conversation_history or deque(maxlen=app.MAX_CONVERSATION_TURNS)
        resolved_question = app.resolve_follow_up_question(question, history)
        context_text = "\n\n".join(document.page_content for document in context)
        return generate_response(model, tokenizer, resolved_question, context_text)

    original_generate_answer = app.generate_answer
    app.generate_answer = fine_tuned_answer
    try:
        return evaluate_rag.main()
    finally:
        app.generate_answer = original_generate_answer


if __name__ == "__main__":
    raise SystemExit(main())
