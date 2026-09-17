# Quick Start Guide: Fine-Tuning Your RAG Model

## ⚡ 5-Minute Overview

Fine-tuning adds a **separate, optional pipeline** that teaches Qwen 2.5 3B to be better at RAG tasks.

```
Before: User → RAG Pipeline → Base Qwen 2.5 3B (via Ollama) → Answer
After:  User → RAG Pipeline → Fine-Tuned Qwen 2.5 3B → Answer (better)
```

## 🚀 Quick Start (Step-by-Step)

### 1. Install Training Dependencies (5 min)
```bash
pip install -r requirements-training.txt
```

### 2. Generate Training Data (1 min)
```bash
python training/prepare_training_data.py
```
Creates: `training/train.jsonl` and `training/validation.jsonl`

### 3. Run Fine-Tuning (2-120 min depending on hardware)
```bash
python training/train.py
```

Outputs to: `models/qwen2.5-3b-lora/`

**Time Estimates:**
- RTX 3090/4080: ~2-5 min
- RTX 4090: ~30 sec
- CPU: 60-120 min

### 4. Test Fine-Tuned Model (1 min)
```bash
python training/test_finetuned_model.py
```

Interactive prompt. Type questions, see responses.

### 5. (Optional) Compare with Base Model
```bash
# Base model performance
python evaluation/evaluate_rag.py

# Fine-tuned model performance
python training/test_finetuned_model.py --batch
```

## ✅ What Doesn't Change

- ✅ Your existing `app.py` still works with Ollama
- ✅ All 40 tests still pass
- ✅ Evaluation still shows 27/27
- ✅ No modifications to documents or retrieval
- ✅ Can be deleted anytime (just delete `models/qwen2.5-3b-lora/`)

## 🎯 What You Get

**Better RAG responses:**
- More concise answers
- Better grounding (answers from context only)
- More consistent refusal of unsupported questions
- Better preservation of document terminology
- Better follow-up handling

## 📊 Status

- ✅ Tests: 40/40 passing
- ✅ Evaluation: 27/27 passing
- ✅ Training Data: Generated (16 train + 4 validation examples)
- ✅ No breaking changes

## 🔧 Troubleshooting

**"No module named peft"**
```bash
pip install -r requirements-training.txt
```

**"CUDA out of memory"**
Edit `training/train.py` and set:
```python
USE_INT4_QUANTIZATION = True  # Reduces memory by 50%
BATCH_SIZE = 4               # Optional: reduce batch size
```

**"No GPU detected"**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## 📚 Documentation

- `training/README.md` - Full documentation
- `FINETUNING_IMPLEMENTATION.md` - Implementation details
- `requirements-training.txt` - All dependencies

## 💡 Key Concepts

**Training Data vs. Evaluation Data:**
- Training: Teaches the model (from `documents/` folder)
- Evaluation: Tests the model (in `evaluation/evaluation_questions.json`)
- These are kept separate to prevent data leakage

**LoRA (Low-Rank Adaptation):**
- Only ~2-3M trainable parameters instead of 3B
- Faster training, less VRAM, easy to save/delete
- Base model never changes

**How It Works:**
```
Base Qwen Model (3B, frozen)
         ↓
    + LoRA Adapter (2-3M, trainable)
         ↓
    = Fine-Tuned Model (3B + adapter)
```

## 🎓 For Your Boss

"We added an optional fine-tuning capability that specializes the language model for RAG tasks. It uses LoRA to train only a small adapter (~0.1% of parameters), keeping training fast and resource-efficient. Training data comes from our project documents, separate from evaluation data, ensuring we test true generalization. The existing RAG pipeline is completely unchanged and can optionally use the fine-tuned model."

## 🔗 Integration

The fine-tuned model can be used in three ways:

1. **Keep Separate** (current)
   ```bash
   python app.py  # Uses base model via Ollama
   python training/test_finetuned_model.py  # Test fine-tuned separately
   ```

2. **Use Directly in Python** (replace Ollama)
   - See `training/README.md` for code changes

3. **Merge & Use with Ollama**
   - Requires GGUF conversion (see `training/README.md`)

---

**Questions?** See `training/README.md` or `FINETUNING_IMPLEMENTATION.md`
