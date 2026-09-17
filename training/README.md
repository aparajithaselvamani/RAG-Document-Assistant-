# Fine-Tuning Pipeline

This directory contains the fine-tuning infrastructure for the Qwen 2.5 3B language model.

## Overview

The fine-tuning pipeline teaches the base Qwen 2.5 3B model to be a better RAG assistant by learning:

- How to answer questions grounded in provided context
- How to refuse to answer when context is insufficient
- How to handle follow-up questions with conversation history
- How to provide concise, factual answers
- How to distinguish between supported and unsupported claims

**Important Separation:**
- **Training data** (`train.jsonl`, `validation.jsonl`) → teaches the model
- **Evaluation data** (`../evaluation/evaluation_questions.json`) → tests the model
- **RAG documents** (`../documents/`) → provide external knowledge at inference time

These three are kept strictly separate.

## Files

- `prepare_training_data.py` - Generates training data from project documents
- `train.py` - Fine-tuning script using LoRA/QLoRA
- `test_finetuned_model.py` - Interactive testing of fine-tuned adapter
- `train.jsonl` - Training dataset (generated)
- `validation.jsonl` - Validation dataset (generated)

## Hardware Requirements

- **GPU strongly recommended:** QLoRA requires CUDA plus a compatible bitsandbytes build.
   - 6-8GB VRAM: 4-bit QLoRA with batch size 1 and gradient accumulation.
   - 12GB+ VRAM: 4-bit or standard LoRA depending on configuration.
   - CPU training is technically possible but may take many hours; no fixed runtime is promised.
  
- **RAM:** 16GB+ recommended for smooth training
- **Disk:** ~10GB for model + adapter weights

**CUDA PyTorch requirement:** `requirements-training.txt` does not install
PyTorch. On the NVIDIA training machine, install a CUDA-enabled PyTorch build
separately using the command recommended at https://pytorch.org for the
machine's CUDA version. Verify it before training:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

The verification must report `True` for CUDA availability. A CPU-only PyTorch
installation will not enable QLoRA.

Detect available GPUs: The `train.py` script will automatically detect and use CUDA if available.

## Quick Start

1. Install training dependencies:
   ```bash
   pip install -r requirements-training.txt
   ```

2. Prepare training data:
   ```bash
   python training/prepare_training_data.py
   ```

3. Run fine-tuning:
   ```bash
   python training/train.py
   ```
   - Default: 3 epochs, per-device batch size 1, gradient accumulation 8, learning rate 2e-4
   - Outputs to: `models/qwen2.5-3b-lora/`

4. Test the fine-tuned adapter:
   ```bash
   python training/test_finetuned_model.py
   ```

5. Integration with RAG (optional):
   - See "Integration with RAG" section below

## Training Data

The training data is generated from project documents using instruction/chat format compatible with Qwen 2.5.

### Format

```json
{
  "instruction": "Answer the question about RAG using only the provided context.",
   "input": "What does retrieval-augmented generation do?",
   "context": "RAG helps language models answer questions using external documents...",
   "output": "RAG helps language models answer questions using external documents...",
   "source": "rag_basics.txt"
}
```

### Content

Training examples teach the model to:
- Answer document-related questions accurately
- Handle follow-up questions using conversation context
- Refuse unsupported questions with the grounded fallback
- Preserve information from source documents
- Stay concise and factual

Generated from:
- `../documents/rag_basics.txt`
- `../documents/embeddings_and_chroma.txt`
- `../documents/search_methods.txt`
- `../documents/upload_test.txt`

### Regenerate Training Data

If you modify documents or want to change training examples:

```bash
python training/prepare_training_data.py
```

This will regenerate `train.jsonl` and `validation.jsonl`.

## Fine-Tuning

### Configuration

Modify `train.py` to adjust:

```python
EPOCHS = 3                    # Number of training epochs
BATCH_SIZE = 8               # Gradient accumulation batch size
LEARNING_RATE = 3e-4         # LoRA learning rate
MAX_SEQ_LENGTH = 512         # Tokenizer max length
LORA_R = 8                   # LoRA rank (8 or 16 common)
LORA_ALPHA = 16              # LoRA alpha (usually 2*R)
LORA_DROPOUT = 0.05          # Dropout for LoRA layers
TARGET_MODULES = ["q_proj", "v_proj"]  # Which modules to apply LoRA
```

### Hardware Detection

The script automatically:
- Detects available CUDA devices
- Falls back to CPU with warnings if needed
- Reports training time estimates
- Adjusts batch size if VRAM is low (use `int4_quantization=True`)

### Training Output

```
models/
├── qwen2.5-3b-lora/
│   ├── adapter_config.json       # LoRA configuration
│   ├── adapter_model.bin         # LoRA weights
│   ├── training_args.bin
│   ├── optimizer.pt
│   └── trainer_state.json
```

The base model is NOT modified. Only the small adapter is saved.

## Testing

### Interactive Testing

```bash
python training/test_finetuned_model.py
```

Prompts you to enter questions. Loads:
- Base model: Qwen 2.5 3B (from HuggingFace)
- Adapter: `models/qwen2.5-3b-lora/`

Example:
```
Enter prompt (or 'exit' to quit): What is RAG?

[Fine-tuned model response...]

Enter prompt (or 'exit' to quit): exit
```

### Evaluation

To compare base vs. fine-tuned models on evaluation questions:

```bash
# Test base model on evaluation set
python evaluation/evaluate_rag.py

# Test fine-tuned model on evaluation set  
python evaluation/evaluate_finetuned.py
```

**Important:** The evaluation dataset is NOT used for training. Evaluation tests whether the model generalizes beyond its training data.

## Integration with RAG

The fine-tuned model can be used as a drop-in replacement for the base Qwen 2.5 3B model.

### Option 1: Merge and Use with Ollama (Recommended)

Merge the LoRA adapter into the base model, then use with Ollama:

```python
# (Example script - not yet implemented)
from peft import AutoPeftModelForCausalLM

# Load merged model
model = AutoPeftModelForCausalLM.from_pretrained(
    "models/qwen2.5-3b-lora"
)

# Save for Ollama (GGUF format)
model.save_pretrained("models/qwen2.5-3b-merged")
# Then: ollama create qwen-finetuned -f Modelfile
```

### Option 2: Use Directly in Python

Replace Ollama calls with direct HuggingFace inference:

```python
# In app.py, generate_answer():
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model = AutoPeftModelForCausalLM.from_pretrained("models/qwen2.5-3b-lora")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=512)
answer = tokenizer.decode(outputs[0])
```

### Option 3: Keep Ollama for Base Model

Keep Ollama for base inference and test fine-tuned separately:

```bash
# Keep using existing RAG with base model
python app.py

# Test fine-tuned model independently
python training/test_finetuned_model.py
```

This approach preserves the current RAG pipeline while allowing comparison.

## Troubleshooting

### "CUDA out of memory"
- Set `int4_quantization=True` in `train.py`
- Reduce `BATCH_SIZE` to 4 or 2
- Reduce `MAX_SEQ_LENGTH` to 256

### "No GPU detected, using CPU" (slow)
- Ensure CUDA is installed: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
- Check PyTorch: `python -c "import torch; print(torch.cuda.is_available())"`

### Training loss not decreasing
- Increase learning rate slightly (3e-4 → 5e-4)
- Increase number of epochs (3 → 5)
- Check training data quality with `prepare_training_data.py --validate`

### Model generates repeated text
- Reduce LoRA rank (16 → 8)
- Increase dropout (0.05 → 0.1)
- Check that training examples aren't duplicated

## Evaluation Metrics

### During Training

- **Training Loss:** Should steadily decrease
- **Validation Loss:** Should track training loss (overfitting if diverges)
- **Perplexity:** Should decrease as model learns

### After Training

Compare on evaluation questions:
- **Grounding:** Does it answer only from context?
- **Refusal:** Does it correctly refuse unsupported questions?
- **Accuracy:** Do answers match expected keywords?
- **Conciseness:** Are answers appropriately brief?

## FAQ

**Q: Will fine-tuning improve RAG evaluation scores?**

A: Yes, the model should learn to be more concise and grounded, potentially improving keyword coverage and refusal rates. However, RAG evaluation tests retrieval and grounding, not fine-tuning alone.

**Q: Can I use this with the existing Ollama setup?**

A: Yes, but requires merging the adapter with the base model and converting to GGUF format for Ollama. See "Integration with RAG" section.

**Q: How much training data do I need?**

A: For LoRA, even 100-200 examples can improve the model. Current dataset has ~80-100 examples from project documents.

**Q: Will fine-tuning make the model worse?**

A: Not if training data is clean and representative. Our training data comes directly from project documents, so the model learns expected behavior.

**Q: Can I fine-tune for domain-specific tasks?**

A: Yes. Modify `prepare_training_data.py` to create examples for your specific use cases.

## Next Steps

1. Install dependencies
2. Prepare training data
3. Run training
4. Test with `test_finetuned_model.py`
5. Compare evaluation results (base vs. fine-tuned)
6. Optionally integrate with RAG pipeline

---

**Key Principle:** Training and evaluation are separate. Training teaches the model. Evaluation measures generalization.
