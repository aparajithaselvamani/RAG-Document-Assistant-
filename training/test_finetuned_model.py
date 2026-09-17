"""Test the fine-tuned Qwen 2.5 3B model with LoRA adapter.

This script loads:
1. The base Qwen 2.5 3B model
2. The trained LoRA adapter
3. Allows interactive testing of the fine-tuned model

The base model is NOT modified. The adapter is loaded on top.
"""

import sys
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

TRAINING_DIR = Path(__file__).resolve().parent
ROOT_DIR = TRAINING_DIR.parent
MODELS_DIR = ROOT_DIR / "models"

ADAPTER_DIR = MODELS_DIR / "qwen2.5-3b-lora"
BASE_MODEL_ID = "Qwen/Qwen2.5-3B"

# Generation parameters
MAX_LENGTH = 512
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 40


def load_model_and_tokenizer():
    """Load the fine-tuned model with LoRA adapter and tokenizer."""
    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(
            f"Fine-tuned model not found at {ADAPTER_DIR}\n"
            f"Please run: python training/train.py"
        )
    
    print("Loading fine-tuned model...")
    
    # Load model with adapter
    model = AutoPeftModelForCausalLM.from_pretrained(
        ADAPTER_DIR,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✓ Model loaded successfully")
    print(f"  Base model: {BASE_MODEL_ID}")
    print(f"  Adapter: {ADAPTER_DIR}")
    
    return model, tokenizer


def generate_response(model, tokenizer, question: str, context: str = "") -> str:
    """Generate a response from a question and retrieved context."""
    messages = [
        {
            "role": "system",
            "content": "Answer the question using only the retrieved context. Be concise. If the context is insufficient, say exactly: I could not find that information in the provided documents.",
        },
        {
            "role": "user",
            "content": f"Retrieved Context:\n{context}\n\nQuestion:\n{question}",
        },
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    inputs = {"input_ids": inputs, "attention_mask": torch.ones_like(inputs)}
    
    # Move to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_tokens = outputs[0, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def main():
    """Interactive testing loop."""
    print("=" * 80)
    print("Fine-tuned Qwen 2.5 3B Model Tester")
    print("=" * 80)
    print()
    
    try:
        model, tokenizer = load_model_and_tokenizer()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    print("\nInteractive testing mode. Enter prompts to test the model.")
    print("Type 'exit' to quit.\n")
    
    while True:
        try:
            user_input = input("Enter question (or 'exit' to quit): ").strip()
            
            if user_input.lower() == "exit":
                print("Goodbye!")
                break
            
            if not user_input:
                print("Please enter a prompt.\n")
                continue
            
            context = input("Paste retrieved context (blank for no context): ").strip()
            print("\nGenerating response...")
            response = generate_response(model, tokenizer, user_input, context)
            print(f"\nResponse:\n{response}\n")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")
            continue


def test_specific_prompts():
    """Test the model on a set of predefined prompts."""
    print("=" * 80)
    print("Fine-tuned Qwen 2.5 3B Model Tester - Batch Mode")
    print("=" * 80)
    print()
    
    try:
        model, tokenizer = load_model_and_tokenizer()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    test_prompts = [
        ("What does RAG do?", "RAG helps language models answer questions using external documents."),
        ("How does semantic search work?", "Semantic search uses vector similarity to find conceptually related passages."),
        ("What is the capital of France?", "RAG helps language models answer questions using external documents."),
    ]
    
    print("Testing on predefined prompts...\n")
    
    for i, (question, context) in enumerate(test_prompts, 1):
        print(f"{i}. Question: {question}")
        try:
            response = generate_response(model, tokenizer, question, context)
            print(f"   Response: {response}\n")
        except Exception as e:
            print(f"   ❌ Error: {e}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        test_specific_prompts()
    else:
        main()
