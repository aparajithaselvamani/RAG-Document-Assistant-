"""Train a Qwen 2.5 3B LoRA/QLoRA adapter without changing the base model."""

from __future__ import annotations

import importlib.util
import json
import difflib
import re
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, default_data_collator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

TRAINING_DIR = Path(__file__).resolve().parent
ROOT_DIR = TRAINING_DIR.parent
MODELS_DIR = ROOT_DIR / "models"
MODEL_ID = "Qwen/Qwen2.5-3B"
EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 768
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
USE_INT4_QUANTIZATION: bool | None = None
USE_INT8_QUANTIZATION = False


def detect_hardware() -> dict[str, Any]:
    """Report hardware and whether this host can realistically run QLoRA."""
    has_cuda = torch.cuda.is_available()
    result: dict[str, Any] = {
        "device": "cuda" if has_cuda else "cpu",
        "cuda": has_cuda,
        "gpu_name": None,
        "vram_gb": 0.0,
        "gpu_count": torch.cuda.device_count() if has_cuda else 0,
        "four_bit_supported": bool(has_cuda and importlib.util.find_spec("bitsandbytes")),
    }
    if has_cuda:
        props = torch.cuda.get_device_properties(0)
        result["gpu_name"] = props.name
        result["vram_gb"] = props.total_memory / (1024 ** 3)
    result["realistically_capable"] = bool(
        (has_cuda and result["vram_gb"] >= 6 and (result["four_bit_supported"] or result["vram_gb"] >= 12))
        or (not has_cuda and __import__("psutil").virtual_memory().total >= 32 * 1024 ** 3)
    )
    print("\nHardware")
    print(f"  Device: {result['device']}")
    print(f"  GPU: {result['gpu_name'] or 'none'}")
    print(f"  VRAM: {result['vram_gb']:.1f} GB")
    print(f"  4-bit QLoRA supported: {result['four_bit_supported']}")
    print(f"  Realistically capable: {result['realistically_capable']}")
    if not has_cuda:
        print("  Warning: CPU training is possible but may take many hours; no fixed time estimate is provided.")
    return result


def choose_config(hardware: dict[str, Any]) -> dict[str, Any]:
    """Choose conservative settings from detected hardware, with no fake timings."""
    if not hardware["cuda"]:
        return {"batch_size": 1, "gradient_accumulation": 16, "gradient_checkpointing": True, "max_length": 512, "quantization": None}
    vram = hardware["vram_gb"]
    use_int4 = USE_INT4_QUANTIZATION
    if use_int4 is None:
        use_int4 = hardware["four_bit_supported"] and vram < 16
    return {
        "batch_size": 1,
        "gradient_accumulation": 8 if vram < 12 else 4,
        "gradient_checkpointing": vram < 16,
        "max_length": MAX_SEQ_LENGTH,
        "quantization": "int4" if use_int4 else ("int8" if USE_INT8_QUANTIZATION else None),
    }


class SFTDataset(Dataset):
    """Tokenized chat examples with loss calculated only on assistant tokens."""

    def __init__(self, path: Path, tokenizer: Any, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: list[dict[str, str]] = []
        with path.open(encoding="utf-8") as data_file:
            for line_number, line in enumerate(data_file, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                required = {"instruction", "input", "context", "output", "source", "category"}
                missing = required - set(item)
                if missing or not all(str(item.get(key, "")).strip() for key in required):
                    raise ValueError(f"{path.name}:{line_number} is missing required fields: {sorted(missing)}")
                self.examples.append(item)
        if not self.examples:
            raise ValueError(f"No examples found in {path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        item = self.examples[index]
        user_content = f"Retrieved Context:\n{item['context']}\n\nQuestion:\n{item['input']}"
        prompt_messages = [{"role": "system", "content": item["instruction"]}, {"role": "user", "content": user_content}]
        full_messages = prompt_messages + [{"role": "assistant", "content": item["output"]}]
        prompt_ids = self.tokenizer.apply_chat_template(prompt_messages, tokenize=True, add_generation_prompt=True)
        full_ids = self.tokenizer.apply_chat_template(full_messages, tokenize=True, add_generation_prompt=False)[: self.max_length]
        prompt_length = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_length + full_ids[prompt_length:]
        padding = self.max_length - len(full_ids)
        full_ids += [self.tokenizer.pad_token_id] * padding
        labels += [-100] * padding
        return {"input_ids": full_ids, "attention_mask": [1] * (self.max_length - padding) + [0] * padding, "labels": labels}


def _normalize_question(question: str) -> str:
    """Normalize questions for exact, embedded, and fuzzy overlap checks."""
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", question.lower())).strip()


def validate_data_separation() -> None:
    """Reject malformed data and exact or suspicious evaluation overlap."""
    evaluation_path = ROOT_DIR / "evaluation" / "evaluation_questions.json"
    eval_questions = [
        (str(item.get("id", "")), _normalize_question(str(item.get("question", ""))))
        for item in json.loads(evaluation_path.read_text(encoding="utf-8"))
        if item.get("question")
    ] if evaluation_path.exists() else []
    examples: list[tuple[str, str, str]] = []
    for path in (TRAINING_DIR / "train.jsonl", TRAINING_DIR / "validation.jsonl"):
        with path.open(encoding="utf-8") as data_file:
            for line_number, line in enumerate(data_file, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                required = {"instruction", "input", "context", "output", "source", "category"}
                if required - set(item):
                    raise ValueError(f"{path.name}:{line_number} has an invalid schema")
                examples.append((path.name, item["input"], _normalize_question(item["input"])))

    duplicate_questions = {
        question for _, _, question in examples
        if sum(candidate == question for _, _, candidate in examples) > 1
    }
    if duplicate_questions:
        raise ValueError("Duplicate questions found across training and validation data")

    near_duplicates = []
    for index, (_, left_text, left) in enumerate(examples):
        for _, right_text, right in examples[index + 1:]:
            score = difflib.SequenceMatcher(None, left, right).ratio()
            if score >= 0.82:
                near_duplicates.append((left_text, right_text, score))
    if near_duplicates:
        raise ValueError(f"Suspicious near-duplicate training questions detected: {near_duplicates[:3]}")

    embedded = []
    fuzzy_overlap = []
    for _, training_text, training_question in examples:
        for evaluation_id, evaluation_question in eval_questions:
            score = difflib.SequenceMatcher(None, training_question, evaluation_question).ratio()
            if evaluation_question in training_question:
                embedded.append((training_text, evaluation_id))
            elif score >= 0.78:
                fuzzy_overlap.append((training_text, evaluation_id, score))
    if embedded:
        raise ValueError(f"Evaluation questions embedded in training inputs: {embedded[:3]}")
    if fuzzy_overlap:
        raise ValueError(f"Suspicious evaluation near-overlap detected: {fuzzy_overlap[:3]}")
    print(f"Data validation passed: {len(examples)} examples, no exact, embedded, or fuzzy evaluation overlap.")


def main() -> None:
    hardware = detect_hardware()
    config = choose_config(hardware)
    validate_data_separation()
    if not hardware["realistically_capable"]:
        print("Warning: this machine is not a recommended training host. Review the configuration before continuing.")
    quantization_config = None
    if config["quantization"] == "int4":
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    elif config["quantization"] == "int8":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    dtype = torch.bfloat16 if hardware["cuda"] and torch.cuda.is_bf16_supported() else (torch.float16 if hardware["cuda"] else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype, quantization_config=quantization_config, device_map="auto" if hardware["cuda"] else None)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if config["quantization"]:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM"))
    train_dataset = SFTDataset(TRAINING_DIR / "train.jsonl", tokenizer, config["max_length"])
    validation_dataset = SFTDataset(TRAINING_DIR / "validation.jsonl", tokenizer, config["max_length"])
    output_dir = MODELS_DIR / "qwen2.5-3b-lora"
    output_dir.mkdir(parents=True, exist_ok=True)
    argument_names = TrainingArguments.__dataclass_fields__
    evaluation_name = "eval_strategy" if "eval_strategy" in argument_names else "evaluation_strategy"
    arguments = dict(output_dir=str(output_dir), num_train_epochs=EPOCHS, per_device_train_batch_size=config["batch_size"], per_device_eval_batch_size=1, gradient_accumulation_steps=config["gradient_accumulation"], learning_rate=LEARNING_RATE, warmup_ratio=WARMUP_RATIO, weight_decay=WEIGHT_DECAY, logging_steps=1, save_strategy="epoch", save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False, fp16=hardware["cuda"] and dtype == torch.float16, bf16=hardware["cuda"] and dtype == torch.bfloat16, gradient_checkpointing=config["gradient_checkpointing"], remove_unused_columns=False, report_to="none", seed=42)
    arguments[evaluation_name] = "epoch"
    trainer = Trainer(model=model, args=TrainingArguments(**arguments), train_dataset=train_dataset, eval_dataset=validation_dataset, data_collator=default_data_collator)
    print(f"Ready to train {len(train_dataset)} examples with {len(validation_dataset)} validation examples.")
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "training_info.json").write_text(json.dumps({"base_model": MODEL_ID, "method": "response-only LoRA/QLoRA SFT", "quantization": config["quantization"], "hardware": hardware}, indent=2), encoding="utf-8")
    print(f"Adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
