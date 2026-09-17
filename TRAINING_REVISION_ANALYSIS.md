# Training Pipeline Revision Analysis

**Date**: 2026-09-06  
**Status**: Design Phase (Before Implementation)  
**User Request**: Review and redesign training pipeline before execution

---

## PROBLEMS IDENTIFIED IN CURRENT IMPLEMENTATION

### 1. **Tokenization Alignment Issue (CRITICAL)**
**Problem**: In `SFTDataset.__getitem__()`, prompt and full sequence are tokenized separately:
```python
prompt_tokens = self.tokenizer(prompt_only, truncation=True, max_length=512, add_special_tokens=True)
full_tokens = self.tokenizer(prompt, truncation=True, max_length=512, padding="max_length", add_special_tokens=True)
prompt_length = len(prompt_tokens["input_ids"])
labels = [-100] * prompt_length + full_tokens["input_ids"][prompt_length:]
```

**Issues**:
- `prompt_tokens` doesn't use `padding="max_length"`, but `full_tokens` does
- Different tokenization calls with different padding strategies ≠ guaranteed alignment
- `add_special_tokens=True` might add `<bos>` tokens at different positions
- Truncation could happen at different points if sequences are long

**Result**: Labels might be misaligned, causing loss to be calculated on wrong tokens

**Fix**: Use a single-pass tokenization where we:
1. Tokenize the full prompt consistently
2. Find the exact byte position where the assistant response starts
3. Convert byte position to token position reliably
4. Calculate labels from token position (not from separate tokenization)

---

### 2. **Chat Template Application Issue**
**Problem**: Current code calls `apply_chat_template()` twice with different parameters:
```python
# Full prompt (with response)
prompt = self.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)

# Prompt only (without response)
prompt_only = self.tokenizer.apply_chat_template(
    prompt_only_messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

**Issues**:
- `add_generation_prompt=True` adds special tokens at the end (e.g., `<|im_start|>assistant\n`)
- `add_generation_prompt=False` does NOT add these tokens
- The strings are different lengths, so finding the "overlap point" is unreliable
- The overlap point might not exist if the template adds unexpected tokens

**Fix**: 
- Always use `add_generation_prompt=False` for both, then tokenize the final result
- Find the prompt boundary by using string position tracking or a cleaner method
- Use the tokenizer's `encode()` method consistently with known parameters

---

### 3. **Data Collator Issue**
**Problem**: Using `DataCollatorForLanguageModeling`:
```python
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Not masked language modeling
)
```

**Issues**:
- This collator was designed for casual LM, not SFT
- It might not handle custom `labels` column correctly
- Should verify it properly preserves our `-100` masking

**Fix**: Use a custom data collator or `DataCollatorForSeq2Seq` designed for SFT that:
- Handles `labels` column correctly
- Preserves `-100` masking
- Properly pads all sequences to the same length

---

### 4. **Hardware Detection Not Influencing Key Settings**
**Problem**: Hardware is detected but doesn't control:
- `TARGET_MODULES` (currently hardcoded `["q_proj", "v_proj"]`)
- Whether to use gradient checkpointing (not implemented)
- Flash attention v2 (not implemented)
- Warmup strategy (currently fixed at `WARMUP_RATIO = 0.1`)

**Fix**: 
- Auto-set more parameters based on detected VRAM
- Enable gradient checkpointing for <8GB systems
- Make all parameters user-overridable at the top of the file

---

### 5. **CPU Training Estimate Still Optimistic**
**Problem**: Says "30-60 minutes per epoch" on CPU:
```python
print(f"   Per epoch: {EPOCHS} * 30-60+ minutes = {EPOCHS * 30}-{EPOCHS * 60}+ minutes")
```

**Issues**:
- For 3B model on CPU: realistically 20-60+ minutes per epoch (often 30-60)
- Total becomes 60-180+ minutes for 3 epochs
- Doesn't account for CPU speed variation
- No mention of option to reduce sequence length on CPU

**Fix**:
- More realistic estimate: 30-90+ min per epoch on CPU
- Clarify this depends heavily on CPU model
- Suggest reducing MAX_SEQ_LENGTH if user wants faster training on CPU

---

### 6. **Version Compatibility Not Tested**
**Problem**: Code assumes transformers>=4.36.0 and peft>=0.7.0 are compatible:
```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
```

**Issues**:
- No import verification
- No fallback if older versions don't have certain features
- `DataCollatorForLanguageModeling` might have API changes across versions
- `prepare_model_for_kbit_training` import might fail with older peft

**Fix**:
- Add version checking at startup
- Provide clear error messages if versions are incompatible
- Document tested version ranges

---

### 7. **SFTDataset Doesn't Handle Edge Cases**
**Problem**: No validation for:
- Empty instruction/input/output fields
- Very long outputs that exceed max_seq_length
- Files that don't exist
- Malformed JSONL

**Fix**: Add robust error handling:
- Skip or warn on empty fields
- Truncate gracefully if output is too long
- Validate file format before loading

---

### 8. **No Validation of Data Separation**
**Problem**: Code claims data is separate from evaluation:
```python
print(f"  Data source: documents/ (NOT evaluation_questions.json)")
```

But there's no actual check that evaluation data isn't being used.

**Fix**:
- Add validation function that checks:
  - train.jsonl doesn't come from evaluation_questions.json
  - No keyword overlap between training outputs and evaluation expected answers
  - Print summary of data sources

---

### 9. **Loss Calculation Not Verified**
**Problem**: No verification that loss is actually being calculated correctly:
- Are `-100` labels being honored by the Trainer?
- Is loss only on response tokens?
- No logging of which tokens contribute to loss

**Fix**:
- Add a test function that:
  - Loads one batch
  - Checks that labels have -100 for prompt and real IDs for response
  - Prints sample token-to-label mapping
  - Runs this validation before training

---

## WHAT I WILL CHANGE

### **Phase 1: Fix Critical Tokenization Issue**
- Replace separate tokenization approach with unified tokenization
- Track prompt/response boundary using reliable method
- Add validation that labels align with token IDs
- Create test case to verify alignment

### **Phase 2: Improve SFTDataset Implementation**
- Use cleaner chat template application
- Add robust error handling
- Validate training data before training starts
- Add data source verification

### **Phase 3: Enhance Hardware Detection**
- Expand auto-configuration based on VRAM
- Enable gradient checkpointing for low-VRAM systems
- Add more detailed hardware reporting
- Make all settings visible and overridable

### **Phase 4: Improve Training Configuration**
- Replace DataCollatorForLanguageModeling with custom collator
- Add version compatibility checks
- More realistic CPU time estimates
- Better logging and progress reporting

### **Phase 5: Add Validation and Testing**
- Validate data separation from evaluation
- Test batch loading before training
- Verify label alignment
- Print sample training examples

### **Phase 6: Documentation Updates**
- Clarify exactly what's being trained
- Explain how loss is calculated
- Document all assumptions
- Provide troubleshooting guide

---

## HARDWARE REQUIREMENTS AFTER REVISION

| Hardware | Config | Time | Notes |
|----------|--------|------|-------|
| **RTX 3090 (24GB)** | LoRA, BS=2, GA=2 | 2-3 min | Best case |
| **RTX 4080 (16GB)** | LoRA, BS=2, GA=2 | 3-5 min | Good |
| **RTX 4070 (12GB)** | LoRA, BS=1, GA=4 | 5-8 min | Acceptable |
| **RTX 4060 (8GB)** | INT4 QLoRA, BS=1, GA=4 | 8-15 min | Works |
| **6GB VRAM** | INT4 QLoRA, BS=1, GA=8 | 15-30 min | Slow |
| **4GB VRAM** | INT4 QLoRA, BS=1, GA=8 | 30-60+ min | Very slow |
| **CPU (Intel i7)** | CPU, BS=1, GA=8 | 60-180+ min | Per epoch |

---

## PACKAGES REQUIRED

Already in `requirements-training.txt`:
```
transformers>=4.36.0       # Model loading/training
peft>=0.7.0                # LoRA/QLoRA
bitsandbytes>=0.41.0       # 4-bit quantization
accelerate>=0.24.0         # Training utilities
datasets>=2.14.0           # Dataset utilities
torch>=2.0.0               # Already installed
```

Will verify version compatibility and add version caps if needed.

---

## IMPLEMENTATION PLAN

### **Step 1**: Fix tokenization in `SFTDataset`
- Implement robust prompt/response boundary detection
- Add validation that labels match token positions
- Create test case that verifies alignment on sample data

### **Step 2**: Improve data handling
- Add error handling for malformed JSONL
- Validate data separation from evaluation
- Print data source verification

### **Step 3**: Enhance hardware configuration
- Expand auto-detection logic
- Add more granular VRAM-based settings
- Enable gradient checkpointing for low-VRAM

### **Step 4**: Fix training configuration
- Replace data collator with proper SFT version
- Add version checks with helpful error messages
- Improve time estimates

### **Step 5**: Add pre-training validation
- Test data loading with actual tokenizer
- Verify batch format before training
- Print sample batch for user review

### **Step 6**: Update documentation
- Explain exactly what's being trained
- Document all assumptions
- Provide troubleshooting

---

## SUCCESS CRITERIA

✅ After revision, the implementation should:
1. Have correct supervised fine-tuning (loss only on response tokens)
2. Work on systems with 4GB+ VRAM
3. Have realistic time/resource estimates
4. Maintain complete data separation from evaluation
5. Provide clear validation before training
6. Have comprehensive documentation
7. Pass version compatibility checks
8. Provide helpful error messages

---

## NO CHANGES TO:
- ✅ `app.py` (unchanged)
- ✅ `evaluation/` (unchanged)
- ✅ `requirements.txt` (unchanged)
- ✅ `evaluation_questions.json` (unchanged)
- ✅ Training data content (from documents/, not evaluation)

---

**Next Step**: Implementation of fixes following this plan
