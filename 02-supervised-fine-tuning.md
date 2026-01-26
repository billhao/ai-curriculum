# Step 2: Supervised Fine-Tuning (SFT)

Transform your base GPT-2 124M from a next-token predictor into an instruction-following assistant.

---

## What is SFT?

**Before SFT:** Model completes text naturally
```
Input:  "What is the capital of France?"
Output: "What is the capital of France? Is it Paris or Lyon? The answer may surprise you..."
```

**After SFT:** Model answers questions
```
Input:  "<|user|>What is the capital of France?<|assistant|>"
Output: "The capital of France is Paris."
```

SFT teaches the model the **format** of a conversation and how to **respond** rather than just continue.

---

## Prerequisites

```bash
source ~/projects/.venv/bin/activate

# Install TRL (Transformer Reinforcement Learning library)
uv pip install trl

# Install other dependencies
uv pip install torch transformers datasets accelerate peft bitsandbytes
```

---

## Step 1: Choose a Dataset

### Option A: Dolly 15k (Recommended for learning)
- 15,000 instruction-response pairs
- Clean, diverse, good quality
- Small enough to iterate quickly

### Option B: OpenAssistant (oasst1)
- 160k messages in conversation trees
- More complex, multi-turn conversations

### Option C: SlimOrca
- 500k high-quality instructions
- Use for production-quality fine-tuning

**Start with Dolly 15k**, then scale up.

---

## Step 2: Define Chat Template

Your model needs to learn a consistent format. Create `chat_template.py`:

```python
"""
Chat template for SFT training
"""

# Simple chat template for GPT-2
CHAT_TEMPLATE = """<|user|>
{instruction}
<|assistant|>
{response}<|endoftext|>"""

# For multi-turn (optional, more advanced)
MULTI_TURN_TEMPLATE = """<|system|>
You are a helpful assistant.
<|user|>
{instruction}
<|assistant|>
{response}<|endoftext|>"""

def format_dolly(example):
    """Format a Dolly dataset example for SFT"""
    instruction = example["instruction"]

    # Include context if available
    if example.get("context") and example["context"].strip():
        instruction = f"{instruction}\n\nContext: {example['context']}"

    response = example["response"]

    return {
        "text": CHAT_TEMPLATE.format(
            instruction=instruction,
            response=response
        )
    }

def format_oasst(example):
    """Format an OpenAssistant example for SFT"""
    # oasst1 has a tree structure, this is simplified
    return {
        "text": CHAT_TEMPLATE.format(
            instruction=example["instruction"],
            response=example["response"]
        )
    }
```

---

## Step 3: Prepare the Dataset

Create `prepare_dataset.py`:

```python
"""
Prepare Dolly 15k dataset for SFT
"""
from datasets import load_dataset
from chat_template import format_dolly

def prepare_dolly_dataset(tokenizer, max_length=512):
    """Load and prepare Dolly 15k for training"""

    # Load dataset
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    print(f"Loaded {len(dataset)} examples")

    # Format for chat
    dataset = dataset.map(format_dolly, remove_columns=dataset.column_names)

    # Split into train/val
    dataset = dataset.train_test_split(test_size=0.05, seed=42)

    print(f"Train: {len(dataset['train'])} examples")
    print(f"Val: {len(dataset['test'])} examples")

    # Preview a few examples
    print("\n" + "="*50)
    print("Example formatted data:")
    print("="*50)
    for i in range(3):
        print(f"\n--- Example {i+1} ---")
        print(dataset['train'][i]['text'][:500])
        print("...")

    return dataset

if __name__ == "__main__":
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-124m-hf")
    dataset = prepare_dolly_dataset(tokenizer)
```

Run it to preview:

```bash
python prepare_dataset.py
```

---

## Step 4: Training Script

Create `train_sft.py`:

```python
"""
SFT Training Script for GPT-2 124M
"""
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TrainingArguments,
)
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from chat_template import format_dolly

# ============================================
# Configuration
# ============================================

MODEL_PATH = "./gpt2-124m-hf"          # Your base model
OUTPUT_DIR = "./gpt2-124m-sft"          # Where to save
MAX_LENGTH = 512                         # Max sequence length
BATCH_SIZE = 8                           # Per device batch size
GRADIENT_ACCUMULATION = 4                # Effective batch = 8 * 4 = 32
LEARNING_RATE = 2e-5                     # Learning rate
NUM_EPOCHS = 3                           # Number of epochs
WARMUP_RATIO = 0.1                       # Warmup proportion

# ============================================
# Load Model and Tokenizer
# ============================================

print("Loading model and tokenizer...")
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)

# GPT-2 doesn't have a pad token by default
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# Add special tokens for chat
special_tokens = {
    "additional_special_tokens": ["<|user|>", "<|assistant|>", "<|system|>"]
}
tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))

print(f"Model parameters: {model.num_parameters():,}")

# ============================================
# Load and Prepare Dataset
# ============================================

print("Loading dataset...")
dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
dataset = dataset.map(format_dolly, remove_columns=dataset.column_names)
dataset = dataset.train_test_split(test_size=0.05, seed=42)

print(f"Train examples: {len(dataset['train'])}")
print(f"Val examples: {len(dataset['test'])}")

# ============================================
# Training Configuration
# ============================================

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,

    # Batch size settings
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,

    # Training duration
    num_train_epochs=NUM_EPOCHS,

    # Learning rate settings
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP_RATIO,

    # Optimization
    optim="adamw_torch",
    weight_decay=0.01,
    max_grad_norm=1.0,

    # Logging
    logging_steps=10,
    logging_first_step=True,
    report_to="none",  # Set to "wandb" if you use Weights & Biases

    # Evaluation
    eval_strategy="steps",
    eval_steps=200,

    # Saving
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,

    # SFT specific
    max_seq_length=MAX_LENGTH,
    packing=False,  # Set True for efficiency with short examples

    # Hardware (for H100s)
    bf16=True,  # Use bfloat16 on H100
    dataloader_num_workers=4,

    # Reproducibility
    seed=42,
)

# ============================================
# Initialize Trainer
# ============================================

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
)

# ============================================
# Train!
# ============================================

print("\n" + "="*50)
print("Starting SFT Training")
print("="*50)
print(f"Model: {MODEL_PATH}")
print(f"Dataset: Dolly 15k")
print(f"Batch size: {BATCH_SIZE} x {GRADIENT_ACCUMULATION} = {BATCH_SIZE * GRADIENT_ACCUMULATION}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Epochs: {NUM_EPOCHS}")
print("="*50 + "\n")

trainer.train()

# ============================================
# Save Final Model
# ============================================

print("Saving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nModel saved to {OUTPUT_DIR}")
print("SFT training complete!")
```

---

## Step 5: Run Training

```bash
# Single GPU
python train_sft.py

# Multi-GPU with accelerate (for your H100 cluster)
accelerate launch --num_processes 4 train_sft.py
```

**Expected training time:**
- Single H100: ~15-30 minutes for 3 epochs on Dolly 15k
- 4x H100: ~5-10 minutes

---

## Step 6: Test Your SFT Model

Create `test_sft.py`:

```python
"""
Test the SFT model with interactive chat
"""
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load SFT model
model_path = "./gpt2-124m-sft"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

def chat(user_input, max_new_tokens=150):
    """Generate a response to user input"""
    prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract just the assistant's response
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1]
    if "<|endoftext|>" in response:
        response = response.split("<|endoftext|>")[0]

    return response.strip()

# Test prompts
test_prompts = [
    "What is the capital of France?",
    "Explain photosynthesis in simple terms.",
    "Write a short poem about coding.",
    "What are three tips for learning a new language?",
    "How do I make a peanut butter and jelly sandwich?",
]

print("="*60)
print("Testing SFT Model")
print("="*60)

for prompt in test_prompts:
    print(f"\nUser: {prompt}")
    response = chat(prompt)
    print(f"Assistant: {response}")
    print("-"*60)

# Interactive mode
print("\n" + "="*60)
print("Interactive Chat (type 'quit' to exit)")
print("="*60)

while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() in ['quit', 'exit', 'q']:
        break
    if not user_input:
        continue

    response = chat(user_input)
    print(f"Assistant: {response}")
```

Run it:

```bash
python test_sft.py
```

---

## Step 7: Evaluate SFT Model

Compare against your baseline:

```bash
# Run the same benchmarks as before
lm_eval --model hf \
    --model_args pretrained=./checkpoints/gpt2-base-19072-hf \
    --tasks hellaswag,arc_easy,arc_challenge,piqa,winogrande,truthfulqa_mc2 \
    --batch_size 16 \
    --device mps \
    --output_path ./output/eval_results/base_model

# Measure perplexity
python measure_perplexity.py ./gpt2-124m-sft
```

**Expected changes after SFT:**
- Benchmark scores may stay similar or slightly decrease (model is now specialized)
- Perplexity on general text may increase slightly
- **Instruction-following ability dramatically improves** (qualitative)

---

## Step 8: Compare Base vs SFT

Create `compare_models.py`:

```python
"""
Side-by-side comparison of base vs SFT model
"""
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_model(path):
    model = GPT2LMHeadModel.from_pretrained(path)
    tokenizer = GPT2Tokenizer.from_pretrained(path)
    model.eval()
    return model, tokenizer

def generate(model, tokenizer, prompt, is_sft=False):
    device = next(model.parameters()).device

    if is_sft:
        formatted = f"<|user|>\n{prompt}\n<|assistant|>\n"
    else:
        formatted = prompt

    inputs = tokenizer(formatted, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Load both models
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading base model...")
base_model, base_tokenizer = load_model("./gpt2-124m-hf")
base_model = base_model.to(device)

print("Loading SFT model...")
sft_model, sft_tokenizer = load_model("./gpt2-124m-sft")
sft_model = sft_model.to(device)

# Compare on test prompts
test_prompts = [
    "What is machine learning?",
    "How do I cook pasta?",
    "Explain why the sky is blue.",
]

print("\n" + "="*70)
print("BASE MODEL vs SFT MODEL COMPARISON")
print("="*70)

for prompt in test_prompts:
    print(f"\n{'='*70}")
    print(f"PROMPT: {prompt}")
    print("="*70)

    print("\n[BASE MODEL]:")
    base_response = generate(base_model, base_tokenizer, prompt, is_sft=False)
    print(base_response)

    print("\n[SFT MODEL]:")
    sft_response = generate(sft_model, sft_tokenizer, prompt, is_sft=True)
    print(sft_response)
```

---
Write your previous answer about this into an empty file
## Troubleshooting

### Out of Memory
```python
# Reduce batch size
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 8  # Keep effective batch same

# Or enable gradient checkpointing
model.gradient_checkpointing_enable()
```

### Loss not decreasing
- Check your chat template formatting
- Verify special tokens are added correctly
- Try lower learning rate (1e-5)

### Model outputs garbage
- Make sure you're using the chat template at inference
- Check that special tokens were saved with the model

---

## Summary Checklist

- [ ] Install TRL and dependencies
- [ ] Create chat template
- [ ] Prepare Dolly 15k dataset
- [ ] Run SFT training (~15-30 min on H100)
- [ ] Test model interactively
- [ ] Run benchmarks and compare to baseline
- [ ] Side-by-side comparison with base model

---

## Next Step

Once your SFT model is working, proceed to **Step 3: Direct Preference Optimization (DPO)** to align the model with human preferences.

Your model now follows instructions. DPO will teach it to give *better* responses.
