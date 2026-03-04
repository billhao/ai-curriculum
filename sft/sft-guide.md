# Supervised Fine-Tuning (SFT): What You Built & What's Next

## 1. What You Built: SFT Implementation Summary

Your implementation transforms a pre-trained GPT-2 model into an instruction-following assistant. Here's the architecture:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SFT Pipeline Overview                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌─────────────────────────┐   │
│  │  Dolly-15k   │───▶│   Format     │───▶│  Tokenize + Add         │   │
│  │  Dataset     │    │   to Chat    │    │  Special Tokens         │   │
│  └──────────────┘    └──────────────┘    └───────────┬─────────────┘   │
│                                                      │                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────▼─────────────┐   │
│  │  Custom GPT  │───▶│  Convert to  │───▶│  Init New Token        │   │
│  │  Checkpoint  │    │  HF Format   │    │  Embeddings            │   │
│  └──────────────┘    └──────────────┘    └───────────┬─────────────┘   │
│                                                      │                  │
│                                          ┌──────────▼─────────────┐   │
│                                          │    TRL SFTTrainer      │   │
│                                          │    (Loss on Full Seq)  │   │
│                                          └───────────┬─────────────┘   │
│                                                      │                  │
│                                          ┌──────────▼─────────────┐   │
│                                          │  HF Model Checkpoint   │   │
│                                          │  (Ready for Inference) │   │
│                                          └─────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Components:**

| Component | File | Purpose |
|-----------|------|---------|
| Data Loading | `sft_load_data.py:26-52` | Loads Dolly-15k, formats to chat template |
| Chat Template | `sft_load_data.py:4-7` | `<\|user\|> {instruction} <\|assistant\|> {response}<\|endoftext\|>` |
| Model Conversion | `sft_model_utils.py:113-164` | Converts custom GPT to HuggingFace GPT2LMHeadModel |
| Token Init | `sft_model_utils.py:24-32` | Initializes `<\|user\|>` and `<\|assistant\|>` embeddings |
| Training | `sft_train.py:31-52` | TRL's SFTTrainer with HF TrainingArguments |

---

## 2. Key SFT Concepts Explained

### Why Special Tokens?

```
Without special tokens:          With special tokens:
┌─────────────────────────┐     ┌─────────────────────────┐
│ What is Python?         │     │ <|user|>                │
│ Python is a language... │     │ What is Python?         │
└─────────────────────────┘     │ <|assistant|>           │
                                │ Python is a language... │
Model can't distinguish         │ <|endoftext|>           │
speaker roles                   └─────────────────────────┘
                                Clear role boundaries
```

Your implementation adds these in `sft_model_utils.py:18-21`:
```python
special_tokens = {"additional_special_tokens": ["<|user|>", "<|assistant|>"]}
tknz.add_special_tokens(special_tokens)
```

### Why Mean Embedding Initialization?

When adding new tokens, you can't leave embeddings random. Your strategy (`sft_model_utils.py:24-32`):

```
Random Init:                     Mean Embedding Init:
┌─────────────────────────┐     ┌─────────────────────────┐
│ <|user|> = [-2.1, 0.8,  │     │ <|user|> = [0.02, 0.01, │
│            3.2, -1.5]   │     │            -0.01, 0.03] │
│                         │     │                         │
│ Far from existing tokens│     │ "Average" token - won't │
│ Unstable gradients      │     │ disrupt existing weights│
│ Slow convergence        │     │ Smooth fine-tuning      │
└─────────────────────────┘     └─────────────────────────┘
```

### Why 2e-5 Learning Rate?

```
Learning Rate Spectrum for Fine-Tuning:
│
│  Pre-training    SFT Sweet Spot    Too High
│     6e-4              2e-5           1e-3
│       │                │               │
├───────┼────────────────┼───────────────┼──────▶
│       │                │               │
│  Learning         Gentle         Catastrophic
│  from scratch     adaptation     forgetting
│                   of existing
│                   knowledge
```

Your choice (`sft_model_utils.py:62`): `learning_rate: float = 2e-5`

This is 30x smaller than pre-training because:
- Model already has useful representations
- We want to adapt behavior, not relearn language
- Too high → catastrophic forgetting of pre-trained knowledge

### Why Single Epoch?

```
Epochs vs Overfitting on Instruction Data:

Loss │
     │\
     │ \        Validation loss starts increasing
     │  ╲╲
     │   ╲ ╲______ Optimal stopping point (1 epoch)
     │    ╲      ╲
     │     ╲       ╲
     │      ╲        ╲___  Severe overfitting
     │                   ╲___ Memorizing Dolly examples
     └──────────────────────────────────────────▶ Epochs
        1     2     3     4     5
```

Instruction datasets like Dolly are:
- Small (15k examples)
- Diverse (many different tasks)
- High-quality (carefully curated)

Multiple epochs → model memorizes specific responses instead of learning patterns.

---

## 3. Scaling to NVIDIA GPU with Larger Datasets

### Recommended Datasets (Progressive Difficulty)

| Dataset | Size | Use Case | Notes |
|---------|------|----------|-------|
| Dolly-15k | 15K | Baseline | What you have now |
| Alpaca | 52K | Next step | GPT-4 generated, diverse |
| OpenAssistant | 161K | Multi-turn | Human conversations |
| ShareGPT | 90K | ChatGPT-like | High quality dialogues |
| FLAN | 1M+ | Task diversity | NLP benchmark mix |

### GPU Configuration Changes

Your current config (`sft_model_utils.py:48-49`):
```python
B: int = 1                    # micro batch size
T: int = 1024                 # sequence length
```

**Scaling for H100/A100:**

```python
# H100 80GB - Aggressive batching
B: int = 16                   # 16x larger micro batch
T: int = 1024                 # Keep sequence length
grad_accum_steps: int = 4     # Effective batch = 64

# A100 40GB - Balanced
B: int = 8
T: int = 1024
grad_accum_steps: int = 8     # Effective batch = 64

# Multi-GPU (4x H100)
# Your DDP code already handles this!
# torchrun --nproc_per_node=4 sft_train.py
# Effective batch = 16 * 4 * 4 = 256
```

**Memory Optimization Already in Place:**
- bfloat16 mixed precision (via TRL's `bf16=True`)
- DDP support (`MySFTConfig._init_device_and_ddp()`)

### Adding a Larger Dataset

```python
# In sft_load_data.py, replace Dolly with OpenAssistant:
from datasets import load_dataset

def load_dataset_openassistant(env):
    dataset = load_dataset("OpenAssistant/oasst1", split="train")

    # Filter to English, high-quality
    dataset = dataset.filter(lambda x: x["lang"] == "en")

    # Format to your chat template
    def format_oasst(example):
        return {
            "text": f"<|user|>\n{example['text']}\n<|assistant|>\n{example['response']}<|endoftext|>"
        }

    dataset = dataset.map(format_oasst)
    return dataset.train_test_split(test_size=0.05, seed=42)
```

---

## 4. Experiments to Try

### Experiment 1: Learning Rate Schedules

Modify `sft_model_utils.py` to compare:

```python
# Option A: Constant (current)
learning_rate: float = 2e-5

# Option B: Linear decay
# Add to training args:
lr_scheduler_type: str = "linear"

# Option C: Cosine decay
lr_scheduler_type: str = "cosine"
```

**What to observe:** Does decay help avoid overfitting on small datasets?

### Experiment 2: Multi-Epoch Training

```python
# In MySFTConfig
num_train_epochs: int = 3  # Change from 1

# Add early stopping callback:
from transformers import EarlyStoppingCallback

trainer = SFTTrainer(
    ...
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
```

**What to observe:** At which epoch does validation loss start increasing?

### Experiment 3: Response Masking Ablation

TRL's SFTTrainer computes loss on the full sequence. To add response-only masking:

```python
# In a custom training loop, create mask:
def create_response_mask(tokens, assistant_token_id):
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    for i, seq in enumerate(tokens):
        # Find <|assistant|> position
        positions = (seq == assistant_token_id).nonzero()
        if len(positions) > 0:
            mask[i, positions[0]:] = True
    return mask

# Apply mask to loss:
loss = loss * mask.float()
loss = loss.sum() / mask.sum()
```

**What to observe:** Does the model learn to follow instructions better, or does it perform worse because it sees fewer gradient signals?

### Experiment 4: Dataset Comparison

Train separate models on:
1. Dolly-15k (current)
2. Alpaca-52k
3. OpenAssistant

**Evaluation prompts:**
```python
test_prompts = [
    "Explain quantum computing to a 10-year-old",
    "Write a Python function to reverse a string",
    "What are the pros and cons of remote work?",
    "Translate 'Hello, how are you?' to Spanish"
]
```

Compare: Which dataset produces more helpful, accurate responses?

---

## 5. What's Next

### Path to More Capable Models

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     LLM Training Evolution                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│    Pre-training          SFT              RLHF/DPO                     │
│   (You did this)    (You did this)     (Next frontier)                 │
│        │                  │                   │                        │
│        ▼                  ▼                   ▼                        │
│   ┌─────────┐       ┌─────────┐        ┌─────────┐                    │
│   │ FineWeb │──────▶│ Dolly   │───────▶│ Human   │                    │
│   │ 10B tok │       │ 15k     │        │ Prefs   │                    │
│   └─────────┘       └─────────┘        └─────────┘                    │
│        │                 │                  │                          │
│   Learns            Learns to          Learns what                    │
│   language          follow             humans prefer                  │
│                     instructions                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### LoRA/QLoRA for Efficiency

Instead of updating all 124M parameters, train ~1% with Low-Rank Adaptation:

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                      # Rank of update matrices
    lora_alpha=32,             # Scaling factor
    target_modules=["c_attn", "c_proj"],  # Which layers to adapt
    lora_dropout=0.1,
)

model = get_peft_model(hf_model, lora_config)
# Trainable params: ~1.5M instead of 124M
```

**Benefits:**
- 10-100x faster training
- Fits larger models in GPU memory
- Can swap adapters for different tasks

### RLHF/DPO After SFT

After SFT, the model follows instructions but may not give the *best* responses. RLHF teaches preferences:

```python
# With TRL's DPO (Direct Preference Optimization)
from trl import DPOTrainer

# Preference data format:
# {"prompt": "...", "chosen": "good response", "rejected": "bad response"}

trainer = DPOTrainer(
    model=sft_model,
    ref_model=sft_model,  # Frozen copy
    train_dataset=preference_data,
    beta=0.1,  # KL penalty
)
```

### Evaluation with lm-eval-harness

Measure progress systematically:

```bash
# Install
pip install lm-eval

# Evaluate on standard benchmarks
lm_eval --model hf \
    --model_args pretrained=./output/sft/checkpoint-1140 \
    --tasks hellaswag,arc_easy,winogrande \
    --batch_size 8
```

---

## Quick Reference: Your Key Files

| Task | Command |
|------|---------|
| Run SFT | `python sft_train.py` |
| Sample from model | `python sft_sample.py` |
| Multi-GPU training | `torchrun --nproc_per_node=4 sft_train.py` |
| Evaluate | `python eval.py` |

Your SFT implementation is a solid foundation. The natural next steps are:
1. **Try larger datasets** (Alpaca, OpenAssistant)
2. **Add LoRA** for efficient fine-tuning
3. **Implement DPO** for preference learning
4. **Set up evaluation** with lm-eval-harness
