# SFT Training Explained

This document explains how Supervised Fine-Tuning (SFT) works using HuggingFace's `SFTTrainer` from the `trl` library.

## Table of Contents
1. [Overview](#overview)
2. [How Training Steps Are Calculated](#how-training-steps-are-calculated)
3. [What Happens in trainer.train()](#what-happens-in-trainertrain)
4. [Input and Labels: The Shifting Trick](#input-and-labels-the-shifting-trick)
5. [Gradient Accumulation (Not Loss Accumulation)](#gradient-accumulation-not-loss-accumulation)
6. [Packing: Efficient Training](#packing-efficient-training)

---

## Overview

SFT fine-tunes a pretrained language model on instruction-response pairs. The model learns to generate appropriate responses given user instructions.

Example training data format:
```
<|user|>
What is the capital of France?
<|assistant|>
The capital of France is Paris.<|endoftext|>
```

---

## How Training Steps Are Calculated

SFTTrainer calculates total training steps as:

```
total_steps = ceil(num_train_samples / effective_batch_size) * num_epochs
```

Where:
```
effective_batch_size = per_device_batch_size * num_devices * gradient_accumulation_steps
```

### Example with Dolly-15k

| Parameter | Value |
|-----------|-------|
| Train samples | ~14,250 (95% of 15k) |
| per_device_batch_size | 1 |
| gradient_accumulation_steps | 64 |
| num_epochs | 1 |

Calculation:
```
effective_batch_size = 1 * 1 * 64 = 64
steps_per_epoch = ceil(14250 / 64) = 223
total_steps = 223 * 1 = 223
```

Note: With `packing=True`, the actual number of steps depends on how examples are packed together.

---

## What Happens in trainer.train()

### Phase 1: Setup

#### 1.1 Data Preparation
```
Raw Dataset: List of {"text": "<|user|>...<|assistant|>...<|endoftext|>"}
     ↓
Tokenization: Each text → token IDs (up to max_length)
     ↓
Packing (if enabled): Multiple short examples → single 1024-token sequences
     ↓
DataLoader: Yields batches of {input_ids, attention_mask, labels}
```

#### 1.2 Optimizer & Scheduler
```python
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
scheduler = linear_warmup + linear_decay
```

### Phase 2: Training Loop

```python
for epoch in range(num_train_epochs):
    model.train()

    for step, batch in enumerate(train_dataloader):
        # batch = {
        #   "input_ids": [batch_size, seq_len],
        #   "attention_mask": [batch_size, seq_len],
        #   "labels": [batch_size, seq_len]
        # }

        # ──── Forward Pass ────
        with torch.autocast(device_type, dtype=torch.bfloat16):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss / gradient_accumulation_steps

        # ──── Backward Pass ────
        loss.backward()  # Gradients accumulate in param.grad

        # ──── Optimizer Step (every N micro-batches) ────
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # Logging, evaluation, checkpointing...
```

### Phase 3: Evaluation & Checkpointing

- **Logging**: Every `logging_steps` (e.g., 10 steps)
- **Evaluation**: Every `eval_steps` (e.g., 200 steps) - compute loss on validation set
- **Saving**: Every `save_steps` (e.g., 500 steps) - save model checkpoint

---

## Input and Labels: The Shifting Trick

In causal language modeling, **input_ids and labels are the same tensor**.

The key insight: the model internally shifts the sequences to create the prediction task.

### What You Pass

```python
input_ids = [Hello, world, I, am, GPT, <eos>]
labels    = [Hello, world, I, am, GPT, <eos>]  # Same!
```

### What the Model Does Internally

Inside `GPT2LMHeadModel.forward()`:

```python
# Model predicts next token at each position
logits = model(input_ids)  # Shape: [batch, seq_len, vocab_size]

# Shift for loss computation
shift_logits = logits[..., :-1, :]    # Positions 0 to N-1
shift_labels = labels[..., 1:]         # Positions 1 to N

loss = CrossEntropyLoss(shift_logits, shift_labels)
```

### Visual Explanation

```
Position:      0        1        2       3       4        5
Tokens:     [Hello]  [world]   [I]    [am]   [GPT]   [<eos>]
               ↓        ↓        ↓       ↓       ↓
Model       [world]   [I]     [am]   [GPT]  [<eos>]  [???]
Predicts:      ↓        ↓        ↓       ↓       ↓
Labels:     [world]   [I]     [am]   [GPT]  [<eos>]

Loss = CrossEntropy(predictions[0:5], labels[0:5])
```

The model learns: **given all tokens before position N, predict the token at position N**.

---

## Gradient Accumulation (Not Loss Accumulation)

A common misconception: "gradient accumulation sums up loss values."

**Wrong.** Gradients accumulate, not loss values.

### How It Actually Works

```python
optimizer.zero_grad()  # Reset all param.grad to 0

for i in range(gradient_accumulation_steps):  # e.g., 64 iterations
    outputs = model(**batch)
    loss = outputs.loss / gradient_accumulation_steps  # Scale the loss
    loss.backward()  # Gradients ADD to param.grad (+=, not =)

# After 64 backward passes:
# param.grad = grad_1 + grad_2 + ... + grad_64 (all scaled by 1/64)

optimizer.step()  # Update weights using accumulated gradients
```

### The Key Insight

When you call `loss.backward()`, PyTorch does:
```python
param.grad += d_loss / d_param  # Accumulates, doesn't replace!
```

Gradients keep adding up until you call `optimizer.zero_grad()`.

### Why Divide Loss by N?

```python
loss = loss / gradient_accumulation_steps
```

This ensures the accumulated gradients equal what you'd get from one large batch:

```
# These are mathematically equivalent:
grad(batch_of_64) ≈ (grad(batch_1) + grad(batch_2) + ... + grad(batch_64)) / 64
```

### What About the Returned Loss?

The loss value returned from the training step is **only for logging**. The actual training uses the gradients stored in `param.grad`, not the loss value.

```python
def training_step(self, model, inputs):
    loss = self.compute_loss(model, inputs)
    loss = loss / self.args.gradient_accumulation_steps
    loss.backward()  # This is what matters for training
    return loss.detach()  # This is just for logging
```

---

## Packing: Efficient Training

When `packing=True`, SFTTrainer concatenates multiple short examples into single sequences.

### Without Packing

```
Sequence 1: [Hello, how, are, you, <pad>, <pad>, <pad>, ...]  (mostly padding)
Sequence 2: [What, is, AI, <pad>, <pad>, <pad>, <pad>, ...]   (mostly padding)
```

Wasted computation on padding tokens.

### With Packing

```
Sequence 1: [Hello, how, are, you, <sep>, What, is, AI, <sep>, Tell, me, ...]
```

Multiple examples packed into one sequence, minimal padding.

### How Attention Is Handled

The attention mask is modified so examples don't attend to each other:

```
Example 1 tokens: [A, B, C]
Example 2 tokens: [D, E]
Packed: [A, B, C, D, E]

Attention mask (1 = can attend, 0 = cannot):
    A  B  C  D  E
A [[1, 1, 1, 0, 0],
B  [1, 1, 1, 0, 0],
C  [1, 1, 1, 0, 0],
D  [0, 0, 0, 1, 1],
E  [0, 0, 0, 1, 1]]
```

### Labels with Packing

Tokens that are "boundaries" between examples have labels set to `-100` (ignored in loss):

```
Packed:  [A, B, C, <sep>, D, E, <sep>, ...]
Labels:  [B, C, -100, D, E, -100, ...]
```

---

## Summary

| Concept | Key Point |
|---------|-----------|
| **Steps calculation** | `total_steps = ceil(samples / effective_batch) * epochs` |
| **Input == Labels** | Same tensor; model shifts internally |
| **Loss computation** | Predict token N given tokens 0..N-1 |
| **Gradient accumulation** | Gradients add up in `param.grad`, not loss values |
| **Packing** | Multiple examples per sequence for efficiency |

---

## Example Configuration

```python
from trl import SFTTrainer, SFTConfig

config = SFTConfig(
    output_dir='./sft_output',

    # Batch settings
    per_device_train_batch_size=1,
    gradient_accumulation_steps=64,  # Effective batch = 64

    # Training duration
    num_train_epochs=1,

    # Optimization
    learning_rate=5e-5,
    weight_decay=0.01,
    max_grad_norm=1.0,

    # SFT specific
    max_length=1024,
    packing=True,

    # Precision
    bf16=True,

    # Logging & saving
    logging_steps=10,
    eval_steps=200,
    save_steps=500,
)

trainer = SFTTrainer(
    model=model,
    args=config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

trainer.train()
```
