# SFT Input Format (from scratch)

**Yes, it's the same B x T matrix.** Two approaches:

## Approach 1: Packing (like train_gpt2.py) - Recommended

Concatenate examples directly, separated by `<|endoftext|>`:

```
[user1 tokens][assistant1 tokens]<|endoftext|>[user2 tokens][assistant2 tokens]<|endoftext|>...
```

```python
# Shape: (B, T) - same as pre-training
# No padding needed, just pack until T tokens
tokens = []
for example in examples:
    tokens.extend(enc.encode(example["text"]))
    tokens.append(enc.eot_token)  # <|endoftext|>

# Chunk into (B, T) batches
```

This is what you're already doing in `fineweb.py` - just tokenize and pack.

## Approach 2: Padding (traditional but wasteful)

```python
# Each row = one example, padded to T
# Shape: (B, T) with padding
# Need attention_mask to ignore pad tokens
```

## Key Difference for SFT: Loss Masking

In pre-training, you compute loss on ALL tokens. In SFT, you typically **only compute loss on assistant responses**, not prompts:

```python
# tokens:  [<|user|> question <|assistant|> answer <|endoftext|>]
# labels:  [-100    -100     -100          answer  <|endoftext|>]
#           ↑ ignored in loss              ↑ loss computed here
```

```python
def forward_sft(self, idx, labels=None):
    # idx shape: (B, T)
    logits, _ = self.forward(idx)  # (B, T, vocab_size)

    if labels is not None:
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # -100 is ignored by cross_entropy
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
    return logits, loss
```

## Creating Labels with Masking

```python
def create_labels(tokens, assistant_start_positions):
    labels = tokens.clone()
    # Mask everything before assistant response
    for i, pos in enumerate(assistant_start_positions):
        labels[i, :pos] = -100
    return labels
```

## Simple Version (no masking, like pre-training)

If you don't care about masking prompts (simpler, works fine for small models):

```python
# Just use tokens as both input and target, same as train_gpt2.py
x = batch[:, :-1]  # input
y = batch[:, 1:]   # target (shifted by 1)
loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
```

## TL;DR

Same B x T matrix, concatenate with `<|endoftext|>` separators, optionally mask prompt tokens in loss computation.
