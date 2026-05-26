# Direct Preference Optimization (DPO)

A step-by-step guide for applying DPO after SFT training.

## Background

**Paper**: [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) (Rafailov et al., Stanford, May 2023)

**Research lineage** — DPO builds on a chain of prior work:

1. **RLHF / InstructGPT** (Ouyang et al., OpenAI, 2022) — Established the SFT → Reward Model → PPO pipeline for aligning LLMs with human preferences. DPO's entire motivation is simplifying this pipeline.

2. **PPO** (Schulman et al., OpenAI, 2017) — The RL algorithm used in RLHF. PPO uses a clipped surrogate objective to stabilize policy gradient updates. DPO eliminates the need for PPO entirely.

3. **Bradley-Terry model** (Bradley & Terry, 1952) — A classical statistical model for pairwise comparisons. DPO uses it to model human preferences: P(A > B) = sigmoid(r_A - r_B). This is the mathematical foundation that enables the DPO loss derivation.

4. **KL-constrained reward maximization** — The standard RLHF objective maximizes reward while staying close to the reference policy via a KL penalty: `max E[r(x,y)] - β·KL(π_θ || π_ref)`. DPO's key insight is deriving the closed-form solution to this objective, bypassing the need to solve it with RL.

**The key contribution**: DPO proved that the optimal policy for the KL-constrained RLHF objective has a closed-form relationship with the reward function. This means you can reparameterize the reward in terms of the policy, eliminating the reward model and RL loop entirely — reducing alignment to a single supervised classification step.

**Related work (parallel, not direct ancestor)**: Constitutional AI (Bai et al., Anthropic, 2022) — used the standard RLHF pipeline but replaced human labelers with AI-generated preferences (RLAIF). Showed preference optimization works at scale, but still relied on reward model + PPO.

## What Problem Does DPO Solve?

SFT teaches a model to follow instructions by imitating examples. But for subjective qualities — helpfulness, safety, tone — there's no single "correct" answer. Instead, we have **preferences**: response A is better than response B.

Traditional RLHF solves this with a 3-stage pipeline:
1. SFT (done)
2. Train a separate reward model on preference pairs
3. Optimize the policy with PPO against the reward model

DPO collapses stages 2-3 into a single supervised learning step. No reward model, no RL loop, no PPO complexity.

## Key Terms

**Preference pair**: A prompt with two responses — one **chosen** (preferred) and one **rejected**.
```
Prompt:    "What is photosynthesis?"
Chosen:    "Photosynthesis is the process where plants convert sunlight into energy..."
Rejected:  "I'm not sure about that topic."
```

**Policy model (π_θ)**: The model being trained. Starts as your SFT checkpoint.

**Reference model (π_ref)**: A frozen copy of your SFT checkpoint. Acts as an anchor — prevents the policy from drifting too far and producing degenerate outputs.

**Bradley-Terry model**: A probability model for pairwise comparisons. Given rewards r_A and r_B:
```
P(A preferred over B) = sigmoid(r_A - r_B)
```
Only the **difference** between rewards matters, not absolute values.

**Implicit reward**: DPO doesn't train a separate reward model. Instead, the reward is embedded in the policy itself:
```
r(x, y) = β · log(π_θ(y|x) / π_ref(y|x))
```
The log-probability ratio between policy and reference IS the reward.

## The DPO Loss Function

```
L_DPO = -E[log σ(β · log(π_θ(y_w|x)/π_ref(y_w|x)) - β · log(π_θ(y_l|x)/π_ref(y_l|x)))]
```

Breaking this down:

1. Compute log-probability of chosen response under both policy and reference
2. Compute log-probability of rejected response under both policy and reference
3. Take the difference of the log-ratios (this is the implicit reward margin)
4. Scale by β
5. Pass through sigmoid + log (binary cross-entropy)

**What the gradient does**: When the model wrongly ranks rejected > chosen, the loss is high and the gradient is large. When the model correctly ranks chosen > rejected, the loss is small. This naturally focuses learning on the hard examples.

**β (beta)**: Controls how aggressively the model chases preferences.
- Low β (0.05): Aggressive updates, risk of mode collapse
- High β (0.5-1.0): Conservative, stays close to reference
- Typical starting point: **0.1**

## Why the Math Works: The Cancellation Trick

The RLHF objective has an intractable partition function Z(x). The key insight of DPO:

```
Implicit reward for y_w: β·log(π_θ(y_w|x)/π_ref(y_w|x)) + β·log(Z(x))
Implicit reward for y_l: β·log(π_θ(y_l|x)/π_ref(y_l|x)) + β·log(Z(x))

Difference: The Z(x) terms cancel!
```

This cancellation transforms an intractable RL problem into simple supervised classification on preference pairs.

## Walkthrough: One DPO Update (Full Numerical Example)

**Setup**:
```
Prompt:   "Explain gravity"     → tokens [8912, 9203, 15]       (3 tokens)
Chosen:   "Gravity is a force"  → tokens [38, 12, 5, 901]       (4 tokens)
Rejected: "I don't know"        → tokens [40, 67, 88]           (3 tokens)
β = 0.1
vocab_size = 50257 (GPT-2)
```

We need 4 forward passes. Let's trace **π_θ on chosen** in full detail, then summarize the other three.

### Pass 1: π_θ on chosen

**Step 1 — Concatenate prompt + chosen:**

```python
input_ids = [8912, 9203, 15, 38, 12, 5, 901]   # shape: (7,)
#            ─── prompt ───  ──── chosen ────
```

**Step 2 — Forward pass → logits:**

Autoregressive: position i predicts token at position i+1.

```python
logits = model(input_ids)   # shape: (7, 50257)
```

**What are logits?** Raw scores from the model's final linear layer (`lm_head`). They are **not probabilities** — they can be any real number (positive, negative, large, small). Higher value = model thinks that token is more likely to come next, but the numbers don't sum to 1 and aren't bounded. Think of them as "votes" for each vocabulary token.

```
              token 5    token 12   token 38   token 901  ... (50257 cols)
              ───────    ────────   ────────   ─────────
position 0:   -2.1       -3.4       -1.8       -5.2       ...  → predicts 9203 (prompt)
position 1:   -4.0       -2.9       -3.1       -6.1       ...  → predicts 15 (prompt)
position 2:   -3.2       -2.0       +1.8       -4.5       ...  → predicts 38 ← FIRST RESPONSE TOKEN
position 3:   -1.5       +2.4       -3.3       -5.0       ...  → predicts 12
position 4:   +2.1       -2.8       -1.9       -4.3       ...  → predicts 5
position 5:   -3.8       -2.1       -4.0       +1.6       ...  → predicts 901
position 6:   ...        ...        ...        ...        ...  → (nothing to predict)
```

At position 2, token 38 ("Gravity") has the highest logit (+1.8) among the shown tokens — the model thinks "Gravity" is a likely next word after "Explain gravity". But +1.8 isn't a probability.

**Step 3 — log_softmax → per-token log-probabilities:**

```python
log_probs = F.log_softmax(logits, dim=-1)   # shape: (7, 50257)
```

**What does log_softmax do?** Two things in one numerically stable operation:

1. **Softmax** converts logits to a probability distribution. For each row (position), it exponentiates every logit and normalizes so they sum to 1:
   ```
   softmax(z_i) = e^(z_i) / Σ_j e^(z_j)
   ```
   After softmax, every value is between 0 and 1, and each row sums to 1 — a proper probability distribution over the vocabulary.

2. **Log** converts probabilities back to log-space. Log-probabilities are always ≤ 0 (since log of a number between 0 and 1 is negative).

**Why not just use the probabilities directly?** Because we need to **multiply** hundreds of per-token probabilities to get P(response|prompt). Multiplying many small numbers (0.4 × 0.9 × 0.3 × ...) quickly underflows to 0 in floating point. In log-space, products become sums: log(a×b) = log(a) + log(b). Sums of numbers like -0.9 + -0.1 + -1.2 are numerically stable.

**Why not just log the raw logits?** Logits aren't probabilities — they don't sum to 1 and can be negative (can't take log of a negative number). Softmax is necessary to turn raw scores into a valid probability distribution first.

Showing position 2 as a concrete example (predicting the first response token):

```
Logits at position 2:    ..., -3.2, -2.0, +1.8, -4.5, ...  (50257 raw scores)
                                              ↓
Softmax at position 2:   ..., 0.001, 0.003, 0.12, 0.0002, ...  (50257 probs, sum=1.0)
                                              ↓
Log at position 2:       ..., -6.9, -5.8, -2.12, -8.5, ...  (50257 log-probs, all ≤ 0)
```

Token 38 ("Gravity") had logit +1.8 → probability 0.12 → log-prob -2.12. It's not the most probable token overall (0.12 < 1.0), but it's among the higher-probability tokens in this 50257-wide distribution.

Full log_probs at response-predicting positions:

```
              token 5    token 12   token 38   token 901  ... (50257 cols)
              ───────    ────────   ────────   ─────────
position 2:   -5.8       -4.6       -0.82      -7.1       ...
position 3:   -4.1       -0.21      -5.9       -7.6       ...
position 4:   -0.35      -5.3       -4.4       -6.8       ...
position 5:   -6.3       -4.6       -6.5       -0.94      ...
```

**Step 4 — Gather the actual next token at each position:**

At each position, pick the log-prob of the token that actually appears next.

```python
# target_ids: what each position should predict
target_ids = [9203, 15, 38, 12, 5, 901, --]
#                       ^^  ^^  ^  ^^^
#                       these are the response tokens we care about

per_token = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
# shape: (7,)
```

```
position 0: log_probs[0, 9203] = -1.10  (prompt → mask)
position 1: log_probs[1, 15]   = -0.45  (prompt → mask)
position 2: log_probs[2, 38]   = -0.82  ← "Gravity"
position 3: log_probs[3, 12]   = -0.21  ← "is"
position 4: log_probs[4, 5]    = -0.35  ← "a"
position 5: log_probs[5, 901]  = -0.94  ← "force"
```

**Step 5 — Mask prompt, sum response tokens:**

```python
response_mask = [0, 0, 1, 1, 1, 1, 0]   # shape: (7,)
masked = per_token * response_mask
# = [0, 0, -0.82, -0.21, -0.35, -0.94, 0]

log_p_chosen_θ = masked.sum()   # shape: scalar
# = -0.82 + (-0.21) + (-0.35) + (-0.94)
# = -2.32
```

### Passes 2, 3, 4: Same Steps, Different Model/Response

**Pass 2 — π_ref on chosen** (same response, frozen model):

```
input:  [8912, 9203, 15, 38, 12, 5, 901]
gather: position 2→-0.91  3→-0.25  4→-0.40  5→-1.02
log_p_chosen_ref = -0.91 + (-0.25) + (-0.40) + (-1.02) = -2.58
```

**Pass 3 — π_θ on rejected:**

```
input:  [8912, 9203, 15, 40, 67, 88]
gather: position 2→-2.30  3→-1.85  4→-1.50
log_p_rejected_θ = -2.30 + (-1.85) + (-1.50) = -5.65
```

**Pass 4 — π_ref on rejected:**

```
input:  [8912, 9203, 15, 40, 67, 88]
gather: position 2→-2.10  3→-1.70  4→-1.40
log_p_rejected_ref = -2.10 + (-1.70) + (-1.40) = -5.20
```

### Summary of All 4 Passes

```
                     π_θ (training)    π_ref (frozen)
                     ──────────────    ──────────────
log P(chosen|prompt)    -2.32             -2.58
log P(rejected|prompt)  -5.65             -5.20
```

### DPO Loss Computation

```python
# Log-ratios: how much has π_θ shifted from π_ref?
chosen_logr  = log_p_chosen_θ - log_p_chosen_ref      # -2.32 - (-2.58) = +0.26
rejected_logr = log_p_rejected_θ - log_p_rejected_ref  # -5.65 - (-5.20) = -0.45

# This is log(π_θ/π_ref), NOT dividing log-probs.
# log(a/b) = log(a) - log(b), so subtraction of log-probs = log of probability ratio.

# Margin
margin = β * (chosen_logr - rejected_logr)   # 0.1 * (0.26 - (-0.45)) = 0.1 * 0.71 = 0.071

# Loss
loss = -log(sigmoid(0.071)) = -log(0.518) = 0.659
```

Slightly better than coin-flip loss (0.693). Gradient pushes π_θ to widen this gap.

### What the Gradient Does Next

The update will increase log P(chosen) and decrease log P(rejected) under π_θ. After many steps:

```
Step 1:    margin=0.071  σ=0.518  loss=0.659
Step 100:  margin=0.8    σ=0.69   loss=0.37
Step 500:  margin=2.1    σ=0.89   loss=0.12
Step 1000: margin=3.5    σ=0.97   loss=0.03
```

### How Backward and Gradient Propagation Works

The DPO loss is a single scalar computed from **both** the chosen and rejected forward passes. `loss.backward()` propagates gradients through both paths.

```
                    π_θ (shared weights)
                   /                    \
      forward(prompt+chosen)    forward(prompt+rejected)
              ↓                          ↓
       log_p_chosen_θ            log_p_rejected_θ
              ↓                          ↓
              └──────── DPO loss ────────┘
                          ↓
                     loss.backward()
                          ↓
                  gradients flow back UP
                  through BOTH paths into
                  the SAME π_θ weights
```

It's one computational graph with two branches sharing the same parameters. Both `log_p_chosen_θ` and `log_p_rejected_θ` contributed to the loss, so both paths produce gradients that **accumulate** (sum) into the same weight tensors.

The gradient direction from each path:
- **Chosen path**: pushes weights to **increase** log_p_chosen_θ (make chosen more likely)
- **Rejected path**: pushes weights to **decrease** log_p_rejected_θ (make rejected less likely)

The π_ref paths don't participate in backprop — they're `torch.no_grad()` or detached tensors, just constants in the loss formula.

**What the gradients look like**: Every parameter tensor in π_θ has a matching `.grad` tensor of the **same shape**:

```
Parameter                        Shape              .grad Shape
─────────────────────────────    ─────────────────  ─────────────────
wte.weight (token embeddings)    (50257, 768)       (50257, 768)
wpe.weight (position embeddings) (1024, 768)        (1024, 768)
h[0].attn.c_attn.weight         (768, 2304)        (768, 2304)
h[0].mlp.c_fc.weight            (768, 3072)        (768, 3072)
...every layer...
lm_head.weight                   (50257, 768)       (50257, 768)
```

One gradient value per weight — "how should this weight change to reduce the loss?"

**How accumulation works during `loss.backward()`**: PyTorch walks the computation graph. When it reaches a weight `W` that was used in both paths:

```
W.grad += ∂loss/∂W via chosen path     # e.g., adds +0.003 to W.grad[i,j]
W.grad += ∂loss/∂W via rejected path   # e.g., adds -0.005 to W.grad[i,j]
                                        # W.grad[i,j] is now -0.002
```

There's no separate "combine" step — it's just addition as each path's gradients arrive. Then the optimizer applies it once:

```python
optimizer.step()
# For each parameter W:
#   W = W - lr * W.grad      (simplified, AdamW is more complex)
```

This is the same principle as any multi-branch loss in PyTorch (contrastive learning, siamese networks, etc.).

### Shape Summary

```
Step                    Shape               Note
──────────────────────  ──────────────────  ────────────────────────
input_ids               (seq_len,)          7 for chosen, 6 for rejected
logits                  (seq_len, 50257)    raw scores, any real number
log_probs               (seq_len, 50257)    after log_softmax, all ≤ 0
per_token (gathered)    (seq_len,)          one log-prob per position
masked                  (seq_len,)          zeros on prompt positions
log P(response|prompt)  scalar              sum of response token log-probs
```

Each of the 4 passes produces one scalar. The DPO loss is computed from those 4 scalars.

**Memory optimization**: Since π_ref is frozen, precompute its log-probs once before training and store them in the dataset.

## DPO vs RLHF

| | RLHF (PPO) | DPO |
|--|-----------|-----|
| Reward model | Train separately | Implicit in policy |
| RL algorithm | PPO (complex) | None (supervised loss) |
| Sampling during training | Required | Not required |
| Models in memory | 4 (policy, ref, reward, value) | 2 (policy, ref) |
| Stability | Hard to tune | Straightforward |
| Performance | Strong | Matches or exceeds |
| Implementation | ~500 lines | ~50 lines |

## Datasets for DPO

**Anthropic HH-RLHF** (~161k pairs)
- Human-written preference pairs on helpfulness and harmlessness
- Format: conversation → chosen/rejected responses
- Good for: general alignment

**UltraFeedback** (~64k pairs)
- GPT-4-labeled preferences across diverse tasks
- Higher quality labels, more diverse prompts
- Good for: instruction-following improvement

**Dataset format**:
```python
{
    "prompt": "What is the capital of France?",
    "chosen": "The capital of France is Paris.",
    "rejected": "France is a country in Europe."
}
```

## Practical Implementation with TRL

```python
from trl import DPOTrainer, DPOConfig

config = DPOConfig(
    beta=0.1,                          # preference strength
    learning_rate=5e-6,                # lower than SFT
    per_device_train_batch_size=4,     # each sample = chosen + rejected
    num_train_epochs=1,                # often 1 epoch is enough
    max_length=512,
    max_prompt_length=256,
)

trainer = DPOTrainer(
    model=sft_model,                   # your SFT checkpoint
    ref_model=None,                    # auto-uses initial model state
    args=config,
    train_dataset=preference_dataset,
    processing_class=tokenizer,
)

trainer.train()
```

**Key metrics to monitor**:
- `rewards/margins`: chosen reward - rejected reward (should increase)
- `rewards/accuracies`: % where chosen > rejected (should approach 100%)
- `logps/chosen`: log-prob of chosen (watch for collapse — shouldn't drop too much)

## Failure Modes

**Likelihood collapse**: DPO only optimizes the ratio π_θ/π_ref, not absolute probabilities. The model can lower both chosen and rejected log-probs while improving the margin. Watch `logps/chosen` — if it drops significantly, the model is degrading overall quality while "winning" on preferences.

**Overfitting**: With small preference datasets, the model memorizes the specific pairs. Use early stopping and monitor held-out preference accuracy.

**Distribution shift**: If preference pairs come from a very different model than your SFT model, the implicit rewards may be noisy. Best results when preference data is generated by or similar to the SFT model.

## When to Use DPO

Good fit:
- You have preference pairs (chosen vs rejected)
- Subjective quality matters (tone, helpfulness, safety)
- You want simplicity over PPO
- After SFT, as a refinement step

Not the best fit:
- Tasks with verifiable correctness (math, code) → use GRPO instead
- No preference data available
- Model needs to learn fundamentally new capabilities (that's SFT's job)

## Applying DPO to Your GPT-2 124M

1. Load your SFT checkpoint as both policy and reference
2. Use Anthropic HH-RLHF or UltraFeedback for preference data
3. Train with β=0.1, lr=5e-6, 1 epoch
4. Compare samples before/after DPO
5. Evaluate: does the model prefer helpful, detailed responses over vague ones?

The improvement on a 124M model will be subtle — the model's knowledge capacity is the bottleneck, not its alignment. But the training mechanics are identical to what labs do at scale.
