# Group Relative Policy Optimization (GRPO)

A step-by-step guide for applying GRPO after SFT training. GRPO is the algorithm behind DeepSeek-R1's reasoning capabilities.

## What Problem Does GRPO Solve?

DPO needs human-labeled preference pairs. For subjective tasks (tone, helpfulness) that makes sense. But for tasks with **verifiable correctness** — math, code, logic — we don't need humans to label preferences. We can just check if the answer is right.

GRPO lets you train with RL using automated reward signals. No human labels, no separate reward model, no critic/value network. The model generates multiple responses, the correct ones get reinforced, the wrong ones get penalized.

This is how DeepSeek-R1 learned to reason — pure RL on math problems with binary correctness rewards.

## Key Terms

**Policy model (π_θ)**: The model being trained. Starts as your SFT checkpoint.

**Reference model (π_ref)**: Frozen copy of SFT model. KL penalty keeps the policy close to it.

**Group**: For each prompt, generate G completions (typically 8-16). These form a "group."

**Reward function**: A function that scores each completion. Can be:
- Binary: 1.0 (correct) or 0.0 (incorrect)
- Continuous: any float
- Composite: weighted sum of multiple signals

**Advantage**: How much better a completion is compared to the group average:
```
advantage_i = (reward_i - mean(rewards)) / std(rewards)
```

**KL penalty**: A term in the loss that penalizes the policy for diverging from the reference model. Controlled by β.

**RLVR (RL with Verifiable Rewards)**: Using GRPO with automated correctness checking — no human labels needed.

## GRPO vs PPO: Why No Critic?

PPO (Proximal Policy Optimization) requires a **critic model** — a separate neural network the same size as the policy that estimates the "value" of each state. This doubles memory and adds training complexity.

GRPO's insight: for LLMs generating complete responses, we don't need per-token value estimates. We can estimate the baseline from the **group** of completions.

| | PPO | GRPO |
|--|-----|------|
| Critic model | Required (same size as policy) | Not needed |
| Baseline estimation | Learned value function | Group mean |
| Models in memory | 3 (policy, ref, critic) | 2 (policy, ref) |
| Reward granularity | Per-token | Per-sequence |
| Memory overhead | ~3x policy | ~2x policy |

## The GRPO Algorithm: Step by Step

### Step 1 — Generate a Group

For each prompt, sample G completions from the current policy using temperature sampling:

```
Prompt: "What is 15 × 7?"
Completions (G=8):
  1. "15 × 7 = 105"
  2. "Let me calculate: 15 × 7 = 115"
  3. "105"
  4. "The answer is 105."
  5. "15 × 7 = 95"
  6. "15 times 7 equals 105"
  7. "I think it's 100"
  8. "15 × 7 = 105. Check: 7×10=70, 7×5=35, 70+35=105"
```

### Step 2 — Score with Reward Function

Apply the reward function to each completion:

```
Rewards: [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
```

For math, the reward is binary: extract the final number, compare to ground truth.

### Step 3 — Compute Group Advantages

Normalize rewards within the group:

```
mean = 0.625
std  = 0.484

Advantages:
  1. (1.0 - 0.625) / 0.484 = +0.775   (correct, above average → reinforce)
  2. (0.0 - 0.625) / 0.484 = -1.291   (wrong, below average → penalize)
  3. (1.0 - 0.625) / 0.484 = +0.775
  4. (1.0 - 0.625) / 0.484 = +0.775
  5. (0.0 - 0.625) / 0.484 = -1.291
  6. (1.0 - 0.625) / 0.484 = +0.775
  7. (0.0 - 0.625) / 0.484 = -1.291
  8. (1.0 - 0.625) / 0.484 = +0.775
```

Correct completions get positive advantage → gradient pushes model to produce them more often. Wrong completions get negative advantage → gradient pushes model away from them.

### Step 4 — Policy Update

The loss combines clipped policy gradient + KL penalty:

```
L(θ) = -E[min(ratio × Â, clip(ratio, 1-ε, 1+ε) × Â)] + β × KL(π_θ || π_ref)
```

Where:
- `ratio = π_θ(completion) / π_old(completion)` — importance sampling
- `Â` — normalized advantage from step 3
- `clip(...)` — PPO-style clipping prevents too-large updates
- `β × KL` — keeps model close to SFT baseline

All tokens in a completion share the same advantage value (sequence-level reward, not per-token).

### Step 5 — Repeat

This is **online learning** — the model generates its own training data each iteration. As the policy improves, the generated completions get better, the reward signal shifts, and the model continues to improve.

## The Advantage Estimation: Why Group Normalization Works

The core insight: **relative performance within a group is more informative than absolute scores**.

Consider two scenarios:

**Easy prompt** — all completions correct:
```
Rewards: [1.0, 1.0, 1.0, 1.0]
Mean = 1.0, Std = 0.0
Advantage = 0 for all → no gradient signal
```

**Hard prompt** — mixed results:
```
Rewards: [1.0, 0.0, 0.0, 1.0]
Mean = 0.5, Std = 0.5
Advantage = ±1.0 → strong gradient signal
```

GRPO automatically focuses learning on problems the model partially solves — where there's signal to learn from. Fully solved and fully unsolved prompts contribute nothing, which is correct: there's nothing to learn from them.

## Reward Design

### Binary Correctness (Verifiable)

The simplest and most robust reward:

```python
def math_reward(completions, answers, **kwargs):
    rewards = []
    for completion, answer in zip(completions, answers):
        extracted = extract_final_number(completion)
        rewards.append(1.0 if extracted == answer else 0.0)
    return rewards
```

### Format Reward

Encourage structured reasoning:

```python
def format_reward(completions, **kwargs):
    rewards = []
    for c in completions:
        has_thinking = "<think>" in c and "</think>" in c
        has_answer = "<answer>" in c and "</answer>" in c
        rewards.append(1.0 if has_thinking and has_answer else 0.0)
    return rewards
```

### Composite Reward

Combine multiple signals:

```python
def combined_reward(completions, answers, **kwargs):
    rewards = []
    for c, a in zip(completions, answers):
        correct = 1.0 if extract_answer(c) == a else 0.0
        formatted = 1.0 if has_reasoning(c) else 0.0
        rewards.append(0.7 * correct + 0.3 * formatted)
    return rewards
```

**Reward design tips**:
- Start with binary correctness — it's the cleanest signal
- Add format rewards only if you want specific output structure
- Avoid continuous rewards that are hard to interpret (e.g., BLEU scores)
- Multiple reward functions can be passed separately to TRL

## GRPO is Secretly DPO

Recent research (arxiv 2510.00977) showed a surprising connection: GRPO with group size 2 reduces to something very close to DPO.

Within a group, every pair of (correct, incorrect) completions forms an implicit preference pair. Group normalization acts as variance reduction. The paper showed that **2-GRPO retains 98% of 16-GRPO's performance** while using 12.5% of the rollouts.

This means:
- DPO = offline preference learning (fixed dataset)
- GRPO = online preference learning (model generates its own pairs)
- The underlying mechanism is the same: contrastive learning from paired comparisons

## Practical Implementation with TRL

```python
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset

dataset = load_dataset("openai/gsm8k", split="train")

def correctness_reward(completions, answer, **kwargs):
    rewards = []
    for completion, ans in zip(completions, answer):
        extracted = extract_number(completion)
        rewards.append(1.0 if extracted == ans else 0.0)
    return rewards

config = GRPOConfig(
    num_generations=8,                  # group size
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=1e-6,                 # very low — RL is sensitive
    beta=0.04,                          # KL penalty strength
    temperature=0.9,                    # generation diversity
    max_completion_length=512,
    max_prompt_length=256,
    logging_steps=10,
)

trainer = GRPOTrainer(
    model=sft_model,
    args=config,
    train_dataset=dataset,
    reward_funcs=correctness_reward,
)

trainer.train()
```

## Key Hyperparameters

| Parameter | Typical Value | Effect |
|-----------|--------------|--------|
| `num_generations` | 8-16 | Group size. Larger = more stable advantage estimates, more compute |
| `learning_rate` | 1e-6 | Very low. RL updates are sensitive |
| `beta` | 0.04 | KL penalty. 0 = no constraint (faster but risky). 0.04 = balanced |
| `temperature` | 0.9 | Generation diversity. Higher = more exploration |
| `max_completion_length` | 256-1024 | Max tokens per completion |

**Tuning guide**:
- If reward_std stays near 0 → increase temperature (need more diverse completions)
- If model quality degrades → increase beta (too much drift from SFT)
- If model doesn't improve → decrease beta or increase learning rate
- If training is unstable → decrease learning rate

## Metrics to Monitor

- **reward** (mean): Should increase over training
- **reward_std**: Should stay > 0. If it collapses to 0, all completions are the same → no gradient signal
- **kl**: KL divergence from reference. Should grow slowly. If it explodes, beta is too low
- **completion_length**: If it grows unboundedly, the model may be gaming the reward

## The Homogeneous Group Problem

When all completions in a group get the same reward (all correct or all wrong), std = 0, advantages are undefined, and there's no learning signal.

Solutions:
- Higher temperature → more diverse generations
- Curriculum learning → train on problems the model partially solves
- Skip flat-reward groups
- Use composite rewards (correctness + format) to create variance

## DeepSeek-R1: GRPO at Scale

DeepSeek-R1 demonstrated two remarkable results:

**R1-Zero**: Applied GRPO directly to a base model (no SFT), using only math correctness rewards. The model spontaneously developed chain-of-thought reasoning, self-verification, and error correction — without being taught these behaviors. AIME 2024: 15.6% → 71.0%.

**R1**: Full pipeline — SFT → GRPO with mixed rewards (correctness + format). Matched OpenAI o1 on reasoning benchmarks.

The key lesson: with the right reward signal and enough compute, RL can discover reasoning strategies that humans never explicitly demonstrated.

## Applying GRPO to Your GPT-2 124M

Realistic expectations: a 124M model won't become a reasoning powerhouse. But GRPO will improve its accuracy on verifiable tasks.

**Best starting point**: GSM8K (grade school math)
1. Load your SFT checkpoint
2. Use binary correctness reward
3. Group size 8, lr=1e-6, beta=0.04
4. Train for 2-3 epochs
5. Measure: what % of GSM8K test problems does it get right before vs after?

The model will likely go from ~0% to maybe 5-10% accuracy (124M params is very small for math). But you'll have implemented the same algorithm that powers DeepSeek-R1.

## DPO vs GRPO: When to Use Which

| | DPO | GRPO |
|--|-----|------|
| Data | Preference pairs (human-labeled) | Prompts + reward function |
| Best for | Subjective quality (helpfulness, safety) | Verifiable tasks (math, code) |
| Training | Offline (fixed dataset) | Online (generates own data) |
| Compute | Lower (no generation during training) | Higher (generates G completions per prompt) |
| Reward signal | Implicit in preferences | Explicit reward function |
| Can discover new strategies | No (limited to dataset) | Yes (explores via sampling) |
