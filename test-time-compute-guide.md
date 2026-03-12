# Test-Time Compute and Scaling It

A guide to the new scaling frontier: spending more compute at inference to get better answers.

## Background

**Foundational paper**: [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314) (Snell et al., UC Berkeley, Aug 2024)

**Research lineage** — test-time compute scaling builds on a chain of prior work:

1. **Chain-of-Thought prompting** (Wei et al., Google, 2022) — Showed that prompting LLMs with "Let's think step by step" dramatically improves reasoning. The first evidence that generating more tokens at inference time helps.

2. **Self-Consistency** (Wang et al., Google, 2023, [arxiv 2203.11171](https://arxiv.org/abs/2203.11171)) — Instead of greedy decoding one chain-of-thought, sample multiple reasoning paths and take a majority vote on the final answer. A simple but powerful form of parallel test-time compute scaling.

3. **Let's Verify Step by Step** (Lightman et al., OpenAI, 2023, [arxiv 2305.20050](https://arxiv.org/abs/2305.20050)) — Introduced process reward models (PRMs) that score each reasoning step, not just the final answer. PRMs solved 78% of MATH problems vs. outcome reward models' lower accuracy. Released PRM800K: 800,000 step-level human feedback labels.

4. **Self-Refine** (Madaan et al., CMU, 2023, [arxiv 2303.17651](https://arxiv.org/abs/2303.17651)) — Model generates output, critiques it, then revises iteratively. No extra training needed — a single LLM acts as generator, critic, and refiner.

5. **OpenAI o1** (OpenAI, Sep 2024) — First production model trained via RL to perform extended chain-of-thought reasoning at inference. Demonstrated that test-time compute scales like a new dimension — more "thinking" yields better answers. Internal chain-of-thought is hidden from the user.

6. **DeepSeek-R1** (DeepSeek, Jan 2025, [arxiv 2501.12948](https://arxiv.org/abs/2501.12948)) — Open-weight reasoning model trained with GRPO. R1-Zero showed that pure RL (no SFT) can produce emergent chain-of-thought reasoning. Matched o1 on reasoning benchmarks.

7. **The Art of Scaling Test-Time Compute** (Agarwal et al., Dec 2024, [arxiv 2512.02008](https://arxiv.org/abs/2512.02008)) — Large-scale empirical study across 8 LLMs (7B-235B), 30B+ generated tokens, 4 reasoning datasets. Key finding: no single test-time scaling strategy dominates — the optimal approach depends on model type, problem difficulty, and compute budget.

**The shift**: For years, the recipe for better AI was simple — train bigger models on more data (GPT-2 → GPT-3 → GPT-4). Test-time compute opened a second axis: keep the model the same size, but let it think longer at inference. This is the new scaling frontier.

## What Problem Does Test-Time Compute Solve?

Traditional scaling (train-time) is hitting diminishing returns. Chinchilla showed optimal data/parameter ratios, but even with optimal allocation, doubling training compute yields diminishing accuracy gains on hard problems. There are also practical limits: training runs cost tens of millions of dollars, take months, and require enormous clusters.

Test-time compute offers a different tradeoff:

```
Train-time scaling:  Spend $100M once → fixed capability for all queries
Test-time scaling:   Spend $0.01-$10 per query → adaptive capability per query
```

The insight is that not all queries need the same amount of compute. A simple factual question ("What's the capital of France?") needs one forward pass. A competition math problem might benefit from 100x more inference compute — generating multiple solution attempts, verifying each step, backtracking from dead ends.

This is the "thinking longer" paradigm: instead of a smarter brain, give the same brain more time to think.

## Key Terms

**Test-time compute (inference-time compute)**: Computation spent when the model is generating a response, as opposed to training. Includes all FLOPs during inference — generating tokens, scoring candidates, running reward models, search.

**Inference-time scaling**: The phenomenon that model performance improves predictably as you increase test-time compute, analogous to how pre-training performance improves with more training compute (scaling laws).

**Best-of-N sampling (BoN)**: Generate N independent responses, score each, return the best one. The simplest form of test-time compute scaling.

**Majority voting / self-consistency**: Generate N responses, extract the final answer from each, return the most common answer. No reward model needed.

**Process Reward Model (PRM)**: A model trained to score each intermediate reasoning step. Enables search over partial solutions.

**Outcome Reward Model (ORM)**: A model trained to score only the final answer/solution. Simpler but less useful for guiding search.

**Tree search**: Exploring multiple reasoning paths in a tree structure (branching at each step), using a reward model to decide which branches to expand. Includes beam search and Monte Carlo Tree Search (MCTS).

**Sequential revision**: The model iteratively refines its own answer — generate, critique, revise, repeat.

**Compute-optimal scaling**: Allocating a fixed inference compute budget optimally across strategies and per-problem difficulty. The analog of Chinchilla's compute-optimal training.

## The Two Paradigms of Scaling

### Train-Time Scaling

The paradigm from 2018-2023. Make the model better by spending more compute during training:

- More parameters (GPT-2 1.5B → GPT-3 175B → GPT-4 ~1.8T MoE)
- More data (Chinchilla: tokens should scale linearly with parameters)
- Longer training (more gradient steps)

This produces a fixed model. Every query gets the same capability — the same weights, the same single forward pass. A trivial question and a PhD-level math problem get the same amount of compute.

**Scaling law**: Loss decreases as a power law of training compute. Roughly, you need 10x more compute for each halving of loss.

### Test-Time Scaling

The paradigm emerging since 2024. Make the model's output better by spending more compute at inference:

- Generate more candidate solutions
- Let the model "think" in longer reasoning chains
- Search over reasoning steps with verifiers
- Iteratively revise answers

This produces variable compute per query. Easy questions are fast and cheap. Hard questions use more compute and cost more, but get better answers.

**Scaling law**: Accuracy improves predictably with inference compute, but the curve shape depends on problem difficulty (more on this in the Compute-Optimal section).

### The Crossover Point

Snell et al. (2024) showed that a smaller model with optimized test-time compute can outperform a much larger model using standard inference. Their key finding:

```
A compute-optimal Llama 3.2 3B + test-time scaling  >  Llama 3.1 405B (single pass)
                                                        on MATH-500
```

That's a 135x smaller model winning by spending more compute at the right time, on the right problems.

When is each approach more efficient?

```
Situation                         Better approach
──────────────────────────────    ─────────────────────────
Serving millions of simple queries    Bigger model, single pass
Hard reasoning/math problems          Test-time compute on smaller model
Need to improve ONE specific answer   Test-time compute
Fixed per-query latency budget        Bigger model
Variable latency acceptable           Test-time compute
Limited training budget               Test-time compute on existing model
```

The general principle: train-time compute is a fixed capital investment amortized over all queries. Test-time compute is a variable cost that can be targeted at queries that need it.

## Methods for Scaling Test-Time Compute

### 1. Best-of-N Sampling

The simplest approach. Generate N independent responses, score each with a reward model (or other verifier), return the highest-scoring one.

```
Prompt: "Prove that sqrt(2) is irrational."

Generate N=8 responses independently (temperature > 0)

Response 1: [correct proof by contradiction]     → reward: 0.92
Response 2: [proof with algebra error in step 3] → reward: 0.31
Response 3: [correct but verbose proof]           → reward: 0.85
Response 4: [incomplete proof]                    → reward: 0.15
Response 5: [correct, elegant proof]              → reward: 0.95  ← return this one
Response 6: [wrong approach entirely]             → reward: 0.08
Response 7: [correct proof, different method]     → reward: 0.88
Response 8: [mostly correct, minor gap]           → reward: 0.72
```

**How to score** — three approaches, from simplest to most powerful:

**a) Majority voting (no reward model needed)**:
Extract the final answer from each response, return the most common one. Works well when answers are discrete (a number, a multiple-choice letter). This is Wang et al.'s self-consistency.

```
Prompt: "What is 23 x 47?"

8 responses → final answers: [1081, 1081, 1081, 1081, 1061, 1081, 1081, 1061]

Majority vote: 1081 (6 out of 8)  ← return this
```

**b) Outcome reward model (ORM)**:
A separate model trained to score complete solutions. Assign a scalar score to each response, pick the highest.

**c) Process reward model (PRM)**:
Score each reasoning step. The overall score can be the product (or min) of step-level scores. More expensive but catches errors earlier.

**Scaling behavior**: Best-of-N has diminishing returns. Going from N=1 to N=4 gives a large boost. Going from N=64 to N=128 gives a small boost. The improvement scales roughly as O(log N) — you need exponentially more samples for linear accuracy gains.

```
N      MATH accuracy (illustrative)
──     ────────────────────────────
1      50%
4      62%
16     70%
64     75%
256    78%
```

**Cost**: Linear in N. Generating 64 responses costs 64x a single response. This is why smarter search strategies (tree search) can be more compute-efficient.

### 2. Sequential Revision

The model generates an initial answer, then revises it one or more times:

```
Round 1 (generate):  "The integral of x^2 is x^3 + C"
Round 2 (critique):  "Wait, I forgot the coefficient. The integral of x^n is x^(n+1)/(n+1)."
Round 3 (revise):    "The integral of x^2 is x^3/3 + C"
```

This is what Self-Refine (Madaan et al., 2023) formalized. The model plays three roles:
1. **Generator**: produce initial output
2. **Critic**: identify errors or weaknesses
3. **Refiner**: fix the identified issues

No extra training or reward model needed — the same LLM does all three via prompting. Typically 2-4 rounds of revision, with diminishing improvements after that.

**When it works**: Tasks where the model can recognize its own errors (code bugs, factual mistakes). When the model can't reliably critique itself, revision can actually make correct answers worse.

**Relation to o1/R1**: These models internalize sequential revision into a single long chain-of-thought. Instead of explicit generate-critique-revise rounds, the model learns (via RL) to self-correct within its reasoning trace: "Wait, that's not right. Let me try again..."

### 3. Tree Search with Reward Models

Instead of generating complete solutions independently (best-of-N) or revising sequentially, tree search explores the space of partial solutions in a structured way.

The idea: treat each reasoning step as a node in a tree. At each node, generate multiple possible next steps (branches). Use a reward model to evaluate which branches are most promising, then expand those.

```
                        "Prove sqrt(2) is irrational"
                       /              |              \
              "Assume it's         "Consider        "By the fundamental
               rational: p/q"      p^2 = 2q^2"      theorem of arithmetic"
              /        \               |
       "Then p^2=2q^2" "So p/q          ...
        (PRM: 0.9)      in lowest terms"
        /       \        (PRM: 0.85)
  "p^2 is even"  "p must be
   (PRM: 0.95)    divisible by 4"
       |           (PRM: 0.2) ← prune
  "so p is even"
   (PRM: 0.93)
       |
      ...
```

**Beam search**: Keep the top-K partial solutions at each step. Expand each by generating next steps. Score with PRM, keep top-K again. Simple, deterministic, efficient.

**Monte Carlo Tree Search (MCTS)**: The algorithm from AlphaGo, adapted for reasoning. Four phases per iteration:
1. **Select**: Walk down the tree choosing branches via UCB (balances exploitation of high-reward paths with exploration of under-visited ones)
2. **Expand**: Generate a new reasoning step at the selected node
3. **Simulate**: Complete the solution from that point (rollout)
4. **Backpropagate**: Update scores of all ancestor nodes based on the outcome

MCTS is more compute-expensive than beam search but can backtrack from dead ends — beam search only moves forward.

**Why PRMs are critical for tree search**: An ORM can only score complete solutions, so you'd need to finish every branch before evaluating it (expensive). A PRM scores partial solutions, so you can prune bad branches early — "this step is wrong, don't continue down this path." This makes search tractable.

**ReST-MCTS*** (Zhang et al., 2024, [arxiv 2406.03816](https://arxiv.org/abs/2406.03816)) showed that MCTS with process reward guidance outperforms both best-of-N and Tree-of-Thought, within the same compute budget.

### 4. Chain-of-Thought / Extended Thinking

The approach used by o1, o3, and R1. Rather than external search (generating multiple responses and selecting), the model performs internal search — producing a long reasoning chain where it explores, backtracks, and self-corrects within a single generation.

```
<thinking>
Let me solve this step by step.

First, I'll try direct computation...
23 × 47 = 23 × 40 + 23 × 7 = 920 + 161 = 1081

Wait, let me verify: 23 × 50 = 1150, minus 23 × 3 = 69, so 1150 - 69 = 1081. Good.

Actually, let me double-check 23 × 7: 20×7=140, 3×7=21, 140+21=161. Yes.
And 23 × 40: 23×4=92, ×10=920. Correct.

920 + 161 = 1081. Confirmed.
</thinking>

The answer is 1081.
```

The model is trained via RL (PPO or GRPO) to produce these reasoning traces. The reward signal comes from whether the final answer is correct. Through RL, the model learns emergent behaviors:
- Breaking problems into sub-problems
- Trying multiple approaches
- Recognizing and correcting mistakes
- Verifying its own work

**How it scales**: Longer thinking → more tokens → more compute → better accuracy. The model learns when to think more (hard problems) and when to answer quickly (easy ones). DeepSeek-R1 produces reasoning traces from hundreds to tens of thousands of tokens depending on difficulty.

**Key difference from methods 1-3**: No external verifier or search algorithm needed. The search is internalized into the model's weights via RL training. This is arguably more elegant but requires specialized training — you can't just prompt a standard model to do this effectively.

## Process Reward Models vs Outcome Reward Models

This distinction is central to effective test-time compute scaling.

### Outcome Reward Model (ORM)

Scores only the final answer or complete solution:

```
Input:  [problem + full solution]
Output: scalar score (e.g., 0.85)
```

Training data: (problem, solution, correct/incorrect) pairs. Straightforward to collect — you just need to check final answers.

**Limitation**: An ORM gives no signal about where a solution went wrong. A proof with an error in step 2 and a correct conclusion gets a different score than a proof with an error in step 7, but the ORM can't tell you which step is faulty.

### Process Reward Model (PRM)

Scores each intermediate step:

```
Input:  [problem + steps 1..k]
Output: scalar score for step k (e.g., 0.93)

Example:
  Step 1: "Let x = sqrt(2), assume x = p/q"      → PRM score: 0.95
  Step 2: "Then p^2 = 2q^2"                        → PRM score: 0.97
  Step 3: "So p^2 is even, meaning p is even"      → PRM score: 0.94
  Step 4: "Let p = 2k, then 4k^2 = 2q^2, q^2=2k^2"→ PRM score: 0.96
  Step 5: "So q is also even — contradiction"       → PRM score: 0.98
```

Training data: requires step-level correctness labels. This is expensive — Lightman et al. collected 800,000 step-level labels from human annotators to train PRM800K.

### Why PRMs Enable Better Search

Consider searching for a correct proof with a budget of 100 model evaluations:

**With ORM (best-of-N)**:
- Generate 100 complete solutions
- Score each with ORM
- Return the highest-scoring one
- You explored 100 independent paths but couldn't reuse partial work

**With PRM (tree search)**:
- Start step 1, generate 5 variants, PRM scores them
- Keep top 2, generate 5 step-2 variants for each (10 total)
- PRM prunes to top 4 partial solutions
- Continue expanding and pruning
- With 100 evaluations, you've explored many more potential paths because you pruned bad branches early

The PRM acts like a heuristic in A* search — it guides you toward promising areas of the solution space without exhaustively exploring everything.

**Lightman et al. (2023) result**: On MATH, best-of-N with a PRM as the scorer solved 78.2% of problems. Best-of-N with an ORM scored lower. The PRM provides strictly more information — it can act as an ORM (by looking at the final step score) but also enables step-level search.

## Compute-Optimal Test-Time Scaling

This is the key contribution of Snell et al. (2024). Not all problems benefit equally from more inference compute. The optimal strategy depends on problem difficulty.

### The Difficulty Spectrum

**Easy problems** (model usually gets them right on the first try):
```
"What is 7 + 5?"

N=1:  98% accuracy
N=16: 99% accuracy  (diminishing returns — 16x compute for 1% gain)
```
More test-time compute barely helps. The model already knows the answer. Spending 16x compute for 1% accuracy gain is wasteful.

**Medium problems** (model sometimes gets them right):
```
"Solve: x^3 - 6x^2 + 11x - 6 = 0"

N=1:  35% accuracy
N=16: 72% accuracy  (sweet spot — 16x compute for 37% gain)
N=64: 81% accuracy
```
This is the sweet spot for test-time compute. The model has the capability but doesn't always activate it. More samples, revision, or search helps significantly.

**Hard problems** (model almost never gets them right):
```
"Prove the Riemann hypothesis for all non-trivial zeros..."

N=1:   0.1% accuracy
N=16:  0.5% accuracy  (still near zero — the model lacks the capability)
N=256: 1.2% accuracy
```
Diminishing returns again, but for a different reason. The model fundamentally lacks the knowledge or reasoning ability. No amount of test-time compute will compensate for a capability that isn't in the weights.

### The Optimal Allocation Strategy

Given a fixed inference compute budget (say, equivalent to 64 total forward passes), how should you allocate it?

Snell et al. proposed **compute-optimal scaling**: adaptively allocate compute per-prompt based on estimated difficulty.

```
Total budget: 64 forward passes across 4 problems

Naive allocation (uniform):
  Problem 1 (easy):   16 passes → 99% (wasted compute)
  Problem 2 (medium): 16 passes → 72%
  Problem 3 (medium): 16 passes → 68%
  Problem 4 (hard):   16 passes → 2%

Compute-optimal allocation:
  Problem 1 (easy):    1 pass  → 98% (save 15 passes)
  Problem 2 (medium): 28 passes → 80%
  Problem 3 (medium): 28 passes → 77%
  Problem 4 (hard):    7 passes → 1.5% (don't waste compute on hopeless cases)
```

The compute-optimal strategy achieves higher aggregate accuracy by redistributing compute from easy (diminishing returns) and hard (near-zero returns) problems toward medium problems (highest marginal returns).

**Result**: Compute-optimal allocation improved efficiency by more than 4x compared to uniform best-of-N sampling on the MATH benchmark.

### How to Estimate Difficulty

In practice, you need a way to estimate problem difficulty before allocating compute:
- **Confidence-based**: Generate one response. If the model's confidence (or self-consistency across a small sample) is high, stop. If low, spend more compute.
- **Adaptive**: Start with a small N. If results are inconsistent, increase N. Stop when answers converge.
- **Learned difficulty predictor**: Train a small classifier to estimate problem difficulty from the prompt alone.

## How o1/o3 and R1 Use Test-Time Compute

These models represent the "internalized search" approach — they spend test-time compute by generating long chain-of-thought reasoning traces rather than external best-of-N or tree search.

### OpenAI o1 / o3

What we know (OpenAI hasn't published full technical details):

- **Training**: RL (likely PPO or a variant) on top of a large base model. The reward signal is answer correctness. The model learns to produce extended reasoning chains that improve final answer accuracy.
- **Inference**: The model generates a long internal chain-of-thought (hidden from the user), then outputs a final answer. The reasoning chain includes self-verification, backtracking, and trying alternative approaches.
- **Scaling**: Performance scales with both training compute (more RL) AND test-time compute (longer reasoning chains). OpenAI showed smooth scaling curves on both axes.
- **o3 efficiency**: o3-mini is 63% cheaper than o1-mini for comparable performance — the model learned to reason more efficiently, not just longer.
- **Benchmark**: o3 scored 87.5% on ARC-AGI, a benchmark for novel reasoning tasks.

### DeepSeek R1

More is known because the paper and weights are public:

- **R1-Zero**: Applied GRPO directly to a base model with only correctness rewards. No SFT, no human demonstrations of reasoning. The model spontaneously learned chain-of-thought, self-verification, and error correction. AIME 2024 accuracy: 15.6% → 71.0%.
- **R1**: Full pipeline — cold-start SFT with a small number of reasoning examples → GRPO with correctness + format rewards → rejection sampling to create better SFT data → final GRPO round. Matched o1 on reasoning benchmarks.
- **Self-evolution**: During training, reasoning chains grew from hundreds of tokens to tens of thousands. The model learned that longer thinking helps on harder problems — an emergent compute-allocation strategy.
- **Test-time behavior**: At inference, R1 produces reasoning inside `<think>...</think>` tags. The length varies with problem difficulty — simple questions get short traces, hard problems get long, multi-attempt traces.

### The Two Scaling Dimensions

Both o1/o3 and R1 scale test-time compute in two ways simultaneously:

```
                Sequential scaling              Parallel scaling
                ─────────────────────           ─────────────────────
Mechanism       Longer chain-of-thought         Generate N solutions,
                within a single generation      pick the best

How it scales   More tokens → more thinking     More samples → higher
                → better accuracy               chance of a correct one

Cost            Linear in chain length          Linear in N

Diminishing     Yes — eventually the model      Yes — O(log N) improvement
returns?        starts repeating itself          per sample

Used by         o1, o3, R1 (primary)            Can be applied on top of
                                                any model (secondary)
```

The most powerful configuration: use a reasoning model (sequential) AND sample multiple reasoning traces (parallel). This combines both scaling dimensions.

## Practical Strategies

### When to Use Test-Time Compute vs a Bigger Model

```
Use test-time compute when:                 Use a bigger model when:
────────────────────────────────           ────────────────────────────────
Task requires reasoning/search              Task requires broad knowledge
Latency is flexible                         Latency must be low
Query volume is low                         Query volume is very high
Problem difficulty varies widely            Most queries are similar difficulty
You can verify correctness                  No good verifier exists
You have a good reward model/verifier       Hard to score outputs automatically
Budget-constrained on training              Budget-constrained on inference
```

**Rule of thumb from Epoch AI**: You can trade ~1 order of magnitude of training compute for ~1 order of magnitude of inference compute without changing performance. So 10x more inference compute ≈ 10x more training compute (roughly).

### Cost-Performance Tradeoffs

Concrete example — solving MATH-500 with different strategies:

```
Strategy                           Accuracy    Relative cost
─────────────────────────────────  ────────    ─────────────
Llama 3.1 8B, single pass         ~45%        1x
Llama 3.1 8B, best-of-64          ~72%        64x
Llama 3.1 8B, compute-optimal     ~75%        64x (but better allocated)
Llama 3.1 70B, single pass        ~68%        ~9x (bigger model)
Llama 3.2 3B, compute-optimal     ~78%        Variable (beats 405B!)
Reasoning model (R1), single pass ~90%+       ~1x (but expensive model)
```

The key insight: compute-optimal test-time scaling on a small model can outperform a much larger model. But a reasoning model that has internalized test-time compute during training is still the most efficient at inference.

### Implementing Basic Test-Time Compute Scaling

**Best-of-N with majority voting** (no reward model needed):

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter
import re

def best_of_n_majority_vote(model, tokenizer, prompt, n=16, temperature=0.7, max_new_tokens=512):
    """Generate N responses and return the most common final answer."""
    answers = []
    for _ in range(n):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
            )
        response = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        answer = extract_final_answer(response)  # task-specific extraction
        if answer is not None:
            answers.append(answer)

    if not answers:
        return None
    # Majority vote
    counter = Counter(answers)
    return counter.most_common(1)[0][0]

def extract_final_answer(response):
    """Extract a numerical answer from a response (task-specific)."""
    # Look for boxed answer (common in math): \boxed{42}
    match = re.search(r'\\boxed\{([^}]+)\}', response)
    if match:
        return match.group(1).strip()
    # Fallback: last number in the response
    numbers = re.findall(r'-?\d+\.?\d*', response)
    return numbers[-1] if numbers else None
```

**Best-of-N with reward model scoring**:

```python
def best_of_n_with_reward_model(model, tokenizer, reward_model, prompt, n=16, temperature=0.7):
    """Generate N responses and return the highest-scoring one."""
    candidates = []
    for _ in range(n):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=512, temperature=temperature, do_sample=True,
            )
        response = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        # Score with reward model
        score = reward_model.score(prompt, response)
        candidates.append((response, score))

    # Return highest-scoring response
    return max(candidates, key=lambda x: x[1])[0]
```

**Adaptive compute allocation** (simple version):

```python
def adaptive_best_of_n(model, tokenizer, prompt, max_n=64, initial_n=4, agreement_threshold=0.75):
    """Start with few samples, increase if answers disagree."""
    all_answers = []
    n = initial_n

    while len(all_answers) < max_n:
        # Generate a batch
        for _ in range(n):
            response = generate_one(model, tokenizer, prompt)
            answer = extract_final_answer(response)
            if answer is not None:
                all_answers.append(answer)

        # Check agreement
        counter = Counter(all_answers)
        top_answer, top_count = counter.most_common(1)[0]
        agreement = top_count / len(all_answers)

        if agreement >= agreement_threshold:
            return top_answer  # confident enough, stop early

        n = min(n * 2, max_n - len(all_answers))  # double the batch size
        if n <= 0:
            break

    # Return majority vote with whatever we have
    counter = Counter(all_answers)
    return counter.most_common(1)[0][0]
```

This adaptive approach implements a simple form of compute-optimal scaling: easy problems (high agreement) stop early with few samples, hard problems use the full budget.

## Key Papers

| Paper | Authors | Year | Key Contribution |
|-------|---------|------|-----------------|
| [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) | Wei et al. (Google) | 2022 | Step-by-step reasoning in LLMs via prompting |
| [Self-Consistency](https://arxiv.org/abs/2203.11171) | Wang et al. (Google) | 2023 | Majority voting over multiple CoT samples |
| [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) | Lightman et al. (OpenAI) | 2023 | Process reward models + PRM800K dataset |
| [Self-Refine](https://arxiv.org/abs/2303.17651) | Madaan et al. (CMU) | 2023 | Iterative self-critique and revision |
| [Scaling LLM Test-Time Compute](https://arxiv.org/abs/2408.03314) | Snell et al. (UC Berkeley) | 2024 | Compute-optimal test-time scaling; small models beat 14x larger ones |
| [ReST-MCTS*](https://arxiv.org/abs/2406.03816) | Zhang et al. | 2024 | MCTS with process reward guidance for reasoning |
| [Learning to Reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/) | OpenAI | 2024 | o1: RL-trained extended chain-of-thought reasoning |
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | DeepSeek | 2025 | Open-weight reasoning via GRPO; emergent CoT from pure RL |
| [The Art of Scaling Test-Time Compute](https://arxiv.org/abs/2512.02008) | Agarwal et al. | 2024 | Large-scale empirical study; no single TTS strategy dominates |
| [Can 1B LLM Surpass 405B?](https://arxiv.org/abs/2502.06703) | Various | 2025 | Rethinking compute-optimal test-time scaling |

## Where Test-Time Compute Fits in Your Training Pipeline

```
Pre-training  →  SFT  →  DPO/GRPO  →  Test-time compute
(base model)     (follow   (align/     (scale inference)
                  instrs)   reason)

You are here ─────────────────────────→ This guide
```

You've already done pre-training (GPT-2 124M), SFT (Dolly/SlimOrca), DPO (hh-rlhf), and understand GRPO. Test-time compute is the final layer — it doesn't change the model weights. It makes any model better at inference by spending more compute on the response.

The connection to your prior work:
- **DPO** trained a reward model implicitly. That implicit reward could be used to score candidates in best-of-N.
- **GRPO** used group sampling and reward functions — best-of-N is essentially the inference-time version of the same idea (sample multiple, keep the best).
- A PRM is like a step-level reward function for GRPO, but used at inference time instead of training time.

The frontier is models like R1 that blur the line — they use RL training (GRPO) to internalize test-time compute strategies into the model's weights, so the model "thinks harder" automatically without external search.
