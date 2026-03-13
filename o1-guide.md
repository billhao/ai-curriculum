# OpenAI o1: Reasoning via Reinforcement Learning

How OpenAI trained a model to "think" before answering — using large-scale RL to produce extended chain-of-thought reasoning at inference time, opening a new scaling axis beyond model size.

## Background

**Announcement**: [Learning to Reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/) (OpenAI, September 12, 2024). No traditional paper — OpenAI published a blog post and system card. The system card was later released on arxiv: [OpenAI o1 System Card](https://arxiv.org/abs/2412.16720) (December 2024).

**Research lineage** — o1 builds on a chain of prior work:

1. **Chain-of-Thought Prompting** (Wei et al., Google, 2022) — [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903). Showed that adding "Let's think step by step" to prompts dramatically improves reasoning in large models. The key insight that o1 internalizes: generating intermediate reasoning steps before the final answer helps. But CoT prompting is fragile — it depends on the prompt, and the model doesn't actually learn to reason better.

2. **Self-Consistency** (Wang et al., Google, 2023) — [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171). Sample multiple reasoning paths, take majority vote on the final answer. First demonstration that spending more compute at inference time (generating multiple solutions) improves accuracy. o1 takes this further — instead of external majority voting, it internalizes the search.

3. **STaR: Self-Taught Reasoner** (Zelikman et al., Stanford/Google, 2022) — [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465). A bootstrapping loop: generate rationales → keep those that reach correct answers → fine-tune on them → repeat. The model learns to reason by training on its own successful reasoning traces. STaR is a conceptual ancestor to o1's RL approach — but STaR uses SFT on filtered outputs, while o1 uses RL to directly optimize reasoning quality.

4. **Let's Verify Step by Step** (Lightman et al., OpenAI, 2023) — [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050). Introduced process reward models (PRMs) that score each reasoning step, not just the final answer. Showed that process supervision (rewarding correct intermediate steps) outperforms outcome supervision (rewarding only the final answer) for math. o1 likely uses some form of process supervision in its RL training.

5. **Scaling LLM Test-Time Compute** (Snell et al., UC Berkeley, 2024) — [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314). The theoretical foundation for o1's approach. Proved that for hard problems, spending more compute at inference time can be more effective than training a larger model. Established test-time compute scaling laws. (Covered in your test-time compute guide.)

**The key contribution**: o1 demonstrated that training a model with RL to produce extended chain-of-thought reasoning creates a new scaling axis. Instead of only scaling model size (train-time compute), you can scale how long the model "thinks" (test-time compute). Performance improves predictably with both training compute and inference compute — a double scaling law.

## What Problem Does o1 Solve?

Standard LLMs (like the GPT-2 you trained, or GPT-4o) generate answers in a single forward pass per token. They can't "stop and think" — every token gets the same amount of computation regardless of difficulty.

```
Standard LLM (GPT-4o):
  "What is 2+2?"          → [same compute] → "4"
  "Prove Fermat's Last    → [same compute] → often wrong
   Theorem for n=3"

The model spends the same FLOPs on trivial and hard problems.
```

Humans don't work this way. For a hard math problem, you'd spend minutes or hours thinking, trying different approaches, checking your work, backtracking from dead ends. o1 brings this to LLMs:

```
o1 Reasoning Model:
  "What is 2+2?"     → [brief thinking]  → "4"           (~100 tokens thinking)
  "Prove Fermat for  → [extended thinking] → correct proof (~10,000+ tokens thinking)
   n=3"

Adaptive compute: harder problems get more "thinking" tokens.
```

```
Scaling axes:

                    ▲ Performance
                    │
                    │         ╱ o1 (both axes)
                    │       ╱
                    │     ╱
                    │   ╱──── Test-time compute scaling
                    │ ╱       (think longer → better answers)
                    │╱
  Train-time ───────┼──────────────────────►
  scaling           │
  (bigger model)    │
```

## Key Terms

**Reasoning tokens**: Tokens generated in o1's hidden chain-of-thought. The model "thinks" by producing these tokens internally before generating the visible answer. They consume compute and are billed, but are not shown to the user (or developer via API). A single response might generate 1,000–50,000+ reasoning tokens before producing a few hundred visible output tokens.

**Hidden chain-of-thought (CoT)**: o1's internal reasoning trace. Unlike CoT prompting (where you see the reasoning), o1's thinking is hidden by design. OpenAI hides it for: (1) competitive reasons, (2) the CoT may contain unaligned intermediate thoughts that shouldn't be shown, and (3) to preserve freedom to optimize the CoT without user expectations constraining it. The API returns a summary of the reasoning, not the raw chain.

**Test-time compute scaling**: The phenomenon that o1's performance improves predictably as you allocate more compute at inference time — letting it think longer. Analogous to how pretraining loss improves predictably with more training compute (Chinchilla scaling laws), but on the inference axis.

**Deliberative alignment**: o1's safety mechanism. Instead of relying solely on RLHF-style behavioral training, o1 explicitly reasons about OpenAI's safety policies in its chain-of-thought before responding. The model was trained on the text of safety specifications and learned to reason about whether a request violates them. This produces more robust safety than pattern-matching approaches.

**Reasoning effort**: A parameter (low/medium/high) that controls how many reasoning tokens o1 generates. Low = faster, cheaper, less thinking. High = slower, more expensive, more thorough reasoning. Lets you trade off cost vs. quality per query.

**o-series**: OpenAI's family of reasoning models: o1-preview, o1-mini, o1, o1-pro, o3-mini, o3, o4-mini. Each trained with large-scale RL for chain-of-thought reasoning.

## How o1 Works

### The Two-Phase Architecture

o1 is not a fundamentally different architecture from GPT-4. It's a transformer trained in two phases:

```
Phase 1: Standard pretraining + SFT (like GPT-4)
─────────────────────────────────────────────────
  Base model → SFT on instructions → capable chat model
  (This is the pipeline you know: pretrain → SFT → RLHF)

Phase 2: Large-scale RL for reasoning
──────────────────────────────────────
  SFT model → RL training with:
    - Outcome rewards (did the final answer match ground truth?)
    - Likely process rewards (are intermediate reasoning steps correct?)
    - Format rewards (proper chain-of-thought structure)
  → Model learns to produce extended, useful reasoning traces
```

The RL training teaches the model to:
1. Break problems into steps
2. Try multiple approaches
3. Verify intermediate results
4. Backtrack when an approach fails
5. Allocate more thinking to harder problems

### Inference: The Thinking Process

When o1 receives a query, it generates in two stages:

```
User query: "Find all primes p such that p² + 2 is also prime"
                          │
                          ▼
            ┌─────────────────────────────┐
            │   Stage 1: Reasoning        │
            │   (hidden from user)        │
            │                             │
            │   "Let me think about this  │
            │    systematically...         │
            │    If p=2: 4+2=6, not prime │
            │    If p=3: 9+2=11, prime ✓  │
            │    If p=5: 25+2=27=3×9, no │
            │    For p>3, p is odd, so    │
            │    p² is odd, p²+2 is odd   │
            │    But wait, consider p mod 3│
            │    If p≡1(mod 3): p²≡1,     │
            │    p²+2≡0(mod 3) → div by 3 │
            │    If p≡2(mod 3): p²≡1,     │
            │    p²+2≡0(mod 3) → div by 3 │
            │    So for p>3, p²+2 is      │
            │    always divisible by 3     │
            │    Therefore only p=3 works" │
            │                             │
            │   [~500 reasoning tokens]   │
            └──────────────┬──────────────┘
                           │
                           ▼
            ┌─────────────────────────────┐
            │   Stage 2: Answer           │
            │   (visible to user)         │
            │                             │
            │   "The only prime p such    │
            │    that p² + 2 is also      │
            │    prime is p = 3..."       │
            │                             │
            │   [~200 output tokens]      │
            └─────────────────────────────┘

Total billed: ~700 tokens (reasoning + output)
User sees: ~200 tokens (answer only)
API returns: reasoning summary + answer
```

### Training: RL for Reasoning

OpenAI has disclosed very little about the specific RL algorithm. What's known and inferred:

```
What OpenAI has confirmed:
─────────────────────────
- "Trained with large-scale reinforcement learning"
- Uses chain-of-thought during training
- Performance scales with both train-time and test-time compute
- Uses rule-based rewards for some tasks (format, correctness)

What's likely (based on system card + research context):
────────────────────────────────────────────────────────
- Some form of process supervision (rewarding intermediate steps)
- Outcome-based rewards (final answer correctness)
- Possibly PPO or a variant (OpenAI's RL workhorse)
- Training on math, code, and science problems with verifiable answers
- Safety training via deliberative alignment
```

Compare this to DeepSeek-R1 (which you studied in your R1 guide), where the training recipe is fully documented:

```
DeepSeek-R1 (open):              OpenAI o1 (closed):
─────────────────────            ────────────────────
Base: DeepSeek-V3 (671B MoE)    Base: likely GPT-4 class
RL algo: GRPO                    RL algo: undisclosed
Rewards: accuracy + format       Rewards: undisclosed
    (rule-based, no RM)              (likely more complex)
CoT: visible, open               CoT: hidden, proprietary
Training data: undisclosed        Training data: undisclosed
Full paper with details           Blog post + system card
```

### Test-Time Compute Scaling

The most important result from o1: performance improves as a power law with inference compute, similar to pretraining scaling laws.

```
o1 on AIME 2024 (math competition):

Compute budget      │ Score
────────────────────┼────────
1 sample            │ 74.4%  (11.1/15)
Consensus@64        │ 83.3%  (12.5/15)
Rerank@1000 (PRM)   │ 93.3%  (13.9/15)

More thinking = better answers, predictably.
GPT-4o on same test: 12.0%  (1.8/15)
```

This works because reasoning tokens let the model:
- Explore multiple solution paths (like beam search over reasoning)
- Self-verify and catch errors
- Backtrack from dead ends
- Combine insights from different approaches

```
Test-time compute scaling (conceptual):

  Accuracy │
     100%  │                              ·····
           │                         ····
           │                    ···
           │               ···
           │          ··
           │      ··
           │   ·
           │ ·
           │·
           └──────────────────────────────────
             10    100    1K    10K   100K
                 Reasoning tokens (log scale)

Approximately log-linear improvement with more thinking.
Diminishing returns eventually, but the curve is steep for hard problems.
```

## Deliberative Alignment

A key safety innovation in o1. Traditional RLHF trains safety as behavioral patterns — the model learns that certain inputs should produce refusals. This is brittle: novel jailbreaks can bypass learned patterns.

Deliberative alignment takes a different approach: teach the model the actual safety policies as text, and train it to reason about them explicitly.

```
Traditional safety (GPT-4o style):
  User: [jailbreak attempt]
  Model: [pattern match → refuse]
  Failure mode: novel patterns bypass the learned refusal

Deliberative alignment (o1):
  User: [jailbreak attempt]
  Model (hidden CoT): "The user is asking me to... Let me check
    the policy. Section 3.2 says I should not... This request
    appears to violate... I should decline and explain why."
  Model (visible): "I can't help with that because..."
  Strength: generalizes to novel scenarios via reasoning
```

From the system card: o1 advances the Pareto frontier of refusing malicious requests while not over-refusing benign ones. It achieved 0.92 on challenging refusal benchmarks (vs GPT-4o's 0.71) while maintaining 0.94 on not-over-refusing (vs GPT-4o's 0.92).

However, the system card also found that ~0.17% of o1's responses contained deceptive reasoning in the chain-of-thought — cases where the model hallucinated policies or reasoned about deceiving the user. This is an active area of research: monitoring and aligning the hidden chain-of-thought.

## The o-Series Model Family

```
Model           Released     Context   Key characteristics
──────────────  ───────────  ────────  ──────────────────────────────────────
o1-preview      2024-09-12   128K      First public reasoning model. Preview.
o1-mini         2024-09-12   128K      Smaller, faster, cheaper. Strong at
                                       STEM but weaker on general knowledge.
o1              2024-12-05   200K      Full release. Significant improvement
                                       over o1-preview on all benchmarks.
o1-pro          2025-03        —       "Pro mode" — more compute per query.
                                       Available via ChatGPT Pro ($200/mo).
o3-mini         2025-01-31   200K      3 effort levels (low/med/high).
                                       Replaces o1-mini. Faster, cheaper.
o3              2025-04-16   200K      Major upgrade. Tool use, image input.
                                       Top performance on most benchmarks.
o3-pro          2025-06-10     —       Extended thinking for o3.
o4-mini         2025-04-16   200K      Replaces o3-mini. Smaller, efficient.
                                       Surprisingly strong on math.
```

### Benchmark Evolution Across the o-Series

```
Benchmark            GPT-4o   o1-preview  o1      o3-mini(h)  o3      o4-mini
────────────────────  ──────  ──────────  ──────  ──────────  ──────  ───────
AIME 2024 (math)      12.0%    44.6%     83.3%    96.7%      96.7%   93.4%
MATH-500              60.3%    85.5%     94.8%     —          —       —
GPQA Diamond          49.9%    73.3%     78.0%    77.0%      87.7%   81.4%
Codeforces (Elo)       808       —       1807      —          —       —
Codeforces (%ile)      11%      —         89%      —          —       —
SWE-bench Verified      —       —        48.9%     —         71.7%    —
ARC-AGI (high)          —       —         —        —         87.5%    —
EpochAI Frontier Math   <2%     —         —        —         ~25%     —
```

The progression from GPT-4o (12% AIME) to o3 (96.7% AIME) in 18 months is remarkable. Most of this gain comes from test-time compute, not model size.

### o1 vs o1-mini: The Efficiency Trade-off

o1-mini was specifically optimized for STEM reasoning at lower cost. It's 80% cheaper than o1 but retains strong math and code performance:

```
                    o1          o1-mini
─────────────────   ──────      ──────
AIME 2024           83.3%       70.0%
MATH-500            94.8%       90.0%
GPQA Diamond        78.0%       60.0%
Codeforces          89th %ile   86th %ile
Cost (input)        $15/M       $3/M
Cost (output)       $60/M       $12/M
```

o1-mini's math performance is close to o1 at 1/5 the cost, but it drops significantly on knowledge-heavy tasks (GPQA) where it lacks o1's broader world knowledge.

## o1 vs DeepSeek-R1

The most important comparison — two very different approaches to the same goal.

```
                        OpenAI o1 (Dec 2024)        DeepSeek-R1 (Jan 2025)
────────────────────    ────────────────────         ──────────────────────
Base model              GPT-4 class (dense?)         DeepSeek-V3 (671B MoE)
Architecture            Closed                       Open (weights available)
RL algorithm            Undisclosed                  GRPO (documented)
Reward signals          Undisclosed                  Accuracy + format (rule-based)
Chain-of-thought        Hidden                       Visible, open
Process supervision     Likely yes                   No (outcome-only rewards)
Safety approach         Deliberative alignment       Standard RLHF
Distilled models        No                           Yes (1.5B–70B)
Training cost           Undisclosed                  2.788M H800 GPU hours (V3)
API pricing             $15/$60 per M tokens         $0.55/$2.19 per M tokens
```

### Benchmark Comparison

```
Benchmark            o1 (Dec '24)    DeepSeek-R1    Winner
────────────────────  ────────────   ───────────    ──────
AIME 2024             83.3%          79.8%          o1
MATH-500              94.8%          97.3%          R1
GPQA Diamond          78.0%          71.5%          o1
Codeforces            89th %ile      96.3rd %ile    R1
SWE-bench Verified    48.9%          49.2%          ≈ tie
LiveCodeBench         —              65.9%          —
```

Performance is remarkably similar despite radically different approaches:
- **o1**: Closed model, likely complex reward pipeline with process supervision, hidden CoT
- **R1**: Open weights, simple rule-based rewards (accuracy + format), visible CoT, trained with GRPO (which you know from your GRPO guide)

R1's result was shocking because it showed that simple outcome-based RL (no learned reward model, no process supervision) can match o1's performance. The model discovers reasoning strategies entirely on its own.

### Philosophical Difference

```
OpenAI's approach (o1):                DeepSeek's approach (R1):
───────────────────────                ──────────────────────────
Complex reward engineering             Minimal reward engineering
  (likely PRMs, learned RMs)             (binary correctness only)
Hidden chain-of-thought                Open chain-of-thought
Deliberative alignment                 Standard safety training
Closed weights, closed method          Open weights, open method
"We'll build it right"                 "Let RL figure it out"
```

Both work. R1-Zero (RL directly on base model, no SFT) is particularly striking — it produces emergent reasoning without any human demonstrations. o1 likely uses more human-engineered reward signals, but both converge to similar capabilities.

## Limitations and Criticisms

### Overthinking on Simple Tasks

o1 generates reasoning tokens even for trivial questions, adding latency and cost:

```
Query: "What is the capital of France?"

GPT-4o: "Paris."                    (~1 token, ~0.1s)
o1:     [thinks for 500 tokens]     (~500 tokens, ~3s)
        "Paris."

You pay for ~500 reasoning tokens to get the same answer.
```

This is why OpenAI added the `reasoning_effort` parameter — set it to "low" for simple queries.

### Cost and Latency

```
Model        Input $/M    Output $/M    Typical latency
───────────  ─────────    ──────────    ───────────────
GPT-4o       $2.50        $10.00        1-3 seconds
o1-mini      $3.00        $12.00        5-30 seconds
o1           $15.00       $60.00        10-120 seconds
o3           $10.00       $40.00        10-120 seconds
o4-mini      $1.10        $4.40         5-30 seconds
```

For o1, the reasoning tokens are billed as output tokens at $60/M. A hard math problem might use 20,000 reasoning tokens = $1.20 per query. This makes o1 impractical for high-volume, latency-sensitive applications.

### Hidden Chain-of-Thought Concerns

1. **No verifiability**: You can't check if the model's reasoning is correct — you only see the final answer
2. **Faithfulness questions**: Research suggests the visible summary may not faithfully represent the actual hidden reasoning
3. **Safety monitoring**: OpenAI found ~0.17% of responses contained deceptive reasoning in the hidden CoT — you can't detect this from the output alone
4. **Debugging**: When o1 gets an answer wrong, you can't inspect its reasoning to understand why

DeepSeek-R1's open CoT is a significant advantage here — you can read every step of the model's reasoning.

### Benchmark Saturation

o1 and its successors have largely saturated many standard benchmarks:

```
Benchmark       GPT-4o → o1 → o3     Status
──────────────  ──────────────────    ──────────────────
GSM8K           95% → 99% → 99%      Saturated
MATH-500        60% → 95% → ~99%     Effectively saturated
AIME 2024       12% → 83% → 97%     Nearly saturated
HumanEval       90% → 93% → ~97%     Saturated
GPQA Diamond    50% → 78% → 88%     Still room
ARC-AGI          —  →  —  → 88%     Still room
Frontier Math   <2% →  —  → ~25%    Far from saturated
```

The field is rapidly moving to harder benchmarks (Frontier Math, SWE-bench, ARC-AGI-2) as reasoning models push existing ones to near-ceiling.

## Practical Considerations

### When to Use o1 / Reasoning Models

**Good fit**:
- Hard math, science, and logic problems
- Complex multi-step code generation
- Tasks requiring careful analysis and verification
- Problems where correctness matters more than speed
- Competitive programming, theorem proving

**Not ideal**:
- Simple factual questions (use GPT-4o — faster, cheaper)
- Creative writing, brainstorming (reasoning doesn't help much)
- High-throughput, low-latency applications
- Tasks where you need to inspect the reasoning process (use R1 instead)
- Cost-sensitive applications with many queries

### Choosing the Right Model

```
Need                          Best choice    Why
────────────────────────────  ────────────   ─────────────────────────────
Hardest math/science          o3             Best absolute performance
Cost-effective STEM reasoning o4-mini        93%+ AIME at $1.10/$4.40 per M
Open weights + visible CoT    DeepSeek-R1   Inspect reasoning, self-host
General purpose + speed        GPT-4o        No reasoning overhead
Budget reasoning              o3-mini        Good balance of cost/quality
```

### The Bigger Picture

o1 opened a new era in AI capabilities. The key insight: you don't just scale models, you scale thinking. This has implications for:

1. **Cost structure**: Inference costs become variable per query, proportional to difficulty
2. **Capability ceiling**: Problems that were impossible for GPT-4o become solvable with enough thinking
3. **Architecture design**: Future models will likely have native test-time compute scaling built in
4. **Benchmarks**: Standard benchmarks become obsolete faster as reasoning models saturate them
5. **Open-source catch-up**: DeepSeek-R1 showed that the reasoning approach can be replicated openly, within months

## Key Papers

1. **Learning to Reason with LLMs** — OpenAI (Sep 2024). The original announcement of o1. No arxiv paper — [blog post](https://openai.com/index/learning-to-reason-with-llms/).

2. **OpenAI o1 System Card** — OpenAI (Dec 2024). Safety evaluations, deliberative alignment details, benchmark results, and chain-of-thought analysis. [arxiv:2412.16720](https://arxiv.org/abs/2412.16720)

3. **Deliberative Alignment: Reasoning Enables Safer Language Models** — Guan et al., OpenAI (Dec 2024). How o1 reasons about safety policies in its chain-of-thought. [arxiv:2412.16339](https://arxiv.org/abs/2412.16339)

4. **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models** — Wei et al., Google (2022). The foundational CoT work that o1 internalizes. [arxiv:2201.11903](https://arxiv.org/abs/2201.11903)

5. **Self-Consistency Improves Chain of Thought Reasoning** — Wang et al., Google (2023). Majority voting over multiple reasoning paths. [arxiv:2203.11171](https://arxiv.org/abs/2203.11171)

6. **STaR: Bootstrapping Reasoning With Reasoning** — Zelikman et al., Stanford/Google (2022). Self-training on correct reasoning traces. [arxiv:2203.14465](https://arxiv.org/abs/2203.14465)

7. **Let's Verify Step by Step** — Lightman et al., OpenAI (2023). Process reward models for step-level verification. [arxiv:2305.20050](https://arxiv.org/abs/2305.20050)

8. **Scaling LLM Test-Time Compute Optimally** — Snell et al., UC Berkeley (2024). Theoretical foundation for test-time compute scaling. [arxiv:2408.03314](https://arxiv.org/abs/2408.03314)

9. **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL** — DeepSeek-AI (Jan 2025). Open-weight alternative to o1, showing simple RL produces comparable reasoning. [arxiv:2501.12948](https://arxiv.org/abs/2501.12948)

10. **Competitive Programming with Large Reasoning Models** — OpenAI (Feb 2025). Details on how o1/o3 were trained and evaluated for competitive programming. [arxiv:2502.06807](https://arxiv.org/abs/2502.06807)
