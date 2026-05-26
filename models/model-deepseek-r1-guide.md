# DeepSeek R1: Emergent Reasoning via Reinforcement Learning

How pure RL — without human demonstrations of reasoning — produces chain-of-thought, self-verification, and error correction in large language models.

## Background

**Paper**: [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) (Guo, Yang, Zhang et al., DeepSeek-AI, January 2025). Published in Nature, Vol. 645, pp. 633-638.

**Research lineage** — DeepSeek R1 builds on a chain of prior work:

1. **Chain-of-thought prompting** (Wei et al., Google, 2022) — Showed that prompting LLMs to "think step by step" dramatically improves reasoning. R1's key question: can a model learn to produce chain-of-thought on its own, without being shown examples?

2. **InstructGPT / RLHF pipeline** (Ouyang et al., OpenAI, 2022) — Established the SFT -> Reward Model -> PPO pipeline. R1 uses a similar multi-stage approach but replaces PPO with GRPO and uses rule-based rewards instead of learned reward models for reasoning tasks.

3. **GRPO** (Shao et al., DeepSeek, 2024) — Group Relative Policy Optimization, which eliminates the critic network from PPO by using group-based advantage estimation. R1 uses GRPO as its RL algorithm throughout. (Covered in your GRPO guide — we won't re-derive it here.)

4. **OpenAI o1** (OpenAI, September 2024) — The first production "reasoning model" that uses extended chain-of-thought at inference time. Demonstrated that test-time compute scaling (thinking longer) can dramatically improve reasoning. R1 is DeepSeek's open-weight answer to o1, achieving comparable performance.

5. **Process Reward Models** (Lightman et al., OpenAI, 2023) — Showed that rewarding individual reasoning steps (not just final answers) improves math performance. R1 takes a different approach: it uses only outcome-based rewards (final answer correctness) but lets the model discover its own reasoning process.

**The key contribution**: DeepSeek R1 proved that **pure reinforcement learning on a base model, with only correctness rewards, can produce emergent reasoning behaviors** — chain-of-thought, self-verification, backtracking, and error correction — without any human demonstrations of how to reason. The model invents reasoning strategies on its own.

This is significant because prior work assumed you needed to show the model examples of step-by-step reasoning (via SFT on human-written chain-of-thought). R1-Zero shows that with the right reward signal and enough scale, RL alone discovers these patterns.

## R1-Zero: RL Without SFT

The most remarkable experiment in the paper. The researchers applied GRPO directly to the DeepSeek-V3-Base model (a 671B MoE pretrained model) with zero supervised fine-tuning. No chain-of-thought examples. No instruction-following data. Just a base model and a reward signal.

### Setup

**RL algorithm**: GRPO (as described in your GRPO guide)

**Reward**: Two components, both rule-based (no neural reward model):
- **Accuracy reward**: Binary (1.0 or 0.0). For math, extract the final answer and compare to ground truth. For code, run against test cases.
- **Format reward**: Requires the model to put its reasoning inside `<think>...</think>` tags and its final answer inside `<answer>...</answer>` tags. Binary — 1.0 if both tags present and properly structured, 0.0 otherwise.

That's it. No reward for reasoning quality, no reward for explanation clarity, no process supervision. Just "did you get the right answer?" and "did you use the right format?"

**Prompt template** (from the paper): The model was given a simple system prompt instructing it to first think about the reasoning process in `<think>` tags, then provide the answer in `<answer>` tags. This template only enforces output structure — it says nothing about how to reason.

### What Emerged

Without being taught, R1-Zero spontaneously developed:

1. **Chain-of-thought reasoning** — The model began producing multi-step reasoning within its `<think>` blocks, breaking problems into sub-problems.

2. **Self-verification** — The model started checking its own work ("Let me verify this..."), going back to confirm intermediate steps.

3. **Error correction and backtracking** — When the model detected an error in its reasoning, it would backtrack and try a different approach.

4. **Dynamic strategy adaptation** — For harder problems, the model allocated more reasoning tokens. It learned to "think longer" on difficult questions without being told to.

5. **Increasing reasoning length** — Over training, the average response length grew steadily from hundreds to thousands of tokens. The model learned on its own that longer, more detailed reasoning produces better answers. This mirrors the test-time compute scaling insight behind o1.

### The "Aha Moment"

The most striking qualitative finding. During training, an intermediate R1-Zero checkpoint produced output containing:

> "Wait, wait. Wait. That's an aha moment I can flag here."

The model was solving a math problem, realized its initial approach was wrong, paused, and re-evaluated — using anthropomorphic language that no one taught it. The researchers describe this as "an aha moment for us, allowing us to witness the power and beauty of reinforcement learning."

This is not the model imitating human chain-of-thought from training data. The base model was never fine-tuned on reasoning demonstrations. The self-reflection behavior emerged purely from the pressure to get correct answers — the model discovered that re-examining its work leads to higher rewards.

Whether the specific phrasing ("aha moment") comes from patterns in the pretraining data is debatable. But the behavioral pattern — stopping mid-reasoning, recognizing an error, and course-correcting — is a genuinely emergent strategy that RL incentivized.

### Results

```
Benchmark          R1-Zero (start)    R1-Zero (end)    R1-Zero (maj@64)
-----------        ---------------    -------------    ----------------
AIME 2024          15.6%              71.0%            86.7%
MATH-500           —                  95.9%            —
GPQA Diamond       —                  73.3%            —
LiveCodeBench      —                  50.0%            —
```

The AIME jump — from 15.6% to 71.0% — is striking. With majority voting over 64 samples (consensus@64), R1-Zero reaches 86.7%, matching OpenAI o1-0912. A base model with no instruction tuning, just RL and correctness rewards, competing with a frontier reasoning model.

### Failure Modes

R1-Zero, for all its emergent reasoning ability, had serious usability problems:

1. **Language mixing** — The model would switch between languages mid-response (e.g., English reasoning with Chinese tokens interspersed, or vice versa). Since the base model was pretrained on multilingual data, RL had no pressure to maintain language consistency — only to get the right answer.

2. **Poor readability** — Reasoning chains were often a stream-of-consciousness dump: no formatting, no structure, hard for humans to follow. The model optimized for correctness, not presentation.

3. **Endless repetition** — Some outputs would loop, repeating similar reasoning steps without converging.

These failures motivated the full R1 pipeline: keep the emergent reasoning, fix the usability.

## R1: The Full Pipeline

R1-Zero proved RL can discover reasoning. But the outputs aren't usable in production. DeepSeek R1 is the productionized version — a 4-stage pipeline that combines the best of SFT and RL.

```
                    Stage 1              Stage 2              Stage 3              Stage 4
                    ───────              ───────              ───────              ───────
Base Model    →    Cold-Start SFT   →   Reasoning RL    →   Rejection Sampling   →   All-Scenario RL
(V3-Base)          (thousands of        (GRPO on math,       + SFT                    (GRPO on all
                    CoT examples)        code, logic)         (~800k samples)          task types)
                                                                                        ↓
                   Fixes readability    Builds reasoning     Adds general             Aligns with
                   and formatting       capability           capabilities             human preferences
```

### Stage 1: Cold-Start SFT

**Problem**: R1-Zero's outputs are unreadable. If you start RL from the raw base model, you get powerful reasoning buried in messy, language-mixed text.

**Solution**: Fine-tune V3-Base on a small set of high-quality reasoning examples before starting RL. This gives the model a "warm start" — it already knows how to format its thinking clearly.

**Data**: Thousands (not millions) of carefully curated chain-of-thought examples, collected from multiple sources:
- Long CoT outputs from R1-Zero, cleaned up and reformatted by humans
- Few-shot prompted outputs from V3
- Direct prompting with reflection verification
- Some human-annotated reasoning traces

**Output format**: The model learns to produce `|special_token|<reasoning_process>|special_token|<summary>` — structured reasoning followed by a clean summary answer.

**Why "cold start" matters**: Using a small, high-quality dataset rather than a massive one is deliberate. Too much SFT data would constrain the model's reasoning style — you'd be teaching it to imitate specific reasoning patterns rather than letting RL discover better ones. The cold start just teaches formatting and basic structure, leaving room for RL to innovate.

### Stage 2: Reasoning-Oriented RL

**Problem**: The cold-start model can format its thinking, but hasn't developed strong reasoning yet.

**Solution**: Apply GRPO on reasoning tasks — math, code, logic, science. Same approach as R1-Zero, but starting from a better-formatted model.

**Rewards**: Same accuracy + format rewards as R1-Zero, plus one addition:
- **Language consistency reward** — Penalizes language mixing within the chain-of-thought. Computed as the proportion of tokens matching the prompt's language. This directly addresses R1-Zero's biggest usability failure.

**Tasks**: All tasks in this stage have **verifiable answers** — you can programmatically check correctness. Math problems with numerical answers, coding problems with test cases, logic puzzles with definitive solutions.

**Result**: The model develops strong reasoning capabilities (like R1-Zero) but with clean, readable, language-consistent outputs.

### Stage 3: Rejection Sampling + SFT

**Problem**: The Stage 2 model reasons well on math/code/logic but can't do general tasks — writing, translation, open-ended QA, summarization.

**Solution**: Use the reasoning-capable model to generate a large supervised dataset, then fine-tune on it alongside general-purpose data.

**Process**:
1. **Reasoning data (~600k samples)**: Take the Stage 2 checkpoint. Generate multiple responses per prompt (rejection sampling). Keep only the ones that pass correctness verification. This creates a large corpus of high-quality reasoning chains.
2. **General data (~200k samples)**: Reuse general-purpose SFT data from the DeepSeek-V3 pipeline — writing, factual QA, translation, conversation, etc. These are non-reasoning tasks where the model needs broad capabilities.
3. **Combined SFT**: Fine-tune on the full ~800k samples for 2 epochs.

**Why rejection sampling?** Not all model outputs are correct. By generating many candidates and filtering to only correct ones, you create a training set where every reasoning chain leads to the right answer. This is more efficient than naive SFT on human-written data because the reasoning style matches what the model naturally produces.

**Why add general data?** RL on reasoning tasks creates a specialist. To make a useful general-purpose model, you need to restore (or add) capabilities on non-reasoning tasks. The ~200k general samples ensure the model can still write, translate, and have normal conversations.

### Stage 4: All-Scenario RL

**Problem**: The Stage 3 model is capable but not fully aligned with human preferences on subjective tasks.

**Solution**: A second round of RL, this time covering all task types — not just reasoning.

**Reward design** (varies by task type):
- **Reasoning tasks** (math, code, logic): Same rule-based accuracy + format rewards as Stage 2. No neural reward model — avoids reward hacking.
- **General tasks** (writing, QA, conversation): Preference-based reward model trained on human judgments. This is where traditional RLHF-style alignment happens.

**Helpfulness evaluation trick**: For reasoning tasks, the reward model only evaluates the final summary/answer, not the full chain-of-thought. This prevents the reward model from penalizing useful-but-verbose reasoning chains.

**Harmlessness**: Evaluated on the full response (including reasoning), since safety issues can appear anywhere.

**Result**: The final R1 model — strong reasoning from RL, broad capabilities from SFT, aligned with human preferences from the final RL stage.

## Reward Design: The Details

One of R1's most important design decisions: **no neural reward model for reasoning tasks**. This is a deliberate departure from the standard RLHF playbook.

### Why Rule-Based Rewards?

Neural reward models are trained on human preference data. They work well for subjective tasks (is this response helpful? safe? well-written?) but have a critical failure mode for reasoning: **reward hacking**.

A neural reward model might learn surface-level correlations — "responses that look confident get higher scores" or "longer responses score higher" — without actually checking correctness. The policy can then exploit these shortcuts, producing confident-sounding wrong answers that fool the reward model.

Rule-based rewards are unhackable for verifiable tasks. The answer is either right or wrong. There's no shortcut.

### Reward Types in R1

| Reward | How it works | When used |
|--------|-------------|-----------|
| **Accuracy** | Extract final answer, compare to ground truth. Math: parse boxed answer, compare numerically. Code: compile and run against test cases. | Stages 2 & 4 (reasoning tasks) |
| **Format** | Check for `<think>...</think>` and `<answer>...</answer>` tags. Binary. | Stages 2 & 4 |
| **Language consistency** | Proportion of CoT tokens matching prompt language. Penalizes mixing. | Stage 2 |
| **Preference reward model** | Neural model trained on human preference judgments. | Stage 4 (general tasks only) |

### What's NOT Rewarded

- **Reasoning quality** — No process reward. The model is free to reason however it wants as long as it gets the right answer. This is what enables emergent strategies.
- **Reasoning length** — No penalty or reward for chain-of-thought length. The model learns to calibrate length to difficulty on its own.
- **Specific reasoning patterns** — No reward for "showing work" or using particular problem-solving heuristics. The model discovers these from scratch.

## R1 vs o1: Performance Comparison

DeepSeek R1 matches or exceeds OpenAI o1-1217 across most reasoning benchmarks:

```
Benchmark              DeepSeek R1    OpenAI o1-1217    o1-mini
─────────────────────  ───────────    ──────────────    ───────
AIME 2024 (pass@1)     79.8%          79.2%             63.6%
MATH-500               97.3%          96.4%             90.0%
GPQA Diamond           71.5%          75.7%             60.0%
Codeforces (rating)    2029           2061              1650
LiveCodeBench          65.9%          63.4%             —
MMLU                   90.8%          91.8%             85.2%
SWE-bench Verified     49.2%          48.9%             —
AlpacaEval 2.0         87.6%          —                 —
```

**Key takeaway**: R1 is competitive with o1 across the board. It edges ahead on AIME, MATH, LiveCodeBench, and SWE-bench, while o1 leads slightly on GPQA Diamond and MMLU. The differences are within noise for most benchmarks.

**Approach differences**:

| | DeepSeek R1 | OpenAI o1 |
|--|------------|-----------|
| Base model | DeepSeek-V3 (671B MoE, open-weight) | Unknown (closed) |
| RL algorithm | GRPO | Undisclosed (likely PPO variant) |
| Reward for reasoning | Rule-based (correctness only) | Undisclosed (likely process reward model) |
| Weights | Open (Apache 2.0) | Closed |
| API cost | ~$0.14/M input tokens | ~$15/M input tokens |
| Reasoning trace | Visible `<think>` blocks | Hidden |

R1's open-weight, open-method approach at ~100x lower API cost was a significant event in the AI industry.

## Distillation: R1's Knowledge in Smaller Models

The paper demonstrates that R1's reasoning capabilities can be distilled into much smaller models via straightforward SFT — no RL needed on the student models.

### Process

1. Take the R1 model (Stage 3 checkpoint, after rejection sampling + SFT)
2. Generate reasoning chains for ~800k diverse prompts
3. Filter to only correct, high-quality outputs
4. Fine-tune smaller base models (Qwen2.5 and Llama-3 families) on this data using standard SFT

The student models learn to imitate R1's reasoning patterns — the extended chain-of-thought, the self-verification, the structured problem decomposition — just from supervised training on R1's outputs.

### Results

```
Model                          AIME 2024    MATH-500    GPQA Diamond    LiveCodeBench
─────────────────────────────  ─────────    ────────    ────────────    ─────────────
DeepSeek-R1 (full)             79.8%        97.3%       71.5%           65.9%
─────────────────────────────  ─────────    ────────    ────────────    ─────────────
R1-Distill-Qwen-1.5B          28.9%        83.9%       —               —
R1-Distill-Qwen-7B            55.5%        92.8%       49.1%           37.6%
R1-Distill-Llama-8B           50.4%        89.1%       49.0%           39.6%
R1-Distill-Qwen-14B           69.7%        93.9%       59.1%           53.1%
R1-Distill-Qwen-32B           72.6%        94.3%       62.1%           57.2%
R1-Distill-Llama-70B          70.0%        94.5%       65.2%           57.5%
─────────────────────────────  ─────────    ────────    ────────────    ─────────────
OpenAI o1-mini                 63.6%        90.0%       60.0%           —
QwQ-32B-Preview                50.0%        90.6%       54.5%           41.9%
```

**Standout results**:
- **R1-Distill-Qwen-7B** (55.5% AIME) surpasses QwQ-32B-Preview (50.0% AIME) — a 7B model beating a 32B model, because it learned from R1's reasoning.
- **R1-Distill-Qwen-32B** (72.6% AIME) surpasses o1-mini (63.6% AIME) — an open-weight 32B model beating OpenAI's smaller reasoning model.
- Even the **1.5B model** gets 28.9% on AIME and 83.9% on MATH — substantially better than GPT-4o on these math benchmarks.

### Distillation vs RL from Scratch

The paper ran a direct comparison: train Qwen-32B-Base with GRPO from scratch (no distillation) for >10,000 RL steps. The result — DeepSeek-R1-Zero-Qwen-32B — underperformed the distilled version (R1-Distill-Qwen-32B) across all benchmarks.

**The lesson**: For smaller models, it's more effective to distill from a strong reasoning teacher than to try to discover reasoning via RL. RL-discovered reasoning requires massive scale (671B parameters in R1's case). At smaller scales, the model doesn't have enough capacity to independently discover sophisticated reasoning strategies through RL — but it can learn to imitate them from a larger model's demonstrations.

This is the same principle behind SFT generally: it's easier to learn from examples than to discover from scratch. RL discovery requires scale; distillation democratizes the result.

## Key Lessons for Practitioners

**1. RL can discover capabilities, not just refine them.** Before R1, the conventional wisdom was that RL/RLHF was for alignment — nudging a model's behavior after SFT taught it what to do. R1-Zero shows RL can teach entirely new capabilities (chain-of-thought reasoning) that weren't in the training data.

**2. Simple rewards beat complex ones.** R1 uses binary correctness rewards — no process reward models, no per-step scoring. The simplicity prevents reward hacking and lets the model discover its own problem-solving strategies. If you can verify the answer, you don't need to score the process.

**3. Verifiable tasks are the sweet spot for RL.** Math, code, logic — anywhere you can programmatically check correctness. RL shines here because you can generate unlimited training signal without human annotation. For subjective tasks, you still need preference data and reward models.

**4. Cold-start SFT is a pragmatic compromise.** Pure RL works (R1-Zero proves it) but produces unreadable outputs. A small amount of SFT on formatting/structure, before RL, gives you the best of both worlds: emergent reasoning with clean presentation.

**5. Distillation is remarkably effective.** You don't need massive compute to get R1-level reasoning in a smaller model. Fine-tune on R1's outputs. A 7B distilled model beats a 32B model trained from scratch. If you have access to a strong reasoning model's outputs, distillation is the most cost-effective path.

**6. Scale matters for RL discovery.** R1-Zero's emergent reasoning required a 671B parameter base model. The same RL approach on a 32B model produces weaker results. Discovery through RL seems to require a critical mass of model capacity. Below that threshold, distillation is the better strategy.

**7. Multi-stage pipelines are the production playbook.** No single training stage produces a complete model. SFT for formatting, RL for reasoning, rejection sampling for data quality, RL again for alignment — each stage solves a specific problem. This mirrors the InstructGPT pipeline but with more stages and rule-based rewards for reasoning.

## Implications for the Field

**Emergent capabilities via RL are real.** R1-Zero is arguably the strongest evidence that RL can produce qualitatively new behaviors in LLMs — not just refine existing ones. The model learned to reason, self-verify, and backtrack without any demonstrations. This suggests there are likely other capabilities waiting to be "unlocked" by the right reward signal.

**The SFT ceiling might not exist.** Prior to R1, there was a common belief that LLMs were fundamentally limited by their training data — they could only remix patterns they'd seen. R1-Zero challenges this: with RL and verifiable rewards, a model can discover strategies not present in its pretraining corpus.

**Open models caught up on reasoning.** Before R1, reasoning was OpenAI's moat (o1). R1 matched o1's performance with open weights, open methods, and dramatically lower cost. This shifted the competitive dynamics significantly, particularly since R1's approach (GRPO + rule-based rewards) is reproducible by any well-resourced lab.

**Distillation democratizes reasoning.** The open release of R1's distilled models (1.5B to 70B) means reasoning capabilities are now accessible to anyone who can run a 7B model. A 7B model on consumer hardware can now outperform much larger models on math reasoning. This compresses the capability gap between frontier labs and the broader community.

**The training paradigm is solidifying.** The field is converging on: Pretrain -> SFT (small, high-quality) -> RL with verifiable rewards (reasoning) -> RL with preference models (alignment). R1 validates this pipeline at scale. The remaining open questions are about reward design, scaling laws for RL, and how far emergent capabilities can go.
