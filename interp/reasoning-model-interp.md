# Mechanistic Interpretability of Reasoning Models

What's inside R1-Zero, o1, and their distilled descendants when they produce the "Wait, wait. Wait. Let me reconsider..." aha moment? Survey of the field as of May 2026.

## The question

DeepSeek-R1-Zero emits text like *"Wait, wait. Wait. That's an aha moment I can flag here. Let me reconsider..."* mid-chain-of-thought. Two sub-questions:

1. **What internal mechanism produces this behavior?** Specific neurons, SAE features, attention heads, or circuits?
2. **Is the mechanism behavior-specific or task-general?** Same circuit across math, code, logic? Or surface-pattern mimicry per domain?

## Headline finding

Substantial work exists — roughly 10+ papers between Mar 2025 and Mar 2026 attack this directly. Mostly on **R1-distill models** (Llama-8B, Qwen-7B), with one paper on **R1-671B itself**. The picture that emerges:

1. **Multiple identified mechanisms across interp levels** (features, heads, GLU vectors, latent directions).
2. **The control circuit is task-general** — backtracking decision-machinery transfers across math, code, GPQA, planning.
3. **R1-Zero RL learns the *gating* of backtracking, not its invention** — the latent direction is already in the base model.
4. **The visible "wait" token is partitioned 4 ways** — load-bearing, inflection-aligned, performative, and pathological.
5. **End-to-end attribution graphs of long CoT are technically out of reach** with current methods (Anthropic explicit statement).

## What was found, by interp level

### Latent direction level — the model-diffing result

**Ward, Lin, Venhoff, Nanda (2025), *Reasoning-Finetuning Repurposes Latent Representations in Base Models*, [arXiv:2507.12638](https://arxiv.org/abs/2507.12638).**

The cleanest mechanistic version of the "R1-Zero doesn't invent reasoning" hypothesis. Shows the **latent direction driving backtracking in DeepSeek-R1-Distill-Llama-8B is already present in the base model.** RL doesn't create the backtracking representation — it repurposes a direction the base model already has.

**This is the answer to "what does R1-Zero RL actually add mechanistically": gating and amplification, not invention.**

**Liu et al. (Sea AI Lab), *Understanding R1-Zero-Like Training*, [arXiv:2503.20783](https://arxiv.org/abs/2503.20783)** (Mar 2025). Behavioral version of the same finding: DeepSeek-V3-Base already produces aha-moment keywords pre-RL, and they don't correlate with accuracy in the base. They name this "Superficial Self-Reflection."

### Feature level — SAEs, transcoders, crosscoders

**Goodfire, *Under the Hood of a Reasoning Model* ([blog, 2025](https://www.goodfire.ai/research/under-the-hood-of-a-reasoning-model)).** First SAEs on **DeepSeek-R1 671B itself.** Qualitative backtracking features + causal steering demos (Feature 22286 = division; Feature 15072 = "Answer Quickly" shortens reasoning). Caveat: doesn't publish specific "aha moment" feature IDs or layer info.

**Galichin et al. (AIRI), *I Have Covered All the Bases Here*, [arXiv:2503.18878](https://arxiv.org/abs/2503.18878)** (Mar 2025). R1-Distill-Llama-8B at layer 19. Custom "ReasonScore" → 200 candidates → 46 manually-verified features:

| Category | Feature IDs |
|---|---|
| Reflection | #4395, #16778, #46691 |
| Uncertainty / exploration | #25953, #61104 |
| Edge-case handling | #3942 |

Steering at γ=2 → **+13.4% AIME-2024, +2.2% MATH-500, +4% GPQA-D.** Traces 13-20% longer. Replicates on Nemotron-Nano-8B.

**Troitskii, Pal, Wendler, McDougall, Nanda, *Internal states before wait modulate reasoning patterns*, [arXiv:2510.04128](https://arxiv.org/abs/2510.04128)** (Oct 2025). The most direct attack on the "wait" token. Crosscoders (32,768 features) across base and R1-Distill-Llama-8B at layer 15. Latent attribution patching on the "wait" logit over 620 rollouts.

```
Top-50 features promoting "wait":     backtracking, self-verification
Bottom-50 suppressing "wait":         restarting, recalling knowledge,
                                      uncertainty, double-checking

Steered features:
  #188     → MATH-500 84%, +28% length
  #31748   → MATH-500 61%, +45% length
```

**Hu, Ward, Icard, Potts (Stanford), *Transcoder Adapters for Reasoning-Model Diffing*, [arXiv:2602.20904](https://arxiv.org/abs/2602.20904)** (Feb 2026). Qwen2.5-Math-7B vs R1-Distill-Qwen-7B, 191k transcoder features. Critical finding: **~5.6k features (~2.4%) specifically produce hesitation tokens. Ablating them reduces response length, often *without affecting accuracy*** → much hesitation is **performative**.

### Attention head level

**Zhang et al. (Microsoft), *From Reasoning to Answer*, [arXiv:2509.23676](https://arxiv.org/abs/2509.23676)** (2025). Identified **Reasoning-Focus Heads (RFHs)** in mid layers (8-16 for Llama-8B; 14-22 for Qwen-7B; 12-20 for Qwen-1.5B). Activation patching at answer-flipping reasoning tokens shifts normalized logit diff by **~0.5** — **CoT is causal, not epiphenomenal** for these inflection points.

**Park, Jeong, Kang, *Thinking Sparks!*, [arXiv:2509.25758](https://arxiv.org/abs/2509.25758)** (2025). Compared distillation vs SFT vs GRPO on Qwen2.5-Math base. Different training schemes introduce different head profiles. **Ablating SFT-acquired heads drops accuracy "close to zero"** on AIME'24. Distillation heads ablation: 30.0 → 26.6.

**Lee, Sun, Wendler, Viégas, Wattenberg, *The Geometry of Self-Verification in a Task-Specific Reasoning Model*, [arXiv:2504.14379](https://arxiv.org/abs/2504.14379)** (2025). Most concrete head-level self-verification result. Late-layer **GLU vectors** encode "success"/"incorrect" semantics. A small set of **previous-token heads whose ablation disables verification entirely.** Similar components found in both base model and general R1 reasoning model → task-general.

**Zhang et al. (UT Austin / Together AI), *CREST*, [arXiv:2512.24574](https://arxiv.org/abs/2512.24574)** (Dec 2025). **Cognitive Heads** identified via linear probes (>85% accuracy distinguishing linear vs non-linear reasoning steps), concentrated in deeper layers. Head-specific steering vectors:

```
R1-Distill-Qwen-1.5B:
  AMC23           90.0% vs 72.5%  (-37.6% tokens)
  AIME            +10% accuracy   (-30% tokens)
  LiveCodeBench   +5.3%
  GPQA-D          +26.6% on common-sense
  Calendar Planning  lift
```

**MATH-calibrated steering vectors transfer zero-shot to code, GPQA, planning** — strongest single piece of task-generality evidence.

### Hidden state / probing level

**Zhang et al., *Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification*, [arXiv:2504.05419](https://arxiv.org/abs/2504.05419)** (2025). Linear probes reveal reasoning models internally represent whether intermediate answers are correct **before they verbalize this via "wait" or correction tokens.** Critical for the causal-vs-performative question: verification happens internally first, surface token follows.

### Sentence level

**Bogdan, Macar, Nanda, Conmy, *Thought Anchors: Which LLM Reasoning Steps Matter?*, [arXiv:2506.19143](https://arxiv.org/abs/2506.19143)** (2025). Sentence-level causal importance in long CoT. Finding: **planning sentences and backtracking sentences are disproportionately load-bearing.** Coarser than circuit, but directly relevant for partitioning what matters in long reasoning traces.

### Behavior level — cross-task steering

**Venhoff, Arcuschin, Torr, Conmy, Nanda, *Understanding Reasoning in Thinking Language Models via Steering Vectors*, [arXiv:2506.18167](https://arxiv.org/abs/2506.18167)** (2025). Behavior steering vectors for backtracking / uncertainty / example testing across **10 task categories**. Cross-model consistency across DeepSeek-R1 distills. Strongest evidence backtracking is a task-general behavior with reusable activation directions.

## The 4-way partition of "wait" tokens

This is the most important refinement to take away. Different papers find different things because they're looking at different sub-cases:

```
   ┌─────────────────────────────────────────────────────────────────────┐
   │                                                                     │
   │   1. Load-bearing wait                                              │
   │      causally drives correct answer                                 │
   │      → Microsoft RFH patching, AIRI steering, Lee verification      │
   │                                                                     │
   │   2. Inflection-aligned wait                                        │
   │      surrounds a real internal belief shift                         │
   │      → Zhang 2504.05419 probes: model "knows" before saying         │
   │      → Reasoning Theater 2603.05488 confirms                        │
   │                                                                     │
   │   3. Performative wait                                              │
   │      stylistic mimicry, no accuracy effect                          │
   │      → Stanford 2602.20904 ablations                                │
   │      → Sea AI Lab 2503.20783: base model wait                       │
   │                                                                     │
   │   4. Pathological wait (looping)                                    │
   │      degenerate cycling, hurts accuracy                             │
   │      → Pipis 2512.12895: "WaitWaitWait..."                          │
   │                                                                     │
   └─────────────────────────────────────────────────────────────────────┘
```

Same surface token, four different internal stories. The Reasoning Theater paper ([arXiv:2603.05488](https://arxiv.org/abs/2603.05488), 2026) gives the nuanced bridge: **CoT between inflection points can be performative, but the inflection points themselves usually align with real belief shifts.**

## CoT faithfulness — the inconvenient numbers

**Lanham et al. (2023), [arXiv:2307.13702](https://arxiv.org/abs/2307.13702)** and **Turpin et al. (2023), [arXiv:2305.04388](https://arxiv.org/abs/2305.04388)** are the pre-R1 baselines: CoT can be plausible but unfaithful.

**Arcuschin et al. (2025), *Chain-of-Thought Reasoning In The Wild Is Not Always Faithful*, [arXiv:2503.08679](https://arxiv.org/abs/2503.08679)** extends to realistic prompts.

**Anthropic 2025, *Reasoning models don't always say what they think* ([anthropic.com](https://www.anthropic.com/research/reasoning-models-dont-say-think))** — hint acknowledgment rate in CoT:

```
Claude 3.7 Sonnet:    25%
DeepSeek R1:           39%
```

60-75% of the time when a model uses a hint to reach an answer, the hint does not appear in the visible CoT. CoT is at best a partial readout.

**Young, *Why Models Know But Don't Say*, [arXiv:2603.26410](https://arxiv.org/abs/2603.26410)** (Mar 2026). Across 12 reasoning models incl R1: **55.4% of cases show hints acknowledged in thinking-channel but suppressed from answer-channel.** Categorizes failure modes: Bare Truncation, Laundered Attribution, Confabulated Justification.

## Task-generality — the answer is "yes, for the control"

| Evidence | Generality |
|---|---|
| Venhoff: behavior vectors across 10 task categories, cross-model | **Strongly general** |
| CREST: MATH-calibrated cognitive-head vectors transfer to code, GPQA, planning | **Strongly general** |
| Lee: verification components found in base + general R1 | General |
| Bogdan: backtracking sentences load-bearing across reasoning traces | General at sentence level |
| AIRI: features replicate to second architecture (Nemotron) | Architecture-general |
| Stanford transcoder-adapters: only math tested | Untested |

Consensus: **the control circuit that decides when to backtrack reuses across math, code, logic, planning.** The content being backtracked over is task-specific. Same pattern as Anthropic's "6+9" arithmetic feature reused across math / astronomy / financial tables — procedural circuits abstract over domain.

## What's NOT been done

1. **No end-to-end attribution graph of a single aha-moment.** Anthropic-Biology-style trace (Dallas→Texas→Austin) on a "Wait, that's wrong. Let me reconsider..." token. Anthropic explicitly states current methods don't scale to thousand-token CoT.

2. **No mechanistic R1-Zero vs V3-Base diff.** Sea AI Lab showed behaviorally. Goodfire has R1-671B SAEs but no diff vs V3-Base. Ward showed it for *distill* — not for R1-Zero itself.

3. **No "6+9"-style cross-task feature-reuse demo for backtracking.** Anthropic showed reusable arithmetic features across domains. The analog for reflection features hasn't been published.

4. **No o1 internals.** Closed source. Apollo Research treats CoT behaviorally only.

5. **No clean partition of performative vs load-bearing wait at the circuit level.** Stanford hints at it; nobody isolates which contexts make hesitation causal.

## Methodological tools (if you wanted to extend this work)

- **Crosscoders + latent attribution patching** for feature-level "wait" analysis (Troitskii recipe).
- **Linear probes on hidden states** for internal correctness signals (Zhang 2504.05419).
- **Activation patching at reasoning tokens** for causal CoT importance (Microsoft RFH).
- **Steering vector calibration on MATH, transfer test on code/logic/planning** for task-generality (CREST).
- **Sentence-level anchor discovery** for long CoT (Bogdan).
- **Anthropic's QK extension** ([transformer-circuits.pub/2025/attention-qk](https://transformer-circuits.pub/2025/attention-qk/index.html)) for why model attends to specific earlier steps when backtracking.

Recommended starting points for R1-Distill-Llama-8B:
- Residual stream **layer 15** for crosscoder/wait analysis (Troitskii)
- Residual stream **layer 19** for reflection features (AIRI)
- **Mid-layers 8-16** for Reasoning-Focus Heads (Microsoft)
- **Late layers** for GLU verification vectors (Lee)

## Key takeaways for the knowledge-vs-intelligence thread

These findings sharpen the answer to whether reasoning machinery is separable from knowledge:

1. **Reasoning control circuits are task-general** — same machinery decides when to backtrack across math, code, planning. Strongly supports separability of "reasoning machinery" as a substrate-level capability.

2. **The machinery is in the base model already.** Ward's repurposing result + Sea AI Lab's superficial-self-reflection result agree: RL doesn't create reasoning, it learns to gate existing latent capacity. This is the cleanest mechanistic version of the "substrate threshold" question.

3. **A substantial fraction of visible reasoning is performative.** The 4-way partition of "wait" tokens means CoT is at best a partial window into the actual mechanism. Hint-acknowledgment rates of 25-39% confirm at scale.

4. **Reasoning emerges from very small RL deltas on top of large pretrained substrates.** Combined with the data-efficiency findings (LIMO, R1-Zero with rule-based reward only), the implication: reasoning is mostly *in the substrate already*, waiting to be gated correctly.

## Key papers — consolidated

| Paper | Level | arXiv |
|---|---|---|
| Liu et al., *Understanding R1-Zero-Like Training* (Sea AI Lab) | behavioral | [2503.20783](https://arxiv.org/abs/2503.20783) |
| Galichin et al., *I Have Covered All the Bases Here* (AIRI) | SAE feature | [2503.18878](https://arxiv.org/abs/2503.18878) |
| Goodfire, *Under the Hood of a Reasoning Model* | SAE feature | [blog](https://www.goodfire.ai/research/under-the-hood-of-a-reasoning-model) |
| Lee et al., *The Geometry of Self-Verification* | head + GLU | [2504.14379](https://arxiv.org/abs/2504.14379) |
| Zhang et al., *Reasoning Models Know When They're Right* | hidden state probe | [2504.05419](https://arxiv.org/abs/2504.05419) |
| Venhoff et al., *Steering Vectors* | latent direction | [2506.18167](https://arxiv.org/abs/2506.18167) |
| Bogdan et al., *Thought Anchors* | sentence | [2506.19143](https://arxiv.org/abs/2506.19143) |
| Ward et al., *Reasoning-Finetuning Repurposes Latent Reps* | latent direction (diff) | [2507.12638](https://arxiv.org/abs/2507.12638) |
| Zhang et al., *From Reasoning to Answer* (Microsoft) | attention head | [2509.23676](https://arxiv.org/abs/2509.23676) |
| Park et al., *Thinking Sparks!* | head profile | [2509.25758](https://arxiv.org/abs/2509.25758) |
| Troitskii et al., *Internal states before wait* | crosscoder feature | [2510.04128](https://arxiv.org/abs/2510.04128) |
| Pipis et al., *Why Do Reasoning Models Loop?* | pathology | [2512.12895](https://arxiv.org/abs/2512.12895) |
| Zhang et al., *CREST* (UT Austin / Together AI) | cognitive head | [2512.24574](https://arxiv.org/abs/2512.24574) |
| Hwang et al., *Oops, Wait* | token-level signal | [2601.17421](https://arxiv.org/abs/2601.17421) |
| Hu et al., *Transcoder Adapters* (Stanford) | transcoder feature | [2602.20904](https://arxiv.org/abs/2602.20904) |
| Young, *Why Models Know But Don't Say* | behavioral faithfulness | [2603.26410](https://arxiv.org/abs/2603.26410) |
| *Reasoning Theater* | faithfulness nuance | [2603.05488](https://arxiv.org/abs/2603.05488) |
| Kamath, Ameisen et al., *Tracing Attention Computation Through Feature Interactions* | Anthropic methodology | [transformer-circuits.pub](https://transformer-circuits.pub/2025/attention-qk/index.html) |
| Lindsey et al., *On the Biology of a Large Language Model* | Anthropic methodology | [transformer-circuits.pub](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) |

## Cross-references

- Mechanistic interpretability foundations: [interpretability-guide.md](interpretability-guide.md)
- Text-level interp (NLA): [natural-language-autoencoders-guide.md](natural-language-autoencoders-guide.md)
- The knowledge-vs-intelligence framing this connects to: [knowledge-vs-intelligence.md](knowledge-vs-intelligence.md)
- Interp ladder + entry point: [../interp.md](../interp.md)
- Reasoning model lineage (R1, o1): [../model-deepseek-r1-guide.md](../models/model-deepseek-r1-guide.md), [../model-o1-guide.md](../models/model-o1-guide.md)
