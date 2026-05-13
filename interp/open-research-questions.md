# Open Research Questions — Knowledge vs Intelligence and Reasoning Mechanism

Significant open problems and research directions on (a) separability of knowledge and general intelligence in LLMs and (b) the internal mechanism of reasoning. Synthesized from the surrounding interp guides. Ranked by significance — these are gaps whose answers would meaningfully advance the field, not parameter sweeps.

## Top tier — deepest gaps, biggest payoffs

### 1. The decisive substrate-decomposition experiment

**Question:** Can RL-from-random-init produce reasoning, or is a knowledge substrate a strict necessity?

The behavioral answer exists (Sea AI Lab: V3-Base already says "wait"); the latent-direction answer exists (Ward: backtracking direction pre-exists in base). What's missing: **a controlled experiment doing RLVR from a model with zero pretraining**, scaled past the point where TinyZero's 0.5B failed. If reasoning *can* emerge from random init given enough compute, the field's "RL elicits existing capacity" framing is wrong. If it can't, we learn a concrete impossibility result.

**Why it matters.** The single experiment that would settle whether reasoning is *in* pretraining or merely *enabled by* it. Every current paper assumes the latter; nobody has tested the former.

**Why it hasn't been done.** Compute cost. And the field implicitly assumes it's hopeless — exactly where insight hides.

### 2. The minimum-knowledge-substrate floor — characterized formally

**Question:** Is there a mathematical / information-theoretic characterization of "minimum world-model substrate sufficient for general reasoning"?

Spelke gives a psychological list (objects, agents, number, space, geometry, causality). Tenenbaum gives a computational version (probabilistic programs over those primitives). LLMs operate on neither — they soak up text. We don't know whether the LLM substrate at, say, Qwen2.5-1.5B, encodes Spelke-equivalent priors or something completely different that happens to be reasoning-supportive.

**Why it matters.** Without a formal characterization we can't tell whether (a) a 3B language-only model has crossed the substrate threshold, or (b) a non-language world model needs the same threshold in a different basis. The Chollet framework needs a substrate axis it currently lacks.

**Concrete experiment.** Train probes for each Spelke primitive across a model lineage (1B → 70B). Plot probe accuracy vs ARC-AGI-2 performance. If the relationship is mediated by Spelke-primitive presence, you have a substrate metric.

### 3. The wholesale knowledge-ablation experiment

**Question:** Can you ablate large *categories* of knowledge (all post-2020 events, all chemistry, all of a specific language) from a trained reasoning model while preserving general reasoning?

MQuAKE shows single-fact edits don't propagate through multi-hop reasoning. But that's a strong negative result *only* for single-fact ROME-style edits — it doesn't address whether large knowledge subspaces can be cleanly excised. SAE feature-set ablation (kill all features whose top-activating examples are 2025 events) is the natural method.

**Why it matters.** If you can cleanly remove chemistry knowledge and reasoning survives, the separability claim is empirically grounded for the first time. If you can't, MQuAKE generalizes — facts and reasoning are non-trivially entangled at scale.

**Untested direction.** Anthropic's Biology paper showed procedural features (6+9 arithmetic) reuse across data domains. The reverse question — domain features reuse across procedures — has not been tested.

### 4. End-to-end attribution graph of one aha moment

**Question:** Decompose a single "Wait, that's wrong. Let me reconsider..." event in R1 or R1-distill into a complete causal graph from prior reasoning error → reflection-triggering features → "wait" token → new reasoning path.

Anthropic explicitly says current circuit-tracing doesn't scale to thousand-token traces. This isn't unimportant — it's the literal load-bearing methodological gap. Every feature-level paper (AIRI, Troitskii, Stanford) sees pieces of the elephant. Nobody has the whole elephant.

**Why it matters.** Without this, we can't verify any of the 4-way partition claims (load-bearing vs inflection-aligned vs performative vs pathological). We're inferring from ablation deltas, not seeing the causal structure.

**The methodological subproblem.** Long-CoT attribution requires either (a) compressed attribution graphs that aggregate across tokens, or (b) sparse approximations that prune subcircuits. Either approach is itself a methodology paper.

## Second tier — still significant, slightly less foundational

### 5. The latent-reasoning experiment

**Question:** If you suppress visible CoT (force immediate answer) but allow internal computation to continue silently in the residual stream, do reasoning models still get the answer right?

Hao 2024's Coconut started this. The 2025-2026 follow-up at reasoning-model scale doesn't exist or hasn't been surfaced. The Anthropic faithfulness numbers (25-39% hint acknowledgment) imply substantial latent reasoning, but nobody has cleanly measured *how much* of the answer-determining computation happens silently.

**Why it matters.** If suppressed-CoT models maintain most of their accuracy, then visible CoT is mostly decorative — and the entire "wait" / aha-moment research program is studying a surface trace rather than the underlying mechanism. If accuracy crashes, visible CoT is causally load-bearing. Either answer reshapes the field.

### 6. Cross-model reasoning circuit identity vs convergence

**Question:** Are the "cognitive heads" and "backtracking features" identified within the DeepSeek-R1-distill family *mechanistically the same* in Anthropic, OpenAI, and DeepSeek frontier reasoning models?

CREST and Venhoff show cross-model consistency *within R1-distill family*. We don't know whether o3, Claude Opus 4.6, and Gemini 3 implement reasoning with the same circuits (universal mechanism) or convergent-but-distinct circuits (multiple realizations).

**Why it matters.** If universal, we have a substrate-independent mechanism worth studying as such. If convergent, "reasoning" is a family of implementations and the substrate determines which one a given model uses. Determines whether interp results transfer between labs.

**Approach.** Build a cross-family steering vector — extract backtracking direction from R1-distill, apply it to Llama-3-Instruct via activation patching. If it elicits backtracking, you have evidence of universality.

### 7. The architectural decoupling test

**Question:** Build a system with an explicit small parametric reasoning core + an explicit external knowledge store + a retrieval mechanism, and measure whether it matches a 70B parametric model on broad benchmarks (MMLU + MATH + GPQA + HumanEval simultaneously).

RETRO/Atlas did this for knowledge-intensive QA in 2022. Nobody has done it post-RLVR / post-R1-Zero with a *reasoning-trained* small core. The LLM-WikiRace 2026 threshold hypothesis predicts this would work; nobody has built it.

**Why it matters.** This is the practical test of whether the "small reasoning core + web search" paradigm can hold. If a 3B Phi-style core + retrieval + RLVR-trained reasoning matches Opus 4.6 on broad benchmarks, the field's central paradigm shifts. If it falls short by a large margin, "reasoning needs ambient knowledge in weights" is structural.

### 8. R1-Zero training-step-by-step mechanistic diff

**Question:** What *exactly* changes in DeepSeek-V3-Base's circuits over R1-Zero's 10,400 training steps, going from 15.6% → 77.9% on AIME?

Ward did a pre-vs-post latent direction diff. Goodfire trained SAEs on R1-671B but not stepwise. Nobody has the trajectory: which features/heads change first? Which changes correlate with the AIME-curve inflection at step 8.2k? Does the "wait"-frequency increase show up in a specific circuit before it shows up in tokens?

**Why it matters.** This is the RLVR analog of the grokking literature. If reasoning emerges in a sharp circuit-level phase transition (like induction heads in Olsson 2022), we know it. If it's gradual accumulation, we know that. Currently we have only behavioral curves.

## Third tier — significant, more incremental

### 9. The metacognition-reasoning entanglement question

**Question:** Is metacognition (Zhang 2504.05419's finding that models know they're right *before* saying so) *necessary* for good reasoning, or separable?

Concrete: ablate the hidden-state directions that encode "this intermediate answer is correct," and measure whether AIME performance survives. If yes, metacognition is decorative. If no, it's load-bearing — and we have a separate fundamental capacity to characterize.

### 10. Sub-Spelke probe set for LLMs

**Question:** Do reasoning models contain learnable analogs of each Spelke core-knowledge primitive (object persistence, agent-as-causal-entity, number sense, geometric intuition, causality)?

The cognitive-science-bridges-LLM-interp experiment. Anthropic Biology shows arithmetic features reuse across domains. Are there equally fundamental "agent" features, "causal" features, "object" features? If yes, we have an empirical mapping between developmental psychology and mech interp — and a concrete substrate-quality metric (Q2).

### 11. Knowledge-density vs reasoning floor

**Question:** Does pretraining-data composition (textbooks vs web vs synthetic) systematically shift the reasoning floor, holding parameter count constant?

Phi line says yes (synthetic textbooks beat raw web at fixed scale). But nobody has tested this against ARC-AGI-2 specifically — only against benchmarks that are themselves knowledge-loaded. The right experiment: train matched 3B models on (a) web-only, (b) textbook-synthetic, (c) Spelke-primitive synthetic data — and measure on a battery of *knowledge-controlled* reasoning tasks (ARC-AGI-2, Mitchell counterfactual analogy, Reasoning-or-Reciting).

### 12. Performative-vs-load-bearing partition at the circuit level

**Question:** Can you cleanly identify, *in advance*, whether a given "wait" token in a generation is load-bearing or performative — from internal state alone?

Stanford 2602.20904 showed 5.6k features generate hesitation; some are causal, some aren't. Nobody has built a real-time classifier on internal activations that predicts before the token is emitted whether ablating it would change the answer. If buildable, you have an empirical partition tool and the foundation for "trimmed-CoT" inference (skip the performative wait, keep the load-bearing one).

## What's NOT on this list and why

- **More SAE training on more models.** Trivial follow-up. The methods are mature; the insights need new questions.
- **Better RL recipes / longer training.** Engineering, not science.
- **More benchmarks.** Eval design matters, but the field doesn't lack benchmarks — it lacks a *substrate metric* (Q2) and a *causal partition tool* (Q12).
- **Cross-lingual / multimodal transfer studies.** Important but extension, not foundation.
- **Better reward design.** Same — engineering improvements.

## The single research direction worth betting on

If forced to pick one project, it would be **the architectural decoupling test (Q7) combined with the wholesale knowledge-ablation experiment (Q3)**. Together they give an empirical handle on the core philosophical question of knowledge-vs-intelligence separability.

```
   Q3 (negative test):  Take a frontier model, surgically remove a
                        large knowledge subspace via SAE-feature ablation,
                        and see what reasoning survives.

   Q7 (positive test):  Build a small reasoning core + retrieval + tools
                        from scratch and see what reasoning it acquires.

   ┌───────────────────────────────────────────────────────┐
   │  Both succeed:   knowledge and reasoning are clean-   │
   │                  ly separable in principle. Field's   │
   │                  paradigm shifts toward small-core    │
   │                  architectures.                       │
   │                                                       │
   │  Both fail:      they're fundamentally entangled.     │
   │                  Scaling parametric knowledge is      │
   │                  necessary, not contingent. Marcus    │
   │                  / LeCun framing wins.                │
   │                                                       │
   │  Mixed result:   you've found exactly where the       │
   │                  separability boundary is — the       │
   │                  best outcome. Gives you a substrate  │
   │                  metric.                              │
   └───────────────────────────────────────────────────────┘
```

Feasibility on 1×H800: the negative test (Q3) is doable at the Gemma-3-12B scale (released NLA checkpoint scale). Ablate feature subsets identified by topic via Neuronpedia, evaluate on MMLU vs ARC-AGI-2 vs Mitchell counterfactual battery. The positive test (Q7) is harder but tractable with a 3B Phi-style core + a Wikipedia retrieval index.

## Cross-references in this curriculum

- The conceptual framing: [knowledge-vs-intelligence.md](knowledge-vs-intelligence.md)
- Reasoning-model interp landscape this builds on: [reasoning-model-interp.md](reasoning-model-interp.md)
- Mech-interp foundations: [interpretability-guide.md](interpretability-guide.md)
- Text-level interp: [natural-language-autoencoders-guide.md](natural-language-autoencoders-guide.md)
- Interp ladder + entry point: [../interp.md](../interp.md)
