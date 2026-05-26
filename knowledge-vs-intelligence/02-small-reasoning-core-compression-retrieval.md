# 02 — Building a Small Reasoning Core: Compression, Unlearning & Retrieval

*Covers Q2 ("reduce a frontier model to reasoning + necessary knowledge") and Q3 ("build a small model that's good enough at reasoning + web search for knowledge"). These are duals: subtract knowledge from a big model vs. add reasoning to a small one.*

---

## The target artifact: Karpathy's "cognitive core"

The popular name for what you're describing is Andrej Karpathy's **"cognitive core"** (dates verified):

- **June 27, 2025** (X post): the ideal LLM is like "the reasoning part of your brain" — it should *maximally sacrifice encyclopedic knowledge for capability*, and ask the world for facts/actions rather than memorizing them. Framed as the "kernel of LLM personal computing": natively multimodal, Matryoshka-style dial-able, aggressively tool-using.
- **Oct 30, 2025** (Dwarkesh Patel interview): a **~1-billion-parameter "cognitive core" may be enough** for productive conversation *if* factual lookup is delegated to tools/search.

It is a **stated aspiration, not a built system.** The closest *academic* formulations of your exact thesis:

- **"General Intelligence Requires Reward-based Pretraining"** — Han, Pari, Gershman, **Pulkit Agrawal** (MIT, arXiv:2502.19402, Feb 2025; *verified title*). Argues **"the core issue is the coupling of knowledge and reasoning in LLMs"** and proposes to *decouple* them: reward-based pretraining from scratch + curriculum of synthetic reasoning tasks + reduced context windows (to kill spurious token correlations) + pairing the reasoning system with external memory/retrieval. This is the most direct statement of your Q2/Q3 program in the literature. **[verified abstract; position+method paper]**
- **RARE: Retrieval-Augmented Reasoning Modeling** (arXiv:2503.23513): explicitly "externalize domain knowledge to retrievable sources, internalize reasoning patterns during training." **[LOW — single-source]**
- **Decoupling Knowledge and Reasoning** (arXiv:2507.18178): across 15 LLMs, finds knowledge concentrates in **lower layers**, reasoning adjustment in **higher layers**, and reasoning gains are **domain-specific** — mechanistic support for partial separability. **[LOW — single-source]**

---

## Q2 — Can you subtract knowledge from a big model and keep reasoning?

### Unlearning: no — reasoning and knowledge are entangled

This is the most *direct* test of "remove memorized knowledge, keep reasoning," and the answer is currently **no, not cleanly.**

- **R²MU — "Reasoning Model Unlearning"** (arXiv:2506.12963, *verified*): standard RMU-style unlearning (perturbing activations along the reasoning trajectory) erases target facts but **severely damages reasoning** — degenerate repetition, broad capability collapse. Their reasoning-aware variant only preserves reasoning by *adding* CoT supervision — i.e., reasoning must be **actively defended**, not freely retained. **[HIGH]**
- **R-TOFU** (arXiv:2505.15214) and IBM's "Rethinking Unlearning for Large Reasoning Models" (ICML 2025) reach the same conclusion: knowledge removal leaks through, or collapses reasoning.
- Unlearning is also often **reversible** — knowledge isn't truly deleted, just suppressed.
- **Knowledge editing has the dual problem** — **MQuAKE** (arXiv:2305.14795): you can edit a fact and have the model *recall* it, but **multi-hop consequences of the edit fail badly.** External memory (MeLLo) beats weight-editing — evidence *for* externalizing knowledge, but also that parametric surgery is brittle.

> **Why this matters for Q2:** the entanglement is not a tooling problem — it's the A.2/B.5 finding (shared weights, computation-coupled knowledge) showing up in practice. You cannot currently excise the "knowledge factor" and keep the "reasoning factor" intact by subtraction.

### Pruning + distillation: works, and knowledge degrades first

NVIDIA's **Minitron** (arXiv:2407.14679) is the most practical guide, and its ablations contain a clue for your thesis:

- Best practice: **width pruning > depth pruning** at ≤15B scale; **retrain with distillation, not ground-truth**; iterative prune-distill > one-shot.
- Iso-compute ablation (4B student):

  | Recipe | MMLU | HellaSwag |
  |---|---|---|
  | 4B trained from scratch | 26.24 | 48.23 |
  | prune+distill 15B→4B | 37.81 | 51.04 |
  | prune+distill 15B→8B→4B | 42.45 | 52.04 |

- **MMLU (broad knowledge) is far more fragile than HellaSwag (lightweight pattern/commonsense) under compression.** Width-only pruning beats depth-pruning after retrain — i.e., **computational depth is disproportionately worth preserving.** **[HIGH]**

This is *weak, indirect* support for the factorization: when you squeeze a model, broad knowledge-heavy capability degrades earlier than the structured machinery — consistent with "knowledge = costly storage, reasoning = cheaper compute."

### Verdict on Q2

> You can **compress** a frontier model and recover much of it with distillation. You can **edit/remove** some knowledge. You **cannot yet** reliably produce a "knowledge-light but broadly reasoning-strong" model by *unlearning/editing alone* — the subtraction damages reasoning. Q3 (build small from the start) is the more promising direction.

---

## Q3 — How small can a model be and still reason? (+ retrieval)

### Small reasoning models (SRMs): the benchmark reality

Reasoning **distills** into small dense models far better than it can be taught from scratch. Verified numbers (official DeepSeek-R1 report, arXiv:2501.12948; Phi-4-reasoning, arXiv:2504.21318):

| Model | Params | AIME'24 | MATH-500 | GPQA-Diamond | LiveCodeBench |
|---|---|---|---|---|---|
| R1-Distill-Qwen-1.5B | 1.5B | 28.9 | 83.9 | 33.8 | 16.9 |
| R1-Distill-Qwen-7B | 7B | 55.5 | 92.8 | 49.1 | 37.6 |
| R1-Distill-Qwen-14B | 14B | 69.7 | 93.9 | 59.1 | 53.1 |
| R1-Distill-Qwen-32B | 32B | 72.6 | 94.3 | 62.1 | 57.2 |
| R1-Distill-Llama-70B | 70B | 70.0 | 94.5 | 65.2 | 57.5 |
| **full DeepSeek-R1** | 671B (37B act.) | 79.8 | 97.3 | **71.5** | — |
| Phi-4-reasoning | 14B | 75.3 | — | 65.8 | 53.8 |
| Phi-4-reasoning-plus | 14B | 81.3 | — | 68.9 | 53.1 |

*(s1-32B, LIMO-32B, Sky-T1-32B ~$450/17K-examples, Bespoke-Stratos-32B, OpenThinker3-7B all land in the same neighborhood on math; treat blog/model-card numbers as non-standardized. **[MEDIUM]**)*

**The crisp pattern — and the core limit of your thesis:**
- On **narrow, verifiable** math/code, 7–32B distilled models reach 90%+ MATH-500 and rival o1. Reasoning distills cheaply.
- On **broad parametric knowledge**, performance degrades **monotonically with size**: GPQA-Diamond falls 71.5 → 62.1 → 49.1 → 33.8 from R1 → 32B → 7B → 1.5B. **Distilling reasoning does *not* distill knowledge.** **[HIGH]**
- Telling clue (Phi-4 on the Kitab benchmark): adding context massively boosts knowledge metrics — **the model can *use* external knowledge much better than it can *store* it.** Evidence *for* the retrieval direction. **[MEDIUM]**

### Tiny recursive reasoners: narrow, not general

The "few-million-param reasoner" headlines are real but **domain-specific**, not knowledge-light general reasoners:

- **HRM** (Hierarchical Reasoning Model, 27M params, arXiv:2506.21734): 40.3% ARC-AGI-1, near-perfect Sudoku-Extreme / Maze-Hard — *no pretraining, no CoT.*
- **TRM** (Tiny Recursive Model, 7M params, Samsung SAIL Montreal, arXiv:2510.04871): 45% ARC-AGI-1, ~8% ARC-AGI-2; beats HRM with a simpler recurrence.
- **But:** the ARC Prize team's independent re-test dropped HRM **41%→32%** and found it **transductive + puzzle-specific** (per-task training, puzzle-ID embeddings; the *recurrence/iteration* matters, the *hierarchy* doesn't). These are narrow trained solvers — they confirm "tiny systems can reason in structured domains," **not** "tiny tool-using LMs can replace knowledge-rich models." **[VERY_NEW]**

> These tie back to your [ARC-AGI 2 guide](../benchmarks/arc-agi-2-guide.md): they reason over Spelke-style core-knowledge priors with *no* world knowledge — the purest existing "reasoning without knowledge," but only because ARC deliberately needs none.

### Retrieval / tool-augmented reasoning: the externalization path

RL-for-search lets small models use external knowledge well:

| System | Base | Result | Source |
|---|---|---|---|
| **Search-R1** | Qwen2.5-7B / 3B | +41% / +20% over RAG baselines on 7 QA sets (RL-only) | arXiv:2503.09516 |
| **R1-Searcher** | 7B | two-stage RL; beats strong RAG + GPT-4o-mini on their settings | arXiv:2503.05592 |
| **ReSearch** | 7B/32B | search *inside* the reasoning chain, no supervised traces; generalizes | arXiv:2503.19470 (NeurIPS'25) |
| **Search-o1** | — | agentic RAG + "Reason-in-Documents"; targets mid-thought grounding | arXiv:2501.05366 |
| **SEM** | — | optimizes *when NOT to search* (cost/latency control) | arXiv:2505.07903 |

**Where the externalization thesis breaks (the hard limits):**
- **Retrieval itself is often the bottleneck** — **BRIGHT** (arXiv:2407.12883): on reasoning-intensive retrieval, the best retriever gets only **22.1 nDCG@10**; LLM query expansion nudges it to ~22.6. A perfect reasoner can't reason over evidence it can't retrieve. **[HIGH]**
- **Multi-hop composition over retrieved facts collapses** — MQuAKE (above); the 2026 "Weakest-Link Law" (arXiv:2601.12499) finds multi-hop accuracy collapses to the least-visible evidence. **[2026 — flag]**
- **Broad expert depth (HLE/GPQA/MMLU-Pro)** needs multi-step synthesis that retrieval breadth doesn't supply, and many items post-date training cutoffs.

### Verdict on Q3

> **Feasibility: MEDIUM-LOW for broad tasks, HIGH for narrow verifiable domains.** A small model can be a strong *procedural* reasoner and a competent *search user* — but no published 2025–26 system cleanly demonstrates a deliberately knowledge-light 1–3B model that, paired with retrieval, broadly matches knowledge-rich models on GPQA/MMLU-Pro/HLE/open-web QA.

---

## The most-promising stack (synthesis)

If you wanted to build toward the cognitive core, the literature points to a **stack, not a single trick**:

```
1. Strong small BASE with the right substrate   (Qwen2.5-class; LIMO shows
   (math/code/logic-rich, behavior-rich)          missing base knowledge can't be
        │                                          cheaply repaired later)
        ▼
2. DISTILL reasoning from a stronger teacher      (R1/QwQ traces — the one cheap
   (preserves depth; width-prune if compressing)   way to add new patterns; Minitron)
        ▼
3. RL for SEARCH/TOOL use (and when NOT to)        (Search-R1 / SEM; verifiable rewards)
        ▼
4. Small general/instruction/tool mixture          (restore assistant behavior)
        ▼
   Evaluate on THREE axes, not one:
   • procedural: AIME, MATH-500, LiveCodeBench, ARC
   • knowledge-heavy: GPQA-D, MMLU-Pro, SimpleQA
   • retrieval-composition: HotpotQA, MuSiQue, Bamboogle, BRIGHT, MQuAKE
```

### A falsification experiment you could run on your 1×H800

The cleanest test of your thesis — which **nobody has published cleanly** — is well within a single-H800 budget (it's distillation + LoRA + retrieval eval, not large-scale RL):

> Take a 1–3B base. Distill strong math/code/science reasoning into it. Then **deliberately suppress or unlearn SimpleQA/MMLU-style factual recall** while measuring whether procedural reasoning survives (per A.2 entanglement, expect damage — that's the interesting measurement). Finally, **bolt on retrieval** and measure how much of the lost knowledge-task accuracy comes back, **at what $/latency.** Report the three-axis suite above + search-rate, tokens/solved-query, and accuracy under *deliberately noisy* retrieval.

If reasoning survives the knowledge suppression and retrieval recovers the knowledge tasks cheaply → strong evidence the factorization is real and buildable. If reasoning collapses with the knowledge → the entanglement (A.2, R²MU) is the real wall. Either result is publishable and directly answers Q2+Q3.

---

## Confidence summary (Q2 + Q3)

| Claim | Confidence |
|---|---|
| Karpathy "cognitive core" framing + dates | HIGH (verified) |
| Unlearning removes knowledge but damages reasoning (entanglement) | HIGH |
| Pruning+distill works; broad knowledge degrades before reasoning | HIGH (Minitron) |
| Reasoning distills into small models; knowledge does not | HIGH |
| GPQA degrades monotonically with size at fixed recipe | HIGH |
| Tiny recursive reasoners are narrow, not general | VERY_NEW |
| Search-RL helps small models use retrieval | MEDIUM-HIGH |
| Retrieval quality + multi-hop are the hard ceilings | HIGH |
| A broad "small core + retrieval" system matching frontier | UNPROVEN (MEDIUM-LOW) |

Continue to [03 — the data that trains such a model](03-data-for-small-reasoning-models.md).
