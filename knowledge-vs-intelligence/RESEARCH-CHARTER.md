# Knowledge vs Intelligence — Research Charter & Continuation Guide

*Purpose: the entry point for resuming this research in a new chat. Captures the original research questions, my motivations, the context, what's been produced, and where to pick up. Last updated May 2026.*

---

## How to resume in a new chat

1. Read **this file**, then [`00-overview-and-field-map.md`](00-overview-and-field-map.md) (the synthesized answers + field map).
2. Skim the deep file relevant to what I want to do next: [`01`](01-knowledge-capacity-and-reasoning-mechanism.md) (science), [`02`](02-small-reasoning-core-compression-retrieval.md) (build), [`03`](03-data-for-small-reasoning-models.md) (data).
3. Browse [`proposals/`](proposals/) for the 13 candidate experiments.

**Suggested kickoff prompt:**
> "Read `knowledge-vs-intelligence/RESEARCH-CHARTER.md` and `00-overview-and-field-map.md` to load context. We're continuing the knowledge-vs-intelligence research. I want to [pick one: turn proposal X.Y into a full experiment spec + H800 repo scaffold / fold the 6 new papers into reports 01 & 03 / write up the pure-reasoning scaling-law thread / brainstorm more on Q__]."

---

## The research questions (verbatim, as I posed them)

This is "Thread B" of my AI curriculum. The five guiding questions, in my own words:

1. **How much of a frontier model's parameters is knowledge vs intelligence?** A lot of them must be knowledge.
2. **How to reduce a frontier model to reasoning ability + necessary knowledge only** (maybe objects, logic, some math, some meta-cognition like planning, backtracking, etc.)?
3. **Or in the opposite direction, how to build a small model that's just good enough for reasoning** (with a web-search tool to obtain knowledge externally)?
4. **What data is necessary to train such a small reasoning model?**
5. **What's the mechanism behind reasoning in LLMs?**

(I also asked: *what research field does this belong to?* → answered in `00`, §"What field is this?")

---

## My motivation & intuitions (why I care)

- I suspect **most of a frontier model's parameters are spent storing knowledge**, not doing reasoning — and that the reasoning "part" is much smaller and more compressible. I want to know how much, and whether the two can be separated.
- The dream artifact: a **small reasoning core** that holds only the *necessary* knowledge substrate — objects, logic, some math, metacognition (planning, backtracking, verification) — and **offloads the rest to web search / tools**. (This is essentially Karpathy's "cognitive core.")
- Two directions to get there, which are duals: **(Q2) subtract** knowledge from a big model, or **(Q3) build** a small one from the start + retrieval. I care which is tractable.
- I want to know **what data** would train such a model, and **what reasoning actually is** mechanistically — because that determines whether the whole program is even possible.
- **Related prior thread (my own, ~May 24–25 2026, still not written up):** "frontier models have huge crystallized knowledge with so many params — what if we train on just reasoning-primitive data (enough to elicit reasoning), then scale params + data? Could it match frontier reasoning with FEWER params? Is there a **Chinchilla-style law for a pure-reasoning model**?" This is the scaling-law face of the same question and should eventually become its own guide (candidates found: ScaleRL 2510.13786, T² 2604.01411).

---

## Context & constraints

- **Me:** strong ML background — trained GPT-2 124M (nanoGPT) from scratch; SFT (Dolly 15k, SlimOrca 520k); DPO (hh-rlhf); GRPO; know mechanistic interpretability, MoE, knowledge distillation, test-time compute, reasoning models. Write educational guides for myself in `ai-curriculum`.
- **Hardware:** 1× H800 80GB. Experiments should have a pilot that fits this.
- **Lineage:** this grew out of the interp Thread B note [`../interp/knowledge-vs-intelligence.md`](../interp/knowledge-vs-intelligence.md) (Chollet skill-acquisition efficiency, Mahowald language≠thought, MQuAKE, substrate concept, TinyStories/Phi/R1-Zero, "is language necessary for reasoning" = no). Connects to [`../arc-agi-2-guide.md`](../benchmarks/arc-agi-2-guide.md) (Spelke core-knowledge priors) and [`../interp/open-research-questions.md`](../interp/open-research-questions.md).
- **How I like to run this research:** launch parallel GPT-5.4 (codex, web) + Claude agents, **hard-wait** for all, cross-check + **web-verify every load-bearing citation** (arXiv IDs confabulate easily), be critical / novelty-check, calibrate experiments to 1×H800. Save outputs as markdown in this directory.

---

## The framing we developed: capability factorization

```
LLM CAPABILITY  ≈  REASONING MACHINERY   ×   STORED KNOWLEDGE
                   (the "intelligence")       (the "knowledge")
        Q5 mechanism?                          Q1 what % of params?
        └──────────► Q2 reduce / Q3 build small + retrieve ◄──────┘
                     Q4 what DATA trains the small core?
```

**The cross-cutting finding that ties all 5 questions together** (verified, multi-source): *pretraining installs the reasoning machinery as latent capability; post-training mostly teaches WHEN to deploy it, not HOW to reason.* (Venhoff 2510.07364: 91% of the thinking-model gap recovered by steering 12% of tokens; Ward 2507.12638; Yue 2504.13837; Gandhi 2503.01307.) → reasoning behaves like cheap, transferable *compute/depth*; knowledge like *storage*. This asymmetry is what makes a small reasoning core conceivable — **but** the weights don't honor a clean split (entanglement, superposition), so "pure reasoning, zero knowledge" is ill-posed.

---

## Answers so far (one line each — full version in `00`)

1. **Q1:** No published work measures the knowledge/reasoning *fraction*; ~2 bits/param is a synthetic *capacity ceiling* (Allen-Zhu 2404.05405), not an allocation. Boundary is a gradient, not a partition.
2. **Q5:** Reasoning ≈ added effective depth / serial compute (looped transformers 2502.17416); can happen latently; CoT is often unfaithful. "WHEN not HOW."
3. **Q2:** Hardest — can't cleanly *subtract* knowledge (unlearning damages reasoning; R²MU 2506.12963). Compression works; broad knowledge degrades before reasoning (Minitron).
4. **Q3:** Most tractable but caps out — small models distill *procedural* reasoning well (R1-Distill, Phi-4-reasoning) but broad knowledge (GPQA) lags at every size; retrieval is often the bottleneck (BRIGHT) and multi-hop collapses (MQuAKE). Karpathy's cognitive core: aspiration, unbuilt.
5. **Q4:** "Less is more" (LIMO 817, s1 1K) is about the **base**, not the data (same 817 ex → +47 AIME by swapping base). Minimum substrate = *transformable structure* (numeracy, state-tracking, comparison, manipulation, metacognition), not facts.

---

## What's been produced (the map)

| Artifact | Contents |
|---|---|
| [`00-overview-and-field-map.md`](00-overview-and-field-map.md) | framing, field map (subfields/labs/venues), TL;DR answers, cross-cutting finding |
| [`01-knowledge-capacity-and-reasoning-mechanism.md`](01-knowledge-capacity-and-reasoning-mechanism.md) | Q1+Q5: capacity, localization limits, mechanism, faithfulness |
| [`02-small-reasoning-core-compression-retrieval.md`](02-small-reasoning-core-compression-retrieval.md) | Q2+Q3: cognitive core, unlearning, SRM benchmarks, retrieval, **falsification experiment for the H800** |
| [`03-data-for-small-reasoning-models.md`](03-data-for-small-reasoning-models.md) | Q4: less-is-more, base-vs-data, minimum substrate, concrete data recipe |
| [`proposals/01-measurement-and-mechanism.md`](proposals/01-measurement-and-mechanism.md) | 4 proposals: used-capacity tomography, edit-transport commutators, trajectory transplants, MoE audit |
| [`proposals/02-architectures-cognitive-core.md`](proposals/02-architectures-cognitive-core.md) | 4 proposals: depth-factored recurrent core, world-swap distillation, swappable knowledge shards, arbitration curriculum |
| [`proposals/03-data-curriculum-substrate.md`](proposals/03-data-curriculum-substrate.md) | 5 proposals: substrate phase-diagram ("Chinchilla for reasoning substrate"), base/data Jacobian, manipulation-first retrieval, on-policy branch-and-repair, task-local symbol-table bootstrapping |

Method note: reports = 6 agents (3 GPT-5.4 + 3 Claude, cross-checked); proposals = 3 lane-specific GPT-5.4 agents, novelty-checked (8 load-bearing citations spot-verified, all real).

---

## Where to continue (open threads / next actions)

1. **Pick a proposal → full spec + H800 repo scaffold.** My top-5 (novel × feasible × answers the questions): **1.1** Used-Capacity Tomography (measures Q1), **3.1** Matched-Entropy Substrate Phase Diagram (the "Chinchilla for reasoning substrate" = my May 24–25 thread), **1.3** Trajectory Transplants (tests "WHEN not HOW"), **2.1** Depth-Factored Recurrent Core, **3.4** On-Policy Branch-and-Repair (metacognition).
2. **Run the falsification experiment** in `02` (distill reasoning into 1–3B → suppress factual recall → measure reasoning survival → add retrieval). Cleanest single-H800 test of the whole thesis; nobody has published it cleanly.
3. **Write up the pure-reasoning scaling-law thread** (May 24–25) as its own guide — still conversation-only.
4. **Fold 6 newer papers into reports 01 & 03** (surfaced by the ideation agents, all verified): Front-Loading Reasoning (2510.03264), Interplay of Pre/Mid/RL (2512.07783), Ward repurposed-representations (2507.12638), Standing-Committee MoE (2601.03425), EMO (2605.06663), Memory Decoder (2508.09874).
5. **Deepen any single question** with more agents, or challenge the cross-cutting "WHEN not HOW" claim (proposal 1.3 is designed to potentially break it).

---

## Anchor citations (verified)

Capacity/localization: Allen-Zhu Physics of LMs 3.3 `2404.05405`, 3.1 `2309.14316`, 3.2 `2309.14402`; Geva FFN-memories `2012.14913`; ROME `2202.05262`; Hase localization≠editing `2301.04213`. Mechanism: looped transformers `2502.17416`; Coconut `2412.06769`; Venhoff "WHEN not HOW" `2510.07364`; Ward `2507.12638`; CoT-unfaithful `2505.05410`; GSM-Symbolic `2410.05229`. Build: Karpathy cognitive core (X, 2025-06-27; Dwarkesh 2025-10-30); decouple-by-pretraining (Han et al.) `2502.19402`; R²MU `2506.12963`; Minitron `2407.14679`; R1-Distill/`2501.12948`; Phi-4-reasoning `2504.21318`; Search-R1 `2503.09516`; MQuAKE `2305.14795`; BRIGHT `2407.12883`. Data: LIMO `2502.03387`; s1 `2501.19393`; Yue `2504.13837`; Gandhi `2503.01307`; SFT-memorizes-RL-generalizes `2501.17161`.
