# Knowledge vs. Intelligence — Overview & Field Map

*A research deep-dive on separating an LLM's reasoning ability from its stored world-knowledge, and building a small "reasoning core." Continuation of the [Thread B notes](../interp/knowledge-vs-intelligence.md). Compiled May 2026 from 6 parallel research agents (3× GPT-5.4 codex + 3× Claude, one pair per cluster, cross-checked).*

> **Resuming this research in a new session?** Start with [`RESEARCH-CHARTER.md`](RESEARCH-CHARTER.md) — it holds the original questions, motivations, context, and where to pick up.

---

## The five questions, as one thesis

You asked five questions. They are one thesis seen from different sides — **capability factorization**: treat an LLM's competence as (roughly) *reasoning machinery* × *stored knowledge*, and ask whether those factors can be measured, separated, and recombined as **{small reasoning core} + {external knowledge}**.

```
LLM CAPABILITY  ≈  REASONING MACHINERY   ×   STORED KNOWLEDGE
                   (the "intelligence")       (the "knowledge")
──────────────────────────────────────────────────────────────────
 made of        planning, backtracking,    facts, entities,
                in-context learning,        domain tails,
                induction & algorithmic     memorized corpora
                circuits, verification
──────────────────────────────────────────────────────────────────
 your question  Q5: what IS the            Q1: what % of params
                mechanism?                  live here?
──────────────────────────────────────────────────────────────────
 recombine →    Q2  reduce big model   = keep machinery, drop knowledge
                Q3  build small model   = small machinery + retrieve knowledge
                Q4  what DATA trains that small machinery?
```

This directory has three deep files plus this overview:

| File | Covers | Your questions |
|---|---|---|
| [01-knowledge-capacity-and-reasoning-mechanism.md](01-knowledge-capacity-and-reasoning-mechanism.md) | The *science*: how much is knowledge, where it lives, what reasoning *is* computationally | Q1, Q5 |
| [02-small-reasoning-core-compression-retrieval.md](02-small-reasoning-core-compression-retrieval.md) | The *engineering*: cognitive core, unlearning/pruning, small reasoning models, retrieval | Q2, Q3 |
| [03-data-for-small-reasoning-models.md](03-data-for-small-reasoning-models.md) | The *data*: minimal/synthetic reasoning data, the data floor, a concrete recipe | Q4 |

---

## TL;DR — the answer to each question (as of May 2026)

**Q1 — How much of a frontier model's parameters are knowledge?**
Your intuition ("a lot must be knowledge") is **directionally right but unprovable as stated.** No published work measures the knowledge-vs-reasoning *fraction* of a real frontier model. The best anchor is Allen-Zhu & Li's **~2 bits/parameter** *capacity* law (arXiv:2404.05405) — but that's a ceiling on synthetic data, not a measurement of allocation. ~2/3 of transformer params are MLP, and MLPs are the established factual store — but MLPs also *compute*. **Confidence: the clean split does not exist; the boundary is a gradient, not a partition.**

**Q5 — What is the mechanism of reasoning?**
Reasoning is best modeled as **added serial computation / effective depth over shared representations**, not a separate "reasoning store." CoT is one way to buy serial steps (a scratchpad); looped/recurrent-depth transformers buy them latently. The keystone 2025 result: **base models already contain the reasoning machinery; post-training mostly teaches *when* to deploy it** (Venhoff, arXiv:2510.07364 — 91% of the thinking-model gap recovered by steering 12% of tokens, no weight updates).

**Q2 — Reduce a frontier model to reasoning + necessary knowledge?**
**Hardest direction.** You cannot cleanly *subtract* knowledge: unlearning methods that remove facts also damage reasoning (entanglement; R²MU arXiv:2506.12963). Pruning+distillation works (NVIDIA Minitron) and notably, **broad knowledge (MMLU) degrades before reasoning under compression** — weak evidence the factors are unequally fragile. But surgical knowledge removal with reasoning intact is unsolved.

**Q3 — Build a small reasoning model + web search?**
**Most tractable direction, but feasibility is MEDIUM-LOW for broad tasks.** Distillation puts strong *procedural* reasoning into 1.5–32B models (R1-Distill, Phi-4-reasoning). On math/code, small models rival frontier; on broad knowledge (GPQA/MMLU-Pro/HLE) they lag at *every* size. Search-RL (Search-R1, R1-Searcher) lets small models use retrieval well — but the **retrieval itself is often the bottleneck** (BRIGHT: best retriever 22.1 nDCG@10), and **multi-hop reasoning over retrieved facts collapses** (MQuAKE). Karpathy's "cognitive core" (a ~1B reasoning kernel that looks facts up) is a stated aspiration; **no published system has cleanly demonstrated it.**

**Q4 — What data trains such a small reasoner?**
The "less is more" results (LIMO 817 examples, s1 1K traces) are **statements about the base model, not the data.** Same 817 examples on Qwen1.5 vs Qwen2.5 = **+47 points AIME**. The post-training data floor is ~10³ curated traces; the *real* floor lives in pretraining. The minimum substrate is **not facts — it's transformable structure**: numeracy, dependency tracking, comparison, multi-step state updates, plus metacognition (backtracking/verification) that is teachable as *behavior* but whose *reliability* needs RLVR.

**The honest bottom line:** Reasoning *is* more compressible and more transferable than knowledge — the mechanism (compute/depth, reused circuits) supports your hope. But "pure reasoning, zero knowledge" is ill-posed: you cannot reason about chemistry without chemistry, and retrieval doesn't remove the need to *manipulate* retrieved facts. The realistic endpoint is exactly what you intuited and what the frontier is already converging on: **a small reasoning core + a necessary-knowledge substrate + external retrieval/tools** — what MIT's Han et al. (arXiv:2502.19402) call decoupling, and Karpathy calls the cognitive core.

---

## What field is this?

There is **no single field** — it's a research *program* at the intersection of six areas, plus a cognitive-science backbone. Verified subfields, lead labs, and venues:

| Subfield | What it contributes | Lead labs / people | Venues |
|---|---|---|---|
| **Knowledge capacity & localization** | how much knowledge, where it's stored | Meta FAIR / **Allen-Zhu, Yuanzhi Li** (Physics of LMs); **Bau Lab** (ROME/MEMIT); Tel Aviv (Geva) | ICLR, NeurIPS, ICML, ACL |
| **Mechanistic interp / science of reasoning** | what reasoning *is*; faithfulness | **Anthropic** (interp + alignment science); DeepMind (Nanda, Geva); academic MI | transformer-circuits.pub, NeurIPS, ICLR |
| **Efficient / small reasoning models** | how small can you reason | DeepSeek, **Microsoft (Phi)**, Berkeley NovaSky, Bespoke/OpenThoughts | COLM, NeurIPS, arXiv |
| **Model compression** | pruning + distillation that preserves capability | **NVIDIA (Minitron/Nemotron)** | arXiv, NeurIPS |
| **Machine unlearning / knowledge editing** | can you subtract knowledge | alignment/safety groups (RMU, R²MU, MQuAKE) | ICML, ICLR, NeurIPS |
| **Data-centric reasoning** | what data elicits reasoning | **GAIR-NLP** (LIMO/LIMR), Stanford (s1), Allen-Zhu (controlled) | COLM, ICML, EMNLP |
| **Retrieval-augmented & agentic reasoning** | externalize knowledge, RL for tools | UIUC (Search-R1), RUC (R1-Searcher) | NeurIPS, EMNLP, SIGIR |

**Conceptual backbone — cognitive science of LLMs:** "language ≠ thought" (Mahowald/Fedorenko), Chollet's *skill-acquisition efficiency*, Spelke core knowledge. (Already covered in your [Thread B notes](../interp/knowledge-vs-intelligence.md) and the [ARC-AGI 2 guide](../benchmarks/arc-agi-2-guide.md).)

**Emerging umbrella labels you'll see:** "the science of LLMs" / "Physics of Language Models" (the controlled-experiment program); "capability / knowledge-reasoning *decoupling*"; on the build side, "small reasoning models (SRMs)" and "efficient reasoning." Karpathy's **"cognitive core"** is the popular name for the target artifact.

---

## The cross-cutting finding (why all three clusters agree)

Independently, the science / build / data clusters converged on the **same mechanism**, which is the thread tying your five questions together:

```
            Pretraining installs the reasoning MACHINERY (latent)
                              │
        ┌─────────────────────┼─────────────────────┐
   Venhoff 2510.07364   Yue 2504.13837        Gandhi 2503.01307
   steer 12% tokens →   RL sharpens sampling, base shows behaviors
   recover 91% gap      doesn't add capability  Qwen 62% / Llama 10%
        └─────────────────────┼─────────────────────┘
                              ▼
   Post-training (SFT/RLVR/distill) mostly teaches WHEN to deploy,
   reweights toward latent good trajectories, stabilizes format.
   Distillation from a STRONGER teacher is the one cheap way to add
   genuinely new reasoning patterns the base lacks.
```

This is good news for your program: if reasoning is **latent machinery the base already has**, and post-training is mostly *elicitation*, then a small model with the right pretraining substrate + a little curated data + RLVR can punch far above its parameter count **on domains its substrate covers.** The catch is the substrate: it must already span the world the reasoning operates over (TinyZero's 0.5B Countdown failure; LIMO's base-swap +47). That substrate — not encyclopedic facts — is the irreducible core.

---

## How this was researched (confidence conventions)

Every claim in these files traces to a web-searching agent's verified citation. Per-cluster I ran **two independent researchers** (GPT-5.4 + Claude) and kept claims they *both* found at higher confidence. I separately web-verified the single-sourced keystones (Venhoff 2510.07364 ✓; corrected 2502.19402's title; R²MU 2506.12963 ✓).

Confidence tags in the files: **HIGH** (verified, multi-source) · **MEDIUM** (verified existence, contested interpretation or non-frontier conditions) · **LOW** (single-source, or numbers from blogs/model-cards) · **2026/VERY_NEW** (recent preprint, abstract-verified only — confirm before relying).

**Known caveat to carry everywhere:** Allen-Zhu's "2 bits/param" and the Physics-of-LMs results are on **controlled synthetic corpora at 10⁶–10⁹ params**, not audits of GPT-4/Claude/Gemini. Treat them as mechanism, not measurement.

---

## Where to go next

- Deep dives: [01 science](01-knowledge-capacity-and-reasoning-mechanism.md) · [02 build](02-small-reasoning-core-compression-retrieval.md) · [03 data](03-data-for-small-reasoning-models.md)
- Origin of this thread: [interp/knowledge-vs-intelligence.md](../interp/knowledge-vs-intelligence.md) (substrate concept, TinyStories/Phi/R1-Zero, language-not-needed-for-reasoning)
- Open experiments worth running on your H800: see the falsification test in [02](02-small-reasoning-core-compression-retrieval.md) and the ranked questions in [interp/open-research-questions.md](../interp/open-research-questions.md)
- Benchmark context: [arc-agi-2-guide.md](../benchmarks/arc-agi-2-guide.md) (fluid reasoning, Spelke priors), [benchmarks-guide.md](../benchmarks/benchmarks-guide.md) (HLE = the knowledge tail)
