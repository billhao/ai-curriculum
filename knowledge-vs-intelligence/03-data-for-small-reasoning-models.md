# 03 — What Data Trains a Small Reasoning Model

*Covers Q4 ("what data is necessary to train such a small reasoning model?"). The short version: the headline "N examples is enough" results are statements about the **base model**, not the data — so the real question splits into "what must pretraining install" vs. "what little post-training data elicits it."*

---

## The "less is more" results (and their fine print)

The lineage starts with **LIMA — "Less Is More for Alignment"** (Zhou et al., Meta, NeurIPS 2023, arXiv:2305.11206): 1,000 curated examples on LLaMA-65B, origin of the **Superficial Alignment Hypothesis** — (C1) knowledge is learned in pretraining; (C2) a few examples saturate task performance; (C3) post-training teaches *style/format*, not new capability.

Applied to **reasoning**:

| Result | Data | Base | Numbers | Source |
|---|---|---|---|---|
| **LIMO** "Less Is More for Reasoning" | **817** curated math traces | Qwen2.5-32B-Instruct | 63.3 AIME'24 / 95.6 MATH-500 (v3); beats 100×-larger SFT sets | arXiv:2502.03387 (COLM'25) |
| **s1** "Simple test-time scaling" | **1,000** traces (from 59K pool) + budget forcing | Qwen2.5-32B-Instruct | beats o1-preview by up to 27%; budget forcing 50→57 AIME | arXiv:2501.19393 |
| **LIMR** "Less is More for RL" | **1,389** RLVR samples (vs 8,523 full) | 7B | matches/beats full set | arXiv:2502.11886 |
| **1-shot RLVR** | **1** example (GRPO) | Qwen2.5-Math-1.5B | MATH-500 36.0 → 73.6 | arXiv:2504.20571 (NeurIPS'25) |

s1's selection criteria — **difficulty + diversity + quality** — are the active ingredient: random-1K and the *full 59K* both underperform the curated 1K, and "1K-longest" captures most of the gain (trace length/format does heavy lifting). LIMO rejects items the base can already solve.

### The fine print that reframes everything

> **It's the base, not the data.** LIMO's own ablation: the **same 817 examples** on Qwen1.5-32B-Chat vs Qwen2.5-32B-Instruct shifts results by **+47.1 AIME / +34.4 MATH-500.** So "817 examples teach reasoning" really means "817 examples *unlock* reasoning the base already installed." LIMO's own hypothesis says minimal post-training works *only when the base model's domain knowledge is already comprehensively encoded.* **[HIGH]**

This is the single most important correction to Q4: separate the **post-training floor** (≈10³ curated traces) from the **total data floor** (still dominated by pretraining).

---

## The base sets the ceiling (the convergent evidence)

Multiple independent results say post-training mostly *elicits/reweights/stabilizes* latent capability rather than *adding* it — the same mechanism as Venhoff (see [01 §B.3](01-knowledge-capacity-and-reasoning-mechanism.md)):

- **Yue et al., "Does RL Really Incentivize Reasoning Beyond the Base Model?"** (arXiv:2504.13837, NeurIPS'25): under **pass@k**, RLVR models win at k=1 but the **base model wins at large k** — RLVR sharpens sampling, doesn't expand the reasoning boundary. Crucially, **distillation *can* add genuinely new patterns** (teacher is stronger); RLVR cannot exceed the base. **[HIGH]**
- **Gandhi et al., "Cognitive Behaviors that Enable Self-Improving Reasoners"** (arXiv:2503.01307, COLM'25): the cleanest base-dependence demo — **Qwen base shows the four behaviors (verification, backtracking, subgoaling, backward-chaining) 62% of the time; Llama only 10%.** Same RL self-improves Qwen, stalls Llama — *until* Llama gets continued pretraining on behavior-rich math (OpenWebMath), after which it catches up. **The data floor is in pretraining.** **[HIGH]**
- **Chu et al., "SFT Memorizes, RL Generalizes"** (arXiv:2501.17161, ICML'25): small-data SFT overfits / fails OOD; RL generalizes — but **SFT is still required first to stabilize output format** before RL works. **[HIGH]**
- **Spurious Rewards** (arXiv:2506.10947) / **RLVR Implicitly Incentivizes Correct Reasoning** (arXiv:2506.14245): even random/weak reward signals can move Qwen-math, because the behavior is latent — the controversy over "what RLVR actually does" is real. **[MEDIUM]**
- **Counterpoint worth holding:** Han et al. (arXiv:2502.19402, see [02](02-small-reasoning-core-compression-retrieval.md)) argue elicitation-from-a-knowledge-rich-base is exactly the trap — reasoning "overfits to training data and is limited in transferability" because knowledge and reasoning are coupled; they advocate **reward-based pretraining from the get-go**. This is the dissenting view to "just distill into a strong base." **[verified position paper]**

---

## What *kind* of data (composition & type)

### Distilled reasoning traces — quality and targeting beat raw count
- Distillation from a strong teacher (R1, QwQ, o3-mini-class) is the workhorse for SRMs. **Quality > quantity; skill-targeted > random.**
- But **breadth still needs scale**: **AM-DeepSeek-R1-Distilled-1.4M** (arXiv:2503.19633) shows simple SFT on 1.4M *verified* traces beats released R1-Distill baselines; **OpenThoughts3** scaled to ~1.2M for SOTA-7B. So: small curated sets *trigger* reasoning on a strong base; large clean corpora *maximize* coverage/transfer.

### RLVR data — verifiable, curriculum-ordered, surprisingly small
- Use RLVR only where verifiers are reliable: **math, code, constrained logic, symbolic planning.**
- **Selection > volume:** **DEPO** (arXiv:2509.01321) — 20% of RLVR data (selected by difficulty/diversity/influence/explorability) gives 1.85×/1.66× AIME'24/'25 training speedups over full-data GRPO. **[MEDIUM]**
- **Difficulty curriculum matters** when the base is weaker — Light-R1 (arXiv:2503.10460) orders data easy→hard across SFT/DPO/RL.
- You already know the GRPO machinery; the lesson here is **data selection and curriculum, not the optimizer.**

### Allen-Zhu's controlled "Physics of LMs" results — what teaches reasoning vs storage
- **Part 2.1, Grade-School Math** (arXiv:2407.20311, ICLR'25): models learn **genuine reasoning, not template memorization** — they form a mental plan *before* emitting the first token, and depth (not just CoT length) governs multi-step capability. Reasoning is a *skill* learned from process-structured data. **[HIGH]**
- **Part 2.2, Learning from Mistakes** (arXiv:2408.16293): putting **error→correction directly in pretraining/continued-pretraining data** beats clean-only data — and self-correction is a *distinct skill*, hard to bolt on late with LoRA. **[HIGH]**
- **Part 3.1 / 3.2** (storage≠extraction; retrieval≠manipulation, see [01 §A.3](01-knowledge-capacity-and-reasoning-mechanism.md)): even with retrieval supplying facts, the model needs trained **manipulation** machinery — so reasoning data is non-negotiable even in a retrieval-heavy design.

### Metacognition data (planning / backtracking / self-correction)
- **Teachable as behavior:** Gandhi shows the four behaviors can be *instilled* by data; strikingly, **traces with correct *behaviors* but wrong *answers* train about as well as correct ones** — echoing s1/LIMO (process/format > final-answer correctness). The "Wait"/backtracking style transfers.
- **But intrinsic reliability is contested:** Huang et al., "LLMs Cannot Self-Correct Reasoning Yet" (arXiv:2310.01798, ICLR'24) — without external feedback, self-correction often *degrades* accuracy. Reliable error-*detection* is better instilled by **RLVR's verifier signal** than by pure SFT.
- Cross-link: this is the training-data side of your [wait-token analysis](../interp/wait-token-analysis-pipeline.md) — performative vs load-bearing backtracking. You can teach the *style* cheaply; making it *causal* needs a verifier.

---

## The minimum viable knowledge substrate

The most useful reframing from the research: **the minimum substrate is not "facts" — it's transformable structure.** What a small reasoning model irreducibly needs in its weights:

```
NOT NEEDED in weights          NEEDED in weights (the substrate)
(offload to retrieval)         ────────────────────────────────────
─────────────────────          • numeracy / arithmetic
• long-tail entities           • dependency & state tracking
• encyclopedic facts           • comparison / classification / inverse-search
• domain knowledge tails       • multi-step state updates
• anything web-lookuppable     • knowledge EXTRACTION under paraphrase (Part 3.1)
                               • knowledge MANIPULATION / CoT (Part 3.2)
                               • metacognition: plan / backtrack / verify
                               • enough core schemas to reason OVER retrieved facts
```

This maps onto your existing Thread B "substrate" definition (world knowledge + symbolic fluency + computational machinery) and onto ARC's **Spelke core-knowledge priors** (objectness, number, geometry, agentness — see the [ARC-AGI 2 guide](../arc-agi-2-guide.md)). The open quantitative question — **how big must this substrate be as the world it reasons over grows** — is the deepest unknown (it's #2 in [interp/open-research-questions.md](../interp/open-research-questions.md)).

---

## A concrete data recipe (synthesis, not a single paper)

For a small (1.5–8B) reasoning model that retrieves facts externally. Confidence **MEDIUM** — this is inferred from s1 / LIMO / DeepSeek-R1 / Allen-Zhu, not a published ratio.

```
PRETRAIN / CONTINUE-PRETRAIN  ── install the substrate, not the world
  • behavior-rich math/code/proof text (OpenWebMath-style) so backtracking
    & verification appear naturally  (Gandhi: this is where the floor really is)
  • paraphrase/diversity augmentation so knowledge is EXTRACTABLE (Part 3.1)
  • include error→correction sequences (Part 2.2)
  • do NOT try to memorize long-tail facts — that's retrieval's job
        │
        ▼
STAGE 1 — SFT (elicitation):  ~1K–17K  hard, diverse, VERIFIED long-CoT traces
  • teacher STRONGER than student (R1/QwQ) → adds real patterns (Yue)
  • select by difficulty+diversity+quality (s1) AND skill-coverage of weak skills
  • keep error/verification/backtracking traces; drop easy & low-quality
  • (scale toward ~1.2M distilled traces ONLY if the base lacks the skill)
        │
        ▼
STAGE 2 — RLVR (generalization + reliable verification):
  • ~1K–2K verifiable math/code/logic problems, difficulty curriculum
    (LIMR/DEPO: selection beats volume); GRPO/PPO
        │
        ▼
STAGE 3 — small general/instruction/TOOL-USE + retrieval mixture
  • restores assistant behavior; teaches when to search (SEM) and how to
    reason over retrieved docs (Search-o1 "reason-in-documents")

Rough post-training mixture:  60–80% verifiable math/code/logic
                              10–20% explicit correction/verification/reflection
                              10–20% general instruction / tool / retrieval
```

### What is genuinely unknown (the data-floor frontier)
- The real floor for **weak bases below ~7B**, for **non-math domains**, and for bases **without Qwen/DeepSeek-style math-rich pretraining** — the literature is immature here. **[LOW]**
- Whether reasoning learned this way *transfers* out of distribution, or overfits to the distilled traces (the Han et al. critique; "Rethinking Generalization in Reasoning SFT," arXiv:2604.06628 — argues generalization is conditional on base strength). **[2026 — flag]**
- The size-of-substrate ↔ reasoning-ceiling scaling law — **no Chinchilla-for-reasoning-substrate exists** (ties to your [May 24–25 pure-reasoning scaling-law thread](../interp/knowledge-vs-intelligence.md)).

---

## Confidence summary (Q4)

| Claim | Confidence |
|---|---|
| LIMO/s1 work *because of the base*, not the data (base-swap +47) | HIGH |
| Post-training floor ≈10³ curated traces; real floor is pretraining | HIGH |
| RL sharpens latent behavior; distillation can add new patterns | HIGH |
| Base must already show cognitive behaviors (Qwen 62% / Llama 10%) | HIGH |
| Quality/skill-targeting > raw count (SFT & RLVR) | HIGH (direction) |
| Error-correction & metacognition teachable as behavior | HIGH |
| Reliable intrinsic self-correction without a verifier | LOW (contested) |
| Minimum substrate = transformable structure, not facts | HIGH (synthesis) |
| Concrete mixture ratios above | MEDIUM (inferred) |
| Data floor for weak/non-math bases | LOW (unknown) |

Back to the [overview & field map](00-overview-and-field-map.md).
