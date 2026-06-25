# Robin: An Agentic System for Automated Scientific Discovery

How a multi-agent LLM system runs the full intellectual loop of biology research — generate hypotheses, design experiments, analyze the resulting wet-lab data, and refine — using humans only as the hands at the bench. The first such system to autonomously discover *and* experimentally validate a novel drug candidate (ripasudil for dry AMD).

## Background

**Primary paper**: [Robin: A multi-agent system for automating scientific discovery](https://arxiv.org/abs/2505.13400) (Ghareeb, Chang, Mitchener, Yiu, Szostkiewicz, Laurent, Razzak, White, Hinks, Rodriques — FutureHouse + University of Oxford, May 2025). Later published in Nature (DOI 10.1038/s41586-026-10652-y). Code: [github.com/Future-House/robin](https://github.com/Future-House/robin).

Robin is an orchestrator built on three pre-existing FutureHouse agents. The contribution is not a new model — it is wiring these into one closed loop:

1. **PaperQA2 / "Language agents achieve superhuman synthesis of scientific knowledge"** (Skarlinski et al., FutureHouse, Sep 2024, [arxiv 2409.13740](https://arxiv.org/abs/2409.13740)) — RAG agent that retrieves, summarizes, and reasons over the literature at expert level. The engine behind Robin's two literature agents, **Crow** (concise reviews) and **Falcon** (deep reviews).
2. **BixBench** (Mitchener et al., FutureHouse, Mar 2025, [arxiv 2503.00096](https://arxiv.org/abs/2503.00096)) — a benchmark for LLM agents on real bioinformatics data-analysis tasks; introduced **Finch**, the data-analysis agent Robin reuses.
3. **Aviary** (Narayanan et al., FutureHouse, Dec 2024, [arxiv 2412.21154](https://arxiv.org/abs/2412.21154)) — the agent-environment ("gymnasium") framework all three agents are instantiated and called from; the same lineage as the RL-trained language agents you've studied.
4. **ReAct** (Yao et al., 2022, [arxiv 2210.03629](https://arxiv.org/abs/2210.03629)) — Finch's think-act-observe loop runs ReAct inside a Jupyter kernel.
5. **Bradley–Terry–Luce** (Bradley & Terry, *Biometrika* 1952) — the 1952 paired-comparison model Robin uses to turn pairwise LLM judgments into a global ranking. The same model behind the RLHF reward model and your DPO loss.

**Contemporary "AI scientist" systems** Robin is positioned against: Google's **AI Co-Scientist** (Gottweis et al., Feb 2025, [arxiv 2502.18864](https://arxiv.org/abs/2502.18864)) — multi-agent Gemini that generates and debates hypotheses but stops before experimental data; **The AI Scientist** (Lu et al., Sakana, Aug 2024, [arxiv 2408.06292](https://arxiv.org/abs/2408.06292)) — end-to-end automation but only in computational ML; **Autonomous chemical research with LLMs / Coscientist** (Boiko et al., *Nature* 2023) — LLM + lab-automation hardware, chemistry-focused, no iterative biological hypothesis↔data loop.

## What Problem Does Robin Solve?

Drug repurposing has a recurring pattern: the insight needed to repurpose a drug often already exists, scattered across the literature, years before anyone connects the dots (the paper cites dabrafenib→hearing loss with a 10-year lag, ketamine→depression 22 years, leucovorin 5 years). The bottleneck is the rate at which humans can synthesize disparate scientific knowledge and close it against experiment.

Prior AI systems each automate *one* slice:

```
                   hypothesis    experiment    wet-lab     data       refine on
System             generation    design        execution   analysis   results
──────────────────────────────────────────────────────────────────────────────
AI Co-Scientist       ✓             ~             ✗           ✗          ✗
  (Google, Gemini)   (in-silico debate only — never touches data)
AI Scientist          ✓             ✓          ✓ (code)    ✓ (code)     ✓
  (Sakana)           (computational ML only — no wet biology)
Coscientist/ChemCrow  ~             ✓          ✓ (robot)     ~           ~
  (chemistry)        (synthesis automation, not bio hypothesis loops)
Robin                 ✓             ✓          humans       ✓            ✓
                     (first to couple lit-grounded hypothesis gen
                      WITH experimental-data analysis in one loop)
```

Robin's claim: it is the first to automate *all* the intellectual steps — generating hypotheses and experimental strategies, analyzing results, and refining hypotheses in light of new data — leaving humans only the manual bench work.

## Key Terms

**Assay**: a laboratory test that turns a biological question into a *number*. Concretely, a controlled procedure that measures the presence, amount, or functional activity of some target or process. In drug discovery an *in vitro* assay is the measuring instrument: you culture cells, apply a drug, and read out a quantitative signal that says whether the drug moved the mechanism you care about. Robin's chosen assay was the **RPE phagocytosis enhancement assay** — culture retinal pigment epithelium (RPE) cells, add a drug, add fluorescent pHrodo beads, and measure by flow cytometry how much the cells "eat" (the readout is mean fluorescence intensity, MFI). The assay is the proxy that links an abstract disease mechanism ("dry AMD involves failing RPE phagocytosis") to something testable. Choosing *which* assay to use is literally the first thing Robin's hypothesis loop decides — see below.

**RPE / dry AMD**: retinal pigment epithelium, the support cells that "clean up" photoreceptor debris by phagocytosis. Dry age-related macular degeneration (dAMD) is the leading cause of irreversible vision loss in the developed world (1.5M with vision-threatening dAMD in the US; no approved treatment at the time); RPE phagocytic failure is a hallmark.

**Bradley–Terry–Luce (BTL) model**: a paired-comparison model. Given many "is A better than B?" judgments, it fits one latent *strength* parameter θᵢ per item by maximum likelihood, where P(i beats j) = σ(θᵢ − θⱼ). It converts noisy pairwise wins into a single self-consistent cardinal ranking. (You already know this model — it is exactly the preference model under RLHF reward modeling and DPO, where P(prefer A) = σ(r_A − r_B).)

**Crow / Falcon / Finch**: Robin's three worker agents — concise literature search, deep literature evaluation, and experimental data analysis, respectively.

**Trajectory (Finch)**: one complete Jupyter-notebook analysis run. Robin launches many in parallel and takes the consensus.

## The Architecture

```
   INPUT                    ROBIN  (orchestrator, a Jupyter notebook on Aviary)              OUTPUT
 ┌─────────┐    ┌──────────────────────────────────────────────────────────────┐    ┌──────────────┐
 │ disease │───>│  HYPOTHESIS GENERATION  <─────feedback──────  EXPERIMENTAL     │───>│ ranked        │
 │  name   │    │  (Crow + Falcon + o4-mini + LLM-judge)        ANALYSIS (Finch) │    │ therapeutic   │
 └─────────┘    │            │                                       ▲           │    │ candidates    │
                └────────────┼───────────────────────────────────────┼───────────┘    └──────────────┘
                             v                                       │
                      LABORATORY EXPERIMENTS  (humans — the only manual step)
```

| Agent  | Built on   | Job in Robin                                                        | Model |
|--------|------------|---------------------------------------------------------------------|-------|
| Crow   | PaperQA2   | Concise literature reviews — answer pathology queries, evaluate assays | (PaperQA2) |
| Falcon | PaperQA2   | Deep literature review — full evaluation report per drug candidate  | (PaperQA2) |
| Finch  | BixBench   | Autonomous data analysis — flow cytometry, RNA-seq, in Jupyter      | ReAct agent |
| Robin  | Aviary     | Orchestrate: query gen, candidate gen, judging, result interpretation | OpenAI **o4-mini** |
| Judge  | —          | Pairwise comparison of hypotheses for BTL ranking                   | Anthropic **Claude 3.7 Sonnet** |

Implementation note: in their original agentic implementation, Robin "almost always called tools in the same order," so they **collapsed it into a deterministic Jupyter notebook** for stability — a nice example of the recurring lesson that once an agentic workflow's control flow is fixed, you should hard-code it rather than re-pay the LLM-decision tax each run.

> **Paper vs released code.** The open-source repo diverges from the paper in several ways: the LLM judge is **o4-mini, not Claude 3.7 Sonnet**; the default config is **3/3/5** (not the paper's `num_queries=5 / num_assays=10 / num_candidates=30`); Finch runs **5** parallel trajectories (not 10); BTL strengths are fit with `choix.ilsr_pairwise` (a spectral ILSR estimator, not a hand-rolled MLE); and `data_analysis` is hard-wired to flow cytometry. The numbers in this guide follow the *paper*; see the companion [`robin-architecture.md`](./robin-architecture.md) for the code-level walkthrough.

## Innovation 1 — Two-Phase Hypothesis Generation

Robin does not free-associate. Generation is a structured *generate → literature-ground → judge-rank* pipeline, run twice: first to pick the **assay** (the experimental strategy), then to pick the **drug candidates** to run through it.

```
PHASE 1 — STRATEGY / ASSAY                       PHASE 2 — THERAPEUTIC CANDIDATE
input: disease name = "dry AMD"                   input: chosen assay = "RPE phagocytosis"
  │                                                 │
  │ o4-mini → num_queries=5 broad                   │ o4-mini synthesizes a candidate-
  │   pathology queries                             │   generation GOAL string from the assay
  │     ↓ Crow answers each                          │     ↓ o4-mini → 2×num_queries queries
  │ o4-mini → num_assays=10 candidate                │     ↓ Crow answers each
  │   causal mechanisms + matched assays            │ o4-mini → num_candidates=30 single-agent
  │     ↓ Crow writes a detailed eval report         │   drug candidates (+ mechanistic hypothesis
  │       per assay                                  │   + reasoning per candidate)
  │     ↓ LLM-judge pairwise tournament → BTL        │     ↓ Falcon writes a deep eval report
  └──> TOP ASSAY defines the strategy ──────────────┘       per candidate
                                                    │     ↓ LLM-judge pairwise tournament → BTL
                                                    └──> RANKED candidate list
                                                          → humans pick top ~5 to test
```

What makes this notable versus earlier "AI hypothesis" work:

- **Hierarchical decomposition** — it separates *what to measure* (assay) from *what to test* (drug). Most prior systems jump straight to "propose a drug." Robin first commits to an experimental proxy for the disease, which constrains and grounds the candidate search.
- **Every hypothesis is literature-grounded before it is judged.** A raw o4-mini idea is never ranked on its own; it is first expanded into an evidence report by Crow/Falcon (real retrieval over the literature), and the *report* is what the judge sees. This pushes the judgment from "does this sound plausible" toward "is the cited evidence strong."
- **The generator prompts encode scientific priors.** The actual candidate-generation system prompt (Supplementary Fig S8) demands each proposal score on Strong Target Validation, Relevant Preclinical/Clinical Evidence, Mechanistic Specificity, and Novelty-balanced-with-validation, and forces single-agent, commercially-available molecules with catalog numbers — i.e., it is tuned to surface *actionable, repurposable* drugs, not exotic ones.

Scale of one run (typical config `num_queries=5, num_assays=10, num_candidates=30`): for dAMD, Robin reviewed **151 papers** to propose the 10 mechanisms, then **~400 papers** to propose the 30 candidates.

## Innovation 2 — Why Bradley–Terry–Luce Instead of Just Asking the Judge to Score?

This is the question worth dwelling on, because the naive design is "ask the LLM to rate each hypothesis 1–10 and sort." Robin instead has the judge make **pairwise** comparisons and fits a BTL model. The benefit over direct scalar scoring:

**1. Pairwise discrimination is far more reliable than absolute scoring — for LLMs and humans alike.** Absolute scores have no stable anchor: the model's "7/10" drifts with prompt wording, ordering, and batch; scores pile up in a narrow band (everything looks like a 7–8); calibration is poor. "Which of these two is better?" is a strictly easier, lower-variance task. This is the same reason RLHF reward models and Chatbot Arena are built on pairwise preferences, not Likert scores — and the same reason your DPO setup trains on (chosen, rejected) pairs rather than absolute reward labels.

**2. BTL recovers a principled global ranking even from inconsistent comparisons.** Pairwise judgments contain cycles (A≻B, B≻C, C≻A) and noise, so you cannot just sort them. BTL fits θᵢ per hypothesis by MLE under P(i≻j)=σ(θᵢ−θⱼ), aggregating all the pairwise outcomes into one self-consistent cardinal score — and it tells you *how much* better, not just the order.

**3. It works from incomplete / sampled comparisons — which is what makes it scale.** Bradley & Terry's 1952 title is literally "Rank Analysis of *Incomplete* Block Designs." Robin exploits this directly:

```
   ≤ 25 hypotheses  →  full round-robin   (all C(n,2) pairs)
   > 25 hypotheses  →  sample 300 random pairwise comparisons, then fit BTL
```

For 30 candidates a full round-robin is C(30,2)=435 judge calls; the count explodes quadratically as the set grows, so sampling 300 pairs and letting BTL estimate the strengths keeps the judge-call budget bounded while still giving a statistically grounded ranking.

**4. Repeated comparisons average out judge noise.** Each hypothesis appears in many pairs, so the MLE pools that evidence — one anomalous judgment can't sink an item the way a single bad absolute score would.

**The judge prompt is itself meta-generated.** Rather than hand-writing the judge's rubric, the authors had domain experts perform their own pairwise comparisons of Robin's hypotheses, then fed those expert preferences to **Gemini 2.5 Pro Preview** to *write the judge prompt* — distilling expert decision criteria into the prompt instead of guessing them. (The resulting prompts, Supp S5/S10, weight evidence strength > mechanism clarity > safety > feasibility > novelty.)

**It was validated, not assumed.** The Claude-3.7 judge's top-10 hypotheses overlapped the experts' top-10 by an average of **7.25/10** (vs **3.33** for random selection), and the judge was *more* self-consistent than humans: presented with identical pairwise comparisons twice, the LLM picked the same winner **88.4%** of the time vs **61.1%** for human experts. Higher intra-rater consistency than the domain experts is the quiet headline here.

## Innovation 3 — The Result-Conditioned Refinement Loop

This is the part no prior system had: experimental results re-enter hypothesis generation as context.

```
 Finch analyzes wet-lab data
        │
        ▼
 Robin distills a STRUCTURED insight blob:
   • Data interpretation   ("among tested compounds, ROCK inhibition by Y-27632 …")
   • Mechanistic insights  ("… may facilitate actin cytoskeleton remodeling")
   • Questions raised       ("what molecular mechanisms underlie …")
   • Proposed follow-up assays ("RNA-seq on Y-27632-treated RPE …")
        │
        ▼
 blob is appended into the next candidate-generation prompt
   (Phase 2 re-runs, now CONDITIONED on prior wet-lab results)
        │
        ▼
 cycle repeats until a human is satisfied with a candidate
```

Crucially this is **in-context conditioning, not fine-tuning** — no weights change; the prior round's distilled findings simply become part of the next generation prompt. Concretely, the round-1 RNA-seq insight (ROCK inhibition + ABCA1/actin involvement) is what led Robin to propose **ripasudil** in round 2 — a more potent, clinically-approved ROCK inhibitor never previously proposed for dAMD.

## Innovation 4 — Finch's Parallel-Consensus Data Analysis

Biological data analysis is judgment-laden (where do you draw flow-cytometry gates? which genes count as differentially expressed?), and an LLM agent's choices vary run-to-run from sampling stochasticity. Instead of fighting that variance, Robin **harvests** it:

```
 raw .fcs / RNA-seq data
        │
        ├─► Finch trajectory 1  ─┐   each = an independent ReAct agent in its own
        ├─► Finch trajectory 2   │   Docker (BixBench-env:v1.0) Jupyter kernel,
        ├─► …  (up to 10)        │   tools = {edit_cell, submit_answer}
        └─► Finch trajectory N  ─┘
                  │
                  ▼
        META-ANALYSIS across trajectories → keep the CONVERGENT findings
```

Robin can launch up to **10 independent Finch trajectories** on the same dataset, each writing and executing its own notebook, then meta-analyzes them for consensus — turning the LLM's stochasticity into an ensemble that both *explores* diverse analytical paths and *delivers* only results that are robust across runs. For the headline RNA-seq result, the consensus over **8 trajectories** showed Finch identified the same genes as significantly differentially expressed in over 50% of trajectories (Fig 3C). Finch's gating (it uses the `flowMeans` R package + k-means clustering to find the main cell population) was checked against an independent human analysis and matched.

## The Concrete dAMD Discovery

```
 ROUND 1 ─ Strategy + first screen
   151 papers → 10 dAMD mechanisms → pick "enhance RPE phagocytosis" assay
   ~400 papers → 30 candidates → top 5 tested:
     Exendin-4, Fingolimod, MFGE8, Y-27632, AICAR+TUDCA
   Assay: ARPE-19 cells, 60 min drug pre-treat, +pHrodo beads, 3 h, flow cytometry (MFI)
     (Robin suggested photoreceptor outer segments; humans used pHrodo beads for availability)
   → HIT: Y-27632, a ROCK inhibitor

 ROUND 1.5 ─ Mechanism (Robin proposes a follow-up assay)
   RNA-seq of Y-27632-treated RPE → Finch (DESeq2, GO enrichment, 8-trajectory consensus)
   → ABCA1 (lipid-efflux pump) upregulated ~3-fold, adjusted p = 2.13×10⁻⁸³
   → GO: actin filament organization, small-GTPase signaling, autophagy
   → novel coupled mechanism: better phagocytosis + better lipid efflux (both fail in dAMD)

 ROUND 2 ─ Refined candidate
   Result-conditioned regeneration → propose ripasudil (approved glaucoma ROCK inhibitor)
   10 drugs tested → ripasudil beats Y-27632:
     +7.5× phagocytosis vs DMSO (Finch analysis) / +1.75× (human analysis)
```

The whole concept-to-submission cycle took **~2.5 months** with a small team. Robin also produced all the main-text data figures itself. As a generality demonstration, Robin generated 10 ranked candidates each for 10 *other* diseases (PCOS, celiac, Charcot-Marie-Tooth, IPF, NASH, sarcopenia, age-related hearing loss, glaucoma, Friedreich's ataxia, CKD — Supp S16–S25).

## Limitations (stated by the authors)

- **No executable protocols.** Robin outputs experimental *outlines*; humans still translate them into runnable wet-lab protocols. This is the main gap to full autonomy.
- **Finch needs per-modality prompt engineering.** Reliable analyses still depend on expert-written prompts; Finch can't yet autonomously adapt to an arbitrary new data type. (Its standalone BixBench score is modest — agent tooling lifts it well above a bare model, but bioinformatics remains hard.)
- **The "discovery" is literature synthesis, not de novo biology.** ROCK-inhibition→phagocytosis had prior support (the lit search surfaced a single paper showing Y-27632 restores phagocytosis in low-phagocytic RPE), and ABCA1/ABCA4 biology was known in macular degeneration. Robin's power is *connecting* existing insights faster than humans — valuable, but not inventing new mechanisms from nothing.
- **In-vitro only, single domain validated.** Ripasudil is a candidate, not a proven therapy; disease-model and clinical validation remain. The 7.5× (Finch) vs 1.75× (human) gap on the same data is itself a verification caution.
- **Evaluation alignment is open.** The authors flag that better aligning hypothesis generation/evaluation with human scientific judgment is future work.

## Robin vs Alternatives

| System            | Domain         | Hypotheses | Wet-lab data analysis | Refines on results | Validated discovery |
|-------------------|----------------|------------|-----------------------|--------------------|---------------------|
| Robin             | wet biology    | ✓ (2-phase, BTL-ranked) | ✓ (Finch, consensus) | ✓ (in-context)     | ✓ (ripasudil, in-vitro) |
| AI Co-Scientist   | biomedical     | ✓ (debate/evolve) | ✗                  | ✗ (no data)        | partial (some wet follow-ups by others) |
| AI Scientist (v1/v2) | computational ML | ✓       | ✓ (code only)         | ✓                  | ✓ (a workshop paper) |
| Coscientist       | chemistry      | ~          | ~                     | ~                  | ✓ (synthesis, robotic) |

## Practical Takeaways

- The reusable engineering ideas here are model-agnostic and transfer to any LLM-judge or agentic-research setup you build: (1) **pairwise + BTL beats absolute scoring** whenever you need to rank LLM-generated candidates; (2) **ground each candidate in retrieval before judging it**; (3) **meta-generate your judge prompt from expert preferences** rather than hand-writing a rubric; (4) **run N stochastic analysis trajectories and take consensus** to convert sampling noise into robustness; (5) **collapse a stabilized agentic workflow into deterministic code** once tool-call order is fixed.
- When *not* to reach for this: anything needing genuinely novel mechanism (Robin recombines known literature), or where the experimental readout can't be reduced to a clean quantitative assay.

## Key Papers

1. Ghareeb et al. *Robin: A multi-agent system for automating scientific discovery.* arXiv 2505.13400 (2025); Nature 10.1038/s41586-026-10652-y. https://arxiv.org/abs/2505.13400
2. Skarlinski et al. *Language agents achieve superhuman synthesis of scientific knowledge* (PaperQA2). arXiv 2409.13740 (2024). https://arxiv.org/abs/2409.13740
3. Mitchener et al. *BixBench: a comprehensive benchmark for LLM-based agents in computational biology.* arXiv 2503.00096 (2025). https://arxiv.org/abs/2503.00096
4. Narayanan et al. *Aviary: training language agents on challenging scientific tasks.* arXiv 2412.21154 (2024). https://arxiv.org/abs/2412.21154
5. Bradley & Terry. *Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons.* Biometrika 39(3/4):324–345 (1952).
6. Yao et al. *ReAct: Synergizing Reasoning and Acting in Language Models.* arXiv 2210.03629 (2022). https://arxiv.org/abs/2210.03629
7. Gottweis et al. *Towards an AI Co-Scientist.* arXiv 2502.18864 (2025). https://arxiv.org/abs/2502.18864
8. Lu et al. *The AI Scientist.* arXiv 2408.06292 (2024). https://arxiv.org/abs/2408.06292
9. Boiko et al. *Autonomous chemical research with large language models.* Nature 624:570–578 (2023).
