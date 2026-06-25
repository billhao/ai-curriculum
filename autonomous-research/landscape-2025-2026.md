# Autonomous Research Landscape, 2025–2026

A ranked map of the "AI scientist" / autonomous-research field as of mid-2026 — end-to-end discovery systems, hypothesis generation and validation, self-driving labs, the frameworks and models underneath, the benchmarks that measure them, and the peer-review milestones and critiques. Domain-general: covers ML, biology, chemistry/materials, and the social sciences, not just biomedicine.

Ranking weighs real-world impact and adoption, novelty, rigor of validation (independently verified vs vendor-claimed), and influence. Maturity tags: `demo` (proof-of-concept), `benchmarked` (evaluated on a public benchmark), `peer-reviewed` (passed external review), `deployed` (in real use). Entries flagged `[unverified]` rest on vendor PR rather than an independently checkable paper/review.

## Must-read shortlist (top of the field)

```
# System / paper            Org            Date      What it is                              Maturity        arXiv / link
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
1 The AI Scientist v1 / v2  Sakana AI      Aug'24 /  Fully autonomous ML research loop;      peer-reviewed   2408.06292 /
                                           Apr'25    v2 = first AI paper to pass workshop     (workshop)      2504.08066
                                                     peer review; v1 later in Nature
2 AI Co-Scientist           Google         Feb'25    6 specialist agents + Supervisor;        demo + 3        2502.18864
                            DeepMind                  generate-debate-evolve, Elo tournament;  wet-lab
                                                     3 wet-lab-validated biomed results        validations
3 Robin                     FutureHouse     May'25    Closed lab-in-the-loop; first to         peer-reviewed   2505.13400
                                                     autonomously discover + validate a        (Nature)
                                                     drug candidate (ripasudil, dry AMD)
4 Popper                    Stanford        Feb'25    Domain-general agentic hypothesis        benchmarked     2502.09858
                                                     VALIDATION; sequential e-values give
                                                     any-time Type-I error control
5 data-to-paper             Technion        Apr'24    Data→hypothesis→code→full paper with     benchmarked     2404.17605
                                                     verifiable claim-to-data provenance
6 PaperBench                OpenAI          Apr'25    Benchmark: replicate an ML paper from    benchmark       2504.01848
                                                     scratch; 8,316 gradable subtasks
```

If you read four systems: **AI Scientist v2** (the reference point + the peer-review milestone), **AI Co-Scientist** (largest effort, strongest validations), **Robin** (the only wet-lab-validated discovery), **Popper** (the rigor the generators lack). Then skim **PaperBench/MLE-bench** to understand how the field is scored.

## 1 — End-to-end autonomous discovery systems

```
Rank System            Org           Date    Domain      Loop covers                          Maturity
──────────────────────────────────────────────────────────────────────────────────────────────────────
1   AI Scientist v2    Sakana        Apr'25  ML          idea→tree-search experiments→paper   peer-reviewed
      [arxiv 2504.08066]                                 →auto-review; agentic tree search     (workshop)
2   AI Scientist v1    Sakana        Aug'24  ML          idea→Aider code→paper→reviewer;      peer-reviewed
      [arxiv 2408.06292]                                 ~$15/paper; now in Nature             (Nature)
3   AI Co-Scientist    Google        Feb'25  biomed      hypothesis gen + Elo tournament;     demo + wet-lab
      [arxiv 2502.18864]                                 NO experiment loop (open loop)
4   Robin              FutureHouse   May'25  wet biology  gen→BTL judge→wet-lab(humans)→       peer-reviewed
      [arxiv 2505.13400]                                 Finch analysis→refine; closed loop    (Nature)
5   data-to-paper      Technion      Apr'24  data sci    data→hypothesis→code→paper w/         benchmarked
      [arxiv 2404.17605]                                 provenance; human co-pilot for hard
6   Agent Laboratory   AMD           Jan'25  ML          PhD/Postdoc/Professor role hierarchy benchmarked
      [arxiv 2501.04227]                                 over lit-review→experiment→report
7   AI-Researcher      (academic)    2025    ML          ~9-agent mentor-student network;     demo
      [arxiv id unverified]                              end-to-end; flag: confirm ID/claims
8   EvoScientist       (academic)    Mar'26  general     multi-agent "evolving" scientists    demo
      [arxiv 2603.08127]                                 for end-to-end discovery
9   Denario            (academic)    Oct'25  general     "deep knowledge" AI agents for        demo
      [arxiv 2510.26887]                                 multi-step scientific discovery
```

Pattern of the field: architectures range from **single-pipeline** (AI Scientist v1) → **role-based teams** (Agent Laboratory's PhD/Postdoc/Professor) → **specialist-agent networks** (AI Co-Scientist's 6 agents + Supervisor; AI-Researcher's mentor-student). The frontier divide is **open-loop** (hypothesis only — AI Co-Scientist) vs **closed-loop** (acts on real results — Robin wet-lab, AI Scientist computational).

## 2 — Hypothesis generation + validation

```
Rank Work               Org        Date    Contribution                                    Maturity
──────────────────────────────────────────────────────────────────────────────────────────────────
1   Popper              Stanford   Feb'25  Agentic sequential falsification; e-values +    benchmarked
      [arxiv 2502.09858]                   e-processes → frequentist Type-I≤α, any-time
                                           valid. The validation layer for any generator.
2   AI Co-Scientist     Google     Feb'25  generate-debate-evolve; tournament w/ online    demo + wet-lab
      (gen component)                       Elo ranking via simulated scientific debate
3   Robin (gen)         FutureHouse May'25 2-phase lit-grounded gen; Bradley-Terry-Luce    peer-reviewed
                                           judge tournament; result-conditioned refine
```

Key contrast for ranking hypotheses: **Robin = batch BTL MLE** over a fixed comparison set (single-turn judgments); **AI Co-Scientist = online sequential Elo** with multi-turn debate as the comparison operator. Both share the logistic σ(θᵢ−θⱼ) kernel (same family as RLHF/DPO reward models). Neither emits a calibrated false-positive rate — that gap is exactly what **Popper** fills (a Type-I≤α verdict on hypotheses it did not generate). Natural pipeline: **generate → rank → Popper adjudicates → spend on wet-lab/policy**.

## 3 — Self-driving labs & autonomous experimentation (physical loop)

```
Rank System            Org          Date    Domain        Contribution                       Maturity
──────────────────────────────────────────────────────────────────────────────────────────────────────
1   GNoME              DeepMind     Nov'23  materials     2.2M predicted crystals; 736       deployed
                                                          synthesized/confirmed in labs       (context, pre-2025)
2   A-Lab              Berkeley     Nov'23  materials     Bayesian-opt + LLM-guided robotic   deployed
                                                          synthesis; closed-loop              (context, pre-2025)
3   Coscientist        CMU          2023    chemistry     GPT-4 + robotic liquid handlers;    deployed
      (Boiko, Nature)                                     plans + executes reactions          (context, pre-2025)
4   SDL 2.0 / "AI       RSC/Royal    2025-26 chem/mat     reviews + "AI advisor" shared-       review/
      advisor" wave      Society                          control SDLs; flexible, scalable    deployed
```

These predate the 2025–26 LLM-agent wave but are the *physical* counterpart to Robin's wet lab — robotics-in-the-loop closing synthesis→characterize→decide. The 2025–26 trend ("SDL 2.0") is wiring LLM reasoning agents on top of existing robotic platforms.

## 4 — Frameworks, infrastructure & scientific models

```
Work          Org          Date    Role                                              Status
─────────────────────────────────────────────────────────────────────────────────────────────
PaperQA2      FutureHouse  Sep'24  Superhuman literature synthesis (agentic RAG);    active (8.7k★)
  [2409.13740]                     the "read the field" primitive
Aviary        FutureHouse  Dec'24  Gym for language agents — env abstraction +       active
  [2412.21154]                     RL/expert-iteration TRAINING of science agents
```

(Robin/Crow/Falcon/Finch are built on these — see `robin-architecture.md`.) The broader trend is treating literature-synthesis and agent-environments as reusable substrate rather than per-project glue.

## 5 — Benchmarks & evaluation

```
Benchmark          Org        Date    What it measures                                    Notes
──────────────────────────────────────────────────────────────────────────────────────────────────
MLE-bench          OpenAI     Oct'24  ML-engineering on 75 offline Kaggle competitions    vs human medalists
  [2410.07095]
RE-Bench           METR       Nov'24  7 open-ended ML research-engineering tasks           human-vs-agent time
PaperBench         OpenAI     Apr'25  Replicate an ML paper from scratch; 8,316 rubric    hardest of the set
  [2504.01848]                        subtasks graded
CORE-Bench         Princeton  2024    Reproduce a paper given its repo (install→run→read) reproduction, not
                                                                                          replication
ScienceAgentBench  OSU        Oct'24  Data-driven discovery tasks across disciplines      data-to-insight
  [2410.05080]
BixBench           FutureHouse Mar'25 Open-ended bioinformatics data analysis (Finch)     ~22% agentic SOTA
  [2503.00096]
DiscoveryBench     AI2        Jul'24  Data-driven hypothesis discovery, 6 domains         used by Popper
MLR-Bench          (academic) May'25  Open-ended ML research                              open-ended
  [2505.19955]
AstaBench          AI2        Oct'25  Rigorous agent suite across the research workflow   broad coverage
  [2510.21652]
```

Takeaway for picking one: **MLE-bench / RE-Bench** for ML-engineering agents; **PaperBench / CORE-Bench** for replication vs reproduction; **ScienceAgentBench / DiscoveryBench / BixBench** for data-driven scientific reasoning. The field is benchmark-rich but the headline numbers stay low (e.g. BixBench ~22% agentic), which is itself the story: autonomous *analysis* is far from solved.

## 6 — Peer-review milestones & critiques

Milestones:
- **AI Scientist-v2 paper passed ICLR 2025 workshop review** (scores 6/7/6, avg 6.33, ~45th percentile, workshop "I Can't Believe It's Not Better"). First fully AI-generated, peer-reviewed-accepted workshop paper. ([Sakana](https://sakana.ai/ai-scientist-first-publication/))
- **The AI Scientist (v1) published in *Nature*** (2026). ([Sakana](https://sakana.ai/ai-scientist-nature/))
- **Robin published in *Nature*** (ripasudil/ABCA1, dry AMD).
- **Zochi (Intology)** and **Carl (Autoscience)** — startup claims of autonomously-authored workshop-accepted papers in 2025. `[unverified]` — vendor PR; not independently checkable to the same degree as Sakana's pre-registered ICLR experiment.

Critiques and limitations (the necessary counterweight):
- **TechCrunch / community pushback on the Sakana milestone**: a workshop (60–70% accept) is not the main track (20–30%); Sakana itself said none of the 3 papers met its internal bar for a main-track accept; the accepted paper had citation errors (e.g. misattributing LSTM). Paper was withdrawn after review per the pre-agreement. ([TechCrunch](https://techcrunch.com/2025/03/12/sakana-claims-its-ai-paper-passed-peer-review-but-its-a-bit-more-nuanced-than-that/))
- *"Evaluating Sakana's AI Scientist… Wishful Thinking or Emerging Reality?"* ([arXiv 2502.14297](https://arxiv.org/abs/2502.14297)).
- *"Why LLMs Aren't Scientists Yet: Lessons from Four Autonomous Research Attempts"* (Jan 2026, [arXiv 2601.03315](https://arxiv.org/pdf/2601.03315)).
- *A Survey of AI Scientists* (Oct 2025, [arXiv 2510.23045](https://arxiv.org/pdf/2510.23045)) — taxonomy and open problems.

## Open problems

1. **Validation rigor lags generation.** Generators (AI Scientist, Robin, Co-Scientist) vastly out-produce any calibrated verification. Popper is the exception, not the norm.
2. **"Discovery" is mostly literature recombination.** Robin, Co-Scientist, and others connect existing insights faster than humans — genuine *de novo* mechanism remains rare.
3. **Autonomous data analysis is weak.** BixBench ~22% agentic SOTA; Finch needs per-modality prompting; provenance (data-to-paper) is a partial fix, not a guarantee of correctness.
4. **Wet-lab / physical execution still human (or robotic-island).** No system both *designs* and *runs* arbitrary wet experiments; the bench is the bottleneck.
5. **Evaluation integrity & hype.** Vendor peer-review claims, contamination, and cherry-picking make independent benchmarks (and pre-registered experiments like Sakana's ICLR deal) essential.

## Sources

Sakana AI Scientist v2 peer-review: [sakana.ai](https://sakana.ai/ai-scientist-first-publication/), [TechCrunch](https://techcrunch.com/2025/03/12/sakana-claims-its-ai-paper-passed-peer-review-but-its-a-bit-more-nuanced-than-that/), [phys.org](https://phys.org/news/2026-03-ai-paper-peer.html) · AI Scientist v2 [arXiv 2504.08066](https://arxiv.org/abs/2504.08066) · PaperBench [arXiv 2504.01848](https://arxiv.org/pdf/2504.01848) · MLE-bench [arXiv 2410.07095](https://arxiv.org/pdf/2410.07095) · AstaBench [arXiv 2510.21652](https://arxiv.org/pdf/2510.21652) · MLR-Bench [arXiv 2505.19955](https://arxiv.org/html/2505.19955v3) · Survey of AI Scientists [arXiv 2510.23045](https://arxiv.org/pdf/2510.23045) · "Why LLMs Aren't Scientists Yet" [arXiv 2601.03315](https://arxiv.org/pdf/2601.03315) · self-driving labs review [Royal Society](https://royalsocietypublishing.org/rsos/article/12/7/250646/235354/) · GNoME / materials [phys.org](https://phys.org/news/2025-12-ai-advisor-labs-creation-generation.html)

> Confidence note: system papers (AI Scientist, Co-Scientist, Robin, Popper, data-to-paper) and the major benchmarks are verified against their arXiv records. Startup peer-review claims (Zochi, Carl) and the AI-Researcher arXiv ID are flagged `[unverified]` — confirm before citing. The pre-2025 self-driving-lab entries (GNoME, A-Lab, Coscientist) are included as physical-loop context, not as 2025–26 releases.
