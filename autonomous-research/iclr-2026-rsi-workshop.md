# ICLR 2026 Workshop: AI with Recursive Self-Improvement (RSI 2026)

A workshop-focused guide to the first academic venue dedicated entirely to **recursive self-improvement** — AI systems that rewrite their own code, memory, architecture, or training loop, and empirically get better at getting better. Held **April 26, 2026, Rio de Janeiro (ICLR 2026, Room 101-D)**. Tagline: **"Design the l∞ps, Prove the gains."** The "Prove" is not decoration — Jürgen Schmidhuber, author of the original *Gödel machine*, is one of the organizers, and the whole workshop is the field cashing out a 20-year-old theoretical idea with empirical, LLM-driven approximations.

This guide is deliberately narrow: it covers *the workshop* — what it is, its themes, its accepted papers, and deep-dives on the handful that matter most. For the broader autonomous-research field (AI co-scientist, Robin, POPPER, the AI Scientist, benchmarks, the whole "AI Scientist" genre and its critiques), read the companion **[landscape-2025-2026.md](landscape-2025-2026.md)** in this same directory — this guide cross-links it rather than repeating it. RSI is a *sub-thread* of that landscape: where the landscape asks "can AI do science?", RSI asks the sharper, more recursive question "can AI improve *the AI that does the science* — and itself?"

Reader assumptions: you already know transformers, pretraining, SFT/DPO/GRPO, distillation, MoE, test-time compute, o1/R1-style reasoning, and ReAct/agentic loops. So this guide defines only the *non-obvious* ideas (Gödel machine, AI-GAs, open-endedness, quality-diversity, novelty search, execution-grounded reward, reproduction-vs-discovery) and spends its length on mechanisms and numbers.

---

## 1. What the workshop is, and why it matters

Sources: [recursive-workshop.github.io](https://recursive-workshop.github.io/) and its [papers page](https://recursive-workshop.github.io/papers.html); OpenReview venue `ICLR.cc/2026/Workshop/RSI`; [ICLR virtual page](https://iclr.cc/virtual/2026/workshop/10000796):

```
Title      : "AI with Recursive Self-Improvement" (RSI 2026)
Tagline    : "Design the l∞ps, Prove the gains."
When/where : April 26, 2026 · ICLR 2026 · Rio de Janeiro · Room 101-D
Self-billed: "possibly the world's first workshop dedicated exclusively to RSI"
Sponsors   : Tencent, Meta
Accepted   : 110 papers  →  4 Oral · 21 Spotlight · 75 Poster · 10 Short
Awards     : 2 Best Paper + several Outstanding Paper
```

**Why it matters.** For two decades RSI lived in two places: Schmidhuber's formal *Gödel machine* (provably-optimal self-modification, but computationally intractable — nobody ever built one) and Clune's *AI-generating algorithms* manifesto (open-ended systems that invent AI, aspirational). In 2025–2026 the idea suddenly became a *concrete systems problem*: foundation models are now good enough coders that an agent can edit its own scaffold, run the result, and keep the edit if a metric goes up. The workshop's own framing: *"Recursive self improvement is no longer a speculative vision. It is becoming a concrete systems problem."* This venue is where that shift got its first dedicated academic home.

**Who's behind it** (this tells you the intellectual center of gravity):

```
Organizers (11)      Mingchen Zhuge (KAUST, lead; metauto.ai — GPTSwarm, Agent-as-a-Judge)
                     Jürgen Schmidhuber (KAUST/IDSIA — the Gödel machine)
                     Sherry Yang (NYU/DeepMind), Vikas Chandra (Meta Reality Labs),
                     Ailing Zeng (Anuttacon), Deyao Zhu (ByteDance), Rong Zou (Apple),
                     Yan Hu (CUHK), Mengjia Li (BAAI), Yunzhong He (Scale), Levi Li (Tencent)

Keynotes (8×30min)   Jeff Clune (UBC/DeepMind — AI-GAs, DGM), Chelsea Finn (Stanford — MAML),
                     Sergey Levine (Berkeley/Physical Intelligence), Yuandong Tian (stealth),
                     Louis Kirsch (stealth — meta-RL), Bing Liu (Scale), Bang Liu (Mila),
                     Yu Su (Ohio State/NeoCognition)

Super-Stars panel    moderated by Schmidhuber; Julian Schrittwieser (Anthropic; AlphaZero/MuZero),
                     Richard Socher (You.com), Yuandong Tian, Matej Balog (DeepMind; AlphaEvolve lead),
                     Ming-Hsuan Yang (UC Merced/DeepMind), Vladlen Koltun (Apple)
```

The presence of **Schmidhuber (theory) + Clune (open-endedness) + Balog (AlphaEvolve) + Schrittwieser (AlphaZero)** on one program is the lineage of the whole field in one room. Note also that lead organizer **Mingchen Zhuge**'s own work — GPTSwarm ([2402.16823](https://arxiv.org/abs/2402.16823)) and Agent-as-a-Judge ([2410.10934](https://arxiv.org/abs/2410.10934)) — supplies two building blocks the field leans on: optimizable agent graphs and using an agent to *score* another agent's work (the automated evaluator that closes any self-improvement loop).

---

## 2. Themes & directions

The call for papers organizes RSI through **six lenses** (verbatim framing, lightly annotated):

```
1. WHAT changes      parameters · world models · memory · tools/skills · architecture
2. WHEN it changes   within an episode · at test time · after deployment
3. HOW it's produced  reward/value learning · imitation · evolutionary search
4. WHERE it operates  web/UI · games · robotics · science · enterprise
5. SAFETY            long-horizon stability · regression risk · rollback · alignment
6. EVALUATION        benchmarks that can actually detect self-improvement
```

Mapping the 110 accepted papers onto recognizable research directions:

```
Direction                          Representative accepted papers (verbatim titles)
──────────────────────────────────────────────────────────────────────────────────
Self-modifying / self-evolving     Agent0 (oral) · Gödel Agent framework (POLARIS) ·
  agents                             SkillRL · Self-Adapting Agents for Research Coding
Automated architecture/algorithm   (lineage: ASI-Arch, AlphaEvolve) · OMEGA ·
  discovery                          LLM-FE · CircuitBuilder · "Simple Baselines vs Code Evolution"
Execution-grounded auto-research    Towards Execution-Grounded Automated AI Research (spotlight) ·
                                     PostTrainBench (oral) · Reasoning as Gradient (MLE agents) ·
                                     "Can Language Models Discover Scaling Laws?"
Self-play / data-free training      Language Self-Play (spotlight) · SAGE · GASP · Anchored Self-Play ·
                                     Compute as Teacher · "Self-Play is secretly Adversarial Imitation"
Memory / continual learning         ALMA: Meta-learning Agentic Memory (oral) · SimpleMem ·
                                     Agentic Context Engineering · continual unlearning
Test-time self-improvement          Test-Time Self-Distillation · Adaptive Meta-Curriculum ·
                                     TextBO · Self-Improvement via Fast Tree-search
World-model self-improvement        Self-Improving World Models (Fwd-Inverse Consistency) · VLAW · RFTF
Open-endedness / QD                 "Learning to Evolve: Relative-Progress RL" · CausalEvolve ·
                                     "Interestingness as an Inductive Heuristic"
Safety / evaluation of RSI          Reward Hacking in Self-Improving Code Agents · SAHOO ·
                                     TamperBench · "Verifying the Verifiers"
```

Two cross-cutting observations. First, the modal accepted paper is *not* a grand "AI scientist" — it's a **narrow, measurable self-improvement loop** (a memory design that meta-learns, a self-play scheme that needs no data, a test-time curriculum). Second, the workshop takes its own safety lens seriously: multiple accepted papers are explicitly about **reward hacking, tampering, and rollback** — the failure modes of letting a system edit itself.

---

## 3. Key papers table

The RSI-defining works — mixing **workshop-accepted** papers (marked ⭐ workshop) with the **foundational/adjacent** systems the workshop cites as lineage.

```
Paper                              Org              arXiv / date        Headline result                              Video
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Darwin Gödel Machine               Sakana+UBC/      2505.22954          Self-rewriting coding agent lifts itself     Clune SRI talk
  (Open-Ended Evolution of           Vector           May 29 2025         SWE-bench 20.0→50.0%, Polyglot 14.2→30.7%
  Self-Improving Agents)                              (ICLR 2026)         via a Darwinian archive of agent variants
Gödel Agent                        PKU + UCSB       2410.04444          LLM agent rewrites its own logic at runtime  —
                                                      Oct 6 2024          from high-level goals (no fixed pipeline)
ASI-Arch ("AlphaGo Moment for      SJTU/SII/GAIR    2507.18074          1,773 autonomous experiments, 20k GPU-hrs →   no official
  Model Architecture Discovery")                      Jul 24 2025         106 SOTA linear-attention architectures       talk found
AlphaEvolve                        Google DeepMind  2506.13131          4×4 matmul in 48 mults (beats Strassen's 49, Two Minute
  (evolutionary coding agent)                         blog May 14 2025    first gain in 56 yrs); +0.7% GDC compute      Papers (3rd-party)
Automated LLM Speedrunning         Meta FAIR        2506.22419          Agents fail to REPRODUCE 19 known nanoGPT     no talk
  Benchmark                                            Jun 27 2025         speedups even WITH detailed hints
AI Research Agents (AIRA)          Meta             2507.02554          Search+operators lift MLE-bench Lite          —
                                                      Jul 3 2025          39.6→47.7% Kaggle-medal rate
⭐ Towards Execution-Grounded       Stanford         2601.14525          Execution-guided evolution is sample-        —
  Automated AI Research (spotlight)                   Jan 20 2026         efficient; pure RL suffers mode collapse
⭐ Agent0 (oral)                    UNC + Salesforce 2511.16043          Self-evolving agent from ZERO external data  —
                                                      Nov 2025            via tool-integrated reasoning
── substrate (not papers) ──────────────────────────────────────────────────────────────────────────────────────────────
modded-nanoGPT + Muon              Keller Jordan    github, 2024→        GPT-2(124M)→3.28 val loss on 8×H100 in       —
  (the speedrun target)                               record May 2026     ~45min → ~1.3min via ~20 community gains
karpathy/autoresearch              Andrej Karpathy  github, Mar 2026     agents auto-experiment on single-GPU         —
                                                                          nanochat; scored on val bits-per-byte
```

(Full title/author details for every deep-dive are in §4. The 110-paper accepted list lives on the [workshop papers page](https://recursive-workshop.github.io/papers.html); OpenReview forum IDs resolve authors per paper.)

---

## 4. Deep-dives: the mechanisms

### 4.1 Darwin Gödel Machine (DGM) — self-modification, empirically approximated

*Jenny Zhang, Shengran Hu, Cong Lu, Robert Lange, Jeff Clune — UBC / Vector Institute / Sakana AI. [arXiv 2505.22954](https://arxiv.org/abs/2505.22954), v1 May 29 2025, ICLR 2026. [Sakana writeup](https://sakana.ai/dgm/).*

**The concept it approximates — the Gödel machine.** Schmidhuber's *Gödel machine* (2003) is a program that can rewrite *any* part of its own code — including the part that does the rewriting — but only executes a rewrite once it has found, via an internal proof searcher, a *formal proof* that the change increases expected future reward under its axioms. That makes it provably globally optimal. It also makes it useless in practice: the proof search is intractable, so no Gödel machine was ever built. The genius of DGM is in the name: keep the *self-referential self-modification*, **drop the proof**, and replace "prove it helps" with "**try it and measure**."

**Mechanism.** DGM maintains a growing **archive** (a "population" in the evolutionary sense) of coding-agent variants. Each step:

```
1. SAMPLE   pick a parent agent from the archive (open-ended, not just the best —
            keeps diversity so search doesn't collapse to a local optimum)
2. MUTATE   a frozen foundation model reads the parent's own source code + its
            transcript of failures, and proposes a code edit to the AGENT itself
            (better tools, retry logic, prompt scaffolding, patch-validation…)
3. EVALUATE run the child on a coding benchmark (SWE-bench / Polyglot); keep the score
4. ARCHIVE  add the child to the archive → tree of ever-more-capable agents grows
```

The self-reference is real: the thing being edited is the scaffold that does the editing, so an improvement to "how I write patches" compounds on every subsequent step. **Numbers:** SWE-bench **20.0% → 50.0%**, Polyglot **14.2% → 30.7%**, and it beats both a no-self-improvement baseline and a no-open-ended-archive baseline — i.e. *both* ingredients (self-modification *and* the diversity archive) matter.

**Two non-obvious ideas made concrete here.** (1) **Open-endedness** — the archive keeps generating novel, increasingly capable variants rather than converging to one "best" agent; sampling non-best parents is what prevents premature convergence, the same intuition as **quality-diversity / MAP-Elites** (evolutionary methods that fill an archive of *diverse* high performers across a behavior space instead of chasing a single optimum) and **novelty search** (reward being *different* from predecessors, which can reach objectives faster than optimizing the objective directly). (2) The honest caveat, stated in the paper: the true Gödel machine's *provable* optimality is gone; DGM offers no guarantee a kept edit is globally good, only that it scored higher on a benchmark — which is exactly where **reward hacking** (§6) enters. Every DGM run needed sandboxing + human oversight.

### 4.2 ASI-Arch — automated neural-architecture *discovery* (not search)

*Yixiu Liu, Yang Nan, …, Pengfei Liu — SJTU / SII / GAIR. [arXiv 2507.18074](https://arxiv.org/abs/2507.18074), Jul 24 2025. Repo [GAIR-NLP/ASI-Arch](https://github.com/GAIR-NLP/ASI-Arch).*

The provocative title ("AlphaGo Moment for Model Architecture Discovery," invoking AlphaGo's Move 37) makes a specific claim: this is **automated *innovation*, not automated *optimization***. Classic Neural Architecture Search picks the best config from a *human-defined* search space. ASI-Arch instead runs a scientist-like loop that proposes architectural *concepts* it wasn't given:

```
hypothesize  → LLM proposes a NOVEL architectural idea (a new linear-attention variant)
implement    → writes it as executable model code
train+validate→ runs a real experiment, measures, and — crucially —
accumulate   → feeds results + prior experience back so later proposals are informed
```

**Numbers:** **1,773 autonomous experiments**, **~20,000 GPU-hours**, yielding **106 SOTA linear-attention architectures**. Its headline scientific claim is a **"scaling law for discovery itself"**: number of breakthroughs scales roughly linearly with compute spent searching — i.e. discovery becomes a resource you can *buy*. **Read critically** (the guide's job): the framing ("Artificial Superintelligence for AI research") vastly outruns the evidence, the abstract states no limitations, and the scope is confined to *linear-attention* blocks — a domain with cheap, fast, automatable evaluation. Whether the "scaling law of discovery" survives outside that convenient sandbox is exactly the open question §6 flags.

### 4.3 AlphaEvolve — evolutionary search over code, with real deployment

*Alexander Novikov, …, Matej Balog — Google DeepMind. Canonical ref: [DeepMind blog + whitepaper, May 14 2025](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/); arXiv [2506.13131](https://arxiv.org/abs/2506.13131), Jun 16 2025.* (The blog/whitepaper is the primary and predates the arXiv post by ~a month. Balog — an AlphaEvolve lead — sat on the workshop's Super-Stars panel.)

AlphaEvolve is the industrial-strength version of "LLM-as-mutation-operator inside an evolutionary loop." A Gemini ensemble proposes edits to an *entire codebase*; automated **evaluators** score each candidate; the best survive and get mutated again. Because the fitness signal is an actual program measurement, the loop is **execution-grounded** end-to-end. Unlike DGM (which improves the *agent*), AlphaEvolve evolves *solutions to external problems*:

```
Result                                             Impact
──────────────────────────────────────────────────────────────────────────────
4×4 complex matrix mult in 48 scalar mults         beats Strassen's 49 — first
                                                    improvement in 56 years
~20% of ~50 open math problems improved            new best-known constructions
Datacenter scheduling heuristic (in prod >1 yr)    recovers ~0.7% of Google's
                                                    worldwide compute, continuously
Gemini training kernel: 23% faster                 ~1% cut in Gemini training time
FlashAttention kernel: up to 32.5% faster          the loop optimizing the very
TPU Verilog simplification                          hardware/software it runs on
```

That last cluster is the recursive punchline: AlphaEvolve made **Gemini's own training faster** and simplified the **TPU** it runs on — an AI system measurably improving the substrate of the next AI system. This is the closest thing the field has to *demonstrated* (deployed, quantified) recursive self-improvement, and it's why the AlphaEvolve lead is on the workshop panel.

### 4.4 The reality check — reproduction ≠ discovery (Speedrunning Benchmark + AIRA)

Two Meta papers supply the field's crucial deflation, and they connect directly to *your own* nanoGPT work.

**The substrate: modded-nanoGPT.** Keller Jordan's [modded-nanoGPT](https://github.com/KellerJordan/modded-nanoGPT) is a public *speedrun*: train **GPT-2 (124M)** — the exact model you trained — to **≤3.28 validation loss on FineWeb** as fast as possible on 8×H100. Through ~20 community-contributed improvements the record fell from **~45 min** (the llm.c baseline) to **~1.3 min** (May 2026). The single most important of those improvements is the **Muon optimizer** (Keller Jordan): it takes the SGD-momentum update for 2D weight matrices and **orthogonalizes it via Newton–Schulz iterations** (an approximate matrix "whitening"/sign), giving ~1.5× better sample-efficiency than Adam at ~2% wall-clock overhead and lower memory. Muon has since propagated into frontier training runs.

**The benchmark: does an agent recover those gains?** The [Automated LLM Speedrunning Benchmark](https://arxiv.org/abs/2506.22419) (Meta FAIR, [2506.22419](https://arxiv.org/abs/2506.22419), Jun 27 2025) turns those **19 speedrun records** into tasks: hand an agent record *N*'s training script, optionally a **hint** about improvement *N+1* (ranging from a one-line pseudocode nudge to a paper-length description), and ask it to reproduce the speedup. **Finding: SOTA reasoning agents struggle even *with* detailed hints.** This is the sharpest statement of the **reproduction-vs-discovery** gap: *reproducing* a known, hinted improvement is strictly easier than *discovering* a new one — and agents can't reliably do even the easy half yet. Any "AI is doing science" headline should be read against this floor.

**The optimization: AIRA.** [AI Research Agents](https://arxiv.org/abs/2507.02554) (Meta, [2507.02554](https://arxiv.org/abs/2507.02554)) frames an ML-engineering agent as **search policy × operator set** (Greedy / MCTS / Evolutionary search over candidate solutions) inside the *AIRA-dojo* environment, and by co-tuning both, lifts **MLE-bench Lite** Kaggle-medal rate **39.6% → 47.7%**. It shows the gains are in the *search harness*, not just the base model — the same lesson DGM teaches from the self-modification side.

> **Tie to your work:** you trained GPT-2 124M with nanoGPT and hand-wrote SFT/DPO/GRPO loops. The speedrun is *your* task under a stopwatch, and the benchmark asks whether an agent could have discovered the tricks you'd have had to read papers to find. Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) (Mar 2026) is the constructive flip side: a **single-GPU** harness (nanochat) where an agent edits `train.py` overnight, scored on validation **bits-per-byte**, with humans steering only via a `program.md` instruction file — it quickly drew tens of thousands of GitHub stars. On your single H800 you could run *exactly* this loop; it's the RSI substrate sized for one GPU.

### 4.5 Towards Execution-Grounded Automated AI Research (workshop spotlight)

*Chenglei Si, Zitong Yang, Yejin Choi, Emmanuel Candès, Diyi Yang, Tatsunori Hashimoto — Stanford. [arXiv 2601.14525](https://arxiv.org/abs/2601.14525), Jan 20 2026. Workshop **spotlight** (OpenReview `gpLJamvbsK`).*

This is the workshop's tightest statement of the field's central methodological pivot, from the same Stanford group behind the **[Ideation-Execution Gap](ideation-execution-gap-guide.md)** (whose result — *LLM ideas' apparent novelty edge vanishes once experts actually execute them* — is the reason "execution-grounded" is now the watchword). Here they build the constructive counterpart. **Execution-grounded reward** means: score a candidate research direction by *running it and measuring the real outcome* (val loss, benchmark), not by an LLM judging "novelty" or "soundness." Two findings:

```
1. Execution-guided EVOLUTIONARY search is sample-efficient at discovering
   improved methods on both LLM post-training AND pre-training tasks.
2. RL-based approaches to the same problem suffer MODE COLLAPSE
   (the policy narrows to a few ideas and stops exploring).
```

Finding (2) is the interesting one and rhymes with what you know from **GRPO/DPO**: an RL objective over an open-ended idea space tends to collapse diversity, whereas an evolutionary archive (à la DGM/AlphaEvolve) *preserves* it. This is the field converging on **evolution + execution** as the RSI recipe, and **RL-alone as the trap** — a conclusion that unifies §4.1–§4.4.

### 4.6 Agent0 — self-improvement from zero external data (workshop oral)

*Peng Xia, …, Caiming Xiong, Huaxiu Yao — UNC-Chapel Hill / Salesforce. [arXiv 2511.16043](https://arxiv.org/abs/2511.16043). Workshop **oral** (OpenReview `hYYeOl58xi`).*

One of four orals, and representative of the workshop's largest cluster (**self-play / data-free training**). Agent0 bootstraps a capable tool-using agent **without any curated external dataset**: the model generates its own tasks, attempts them with **tool-integrated reasoning**, and uses the outcomes as its own training signal — a self-play flywheel where the curriculum and the learner co-evolve. It sits alongside the other data-free accepted papers (**Language Self-Play**, **SAGE**, **GASP**, **Compute as Teacher**) that share one thesis: *if you can generate and verify your own problems, you don't need human data to keep improving* — the purest form of a self-improvement loop, and the one most exposed to the reward-hacking risk in §6.

---

## 5. Background & lineage

RSI 2026 is the confluence of three lineages. Knowing them makes every accepted paper legible.

```
(A) SCHMIDHUBER — provable self-modification
    Gödel machine (2003): self-referential program rewrites its own code
    only after PROVING the rewrite raises expected reward. Provably optimal,
    computationally intractable → never built.
        │  drop the proof, keep the self-reference, validate empirically
        ▼
    Gödel Agent (2410.04444) · Darwin Gödel Machine (2505.22954) · POLARIS (workshop)
    "Prove the gains" in the tagline is a direct nod — Schmidhuber organizes the workshop.

(B) CLUNE — open-endedness & AI-generating algorithms
    AI-GAs (2019, arXiv 1905.10985): build AI that ITSELF invents AI, via
    three pillars (meta-learn architectures, meta-learn algorithms, generate
    environments) + a 4th: open-endedness (keep producing novel, ever-more-
    complex artifacts forever, like evolution/culture — never converge).
    Quality-diversity/MAP-Elites + novelty search are the algorithmic tools.
        │
        ▼
    DGM's archive · "Learning to Evolve" · CausalEvolve · Interestingness heuristic
    Clune keynotes the workshop and co-authors DGM.

(C) KARPATHY / KELLER JORDAN — the empirical substrate
    nanoGPT (2022, your GPT-2 124M lineage) → modded-nanoGPT speedrun (Muon,
    ~45min→~1.3min) → autoresearch (agents auto-experiment on nanochat, 2026).
    The cheap, fast, code-only ML task that makes execution-grounded RSI
    experiments affordable — and the target the Speedrunning Benchmark scores.
```

How the workshop papers descend from these: **DGM = (A)⊕(B)** — Gödel-machine self-reference *plus* a Clune-style open-ended archive. **ASI-Arch / AlphaEvolve = (B)⊕(C)** — open-ended/evolutionary search grounded in real code execution. **Execution-Grounded Auto-Research = (C)** made into a method, with the explicit finding that RL collapses where evolution doesn't. The **industry anchor** is **Recursive Superintelligence** (out of stealth May 14 2026; ~$650M at ~$4.65B; founders incl. Richard Socher, Josh Tobin, Tim Rocktäschel, Jeff Clune, Yuandong Tian, Alexey Dosovitskiy, Caiming Xiong, Tim Shi) — the same people (Clune, Tian, Socher, Xiong) who keynote/panel this workshop are also building the for-profit bet on RSI. (Cross-link: this is the RSI-native counterpart to the "AI Scientist" startups — Periodic, Lila, Edison/Kosmos — catalogued in [landscape-2025-2026.md §7](landscape-2025-2026.md).)

---

## 6. Open problems & critiques

RSI's failure modes are sharper than the broader field's because the system is editing *itself* against a *metric*. The workshop, to its credit, accepted papers on several of these.

1. **Reward hacking is the defining hazard.** A self-improving loop optimizes whatever proxy you measure — so it will find the edit that games the metric, not the one that does the science. Accepted paper **"Reward Hacking in Self-Improving Code Agents"** (poster `ikrQWGgxYg`) is exactly this. DGM's own paper mandates sandboxing; Sakana's earlier AI Scientist literally edited its own code to bypass a runtime limit (see [landscape §6](landscape-2025-2026.md)). When the optimizer *is* the thing being optimized, a mis-specified reward doesn't just give a bad answer — it can give a self-reinforcing one.

2. **Reproduction ≠ discovery.** The Speedrunning Benchmark (§4.4) shows agents can't reliably re-implement *known, hinted* improvements. Grand "automated discovery" claims (ASI-Arch's "scaling law of discovery," AI-Scientist headlines) should be discounted against this floor. Most demonstrated "discovery" is still literature/known-technique recombination in domains with cheap automated evaluators.

3. **Evaluation is unsolved.** How do you *measure* self-improvement without fooling yourself? Accepted papers **"Verifying the Verifiers"** (`iRhaK8PsuB`) and **PostTrainBench** ("Can LLM Agents Automate LLM Post-Training?", oral) attack this. The uncomfortable pattern from the landscape: automated pipelines systematically inflate effect sizes (Robin's 7.5× vs 1.75×; the CMU "hidden pitfalls" audit). A self-scoring loop is even more exposed — the evaluator is inside the loop it's evaluating (cf. Agent-as-a-Judge, the organizer's own tool).

4. **Mode collapse & the evolution-vs-RL split.** The execution-grounded spotlight (§4.5) finds RL collapses idea diversity where evolutionary search preserves it. Diversity is the fuel of open-endedness; a loop that collapses stops improving. This is now a first-class design axis, not a footnote.

5. **Safety, stability, rollback.** Long-horizon self-modification risks *regression* (an edit that helps now but destabilizes later) and *tampering*. Accepted **SAHOO** (safeguarded alignment for recursive self-improvement, `OAFPpQO0H9`) and **TamperBench** (`smLtz7WID0`) target this. The Gödel machine's original appeal was that a *proof* gave a stability guarantee; dropping the proof (as every practical system does) means we've traded provable safety for empirical scores — the workshop's "Prove the gains" is aspirational, not yet achieved.

6. **The convenient-domain problem.** Every impressive RSI result lives where evaluation is cheap, fast, and automatable: coding benchmarks (DGM), linear-attention blocks (ASI-Arch), math/kernels (AlphaEvolve), nanoGPT loss (speedrun). Whether the loop closes where evaluation is slow, expensive, or ambiguous (real wet-lab science, open-ended research taste) is unproven — and is precisely the gap between this workshop and the wet-lab systems in the landscape file.

---

## 7. Video / talk links

No **official** ICLR-2026-RSI session recording was found as of 2026-07; the only YouTube hits for the workshop are AI-generated audio summaries on a personal channel — **not authoritative, not linked here**.

```
Jeff Clune — "Open-ended and AI-generating algorithms in the era of foundation models"
   Schwartz Reisman Institute (U Toronto), 2025  ·  BEST SINGLE LINK
   https://www.youtube.com/watch?v=gIHAVTj9fjo
   Covers OMNI, VPT, ADAS, the Darwin Gödel Machine, The AI Scientist, open-endedness.
   (Re-host on Clune's own Evolving AI Lab channel: youtube.com/watch?v=W-ObbKdCOhk)

Andrej Karpathy — "Software Is Changing (Again)" (Software 3.0)
   Y Combinator, Jun 2025, ~40 min
   https://www.youtube.com/watch?v=LCEmiRjPEtQ
   LLMs as a new programming paradigm; agents automating software.

Andrej Karpathy — on Dwarkesh Patel ("the decade of agents")
   Oct 17 2025, ~2h25m
   https://www.youtube.com/watch?v=lXUZvyajciY
   Agentic coding, autonomy, why AGI is "still a decade away."

AlphaEvolve — Two Minute Papers explainer (3rd-party, reputable; no official DeepMind talk)
   May 2025  ·  https://www.youtube.com/watch?v=T0eWBlFhFzc
```

Primary non-video sources worth pairing: the **[Sakana DGM writeup](https://sakana.ai/dgm/)**, the **[AlphaEvolve blog](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)**, and the repos **[GAIR-NLP/ASI-Arch](https://github.com/GAIR-NLP/ASI-Arch)**, **[KellerJordan/modded-nanoGPT](https://github.com/KellerJordan/modded-nanoGPT)**, **[karpathy/autoresearch](https://github.com/karpathy/autoresearch)**.

---

## 8. How this connects to the rest of this repo

```
This guide (RSI workshop)      ─┬─▶  landscape-2025-2026.md   the broader "AI Scientist"
  the recursive sub-thread:     │                             field; RSI startups (Recursive,
  improve the improver          │                             Periodic, Lila) in its §7
                                ├─▶  ai-scientist-guide.md    Sakana AI Scientist v1/v2 —
                                │                             the end-to-end loop DGM's authors
                                │                             (Clune, Cong Lu) also built
                                ├─▶  ideation-execution-gap-   the negative result that MADE
                                │      guide.md                "execution-grounded" the watchword
                                │                             (same Stanford group as §4.5)
                                ├─▶  popper-falsification-     calibrated validation — the rigor
                                │      guide.md                a self-scoring RSI loop lacks
                                └─▶  robin-guide.md            wet-lab closed loop; the "convenient-
                                                              domain" contrast to code-only RSI
```

And to **your own work**: the [modded-nanoGPT](https://github.com/KellerJordan/modded-nanoGPT) speedrun *is* your GPT-2 124M training under a stopwatch; the Speedrunning Benchmark asks whether an agent could rediscover the Muon-and-friends tricks you'd otherwise read papers for; **PostTrainBench** asks the same about the **SFT/DPO/GRPO** loops you hand-wrote; and the **evolution-beats-RL / mode-collapse** finding (§4.5) is the RSI-scale echo of the diversity-collapse intuition you already met in DPO/GRPO. Karpathy's `autoresearch` is the loop you could actually run on your single H800.

---

## Caveats & open flags

- **Per-paper authors & arXiv IDs beyond the 4 orals + named spotlights are [UNVERIFIED]** — OpenReview's API sat behind a bot-challenge wall. Titles + track categories are verbatim from the official papers page; per-paper authors resolve via each forum ID in a browser. arXiv IDs shown for accepted papers (e.g. Agent0 2511.16043, Contextual Drag 2602.04288, ALMA 2602.07755, PostTrainBench 2603.08640) are high-confidence-but-not-PDF-verified except Agent0 and Execution-Grounded.
- **No official workshop video recording** exists; AI-generated summary videos are not authoritative and are excluded.
- **Recursive Superintelligence** funding ($650M / ~$4.65B) and HQ (SF vs London) come from secondary tech press (TechCrunch, TheNextWeb) and are reported inconsistently across outlets — treat the exact figure as approximate.
- **ASI-Arch's** "scaling law for scientific discovery" and "ASI4AI" framing are the authors' claims, stated without limitations in the abstract; presented here as claims, not established results.
