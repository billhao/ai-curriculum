# Towards an AI Co-Scientist: A Multi-Agent System That Debates Its Way to Better Hypotheses

How Google built a "generate, debate, evolve" colony of Gemini 2.0 agents that proposes novel, literature-grounded research hypotheses, ranks them in an Elo tournament run through *simulated scientific debate*, and gets measurably better the longer you let it think. Unlike Robin, it stops at the hypothesis — humans (and three co-timed wet-lab papers) close the experimental loop afterward.

## Background

**Primary paper**: [Towards an AI co-scientist](https://arxiv.org/abs/2502.18864) (Gottweis, Weng, Daryin, Tu, Palepu, Sirkovic, ... Karthikesalingam, Natarajan — Google Cloud AI Research, Google Research, Google DeepMind, with Houston Methodist, Sequome, Fleming Initiative / Imperial College London, Stanford School of Medicine; Feb 2025). A 81-page report: ~33 pages of main text, the rest references (143) and an appendix of agent prompts and example outputs.

This is the contemporary system Robin positions itself against. Where Robin (FutureHouse, May 2025) wires three pre-existing agents into a *closed* loop that touches wet-lab data, the AI co-scientist is a freshly designed *colony of six specialized agents plus a Supervisor*, all running on one base model (Gemini 2.0), that stops at hypothesis generation. The two papers are best read as a matched pair: same goal (augment the scientist), opposite halves of the research loop automated, and two different paired-comparison rankers (Elo-via-debate vs Bradley-Terry-Luce). The explicit comparison is at the end of this guide.

Research lineage the paper builds on (you know most of these):

1. **Test-time compute scaling** — Snell et al. ([2408.03314](https://arxiv.org/abs/2408.03314)), s1 ([2501.19393](https://arxiv.org/abs/2501.19393)), DeepSeek-R1 ([2501.12948](https://arxiv.org/abs/2501.12948)). The co-scientist's central claim is that it *scales test-time compute without any training* — no SFT, no RL, no fine-tuning. The "thinking" budget is spent on more agents, more debates, more tournament rounds.
2. **Multi-agent / agentic LLMs** — the same agent paradigm behind ReAct (Yao et al., [2210.03629](https://arxiv.org/abs/2210.03629)) that you studied; here specialized into role-based agents (generator, reviewer, ranker, ...).
3. **Elo rating** (Elo & Sloan, *The Rating of Chessplayers*, 1978; Coulom 2007 applied Elo to ranking Go move patterns; [2412.14427](https://arxiv.org/abs/2412.14427) on Elo under intransitivity). The tournament ranker.
4. **Multi-agent debate** (Khan et al., [2402.06782](https://arxiv.org/abs/2402.06782), "debating with more persuasive LLMs leads to more truthful answers") — the basis for using *simulated scientific debate* as the comparison operator inside the tournament.
5. **AI-for-science peers** — The AI Scientist (Lu et al., Sakana, [2408.06292](https://arxiv.org/abs/2408.06292)); Coscientist (Boiko et al., *Nature* 2023, chemistry + robots); Virtual Lab (Swanson et al., a "PI" LLM directing agent specialists); PaperQA2 (Skarlinski et al., [2409.13740](https://arxiv.org/abs/2409.13740)); HypoGeniC (Zhou et al.); Robin (Ghareeb et al., [2505.13400](https://arxiv.org/abs/2505.13400)).

The three wet-lab validations are reported in **separate, co-timed papers** — most notably the antimicrobial-resistance recapitulation: Penadés, Gottweis et al., *AI mirrors experimental science to uncover a novel mechanism of gene transfer crucial to bacterial evolution* (2025), with the underlying biology from He et al. ([2025-02](https://arxiv.org/abs/2502.02292) companion) and the original cf-PICI discovery in Alqurainy et al. (*Cell Host & Microbe* 2023).

## What Problem Does It Solve?

Biomedical breakthroughs increasingly require bridging *across* disciplines (the paper opens with CRISPR's microbiology+genetics+molecular-biology synthesis, and Hinton/Hopfield's physics+neuroscience), yet the literature has exploded past any one human's reach. The bottleneck is generating *novel, original, testable* hypotheses that are simultaneously grounded in prior evidence and aligned to a scientist's specific goal and constraints.

Existing tools each fall short:
- "Deep research" / literature-summary tools *synthesize* existing knowledge but do not propose novel hypotheses.
- Single-shot LLM prompting produces plausible-sounding but ungrounded, uncalibrated ideas with no mechanism to *improve* them.
- Earlier hypothesis systems (HypoGeniC, data-to-paper) evaluate against retrospective data or only recapitulate existing papers, leaving genuine novelty unproven.

The AI co-scientist targets the *cognitive* half of the loop: given a research goal in natural language (from a one-line prompt to tens of thousands of tokens including hundreds of PDFs), produce a *ranked* set of novel, literature-grounded, experimentally-testable hypotheses and research proposals, with citations and reasoning, that a domain expert then takes to the bench. It is explicitly a "scientist-in-the-loop" collaborator, not a replacement.

## Key Terms

**Test-time compute scaling (here)**: spending more inference compute to get better answers — but instead of longer chains-of-thought inside one model, the co-scientist spends it on *more agent activity*: more generated hypotheses, more tournament matches, more evolution rounds, more meta-review iterations. No weights change. The paper's headline empirical result is that hypothesis quality (measured by Elo) keeps rising with this spend and shows *no saturation*.

**Elo rating**: a paired-comparison rating system. Each item has a scalar rating; after item *i* beats *j*, ratings update by `Δ = K·(outcome − expected)`, where `expected = σ((θ_i − θ_j)/400·ln10)` — i.e. the same logistic `P(i beats j) = σ(θ_i − θ_j)` kernel you know from RLHF reward models and your DPO loss. The difference from a one-shot fit: Elo updates *online and sequentially*, one match at a time, with a learning rate K. New hypotheses enter at a fixed initial rating (the co-scientist uses **1200**) and climb as they win debates. (Contrast with Robin's BTL below.)

**Simulated scientific debate**: the *comparison operator* inside the tournament. Rather than asking the judge "rate hypothesis 1–10," the Ranking agent stages a multi-turn debate between two hypotheses (3–5 turns, up to 10) evaluating correctness, utility, specificity, novelty, and feasibility, ending in "better idea: 1 or 2." Lower-ranked hypotheses get a cheaper single-turn comparison instead.

**Research goal → research plan configuration**: the Supervisor parses the scientist's natural-language goal into a structured config — preferences, attributes (e.g. novelty, feasibility), constraints (e.g. "should be novel," "should be correct"), and the evaluation criteria the system will self-grade against.

**Asynchronous task framework**: the execution substrate. Specialized agents run as *worker processes* pulling from a task queue that the Supervisor fills; this is what lets the system flexibly trade compute for quality and run continuously.

## The Architecture

The system has four components: a **natural-language interface**, an **asynchronous task framework**, six **specialized agents**, and a persistent **context memory**. One **Supervisor** agent orchestrates everything. All agents run on **Gemini 2.0** (the design is model-agnostic).

```
  SCIENTIST                THE AI CO-SCIENTIST (multi-agent system on Gemini 2.0)              OUTPUT
 ┌──────────┐   goal     ┌─────────────────────────────────────────────────────────┐   ┌──────────────┐
 │ research │──────────> │  parse → RESEARCH PLAN CONFIG  ──>  SUPERVISOR AGENT      │   │ research      │
 │  goal +  │            │  (preferences, attributes,          (fills task queue,    │──>│ overview +    │
 │ feedback │ <───chat── │   constraints, eval criteria)        assigns agents to    │   │ ranked, novel │
 └──────────┘            │                                      workers, allocates   │   │ hypotheses    │
       ^                 │                                      compute, computes     │   │ + proposals   │
       │                 │      ┌──────── WORKER PROCESSES (async task queue) ──────┐ │   └──────────────┘
       │ refine goal,    │      │                                                   │ │
       │ add own ideas,  │      │  GENERATION ──> REFLECTION ──> RANKING (tourney)  │ │
       │ manual reviews  │      │      ^              ^   |          |   ^           │ │
       └─────────────────┤      │      |              |   v          v   |           │ │
                         │      │   META-REVIEW <── PROXIMITY <─> EVOLUTION          │ │
                         │      │   (feedback to all agents, no weight updates)      │ │
                         │      └───────────────────────────────────────────────────┘ │
                         │                          ▲                                   │
                         │              CONTEXT MEMORY (state, stats, restart)          │
                         │              TOOLS: web search · DBs · AlphaFold             │
                         └─────────────────────────────────────────────────────────────┘
```

| Agent          | Role (peer-review / scientific-method analogy)                          | Key mechanisms                                                                                  |
|----------------|------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| **Supervisor** | lab director / scheduler                                                | parses goal→config; fills + samples the task queue; weights agents; writes stats to memory; decides terminal state; enables restart |
| **Generation** | the idea person                                                         | literature exploration via web search; **simulated scientific debates** (self-play); iterative assumption identification; research expansion |
| **Reflection** | peer reviewer                                                           | initial review (no tools) → full review (web search) → **deep verification** (decompose into assumptions + sub-assumptions) → observation review → simulation review → recurrent/tournament review |
| **Ranking**    | program committee                                                       | **Elo tournament** (init 1200); multi-turn debate for top pairs, single-turn for lower; prioritizes similar + new + top-ranked pairs |
| **Proximity**  | librarian / clustering                                                  | builds a similarity graph over hypotheses → dedup, clustering, picks which pairs the tournament compares |
| **Evolution**  | the editor / improver                                                   | grounding, coherence+feasibility, inspiration, **combination**, simplification, out-of-box thinking. Always creates a *new* hypothesis (never edits in place) so it must re-win the tournament |
| **Meta-review**| the chief editor / "lessons learned"                                    | synthesizes recurring patterns across all reviews + debates into a **meta-review critique** appended to every agent's next prompt; writes the final NIH-style research overview; suggests expert contacts |

A few design points worth dwelling on, because they are the reusable ideas:

- **Feedback without back-propagation.** The Meta-review agent is the system's learning signal. It reads all the tournament debates and reviews, distills the recurring failure modes (the paper's example: "only 90% of reviews caught a blood-brain-barrier permeability issue → make every future review check this"), and simply *appends that critique to the next round's prompts*. This is in-context self-improvement — the same trick Robin uses to fold prior wet-lab findings back into generation. No gradients, no fine-tuning; it works because Gemini 2.0's long context can carry the accumulated critique.

- **Evolution never mutates in place.** It generates a *new* hypothesis inspired by/combining the top-ranked ones, which then re-enters the tournament. This protects the incumbent quality — a flawed "improvement" just loses its matches instead of corrupting a good hypothesis.

- **The Supervisor turns quality into a scheduling problem.** It periodically computes statistics (how many hypotheses exist, how the tournament is progressing, whether new-idea generation vs. evolution is paying off) and re-weights which agents get worker slots. This is where test-time compute is actually allocated.

## The Generate–Debate–Evolve Loop and the Elo Tournament

The whole system is one self-improving cycle:

```
        ┌─────────────────────────────────────────────────────────────────┐
        │                                                                   │
        v                                                                   │
  GENERATION ──> new hypotheses (web-grounded, debate-refined, w/ rationale)│
        │                                                                   │
        v                                                                   │
  REFLECTION ──> each hypothesis reviewed (correctness/novelty/safety,      │
        │         deep verification into assumptions)                       │
        v                                                                   │
  PROXIMITY  ──> similarity graph: cluster + choose which pairs to compare  │
        │                                                                   │
        v                                                                   │
  RANKING    ──> ELO TOURNAMENT via simulated scientific debate             │
        │         (top pairs: 3–5 turn debate; lower pairs: 1-turn compare) │
        │         win/loss → Elo update; top hypotheses surface             │
        v                                                                   │
  EVOLUTION  ──> combine / ground / simplify top hypotheses → NEW ones ─────┤
        │                                                                   │
        v                                                                   │
  META-REVIEW ──> distill recurring critique → append to ALL agent prompts ─┘
                  (+ build research overview for the scientist)
```

Why a tournament with *debate* instead of asking the judge for a score? The reasoning is exactly the argument you already know from preference-based RLHF and DPO, applied one level up:

1. **Pairwise beats absolute.** "Which of these two is better?" is lower-variance and better-calibrated than "rate this 7/10," for LLMs as for humans. The Ranking prompt even instructs the judge to *disregard the numerical scores* attached to each hypothesis's prior review, because those scores "may not be directly comparable across reviews" — an explicit refusal to anchor on absolute ratings.
2. **Debate extracts more signal than a single judgment.** For the top hypotheses (where precision matters most), the comparison is a 3–5-turn (up to 10) simulated panel that poses clarifying questions, weighs correctness/utility/specificity/novelty/feasibility, and surfaces flaws before concluding. Cheaper single-turn comparisons are reserved for lower-ranked pairs to bound cost.
3. **Elo aggregates noisy, incomplete comparisons online.** You never run all C(n,2) matches. The Proximity graph picks informative pairs (similar hypotheses, newly-added ones, and current leaders), and Elo updates incrementally as matches stream in.

Pseudocode for one tournament tick:

```python
INIT_ELO = 1200
def tournament_round(hypotheses, proximity_graph, goal):
    pairs = select_pairs(proximity_graph,          # prioritize: similar pairs,
                         prefer_new=True,          #   newly generated, and
                         prefer_top_ranked=True)   #   current top-Elo hypotheses
    for h1, h2 in pairs:
        if top_ranked(h1) and top_ranked(h2):
            winner = scientific_debate(h1, h2, goal, turns="3..10")  # multi-turn
        else:
            winner = single_turn_compare(h1, h2, goal)               # cheap
        # logistic expectation — same kernel as your DPO / BTL reward model
        exp1 = 1 / (1 + 10 ** ((h2.elo - h1.elo) / 400))
        s1 = 1.0 if winner is h1 else 0.0
        h1.elo += K * (s1 - exp1)        # ONLINE, sequential update
        h2.elo += K * ((1 - s1) - (1 - exp1))
    return sorted(hypotheses, key=lambda h: -h.elo)
```

## Test-Time Compute Scaling: The Results

All quality numbers use the system's own **Elo** as the metric (auto-evaluated, not ground truth — the paper repeatedly flags this caveat).

**1. Elo is concordant with real accuracy (Fig 3).** On the GPQA diamond set (the hard expert-curated MCQ benchmark) reframed as research goals, higher co-scientist Elo correlates with higher answer accuracy; picking the top-Elo response per question yields **top-1 accuracy 78.4%**. The Gemini-2.0 reference baseline stays flat (~0.69) across Elo buckets, showing the gain is from the *system*, not the base model. This validates Elo as a self-improvement signal.

**2. Quality keeps rising with compute (Fig 4).** Across **203 research goals**, hypotheses partitioned into 10 temporal buckets (bucket 10 = most recent compute):
```
                  bucket 1   →   bucket 10
  Best Elo          ~1350     →    ~1620
  Top-10 avg Elo    ~1260     →    ~1500
```
Both rise monotonically; the trend does not flatten.

**3. It outperforms frontier reasoning models and human experts (Fig 5).** On a harder subset of **15 expert-curated goals**, the co-scientist's Elo passes every baseline as compute scales: **Gemini 2.0 Pro Experimental, Gemini 2.0 Flash Thinking Exp 12-19, OpenAI o1, OpenAI o3-mini-high, DeepSeek R1, and the human experts' "best guess."** Newer reasoning models (o3-mini-high, R1) were competitive at *less* compute, but the co-scientist surpassed all of them with scaling, and **no saturation** was observed.

**4. It can improve the experts' own ideas (Fig 6).** Fed expert "best guess" hypotheses, the co-scientist's refinements first matched, then exceeded, the experts' originals by Elo — a preliminary "AI-augmented expertise" result.

**5. Experts and LLM judges prefer its outputs (Figs 7–8).** On 11 of the 15 goals, domain experts rated the co-scientist **best**: average preference rank **2.36** (lower = better, 4 systems), **novelty 3.64/5**, **impact 3.09/5**, beating Gemini 2.0 Flash Thinking (2.73), Gemini 2.0 Pro (2.45), and OpenAI o1 (2.45). Four independent LLM-judge panels (o3-mini, o1-preview, Gemini 2.0 Pro, Gemini Flash Thinking) all ranked it most-preferred (2.18–2.32).

**6. Safety (Section 4.4).** Against **1,200 adversarial research goals across 40 topics**, the system rejected all unsafe goals in this preliminary red-team. Safeguards: automated review of the input goal, per-hypothesis safety review (unsafe ones never enter the tournament), continuous monitoring of research directions by the Meta-review agent, full logging, and a planned Trusted Tester program.

## The Three Wet-Lab Validations

All three were run with expert scientists guiding and prioritizing, and are detailed in separate co-timed reports. The complexity ladder is deliberate: drug repurposing (medium, constrained search) → target discovery (high, large unknown space) → mechanism explanation (very high, vast/dynamic).

### 1. Drug repurposing for acute myeloid leukemia (AML)

Setup: constrained the system to **2,300 approved drugs × 33 cancer types**. Candidates ranked by combining the co-scientist review score (1–5) with the **DepMap** dependency probability (0–1, a sanity-check filter); kept only pairs with review score ≥4 **and** DepMap ≥0.99. Six board-certified hematologists/oncologists evaluated **78** hypotheses formatted as **NIH Specific Aims pages** and rated them high across the rubric.

- **Validating against existing evidence** — 5 expert-selected drugs (Binimetinib, Pacritinib, Cerivastatin, Pravastatin, DMF). In vitro on AML cell lines, **Binimetinib, Pacritinib, and Cerivastatin inhibited cell viability**; Binimetinib showed an **IC50 as low as 7 nM** (MOLM-13).
- **Genuinely novel candidates** — the system was then run with *no* DepMap/expert priors, free to propose drugs with no prior AML evidence. Top-3 novel candidates: **Nanvuranlat, KIRA6, Leflunomide**. **KIRA6** (an IRE1α inhibitor) inhibited viability in **three** AML lines: **KG-1 IC50 13 nM**, **MOLM-13 517 nM**, **HL-60 817 nM** — all in the nanomolar range.

### 2. Novel target discovery for liver fibrosis

In a **human hepatic organoid** system with live-cell imaging, the co-scientist was asked to propose epigenetic targets driving myofibroblast formation. Experts picked **3** top-ranked hypotheses (from 15) with full proposals. The system identified **three novel epigenetic targets**; drugs hitting **2 of the 3** showed **significant anti-fibrotic activity without cytotoxicity** (Fig 12, p < 0.0001 to < 0.01 vs. fibrosis-inducer control). One of the active drugs is **FDA-approved for another indication** — a repurposing opportunity surfaced as a side effect.

### 3. Recapitulating an unpublished AMR mechanism (cf-PICIs)

The hardest test, and the most striking. The question: why are *identical* capsid-forming phage-inducible chromosomal islands (cf-PICIs) — mobile genetic elements that spread antibiotic-resistance and virulence genes — found across *different* bacterial species (incl. WHO priority pathogens *E. coli*, *K. pneumoniae*)? A research group (Penadés et al.) had answered this experimentally but **not yet published** it. Given only a one-page background plus two reference papers (the original cf-PICI discovery and a phage-satellite identification tool), the co-scientist **independently proposed the correct top-ranked mechanism — that cf-PICIs interact with diverse phage tails to expand their host range — in ~2 days**, recapitulating the unpublished wet-lab result it had no access to. (It was, of course, building on decades of open literature.)

## AI Co-Scientist vs Robin

The two systems are the cleanest available contrast in autonomous biology. Read together they bracket the field: same north star, opposite halves of the loop, two different paired-comparison rankers.

```
                         AI CO-SCIENTIST (Google)          ROBIN (FutureHouse)
─────────────────────────────────────────────────────────────────────────────────────
Base model(s)            Gemini 2.0 (all agents)           o4-mini (orchestrator) +
                                                           Claude 3.7 Sonnet (judge),
                                                           PaperQA2 agents
Architecture             6 purpose-built agents +          orchestrator wiring 3 reused
                         Supervisor, async task queue      agents (Crow/Falcon/Finch),
                                                           collapsed to a deterministic
                                                           Jupyter notebook
Hypothesis generation    generate–debate–evolve;           2-phase (assay → drug);
                         self-play debate, evolution,      each candidate literature-
                         meta-review feedback              grounded before judging
─────────────────────────────────────────────────────────────────────────────────────
RANKING METHOD           Elo tournament                    Bradley–Terry–Luce
  comparison operator    multi-turn SIMULATED DEBATE       single LLM pairwise judgment
                         (3–5, up to 10 turns) for top;    (judge prompt meta-generated
                         single-turn for lower pairs       from expert preferences)
  aggregation            ONLINE, sequential Elo update     BATCH MLE: one global fit of
                         (K-factor), init rating 1200;     θᵢ over all comparisons
                         pairs chosen by Proximity graph   (full round-robin ≤25 hyps,
                         (similar/new/top-ranked)          else sample 300 pairs)
  shared kernel          both use P(i≻j)=σ(θᵢ−θⱼ) — the same logistic preference model
                         as your RLHF reward model and DPO loss
─────────────────────────────────────────────────────────────────────────────────────
Experimental loop        OPEN — stops at hypothesis;       CLOSED — Finch analyzes the
                         humans + 3 co-timed papers do     wet-lab data and feeds results
                         the wet-lab                       back into next-round generation
Refines on real data?    no (no data in the loop)          yes (in-context, not fine-tune)
Validated discovery      AML candidates (KIRA6 etc.),      ripasudil for dry AMD
                         liver-fibrosis targets, cf-PICI   (novel, in-vitro validated)
                         mechanism recapitulation
Test-time-compute claim  central + quantified (Elo scales, no saturation)   implicit
```

The two key axes:

1. **Elo-via-debate (co-scientist) vs Bradley-Terry-Luce (Robin).** Both convert noisy pairwise wins into a cardinal ranking using the *same* logistic kernel `σ(θ_i − θ_j)` you know from DPO. They differ in two ways. (a) **Aggregation:** Elo is an *online, order-dependent* update — each match nudges ratings by `K·(outcome − expected)`, new hypotheses enter at 1200 and climb, and you never need a fixed comparison set (ideal for a tournament where hypotheses are continuously generated and evolved). BTL is a *batch* MLE that fits all strengths jointly over a frozen set of comparisons (Robin runs a full round-robin up to 25 candidates, else samples 300 pairs). (b) **Comparison operator:** the co-scientist's match is a *multi-turn simulated debate* (richer, costlier signal, reserved for top pairs); Robin's is a *single* LLM pairwise judgment per pair. So: co-scientist spends more compute *per comparison* and updates incrementally; Robin spends less per comparison and fits globally. Online Elo naturally suits the streaming, ever-growing hypothesis pool; batch BTL suits Robin's fixed candidate slate.

2. **Hypothesis-only (open loop) vs closed experimental loop.** This is the bigger difference. The co-scientist's product is *ranked hypotheses and proposals*; it never sees experimental data. Robin's product is a *refined candidate after wet-lab feedback* — Finch analyzes the flow-cytometry/RNA-seq and the distilled findings re-enter generation (the round-1 RNA-seq result is what led Robin to ripasudil in round 2). The co-scientist's three "validations" are real wet-lab results, but they were produced by *humans and separate papers* after the system handed off — the loop is closed by people, not by the system. Robin's claim is to be the first to close the hypothesis↔data loop autonomously; the co-scientist's claim is breadth (general-purpose, three domains, quantified compute scaling) and depth of the *generation* machinery.

## Limitations (stated by the authors)

- **Open-literature only.** Reviews can miss paywalled prior work, and the system has limited access to *negative results / failed experiments* — knowledge experienced scientists use heavily to prioritize.
- **Weak multimodal / tool reasoning.** Much scientific signal lives in figures, charts, and multi-omics datasets the system does not yet read well; it assesses text, not images or databases.
- **Inherited LLM flaws.** Factuality errors and hallucinations from Gemini 2.0 propagate through the system.
- **Elo is an auto-evaluation metric, not ground truth.** It may favor attributes that do not align with scientists' true preferences; every quality figure carries this caveat. Better, less self-favoring metrics are future work.
- **No drug-delivery / PK / clinical-trial design.** Outputs are targets and mechanisms, not formulations or trial protocols; a translational team is still required.
- **Discovery is recombination, not de novo invention** (same caveat as Robin) — the cf-PICI result recapitulates known-but-unpublished biology from open literature, fast; it did not invent a mechanism from nothing.
- **Automation bias.** Over-reliance could homogenize research ideation; the paper cites mixed evidence on LLMs and creative diversity.

## Practical Takeaways

- The reusable engineering ideas, all model-agnostic: (1) **pairwise + Elo/BTL beats absolute scoring** for ranking LLM-generated candidates — and explicitly tell the judge to *ignore* prior numeric scores; (2) **use multi-turn debate as the comparison operator** for your most important pairs, single-turn for the rest, to bound cost; (3) **separate generation, review, ranking, and improvement into distinct agents** so each can be prompted and scaled independently; (4) **feed back via in-context meta-review, not fine-tuning** — distill recurring critiques and append them to the next round's prompts; (5) **make "evolution" additive** (new candidate re-enters the tournament) so improvements can't corrupt incumbents; (6) **let a Supervisor turn quality into a scheduling problem** — spend test-time compute where the statistics say it pays.
- When to reach for the co-scientist pattern: broad, open-ended hypothesis generation across a large literature where you want a *ranked, grounded* slate and a human will run the experiments. When to prefer Robin's pattern: when you have a wet-lab readout and want the system to *refine on the data* itself.

## Key Papers

1. Gottweis, Weng, et al. *Towards an AI co-scientist.* arXiv 2502.18864 (2025). https://arxiv.org/abs/2502.18864
2. Penadés, Gottweis, He, et al. *AI mirrors experimental science to uncover a novel mechanism of gene transfer crucial to bacterial evolution* (co-timed AMR validation, 2025).
3. Ghareeb et al. *Robin: A multi-agent system for automating scientific discovery.* arXiv 2505.13400 (2025). https://arxiv.org/abs/2505.13400
4. Lu et al. *The AI Scientist.* arXiv 2408.06292 (2024). https://arxiv.org/abs/2408.06292
5. Boiko et al. *Autonomous chemical research with large language models* (Coscientist). Nature 624:570–578 (2023).
6. Khan et al. *Debating with more persuasive LLMs leads to more truthful answers.* arXiv 2402.06782 (2024). https://arxiv.org/abs/2402.06782
7. Snell et al. *Scaling LLM test-time compute optimally can be more effective than scaling model parameters.* arXiv 2408.03314 (2024). https://arxiv.org/abs/2408.03314
8. Skarlinski et al. *Language agents achieve superhuman synthesis of scientific knowledge* (PaperQA2). arXiv 2409.13740 (2024). https://arxiv.org/abs/2409.13740
9. Elo, A. E. & Sloan, S. *The Rating of Chessplayers: Past and Present* (1978); Coulom, R. *Computing "Elo ratings" of move patterns in the game of Go.* ICGA Journal 30 (2007).
10. Bradley & Terry. *Rank Analysis of Incomplete Block Designs.* Biometrika 39 (1952). [Robin's ranker, for contrast.]
