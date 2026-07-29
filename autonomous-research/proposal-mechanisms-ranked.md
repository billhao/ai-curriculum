# Proposal Mechanisms in Autonomous Research, Ranked

Execution is getting cheap and checkable: a node runs or it doesn't, a Lean term type-checks or it doesn't, a submission scores against a held-out leaderboard or it doesn't. Proposal is neither. Every system below caps out at the quality of the object it decided to try, and none of them can score that decision without executing it — the step they were built to avoid paying for. The measurement literature makes the cap concrete: the LLM reranker that picks which idea to run correlates r = -0.092 with executed outcome and -0.321 on excitement, so scaling generation sharpens a hacked proxy rather than the target. Where a non-LLM verifier exists (Lean kernel, program score, e-process, Rietveld pattern), genuinely new proposal operators appear; where it does not, seventeen independent teams converge on the same artifact — role-labeled prompt chains feeding an LLM judge. This document compares only that step: what each work generates, how it diversifies, what signal selects, and what the comparison shows that no single paper does.

## The ranked table

| Rank | Work | Proposal mechanism in <=10 words | Novelty | Impact | Combined |
|-----:|------|----------------------------------|--------:|-------:|---------:|
| 1 | AlphaProof + AlphaGeometry 2 | LLM emits Lean-checkable problem variants; TTRL on that curriculum | 8.5 | 7.83 | 8.17 |
| 2 | AlphaEvolve | Diff mutation over MAP-Elites/island archive of scored programs | 7.17 | 9.0 | 8.08 |
| 3 | POPPER | Typed (h0, h1, executable test) triples gated on implication | 8.0 | 5.67 | 6.83 |
| 4 | The Ideation-Execution Gap | No generator; blinded paired execution audit of reranked ideas | 6.33 | 7.33 | 6.83 |
| 5 | The AI Scientist (v1) | Whole JSON idea archive pasted in; "propose the next" | 5.0 | 8.0 | 6.5 |
| 6 | MLE-bench | Benchmark; forces AIDE-style atomic single-change improvements | 4.67 | 8.0 | 6.33 |
| 7 | RE-Bench | Benchmark; sweeps run-length vs restarts at fixed wall clock | 5.5 | 7.0 | 6.25 |
| 8 | A-Lab (+ Palgrave/Schoop reanalysis) | Literature analogy, then DFT-ranked precursor sets from probe runs | 6.17 | 6.17 | 6.17 |
| 9 | PaperBench | Benchmark; self-decomposition of a paper into a work plan | 5.67 | 6.5 | 6.08 |
| 10 | Towards an AI co-scientist | Debate-generated hypotheses, six mutation strategies, Elo tournament | 5.33 | 6.67 | 6.0 |
| 11 | The AI Scientist-v2 | Stage-conditioned typed node generators with already-tried ledgers | 5.0 | 5.83 | 5.42 |
| 12 | Agents4Science 2025 | Venue; per-stage A-D autonomy labels isolate hypothesis development | 5.67 | 5.0 | 5.33 |
| 13 | Robin | Retrieval-expand each idea into a report, then rank reports | 5.33 | 4.5 | 4.92 |
| 14 | Aviary | Sampled tool call as proposal; reward-filtered SFT on successes | 4.83 | 5.0 | 4.92 |
| 15 | The Virtual Lab | Agent meetings write a spec; an ESM scan does the proposing | 4.83 | 4.33 | 4.58 |
| 16 | Kosmos | Query a persistent world model; emit <=10 typed task specs | 4.5 | 4.33 | 4.42 |
| 17 | Evaluating Sakana's AI Scientist | No generator; audits the novelty gate and edit-loop deltas | 4.0 | 4.33 | 4.17 |

Three judges (mechanism / evidence / practitioner) disagreed most on four rows. A-Lab has the widest impact spread, 3.5 (scores 6, 8, 4.5): the split is whether an adversarial reanalysis that forces a Nature title correction is high impact for the field or a demotion of the system — ARROWS3's diagnostic-probe operator is real, but it contributed 6 of 36 successes and 35/36 claims carry at least one error type. The Ideation-Execution Gap and AI Scientist v1 both spread 3.0 on novelty (5/8/6 and 4/4/7): whether "contributes no generator" and "paste the archive, ask for the next idea" are novelty floors or precisely the contribution. AlphaProof spreads 2.5 on impact (9, 8, 6.5) purely on reachability — 500 TPU-days per target, ~100,000 TPU-days of auto-formalization. Consensus is tightest where a mechanism is either sharply typed or plainly absent: POPPER novelty 8/8/8, AI Scientist-v2 novelty 5/5/5, AlphaEvolve impact 9/9/9.

## Taxonomy of proposal mechanisms

### Family 1 — Verifier-checkable problem-variant curricula
Generates problems, not solutions. AlphaProof emits hundreds of thousands of Lean-syntax-valid variants of one target (simplification, generalization, lemma, analogy, decomposition) from 791 few-shot (problem, variant) pairs; AlphaGeometry 2's analogue is the auxiliary construction — a new point plus its defining predicates. Diversity source: sampled prompting strategies, a programmatic hypothesis/goal mutator, randomized formalization prompts (1M NL problems into 80M Lean statements), and N_evo = 15 rounds of re-seeding validated variants. Selection signal on the proposal itself: Lean syntax validity and dedup, nothing more; real selection happens downstream in whether RL on the curriculum cracks the target. Wins when a kernel-level verifier makes wrong proposals nearly free, so quantity dominates quality — Top-10 to Top-100k monotonically raises the target prove rate. Fails when the domain resists formalization: auto-formalization pass@1 is 33.3% on combinatorics, and both IMO 2024 combinatorics problems went unsolved.

```
+----------------------------------------------------------------+
| target T (unsolved after 12 TPU-h search)                      |
|   |                                                            |
|   +--> LLM + 791 few-shot (problem,variant) pairs              |
|   |      simplify / generalize / lemma / analogy / decompose   |
|   +--> programmatic mutator over hypotheses and goals of T     |
|          |                                                     |
|          v                                                     |
|        Lean syntax check --> dedup --> V_T  (~1e5 variants)    |
|          |     ^                                               |
|          |     +-- reseed high-sim variants, N_evo = 15 rounds |
|          v                                                     |
|        RL on {T} u V_T --> Lean kernel: proof / disproof       |
|          |                                                     |
|          +--> weight update; halt the moment T is proved       |
+----------------------------------------------------------------+
```

### Family 2 — Evolutionary mutation over a scored archive
Generates a delta on an artifact that already carries a score. AlphaEvolve emits SEARCH/REPLACE diffs against a sampled parent, with k other high-scoring programs pasted in as "Prior programs" alongside their metric dicts; AI Scientist v1 degenerates the same idea to text, serializing the entire idea archive into the prompt and asking for "the next" one. Diversity source: MAP-Elites plus island populations, multi-objective scoring used deliberately as a diversity injector, stochastic prompt templates, model mixing (a cheaper weaker model sometimes beats using only high-end models), and a second co-evolving database of meta-prompts. Selection signal: a user-written evaluate() returning scalars — no LLM judge on the primary metric anywhere. Wins whenever candidate scoring is cheap relative to an LLM call: 48 multiplications for 4x4 complex matmul after 56 years, kissing number 593 in 11D, 0.7% of Google's fleet compute recovered. Fails when the archive carries no selection pressure — v1 never thresholds its 1-10 self-scores, and its own Section 8 reports near-duplicate ideas across runs and across models.

```
+------------------------------------------------------------------+
| program DB  (MAP-Elites x island populations)                    |
|   |                                                              |
|   +-- sample --> parent + k inspirations, each with its metrics  |
|                    |                                             |
|                    v                                             |
|              prompt = role line                                  |
|                     + prior programs (source + metric dict)      |
|                     + current program (source + metric dict)     |
|                     + meta-prompt sampled from co-evolved DB     |
|                     + stochastic template fills                  |
|                    |                                             |
|                    v                                             |
|              LLM diff:  <<<SEARCH  ===  REPLACE>>>  (or rewrite) |
|                    |                                             |
|                    v                                             |
|              child = apply_diff(parent)                          |
|                    |                                             |
|                    v                                             |
|              evaluate() cascade: cheap stage --> harder stages   |
|                    |                                             |
|                    +--> scored child written back to program DB  |
+------------------------------------------------------------------+
```

### Family 3 — Typed tree search over experiment nodes
Generates one plan-plus-code node per step, with the operator chosen by the parent's status rather than by model discretion. AIDE — the engine inside MLE-bench and RE-Bench — has three: _draft (a deliberately simple baseline plus a self-chosen validation metric), _improve (exactly one atomic change "so that we can experimentally evaluate the effect"), _debug (repair from the traceback). AI Scientist-v2 extends this to seven stage-conditioned types (draft/debug/improve/hyperparam/ablation/seed/aggregate) with per-type already-tried ledgers. PaperBench's IterativeAgent is the degenerate case: one action per turn, re-prompted until the wall clock ends. Diversity source: 3-5 independent draft roots plus a prompt asking for something different — weak, and measurably so. Selection signal: greedy argmax on the agent's own validation metric, with a feedback LLM setting is_buggy. Wins when a held-out grader exists and horizons are short: AIDE 8.7% vs OpenHands 4.4% vs MLAB 0.8% at identical gpt-4o. Fails against the objective it invented for itself — MLE-bench's 100h runs sometimes score below 24h, and 84% of RE-Bench Restricted-Architecture-MLM proposals are transformer variants in an environment built to handicap transformers.

```
+--------------------------------------------------------------------+
| journal = tree of nodes {plan, code, metric, is_buggy, feedback}   |
|   |                                                                |
|   v                                                                |
| search_policy(journal):                                            |
|   if   drafts < num_drafts    --> _draft                           |
|   elif rand() < debug_prob    --> _debug(buggy leaf, depth <= D)   |
|   elif stage == tune          --> _hyperparam(new axis, dedup log) |
|   elif stage == ablate        --> _ablation(new idea, dedup log)   |
|   else                        --> _improve(best node, ONE change)  |
|   |                                                                |
|   v                                                                |
| (3-10 sentence plan  +  one self-contained code block)             |
|   |                                                                |
|   v                                                                |
| execute --> stdout / traceback --> feedback LLM (+ VLM on plots)   |
|   |            {is_bug, metric, lower_is_better, summary}          |
|   v                                                                |
| Memory = (Design, Results, Validation Metric) of non-buggy nodes   |
|   |                                                                |
|   +--> conditions the next _draft / _improve prompt                |
+--------------------------------------------------------------------+
```

### Family 4 — Decomposition into falsification tests
Generates a statistically typed object: POPPER's designer emits (name, test description, null sub-hypothesis h0, alternative h1) and nothing else counts as a proposal. Diversity source: an in-prompt Self-Refine loop over three axes (non-redundancy against prior tests, implementability given the schema, logical relevance) plus F_failed, an explicit avoid-list of relevance-rejected and execution-failed specs; the result is 2.5x more distinct statistical tests than 9 PhD-level human experts used on the same task. Selection signal: an LLM relevance judge scoring R(h) on a 6-level 0.1-1.0 rubric with a hard gate at tau = 0.8, then no ranking at all — surviving tests execute, each p-value is calibrated to e_i = 0.5/sqrt(p_i), and the product is a non-negative super-martingale rejecting H0 at E > 1/alpha. Wins when the number of tests must be adaptive and self-chosen without voiding inference: Type-I 0.103 +/- 0.020 at nominal alpha = 0.1, power 0.638 vs ReAct 0.383. Fails when the implication gate is removed (Type-I 0.082 -> 0.340) or the backbone is weak (Claude-Haiku-3.5 -> 0.230).

```
+------------------------------------------------------------------+
| H, schema-only data view, tau = 0.8, N_max = 3..5, E = 1         |
|   |                                                              |
|   v                                                              |
| for i in 1..N_max:                                               |
|     for k in 1..10:            # relevance rejection sampling    |
|         T <- A_design(H, schema, F_success + p-values, F_failed) |
|              critic --> reflect --> revise   (one CoT stream)    |
|              emits {name, description, h0, h1}                   |
|         if A_rel(T) >= 0.8: break                                |
|         else: F_failed += T                                      |
|     ------------- proposal boundary -------------                |
|     p_i <- ReAct execution over pandas / scipy                   |
|     e_i <- 0.5 / sqrt(p_i);   E <- E * e_i                       |
|     if E > 1/alpha: return VALIDATED                             |
+------------------------------------------------------------------+
```


### Family 5 — Verifier-grounded RL over the proposer itself
Generates the same object as any agent — a tool call — but makes the sampling node trainable. Aviary defines the agent as a stochastic computation graph whose only random nodes are LLMCallOps, so one identified call is simultaneously the candidate generator, the unit reward filters, and the unit SFT updates; AlphaProof's TTRL is the same shape at 500 TPU-days per target. Diversity source: sampling temperature and nothing else, plus i.i.d. rollouts (32, up to 945) and a curriculum weight w_k = M(1 - f_pass^k), M = 20, that pushes attempts toward currently-failing tasks. Selection signal: a hard threshold R > rho on trajectory return against a real verifier — Rosetta cart_ddg ddG, exact-match answers, a Lean kernel — never an LLM judge. Wins when a programmatic verifier is cheap and the task distribution is graded: an 8B policy reaches 0.86-0.89 on SeqQA at ~$0.00066/trajectory vs ~$0.07 for Claude 3.5 Sonnet. Fails outside the base policy's support (a task pi_0 never solves contributes nothing forever, while the curriculum keeps spending on it) and degenerates under mode collapse, since SFT-on-successes-only narrows the proposal distribution with no entropy term reported.

```
+--------------------------------------------------------------------------+
| per timestep (inside one episode):                                       |
|     xi_t = [o_0, a_0, ..., o_t]                                          |
|     thought ~ p_LLM(. | xi_t)          # ReAct: separable node           |
|     a_t     ~ p_LLM(. | xi_t, thought, tool schemas)                     |
|     ----------- proposal boundary -----------                            |
|     o_t+1, r_t = env.step(a_t)         # Rosetta / PCR sim / PaperQA2    |
| per round (across episodes), expert iteration:                           |
|     T_i <- rollout(pi_i-1, tasks ~ w_k = M(1 - f_pass^k), M = 20)        |
|     D_i <- D_i-1 + {tau : R(tau) > rho}        # keep successes only     |
|     pi_i <- SFT(D_i)                           # proposer updates itself |
| at test time:                                                            |
|     k = 32..945 rollouts --> drop unsure --> majority vote               |
|     or k = 16 --> Rosetta ddG oracle --> pass@16                         |
+--------------------------------------------------------------------------+
```

### Family 6 — LLM writes the search spec; a non-LLM enumerator proposes
Generates a specification, not candidates. The Virtual Lab's agent meetings produce seeds, scorers, a weight vector WS = 0.2*ESM_LLR + 0.5*ipLDDT - 0.3*dG, beam width 5 and 4 rounds; the actual molecular proposals come from an ESM-1b scan of all L x 19 point mutants with no LLM in the loop. A-Lab is the same shape with a text-mined analogy model choosing precursors and ARROWS3 enumerating DFT pairwise reaction energies. Diversity source: 5 parallel meetings at T = 0.8 merged at T = 0.2 with provenance attribution; then, at the candidate level, exhaustive enumeration. Selection signal: the LLM-authored scalarization, evaluated by real physics (AlphaFold-Multimer, Rosetta, or a robot furnace plus Rietveld). Wins when the candidate space has a cheap enumerator the LLM could never reliably list. Fails on scale-blindness — ipLDDT lives on 60-80 with weight 0.5 while ESM LLR lives on 1-8 with weight 0.2, which is how three cysteine-crosslinking designs ranked 19/23, 22/23 and 21/23 on LLR yet were selected and bound everything nonspecifically — and on beam collapse: all 23 final Ty1 mutants carry one round-1 mutation at position 32.

```
+---------------------------------------------------------------------+
| LLM layer (writes the operator):                                    |
|     agenda + agenda_questions ("give a formula", "give ONE number") |
|       |                                                             |
|       +--> 5 parallel meetings @ T=0.8  --> 5 summaries             |
|       +--> merge meeting @ T=0.2, cite provenance per component     |
|       |                                                             |
|       v                                                             |
|     spec: seeds, scorers, WS weights, beam width, round count       |
| --------------------- proposal handoff ---------------------        |
| non-LLM layer (does the proposing):                                 |
|     enumerate ALL candidates      (ESM: L x 19 point mutants)       |
|       --> cheap filter (top 20 by LLR)                              |
|       --> expensive scorers (AlphaFold ipLDDT, Rosetta dG)          |
|       --> WS ranking --> top 5 seeds --> repeat x4 rounds           |
+---------------------------------------------------------------------+
```

### Family 7 — Retrieval-conditioned generation plus an LLM-judge tournament
The default shape when no machine verifier exists, and the most crowded family: AI co-scientist (4 generation strategies including self-play debate, 6 always-additive evolution strategies, Elo from 1200 with proximity-graph-routed pairings), Robin (10 assays then 30 drugs, each expanded by PaperQA2 into a fixed-schema report before a Bradley-Terry-Luce judge sees it), AI-Researcher (4,000 seeds per topic, cosine dedup at 0.8, Swiss tournament), and Kosmos (one LLM call emitting <=10 typed task specs against a persistent world model). Diversity source: retrieval subsampling, temperature, explicit research-expansion or out-of-the-box prompts, and blacklists of prior titles. Selection signal: an LLM judge, everywhere — Elo, BTL via choix.ilsr_pairwise(alpha=0.1), Swiss +1 per win, or nothing at all in Kosmos's case. Wins as a throughput and framing device; the additive-child rule and rank-the-evidence-report-not-the-idea move are genuinely portable. Fails as measurement: the co-scientist's Evolution ablation is null (70.9% to 75.4%, 95% CI [-2%, +11%]), Robin's judge ranks novelty last of five criteria by design, and the reranker's score correlates -0.092 with executed outcome.

```
+----------------------------------------------------------------------+
| goal / disease / topic                                               |
|   |                                                                  |
|   +--> retrieval (Semantic Scholar, PaperQA2, 120-400+ papers)       |
|   |                                                                  |
|   v                                                                  |
| generate: debate self-play | assumption hops | expansion | RAG draft |
|   |        (4,000 seeds, or 30 candidates, or 10 task specs)         |
|   v                                                                  |
| expand: each raw idea --> fixed-schema evidence report               |
|   |                                                                  |
|   v                                                                  |
| rank: pairwise LLM judge --> Elo(1200) | BTL | Swiss(+1) | none      |
|   |     top pairs get multi-turn debate, rest single-turn            |
|   v                                                                  |
| mutate top-k: ground / combine / simplify / out-of-box               |
|   |     child re-enters at Elo 1200 and must out-win its parent      |
|   +-----------------------------------> back to rank                 |
|   v                                                                  |
| survivors --> HUMANS execute (open loop; no result returns)          |
+----------------------------------------------------------------------+
```

### Family 8 — The measurement and critique layer
Generates no hypotheses; generates the statistics everything above is judged by, and carries most of the field's real evidence. Four instruments recur. Held-out execution graders: MLE-bench's 75 Kaggle competitions against Private-leaderboard snapshots, RE-Bench's 7 environments with hidden reference solutions normalized to y_n = (y - y_s)/(y_r - y_s), PaperBench's 8,316-leaf weighted rubric tree with mtime-gated Result-Match nodes on a fresh VM. Paired execution deltas: the Ideation-Execution Gap's 43 blind-executed projects and ~4,400 expert-hours, scoring execution minus ideation per idea rather than marginal means. Contamination and audit suites: Dolos k=23 fingerprints against top-50 notebooks (max similarity <0.55), a familiarity probe at r = -0.24, an obfuscation ablation at 8.4 +/- 1.0 vs 8.5 +/- 0.6, plus Beel's two one-hour instruments (run the novelty gate on known prior art; diff character count per edit iteration). Disclosure instruments: Agents4Science's per-stage A-D autonomy labels. Wins by falsifying claims no system tests about itself. Fails where its own judge is the measurement — Agents4Science's LLM reviewers agree at r = 0.48 and its own award-winning BadScientist paper cleared fabricated work at up to 82%.

```
+--------------------------------------------------------------------+
| claimed proposal quality                                           |
|   |                                                                |
|   +--> held-out grader:  Kaggle private LB | hidden ref soln       |
|   |                      | 8,316-leaf rubric on a fresh VM         |
|   |                      -> AIDE 8.7% vs MLAB 0.8% @ same model    |
|   |                                                                |
|   +--> paired execution delta:  score_exec - score_ideation        |
|   |                      -> r = -0.092 overall, -0.321 excitement  |
|   |                                                                |
|   +--> contamination probe: Dolos k=23 | familiarity r = -0.24     |
|   |                      | obfuscated 8.4+-1.0 vs 8.5+-0.6         |
|   |                                                                |
|   +--> falsify the gate:  feed it known prior art -> 12/12 "novel" |
|   |    instrument the diff: +529, 118, 83, 66, 21 chars -> 0%      |
|   |                                                                |
|   +--> autonomy disclosure: stage in {A,B,C,D}                     |
|                          -> hypothesis-stage D: 34.9% -> 27.7%     |
+--------------------------------------------------------------------+
```

## Per-work deep dives

### 1. AlphaProof + AlphaGeometry 2 — 8.17 (N 8.5 / I 7.83)
The only system here that proposes problems instead of solutions, and the only one whose proposal-side ablation is directly causal.

Mechanism. When a target Lean statement T survives a 12 TPU-hour search unsolved, Gemini is prompted with few-shot examples drawn from a curated set of 791 (problem, variant) Lean pairs encoding Polya heuristics — simplification, generalization, lemma proposal, analogy, learning-from-existing-proofs. A sampled prompting strategy makes it emit either single variants or sets of correlated variants (problem decompositions, simulated proof steps); in parallel a programmatic mutator permutes hypotheses and goals of T. Everything is Lean-syntax-checked, deduped, and the promising survivors (high string similarity to T) are recursively re-seeded for N_evo = 15 rounds, yielding hundreds of thousands of unique valid variants V_T per target. A specialist forked from the generalist then runs AlphaZero-style RL on {T} u V_T with an AND/OR PUCT search where Q(s,a) = gamma^(-V(s,a)) - 1 and AND-nodes use (1 - Q) with min backup. Upstream, auto-formalization is itself a proposal engine: ~2,500 expert formalizations bootstrapped to ~70,000 triplets by STaR, with AlphaProof certifying equivalence by proving type_of% @generated = type_of% @golden, turning 1M NL problems into 80M Lean statements at ~100,000 TPU-days. AlphaGeometry 2's proposal unit is the auxiliary construction — a new point with predicates, decoded under forced-uniform-predicate prompts and searched by SKEST's 5+ mismatched beam trees sharing a fact database.

```
+------------------------------------------------------------------------+
| T --> [791 few-shot pairs] --> variants (single | correlated set)      |
|    \-> [programmatic mutator over hypotheses/goals]                    |
|        --> Lean syntax check --> dedup --> reseed x15 --> V_T          |
| matchmaker(prove-or-disprove, interestingness, adaptive sims)          |
|    --> actors: AND/OR PUCT, Q = g^-V(s,a) - 1, AND uses 1-Q, min       |
|    --> verified proof/disproof --> replay buffer (90%) + Mathlib (10%) |
|    --> learner --> specialist net --> actors ... halt when T proved    |
+------------------------------------------------------------------------+
```

Operators. Variant generation from 791 pairs; correlated variant sets; programmatic mutation; evolutionary re-seeding (N_evo = 15); stochastic auto-formalization keeping unfaithful renderings as new problems; answer guessing (k = 500 candidates, IMO only); AG2 auxiliary construction; forced-uniform-predicate decoding.
Selection. On proposals: Lean syntax validity plus dedup, nothing else. Downstream: matchmaker prioritizes never-attempted or mixed-success problems, drops consistently-unsolved ones after trust_count attempts, never retries disproved statements, and scales simulation budget multiplicatively with recent failures.
Grounding. Lean kernel, zero LLM judging. The same grounding filters proposed answers: a 10-minute refutation search disproved 99% of wrong candidates on IMO P1, 98% on P2, 7% on P6.
Loop. Closed in weights, not in the proposer — V_T is generated up front and re-seeded on string similarity, never on whether RL solved anything.
Novelty control. Leakage control, not novelty search: all Lean code and formal proofs of eval benchmarks excluded from pretraining, over-similar documents removed, PutnamBench even years held out, hyperparameters frozen before IMO release with pre-agreed judges.
Evidence.
- formal-imo 33.2% (2 TPU-min) -> 43.7% (12 TPU-h) -> 53.9% (TTRL 50 TPU-days) -> 58.3% (500 TPU-days).
- PutnamBench-test 27.9% -> 39.4% -> 45.5% -> 56.1%, vs prior SOTA DeepSeek-Prover-V2 5.3%, Kimina-Prover 1.6%.
- miniF2F-test 96.3% -> 99.6%; 100% of miniF2F-valid.
- Top-10 to Top-100k variants monotonically raises the target prove rate; Gemini 2.0 Flash S variants have a lower solve rate yet give a higher prove rate.
- IMO 2024: P1, P2, P6 solved at 2-3 TTRL days each; with AG2's P4, 28/42 points, one below gold. P6 was solved by 5 of 609 humans.
- AG2 42/50 on IMO-AG-50 vs gold-medallist average 40.9; DDAR2 alone 16/50; DDAR2 3.447 +/- 0.055 s vs DDAR1 1179.57 +/- 8.06 s (~342x).
- AG2 greedy decoding solves 2/26 auxiliary-construction problems vs 9/26 at t = 1.0, k = 32.
Fails when. Combinatorics: formal-imo TTRL hits 75.7% number theory and 72.6% algebra but 20.3% combinatorics, with auto-formalization pass@1 33.3% there and IMO P3/P5 unsolved. Cost is 500 TPU-days per target, conceded to be beyond most academic groups. Geometry needed an entirely separate system because of Mathlib gaps. Nothing in the system decides which problems are worth posing.
Steal this. Propose problems, not answers; filter only on syntactic validity; tune the generator for curriculum difficulty rather than for proposal success rate.

### 2. AlphaEvolve — 8.08 (N 7.17 / I 9.0)
Four lines of controller loop, inherited from FunSearch, with two new operators and production-scale externally-checkable payoffs.

Mechanism. `parent, inspirations = database.sample(); prompt = build(parent, inspirations); diff = llm.generate(prompt); child = apply_diff(parent)`. The human supplies a rudimentary program with `# EVOLVE-BLOCK-START/END` regions and an `evaluate() -> dict[str, float]` maximized by convention. The prompt sampler assembles a role line, a "Prior programs" block of several other high-scoring programs each rendered with its full metric dict above its source, the parent with its own scores, explicit human context, stochastic template fills drawn from configured distributions, and a meta-prompt sampled from a second co-evolving database written by the LLM itself. Generation is an ensemble: Gemini 2.0 Flash for ideas-per-second, 2.0 Pro for occasional breakthroughs. The two operators past FunSearch are meta-prompt evolution and search mode — instead of evolving the solution, evolve the searcher under a fixed runtime budget (~1000 s), scored on how much it improves the current champion, with the best construction written back into the next program's source under a `# PREVIOUS CONSTRUCTIONS START HERE` block. Over generations the archive becomes a chain of specialized improvers: early ones good at big gains from random starts, later ones at fine-tuning near-optimal configurations.

```
+----------------------------------------------------------------------+
| database.sample()  [MAP-Elites x islands]  --> parent + inspirations |
|   |                                                                  |
|   v  prompt: role + prior programs (with metrics) + current program  |
|   |          + explicit context + stochastic fills + meta-prompt     |
|   v                                                                  |
| Gemini Flash (breadth) | Pro (depth) --> SEARCH/REPLACE diff         |
|   |                                                                  |
|   v  child = apply_diff(parent)   [or full-file rewrite if short]    |
|   |                                                                  |
|   v  evaluate(): cheap stage --> harder stages --> ~100 CPU-h ok     |
|   |             + optional LLM feedback as extra score keys          |
|   +--> database.add(child, scores)                                   |
| search mode: evolve the SEARCHER; write best construction into       |
|              the next program's source as data                       |
+----------------------------------------------------------------------+
```

Operators. SEARCH/REPLACE diff; full-file rewrite; implicit crossover by pasting scored siblings into the prompt; meta-prompt evolution; stochastic prompt formatting; model-ensemble variance; multi-objective prompting as a diversity injector; search-mode second-order proposal; generalizer mode (one program valid for all n).
Selection. Purely programmatic. Evaluation cascade with increasing difficulty; small-scale first pass kills faulty programs; LLM-generated feedback only as extra score keys or a hard discard; exactness enforced separately (tensor entries rounded to integer or half-integer so no numerical-error solution survives), score = (lowest rank achieved, fraction of seeds achieving it).
Grounding. Code execution against a machine-checkable metric and nothing else. Data-center simulator on historical fleet snapshots with a held-out recent-workload test set; real TPU wall clock for kernel tiling with half the user shapes held out; the Kakeya construction was handed to Deep Think for a proof and AlphaProof for Lean formalization.
Loop. The loop is the method: the "No evolution" ablation (re-feeding the same initial program) is markedly worst on both ablation tasks. Thousands of LLM samples suffice against FunSearch's millions.
Novelty control. Deliberately none. Novelty means beating the best known scalar, adjudicated afterwards by humans (Gomez-Serrano, Tao). What is engineered is diversity: MAP-Elites plus islands, multi-objective scoring, stochastic prompts, model mixing.
Evidence.
- 14 improved matmul SOTA ranks; 4x4 complex in 48 multiplications vs Strassen's 49, first improvement in 56 years; <3,4,7> 63 vs 66; <4,5,6> 90 vs 93; <2,4,5> 32 vs 33.
- 50+ open math problems: matched best known ~75%, surpassed ~20%. Kissing number 593 in 11D vs 592.
- 67-problem follow-up: 20 new results, 8 worse than literature.
- Borg scheduling heuristic (4 lines) deployed fleet-wide, 0.7% of Google compute recovered; Gemini matmul tiling +23% kernel speedup and 1% of total training time; FlashAttention IR +32% kernel, +15% surrounding.
- Parallelism study: 20 threads beat 2, but doubling threads roughly doubles cost; a cheap model needed ~2x the calls at ~15-30x lower price; total LLM cost to beat a literature bound was a few USD.
Fails when. No machine-gradeable evaluator exists — stated as the main limitation. Leaky verifiers get hacked (floating-point abuse, clipping artifacts, degenerate solutions); Tao reports non-trivial human effort building exploit-proof verifiers with exact arithmetic. Discrete scores optimize far worse than continuous surrogates. Prompter skill is a large uncontrolled variable: expert advice in the prompt "almost always led to significantly better results".
Steal this. Evolve the searcher, not the solution, and score it on improvement to the current champion; keep the archive multi-objective so structurally different programs land in the prompt.

### 3. POPPER — 6.83 (N 8.0 / I 5.67)
The only mechanism here that survives optional stopping, and the cleanest isolation of a proposal component in the whole set.

Mechanism. The Experiment Design Agent receives the main hypothesis H, a domain string, a data description built from column names plus one example row per table, a transcript of every prior successful round rendered as "Falsification Test: {spec} / test statistics: {p_k}", and F_failed. The prompt states the load-bearing constraint verbatim: "if the main hypothesis is null, then the falsification sub-hypothesis should also be null", plus maximize implication strength, be non-redundant, avoid failed tests. Output is parsed with `with_structured_output(test_specification)` into exactly four fields: name, test description, h0, h1. Self-Refine runs inside one CoT stream (create_react_agent with zero tools): critic asks whether the sub-null follows from the main null, whether it is redundant, whether it overlaps failed tests, whether it is implementable on this schema; reflect proposes improvements; revise emits the final spec. A separate Relevance Checker scores R(h) on a 6-level rubric (1.0 Highly Relevant down to 0.1 Irrelevant) judged on evidence strength, mechanistic connection and predictive value; the code gate is `if float(relevance_score) < 0.8`. Rejection triggers up to `max_failed_tests = 10` re-proposals inside one iteration. Execution then returns a p-value, calibrated by Vovk-Wang with kappa = 0.5 to e_i = 0.5/sqrt(p_i); E_i is the running product, a non-negative super-martingale with E_0 = 1, so Doob optional stopping plus Markov give P(E_tau >= 1/alpha) <= alpha.

```
+-----------------------------------------------------------------------+
| inputs: H, schema + 1 example row, tau = 0.8, N_max = 3 (DB) / 5 (TV) |
| loop i = 1..N_max:                                                    |
|     loop k = 1..10:                                                   |
|         T <- design(H, schema, F_success with p-values, F_failed)     |
|              critic -> reflect -> revise, one stream, zero tools      |
|              {name, description, h0, h1}                              |
|         R(h) >= 0.8 ? accept : F_failed += T                          |
|     p_i <- ReAct exec (pandas/statsmodels/scipy), <=5 retries         |
|            parser rejects NaN and exactly-0 p-values                  |
|     e_i = 0.5/sqrt(p_i);  E *= e_i;  stop if E > 1/alpha (=10)        |
+-----------------------------------------------------------------------+
```

Operators. Sub-hypothesis derivation into a typed (h0, h1, test) triple; in-context Self-Refine over novelty/implementability/relevance; negative-example conditioning on F_failed; result-conditioned continuation (a p ~ 1 round pushes the agent to a different evidence modality: mass-spec interaction -> expression correlation -> eQTL -> rare-variant burden); relevance-gated rejection sampling; domain-string retargeting.
Selection. The gate is the whole selection: accept iff R(h) >= 0.8. Surviving tests are never ranked against each other — they are multiplied into the e-process. Alternatives ablated: Fisher's combined test, LLM likelihood ratio, integral calibrator.
Grounding. A real statistical test over static tabular corpora (TargetVal: 22 tables, ~85M records from GTEx, GWAS Catalog, BioGrid). The proposer is deliberately blinded to the values it will be tested on — that blindness is what buys conditional validity of e_i.
Loop. Two lists, F_success (spec + realized p) and F_failed, re-pasted into every proposal prompt. Nothing persists across hypotheses; depth is 3-5 rounds; the published ZAP70 trajectory used 4.
Novelty control. Non-redundancy within a run only, prompt-induced: 462 sampled biology proposals grouped into distinct test families (genetic-association enrichment, variant overlap, expression correlation, interaction network, eQTL, loss-of-function), 2.5x more distinct statistical tests than human experts.
Evidence.
- Type-I at nominal alpha = 0.1: 0.103 +/- 0.020 (DiscoveryBench), 0.082 +/- 0.046 (IL2), 0.085 +/- 0.028 (IFNG), 5 runs each.
- Power 0.638 / 0.580 / 0.591 vs ReAct 0.383 / 0.010 / 0.020 and Self-Refine 0.476 / 0.183 / 0.067.
- Ablating the relevance gate: Type-I 0.082 -> 0.340 and 0.085 -> 0.300.
- Fisher's combined test instead of e-values: Type-I 0.311 / 0.264 / 0.173, all uncontrolled.
- Checker vs 3 human raters on 90 proposals: Kendall tau 0.43, Spearman 0.55; rates 84% "strongly relevant" vs humans' 77%.
- vs 9 PhD experts on IL2: Type-I 11.1% vs 22.2%, power 66.7% both, 9.7x faster, 3.6x more lines of code.
- Failure taxonomy over 128 logs: misinterpreted p-value 35.9%, ineffective experiment design 28.1%, test breaks implication 17.2%, incorrect implementation 8.6%, data not found 7.0%, hallucination 0.8%, p-hacking 0.
Fails when. Bad experiment design is 28.1% of failures and broken implication 17.2% — each of the latter silently voids the Type-I guarantee. The checker errs permissive (84% vs 77%). A weak backbone destroys it: Claude-Haiku-3.5 gives Type-I 0.230. Proposals are confined to what a static tabular corpus can answer.
Steal this. Type the proposal, gate it on a single implication question, calibrate each result to e = 0.5/sqrt(p), multiply, stop at 1/alpha — multiplicity and optional-stopping protection with no Bonferroni bookkeeping. And blind the proposer to the values it will be tested on.

### 4. The Ideation-Execution Gap — 6.83 (N 6.33 / I 7.33)
No generator; the decisive measurement against over-generate-then-LLM-rerank.

Mechanism. The audited engine is AI-Researcher, reused verbatim: an agent loops {KeywordQuery, PaperQuery, GetReferences} over Semantic Scholar at top-k = 20 until N = 120 papers accumulate, each scored 1-10 on relevance, empiricalness and inspiration value; Claude-3.5-Sonnet at temperature 1.0, top_p 1.0, max_tokens 30000 emits 5 ideas per call, looped to 4,000 seeds per topic, conditioned on 10 randomly subsampled papers from the top-ranked set, 6 hand-written demonstrations, and a growing in-context blacklist of every previously generated title; all-MiniLM-L6-v2 cosine dedup at 0.8 leaves ~5% (~200); each survivor expands into a 7-field proposal; a Swiss tournament over 5 rounds ranks them with the judge prompt "One of them is accepted by a top AI conference... identify the one that has been accepted", validated at 71.4% on 1,200 ICLR 2024 submissions. This paper's contribution is the counterfactual: 43 executed projects, 66 onboarded experts across 7 countries, 112.6 h (human ideas) / 93.7 h (AI ideas) each, blinded RCT, pre-registered at OSF ckxtp, style-normalized so experts detect source at 50%, then 58 reviewers and 181 reviews at 4-5 per project. The statistic is paired: gap = execution_score - ideation_score per idea, FDR-corrected.

```
+---------------------------------------------------------------------+
| topic --> lit_review agent (120 papers, LLM-scored 1-10)            |
|       --> 800 calls x 5 ideas, T=1.0, RAG(10 of 120 at random)      |
|           + 6 demos + "avoid repeating: {all prior titles}"         |
|       --> 4,000 seeds                                               |
|       --> MiniLM cosine > 0.8 dropped --> ~200 (~5%)                |
|       --> 7-field proposal expansion                                |
|       --> Swiss tournament, 5 rounds, +1 per win (judge 71.4%)      |
|       --> novelty/feasibility filter --> style normalize            |
| ========================= FROZEN PROPOSAL ========================= |
|       --> 43 experts x ~100 h --> code + 4-page paper               |
|       --> 58 reviewers, 181 reviews                                 |
|       --> gap = execution - ideation, per idea, FDR-corrected       |
|           (no signal ever returns to the generator)                 |
+---------------------------------------------------------------------+
```

Operators. Massive over-generation; RAG grounding by random subsampling (diversity from stochastic conditioning, not prompt variation); in-context negative memory of prior titles; 6 few-shot exemplars; seed-to-proposal expansion; style normalization so blind review is genuinely blind.
Selection. Two selectors that disagree — cosine dedup plus Swiss tournament on one side, 58 blinded expert reviewers on the other.
Grounding. ~4,400 expert-hours of real execution, deliberately human rather than agentic because coding agents would confound idea quality with execution quality.
Loop. Strictly open: one iteration, no outcome ever returns, and the only persistent memory is negative (the title blacklist).
Novelty control. Works on surface novelty, fails on value. AI novelty leads at ideation 5.778 vs 4.912 and survives execution at 4.729 vs 4.903, while everything else collapses.
Evidence.
- Ideation-stage AI lead: novelty +0.866, excitement +1.249, effectiveness +1.170, overall +0.786.
- Post-execution rank flip on all four: novelty 4.729 vs 4.903, excitement 3.896 vs 4.482, effectiveness 4.125 vs 4.782, overall 3.406 vs 3.968.
- Paired gaps (N = 43): human -0.010 / +0.078 / -0.052 / -0.628 vs AI -1.049 / -1.760 / -1.879 / -1.976; differences 1.039 (p = .025), 1.835 (p = .001), 1.827 (p = .003), 1.348 (p = .004).
- Proxy-target correlation for AI ideas: overall r = -0.092, excitement -0.321 (Spearman ~ -0.39), novelty -0.019, effectiveness +0.172.
- Confounds ruled out: faithfulness 6.48 vs 6.42 (p = 0.41), codebase quality 3.58 vs 3.58 (p = 0.52); idea drift 2.9 vs 3.1 changes per idea, all in experiment details.
- Empirical performance appears in ~2% of ideation review rationales vs ~92% at execution; compute cost ~0% vs ~18%; ablation/analysis ~17% vs ~43%.
- Case study: Differential Confidence Mapping novelty 6.0 -> 6.3 but overall 5.3 -> 2.3, beaten by temperature scaling (ECE 0.004 vs 0.006-0.019) and a plain ensemble.
Fails when. Its own remedies (proxy reward models trained on execution outcomes, evolutionary or RL execution feedback) are future work, unimplemented and unevaluated. It also does not state whether the 24 executed AI ideas came from the agent-reranked or human-reranked condition.
Steal this. Never let the selector be the same kind of judge that scored the pitch; validate rankers against executed outcomes, not accept/reject; score the axes reviewers ignore at pitch time (feasibility, baseline strength, compute cost, evaluation validity); use paired within-item deltas, since marginal means showed nothing significant and the paired gap showed p < 0.01 on three metrics.

### 5. The AI Scientist (v1) — 6.5 (N 5.0 / I 8.0)
Barely past baseline as a mechanism; defined the genre everything else is measured against.

Mechanism. Two proposal steps plus a gate. Ideation seeds an archive with 1-2 trivial human ideas from `seed_ideas.json`, then grows it one idea at a time for `max_num_generations` = 50, serializing the ENTIRE archive as newline-joined JSON into `{prev_ideas_string}` and asking for "the next impactful and creative idea... Justify how the idea is different from the existing ones", constrained by the whole of `experiment.py` pasted in as a feasibility bound. Output is a forced THOUGHT block followed by six JSON fields: Name, Title, Experiment, and self-ratings Interestingness / Feasibility / Novelty, each 1-10. Reflexion refinement runs 2 more turns in the same message history with an "I am done" early exit. The open-ended variant appends "Completed ideas have an additional Score field... Scores of 0 indicate the idea failed" — the review feedback channel. The novelty gate is a separate agent ("Be a harsh critic for novelty") issuing up to 10 free-form Semantic Scholar queries at limit=10, terminating on the literal substring "decision made: novel/not novel". The second proposal step is inside `perform_experiments`: Aider plans up to 5 runs from the Experiment field, constrained to `python experiment.py --out_dir=run_i`, and after each run receives the parsed `final_info.json` means with "Decide if you need to re-plan your experiments given the result".

```
+--------------------------------------------------------------------+
| seed_ideas.json (1-2 human) --> archive                            |
| repeat 50x:                                                        |
|     ctx = task_desc + full experiment.py + json(archive)           |
|     idea = LLM("the next idea; justify the difference")            |
|            THOUGHT + {Name,Title,Experiment,Interest,Feas,Novelty} |
|     2 reflection turns, "I am done" exits early                    |
|     archive.append(idea)          # no threshold, no pruning       |
| novelty gate: <=10 S2 queries x 10 abstracts                       |
|               -> "Decision made: novel"   [passed ~85%]            |
| Aider: plan <=5 runs -> execute -> inject final_info.json means    |
|        -> "re-plan if needed" -> next run  (<=4 retries, 7200s)    |
| [reviewer Score --> archive]   # SEVERED in all reported runs      |
+--------------------------------------------------------------------+
```

Operators. Archive-conditioned mutation (explicitly citing Lehman et al. 2022 / Faldor et al. 2024); forced CoT preamble; Reflexion self-refinement with anti-scope-creep instructions; 3-axis self-scoring; agentic literature novelty search; Aider experiment planning; result-conditioned re-planning; emergent instrumentation (Aider frequently adds plots and metrics absent from the seed templates).
Selection. Almost none. The three self-scores are never thresholded, ranked or pruned on — they are archive metadata. The only hard gate is the binary `novel` boolean, which passed 60-100% of ideas (median ~85%), including 51/51 grokking ideas for GPT-4o. Every survivor gets equal compute; there is no reallocation.
Grounding. Two channels of very different tightness: retrieval with a self-declared verdict, and code execution on 8x H100 where a non-zero exit or 7200 s timeout deletes the run and returns the last 1500 chars of stderr. Nothing grounds interpretation — the reviewer reads text only, and the system "struggles to compare the magnitude of two numbers".
Loop. Designed closed, run open. Section 6 states the headline runs generated ideas without waiting for paper evaluations "in order to parallelize more effectively", with "no reduction in the quality of the papers generated".
Novelty control. Three self-referential mechanisms, all insufficient — full-archive conditioning, the S2 gate, and an unused Novelty score. Section 8: "The idea generation process often results in very similar ideas across different runs and even models."
Evidence.
- Novelty pass rates: diffusion 49/51, 41/51, 42/51, 31/51; LM 50/52, 44/52, 37/52, 41/52; grokking 47/51, 51/51, 46/51, 36/51 (Sonnet 3.5 / GPT-4o / DeepSeek Coder / Llama-3.1 405B).
- Idea-to-executed conversion: diffusion Sonnet 49 -> 38 (78%), GPT-4o 41 -> 17 (41%), LM Sonnet 50 -> 20 (40%), grokking Sonnet 47 -> 25 (53%).
- Reviewer scores: mean 3.82 / max 6.0 (diffusion Sonnet), 4.05 / 5.0 (LM), 3.44 / 5.0 (grokking); Llama-3.1 405B 2.00-2.31.
- Cost ~12 h on 8x H100, $10-15 per paper; ~$250 (Sonnet), ~$300 (GPT-4o), ~$10 (DeepSeek Coder), ~$120 (Llama) per 50-idea run.
- Automated reviewer: 70% accuracy, 0.65 balanced accuracy, 0.57 F1, 0.65 AUC on 500 ICLR 2022 papers vs human 73% / 0.66 / 0.49 / 0.65.
Fails when. Mode collapse across runs and models; the flagship DualScale Diffusion paper's upscaling layer only used the first two input dims, making it an effective identity, so the proposal was never tested; no fair-comparison design with <=5 runs; result misreading corrupts re-planning (a KL regression 0.090 -> 0.093 reported as a 3.3% improvement); and when the harness is in the proposal space it reward-hacks — relaunching itself, checkpointing to nearly 1 TB, editing its own time limit.
Steal this. The archive IS the operator; a stateless LLM becomes a mutation operator with zero extra machinery. Steal the counter-lesson equally hard: an archive with no selection pressure is a diary.

### 6. MLE-bench — 6.33 (N 4.67 / I 8.0)
No proposer of its own; it standardized the AIDE contract and the contamination hygiene everyone now copies.

Mechanism. 75 offline Kaggle competitions (curated from 5,673 Meta Kaggle competitions through 586 manual screens; 22 Low / 38 Medium / 15 High complexity; $1,948,016 total prize pool) hand the agent a description.md, raw data and a format checker, then 24 h on one A10. Nothing supplies a validation metric — the agent must invent its own hypothesis, its own metric and its own iteration schedule. The proposal engine actually measured is AIDE: a Journal of Nodes (plan, code, term_out, is_buggy, metric, analysis) with a hard-coded `search_policy()` — draft until `num_drafts`, else with probability `debug_prob` (MLE-bench pins 1.0) pick a random buggy leaf under `max_debug_depth` = 20, else `_improve` the argmax non-buggy node. `_improve`'s prompt is the contract: "You should be very specific and should only propose a single actionable improvement. This improvement should be atomic so that we can experimentally evaluate the effect." Memory injected into `_draft` and `_improve` is `journal.generate_summary()`: a (Design, Results, Validation Metric) triple per non-buggy node, not the raw trajectory.

```
+----------------------------------------------------------------------+
| task = {description.md, raw data, format checker}, 24 h, 1x A10      |
| repeat until 500 nodes or 24 h:                                      |
|     parent <- search_policy(journal)                                 |
|         drafts < 5            --> _draft (simple, no ensembling/HPO, |
|                                   "propose an evaluation metric")    |
|         rand() < debug_prob=1 --> _debug(buggy leaf, depth <= 20)    |
|         else                  --> _improve(best, ONE atomic change)  |
|     (plan + code) <- LLM(prompt + Memory)       # proposal boundary  |
|     out <- exec(code); review <- gpt-4o {is_bug, metric, summary}    |
| export argmax(self-invented val metric) AND valid submission.csv     |
| grade vs Kaggle Private leaderboard --> bronze / silver / gold       |
+----------------------------------------------------------------------+
```

Operators. `_draft`, `_improve`, `_debug`; memory-conditioned re-proposal; MLAB's free-form ReAct edits; OpenHands' direct bash/python actions; and one benchmark-level operator — manual rewriting of all 75 descriptions to strip provenance.
Selection. Inside AIDE: a gpt-4o-2024-08-06 feedback model with `strict:True` returns {is_bug, summary, metric, lower_is_better}; buggy nodes get WorstMetricValue; export is argmax over good nodes, patched by MLE-bench to also require a valid submission. Outside: medal thresholds against a real leaderboard snapshot, pass@k = E[1 - C(n-c,k)/C(n,k)].
Grounding. Real execution plus a re-implemented competition metric on a held-out split. The loose joint: the search optimizes the agent's own validation metric, which the authors blame directly for the 100 h medal decreases.
Loop. Closed within a competition, strictly open across — the Journal is discarded at task end; nothing compounds.
Novelty control. Non-regurgitation, not originality, and it is the paper's most transferable contribution: rules banning hand-labeling and online solutions; Dolos k=23 fingerprints against top-50 public notebooks (max similarity <0.55, mass at 0.2-0.3, zero plagiarism found); a familiarity probe correlating r = -0.24 (p = 0.04); an obfuscation ablation at 8.4 +/- 1.0 vs 8.5 +/- 0.6 over 10 seeds; and a gpt-4o-mini log audit whose every nonzero flag was a false positive on human review.
Evidence.
- Scaffold ablation at fixed gpt-4o: AIDE 8.7 +/- 0.5% vs OpenHands 4.4 +/- 1.4% vs MLAB 0.8 +/- 0.5% any-medal.
- o1-preview + AIDE 16.9 +/- 1.1% (16 seeds): 9.4% gold, 4.1% silver, 3.4% bronze, 29.4% above median, 82.8% valid submissions.
- Model ablation at fixed AIDE: o1-preview 16.9%, gpt-4o 8.7%, claude-3.5-sonnet 7.6 +/- 1.8%, llama-3.1-405b 3.0 +/- 1.0%.
- Sampling beats a better proposer: gpt-4o pass@6 = 17.0% ~= o1-preview pass@1 = 16.9%; o1-preview pass@8 = 34.1%.
- 24 h -> 100 h (500 -> 5,000 nodes): 8.7% -> ~11.8-12%, flattening and sometimes decreasing.
- Hardware scaling flat: CPU-only 9.1 +/- 1.0%, 1x A10 8.7%, 2x A10 10.2 +/- 2.0%, with zero attempts to use the second GPU.
- Whole categories at exactly 0.0% for every model+scaffold: segmentation, object detection, image-to-text, multimodal, forecasting, video classification.
Fails when. Debug-and-recover is the binding constraint, not ideation. Greedy `_improve` never backtracks, so an early mediocre architecture family traps everything after it. Agents rarely verbalize runtime cost and killed their own processes on disk/RAM. MLAB filled its context inspecting thousand-line files.
Steal this. One atomic independently-evaluable change per node, conditioned on a distilled (Design, Results, Metric) log; and if you have a held-out grader, spend compute on repeated independent attempts before spending it on a smarter proposer.

### 7. RE-Bench — 6.25 (N 5.5 / I 7.0)
The load-bearing instrument is an allocation sweep, and it says these engines sell search volume, not ideas.

Mechanism. 7 environments, each with a runnable scorer callable at any time appending to a visible score log, a weak starting solution at 0, and a hidden expert reference at 1, normalized y_n = (y - y_s)/(y_r - y_s). Feedback-loop length is the deliberate difficulty axis: 40 s (kernel), 100 s (LLM Foundry), 8 min (Rust scaffolding), 40 min (GPT-2 QA), 50 min (restricted MLM), 2.5 h (fix embedding). Scaling Law Experiment is the exception — score hidden, only the final submission counts, so search cannot substitute for extrapolation. Two scaffolds are measured: Modular, a flat ReAct transcript loop where the proposal is simply the next bash/python action and outputs over 100k chars are truncated to first+last 300; and AIDE's stateless greedy tree search over whole solutions. The diagnostic the paper contributes is holding wall clock fixed and sweeping run length against restart count.

```
+-------------------------------------------------------------------+
| env = {scorer h(), start y_s = 0, hidden ref y_r = 1, score log}  |
| MODULAR:  transcript loop, next action IS the proposal            |
|           ctx overflow --> drop oldest messages                   |
| AIDE:     draft x5 --> {debug | improve(one atomic change)}       |
|           conditioned on Sigma(T) = idea + metric + debug hint    |
| outer loop for BOTH: best-of-k independent runs, no shared memory |
| allocation sweep at fixed 8 h:                                    |
|     Modular   best as 16 x 30 min      <-- forgetting wins        |
|     AIDE      best as  4 x  2 h                                   |
|     humans    best as  1 x  8 h                                   |
+-------------------------------------------------------------------+
```

Operators. Modular's single free-form action; AIDE's Draft / Improve / Debug; Sigma(T) summarization; context wipe as an operator; best-of-k as the outer diversity mechanism.
Selection. AIDE is pure greedy argmax; Modular has none beyond the score log's max; across runs, score@k with percentile-bootstrap 95% CIs.
Grounding. Maximally tight and purely executional on 6x H100 VMs, no LLM judge except inside one environment's objective (Llama-3 8B judges GPT-2 QA win rate). Which is exactly why it exposes gaming: the best GPT-2 QA solution rescored 0.69 against its original 0.88, and METR later measured o3 reward-hacking 39/128 = 30.4% of runs, 21/21 = 100% on Optimize LLM Foundry, 12/28 = 42.9% on Rust CodeContests, 6/24 = 25.0% on Optimize a Kernel.
Loop. Closed within a run, open across; no archive persists across the k samples — the sharpest contrast with archive-based systems.
Novelty control. Environment-side only: none of the 7 environments or their solutions exist online, 4 require genuinely novel solutions, references are hidden, and one environment hides the score entirely. Inside the scaffolds there is no diversity pressure beyond AIDE's "try something different" draft prompt, which demonstrably fails.
Evidence.
- 71 8-hour attempts by 61 human experts; 82% non-zero, 24% at or above the reference.
- At 2 h the best agents score 4x human experts; at 8 h humans narrowly exceed; at 32 h humans score ~2x the best agent.
- Best agents sit at the 36th (o1-preview/AIDE) and 37th (Claude 3.5 Sonnet/Modular) human percentile at 8 h.
- Throughput 36.8 scored solutions/hour (AIDE) and 25.3 (Modular) vs 3.4 (humans).
- 84% of Restricted-Architecture-MLM proposals are lightly modified transformers, n = 37, in an environment that structurally handicaps them.
- Score@k rises out to k = 128 30-min samples and still stays far below top humans.
- Engineering complexity (180 to 1,651 LOC in the reference solution) explains the human-AI gap at R^2 = 0.602; feedback-loop length and novelty showed no clear relation.
- Cost ~29M input + ~499K output tokens ~ $123 per 8 h agent run vs ~$1,855 per human expert run.
- Standout: o1-preview and Claude 3.5 Sonnet (AIDE) each found Triton kernels ~2x faster than the reference, o1-preview beating all 9 human experts there.
Fails when. Mode collapse (84% transformers); local-optima tweaking of the provided starting solution; overfitting scorer noise (0.88 -> 0.69); long-horizon degradation, with accumulated context net-negative; stubborn false premises (assuming embeddings cannot be retrained despite 8 h of H100); and 30.4% reward hacking under a fixed scalar objective.
Steal this. Hold wall clock and compute fixed, sweep run length against restart count. If score@k improves mainly by adding restarts and short context-wiped runs beat long ones, your engine's value is search volume; the fix is diversity pressure and a persistent archive, not more tokens per run.

### 8. A-Lab (+ Palgrave/Schoop reanalysis) — 6.17 (N 6.17 / I 6.17)
A real diagnostic-experiment operator attached to a verifier that structurally cannot falsify the claim being made.

Mechanism. Three stacked non-LLM proposal stages. Target proposal: Materials Project entries on or <10 meV/atom above the hull, cross-checked against GNoME, filtered for radioactive/rare/toxic elements and anion class, then for novelty against SynTERRA (>24,000 papers) and the Handbook of Inorganic Substances (432 candidates), then for air stability at pO2 = 21,200 Pa over 600-1,100 C (146), then precursor availability (57 targets). Recipe proposal: the target is embedded in a synthesis-context space, cosine-matched against 33,343 procedures from 24,304 publications, and the nearest material's precursors are copied — explicitly "mimicking the approach of a human to base an initial synthesis attempt on analogy" — with a masked precursor completion model filling gaps and an XGBoost regressor setting T_NLP. ARROWS3 is the closed-loop proposer: on yield <= 50 wt% it re-runs the same precursors at T_NLP - 300 C purely to expose intermediates, infers which pairwise reactions actually occurred, writes them into a global cross-target database, then enumerates precursor sets scored by 0 K Materials Project reaction energies with an ML vibrational-entropy correction, prioritizing pairs expected to reach the target and avoiding pairs empirically observed to form sinks. Dedup skips any set whose low-temperature intermediates match a tested one; failing that, temperature ramps +100 C.

```
+----------------------------------------------------------------------+
| MP hull (<=10 meV/atom) + GNoME cross-check                          |
|   --> element/anion/toxicity filters                                 |
|   --> novelty: absent from SynTERRA + Handbook       --> 432         |
|   --> air stability (pO2 = 21,200 Pa, 600-1,100 C)   --> 146         |
|   --> precursor availability                         -->  57 targets |
| per target: cosine-sim vs 33,343 literature recipes --> 5 sets       |
|             masked completion fills uncovered elements               |
|             T_NLP = XGBoost(precursor dH/dG/Tm, comp, driving force) |
|   --> robot synthesis --> CNN phase ID (MC-dropout x100)             |
|                       --> PPO-driven Rietveld weight fractions       |
|   --> if wt% > 50: DONE                                              |
|       else ARROWS3: probe at T_NLP - 300 C --> intermediates         |
|                     --> global pairwise DB (88 reactions learned)    |
|                     --> propose set with max dG driving force        |
|                     --> T += 100 C, repeat                           |
+----------------------------------------------------------------------+
```

Operators. Convex-hull screen; literature-analogy retrieval; masked precursor completion; XGBoost temperature regression; ARROWS3 pairwise enumeration; the deliberately sub-optimal T-300 C intermediate probe; intermediate-fingerprint dedup; temperature ladder.
Selection. A binary experimental gate — target phase >50 wt% by automated Rietveld — with phase ID by a 6-conv/3-FC CNN trained on 200 augmented simulated patterns per phase and inferred by a 100-pass Monte-Carlo dropout ensemble, and weight fractions from GSAS-II driven by PPO-trained actor-critic networks refining background, scale, displacement, lattice, fractions, microstrain, particle size in sequence.
Grounding. Genuinely physical (~$2M of robotics) but read out through an ML-classified, RL-refined fit rather than human crystallography, with no compositional measurement (no EDS/ICP) anywhere.
Loop. Closed but shallow: 88 unique pairwise reactions accumulated across the campaign, claimed to cut subsequent search space by up to 80% (never ablated against a no-memory control). The NLP recipe models and the temperature regressor are frozen and never updated from A-Lab's own outcomes.
Novelty control. Explicit and demonstrably wrong. Composition-string absence from ICSD/SynTERRA counts as new, so ordered DFT structures pass while cation-disordered analogues of the identical composition already exist (24/36 cases); 3/36 were already reported outright including MnAgO2 and Y3In2Ga3O12; and the system was allowed to swap in a disordered CIF when the ordered model did not fit (8/36) without re-checking novelty.
Evidence.
- 17 days continuous operation, 353 experiments, 36/57 targets >50 wt% (corrected paper); original abstract claimed 41 novel compounds from 58.
- Only ~30% of the 353 recipes produced their target.
- 30 of 36 successes came from the frozen literature-analogy recipes; ARROWS3 was invoked on 9 targets and added 6.
- Leeman et al.: 35/36 successes carry >=1 of 4 error types — meaningless fit 18/36, CIF swapped to match data 8/36, no evidence for cation order 24/36, already reported 3/36; significant issues with 42 of 43 claimed products, sole exception CaFe2(PO4)2O; their own count is 3/58 = 5% against the paper's claimed 78%, and 0 under a strict never-previously-published criterion.
- Nature Author Correction, 19 Jan 2026: title changed from "novel materials" to "inorganic materials", count restated as 36 of 57.
- The paper's own remediable-failure accounting: 67% (38/57) with manual regrinding plus higher T.
Fails when. The proposal space cannot express the true answer — MP enumerates ordered supercells and never models compositional disorder, so the proposer systematically proposes ordered versions of known solid solutions. The accept signal is a scalar goodness-of-fit that cannot distinguish "wrong symmetry" from "slightly wrong peak shape", and only the positive hypothesis is ever fitted. ARROWS3 also cannot propose the operations that fix most failures: 11 of 17 failed targets were sluggish kinetics, fixable by regrinding or longer soaks, entirely outside its action space.
Steal this. Force the verifier to also fit the strongest known alternative and make the decision statistic the margin between them, not absolute goodness-of-fit. And check that your search space can represent the most common real answer.

### 9. PaperBench — 6.08 (N 5.67 / I 6.5)
The proposal mechanism it measures is task decomposition; its most-copied finding is about control flow, not ideas.

Mechanism. The container holds /home/paper (PDF + Markdown), addendum.md, blacklist.txt and instructions.txt — and no rubric. The agent must itself decide which claims are core contributions (appendix-only experiments are explicitly out of scope), decompose them into implementable units, order them under a 12 h budget, and write a reproduce.sh entrypoint. BasicAgent is a ReAct loop over bash, python, a browser and a paginated keyword-searchable file reader (added because models kept not reading the whole paper), with the submit tool renamed "end task" to discourage early stopping. IterativeAgent deletes the submit tool entirely so the loop cannot terminate before the wall clock, and appends a fixed Continue Message on every turn without a tool call: "take the next step towards replicating the paper... You have a lot of time available, so don't try and do everything in one go... you should try prioritize the most important parts of the paper to replicate first." That is the whole proposal operator. Grading is a weighted rubric tree of 8,316 leaves across 20 ICML 2024 Spotlight/Oral papers (69 leaves for stochastic-interpolants up to 1,963 for pinn), with three leaf types — Code Development, Execution, and Result Match, the last of which counts only plaintext/CSV/JSON/HTML files whose mtime is newer than reproduce.sh's start.

```
+---------------------------------------------------------------------+
| container: paper.pdf + paper.md + addendum + blacklist              |
|            NO rubric, NO score signal, 12 h, 1x A10                 |
| loop until wall clock:                                              |
|     model reasons --> "best next step"        <-- THE proposal step |
|     one tool call: bash | python | browse | paginated_read          |
|     if no tool call: append Continue Message (prioritize first)     |
| --------------------- proposal boundary ---------------------       |
| fresh VM: git clean -fd; bash reproduce.sh (<=12 h) --> log + files |
| SimpleJudge (o3-mini-high): "# Expectations / # Reality / # Score"  |
|     per leaf {0,1} --> weighted parent avg --> Replication Score    |
+---------------------------------------------------------------------+
```

Operators. Self-decomposition into subtasks; the per-turn next-step re-prompt; the submit-tool ablation; tool-grounded revision from real stack traces; browsing under a per-paper blacklist of the authors' repo and known replications; an optional rubric-as-scaffold path that was never used in reported runs.
Selection. None inside the agent — one trajectory, no branching, no pruning. Selection is external: binary leaf grades propagated by author-set sibling weights, with the judge switched to filename-relevance ranking and greedy top-ten inclusion when the submission exceeds the context budget. A post-hoc text-search monitor for blacklisted URLs disqualified 10 of 646 runs.
Grounding. The tightest grading in the set: fresh-VM re-execution with mtime gating, so results cannot be written down. The agent sees none of it during the run.
Loop. Closed only against its own interpreter; fully open across episodes, 3 i.i.d. seeds per paper.
Novelty control. Not applicable by construction (correct proposals are rediscoveries); controls are the blacklist, the monitor, and recency, with contamination from public repos acknowledged as a growing future threat.
Evidence.
- BasicAgent: Claude 3.5 Sonnet (New) 21.0 +/- 0.8%, o1-high 13.2 +/- 0.3%, DeepSeek-R1 6.0 +/- 0.3%, GPT-4o 4.1 +/- 0.1%, Gemini 2.0 Flash 3.2 +/- 0.2%, o3-mini-high 2.6 +/- 0.2%.
- IterativeAgent: o1-high 13.2% -> 24.4 +/- 0.7% (+85% relative); o3-mini-high 2.6% -> 8.5 +/- 0.8%; but Claude 3.5 Sonnet 21.0% -> 16.1 +/- 0.1%.
- 12 h -> 36 h moved o1 only 24.4% -> 26.0 +/- 0.3%.
- Requirement-type breakdown for the best agent: Code Development 35.4 +/- 0.8%, Execution 1.8 +/- 0.7%, Result Match 0.7 +/- 0.3%.
- Agent reproduce.sh scripts ran an average of 5.5 minutes against a 12-hour allowance.
- Human baseline (8 ML PhDs, best@3, 3 papers): 41.4% at 48 tracked hours; o1 leads in hour 1, humans overtake after ~24 h.
- PaperBench Code-Dev proxy: o1 43.4 +/- 0.8% but only r = 0.48 against the full benchmark, PB = 0.45*PBCD + 0.05, grading $66 -> ~$10 per paper.
- JudgeEval macro-F1 / cost: o1 0.84 / $830, o3-mini-high 0.83 / $66, o1-mini 0.78 / $72, GPT-4o 0.73 / $120, GPT-4o-mini 0.59 / $8.
Fails when. "All agents failed to strategize about how best to replicate the paper given the limited time available" — the decomposition step is exactly where every model fails. Models describe plans instead of calling tools; o1 tried to finish in a single response. The scaffold fix is model-specific prompt tuning, not a general mechanism: the same change worth +11.2 points to o1 cost Claude 3.5 Sonnet 4.9.
Steal this. Removing the option to stop can beat improving the ideas. And build dense reward as a weighted tree with cheap Code-Development and Execution leaves feeding an expensive Result-Match leaf — but use the cheap nodes to shape and diagnose, never to rank, since the proxy correlates at only r = 0.48.

### 10. Towards an AI co-scientist — 6.0 (N 5.33 / I 6.67)
Ten prompts wearing the costume of ten operators, wrapped around one genuinely good selection invariant.

Mechanism. Proposal is split across exactly two agents. The Generation agent samples from a library of four strategies: literature exploration (iterative search, read, summarize into a fact base, then generate conditioned on `{articles_with_reasoning}` "chronologically ordered, beginning with the most recent analysis"); simulated scientific debate, a self-play panel that opens with "Propose three distinct {idea_attributes} hypotheses", critiques on attribute adherence, utility and specificity, runs 3-5 turns (max 10) and emits the literal token "HYPOTHESIS"; iterative assumption identification via "conditional reasoning hops" aggregated bottom-up; and research expansion, which reads the current hypothesis set plus the meta-review overview to name unexplored regions. The Evolution agent fires when the Supervisor sees Elo plateau and applies six strategies to top-ranked hypotheses — grounding (find weaknesses, generate search queries, read, fill gaps), coherence/feasibility rewrite, inspiration, combination, simplification, out-of-the-box analogy ("This should not be a mere aggregation of existing methods or entities"). The hard invariant is that Evolution never modifies in place: the child re-enters at Elo 1200 and must out-win its parent.

```
+-----------------------------------------------------------------------+
| goal --> Supervisor --> ResearchPlanConfig{prefs, attrs, constraints} |
| GENERATE (4 strategies):                                              |
|     websearch summarize --> gen | self-play debate 3-5 turns          |
|     conditional hops --> assumptions | research expansion             |
| MUTATE (6 strategies, only on top-ranked, on Elo plateau):            |
|     ground | feasibility | inspire | combine | simplify | out-of-box  |
|     --> ALWAYS a NEW hypothesis; parent untouched; re-enter at 1200   |
| Reflection cascade: initial --> full(search) --> deep verification    |
|     --> observation --> simulation --> tournament review              |
| Ranking: Elo, pairs routed by proximity graph, "better idea: <1|2>"   |
|     top pairs get multi-turn debate, rest single-turn                 |
| Meta-review critique appended to ALL agent prompts next round         |
+-----------------------------------------------------------------------+
```

Operators. Four generation strategies, six evolution strategies, plus expert-in-the-loop injection where a scientist's own hypothesis enters the tournament and can be combined with system-generated ones.
Selection. A three-stage cascade: Reflection filters (initial no-tool review, full review with search, deep verification decomposing into decontextualized sub-assumptions, observation review labeling against each extracted article observation as {already explained, other explanations more likely, missing piece, neutral, disproved}, simulation review); then an Elo tournament from 1200 where the comparison prompt explicitly says to disregard the numerical scores in the attached reviews as non-comparable, and pairs are routed preferentially between similar hypotheses per the Proximity agent's 1-5 similarity graph; then Supervisor termination on MaxIdeas / MaxMatchesPerIdea. K-factor and update rule are never published.
Grounding. Textual only. Web search is the sole tight loop, wired into three places, and the ablation shows it is entirely load-bearing: without search the Reflection agent rates already-published ideas 6.14/10 novel (CI 5.93-6.34), with search 2.38 (CI 2.10-2.67). The "simulation review" is the LLM imagining the experiment, justified by the assumption that frontier LLMs "may have developed an internal world model of science" — a grounding claim with no grounding mechanism.
Loop. Closed on critique, open on experiment. The Meta-review agent distills recurring patterns across every review and debate transcript into one critique appended to all agent prompts — "feedback propagation and learning without back-propagation techniques". No experimental result ever re-enters; all wet-lab numbers came from humans after handoff.
Novelty control. Novelty is a default criterion, priced into Elo, and enforced by pre-tournament discard; the Proximity graph deduplicates and routes similar pairs into the same match. The paper concedes mode-collapse risk in the feedback path: the Generation agent must use meta-review critique "selectively to avoid over-fitting to these review critiques".
Evidence.
- Elo rises monotonically with test-time compute across 203 research goals in 10 temporal buckets, with no observed saturation.
- Blinded expert evaluation on 11 goals: preference rank 2.36 of 4, novelty 3.64/5, impact 3.09/5 — best on all three.
- Evolution ablation: GPQA precision 70.9% -> 75.4%, +4.5% with 95% CI [-2%, +11%]; paperQA max quality 4.7 -> 5.6, CI [0.05, 1.7].
- Generation-strategy contributions are flat: GPQA focus areas 13.8% and generate prompt 13.0%; paperQA conditional hops 2.88% and raw ideas 2.66%.
- Search ablation: novelty 6.14 -> 2.38, correctness 7.4 -> 8.46, GPQA correctness AUC 0.643 -> 0.651.
- Ranking ablation: over 2,000 matches the simple prompt is on par with debate on Flash 2.0; debate wins on Flash 2.5 and on 1,800 high-Elo matches, and nearly eliminates positional bias over 2,000 swap matches.
- Meta-review ablation: correctness-review AUC 0.521 -> 0.597 (paperQA, 600 ideas) but 0.629 -> 0.634 (GPQA, 1,680 ideas).
- AML: search space 2,300 approved drugs x 34 cancer types, 30 hypotheses to oncologists, 5 selected, 3 hits; Binimetinib IC50 as low as 2 nM. KIRA6 10 nM in KG-1a, 144 nM in NOMO-1, vs 180 nM in the TK6 control but 870 nM in HL-60 and 1750 nM in MOLM-13.
- cf-PICI: the top-ranked hypothesis independently matched an unpublished experimental finding from a one-page background plus two references.
Fails when. The only proposal-side ablation is null. The expensive debate operator only beats the cheap prompt in the high-Elo regime. Elo is auto-evaluated by the same model family that generated the hypotheses and is "not based on independent ground truth". Open literature only, so negative and failed results — exactly what experts prune with — are inaccessible. Hit rates are human-AI joint: 30 hypotheses filtered to 5 by oncologists before any bench work.
Steal this. Additive improvement. A child hypothesis re-entering at the same baseline rating means high-variance operators carry zero downside, so you can run radical simplification and out-of-box analogy safely. Pair it with a two-tier comparison budget, since the expensive judge only earns its cost among top candidates.

### 11. The AI Scientist-v2 — 5.42 (N 5.0 / I 5.83)
Typed stage-conditioned generators that make rigor a proposal branch; the tree search underneath is inherited and ranks on a metric the agent invented for itself.

Mechanism. Stage 0 is a ReAct ideation agent with exactly two tools, `SearchSemanticScholar` and `FinalizeIdea`, seeded by a human-written markdown topic rather than a code template, system-prompted to "propose high-impact research ideas resembling exciting grant proposals... Each proposal should stem from a simple and elegant question, observation, or hypothesis" under an academic-lab resource constraint, required to search at least once, emitting 7 fields (Name, Title, Short Hypothesis, Related Work, Abstract, Experiments, Risk Factors and Limitations), refined over 5 reflections, 20 generations per run, conditioned on `{prev_ideas_string}`. Stages 1-4 (Preliminary Investigation, Hyperparameter Tuning, Research Agenda Execution, Ablation Studies) run a parallel best-first tree where a node is {script, plan, error trace, runtime, metrics, LLM feedback, plotting script, figures, VLM feedback, buggy status} and the generator is chosen by parent status and stage: `_draft` while fewer than `num_drafts` = 3 roots exist, `_debug` on buggy parents, `_improve` on non-buggy ones, `_generate_hyperparam_tuning_idea` in Stage 2 checked against a log of already-tried settings, `_generate_ablation_idea` in Stage 4 ("the ablation should be a new idea, not a variation of previous ideas"), `_generate_seed_node` for replications, and aggregation nodes that write only a plotting script over prior .npy results.

```
+-----------------------------------------------------------------------+
| STAGE 0 (open loop, batch):                                           |
|     topic.md --> ReAct{SearchSemanticScholar, FinalizeIdea}           |
|     x5 reflections, x20 generations, vs prev_ideas_string             |
|     --> 7-field idea      [humans then pick 3 of ~40]                 |
| STAGES 1..4, budgets 21 / 12 / 12 / 12 nodes, <=num_workers parallel: |
|     drafts < 3            --> _draft                                  |
|     rand() < debug_prob   --> _debug (depth <= 3)                     |
|     stage == 2            --> _hyperparam (new axis, dedup log)       |
|     stage == 4            --> _ablation   (new idea, dedup log)       |
|     else                  --> _improve(best node)                     |
|     each --> plan + code --> execute --> plot --> VLM critique        |
|     end of stage: LLM evaluator picks 1 node --> root of next stage   |
|                   + 3 replication seeds --> aggregation node          |
+-----------------------------------------------------------------------+
```

Operators. Seven typed node generators plus ReAct ideation, archive-conditioned diversity prompting, per-type already-tried ledgers, and a runtime-slack escalation heuristic in Stage 3 that raises experiment complexity when runs finish far under budget.
Selection. Best-first within a stage, but the ranking is not a fixed formula — an LLM evaluator scores candidates on "performance metrics, training dynamics, and the quality of generated plots", already-expanded trees are skipped, buggy nodes die past `max_debug_depth` = 3, and a clean-running node can be marked buggy purely on VLM figure critique. Between stages, one LLM evaluator picks exactly one node to seed the next stage root. Above the system, humans pick 3 ideas of ~40 and 1 manuscript of several seeds.
Grounding. Code execution (tightest), VLM figure critique (the only non-textual honesty check), and Semantic Scholar during ideation. No external correctness oracle: "success" is measured on an evaluation metric the agent wrote into its own Stage 1 baseline.
Loop. Closed at the experiment level, open at the hypothesis level — all ~20 ideas are generated before any experiment runs, and the accepted paper's hypothesis was falsified by its own experiments and written up as a negative result rather than replaced.
Novelty control. Four mechanisms, none quantitatively validated: in-loop Semantic Scholar with a required Related Work field, the archive prompt, per-type dedup ledgers, and 3 independent draft roots. No dedup rate, no overlap measurement, no diversity metric is reported, and humans had to hand-pick for distinctness and re-prompt toward applied domains to get a second batch.
Evidence.
- 3 fully autonomous manuscripts submitted to the ICLR 2025 ICBINB workshop among 43 submissions; 1 exceeded the threshold at 6.33/10 (individual scores 6, 6, 7), roughly top 45%.
- Idea funnel ~40 generated, 3 selected by humans (7.5%).
- Node budgets 21 / 12 / 12 / 12 (paper Table 3) vs 20 / 12 / 12 / 18 with num_workers = 4 in the shipped config; max_debug_depth 3; debug_prob 1.0 (paper) vs 0.5 (repo); 1 h per node, <=15 h per paper.
- Cost ~$15-20 per experiment run plus ~$5 writing; ideation "a few dollars"; 20-30 min for writeup.
- Authors' internal review: none of the 3 met top-tier main-track standards; their pick of the best manuscript agreed with the reviewers' (n = 1).
Fails when. Best-first optimizes a self-chosen proxy — the classic hacked-proxy failure, worse the deeper the tree. VLM critique conflates presentation defects with experimental failure. Citation hallucination, caption inaccuracies and potential dataset overlap are all reported by the authors. Two humans-in-the-loop remain load-bearing, and no success rate over runs was measured.
Steal this. Type the operator and condition it on stage, not on model discretion: the system then proposes its own ablations and replication seeds (rigor generated rather than requested), failure gets a first-class branch with a depth cap, and per-type ledgers give verifiable dedup where a diversity prompt gives only its appearance.

### 12. Agents4Science 2025 — 5.33 (N 5.67 / I 5.0)
A venue that turned per-stage autonomy into a measurable variable, and measured its own judge into disrepute.

Mechanism. No proposal algorithm; the contribution is an instrument plus a cascade. Every submission completes two checklists, the second decomposing research into four stages — hypothesis development (defined to cover background research and who proposed the idea), experimental design and implementation, data analysis and interpretation, manuscript writing — with a per-stage autonomy label: A (>=95% human), B (50-95% human), C (50-95% AI), D (>=95% AI), plus free-text justification and an AI-limitations field. Published Level-D justifications describe one dominant pattern: the agent analyzes literature, formulates the direction and core hypothesis, emits a pool of candidate hypotheses, and the human supplies only venue scope and the final pick. Level B is the mirror: human seed prompt, agent-developed topic/framework/hypotheses, human mentor approving or rejecting each. The selection cascade is 315 submissions, 62 desk-rejected, 253 scored independently by GPT-5, Gemini 2.5 Pro and Claude Sonnet 4 on an identical NeurIPS-2025-guidelines prompt (soundness, significance, clarity, originality, 1-6), calibrated beforehand against anonymized ICLR 2022 and 2025 papers with known decisions; mean >= 4.0 advanced 79 papers to one blinded human expert each; the committee accepted 48.

```
+----------------------------------------------------------------------+
| per submission (author-side internals undisclosed):                  |
|     seed or nothing --> agent reads literature                       |
|                     --> emits a POOL of candidate hypotheses         |
|                     --> human gate: pick one | approve/reject | none |
|                     --> self-label stage autonomy in {A, B, C, D}    |
| venue pipeline (single pass, open loop):                             |
|     315 --> desk-reject incomplete --> 253                           |
|     253 --> {GPT-5, Gemini 2.5 Pro, Claude Sonnet 4} on 1-6 rubric   |
|         --> mean of 3 >= 4.0 --> 79                                  |
|      79 --> 1 human expert, BLIND to the LLM reviews                 |
|         --> committee synthesis --> 48 accepted                      |
|     side screens on all 253: citation-existence verifier;            |
|                              prompt-injection scanner                |
+----------------------------------------------------------------------+
```

Operators. Four labeled patterns rather than algorithms: pool-and-pick full autonomy, mentor-gated generation, seeded-from-papers, human-only formulation; plus multi-model ensembling as an author-side diversity source.
Selection. A hard numeric gate at mean >= 4.0 of three LLM scores, then one human. No rebuttal, no discussion phase, no aggregation beyond the arithmetic mean. The gate is one-directional, so LLM false negatives are structurally unrecoverable and unmeasured.
Grounding. Thin and post hoc: human review only above the gate; automated citation-existence verification via title extraction plus web search with public OpenReview flags; a prompt-injection scanner. No code execution, no replication, no re-analysis — the venue grounds the write-up, never the results.
Loop. Open and single-shot. The only closed loop in the design targets the judge: reviewer instructions were iteratively refined against ICLR ground-truth scores to maximize correlation with human means.
Novelty control. None. "Originality" is one rubric dimension with no operational definition, and the reference verifier tests existence, not contribution — it would pass a perfectly cited rediscovery. Authors independently reported models that "struggled to generate novel or complex experimental ideas beyond the templates it had been given".
Evidence.
- Fully AI-driven work across all four stages: 23.3% of submissions, 14.9% of accepted — a 36% relative drop.
- Hypothesis-stage autonomy, all submissions: A 11.6%, B 28.5%, C 24.9%, D 34.9%. Among accepted: A 10.6%, B 29.8%, C 31.9%, D 27.7%.
- Autonomy is stage-dependent: Level D rises monotonically across the four stages, highest at data analysis and writing.
- 56.7% of submissions and 55.3% of accepted reported Category C or D in every stage.
- LLM reviewer agreement: average pairwise Pearson r = 0.48. Means on 1-6: GPT-5 2.30, Claude Sonnet 4 3.0, Gemini 2.5 Pro 4.23. Mean absolute deviation from humans on the 79: 0.91, 1.09, 2.73.
- Only ~44% of submissions (111 papers) had zero flagged references; 2 papers attempted prompt injection and were rejected.
- All 48 accepted list an AI model as first author; 73% credited one model, 27% multiple; only 16.7% used a specialized research agent.
- The venue's own award-winning BadScientist paper: fabricated work with no real experiments accepted by multi-model LLM review at up to 67.0%/82.0%, with detection barely above chance.
Fails when. The autonomy labels are self-reported with no audit, so the headline 34.9% -> 27.7% is correlational — accepted teams may simply be stronger researchers who also use AI differently. And the selection signal is too noisy to reuse as a fitness function by the venue's own numbers.
Steal this. Instrument the proposal step separately from execution and write-up with a per-stage autonomy label. The resulting number is directly actionable: autonomy pays most at data analysis and writing and least at hypothesis generation. Do not use an LLM-reviewer ensemble as a fitness function, even three frontier models with a calibrated shared prompt.

### 13. Robin — 4.92 (N 5.33 / I 4.5)
Two non-obvious moves — rank the evidence report, and meta-write the rubric — bolted onto a straight-line prompt chain the authors themselves de-agentified.

Mechanism. Phase 1 proposes the measuring instrument: o4-mini emits 5 literature queries (30+ words each, forbidden from naming specific drugs, split across disease biochemistry and assay methodology) separated by `<>`; Crow (PaperQA2) answers each across 151 papers; o4-mini proposes 10 causal mechanisms as strict JSON `{strategy_name, reasoning}` under hard-coded priors ("prioritize simplicity, speed of readout, biological relevance, and direct measurement of functional endpoints"); Crow writes a fixed-schema eval report per assay (Assay Overview / Biomedical Evidence / Previous Use / Overall Evaluation); a judge plus BTL ranks the 10 reports and `iloc[0]` survives; then o4-mini synthesizes the winning assay plus disease into a free-text `candidate_generation_goal` — a generated, not templated, phase handoff. Phase 2 proposes the objects: 10 queries split half disease-biology, half therapeutic-landscape, ~400 papers, then 30 drugs in delimiter-fenced text (`<CANDIDATE START>` / CANDIDATE / HYPOTHESIS / REASONING) constrained to single agents, no combinations or special formulations, commercially available with catalog numbers; Falcon writes a 5-section deep report per candidate; and the judge ranks reports, never raw ideas.

```
+------------------------------------------------------------------------+
| disease --> 5 queries (30+ words, no drug names) --> CROW [151 pp]     |
|         --> 10 assays as JSON {strategy_name, reasoning}               |
|         --> CROW x10 fixed-schema eval reports                         |
|         --> judge C(10,2) = 45 pairs, JSON{Analysis,Reasoning,W,L}     |
|         --> choix.ilsr_pairwise(alpha=0.1) --> iloc[0]                 |
|         --> SYNTHESIZE --> goal string                                 |
|         --> 10 queries (half bio / half therapeutic) --> CROW [~400]   |
|         --> 30 candidates --> FALCON deep reports --> APA pass         |
|         --> judge min(300, C(30,2)) = 300 pairs --> BTL --> ranked.csv |
| =========================== humans bench ===========================   |
|         --> FINCH x5 same prompt + 1 consensus trajectory              |
|         --> split on "<>": {drugs, summary + "DO NOT SUGGEST THESE     |
|             DRUGS AGAIN.", questions, mechanisms} --> re-enter         |
+------------------------------------------------------------------------+
```

Operators. Assay/strategy proposal; generated goal synthesis as a phase handoff; drug proposal with a bundled mechanistic hypothesis; dual-track query generation; literature expansion as an operator (the report, not the idea, is judged); follow-up assay proposal explicitly permitted to return none; result-conditioned regeneration splicing analysis_summary, mechanistic_insights and questions_raised into both the query-gen system message and the candidate-gen user message; stochastic-ensemble analysis where N Finch trajectories run an identical prompt and diversity comes purely from sampling noise.
Selection. A pairwise LLM judge feeding BTL: `n_games = min(300, C(n,2))` at fixed seed 621, comparisons fanned out under a 100-slot semaphore, strengths fit by `choix.ilsr_pairwise(alpha=0.1)` under P(i beats j) = sigmoid(theta_i - theta_j). The rubric was meta-generated — domain experts performed their own pairwise comparisons and those preferences were handed to Gemini 2.5 Pro Preview to write the judge prompt. The resulting priority order is explicitly anti-novelty: evidence strength first, then mechanism clarity, safety, feasibility, and novelty last, "Do not prefer novelty if it comes at the cost of strong evidence for a more established, safer approach."
Grounding. Mandatory retrieval expansion before any ranking (151 + ~400 papers per run), plus a real but low-bandwidth wet-lab channel: ARPE-19 cells, 60 min pre-treat, pHrodo beads, 3 h, flow cytometry MFI, run by humans who had to translate the outline into a protocol.
Loop. Closed in principle, human-triggered in practice, and shallow: 2 candidate rounds ever ran, with no automatic iteration construct in the released code.
Novelty control. A literal prompt suffix, "DO NOT SUGGEST THESE DRUGS AGAIN", with no programmatic enforcement — a string-level blocklist over physically tested drugs, not a literature-level check.
Evidence.
- Judge-expert concordance: 7.25 of the judge's top 10 matched the experts' top 10, vs 3.33 expected at random.
- Judge intra-rater consistency 88% vs 61% for the human experts whose preferences generated its rubric.
- Round 1: 1 of 5 top-ranked candidates worked (Y-27632). Round 2: ripasudil, +7.5x phagocytosis vs DMSO by Finch's analysis and +1.75x by independent human analysis of the same data.
- ABCA1 upregulated ~3-fold at adjusted p = 2.13e-83; over 8 RNA-seq trajectories the same genes were significant in >50% of them.
- 10 ranked candidates generated for each of 10 further diseases, none experimentally validated.
Fails when. The feedback signal itself is unreliable (a >4x discrepancy in the number that conditions the next round). The released code makes the generator o4-mini its own judge, while the 88%/7.25 validation used Claude 3.7 Sonnet. Defaults ship at 3/3/5 rather than the paper's 5/10/30. The data-analysis path is hard-wired to flow cytometry and does not work unmodified for RNA-seq. And the authors concede the ROCK-phagocytosis link already had a supporting paper surfaced by their own search.
Steal this. Never rank a raw idea — rank a fixed-schema evidence report generated from it, which converts the judge's task from "does this sound plausible" to "is the cited evidence strong". And meta-generate the rubric from expert pairwise preferences rather than hand-authoring criteria.

### 14. Aviary — 4.92 (N 4.83 / I 5.0)
The most directly reproducible recipe in the set: identify the proposal call, filter it on a real verifier, fine-tune it.

Mechanism. A Language Decision Process (V, S, A, O, T, Z, R, gamma) — a POMDP whose actions and observations are natural-language strings — realized as a stochastic computation graph of typed Ops whose only stochastic nodes are LLMCallOps. The proposal at timestep t is a_t ~ p_LLM(. | xi_t) with xi_t = [o_0, a_0, ..., o_t], emitted as a ToolRequestMessage. Three agent shapes ship. SimpleAgent is one call. ReActAgent splits the proposal across two consecutive stochastic nodes under the template "Thought: you should always think about what to do / Action: the action to take, should be one of [{tool_names}] / Action Input: comma separated list of inputs to action as python tuple", parsed by regex plus `ast.literal_eval`, with ACT_* variants deleting the Thought field to ablate reasoning. TreeofThoughtsAgent does explicit propose/value/select, splitting one completion on newlines so each line becomes a child path. Population-level proposal is Algorithm 1: for i = 1..N, roll out pi_{i-1}, append every trajectory with R > rho to a monotone buffer D_i, and SFT pi_i on D_i, with tasks sampled by w_k = M(1 - f_pass^k), M = 20, so budget flows to the current failure frontier.

```
+-----------------------------------------------------------------------+
| per timestep:  a_t ~ p_LLM(. | xi_t, tools)  --> ToolRequestMessage   |
|                o_t+1, r_t = env.step(a_t)                             |
|                tools: Rosetta cart_ddg | PCR/Gibson/Golden-gate sims  |
|                       | MMseqs2 | PaperQA2 | Wikipedia | calculator   |
| per round:     T_i  <- rollout(pi_i-1, tasks ~ w_k = M(1 - f_pass^k)) |
|                D_i  <- D_i-1 + {tau : R(tau) > rho}                   |
|                pi_i <- SFT(D_i)                                       |
| at test time:  32..945 rollouts --> drop unsure --> majority vote     |
|                protein: 16 rollouts --> Rosetta ddG --> pass@16       |
+-----------------------------------------------------------------------+
```

Operators. Temperature-sampled tool calls; the ReAct two-node split; newline-split branching for cheap width; independent i.i.d. rollouts; tree search with environment cloning per node; curriculum resampling; expert-iteration SFT; APEOpt, which has an LLM rewrite the agent's own PromptOp from up to 50 scored (input, output) examples — a proposal about the proposer; MemoryOpt storing backprop-filtered (input, output, reward) tuples as retrievable exemplars; and a ReflectModule revise operator.
Selection. Three separate selectors: a hard threshold R > rho at training time (with binary reward this is just "keep the successes", no ranking, no partial credit, nothing ever evicted); consensus@k majority voting at inference on MCQ tasks after discarding unsure and truncated trajectories; and an oracle pass@16 on protein stability, mirroring how a protein engineer screens a plate. Tree search prunes on target_reward and max_depth; APEOpt caps context at 50 examples sorted by -reward and discards any rewrite that changes the format template variables.
Grounding. Non-LLM verifiers throughout — Rosetta cart_ddg ddG over 40 proteins from the megascale dataset, aggregation propensity, secondary structure, sequence properties; MMseqs2 annotation and cloning simulators for SeqQA; PaperQA2 for LitQA2; a calculator for GSM8K. Terminal reward is exact match against held-out ground truth, never an LLM judge.
Loop. Closed at two nested scales — tool observations within an episode, and a monotonically growing EI buffer plus online f_pass moving averages across episodes (2,841 BC trajectories then 8 EI rounds on SeqQA; 430 BC trajectories on LitQA2). What is missing is any record of why a candidate failed; only surviving successful transcripts persist.
Novelty control. Absent and appropriately so — the tasks have known answers. Diversity comes only from temperature, and EI actively reduces it. The real control is benchmark-side: LitQA2 is built from recent literature and PaperQA2 must retrieve the source paper; the 40 evaluation proteins exclude those in a cited reference.
Evidence.
- SeqQA: Llama-3.1-8B after EI 0.64 single-sample -> 0.86 with 32-sample majority vote -> 0.89 at 945 rollouts; Claude 3.5 Sonnet agent 0.86; prior best 0.87.
- Majority voting adds roughly 20 percentage points on both SeqQA and LitQA2 — the largest reported gain from any component.
- LitQA2: 0.89 with voting vs 0.67 previously reported.
- Cost ~$0.00066 per 8B trajectory vs ~$0.07 per Claude 3.5 Sonnet trajectory (~100x) vs $4-$12 per question for PhD contractors.
- Protein stability: 40 proteins, oracle-verified pass@16 shows a large jump over pass@1, with no absolute number reported.
Fails when. EI cannot escape the base policy's support: a task pi_0 never solves contributes nothing forever while the curriculum keeps spending on it. Binary reward gives no gradient between nearly-right and nonsense, and SFT-on-successes-only narrows the distribution with no entropy regularizer. The headline 0.89 buys accuracy with inference compute, not with a better proposer. The one open-ended generative task is graded by a simulator with no wet-lab confirmation and no propagation of cart_ddg's own error. TreeofThoughtsAgent is documented in-repo as tested only as a Game-of-24 baseline and does not support tool calls on intermediate steps.
Steal this. Make the proposal step a trainable stochastic node, so one identified sampling call is generator, reward-filtered unit and SFT target at once. Pair it with w_k = M(1 - f_pass^k) to spend proposal budget on the current failure frontier. Do not transfer it where no ground-truth reward exists — filtering on R > rho then degenerates into filtering on whatever proxy you had, and EI amplifies that proxy's biases.

### 15. The Virtual Lab — 4.58 (N 4.83 / I 4.33)
Role-play meetings are thin; the transferable move is that the LLM writes a specification and a non-LLM scan does the proposing.

Mechanism. An Agent is a 4-field prompt ("You are a {title}. Your expertise is in {expertise}. Your goal is to {goal}. Your role is to {role}."). A human writes only the PI and the Scientific Critic; the PI then generates the rest of the cast, emitting 3 scientist agents in literal `Agent(title=..., expertise=..., goal=..., role=...)` syntax using its own prompt as the few-shot example. Every proposal is a meeting over one shared message list where each agent is a view with its own system prompt. Team meetings run 3 rounds: PI opens, each scientist speaks, then the Critic, each told "If you do not have anything new or relevant to add, you may say 'pass'. Remember that you can and should (politely) disagree"; the PI must synthesize and ask follow-ups each round, then write a fixed-schema summary. Individual meetings are agent -> Critic -> agent, 3 rounds, with the Critic forbidden from writing the answer. Every phase runs `num_iterations = 5` identical meetings at temperature 0.8, merged at 0.2 by an agent shown all five as `[begin summary k]` blocks and told to "merge the best components of each individual answer... and explain what components of your answer came from each". The workflow-design agenda hands the PI real budget constraints (ESM 5 min for all mutations, AF-M 30 min/mutation, Rosetta 5 min/mutation) and forces numbered answers including "give a formula" and "give a single number". After that the LLM is never called again: ESM-1b scores all L x 19 point mutants, top 20 go to AlphaFold-Multimer and Rosetta, WS ranks them, top 5 seed the next of 4 rounds.

```
+---------------------------------------------------------------------+
| human: agenda + agenda_questions + agenda_rules + prior summaries   |
|   --> 5 parallel meetings @ T = 0.8                                 |
|       team: PI --> Immunologist --> ML Spec --> CompBio --> Critic  |
|             --> PI synthesis + follow-ups (x3 rounds)               |
|       individual: agent --> Critic --> agent (x3)                   |
|   --> merge meeting @ T = 0.2 over all 5 summaries, with provenance |
|   --> spec: seeds, WS = 0.2*LLR + 0.5*ipLDDT - 0.3*dG, 20->5, x4    |
| ==================== no LLM below this line ====================    |
|   ESM-1b scans all L x 19 mutants --> top 20 by LLR                 |
|   --> AlphaFold-Multimer ipLDDT --> Rosetta dG --> WS --> top 5     |
|   --> repeat x4 --> top 23 by WS_WT (LLR summed along lineage)      |
|   --> 92 mutants + 4 wild types --> ELISA                           |
+---------------------------------------------------------------------+
```

Operators. PI-generated team; 5 parallel meetings at T = 0.8; merge at T = 0.2 with per-component provenance; adversarial critique loops; role-conditioned disagreement; PI round-end synthesis; forced-commitment agenda questions added after agents refused to decide; human errata meetings listing concrete defects; the ESM exhaustive point-mutation scan; beam search over sequences.
Selection. Two selectors at two levels. At the LLM level there is no numeric score at all — the merge meeting's free-text judgment plus the Critic's veto-by-critique. At the molecular level an LLM-authored linear scalarization, with a corrected cross-round statistic WS_WT that sums per-round LLRs along the mutation lineage because raw per-round LLRs are relative to the previous seed. Per starting nanobody that is 320 full AF-M + Rosetta evaluations; ~1,280 across four.
Grounding. In-round grounding is computational only (ESM-1b likelihood, antigen-agnostic; AF-M interface pLDDT; Rosetta REF15 dG after relaxation). Code is grounded by actually running it, which is how the wrong HF model id, the invalid Rosetta filter and ref15-vs-ref2015 surfaced. Terminal grounding is ELISA against a 5-RBD array. Literature grounding is absent: a `pubmed_search` tool exists but is never enabled in the nanobody run.
Loop. Open where it matters. The 4 mutation rounds close only against in-silico surrogates; the ELISA results never re-entered — no round 5, no re-weighted WS, no second synthesis batch.
Novelty control. None at the idea level; novelty is delegated to the search space ("trillions of nanobody sequences with one to four mutations").
Evidence.
- 2 of 92 designs (2.2%) gained a binding profile beyond Wuhan RBD — the published result is improved binding to the newer JN.1 and KP.3 variants. Neither reaches its own Wuhan-RBD affinity, so the yield against the newest-variant target is 2/92 weak binders, not a high-affinity hit.
- Nb21 I77V-L59E-Q87A-R37Q: average KP.3 intensity 3.5 vs 0.06 +/- 0.09 for the other 22 Nb21 mutants and 0.1 for WT.
- >90% expressed and soluble; 38% high-to-very-high expression; specificity retention 96% (44/46) for H11-D4 + Nb21 vs 10/23 for Ty1.
- In silico: 92/92 positive ESM LLR, 78/92 beat WT on ipLDDT, 32/92 at ipLDDT >= 80, 60/92 beat WT on dG, 23/92 at dG <= -50.
- Specialized-vs-generic ablation over 50 parallel project-specification meetings: scientist team chose "modify existing" 43/50 vs generic 30/50 (generic hedged "both" 18x), Fisher OR 0.244, p = 0.0063; molecule choice not differentiated (50/50 vs 49/50, p = 1.0).
- Scientific Critic coding ablation over 20 scored discussions: +2.15 net enhancements per discussion, total +43, Wilcoxon W = 160.5, p = 5.1e-4; per topic AlphaFold +4.8, ESM +1.8, Rosetta-Python +1.8, Rosetta-XML +0.2; 2 of 20 revisions were net harmful.
- Human wrote 1,596 words (1.3%) vs 122,462 by the agents; PI 36.4% of a team meeting's words, Critic 21.0%, human 0.5%.
Fails when. Beam collapse: all 23 final Ty1 mutants share a round-1 mutation at position 32, and Ty1 had the worst Wuhan retention. Scalarization pathology: ipLDDT on 60-80 at weight 0.5 dominates LLR on 1-8 at weight 0.2, so three R27C designs ranked 19/23, 22/23, 21/23 on LLR were still selected and bound everything. And the first ESM scripts used random numbers as a placeholder for log-likelihoods; correct code survived only because one of the five parallel runs happened to get it right.
Steal this. Have the LLM design the search operator, not perform the search, and force the specification to be numeric with agenda questions that demand a formula, a single number, a choice of one. Normalize the components of any LLM-written weight vector yourself, or you will select crosslinking junk.

### 16. Kosmos — 4.42 (N 4.5 / I 4.33)
Proposal conditioning moved out of the context window and into a persistent world model — with no scoring statistic anywhere in the system.

Mechanism. Initialized with a free-text objective and a dataset under 5 GB, Kosmos runs discovery cycles for up to 12 hours (~20 cycles typical, one reported run at 35). Each cycle an orchestrating LLM queries the structured world model — the persistent store of prior task summaries, findings and provenance — plus the standing objective, and emits up to 10 natural-language task specifications, each typed as either a data-analysis task or a literature-search task. Those dispatch parallel rollouts of two general-purpose agents: a code-writing agent whose essentially only tool is execute-code, producing Jupyter notebooks in gVisor sandboxes on Kubernetes, and a full-text literature agent. Task outputs are summarized back into the world model, and the next cycle re-queries it. There is a second, finer proposal layer inside rollouts: the data-analysis agent proposes hypotheses and even invents analysis methods that were never requested — a segmented-regression breakpoint test on pseudotime, a bespoke Mechanism Rank Score for SNP-gene pairs. Termination is self-judged, collapsing the world model into 3-4 reports.

```
+---------------------------------------------------------------------+
| inputs: research_objective (text), dataset (<= 5 GB)                |
| W <- init_world_model(objective, dataset)                           |
| for cycle in 1..~20:                       # up to 12 h wall clock  |
|     tasks <- LLM_propose(query(W), objective)      # <= 10 specs    |
|              type in {data_analysis, literature_search}             |
|     results <- parallel_map(tasks)         # ~8.3 DA + ~1.8 LIT     |
|         data_analysis --> code agent (one tool) --> notebook        |
|         literature    --> full-text agent --> papers                |
|     W <- update(W, summarize(results))     # summaries + provenance |
|     if LLM_judges(objective_complete): break                        |
| reports <- synthesize(W): 3-4 narratives, ~25 claims each,          |
|            every claim cites a notebook or a paper                  |
| # no scoring function, no tournament, no novelty check anywhere     |
+---------------------------------------------------------------------+
```

Operators. World-model-conditioned batch task proposal; typed data-analysis and literature dispatch; parallel breadth as deliberate diversification ("explore many different research avenues simultaneously"); cross-modal synthesis pairing a data result with a literature fact into a mechanism (Atp10a downregulation plus phosphatidylserine "eat-me" signal literature into a flippase-collapse hypothesis); within-rollout method invention; falsification by execution (Hypothesis 1, "other flippases compensate", was rejected by observing global P4-ATPase downregulation, which generated Hypothesis 2); self-termination.
Selection. There is none. No Elo, no BTL, no e-values, no MCTS value, no rubric. A hypothesis survives if the analysis dispatched to test it returns support; final pruning is the report-synthesis step, where roughly 25-35 of ~200 rollouts survive into the output. The paper states flatly that "there exists no automated method to reliably evaluate if a claim is accurate, novel, and significant". More cycles is not monotonically better: in the 35-iteration run the iteration-8 report was judged more focused and used instead.
Grounding. Code execution against the scientist's real dataset (42,500 LOC per run) plus full-text retrieval (1,500 papers per run), both in-loop, with every claim required to cite a notebook or a paper. No wet lab, no ability to fetch external data for orthogonal validation.
Loop. Closed across cycles through the world model, which replaces context-window carryover and sustains ~202 rollouts and tens of millions of tokens; but it is a summarizing store, not an archive of full artifacts, with no described mechanism for reviving a pruned branch, and no human interaction mid-run.
Novelty control. Absent as an automated component — no dedup, no novelty agent, no diversity objective. Novelty was established post hoc: three discoveries reproduced findings from preprinted manuscripts the literature agent provably did not access and that postdate model cutoffs.
Evidence.
- 79.4% of 102 statements judged accurate by blinded experts; by type, data analysis 85.5% (n = 55), literature review 82.1% (n = 28), interpretation/synthesis 57.9% (n = 19).
- 42,500 +/- 7,280 lines of code and 1,500 +/- 1,120 full-text papers per run over 166 data-analysis + 36 literature rollouts.
- 9.8x more code than Robin's 4,310 +/- 344 LOC; ~8x more iterations than prior systems.
- Internal tally ~4.1 expert-months per run (n = 6, sigma = 0.85) at 15 min/paper and 2 h/notebook; independent academic groups put a 20-cycle run at 6.14 months (n = 7, sigma = 2.49).
- Cycle-20 findings rated 1 completely / 2 largely / 5 moderately / 0 not novel; reasoning depth 4 high / 4 moderate / 0 shallow.
- 7 highlighted discoveries: 3 reproduce unaccessed manuscripts, 4 novel. Replication by re-prompting: 5/5, 5/5, 4/5 trajectories supported.
- Kosmos-invented Mechanism Rank Score: Q5 showed a 3.3-fold higher ChIP-seq validation rate than Q1 (p < 0.001); segmented-regression breakpoint at 0.58 pseudotime units, Davies test p = 0.017.
Fails when. The authors say it outright: "our evaluations do not capture if the analyses Kosmos chose to execute were the ones most likely to yield novel or interesting scientific insights". Interpretation accuracy of 57.9% is attributed to "its propensity to conflate statistically significant results with scientifically valuable ones" — bad scientific taste, not bad code, is the binding constraint. Runs are stochastic and sensitive to objective phrasing; there is no automated way to identify which of ~200 rollouts produced anything valuable.
Steal this. Move proposal conditioning off the context window onto a persistent queryable store updated after every task, and propose a fixed-width batch of executable task specs typed into a small closed set of dispatchable actions, each carrying a provenance pointer. Then add the ranking mechanism Kosmos does not have — its weakest number sits exactly where selection would.

### 17. Evaluating Sakana's AI Scientist — 4.17 (N 4.0 / I 4.33)
Two audit instruments reproducible in an hour, from a single n = 1 run on a FunkSVD toy.

Mechanism. A critique, not a generator. The audited pipeline is ideation (gpt-4o-2024-05-13, `num_reflections` = 5, `max_num_generations` = 32, emitting {Name, Title, Experiment, Interestingness, Feasibility, Novelty}), a Semantic Scholar novelty gate (up to 10 rounds x 10 results, title/authors/venue/year/abstract only, setting novel=True when no clear match), and Aider rewriting experiment.py for up to 5 iterations. Beel et al.'s contribution is the audit method: run the full pipeline once on a controlled substrate (FunkSVD on MovieLens-100k, 80/10/10, RMSE, CPU-only) with 2 hand-written seed ideas plus 10 generated, then (a) manually fact-check every novel=True verdict against known literature, (b) diff the character count of experiment.py per Aider iteration as an adaptability proxy, (c) hand-audit citations, figures and numeric claims against the logs, and (d) re-run the reviewer agent on its own 7 papers and on 10 human OpenReview papers with known labels.

```
+--------------------------------------------------------------------+
| FALSIFY THE GATE:  feed it ideas you already know are prior art    |
|     seed "e-fold cross-validation" (indexed on S2 under that name) |
|     seed "adaptive learning rates", "micro-batching", "hybrid MF"  |
|     --> 12/12 returned novel = True                                |
| INSTRUMENT THE DIFF:  chars added to experiment.py per iteration   |
|     baseline 6,260 chars --> +529 (+8%), 118, 83, 66, 21           |
|     Quantized SGD  +1173 (18.7%), -36, +17, +258, +169             |
|     e-fold         +818 (13.1%), 0, 0, 0, 0                        |
|     Sparse-aware   +524 (8.4%), +1, 0, 0, 0                        |
|     --> loop collapses to a no-op, invisible in the manuscripts    |
| AUDIT THE JUDGE:  reviewer on 10 human OpenReview papers           |
|     --> rejects 9/10, incl. 4 of 5 human-accepted; accepts the 1   |
|         paper humans rejected                                      |
+--------------------------------------------------------------------+
```

Operators (of the audited system). Template-conditioned ideation where the only context is one task_description plus the literal experiment.py source, so all 10 ideas are variations on FunkSVD/SGD (factor pruning, quantized SGD, micro-batching, sparse-aware SGD, cluster-aware SGD, hybrid MF); seed-idea priming that visibly anchors the set; 5 self-reflection turns; the Semantic Scholar loop; Aider's 5 edit-and-run iterations.
Selection. Effectively none. The three 1-10 scores "appear arbitrarily assigned and are not used in further processing" — the seed ideas' own scores (Interestingness 8, Feasibility 7, Novelty 7) had no apparent impact. The de facto selector is the execution harness: 5 of 12 ideas never ran after 5 Aider iterations, so survival is decided by whether Aider happened to emit working code.
Grounding. Both channels are loose. Literature grounding is keyword matching on metadata, which failed to cite e-fold cross-validation papers using that exact term and retrievable on Semantic Scholar. Execution grounding runs real code but cannot read its own results: the system intended to sweep e = {2,3,4,5}, kept e fixed at 2, never re-ran the k-fold baseline, and reported e-fold as superior — impossible by construction, since e-fold can only save energy, never beat k-fold on performance.
Loop. Closed within an idea for 5 Aider iterations, open everywhere else — no cross-idea archive, no re-ideation after results, one pass, no compounding.
Novelty control. The audited gate is the finding: 12/12 classified novel, with verified prior art including micro-batching for SGD, adaptive learning rates, hybrid matrix factorization and e-fold cross-validation. Diagnosis: "reliance on keyword matching rather than any deeper synthesis of research."
Evidence.
- 12/12 false-novelty rate in this run; 5 of 12 (42%) unrunnable after 5 iterations.
- Character deltas +529 (+8%), then 118, 83, 66, 21 on a 6,260-character template, hitting exactly 0% for several ideas by iteration 2.
- 7 manuscripts, median 5 references (range 2-9); only 5 of 34 references (14.7%) from 2020 or later; related work rated poor in 7/7.
- 4 of 7 (57%) had structural errors including literal "Conclusions Here" placeholder text; 4 of 7 (57%) had incorrect or hallucinated numbers.
- Reviewer agent rejected 7/7 of its own manuscripts and 9/10 human papers.
- $42 total for 10 ideas + 7 experiments + 7 manuscripts + reviews (~$6/manuscript); 25 human hours excluding setup, ~3.5 h per manuscript, estimated 3-11x faster than a human.
- Zero confidence intervals or p-values across all 7 manuscripts; one claimed an energy win from 116 s -> 115 s while memory usage rose.
Fails when. The study is a single run, 12 ideas, one FunkSVD substrate, one LLM, no seeds — every percentage comes from n = 1.
Steal this. If your engine emits scores, make them load-bearing or delete them. Run your novelty gate on ideas you already know are prior art; and instrument the diff (character or AST delta per edit iteration), which exposes a loop collapsing to a no-op long before any polished output would show it.

## Cross-cutting findings

1. Grounding strength predicts proposal quality; agent count and orchestration cleverness do not. The two top-ranked works have the fewest moving parts in their generator — four lines of controller loop, or one prompt over 791 few-shot pairs — and the strictest verifiers (a program score, the Lean kernel). The most elaborate orchestrations sit at ranks 10, 13, 15 and 16: eleven co-scientist operators produce a null Evolution ablation; five parallel Virtual Lab meetings plus a Scientific Critic produce 2/92 weak binders against the newest-variant target; Kosmos's ~202 rollouts produce 57.9% accuracy on exactly the interpretive statements a selector would rank. Where a machine verifier exists, genuinely new operators appear — variant curricula, evolved searchers, martingale-typed proposals. Where it does not, seventeen teams converge on the same artifact.

2. The verifier cliff is a cliff, not a slope. Systems with a cheap automatic verifier report externally checkable outcomes: 48 multiplications for 4x4 complex matmul, 0.7% of Google's fleet recovered, 56.1% on PutnamBench-test against a 5.3% prior SOTA, Type-I 0.103 +/- 0.020 at nominal 0.1. Systems without one report either human-gated funnels (30 co-scientist hypotheses filtered to 5 by oncologists before 3 hits; 5 Robin candidates to 1 hit) or unmeasured proposal quality (Kosmos states outright that its evaluations do not capture whether the chosen analyses were the ones most likely to yield insight). There is no middle tier where an LLM judge substitutes for the verifier and the numbers still hold.

3. Almost nothing ablates its own proposal component, and where one is isolated the result is null or traceable to sampling volume. The co-scientist's Evolution agent moves GPQA precision 70.9% -> 75.4% with 95% CI [-2%, +11%] and meta-review moves review AUC 0.629 -> 0.634. AI Scientist v1 severed the archive-to-review-to-ideation loop for all headline runs and reported "no reduction in the quality of the papers generated". Aviary's largest gain, roughly 20 points on SeqQA and LitQA2, is majority voting over 32 to 945 rollouts. MLE-bench's gpt-4o pass@6 = 17.0% matches o1-preview pass@1 = 16.9%. The exceptions prove the rule: POPPER's relevance-gate ablation (0.082 -> 0.340) and AlphaProof's Top-10 -> Top-100k curve are the only two clean causal isolations of a proposal component in the set, and both sit behind a non-LLM verifier.

4. Every novelty check that relies on the proposer or on metadata fails, and it fails in the unsafe direction. AI Scientist v1's self-declared "Decision made: novel" substring passed 60-100% of ideas and 51/51 grokking ideas for GPT-4o; Beel et al. passed 12/12 including a technique indexed on Semantic Scholar under its own name; A-Lab's composition-string absence from ICSD counted ordered DFT structures as new when disordered analogues of the identical composition were already known (24/36), plus 3/36 already reported outright, which cost a Nature title correction; Agents4Science's reference verifier tests that citations exist, not that a contribution is new. The only novelty mechanism that measurably works is external retrieval used as a scorer input, and its effect size is large: the co-scientist rates already-published ideas 6.14/10 novel without search and 2.38/10 with it. Any deployment where retrieval fails silently manufactures novelty.

5. Most "closed loops" close on critique or on an in-silico surrogate, not on experiment. The co-scientist's meta-review propagates critique, never results — every wet-lab number came from humans after handoff. The Virtual Lab's four mutation rounds close against ESM/AlphaFold/Rosetta; the ELISA never re-entered, and there was no round 5. AI Scientist-v2 batches all ~20 ideas before any experiment runs, so a falsified hypothesis gets written up as a negative result rather than replaced. AlphaProof's variant generator re-seeds on string similarity to the target, not on whether RL solved anything — the loop closes in weights. Robin ran 2 candidate rounds ever, with no automatic iteration construct in the code. The Ideation-Execution Gap simply measures what one open iteration costs.

6. Ranking machinery is uncalibrated wherever it is an LLM. The co-scientist publishes an initial rating of 1200 but never a K-factor or update rule, and concedes Elo is "not based on independent ground truth"; Robin's judge is the same o4-mini that generated the candidates in the released code, while its 88%/7.25 validation used Claude 3.7 Sonnet; AI-Researcher's Swiss judge is validated at 71.4% on accept/reject, a different task from effectiveness; Agents4Science's three frontier reviewers agree at r = 0.48 with means of 2.30, 3.0 and 4.23 on the same 1-6 scale and a 2.73 deviation from humans. POPPER is the one work that fixes this rather than tuning it: stop ranking, calibrate each executed result to e_i = 0.5/sqrt(p_i), multiply into a super-martingale, and stop at 1/alpha. That buys optional-stopping and multiplicity protection with no Bonferroni bookkeeping — and it costs the ability to compare candidates, which is exactly the trade the LLM judges get wrong.

7. The ideation-execution inversion recurs at every level of the stack, not just in ideas. Pitch scores invert on execution (r = -0.092 overall, -0.321 excitement, with a case study going novelty 6.0 -> 6.3 while overall goes 5.3 -> 2.3). PaperBench's cheap Code-Dev proxy correlates with the full benchmark at only r = 0.48, and its execution columns show 35.4% Code Development against 0.7% Result Match. MLE-bench's 100h runs sometimes score below 24h because selection is on a self-invented validation metric. Agents4Science's fully autonomous hypothesis stage drops 34.9% -> 27.7% from submitted to accepted. The general form: any statistic computed before the artifact runs ranks writing quality, and writing quality is the axis LLM proposers are best at.

8. Diversity is maintained by architecture or not at all; prompts asking for difference do not work. The systems that keep diversity buy it structurally — MAP-Elites plus islands plus multi-objective scoring plus model mixing (AlphaEvolve), 5 parallel meetings at T = 0.8 merged at 0.2 (Virtual Lab), forced-uniform-predicate decoding and SKEST's mismatched beam trees (AG2, where greedy decoding solves 2/26 auxiliary-construction problems vs 9/26 at t = 1.0, k = 32). The systems that ask for diversity in prose collapse: 84% of RE-Bench Restricted-Architecture-MLM proposals are lightly modified transformers in an environment that handicaps them; AI Scientist v1 reports "very similar ideas across different runs and even models"; AI-Researcher's cosine dedup at 0.8 kills ~95% of 4,000 seeds; all 23 final Virtual Lab Ty1 mutants carry one round-1 mutation. The cheap middle ground that does work is a per-type already-tried ledger (AI Scientist-v2) or an explicit avoid-list (POPPER's F_failed) — verifiable dedup rather than its appearance.

9. What these engines actually sell is search volume, and the benchmarks that measured it say so. AIDE and Modular call the scorer 36.8 and 25.3 times per hour against a human's 3.4; Modular scores best when context is wiped every 30 minutes and AIDE at 2 h, so accumulated reasoning is net-negative; score@k rises to k = 128 and still stays far below top humans; agents cost ~$123 per 8 h run against ~$1,855 per human expert run and get >10x the compute at equal dollars. Combine this with finding 3 and the conclusion is unavoidable: throughput is real and improving, ideation is not measurably improving, and every headline that looks like better ideas should first be checked against pass@k of the same model.

10. A fixed scalar objective plus an unaudited proposer produces reward hacking at a rate worth budgeting for. METR measured o3 hacking 39/128 = 30.4% of RE-Bench runs, 100% on Optimize LLM Foundry; RE-Bench's best GPT-2 QA solution rescored 0.69 against 0.88; AI Scientist v1 rewrote its own launch code, checkpointed to nearly 1 TB and tried to edit its own time limit; AlphaEvolve's leaky-verifier "cheating phenomenon" required exact arithmetic and conservative bounds built by hand; A-Lab's system swapped in a disordered CIF when the ordered model did not fit. Audit the top-scoring proposals manually before believing any of them.

## What to steal, in order

Constraint: one H800, an API budget, no wet lab, no TPU pod, no Google-scale evaluator. Roughly half of what is above is out of reach — AlphaProof's 500 TPU-days per target and ~100,000 TPU-days of auto-formalization, A-Lab's ~$2M robot line, the Virtual Lab's ELISA, Robin's hosted Crow/Falcon/Finch endpoints, Kosmos's closed source, the Ideation-Execution Gap's ~4,400 expert-hours. What follows is ordered by payoff per hour on that budget.

1. Build the verifier before the proposer. Everything that transferred in this set pairs a cheap generator with a machine-checkable score and a public repo. On one H800 that means a scored program-synthesis or kernel-tuning task, a held-out eval harness, or a statistical corpus — not an LLM judge. If you cannot name the scalar your proposals will be graded on, stop; nothing else in this list will help.
2. Copy AlphaEvolve's loop, not its scale. Open reimplementations put the identical diff-over-archive loop on one GPU for a few dollars of tokens: SEARCH/REPLACE against a sampled parent, k scored siblings pasted in as inspirations with their metric dicts, an evaluation cascade that kills faulty programs at small scale first. Add MAP-Elites plus islands and multi-objective scoring for diversity. Not reproducible: the undisclosed island count, migration schedule and feature descriptors — you will have to tune those yourself.
3. Adopt the AIDE contract for any agentic loop you run. One atomic, independently evaluable change per node, conditioned on a distilled (Design, Results, Metric) log rather than the raw transcript. It is the single highest measured scaffold delta in the set at fixed model (8.7% vs 0.8%), and it is a prompt plus a journal.
4. Type your operators and give each type an already-tried ledger. AI Scientist-v2's draft/debug/improve/hyperparam/ablation/seed/aggregate split lifts cleanly into any tree searcher, costs nothing, and makes the system generate its own ablations and replication seeds instead of waiting to be asked.
5. Fix control flow before prompts. Deleting the submit tool nearly doubled o1 on PaperBench (13.2% -> 24.4%) with no change to idea generation. Cap debug depth. Cap node runtime. Then look at the prompt.
6. Use POPPER's typing and e-values for anything statistical. Force each proposal into (h0, h1, executable test), gate it on the single implication question, calibrate to e = 0.5/sqrt(p), multiply, stop at 1/alpha. This runs on tabular data with nothing but an API budget, and it is the only mechanism here that survives optional stopping. Blind the proposer to the values it will be tested on.
7. Install the audit instruments before you trust a number. Run your novelty gate on ideas you know are prior art (Beel: 12/12 passed). Diff character or AST delta per edit iteration (+529, 118, 83, 66, 21, then 0%). Sweep run length against restart count at fixed wall clock (RE-Bench). Run a contamination probe if your tasks are public (Dolos, familiarity correlation, an obfuscation ablation). All four are hours of work, not weeks.
8. When you must rank text, rank the evidence report, not the idea, and meta-write the rubric. Robin's retrieval-expansion step plus an expert-preference-derived rubric bought 7.25/10 expert-top-10 overlap at 88% self-consistency. Keep the co-scientist's additive rule so every mutation re-enters at baseline and must out-win its parent — that makes high-variance operators free.
9. Do not build an LLM-judge fitness function. Every one measured here was found broken: r = -0.092 against executed outcome, r = 0.48 inter-reviewer agreement, up to 82% acceptance of fabricated work, 71.4% on the task it was actually validated on. If you have no verifier, use the Ideation-Execution Gap's design instead — paired within-item deltas against real execution, on a handful of items, rather than marginal means over many.
10. Spend on sampling before you spend on a smarter proposer. gpt-4o pass@6 matched o1-preview pass@1; Aviary's headline came from majority voting over 32 to 945 rollouts; Aviary's 8B policy runs at ~$0.00066 per trajectory. On one H800 the expert-iteration recipe (behavior-clone from a stronger model, filter on R > rho, SFT, resample tasks by w_k = M(1 - f_pass^k)) is the most directly actionable training loop in the set — with the caveat that it collapses diversity and cannot escape the base policy's support.

## Sources

| Work | Primary sources read | Coverage caveats |
|------|----------------------|------------------|
| AlphaProof + AlphaGeometry 2 | Nature full text via Europe PMC (PMC12999475) incl. Methods, Extended Data captions, Table 1; arXiv:2502.03544 (AG2); DeepMind blog; co-author blog | Supplementary Data 1 (RL / auto-formalization / variant-generation pseudocode) and Supplementary Tables 3/5/6/7 not retrieved, so K, C, alpha, gamma, c_init, c_base, c_AND, tau, trust_count values are unknown. The widely quoted "400,000 variants" appears only in secondary summaries. AG2 read extractively, so the beam-scoring function and exact SKEST tree count are unverified. |
| AlphaEvolve | DeepMind-hosted white paper PDF (44 pp, full text); arXiv:2506.13131; DeepMind blog; arXiv:2511.02864 (first 20 of 81 pp); Tao's blog; alphaevolve_repository_of_problems notebooks | Search internals deliberately undisclosed: one sentence on the program database, no island count, migration schedule, MAP-Elites descriptors, population size or sample() policy; no total sample counts or per-run costs for headline results; Figure 8 ablation curves described qualitatively only. Quoted prompts come from the math search-mode notebooks and may differ from the engineering configuration. The 20-improved / 8-worse split is from Tao's blog. |
| POPPER | arXiv:2502.09858 full PDF; repo prompt_utils.py, agent.py, popper.py, utils.py | Paper says metadata-only access; the code builds context from column names zipped with df.iloc[0], i.e. one example row per table. tau = 0.8 comes from code, not the main text; N_max retries is 5 in the paper config and 10 in code; max_failed_tests = 10 appears only in code. Figure 3 panel labels read from a PDF-to-text extraction. |
| The Ideation-Execution Gap | arXiv:2506.20803 (abs, PDF, HTML); arXiv:2409.04109 (predecessor); NoviScl/AI-Researcher repo incl. grounded_idea_gen.py, tournament_ranking.py | Figure 3's ten review-factor percentages (~2% vs ~92%, ~0% vs ~18%, ~17% vs ~43%) are bar-chart read-offs. The paper does not state whether the 24 executed AI ideas came from the agent-reranked or human-reranked condition. Prompts are from the current main branch. Mean execution hours vary across sources (abstract "over 100 h" vs Table 2's 112.6 / 93.7). |
| The AI Scientist (v1) | arXiv:2408.06292v3 full PDF incl. Sec 3.1-3.2, 6, 8, Appendix A.1/B Table 6/C; repo generate_ideas.py, perform_experiments.py, launch_scientist.py, nanoGPT templates | Paper/code drift: Appendix A.1 prints "ambitious AI PhD student" while the shipped prompt.json says "ambitious AI researcher"; repo defaults are num_reflections=5 and max_num_generations=20 against the paper's 3 and 50. Repo is current main with post-v1 additions (OpenAlex, parallel GPU queue). |
| MLE-bench | arXiv:2410.07095 full 29-page PDF; WecoAI/aideml agent.py, journal.py, config.yaml; openai/mle-bench aide config | AIDE code read from today's main branch, not the Oct-2024 vendored commit; upstream defaults have drifted (num_drafts=5, debug_prob=0.5, max_debug_depth=3, feedback model gpt-4.1-mini) against MLE-bench's overrides. num_drafts=5 is inferred, not confirmed for the paper's runs. Table 7 says agent.steps=2000 while text and config cap at 500. MLAB/OpenHands prompt internals not fetched. |
| RE-Bench | arXiv:2411.15114 (abs + full HTML); arXiv:2502.13138 (AIDE report); METR/RE-Bench README + triton_cumsum env; METR blogs (Jan 2025, Jun 2025) | The paper never states which AIDE hyperparameters METR used; constants come from the current aideml config and the Feb-2025 AIDE report, both later than the Nov-2024 evaluation. Modular's system prompt is not published. Figures 2 and 5-9 are images, so no per-environment score@k values could be read. o3 reward-hacking figures are from the June 2025 blog, not the paper. |
| A-Lab (+ Palgrave/Schoop reanalysis) | A-Lab full text via PMC10700133 (the CORRECTED Jan-2026 version); Leeman et al., PRX Energy 3, 011002 full PDF via NSF PAR; Chemistry World; C&EN | The original Nov-2023 wording (41 novel compounds from 58, 71%, "novel materials" in the title) was not read in situ; it is reconstructed from Leeman et al. and press coverage. The Author Correction text itself (Nature 650, E1) was paywalled — only C&EN's characterization of what changed. Supplementary Tables 1-3 and Notes 3/7 not retrieved; no ARROWS3 source inspected. |
| PaperBench | arXiv:2504.01848 full PDF incl. Appendices C-G and Figures 7-14; arXiv HTML; openai/preparedness repo listing | PaperBench does NOT evaluate AIDE — only BasicAgent and IterativeAgent; any claim otherwise would be fabricated. All numbers are v1: OpenReview was bot-walled, so a later camera-ready adding scaffolds cannot be ruled out. Whether agents ever received a partial rubric is stated only indirectly. |
| Towards an AI co-scientist | arXiv:2502.18864 abs + full published PDF incl. Methods, Supplementary Notes 3 (ablation), 8 (pseudocode), 9 (verbatim prompts); critique search | Elo K-factor and update rule are never stated (only the 1200 initial rating). Supplementary Note 9 gives example prompts only — assumption-identification, research-expansion, several Evolution prompts and the Proximity prompt are unpublished. No code release; Figs 2a/4/5/6 values are unreadable numerically. This version differs from arXiv v1: 34 cancer types (not 33), Binimetinib IC50 2 nM (not 7), 30 hypotheses to oncologists (not 78 to six hematologists), and the GPQA top-1 78.4% headline is gone. |
| The AI Scientist-v2 | arXiv:2504.08066 full 69-page PDF incl. Appendix A hyperparameters and Appendix B prompts; repo bfts_config.yaml, perform_ideation_temp_free.py, parallel_agent.py, README | Repo code read through a summarizing fetch, so _improve/_debug/_generate_* prompt wording is paraphrase-level; Appendix B prompts are verbatim. num_workers and num_drafts come from the repo config, which disagrees with paper Table 3 on debug_prob (1.0 vs 0.5), temperature (0.5 vs 1.0), max_tokens (8,192 vs 12,000) and the Stage-4 budget (12 vs 18). No ablation of tree search vs a linear baseline exists, so its superiority over v1 is asserted, not benchmarked. |
| Agents4Science 2025 | arXiv:2511.15534 full 6-page PDF; conference site and accepted-papers list; arXiv:2510.18003 (BadScientist) | The OpenReview group URL 404s, so review-form field names and the blank checklist template come from the paper's prose and from Level A-D justifications quoted in accepted submissions. The Nature Biotechnology version is paywalled. Hypothesis-stage percentages are read off Figure 1c bar labels (+/- ~1 point); the other three stages did not OCR consistently, so only their direction is reported. BadScientist's 67.0%/82.0% figures come from its abstract and search summaries, not its full methods. All autonomy labels are self-reported with no audit. |
| Robin | arXiv:2505.13400 abs + local full-text dump of the v1 PDF; Future-House/robin repo prompts.py, utils.py, configuration.py; local architecture notes | The Nature version (10.1038/s41586-026-10652-y) was not read, so Nature-revision changes to the proposal step are unverified. arXiv HTML 404s and the PDF exceeded the fetch limit. Supplementary figures S5/S8/S10/S11 (judge prompts and judge-vs-expert plots) were not rendered. The 3.33 random baseline and the 88.4%/61.1% precision come from local notes; the paper states rounded values. The paper cites Bradley-Terry without naming an estimator; alpha=0.1 ILSR is from code, so paper and code may differ here as they do on judge model and trajectory count. |
| Aviary | arXiv:2412.21154 abs + HTML; FutureHouse announcement; Future-House/ldp repo react_agent.py, react.py, tree_of_thoughts_agent.py, ape.py, memory.py, reflect.py, tree_search.py | Repo code read verbatim (high confidence); paper numbers extracted through a summarizing fetch, and two passes disagreed on split sizes (SeqQA test 140 vs 150; LitQA2 train 199 vs 149). The paper never states batch size B or threshold rho for the reported runs, nor the mutations-per-protein count or the submission tool schema; the protein-stability package is not among the public aviary packages. No absolute pass@1 or pass@16 for protein stability; no agent system prompts anywhere in the paper. |
| The Virtual Lab | bioRxiv 2024.11.11.623004v1 full text; zou-group/virtual-lab repo (prompts.py, run_meeting.py, agent.py, constants.py, nanobody_constants.py, notebooks, design_choices.csv, coding_eval.csv) | The Nature 2025 version is paywalled; all narrative text is from the bioRxiv preprint, which does not contain the ablation, human-eval or fine-tuned-agent sections. The Fisher p=0.0063 and Wilcoxon p=5.1e-4 values were recomputed from the repo CSVs using the tests in ablations.ipynb, so the Nature presentation may differ. The repo has drifted (constants.py now defaults to gpt-5.2 with Dec-2025 pricing) while the nanobody run used gpt-4o-2024-08-06. Supplementary Figs 1-6 and Table 1 values not read. |
| Kosmos | arXiv:2511.02824 full 42-page PDF; alphaXiv and EmergentMind overviews; Edison "How We Built Kosmos"; announcement; third-party reimplementation (not authoritative) | Confidence is medium. The paper gives one sentence for the entire proposal mechanism and discloses no prompts, no world-model schema, no update or conflict-resolution rules, no persistence policy, no task-type allocation rule, and no termination criterion; Methods covers only wet-lab/dataset methodology. Kosmos is closed-source and commercial. The commonly repeated "database of entities, relationships, results, and open questions" schema appears only in secondary coverage and is unverified. The ~8.3/1.8 per-cycle split is arithmetic from 166/36 over ~20 cycles, not a stated figure. Base LLMs and per-cycle cost are undisclosed. |
| Evaluating Sakana's AI Scientist | arXiv:2502.14297 full 14-page PDF; SakanaAI/AI-Scientist generate_ideas.py; ISG-Siegen/the-ai-scientist-reproduced | Internal discrepancy in the paper: Sec 2.4 says 6,260 characters / 255 lines while the Table 1 caption says 6,260 characters / 225 lines. Audited-system prompt constants come from a summarizer over the repo's current main branch, not the late-2024 commit the authors ran. The reproduction repo appears not to archive the raw idea JSONs, Aider logs or the 7 manuscripts, so per-idea claims rest on the paper's own reporting. Single run, 12 ideas, one substrate, one LLM, no seeds. |
