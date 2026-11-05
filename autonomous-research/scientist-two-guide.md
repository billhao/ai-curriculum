# ScientistTwo: Turning Experimental Criticism into the Next Experiment

How a research agent evolves a method through benchmark screening, ablations,
and experiment-running rebuttals, and how to interpret its claims of advancing
the scientific frontier.

## Background

**Primary paper:** [ScientistTwo: Pioneering the Human Knowledge Frontier with
Autonomous AI](https://arxiv.org/abs/2609.19644), submitted 17 September 2026.
Jaehyun Nam, Jinsung Yoon, Yanzhou Pan, Rui Meng, Parthasarathy Ranganathan,
and Tomas Pfister are affiliated with **Google Cloud AI Research**; Yubo Wang
is affiliated with the **University of Waterloo**, as printed on PDF p. 1.
The [project page](https://scientist-two.github.io/) hosts generated papers.

This is a useful next read after the
[AI Scientist](ai-scientist-guide.md) and
[execution-grounded research](execution-grounded-auto-research-guide.md).
You already know how agents propose and execute training changes. ScientistTwo
addresses the next question: **after a change works, can the agent determine
why, revise the mechanism, and run the additional experiments needed to defend
the resulting paper?** Its main contribution is the research workflow.

The relevant lineage is:

| Work | Authors / organization | Year | Contribution to this story |
|---|---|---|---|
| AI Scientist | Chris Lu et al.; Sakana, Oxford, UBC | 2024 | Full idea-to-paper loop |
| AI Scientist-v2 | Yutaro Yamada et al.; Sakana-led | 2025 | Experiment tree search |
| MLE-STAR | Jaehyun Nam et al.; Google Cloud, KAIST | 2025 | Search and targeted refinement |
| ScholarPeer | Palash Goyal et al.; Google | 2026 | Literature-aware review |
| PaperOrchestra | Yiwen Song et al.; Google | 2026 | Research materials to manuscript |
| ScientistOne | Rui Meng et al.; Google Cloud AI Research | 2026 | Claims linked to artifacts |
| ScientistTwo | Nam et al.; Google Cloud AI Research, Waterloo | 2026 | Ablation and rebuttal feedback |

[ScientistOne](https://arxiv.org/abs/2605.26340) supplies the closest conceptual
predecessor: **Chain-of-Evidence**, which connects paper claims to executable
code, measured outputs, and references. ScientistTwo adds active refinement:
criticism can change the experiment and method, not just the explanation.
[PaperOrchestra](https://arxiv.org/abs/2604.05018) and
[ScholarPeer](https://arxiv.org/abs/2601.22638) are incorporated writing and
review modules. MLE-STAR's originating preprint and NeurIPS publication are
from 2025, although ScientistTwo's bibliography labels that entry 2026.

All page references below use the **71-page PDF**, including its appendices;
the embedded generated paper also has its own page numbers.

## 1. The problem: a working modification is only the beginning

Imagine your GPT-2 124M run improves after you simultaneously introduce a new
residual connection, change the learning rate, and average checkpoints.
The validation loss moved, but the proposed residual mechanism may contribute
nothing. A paper claiming that mechanism caused the improvement would be weak
even if its code reproduced perfectly.

ScientistTwo tries to automate three different judgments:

1. **Performance:** does the candidate improve the relevant benchmark suite?
2. **Attribution:** which component accounts for the improvement?
3. **Adequacy:** what additional evidence would answer a critical reviewer?

These judgments have different outputs. A performance critic may request
hyperparameter engineering. An ablation critic may delete the main proposed
component. A reviewer may request another dataset or backbone. Treating all
three as a request to rewrite the introduction misses the central idea.

The input is a research problem grounded in an existing paper and codebase.
The evaluation uses 107 problems derived from accepted ML papers. Thus the
demonstrated autonomy starts from substantial human research infrastructure,
including established tasks and benchmarks (§4, pp. 10–12; Appendix A.1,
pp. 28–29).

## 2. The complete loop

```text
Research problem + prior paper/code
                 |
                 v
Extract limitations -> Generate and rank seed ideas
                 |
                 v
Reproduce subset baseline -> Implement candidate on subset
                                      |
                         critic: reject / engineer / scale
                                      |
                                      v
                            Full benchmark evaluation
                                      |
                   history of successes AND failures
                                      |
                     evolve ideas + try unused seeds
                                      |
                                      v
                          Select strongest valid idea
                                      |
                                      v
                        Ablate -> refine -> compare
                                      |
                                      v
                          Draft -> review -> rebuttal
                                      |       |
                                      |       v
                                      |   Run new experiments
                                      |       |
                                      +<------+
                                      |
                                      v
                         Meta-review -> final artifacts
                              |
                   revise method if evidence improves
                   then repeat ablations and drafting
```

The recurring unit is a **candidate, critic, refinement loop**. A critic can
accept the artifact, request a bounded refinement, or reject it. The candidate
can be a limitation list, code and results, an ablation breakdown, or a draft.
Listing 1 and Table 1 formalize this shared controller (p. 5); Figures 4–7 and
§§3.1–3.6 describe its specialized instances (pp. 6–9).

The system uses inference from existing models to run this process. There is
no new foundation-model training objective, SFT stage, or GRPO update in the
ScientistTwo controller. Some **experiments proposed by the controller** do
train models; that is distinct from training the research agent itself.

## 3. Search over experimental records

### Limitations become candidates

A Limitation Extractor identifies actionable weaknesses in the existing
method. A verifier requests missing limitations. The Idea Generator then
proposes seed ideas and ranks them by a novelty check. In the reported
configuration, this check retrieves **two reference papers through Google
Search** (Appendix A.2, p. 30). This is a practical novelty screen, not an
exhaustive test that an idea has never appeared in the literature.

### A cheap experiment gates an expensive experiment

The baseline coder first reproduces the original method on a benchmark subset.
A candidate modifies that implementation. The subset critic can:

1. discard an unpromising candidate;
2. send it to an engineer for code or hyperparameter changes;
3. promote it to the full benchmark suite.

The comparison changes at promotion: the subset result is checked against the
**reproduced subset baseline**, while the full-set critic checks the candidate
against the **original state-of-the-art result** (Table 1, p. 5; §3.2,
pp. 6–7). Those references can differ. Recording both is essential when
hardware, implementations, or training recipes produce baseline drift.

The benefit is straightforward. In a teaching example, if one subset test
costs 10 GPU-minutes and one full test costs 100, screening six candidates and
promoting two costs $6(10)+2(100)=260$ GPU-minutes, rather than $6(100)=600$.
The risk is equally concrete: a candidate that helps long sequences but hurts
the chosen short-sequence subset could be discarded before its strength is
measured. Subset selection is part of the scientific design.

### Failed experiments remain useful

For each idea, the system retains a record equivalent to:

$$
z_h=(h,E_h,C_h,d_h,r_h),
$$

where $h$ is the hypothesis, $E_h$ the experimental outputs, $C_h$ the code,
$d_h$ the decision, and $r_h$ the diagnostic feedback. The Idea Evolver reads
the accumulated records. It can improve a successful idea or use a failed
implementation to diagnose what needs to change. Previously untested seed
ideas are also introduced, preserving an exploration path outside the early
winners (§3.3, Eqs. 2–4, pp. 7–8).

This resembles execution-grounded evolutionary search you know: the memory
changes the next proposal distribution without updating model weights.
ScientistTwo additionally asks whether a gain has a defensible mechanism and
enough evidence for a manuscript.

## 4. The distinctive step: ablation can replace the idea

An **ablation** changes or removes a component while keeping the rest of the
experiment fixed. Here it is a decision input, not merely an extra paper table.
The ablation critic can ask for a simpler or different method. An improved
candidate replaces the current best only after the result comparator prefers
its measured results; the system then repeats the ablations (§3.4, p. 8).

The appendix's OOD detection example makes this tangible. OOD detection means
distinguishing inputs outside the training distribution. The initial
**Procrustes-DS** proposal combines layer alignment, kernel-based layer
weighting, feature gating, regularized covariance estimation, and evidential
fusion. Several components fail their own empirical tests.

| Proposed component | What the experiments reveal | Resulting change |
|---|---|---|
| Kernel-based layer weights | Nearly uniform weights | Use class separation |
| Procrustes alignment | About +0.01% AUROC | Remove alignment |
| Feature-level gating | Hurts near-OOD detection | Gate scores instead |
| Fixed covariance penalties | Wrong scale for features | Scale-matched shrinkage |
| Dempster–Shafer fusion | Reduces to logit addition | Use weighted evidence |

The proposal, refined report, ablations, and criticism appear on pp. 36–46.
The name survives, but the successful implementation differs substantially
from the original proposal. That is a useful example of empirical research
changing its mind.

For a numerical walkthrough, the refined report gives the following means
over six OOD datasets (pp. 40–42):

| Method | AUROC (%) | FPR95 (%) |
|---|---:|---:|
| Published X-Maha baseline | 99.30 | 3.76 |
| Reproduced X-Maha baseline | 99.16 | 4.17 |
| Refined candidate | 99.56 | 2.17 |

**AUROC** measures discrimination across thresholds; higher is better.
**FPR95** is the false-positive rate at 95% true-positive rate; lower is
better. The FPR95 reduction is **2.00 percentage points** against the rounded
reproduced baseline, or about **48.0% relative**:

$$
100\frac{4.17-2.17}{4.17}\approx47.96\%.
$$

Against the published baseline it is **1.59 points**, about **42.3% relative**.
The excerpt itself prints 1.99 and 1.58 points, so its prose and displayed
rounded cells differ by 0.01 point. The distinction between reproduced and
published baselines matters far more than that rounding discrepancy.

The engineering lesson for your training experiments is to compare the final
recipe with a matched control, then ablate the advertised mechanism. A
successful bundle does not establish that every component helps.

## 5. Rebuttal as experiment scheduling

After PaperOrchestra drafts the manuscript, ScholarPeer reviews it. A low
score triggers a Rebuttal Planner that converts objections into supplementary
experimental tasks. Coders run those tasks; the Draft Enhancer incorporates
their results and revises the claims. A meta-reviewer can request a deeper
method change, followed by renewed ablations and drafting (§§3.5–3.6, p. 9).

An appendix example asks whether **TABHARMONY** transfers from its original
tabular backbone to **TabPFN v2**. The response is an actual second-backbone
experiment: 50 datasets, five seeds, paired baseline/candidate evaluation.
The report separates a small classification improvement from a stronger
regression improvement, rather than claiming universal wins (pp. 51–55).

In your SFT work, an analogous objection might be: “Does the proposed loss
masking change still help on another instruction dataset?” A useful rebuttal
artifact would contain the second dataset's training configuration, code
revision, checkpoints, and results. A persuasive paragraph alone cannot
answer that objection.

The meta-review loop also has rollback: if a revised idea fails to improve the
measured results, the system keeps the previous best paper and code. An
implementation should version hypothesis, code, results, and manuscript
together; otherwise it can accidentally describe a rejected method using
numbers from another version.

## 6. Configuration and the unresolved comparison rule

The reported settings are bounded, not indefinite autonomous deliberation
(Appendix A.2, p. 30):

| Control | Reported setting |
|---|---|
| Most planning / critic agents | Gemini 3.6 Flash |
| Experiment, ablation, rebuttal coding; enhancement | Claude Code, Opus 4.8 |
| Limitation extraction | At most 16 rounds |
| Experimentation | Up to 4 rounds; 2 candidates: seed + evolved |
| Early stopping | 4 successful ideas |
| Engineering per flagged idea | At most 2 rounds |
| Ablation-driven refinement | At most 1 round |
| Simulated peer review | At most 2 rounds; stop at score 8 |
| Review-driven method refinement | At most 1 round |

The paper describes critics as judging whether results are consistently or
strictly better, but does not provide a universal numerical tolerance or a
fixed scalarization of competing metrics (§§3.2–3.6). The Selector Agent
compares results and logs; Eq. 4 is an agent interface, not a specified
mathematical argmax that another implementation can reproduce exactly.

For your GPT-2 experiments, write the comparison rule before the search:
for example, lower held-out loss at equal tokens and GPU time, with fixed
evaluation code, repeated seeds, and an explicit tolerance for noise. For
SFT, specify whether a quality gain can compensate for higher latency or
lower accuracy elsewhere. More metrics help only when the decision rule is
clear and the final test remains separate from repeated search feedback.

## 7. What the evidence establishes

### Generation success and automated review are separate outcomes

ScientistTwo reports successful improvements on **86 of 107 tasks**, or
**80.4%**. Those 86 completed papers receive the headline review scores.
Table 3 gives the task breakdown (p. 12):

| Source of task | Completed / attempted | Stanford rating | Automated acceptance |
|---|---:|---:|---:|
| ICLR 2026 | 4 / 5 | 5.4 | 75.0% |
| NeurIPS 2025 | 33 / 38 | 5.6 | 75.8% |
| ICML 2026 spotlight | 49 / 64 | 5.7 | 69.4% |
| Overall | 86 / 107 | 5.7 | 72.1% |

ScholarPeer gives the completed papers an average **7.5/10** and **91.9%**
automated acceptance; Stanford Agentic Reviewer gives **5.7/10** and **72.1%**.
ScholarPeer also supplies optimization feedback. Stanford is held out from
development (§4, p. 10).

The 72.1% figure is conditional on completion. Counting unsuccessful tasks as
failures gives approximately **62 of 107**, or **57.9%**, yielding both a
completed artifact and a Stanford-accepted judgment. This is a denominator
conversion, not a new reported benchmark result.

These are **automated reviewer judgments**, not actual acceptance decisions
by ICLR, ICML, or NeurIPS. The
[Stanford reviewer's own description](https://paperreview.ai/tech-overview)
positions it as rapid feedback calibrated against human review scores.
Different public paper collections are used for prior-system comparisons in
Table 2; this is not a matched rerun of every research agent on all 107 tasks.

### A first rebuttal helps; a second does not uniformly help

On the 49 successful ICML-derived tasks, Table 5 shows (p. 13):

| Review rounds | ScholarPeer rating | ScholarPeer accept | Stanford rating | Stanford accept |
|---|---:|---:|---:|---:|
| 0 | 5.2 | 46.9% | 5.6 | 49.0% |
| 1 | 6.9 | 79.6% | 5.8 | 73.5% |
| 2 | 7.6 | 93.9% | 5.7 | 69.4% |

The first review/rebuttal round improves both evaluators. The second improves
the reviewer being optimized while reducing held-out acceptance. This pattern
is consistent with diminishing returns or reviewer-specific optimization;
the table alone does not determine the cause. It does show why an in-loop
review score cannot be treated as scientific quality itself.

### Integrity is necessary, with a narrower audited population

The four Chain-of-Evidence audit dimensions are score reproduction,
specification compliance, reference existence, and method–code alignment.
Table 7 reports **49/49** score reproductions and method–code alignments,
**0/49** specification violations, and **0/1,814** hallucinated references
for the final audited variant (p. 14).

This table covers the successful ICML subset, not all 86 final papers. The
variant without specification filtering completes an additional task, but its
solution exploits the rules and is discarded (p. 15, footnote 2). Passing an
artifact audit establishes neither novelty nor generalization. A wrong
scientific conclusion can be faithfully implemented and reproducible.

### Gains, human review, and cost need their own denominators

The headline **25.2% mean relative gain** has a **7.7% median** over successful
cases (Table 4, p. 12). Gains are extracted from main result tables using
Gemini, repeated ten times and averaged. They span different tasks and metrics;
they are not a common loss reduction comparable to one GPT-2 training run.
Appendix B explicitly says the AutoSOTA gain columns are not a controlled
head-to-head under common metrics and hardware (pp. 30–32).

Human evaluation covers **33 NeurIPS-derived papers and nine reviewers**.
The comparative overall mean is **3.0/5**, defined as parity with the human
paper; methodology is **2.9**, slightly favoring humans, and experimental
benchmark breadth is **3.5**, favoring ScientistTwo (Table 10, p. 16). The
main description does not provide confidence intervals sufficient to turn
those means into a claim of general superiority.

The same 33-task subset supplies the cost analysis: about **2.5 days** and
**$3,765** per completed research cycle, including model tokens and virtual
machines (Figure 10 and §4.3, pp. 15–16). That is not an all-107-task cost
accounting that includes every unsuccessful attempt.

For implementation, the official project page provides a paper gallery and
demo. As of 5 November 2026, no reusable framework-source link was found on
that page; a generated-paper release is different from a runnable research
controller release.

## 8. A generated method: DynaSpec-RAG

The full generated manuscript on pp. 56–71 is useful for seeing what the
workflow produces. **DynaSpec-RAG** modifies time-series forecasts using
retrieved historical analogues. Its backbone forecast stays frozen; a small
trainable module decides which retrieved information to use.

The intuition is that a retrieved trajectory may have a useful slow trend
and an unreliable fast oscillation. Trusting the whole curve equally throws
away that distinction. The method aligns the retrieved context boundary,
splits the future trajectory into low/high frequency bands using real FFT,
and gates the residual corrections separately at each forecast step.

The generated paper's Eq. 1 can be written as:

$$
\hat y_{\mathrm{final}}=\hat y_{\mathrm{base}}
+\beta\,\mathrm{scale}_q
\left(g_{\mathrm{low}}\odot r_{\mathrm{low}}
+g_{\mathrm{high}}\odot r_{\mathrm{high}}\right).
$$

Here $r$ is the retrieved-minus-baseline residual in each band, $g$ is a
per-step gate between zero and one, $\mathrm{scale}_q$ restores the query's
units after normalization, and $\beta$ is a validation-calibrated global
scale. The lookback is **512**, forecast horizon **64**, and retrieval uses
**10 neighbors** (pp. 59–61).

For a teaching example at one forecast step, let the baseline be 10, query
scale 2, low/high residuals 0.5 and -0.2, gates 0.8 and 0.1, and
$\beta=0.5$. Then:

$$
\hat y_{\mathrm{final}}
=10+0.5(2)\left(0.8(0.5)+0.1(-0.2)\right)=10.38.
$$

The trend correction receives much more trust than the noisy fast component.
Setting $\beta=0$ recovers the baseline exactly.

### What its safety guarantee actually guarantees

The calibration grid includes $\beta=0$ and minimizes (p. 61, Eq. 7):

$$
J(\beta)=
\frac{\mathrm{MSE}_{\mathrm{val}}(\beta)}
     {\mathrm{MSE}_{\mathrm{base,val}}}
+\frac{\mathrm{MAE}_{\mathrm{val}}(\beta)}
      {\mathrm{MAE}_{\mathrm{base,val}}}.
$$

Since $J(0)=2$, selecting the best grid point guarantees **$J\leq2$ on the
calibration validation set**. It does not guarantee that each metric improves:
normalized MSE 0.90 and normalized MAE 1.05 sum to 1.95, despite worse MAE.
It also provides no guarantee on unseen test data.

The actual table illustrates the limit. Average MSE decreases
**0.1940 to 0.1892**, about **2.47%**, while average MAE decreases
**0.2492 to 0.2488**, about **0.16%**. Electricity worsens from
**0.1120 to 0.1135 MSE** and **0.2002 to 0.2059 MAE**. ETTh1's MAE also
slightly worsens, **0.3624 to 0.3627**, although its MSE improves
(main Table 11, p. 17). The generated appendix explicitly counts **2/7**
datasets with a regression on either metric (p. 70, its Table 7).

The word **zero-shot** also needs scope. The backbone is frozen, but the
**265,746-parameter** module is trained for **40 epochs** on pooled benchmark
training splits with AdamW at **0.001**, then calibrated with dataset-specific
validation labels (pp. 61–62). It is not an entirely untrained system on
those seven datasets. Transfer to five held-out GIFT-Eval datasets without
module retraining is stronger transfer evidence, although it still uses
per-dataset validation calibration (pp. 63, 70).

This example captures both sides of ScientistTwo: the system can produce a
coherent mechanism with experiments, yet the reader must still distinguish
the precise mathematical claim from the manuscript's broader language.

## 9. Does it demonstrate recursive improvement?

Table 9 reports three successive improvements to **one incremental BPE
tokenization task**: VD-STrans, BXT-Transducer, and SBR-Transducer, with
relative gains **10.9%, 9.6%, and 8.2%** against the immediately previous
method (p. 16).

This is evidence that the same research process can continue improving its
own output on that task. It does not show that the researcher becomes more
capable across new tasks, that its controller improves, or that gains compound
indefinitely. The BPE percentages should not simply be added; throughput
increases and latency decreases have different denominators.

In your RSI framework, distinguish **improving the research product** from
**improving the research process**. ScientistTwo's repeated BPE example
directly addresses the former.

## 10. Comparison with the systems you know

| System | Main feedback | What changes | Distinctive emphasis |
|---|---|---|---|
| AI Scientist v1/v2 | Runs and reviews | Experiments and manuscript | Full pipeline; v2 tree search |
| AI co-scientist | Debate and ranking | Hypotheses | Broad hypothesis exploration |
| Execution-grounded evolution | Measured task reward | Idea population | Real execution guides selection |
| ScientistOne | Evidence and integrity checks | Artifacts and claims | Traceable, auditable research |
| ScientistTwo | Runs, ablations, rebuttals | Method, evidence, manuscript | Criticism schedules experiments |

The [AI co-scientist](ai-co-scientist-guide.md) concentrates on hypothesis
generation and ranking, with human collaborators closing experimental loops.
Execution-grounded evolution tightly tests whether ideas improve a registered
objective. ScientistTwo spends additional computation on attribution,
benchmark breadth, and manuscript defense.

These are partly complementary tools. If the goal is to tune your existing
GPT-2 recipe under a fixed budget, a simpler execution-grounded search may
answer the question more cheaply. If the goal is to establish a new training
mechanism, the ablation and rebuttal machinery becomes more relevant.

The evidence types also differ: AI Scientist-v2 reports an actual ICLR
workshop review outcome; ScientistTwo's headline venue-style acceptance
figures are produced by automated reviewers. Higher simulated scores alone
do not establish a stronger real publication result.

## 11. What to reuse in your experiments

The most reusable design is a persistent record of **hypothesis, code,
configuration, results, and criticism**. Start with a fixed evaluator and
budget, use representative subsets for screening, and retain failed-run
diagnostics so the next proposal can change for a concrete reason.

For a GPT-2 research question, require a matched run and an ablation that
removes the proposed mechanism while preserving generic improvements. For
an SFT question, separate whether training ran, whether a measured gain
survived another dataset, and whether the claimed explanation matches the
implementation. Keep final evaluation outside the search loop.

The broader significance is that an autonomous research harness can allocate
compute to **resolving uncertainty about a result**, rather than spending
every additional token on proposing more ideas. The unresolved challenge is
deciding what evidence is sufficient without letting the proposer optimize
its own scientific judge.

## Key Papers and Reading Route

1. **ScientistTwo**, Nam et al. (2026):
   [paper](https://arxiv.org/abs/2609.19644),
   [PDF](https://arxiv.org/pdf/2609.19644),
   [project](https://scientist-two.github.io/).
   Read §§3.4–3.6 first for the distinctive loops, then Table 5 and the
   appendix criticism on p. 46. Use the generated paper to inspect claims
   against its own evidence.
2. **ScientistOne: Towards Human-Level Autonomous Research via
   Chain-of-Evidence**, Meng et al. (2026):
   [paper](https://arxiv.org/abs/2605.26340).
   Read for the integrity standard that ScientistTwo inherits.
3. **PaperOrchestra: A Multi-Agent Framework for Automated AI Research
   Paper Writing**, Yiwen Song et al. (2026):
   [paper](https://arxiv.org/abs/2604.05018).
   The manuscript-production module; distinct from discovering the method.
4. **ScholarPeer: A Context-Aware Multi-Agent Framework for Automated
   Peer Review**, Goyal et al. (2026):
   [paper](https://arxiv.org/abs/2601.22638).
   The review module and in-loop feedback source.
5. **The AI Scientist**, Chris Lu et al. (2024):
   [paper](https://arxiv.org/abs/2408.06292);
   **The AI Scientist-v2**, Yutaro Yamada et al. (2025):
   [paper](https://arxiv.org/abs/2504.08066).
   Earlier complete research pipelines and the experimental-tree alternative.
6. **MLE-STAR: Machine Learning Engineering Agent via Search and
   Targeted Refinement**, Nam et al. (2025):
   [paper](https://arxiv.org/abs/2506.15692).
   The related lineage of targeted, ablation-guided engineering.
7. **Towards an AI co-scientist**, Gottweis et al. (2025):
   [original paper](https://arxiv.org/abs/2502.18864v1), revised as
   [Accelerating scientific discovery with Co-Scientist](https://arxiv.org/abs/2502.18864)
   in 2026.
   Compare hypothesis improvement with ScientistTwo's experiment-driven
   method improvement.
8. **Towards Execution-Grounded Automated AI Research**, Si et al. (2026):
   [paper](https://arxiv.org/abs/2601.14525).
   Compare population search grounded in actual training results with the
   broader evidence requirements of a paper-producing system.
