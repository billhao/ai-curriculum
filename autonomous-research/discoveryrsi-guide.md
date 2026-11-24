# DiscoveryRSI and Learning from Hypothesis Development

How a research agent turns a tentative idea into a testable plan, traces the
contributions behind its revisions, and uses them to train the next agent.

## Background

*DiscoveryRSI: Recursively Learning to Discover Valuable Hypotheses* credits
Jianyuan Zhong (PhAI Labs and The Chinese University of Hong Kong), Chenwei
Wu (University of Michigan), and Zhaochen Yu (National University of
Singapore) as equal first authors. arXiv records the first submission on
16 October 2026 and v2 on 20 October UTC. The v2 cover prints 21 October
and includes Jiajun Fan, who is absent from the abstract-page author list.
Page and figure references below follow the 40-page v2 PDF.
[arXiv record](https://arxiv.org/abs/2610.23839v2) ·
[Local v2 PDF](discoveryrsi-2610.23839v2.pdf)

DiscoveryRSI extends two lines of work. AI Co-Scientist (Gottweis et al.,
Google and collaborators, 2025) uses a tournament to refine and rank
hypotheses. AI Scientist-v2 (Yamada et al., Sakana AI, UBC, Vector Institute,
and Oxford, 2025) searches over runnable experiment implementations.
DiscoveryRSI versions the claim and its investigation plan together, then
uses documented contributions to teach future agents. The relation to
Darwin Gödel Machine is persistent improvement across runs. DGM's archive
stores agent implementations, while this paper's research graph stores
versions of scientific hypotheses (v2 §§1, 8, pp. 2–4, 19).
[AI Co-Scientist](https://arxiv.org/abs/2502.18864) ·
[AI Scientist-v2](https://arxiv.org/html/2504.08066v1) ·
[Darwin Gödel Machine](https://arxiv.org/abs/2505.22954)

Read this after the [execution-grounded research guide](execution-grounded-auto-research-guide.md).
Si et al. use actual GPU experiment outcomes to evaluate ideas. DiscoveryRSI
asks what an agent can learn earlier, while the proposed tests are still
being shaped. That makes it relevant to your RSI agenda's question of what
each possible experimental result would teach before spending compute.
[Execution-grounded research](https://arxiv.org/abs/2601.14525) ·
[Your research agenda](_rsi-research-agenda.md)

## 1. Specify what the experiment would teach

A plausible claim can leave a scientist with no way to choose the next
experiment. DiscoveryRSI defines a hypothesis as $H=(c,\pi,d,a)$ relative
to documented evidence $E$ and constraints $B$. It is **valuable** if
settling the claim could change an important conclusion or decision. It is
**testable** if the experiment fits the constraints and its possible
observations can strengthen, weaken, or bound the claim. A null result can
be valuable if it changes what you pursue (v2 §2, pp. 5–7).

For an illustrative GPT-2 124M study on your H800, suppose profiler traces
suggest that AdamW update overhead matters for short runs.

| Component | What the agent would commit to |
|---|---|
| Claim $c$ | Fused AdamW cuts time to the same validation loss by at least 5%. |
| Plan $\pi$ | Compare fused and unfused AdamW at a fixed held-out loss target. Match data order, seeds, optimizer settings, and evaluation cadence. |
| Decision rule $d$ | Require the lower bound of a paired interval for time reduction to exceed 5%, with fixed-token loss equivalent within a prespecified tolerance. |
| Alternatives $a$ | Measurement warmup, uneven evaluation overhead, or different optimization trajectories could explain the timing gap. |

A mean reduction from 32 to 30 minutes is 6.25%. If repeated paired runs
put its interval at 2–10%, the 5% claim is still inconclusive. If fixed-token
loss is stable and profiler traces isolate the optimizer, the experiment can
still narrow where to look next. Those numbers illustrate the decision rule.

## 2. Search over versions and preserve the work behind them

The **Lead** chooses a direction, parent versions, and an operation.
**Generation** and **Evolution** inspect evidence and publish versions.
An *evolve* step develops a direction, *repair* fixes a defect, *crossover*
combines contributions from several parents, and *restart* opens a new
question. A restart can reuse prior context. Only actual revisions or
combinations create ancestry edges. Crossover makes the research tree a
directed acyclic graph (v2 §3.1, pp. 6–8, Appendix A).

```text
Lead picks a direction and operation
               |
               v
Generation / Evolution publish a hypothesis version + evidence
               |
               +--> Blinded Critique: which proposal is better?
               +--> Progress: what changed from the actual parent?
               +--> Compiler: package the selected computational plan
               |
               v
Linked records feed weighted SFT and harness diagnosis
               |
               v
Development checks select a model and harness for fresh investigations
```

For example, a research path might go from *try fused AdamW* to *compare
at matched tokens* to *control for evaluation cadence and inspect the
optimizer time*. The path tracks how the scientific question changed. A
**trajectory** records one agent invocation and its tool interactions.
One node can depend on several trajectories. Evidence links, artifact
paths and hashes, assignments, and publication records let the learner
recover those links independently of ancestry (v2 Figure 2, Appendix A).

The **Compiler** packages selected plans as Harbor tasks with inputs, an
environment, required outputs, and checks. For the AdamW example a verifier
could inspect whether the runs were correctly paired and produced loss and
timing logs. Readiness checks concern the package. Actual outputs supply
the measurements for the statistical decision, and independent reproduction
would strengthen the claim. Unavailable inputs remain explicit dependencies.
The paper shows how to compile plans and reports quality gains at the
proposal stage. Its conclusion leaves broader solver execution and scientific
validation for future work (v2 §§3.2, 9, Appendix D.3).
[Harbor task format](https://docs.harborframework.com/tasks/overview)

## 3. Turn a good revision into weighted SFT

### Rate the hypothesis

Critique compares proposals under compatible data and review conditions.
TrueSkill assigns each hypothesis $h$ a latent rubric preference $z_h$.
Comparison performance adds independent Gaussian noise of standard
deviation $\beta$. The probability that $a$ wins is given by Eq. (2).

$$
P(a\succ b\mid z_a,z_b)=
\Phi\left(\frac{z_a-z_b-\varepsilon}{\sqrt{2}\beta}\right).
$$

$\Phi$ is the standard normal CDF and $\varepsilon$ is the draw margin.
In this model the performance difference inside $[-\varepsilon,\varepsilon]$
produces a tie. An abstention supplies no likelihood update. Approximate
Bayesian updates yield a marginal mean $\mu_h$ and uncertainty $\sigma_h$.
One win moves the winner's mean up and the loser's down, with the amount
depending on incoming uncertainty. These beliefs are about preferences
under the critique rubric. Evidence for the claim itself comes from its
investigation (v2 §3.3 and Appendix B.1, Eqs. 8–13).

### Allocate credit to documented contributions

Rank $s_h=\mu_h-2\sigma_h$ within each compatible comparison stratum.
The rank $r_h$ is the fraction strictly below $s_h$ plus half the fraction
tied. Effective comparison count $n_h$ and residual uncertainty damp the
outcome weight. A valid, compared hypothesis without a marginal-repetition
judgment has eligibility $v_h=1$. Recorded substantive progress or necessary
repair sets $p_h=1$ (v2 Appendix B.2, Eqs. 14–17).

$$
\rho_h^{\mathrm{cov}}=\frac{n_h}{n_h+2},
\qquad \rho_h^{\mathrm{unc}}=\frac{1}{1+\sigma_h/8.3333}.
$$

$$
q_h=v_h\min\left(2,\ 0.5+1.5r_h\rho_h^{\mathrm{cov}}
\rho_h^{\mathrm{unc}}+0.25p_h\right).
$$

After exact deduplication the method limits each local ancestry family's
total $q$ to 2 by proportional scaling. It then reserves 20% of a node's
remaining credit for verified coordination and 80% for verified production
or support. Each pool splits its share equally among eligible invocations.
Missing shares stay unallocated. A global budget of 2 caps allocated credit
for related hypotheses across snapshots. An invocation $i$ then receives
$w_i=\min(2,\sum_h b_{ih})$ over its final allocated shares $b_{ih}$.
These caps curb repeated credit when versions are linked to the same
question. Unlinked paraphrases can still receive credit. Matching a draft,
artifact, or assignment establishes provenance. The 20/80 split encodes
assumed responsibility for the work
(v2 Appendix B.3–B.7, Eqs. 18–23).

**Worked example.** Suppose the repaired timing plan has $r_h=0.75$,
$n_h=4$, $\sigma_h=\sigma_{\mathrm{ref}}/2$, and $v_h=p_h=1$. Both dampers
are $2/3$, so its raw credit is $0.5+1.5(0.75)(2/3)^2+0.25=1.25$.
If two versions in the local family both have credit 1.25, the 2-point cap
scales each to 1.00. For a version with one documented Lead assignment,
one Generation draft, and one Evolution contribution, the shares are 0.20,
0.40, and 0.40. An invocation linked only by ancestry earns zero for this
version. These are illustrative allocations.

### Weight the masked next-token loss

Within a domain $d$, the objective assigns equal shares of its mass
$\eta_d$ to the shared-memory groups in $G_d$ with retained targets. The
domain masses sum to 1. Within group $g_i$, trajectory $i$ receives a
share proportional to $w_i$. Groups capture dependencies such as resumed
sessions and shared campaign memory. Scientific-family caps apply before
this balancing. The trajectory coefficients are fixed across batches
(v2 Appendix B.5, Eq. 21).

$$
\omega_i=\frac{\eta_{d_i}}{|G_{d_i}|}
\frac{w_i}{\sum_{k:\ d_k=d_i,\ g_k=g_i} w_k}.
$$

The update is offline SFT. Extend the loss mask from your Dolly or SlimOrca
loop to retain assistant text and tool-call arguments supported by a
documented contribution. Observations and known failed actions can condition
later targets with zero direct loss. For the eligible token set $U_i$,
Eq. (4) averages within the original trajectory before multiplying by
$\omega_i$.

$$
\mathcal{L}(\theta)=-\sum_i\frac{\omega_i}{|U_i|}
\sum_{u\in U_i}\log\pi_\theta(a_{iu}\mid x_{iu}).
$$

Here $x_{iu}$ is the context available at the target token $a_{iu}$. If
$\omega_i=0.10$, a 100-target history gives each token coefficient 0.001.
At 400 targets each coefficient is 0.00025. Both histories contribute
0.10 times their own mean loss. The implemented uniform positive token
credit of 0.5 cancels in the mean (v2 Appendix C.2–C.4, Eq. 24).

The authors report full-parameter language-model fine-tuning from
Qwen3.8-27B at a 262,144-token window. The funnel is **15,792 collected →
13,270 structurally reviewable → 13,269 integrity-passing → 2,432 attributable
→ 2,397 retained**. The final 35 exclusions had developer messages that the
native training template could not represent. The retained set contains
634 Lead, 861 Generation, and 902 Evolution invocations. Critique, Progress,
and Compiler supply records and receive zero direct supervision. Only
**6,177,605 of 291,121,000 sequence tokens (2.12%)** receive loss. Training
uses one epoch, 150 Adam steps,
learning rate $5\times10^{-6}$, global batch 16, and 64 GPUs in BF16
(v2 §6, Appendix C, Table 4).

Splitting seven oversized histories brings the total to 2,404 windows.
Targets are counted once, repeated context is masked, and split windows
retain one trajectory denominator. Exact gradient equivalence also requires
identical context at every target. Bounded overlap can omit earlier context
in practice (v2 Appendix C.5, Proposition 2).

## 4. Update the workflow and try fresh questions

**Harness replay** joins decisions, artifacts, and execution records to
diagnose failures such as rediscovering the same result or losing an output
between agents. The harness here is research coordination, context, and
tools layered over Codex or Claude Code. A candidate edit could make the
Lead check a predecessor's timing artifacts before commissioning a repeat.
Recorded continuations support replay. An unobserved tool outcome requires
a new run from compatible state. Compare edits to the incumbent with the
checkpoint, inputs, and resource limits matched (v2 §3.4, Appendix A.4).

**Rediscovery** is the next run on fresh questions using the accepted
checkpoint and workflow. Algorithm 1 defines the development comparison
over four pairs formed from the incumbent and candidate model and harness. Acceptance
considers quality, exploration breadth, reliability, and cost. The selected
pair collects the next batch of experience. Final evaluation uses separate
test families. The protocol supports isolating effects, but §7 publishes
aggregate round comparisons. The four matched cell results needed to
attribute the improvement to specific updates are unavailable in v2
(v2 §§3.4–3.5, pp. 9–10, Appendix A.4).

## 5. Read the results by what was measured

DFM-Bench distinguishes five objects. A **hypothesis** gets a profile of
proposal quality. A **revision** gets a judgment of progress, repair,
repetition, or insufficient evidence against its parent, with score change
tracked separately. A **path** follows one longest route per terminal node
with ancestors and assesses cumulative inquiry. A **run** assesses the
full graph and effort allocation. A **contribution family** records a
distinct result or advance and the maturity of its evidence. Delivery stages
from proposal to
independent review are recorded separately (v2 §4 and Appendix D).

### Automated grading improved most on test design

Across 241 research questions in four life-science domains, GPT-5.6 Sol
graded base, round 1, and round 2 hypotheses at **4.86, 5.18, and 5.39**
out of 10. The protocol uses three reviewer prompts from the same model
family. Each axis takes the median of at least two known scores. An overall
score requires all three reviews, majority eligibility, and all eight
axis medians. The weights are fixed rubric choices. Independent human
calibration of the automated rubric is pending (v2 §6, Figure 6, Appendix D).

| Axis | Weight | Base | Round 1 | Round 2 |
|---|---:|---:|---:|---:|
| Importance | 20% | 6.48 | 6.21 | 6.04 |
| Novelty | 20% | 4.87 | 4.94 | 4.58 |
| Grounding | 15% | 4.24 | 4.97 | 4.56 |
| Falsifiability | 10% | 4.00 | 5.24 | 6.20 |
| Data resolvability | 10% | 3.73 | 4.33 | 4.99 |
| Feasibility | 10% | 4.79 | 4.21 | 5.87 |
| Alternative explanations | 10% | 4.44 | 5.64 | 5.98 |
| Actionability | 5% | 5.20 | 5.30 | 5.47 |

Falsifiability contributes about $2.20\times0.10=0.22$ points to the
overall increase. Alternatives add 0.154 points, and lower importance
and novelty subtract 0.146 combined. The measured gains concentrate in
writing a discriminating, feasible test. They taper from +0.32 points in
round 1 to +0.21 in round 2. Figure 6 marks significant comparisons with
the base using permutation tests with false-discovery correction.

Across six 20-episode campaigns per variant, evolve or repair rises from
93/120 to 108/120 to 110/120 episodes. Evidential discipline along paths
gains 1.25 on its 0–4 axis. Cumulative progress (+0.07), stopping (+0.03),
and budget use (+0.08) shift little. More refinement describes a change in
search behavior. Its value depends on which questions receive that effort
(v2 §7.2, Figure 7, Appendix D.4).

### Humans saw a mixed profile and more useful revisions

The study randomized 292 anonymous proposals from six dataset groups for
27 reviewers with relevant backgrounds under three senior supervisors.
They judged proposals individually, with model identity, prior versions,
and automated scores hidden. The overall research-value scale is 1–5 and
the six criterion ratings use 1–4 (v2 §5, Appendix E).

| Model | Mean change per assessed revision | Improved / same / worse |
|---|---:|---:|
| DiscoveryRSI-27B | +0.07 | 41% / 47% / 12% |
| GPT-5.6 Sol | +0.03 | 31% / 37% / 32% |
| GPT-6 Luna | -0.01 | 25% / 44% / 31% |

DiscoveryRSI-27B scored higher than Luna on importance (3.10 vs 2.94) and
evidence-bounded claims (3.37 vs 3.16). Both scored 2.31 for advance over
known work. Luna scored higher on data fit, alternatives, and informative
outcomes. The revision figures join ratings for actual parent–child pairs.
Edges share ancestors, and per-model pair counts are unavailable in v2.
The comparison describes the assessed branches (v2 §7.3, Figures 8–9,
Appendix E.3).

### Later literature matches depend on retrieval coverage

A hit means an automated judge found a concrete comparator, design,
measurement, or testable gap in retrieved literature after the group's
recorded cutoff. The denominator includes **all generated versions**.
DiscoveryRSI recorded **74/1,176 (6.3%)** over two pooled runs, DeepSeek
14/257 (5.4%), GLM 1/36 (2.8%), the GPT group 59/2,341 (2.5%), and Claude
3/131 (2.3%) (v2 §7.4, Figure 10, Appendix F).

The 6.3% is $355/1176\times74/355$. Claude passed retrieval less often
(15/131), while positivity given retrieval was similar (3/15, or 20.0%,
versus 74/355, or 20.8%). Claude's median proposal was 41,134 characters.
Retrieval saw the first 6,000 and the judge the first 2,500, so provenance
notes could crowd out the scientific content. Using only titles as queries
raised Claude's retrieval count to 95/131. The diagnostic measured passage
through the retrieval gate. GPT-6 Luna supplied 145 successful audit
judgments from 149 sampled records, with labels propagated to 903 registered
versions across groups.
Operational cutoffs include proxies for some models. Correlated versions,
domain composition, and uneven coverage make the rates descriptive.
Evidence of adoption or influence would require following hypotheses
prospectively (v2 Appendix F, pp. 39–40).

### A battery campaign shows use in another domain

The same workflow yielded 351 hypotheses across 200 episodes. Of the 165
with complete quality scores, the mean was 5.61. The paper classifies
214/366 transitions as substantive progress and seven of 20 contribution
families as carrying diagnostic or research evidence. These show that the
workflow and rubric can be applied to another domain. The paper reports
one battery panel, so the change relative to a battery-domain baseline is
unknown (v2 §7.5, p. 18).

## 6. What to take into your research loop

In a nanoGPT campaign, save the comparison and decision rule before each
run, link the measured outcome and agent actions to the version, and keep
an inconclusive result visible when it redirects the next experiment. Your
speedrunning verifier can supply the performance signal once a study runs.
Rating a plan earlier can guide how to spend the GPU budget. The
Ideation–Execution Gap gives a concrete reason to track both signals, since
human rankings of plans changed after the ideas were implemented.
[Si et al., 2025](https://arxiv.org/abs/2506.20803v1)

To test learning across cycles, compare the four checkpoint/harness pairs
at the same budget on unseen **question families**, keeping revisions of
one hypothesis together. Score predeclared experimental outcomes and report
failures and uncertainty. A retrospective best-so-far curve can only rise
when scored against the same final snapshot, so it diagnoses the archive
and leaves transfer to fresh runs an open question (v2 Appendix D.3).
This would distinguish learning to design a clean test from learning to
find a faster GPT-2 recipe.

## Key sources

1. Start with Definition 1 and Figure 2, then read Algorithm 1 to follow
   the two updates. Appendices B–C explain the loss and why only 2.12% of
   sequence tokens are targets. Figures 6–10 and Appendices D–F give the
   interpretation of the results. [DiscoveryRSI v2](https://arxiv.org/pdf/2610.23839v2)
2. Read the comparison model in the original rating paper if you want to
   derive the Gaussian moment update used in Appendix B.1.
   [TrueSkill, Herbrich, Minka, and Graepel, 2006](https://proceedings.neurips.cc/paper/2006/hash/f44ee263952e65b3610b8ba51229d1f9-Abstract.html)
3. The authors release the harness, evaluation code, and round-2 model
   checkpoint. The private research traces and artifacts used for training
   are unavailable in the supplement (v2 Appendix C.4).
   [Harness](https://github.com/Gen-Verse/DiscoveryRSI) ·
   [DFM-Bench](https://github.com/Gen-Verse/DFM-Bench) ·
   [DiscoveryRSI-27B checkpoint](https://huggingface.co/Gen-Verse/DiscoveryRSI-27B)
