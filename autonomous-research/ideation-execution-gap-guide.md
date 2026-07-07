# The Ideation–Execution Gap: Why LLM Research Ideas Lose Their Edge Once Executed

The field's most important negative result: when 43 experts each spent 100+ hours actually *building* a randomly-assigned research idea into a real 4-page paper, the apparent novelty advantage of LLM-generated ideas — significant at the pitch stage — not only vanished but *reversed*. The lesson is a measurement-validity failure you already know from reward modeling: idea-stage review scores are a cheap proxy for research value, and optimizing the proxy (novel-sounding pitches) diverges from the target (methods that actually beat baselines).

## Background

**Primary paper**: [The Ideation–Execution Gap: Execution Outcomes of LLM-Generated versus Human Research Ideas](https://arxiv.org/abs/2506.20803) (Chenglei Si, Tatsunori Hashimoto, Diyi Yang — Stanford SALT Lab / Tatsu Lab, Jun 2025). Data + code: [github.com/NoviScl/AI-Researcher](https://github.com/NoviScl/AI-Researcher). Pre-registered on OSF (`ckxtp`); Stanford IRB 74246.

This is the pay-off to a two-paper arc by the same authors, and it is best read as the *execution* half of a setup–punchline pair.

1. **Can LLMs Generate Novel Research Ideas?** (Si, Yang, Hashimoto — Stanford, Sep 2024, [arxiv 2409.04109](https://arxiv.org/abs/2409.04109), ICLR 2025). The setup. 100+ NLP researchers wrote ideas; 79 reviewers blindly scored human ideas vs. ideas from an LLM ideation agent (Claude-3.5-Sonnet + retrieval + over-generate-then-rerank). Headline: **LLM ideas were judged significantly more novel than expert ideas (5.64 vs 4.84 on a 1–10 scale, p<0.01)** and more exciting, while being **comparable-or-lower on feasibility (6.34 vs 6.61, n.s.)**. That single caveat — feasibility is the *one* axis where AI didn't win — is the crack this second paper drives a wedge into. Crucially, the authors pre-registered the execution study *then*, so this is a confirmatory test, not a post-hoc fishing trip.
2. **The Ideation–Execution Gap** (this paper, Jun 2025). The punchline. Take those same ideas, have experts execute them for real, re-review the executed projects blind, and compare before-vs-after scores.

Where the rest of this repo's autonomous-research guides sit relative to it:
- **[The AI Scientist](ai-scientist-guide.md)** (Sakana, [2408.06292](https://arxiv.org/abs/2408.06292)) and **[Robin](robin-guide.md)** / **[AI Co-Scientist](ai-co-scientist-guide.md)** are *generation* systems that emit ideas/papers and score them with LLM judges or BTL/Elo rankers. This paper is the empirical warning label: the idea-stage score those systems optimize is **not** a valid proxy for what the idea is worth once run.
- **[POPPER](popper-falsification-guide.md)** ([2502.09858](https://arxiv.org/abs/2502.09858)) is the validation-rigor counterpart. POPPER answers "how do I get a calibrated false-positive rate on a *hypothesis*?"; this paper answers "why can't I trust the *review score* of a research *idea*?" Together they make the same argument from two directions — generation is cheap and untrustworthy; rigor has to come from execution and calibrated verification, not from a reviewer's (or an LLM's) sense of novelty.
- The landscape guide lists this as **Section 2 #2** and as **Open Problem #1** ("Ideation novelty does not survive execution — treat any 'more novel than humans' headline as ideation-stage only"). See [landscape-2025-2026.md](landscape-2025-2026.md).

## What Problem Does This Study Solve?

Every "AI scientist" pipeline and every idea-generation benchmark evaluates ideas *at the proposal stage* — an LLM judge or a small human panel reads a one-page pitch and scores its novelty, excitement, and expected effectiveness. This is the only thing anyone measures because it is cheap: reading a pitch takes minutes; *executing* it takes a research quarter. The entire "LLMs can out-ideate humans" narrative rests on those proposal-stage scores.

But a good idea is not one that *sounds* novel — it is one that, once built, actually works: beats real baselines, survives ablations, reproduces. Nobody had checked whether proposal-stage scores predict that, because the experiment is brutally expensive. Prior "AI scientist" papers validated only 1–2 cherry-picked AI ideas through execution — far too few for a statistical claim.

```
 THE UNTESTED ASSUMPTION every idea-gen system relies on:

   ideation score  ──────────( assumed ∝ )──────────▶  executed research value
   (novelty/excitement of                              (does the method beat
    a 1-page pitch, ~30 min                              baselines after 100+ hrs
    to review)                                           of real implementation?)

 THIS PAPER measures the join directly, at scale (N=43 executed projects):

   AI pitch scores HIGH ─────▶ [ 43 experts × 100+ hrs ] ─────▶ scores LOW
   Human pitch scores lower ──▶ [ same execution + review ] ──▶ scores hold / rise
                                                                └─▶ RANK FLIP
```

The contribution is the first quantitative, adequately-powered execution study: enough real projects (N=43) to draw statistically significant conclusions about *post-execution* quality, and a within-idea before/after design that cancels out the enormous heterogeneity in idea quality.

## Key Terms

**Ideation evaluation (Study 1 / "before")**: blind review scores of the *idea proposal* — a structured 1-page pitch (problem, motivation, method, step-by-step plan). Metrics: Novelty, Excitement, Feasibility, Expected Effectiveness, Overall, each 1–10. Taken from the predecessor study.

**Execution evaluation (Study 2 / "after")**: blind review scores of the *executed project* — the 4-page ACL-format paper plus full codebase — by a fresh pool of 58 reviewers (181 reviews, 4–5 per project). Metrics: Novelty, Excitement, **Soundness**, **Effectiveness**, Overall (1–10), plus two control metrics: **Faithfulness** (did the paper stay true to the original idea outline?) and **Codebase Quality** (1–5). Feasibility is dropped (you no longer speculate about feasibility once it's built); Soundness is added.

**The ideation–execution gap**: for one idea, `gap = execution_score − ideation_score` on a shared metric (Novelty, Excitement, Effectiveness, Overall). A negative gap means the idea scored *worse* after being built. This within-idea difference is the paper's central quantity — it removes the "some ideas are just better than others" variance that swamps a direct human-vs-AI comparison.

**Condition**: whether an idea came from a **Human** expert (N=19 executed) or an **AI** agent (N=24 executed), spanning 7 NLP topics (bias, coding, safety, multilingual, factuality, math, uncertainty). Reviewers were blind to the source. Executors were randomly assigned an idea from their preferred topic — no self-selection of "good" ideas.

**Faithfulness / codebase controls**: the guardrail against the obvious confound "maybe AI ideas were just executed worse." They weren't — Faithfulness (6.48 human vs 6.42 AI, p=0.41) and Codebase Quality (3.58 vs 3.58, p=0.52) are statistically identical across conditions. Both kinds of idea were implemented equally faithfully and cleanly. The gap is about the *ideas*, not sloppy execution of the AI ones.

## The Reversal, Traced Number by Number

This is the whole paper in one table. Read each row left to right: AI starts ahead of Human (positive `AI−Hum`), and ends behind (negative `AI−Hum`). Every sign flips.

```
                 IDEATION (before)          EXECUTION (after)          GAP (after − before)
Metric           Human    AI    AI−Hum      Human    AI    AI−Hum      Human    AI     Δ(H−AI)   p(FDR)
──────────────────────────────────────────────────────────────────────────────────────────────────────
Novelty          4.912  5.778  +0.866       4.903  4.729  −0.174       −0.010  −1.049   1.039    .025 *
Excitement       4.404  5.653  +1.249       4.482  3.896  −0.586       +0.078  −1.760   1.835    .001 **
Effectiveness    4.833  6.003  +1.170       4.782  4.125  −0.657       −0.052  −1.879   1.827    .003 **
Overall          4.596  5.382  +0.786       3.968  3.406  −0.562       −0.628  −1.976   1.348    .004 **
──────────────────────────────────────────────────────────────────────────────────────────────────────
   (* p<0.05, ** p<0.01; Tables 4 & 5. N=19 human / 24 AI ideas; each idea's mean is one data point.)
```

Walk the novelty row, because novelty is the metric the whole "AI out-ideates humans" story was built on:

- **Before:** AI 5.778 vs Human 4.912 → **AI leads by +0.866** (≈ +0.87), significant. This is the [2409.04109](https://arxiv.org/abs/2409.04109) headline, reproduced on the executed subset.
- **After:** AI 4.729 vs Human 4.903 → **AI trails by −0.174** (≈ −0.17). The lead didn't just shrink; it crossed zero. **Rank flip.**
- **The gap that drives it:** the human idea's novelty barely moves (−0.010 — statistically a flat line). The AI idea's novelty drops **−1.049**. The difference in drops, 1.039 points, is significant at p=.025.

The same story repeats on all four metrics, and it is *sharper* on the ones that matter more than novelty:
- **Human ideas are stable.** Their gaps are −0.010, +0.078, −0.052, −0.628 — essentially zero on novelty/excitement/effectiveness, with only Overall dipping (executing anything into a rushed 4-page short paper costs a little).
- **AI ideas fall off a cliff.** Gaps of −1.049, −1.760, −1.879, −1.976 — down 1 to 2 full points on *every* axis.
- **Before, AI wins all four (all significant). After, Human wins all four (none significant).** The post-execution human lead isn't statistically significant at N=43 (idea heterogeneity is large), but the *direction* flipped on every metric, and the *gap* comparison is significant on every metric. As the abstract puts it: the execution scores "flip in rankings where human ideas score higher than LLM ideas."

Notice what this means numerically: **AI ideas are not penalized for being AI; they are penalized for being *inflated*.** They were scored ~1–2 points too high at the pitch stage relative to what they were worth. Human ideas were priced about right.

## Why the Scores Collapse: Reviewers See Different Things Before and After

The authors categorized every free-text review comment into 10 factors and measured how often each is mentioned at ideation vs. execution (Figure 3; percentages read off the bars, approximate):

```
Factor mentioned in review rationales        Ideation   Execution
────────────────────────────────────────────────────────────────
Novelty / Motivation                            ~82%       ~95%
Empirical Performance                            ~2%       ~92%   ← the chasm
Experiment Design                               ~25%       ~60%
Ablation / Analysis                             ~17%       ~43%
Baseline Comparison                             ~10%       ~40%
Feasibility / Resource                           ~0%       ~18%
Significance / Impact (speculative)             ~38%       ~12%   ← speculation drops out
Flaws of Method (speculative)                   ~33%       ~15%
────────────────────────────────────────────────────────────────
```

Two mechanisms, both intuitive once you see the numbers:

**1. Ideation review is speculation; execution review is measurement.** At the pitch stage there are no results, so reviewers *condition on the method working* and reward the promise. One reviewer, scoring the AI idea *"Conceptual Pivot Prompting for Bias Mitigation"* at ideation, literally wrote:

> "This is pretty hard to predict. Assuming the experiments are successful and thorough, it would be a solid paper worthy of acceptance at any conference. However, this is entirely dependent on how the experiments turn out. It is entirely possible that the proposed method is ineffective, and we don't learn anything substantial from it, in which case a paper might not even exist."

Empirical performance is mentioned in **~2%** of ideation reviews and **~92%** of execution reviews. That 90-point swing *is* the gap: the AI pitch was optimized to win the speculative game, and the speculative game stops being played the moment there are real numbers on the table.

**2. Execution imposes a rigor bar that pitches never face.** Once a method is run, reviewers see the missing baselines, the weak metrics, the cost, the failed comparisons — all invisible in a pitch. Real execution-stage complaints from the paper, each about an AI idea that had scored well at ideation:
- *"the method does not show marked improvements over a basic empathy prompting approach"* (Empathetic Cascading Networks)
- *"lacks comparison with previous work: the method is only compared with the simplest baselines despite well-acknowledged benchmarks"* (Contrastive Semantic Pivot Prompting)
- *"not using the same metrics as other works to compare the efficacy of this method"* (Temporal Bias Decay Simulation)
- *"the method is also very computationally expensive"* (Adaptive Contextual Pruning)

Could these drops just be executors mangling the AI ideas, or changing them beyond recognition? The paper closes that door three ways: (a) executors made only **2.9 (human) vs 3.1 (AI)** changes on average, all confined to *experiment details* (dataset, baseline, metric, hyper-parameters) — verified not to alter the core method; (b) the Faithfulness and Codebase controls are identical across conditions; (c) even after dropping the 6 AI ideas whose proposed human-eval was swapped for LLM-as-judge, the gaps are unchanged (AI gaps still −1.1 to −2.0, all significant). The ideas were built faithfully. They were just worth less than they looked.

## Worked Example: Same Reversal, One Idea at a Time

The aggregate reversal is a population effect, but Appendix H gives per-idea before/after scores, and the mechanism is vivid at the single-idea level. Two contrasting cases — one AI idea that cratered, one human idea that rose — from the paper's randomly-selected examples.

**AI idea (Uncertainty): "Differential Confidence Mapping" (DCM).** Pitch: LLMs are miscalibrated, so build a *confidence map* — generate contrastive question variants at different difficulty levels, elicit the model's relative confidence between them, embed those comparisons into a graph with node2vec, and use the query's position in that "confidence landscape" to correct its raw confidence. It reads like a genuinely clever, novel calibration method. The reviewers agreed — at the pitch stage.

```
Differential Confidence Mapping (AI, Uncertainty)     Ideation → Execution     Δ
──────────────────────────────────────────────────────────────────────────────────
Novelty                                                  6.0  →  6.3         +0.3   ← survives!
Excitement                                               6.0  →  3.5         −2.5
Effectiveness   (Expected-Effectiveness → measured)      5.0  →  2.3         −2.7
Overall                                                  5.3  →  2.3         −3.0
──────────────────────────────────────────────────────────────────────────────────
```

This is the whole paper compressed into one idea. **Novelty is essentially preserved (6.0 → 6.3)** — it *is* a novel-sounding method, and building it didn't make it less novel. But **Overall craters by 3 full points (5.3 → 2.3)** because, once executed, DCM simply **lost to trivial baselines**: plain temperature scaling had the lower calibration error (ECE 0.004 vs DCM's 0.006–0.019), and a plain ensemble had the better Brier score and accuracy (0.141 / 0.818 vs DCM's ~0.15 / ~0.81). The elaborate graph-embedding-plus-contrastive-prompting apparatus didn't beat a one-line logit scaler. The *proxy* the pitch was optimized for (novelty, excitement) held; the *thing you actually wanted* (a method that calibrates better than the baseline) never materialized.

**Human idea (Safety): "A Compound LLM System to Mimic Knowledge Unlearning."** Pitch: instead of fine-tuning to *forget* dangerous knowledge, wire up a compound system — an orchestrator routes each query, a responder answers benign ones, a deflector safely refuses ones probing the forbidden topic, and a filterer double-checks the output — so the model *pretends* to have unlearned, no weight edits.

```
A Compound LLM System to Mimic Unlearning (Human, Safety)   Ideation → Execution   Δ
──────────────────────────────────────────────────────────────────────────────────
Novelty                                                       5.5  →  6.0        +0.5
Excitement                                                    5.5  →  5.3        −0.2
Effectiveness                                                 4.0  →  7.0        +3.0   ← rose
Overall                                                       4.5  →  5.3        +0.8
──────────────────────────────────────────────────────────────────────────────────
```

Note it started *below* the DCM pitch on Overall (4.5 vs 5.3) — a less flashy pitch. But it went **up** on execution: built out on the WMDP unlearning benchmark it hit strong suppression (24.6% cyber / 26.3% bio / 27.2% chem accuracy, beating the fine-tuning-based RMU baseline) while retaining general utility (58.4% MMLU). Effectiveness leapt 4.0 → 7.0. A modest-sounding, feasible, grounded idea *under-promised and over-delivered* — the opposite failure mode from DCM.

**A necessary nuance — humans aren't magic.** The human idea *"Self-Improving Memory Ignites Mathematical Reasoning"* dropped 4.0 → 3.0 overall (its ideation feasibility was already a weak 3.5). Human ideas are *stable on average*, not individually bulletproof. The claim is distributional: AI pitches are systematically inflated by ~1–2 points; human pitches are priced roughly right.

## The Real Lesson: Ideation Score Is an Invalid Proxy (a Goodhart / Reward-Overoptimization Story)

Strip away the "AI scientist" framing and this is a measurement-validity result you already understand from post-training. The ideation reviewer is a **reward model**. It maps a cheap observable (a pitch) to a scalar. The thing you actually care about — executed research value — is expensive and unobservable at pitch time. The ideation score is a *proxy* for it.

```
Reward-model / DPO world (you know)              Research-ideation world (this paper)
──────────────────────────────────────────────────────────────────────────────────────
reward model  r_φ(prompt, response)              ideation reviewer  →  novelty/excitement
true human value / gold preference               executed value (beats baselines? reproduces?)
policy π optimized against r_φ                    LLM idea generator + LLM reranker, tuned to score well
reward hacking / over-optimization                novel-SOUNDING pitches that fail on execution
KL(π‖π_ref) ↑  →  true reward peaks then FALLS    AI pushes proxy score up; true value ends up BELOW human
Gao et al. reward-overopt scaling law             ideation↔execution correlation ≈ 0
early stopping / a better reward model            require EXECUTION or calibrated validation (POPPER)
──────────────────────────────────────────────────────────────────────────────────────
```

Two facts nail the analogy:

1. **The proxy is nearly uncorrelated with the target — and for the flashiest metric, anti-correlated.** Appendix D correlates each idea's ideation score with its execution score. For AI ideas, Overall r = −0.09, and **Excitement r = −0.32** (Spearman ρ = −0.39). Read that literally: *the more excited a reviewer was by an AI pitch, the worse it did once built.* An exciting AI pitch is a mild negative signal. This is exactly reward-model over-optimization — the proxy doesn't just plateau, it inverts, because the generator has learned to produce pitches that hit the proxy's buttons without carrying the underlying value.

2. **What the reviewers were rewarding is knowable — and it's the hackable part.** In the predecessor study, the *overall* ideation score correlated 0.73 with novelty and 0.85 with excitement, but **≈0 with feasibility** (r<0.1). Reviewers price a pitch on how novel and exciting it sounds, and ignore whether it will work. An LLM tuned (implicitly, via over-generate-and-rerank on the same kind of judge) to maximize that score will learn to maximize surface novelty — the definition of Goodhart's law, *"when a measure becomes a target, it ceases to be a good measure."* You saw the mechanical version of this when a DPO policy drifts from its reference to chase reward-model score and produces high-scoring degenerate text; here the "policy" is an idea generator and the degenerate output is a plausible-but-hollow research pitch.

The corrective is the same one you apply to DPO: don't trust the proxy past the region where it's calibrated. In research terms, that means **you cannot rank ideas by ideation score and act on the ranking** — you have to either execute them (expensive, this paper) or route them through calibrated validation ([POPPER](popper-falsification-guide.md)'s e-process gives a Type-I-error-controlled verdict on a hypothesis). The generators ([AI Scientist](ai-scientist-guide.md), Robin, co-scientist) are the policy; this paper is the measurement showing the policy has reward-hacked the ideation judge.

## A Statistics Subtlety Worth Copying: Three Ways to Slice the Same Data

The paper reports the human-vs-AI comparison three ways, and which one you use changes whether you see a significant effect. This is a clean lesson in matched-design variance reduction — the same reason paired tests beat unpaired ones.

```
Analysis (what is one data point?)          Property                              Result: AI vs Human
────────────────────────────────────────────────────────────────────────────────────────────────────
Table 3: each REVIEW  (N=181)               high power; ignores idea clustering    AI sig. LOWER on Exc/
                                            (pseudo-replication risk)               Eff/Sound/Overall (not Novelty)
Table 4: each IDEA, scores direct (N=43)    honest unit; idea heterogeneity        NO metric significant
                                            swamps the signal
Table 5: each IDEA, the GAP direct (N=43)   within-idea diff cancels the           AI drops sig. MORE on
                                            "some ideas are better" variance         ALL 4 metrics
────────────────────────────────────────────────────────────────────────────────────────────────────
```

Table 4 (direct human-vs-AI on executed scores) shows *nothing significant* — not because there's no effect, but because ideas vary so wildly (a great idea and a dud in the same condition) that 43 points can't resolve a ~0.5-point mean difference. Table 5 fixes this by measuring each idea *against itself* (before vs after): the huge between-idea variance cancels, and the AI-drops-more effect pops out at p<0.01 on excitement/effectiveness/overall. If you run before/after evals on your own checkpoints (SFT vs the base model, DPO vs SFT), this is the same move — compare paired deltas per prompt, not marginal means, when the per-item variance is large.

## Critiques and Limitations (stated by the authors)

- **Idea scope is narrow by construction.** All ideas were prompting-based NLP methods, deliberately scoped in the predecessor study to be executable in ~3 months on a modest compute/data budget. Ideas requiring large-scale pretraining, new datasets, or long research programs are out of frame — and those are arguably where human research taste matters *most*, so the gap could be even larger there, or different.
- **Modest N.** 43 executed projects is a lot of human-quarters (≈4,400 hours of expert labor) but still statistically modest; the significant signal lives in the paired *gap* analysis, not the direct comparison. Per-condition, granular breakdowns are underpowered.
- **Human executors, not autonomous agents.** Execution was done by paid expert humans precisely to guarantee faithful, high-quality implementation (the thing autonomous agents can't yet promise — see the AI Scientist's "silent wrong implementations" failure mode). So this measures *idea quality under competent execution*, not what today's autonomous pipelines would produce end-to-end.
- **The gap is about *current* LLM ideation, not a law of nature.** It measures Claude-3.5-Sonnet-era ideation agents scored by human reviewers who reward novelty. Both halves can move: better idea generators, or reviewers/judges trained to weight executed value.
- **"Novelty is hard to judge" cuts both ways.** The authors flag (in the predecessor) that even expert novelty judgments are noisy; some of the ideation inflation is reviewer error, not just generator gaming.

## Practical Takeaways

- **Never rank ideas (yours or an LLM's) by proposal-stage novelty/excitement and act on the ranking.** It's a proxy with ≈0 correlation to executed value, and *negative* correlation for AI excitement. If you're building an idea-generation loop, the ideation score is a reward model you should assume is being hacked.
- **The honest signal requires grounding in outcomes.** Either execute (expensive) or wrap generation in calibrated validation. This is the exact hand-off to [POPPER](popper-falsification-guide.md): generate cheaply, then adjudicate the survivors with a Type-I-controlled test before spending real resources. Generation without a rigor gate is where the field over-produces.
- **Under-promise, over-deliver beats the reverse.** The idea that *rose* on execution (compound unlearning) had a modest, feasible, grounded pitch and strong real results; the one that cratered (DCM) had a flashy, novel-sounding pitch that lost to temperature scaling. When you evaluate a research direction, weight the boring feasibility axis the reviewers ignore.
- **Use paired before/after deltas for noisy evals.** When per-item variance is large (research ideas, or your own SFT/DPO checkpoints across a heterogeneous prompt set), compare matched deltas per item, not marginal means — the paper's Table-4-vs-Table-5 contrast is the textbook demonstration.
- **The forward direction is execution-as-reward.** The authors point at exactly the fix an RL person would reach for: train a **proxy reward model that predicts execution outcomes** from a pitch (Wen et al., *Predicting Empirical AI Research Outcomes with LLMs*, [2506.00794](https://arxiv.org/abs/2506.00794)), and/or close the loop by using **empirical execution results as the reward signal** for idea generation (RL / GRPO-style, or evolutionary selection on real outcomes) instead of ideation scores. That replaces the hackable proxy with the target — the same reason you'd prefer a verifiable reward over a learned one in RLHF.

## Key Papers

1. Si, Hashimoto, Yang. *The Ideation–Execution Gap: Execution Outcomes of LLM-Generated versus Human Research Ideas.* arXiv 2506.20803 (2025). https://arxiv.org/abs/2506.20803 — this guide.
2. Si, Yang, Hashimoto. *Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers.* ICLR 2025, arXiv 2409.04109 (2024). https://arxiv.org/abs/2409.04109 — the setup; the "AI ideas are more novel (5.64 vs 4.84)" study that this paper's execution phase pays off.
3. Wen, Si, Chen, He, Feng. *Predicting Empirical AI Research Outcomes with LLMs.* arXiv 2506.00794 (2025). https://arxiv.org/abs/2506.00794 — the proposed fix: a proxy model that predicts execution outcomes, cited in Future Work.
4. Simsek, de Vaan, van de Rijt. *Do grant proposal texts matter for funding decisions? A field experiment.* Scientometrics 129:2521–2532 (2024). — external evidence that even expert evaluators struggle to judge proposals before outcomes; motivates the "evaluation is hard" framing.
5. Lu, Lu, Lange, Foerster, Clune, Ha. *The AI Scientist.* arXiv 2408.06292 (2024). https://arxiv.org/abs/2408.06292 — the generation-side system whose idea-stage scores this paper warns against trusting. See [ai-scientist-guide.md](ai-scientist-guide.md).
6. Huang, Jin, R. Li, M. Li, Candès, Leskovec. *Automated Hypothesis Validation with Agentic Sequential Falsifications (POPPER).* arXiv 2502.09858 (2025). https://arxiv.org/abs/2502.09858 — the calibrated-validation counterpart. See [popper-falsification-guide.md](popper-falsification-guide.md).
