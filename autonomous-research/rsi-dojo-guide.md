# RSI-Dojo: Teaching an Agent to Run Post-Training

How executable research environments and filtered tool-use demonstrations teach
an LLM to select data, train smaller models, diagnose failures, and improve them.

## Background

**Paper:** [Training LLM Agents to Automate LLM Post-Training with
RSI-Dojo](https://arxiv.org/abs/2611.03698), Shuzheng Gao, Zhaojian Yu,
Penghao Yin, Yuling Shi, Kai Cai, Shilin He, and Michael R. Lyu, 2026.
The arXiv record lists v1 as submitted **29 October 2026**, despite its November
identifier. This guide uses the complete **12-page v1 PDF**, with references on
pp. 10–12 and no appendix. Page references below mean PDF pages.

The title page assigns Gao and Lyu to the Department of Computer Science and
Engineering, The Chinese University of Hong Kong; Yu, Cai, and He to ByteDance;
Yin to Tsinghua University; and Shi to Shanghai Jiao Tong University.

The closest research lineage is:

| Work | Authors / group | Year | Contribution |
|---|---|---|---|
| PostTrainBench | Ben Rank et al.; Tübingen-led | 2026 | Evaluate agents that train models |
| AutoTrainess | Zhaojian Yu et al. | 2026 | Encode training expertise in workflows |
| RSI-Dojo | Gao et al.; CUHK, ByteDance, Tsinghua, SJTU | 2026 | Learn training work from trajectories |

Read this after the
[execution-grounded research guide](execution-grounded-auto-research-guide.md).
You already know how an agent proposes a recipe and uses an experiment as
feedback. RSI-Dojo asks a complementary question: **can we put the ability to
execute that research workflow into an open model's weights?** It is especially
relevant to your SFT work because the outer learner is trained on examples of
doing SFT and related model-improvement work.

## 1. The problem: the agent needs to learn the whole job

When you fine-tuned GPT-2 124M, the training loop was only part of the work.
You also chose examples, matched their format to the model, handled EOS and
loss masking, examined failures, and decided whether another run was useful.
A code-capable agent can produce a plausible training script while still
failing at these surrounding decisions.

Earlier systems often supply expertise through the **harness**: the program,
instructions, tools, and workflow surrounding the model. AutoTrainess supplies
guidance using `AGENTS.md`, skills, hooks, and reprompting. RSI-Dojo keeps this
useful infrastructure, then trains the controller on successful demonstrations
of using it (§§2.2–3.3, pp. 3–5).

Two models play different roles:

1. The **controller** is the coding agent. Its outputs include shell commands,
   code edits, diagnoses, and decisions about the next experiment.
2. The **target** is the smaller pretrained model being improved. It receives
   the data and gradient updates selected by the controller.

The 35B controller's PostTrainBench score measures how well the **targets it
produces** perform. It is not the controller's own math accuracy.

```text
OUTER LEARNING: teach the controller

Research artifacts -> Executable tasks -> Teacher-agent sessions
                                              |
                                  Quality + outcome filters
                                              |
                                              v
                                    SFT controller weights

INNER WORK: the trained controller improves a target

Task + target weights -> Choose data -> Train -> Evaluate
                            ^                      |
                            +--- Diagnose failures-+
                                              |
                                              v
                                     Submit target weights
```

The outer procedure is supervised trajectory learning. The inner procedure
can use ordinary gradient-based model training. Direct numeric weight editing
from the earlier weight-space-programming discussion is a different mechanism.

## 2. RSI-Craft: turn a paper into an executable task

**RSI-Craft** constructs environments. **RSI-Dojo** is the resulting collection
used to gather demonstrations. Neither name refers to a new optimizer.

The input is a research artifact with a usable dataset and evaluation target,
or an existing InspectAI task. The output needs an accessible dataset,
dependencies, an executable scorer, and a common evaluation interface.
This converts literature into places where agents can practice, rather than
merely adding paper text to a prompt (§3.1, pp. 4–5).

### Construction funnel

| Stage | Count | What survives |
|---|---:|---|
| Candidate seeds | 547 | Dataset/task screening |
| Runnable initial environments | 315 | Access, dependencies, executable evaluation |
| Final environments | 281 | Reliability and useful difficulty separation |

Paper sources include NeurIPS, ICML, ICLR, ACL, EMNLP, NAACL, and COLM from
the preceding three years. Keyword matches are screened by an agent: a paper
about evaluation is insufficient if it supplies no reusable dataset or
executable target. Dataset subsets with different metrics can become separate
tasks, so the counts do not imply 281 independent papers.

Tasks with fewer than **50 examples in total** are removed. Tasks above
**300 examples** are randomly downsampled to 300 for tractable evaluation.
This cap concerns the instantiated task data; it is not a claim that each
agent's eventual training dataset contains at most 300 examples.

The reliability screen rejects unstable evaluation, infrastructure-heavy
cases such as kernel tasks, and metrics that do not use generated output.
Unstable LLM-judge tasks can also fail this screen.

### Why compare a small and a stronger model?

The pipeline evaluates Qwen3-4B-Base and GPT-5-mini. It retains a task only
when their scores differ by at least **10 percentage points**, and removes
tasks where both exceed **80%** or both fall below **20%**.

For example, scores of 35% and 62% pass these stated difficulty tests. Scores
of 88% and 99% fail because both models are already strong. Scores of 5% and
18% fail because both are near the floor. These are teaching examples, not
individual datasets reported by the paper.

The intuition is to seek a task with room for improvement and a detectable
capability signal. However, a gap between these two models is not proof that
a small target can learn the task within five GPU-hours. It is a selection
heuristic. The resulting 281 tasks span ten domains; General QA contributes
31.7%, and Safety 14.6% (§3.1, Figure 3, p. 5).

## 3. Collect demonstrations worth imitating

A teacher rollout contains the complete attempt to improve a target model:
tool calls, code, experiment feedback, and revisions. Each collection session
is capped at **five hours**. Suitable training resources are downloaded
beforehand, and the custom web-search tool is disabled during collection
(§4.3, p. 6).

The authors first improve the demonstration harness. Repeated failures include
using tiny datasets, changing recipes without inspecting failed cases, getting
special tokens wrong, and finishing without a valid submitted model.
Additional guidance explicitly asks the teacher to diagnose, choose data based
on observed failures, check configuration, and submit the correct artifact.

This matters because a demonstration learner can copy bad process even when
the teacher is generally capable. A session that repeatedly launches tiny
LoRA jobs might teach activity without teaching useful experimentation.
The paper's EOS findings concern the observed configurations and workflows;
they do not establish that LoRA inherently causes termination failures.

### Process and outcome are separate filters

The retained demonstration must pass both gates (§3.3, p. 5):

1. **Behavior gate:** remove cheating and recurring operational mistakes,
   such as recipe changes without diagnosis or an invalid final submission.
2. **Outcome gate:** require a measurable improvement by the submitted model
   on the target task.

A controller can follow a tidy procedure and choose ineffective data. It can
also get a lucky gain through a poor procedure. Both cases are weak teaching
examples for a general training agent.

The final dataset contains **324 CLI sessions**, split into **1,240 training
trajectories** at context-compaction boundaries. These are about 3.83 segments
per retained session, not 1,240 independent successful experiments. Compaction
reasoning summaries are excluded. Custom `exec` and `wait` calls are converted
to standard tool-call representations where possible; unconvertible calls are
masked. An open-source implementation supports `apply_patch` at inference.

## 4. Train the controller with supervised examples

The paper fine-tunes Qwen3.5-9B and Qwen3.6-35B-A3B for **four epochs**.
The learning rate begins at $10^{-5}$ and falls to $2\times10^{-6}$
(§4.3, p. 6). It does not describe an outer GRPO or policy-gradient stage.

Conceptually, this is the same supervised objective you used for SFT, now
applied to agent actions in a tool-use transcript:

$$
\mathcal{L}(\theta)=
-\sum_{(x,y)\in D}\sum_t m_t
\log p_\theta(y_t\mid x,y_{<t}).
$$

Here $x$ is the available task/transcript context, $y_t$ is a supervised output
token, and $m_t$ indicates whether that token contributes to the loss.
This equation explains the standard SFT interpretation; the paper does not
print its own training-loss equation or specify a complete token-mask recipe.
Its explicit masking statement concerns custom calls that cannot be converted.

Contrast this with your GRPO implementation: the experiments' scores here
**select demonstrations before training**. They are not reported as rewards
used to compute advantages and update the controller online. Outcome filtering
is closer to rejection-sampled imitation learning.

The extra collection-time guidance is removed during evaluation. The standard
AutoTrainess harness and skills remain. This helps test whether learned
behavior transfers without the extra reminders; it does not isolate a bare
model operating without a harness (§4.2, p. 6).

## 5. Understand what the benchmark scores measure

### Full PostTrainBench: target benchmark performance

The full evaluation uses **four target base models × seven benchmarks = 28
settings**, with one GPU and a **ten-hour budget per setting**. The targets
are Qwen3-1.7B, Qwen3-4B, SmolLM3-3B, and Gemma3-4B. The seven benchmarks
cover AIME 2025, Arena-Hard, BFCL, GPQA Main, GSM8K, HealthBench, and
HumanEval (§4.1, pp. 5–6; Table 1 and Figure 5).

For target model $m$ and benchmark $b$, let $s_{m,b}$ be the submitted
model's result and $s^{\mathrm{base}}_{m,b}$ the starting model's result.
The protocol floors the submitted score at the base score:

$$
\tilde{s}_{m,b}=\max(s_{m,b},s^{\mathrm{base}}_{m,b}),
\qquad
S=\frac{1}{|M|}\sum_{m\in M}\sum_{b\in B}w_b\tilde{s}_{m,b}.
$$

$w_b$ are the benchmark-level weights. The result is an aggregate score,
not the percentage of the 28 settings solved. A regressed target does not
lower its term below the pretrained starting point. Completion counts and
improvement counts therefore supply useful information alongside the score.

### Lite: fraction of the base-to-instruct gap closed

Lite uses three target models and four benchmarks, giving **12 settings**,
with a **five-hour budget** each. Gemma is omitted; the benchmarks are AIME
2025, BFCL, GSM8K, and HumanEval. Its task score is:

$$
r_{m,b}=\mathrm{clamp}\left(
\frac{s^{\mathrm{trial}}_{m,b}-s^{\mathrm{base}}_{m,b}}
{s^{\mathrm{instruct}}_{m,b}-s^{\mathrm{base}}_{m,b}},0,1\right).
$$

For an illustrative task with base accuracy 20%, official instruct accuracy
60%, and submitted accuracy 40%, the normalized reward is
$(40-20)/(60-20)=0.5$. A 70% submission still receives 1 because the score
is capped. Invalid strategies receive zero under the protocol.

Thus **53.29 on Lite means an average normalized score of 0.5329**. It does
not mean 53.29% raw accuracy or 53.29% of tasks solved. Full and Lite scores
have different definitions and cannot be compared as if they shared a scale.

## 6. What changed after learning?

The most informative comparisons use the authors' own evaluation setting
(Table 1, p. 7):

| Controller | Full score | Lite score |
|---|---:|---:|
| Qwen3.5-9B | 7.72 | 0.56 |
| RSI-Dojo-9B | 16.46 | 15.71 |
| Qwen3.6-35B-A3B | 13.08 | 5.80 |
| RSI-Dojo-35B-A3B | 29.95 | 53.29 |
| GPT-5.5 | 27.80 | 52.96 |
| GLM-5.2 | 29.52 | 56.37 |
| DeepSeek-V4.1-Flash | 36.36 | 62.92 |

The full-score gains are **8.74 points** for 9B and **16.87 points** for
35B-A3B. The larger trained controller is competitive with GPT-5.5 and
GLM-5.2 on this particular job. DeepSeek remains ahead in both columns.

### A large part of the gain is becoming a reliable operator

| Controller | Any valid training | Final target improved | Effective / all iterations |
|---|---:|---:|---:|
| Qwen3.5-9B | 10/28 | 1/28 | 2/13 |
| RSI-Dojo-9B | 24/28 | 15/28 | 29/247 |
| Qwen3.6-35B-A3B | 15/28 | 11/28 | 17/68 |
| RSI-Dojo-35B-A3B | 28/28 | 26/28 | 61/175 |

For the larger model, valid-training coverage goes from 15/28 to 28/28,
and improved final targets from 11/28 to 26/28 (Table 2, pp. 7–8).
The effective-iteration fraction rises from 25.00% to 34.86%.
This is evidence of learning to conduct experiments, not just generating
better prose about training.

For 9B, the effective fraction actually falls from 15.38% to 11.74%, despite
more successful tasks and a better score. Learning lets it run many more
experiments, but does not improve every notion of efficiency. None of these
iteration ratios directly measures GPU cost, token cost, or dollar efficiency.

## 7. Why the two filters matter

Table 3 (p. 8) compares training the 35B controller with different filters:

| Demonstration filtering | Lite score | Improved targets | Any valid training |
|---|---:|---:|---:|
| Behavior only | 30.66 | 7/12 | 12/12 |
| Outcome only | 26.13 | 8/12 | 10/12 |
| Both | 53.29 | 9/12 | 12/12 |

The combined filter gives a much larger aggregate gain than either alone.
Outcome-only improves more tasks than behavior-only while obtaining a lower
aggregate score: the size of each gain matters, not just whether it is positive.
The paper does not provide matched dataset sizes or multi-seed uncertainty
for these ablations, so the table supports the combined recipe without
establishing a precise causal contribution for each filtering rule.

Guidance also improves the demonstrations before filtering. In samples of
100 sessions per condition, retention increases from **42% to 56%**.
Mean iterations fall from **11.93 to 9.25**, while mean improving iterations
rise from **3.05 to 3.32**. EOS failures fall from 23 to 14, too-little-data
cases from 22 to 3, and missing final models from 4 to 0 (§5.2, Figure 4).

The design lesson is concrete: improve the process that generates your SFT
data, then filter the resulting examples. Filtering alone cannot reliably
recover behaviors the teacher almost never demonstrates.

## 8. What the trained agent actually does

The authors' GSM8K example shows data selection followed by increasingly
targeted continuation (§5.3, Figure 6, p. 9). RSI-Dojo selects MetaMathQA
and OpenMathInstruct-2, then adjusts the data and continues training at a
lower learning rate. Their broader trace analysis identifies a recurring order:

1. Make training, formatting, termination, and submission work.
2. Choose a useful data source and obtain a substantial first improvement.
3. Inspect failed cases, filter data, and continue training more selectively.
4. Return from cheap subset evaluation to the full permitted development
   evaluation when the subset stops discriminating candidates.

For your SlimOrca work, an analogous change would be moving from “try another
learning rate” to “which errors remain, which examples address them, and is
the small validation subset still representative?” The fourth step should
use the evaluation split permitted for iteration, with an independent final
test retained. It does not authorize training on held-out benchmark examples.

The learned strategy is uneven. Hard-math rollouts frequently fail to improve,
so the outcome filter removes them; this leaves little successful supervision
for AIME 2025. The combined-filter model obtains **0.00 normalized Lite AIME
score** in Table 3. Successful teacher-distillation strategies are absent from
its demonstrations, which the authors associate with weaker Arena-Hard
performance. They observe such strategies in some stronger competitors.

This is a selection problem: retaining only successful work can make easy
operational improvements abundant while erasing the difficult tasks where
better exploration is most needed. More environments alone do not guarantee
coverage of the behaviors required for the hardest targets.

## 9. Scope, limitations, and reproducibility

1. **The recursive loop is still open.** The trained controller improves
   smaller target models. The paper does not use that controller to train an
   improved successor of itself and repeat the process. Section 7 (p. 10)
   explicitly leaves that experiment to future work. This is a useful
   prerequisite for RSI, not evidence of sustained recursive acceleration.
2. **Benchmark parity has a narrow meaning.** The paper uses PostTrainBench
   **v1.1**. The official site already lists v1.2, with changes including BFCL
   removal and HumanEval fixes. The 29.95 figure should not be inserted into
   a current v1.2 leaderboard. Nor does matching a frontier model on this
   aggregate establish comparable general capability.
3. **Hardware and resource access differ from the official protocol.** The
   authors use an unnamed GPU weaker than the official benchmark GPU. Their
   self-hosted controllers lack the custom web-search tool, retain general
   network access, and receive predownloaded resources, including datasets
   found in public benchmark traces. Prefer their in-paper matched comparisons
   to cross-leaderboard rankings; even those preserve tool-access asymmetry
   (§4.2, p. 6).
4. **Replication details are incomplete.** The 12-page report does not name
   the trajectory teachers, exact GPU, complete training configuration, or
   precise rubric thresholds. It provides no independent-seed error bars or
   detailed environment-to-evaluation overlap audit. No official RSI-Dojo
   code, environment release, or weights were located in the paper links or
   author release material as of 6 November 2026. These are missing details,
   not evidence that leakage or invalid training occurred.

For a practical follow-up on your own training setup, the useful first study
would be controller learning on a small set of reproducible SFT tasks.
Keep the target family, GPU budget, resources, and harness fixed; split
evaluation tasks by dataset family; record valid runs, actual target gains,
cost, and failure diagnoses. Compare an unchanged controller, extra guidance,
trajectory SFT, and both. Keep unsuccessful attempts in the research archive
even when excluding them from the imitation dataset: otherwise you lose the
evidence needed to see the same selection bias affecting hard mathematics.

## Key Papers and Sources

1. Gao et al., [RSI-Dojo](https://arxiv.org/abs/2611.03698), 2026.
   Main source: construction pp. 4–5; training/evaluation pp. 5–6;
   results pp. 7–9; limitations p. 10.
   [Downloaded PDF](../output/rsi-dojo-2611.03698.pdf).
2. Rank et al., [PostTrainBench](https://arxiv.org/abs/2603.08640), 2026.
   The originating automated-post-training benchmark.
3. Yu et al., [AutoTrainess](https://arxiv.org/abs/2606.31551), 2026;
   [official repository](https://github.com/simple-agent-lab/AutoTrainess).
   The workflow and harness ancestor used for collection and evaluation.
4. PostTrainBench team,
   [benchmark site and version history](https://posttrainbench.com/?version=v1.1).
   Distinguishes v1.1 scores from the subsequent v1.2 protocol.
