# Vals RSI Index: Measuring Autonomous LLM R&D

The Vals RSI Index tests whether frontier agents can autonomously improve
compression systems, language-model training, evaluation harnesses, and
post-trained models under fixed research budgets. It is best understood as an
external benchmark of capabilities needed for recursive self-improvement, not
as evidence that a completed recursive loop already exists.

**Primary sources and versions.** The [live benchmark
page](https://www.vals.ai/benchmarks/rsi_index) was accessed on **20 November
2026** and says **Updated 11/19/2026**. Current rankings come from that page.
The [October 2026 preprint](vals-rsi-index-preprint.pdf) supplies the scoring
equations and detailed protocol. The live page credits Oliver Chen, Geby Jaff,
Langston Nashold, and Rayan Krishnan; the preprint uses the collective byline
**Vals AI Team**.

## Background

1. **Speculations Concerning the First Ultraintelligent Machine** (I. J. Good,
   1965) — defines an ultraintelligent machine as one that can surpass all
   intellectual activities of any human. Because machine design is itself an
   intellectual activity, Good argues that such a machine could design better
   machines, producing an intelligence explosion. [Original
   excerpt](https://vtechworks.lib.vt.edu/items/5085379d-b24c-424e-8861-e70a47b4b2fb)
2. **Gödel Machines** (Jürgen Schmidhuber, 2003) — formalizes self-rewriting
   systems that accept a modification only with a proof that it improves the
   system's utility. This supplies a precise but computationally difficult
   ideal, not a practical LLM agent. [arXiv:cs/0309048](https://arxiv.org/abs/cs/0309048)
3. **The AI Scientist** (Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob
   Foerster, Jeff Clune, David Ha; Sakana AI, 2024) — connects idea generation,
   code execution, paper writing, and automated review in one research
   pipeline. Its output is a research artifact, not a researcher reused in the
   next generation. [arXiv:2408.06292](https://arxiv.org/abs/2408.06292)
4. **RE-Bench** (Hjalmar Wijk et al., METR, 2024) — compares agents with human
   experts on seven ML research-engineering environments under matched
   eight-hour budgets. It establishes the value of time-matched execution
   baselines. [arXiv:2411.15114](https://arxiv.org/abs/2411.15114)
5. **Darwin Gödel Machine** (Jenny Zhang, Shengran Hu, Cong Lu, Robert Lange,
   Jeff Clune; UBC and collaborators, 2025) — replaces proof-gated rewriting
   with an archive of coding-agent variants selected by empirical performance.
   Accepted agent changes persist into later exploration.
   [arXiv:2505.22954](https://arxiv.org/abs/2505.22954)
6. **The Automated LLM Speedrunning Benchmark** (Bingchen Zhao et al., Meta
   FAIR and collaborators, 2025) — turns 19 transitions between known nanoGPT
   training records into reproduction tasks. Its Fraction of Speedup Recovered
   asks whether an agent can recover a published improvement, not discover an
   unknown one. [arXiv:2506.22419](https://arxiv.org/abs/2506.22419)
7. **MLS-Bench** (Bohan Lyu et al., 2026) — evaluates whether invented ML
   methods generalize across controlled settings and scales. RSI-Index adopts
   its idea of calibrating a transformed score so that a reference lands at
   0.5. [arXiv:2605.08678](https://arxiv.org/abs/2605.08678)
8. **Vals RSI Index** (Vals AI, launched August 2026; preprint October 2026) —
   provides an external, cross-provider comparison of long autonomous AI R&D
   campaigns against fixed references. [Launch
   post](https://www.vals.ai/blogs/series-a)
9. **Recursive Self-Improvement of AI Research Agents / AIDE²** (Dhruv
   Srikanth, Bingchen Zhao, Dixing Xu, Yuxiang Wu, Zhengyao Jiang; Weco AI,
   2026) — modifies the research agent's own code and carries accepted changes
   forward. This is closer to testing improvement of the researcher itself.
   [arXiv:2609.26457](https://arxiv.org/abs/2609.26457)
10. **Flex-RSI** (Flex-RSI Team, 2026) — explicitly separates an executor that
    improves a task solution from an improver that modifies the executor and
    itself. Validation gates accepted changes before hidden-test scoring.
    [arXiv:2611.08946](https://arxiv.org/abs/2611.08946)

## What Problem Does the RSI Index Solve?

Claims about AI R&D are often produced inside model developers' own labs, on
private tasks chosen by those labs. That makes progress difficult to compare
across providers and releases. Vals instead gives every evaluated system the
same four task specifications, fixed resource budgets, and independent artifact
verification.

This changes the question from **"did a lab report that its model helps its
researchers?"** to **"what measurable research artifact did the agent produce
when another organization controlled the test?"**

It also separates two ideas that are often conflated:

```text
VALS RSI INDEX: FOUR BOUNDED RESEARCH CAMPAIGNS

Fixed researcher + native agent harness
              |
              +--> compressor          --> verifier --> task score s_1
              +--> LM training program --> verifier --> task score s_2
              +--> judge harness       --> verifier --> task score s_3
              +--> tuned checkpoint    --> verifier --> task score s_4
                                                   |
                                                   v
                                        index = mean of four scores

RECURSIVE SUCCESSION: WHAT THIS DOES NOT SHOW

Generation 0          Generation 1          Generation 2
researcher v0 -------> researcher v1 -------> researcher v2
                       improved child        improved grandchild

The child must become the researcher that designs the next child.
```

The preprint states this limitation directly: no agent trains a model that is
then asked to train a successor. Improving a compressor, training recipe, or
finance model is therefore evidence about **research capability**, not a
demonstration of Good's full feedback loop.

## Key Terms

**Agent campaign**: One attempt by one model and harness on one task under a
fixed time and compute budget. The October preprint reports one campaign per
system-task pair.

**Harness**: The software surrounding the model: prompting, tools, memory,
context management, code execution, and session control. A result belongs to a
model-plus-harness system, not an isolated model.

**Artifact**: The final submitted object: a compressor, training program, judge
harness, or checkpoint with generation settings. The verifier grades this
artifact rather than the agent's own summary.

**BPB**: Bits per byte. For next-byte prediction, this is mean negative
log-base-2 probability per byte. For compression, it includes all submitted
bytes needed to reconstruct the input.

**Baseline $b_t$**: The raw starting result that anchors normalized score 0.

**Reference $r_t$**: A selected published, human, or frontier-model result that
anchors score 0.5. The four references are not a uniform human-SOTA standard.

**Ceiling $u_t$**: The task's theoretical-best result, anchoring score 1.

**Research-run variance**: Variation between complete independent agent
campaigns. Repeatedly grading one frozen artifact does not measure this.

## Benchmark Protocol

Each task supplies a Docker environment, instructions, a time limit, and a
verifier. Agents can modify their environments, run experiments, and document
findings, but must submit exactly one final artifact. Models run at their
highest supported reasoning effort.

| Model family | Harness used by Vals |
|---|---|
| Claude, GLM | Claude Code |
| GPT | Codex |
| Gemini | Antigravity |
| Kimi | Kimi Code |
| Qwen | Qwen Code |
| Muse | Muse Code |
| Grok | Grok Build |

This is an ecological test of systems as their providers expect them to be
used. It is not a controlled model-only comparison: Claude and GPT, for
example, differ in both weights and harness.

Agents maintain a **Marimo experiment ledger**. If an agent exits before time
expires, a supervisor relaunches a session with the original objective,
remaining budget, and preserved workspace. This detail matters: in the
preprint, most agents exited early at least once, and the supervisor launched a
median of **10 further sessions per run**.

### Do not mix the two snapshots

| Snapshot | Coverage | Reported leader |
|---|---|---|
| October preprint | 19 systems, 8 providers | Opus 5.5, index 0.373 |
| Live page, 20 Nov 2026 | 29 configurations | Fable 5.2, index 37.88% |

The older preprint remains the source for equations and analysis. The dated
live page supersedes its rankings.

## The Four Research Tasks

### 1. Compression: build a self-contained lossless compressor

The corpus is **enwik8**, the first $10^8$ bytes of enwik9. The agent receives
**24 hours, one H100, and eight CPU cores**, without internet. A clean sandbox
reruns the submission and requires byte-for-byte reconstruction.

The score counts the self-extracting archive and every required runtime
dependency. This prevents an apparent compression win obtained by hiding a
large decoder outside the submission.

The NNCP reference illustrates the accounting. Mahoney's Large Text
Compression Benchmark lists 14,915,298 compressed bytes and a 628,955-byte
decompressor:

```text
NNCP total = 14,915,298 + 628,955 = 15,544,253 bytes
BPB = 15,544,253 * 8 / 100,000,000 = 1.24354024
```

NNCP's own page reports 1.19 BPB for compressed data alone. Vals uses the
larger, packaging-inclusive figure.

### 2. LM Training: improve a small byte-level GPT

The starter is a **four-layer, width-256 GPT with a 2,048-byte context**. The
enwik9 split is **800 MB train / 100 MB development / 100 MB hidden test**. The
agent has **24 hours and one H100-80GB**, with no internet or package
installation; the live protocol reserves the final hour for packaging.

After the campaign, Vals freezes the submitted program and retrains it from
scratch for **30 minutes on each of three hidden seeds**. Each run evaluates
next-byte BPB at **32,768 held-out positions** with 2,048 bytes of context; the
reported result is the median across seeds.

```text
def verify_lm(program, hidden_seeds, hidden_windows):
    seed_results = []
    for seed in hidden_seeds:                 # three seeds
        model = train_from_scratch(program, seed, minutes=30)
        bpb = evaluate_next_byte_bpb(
            model, hidden_windows[seed],      # 32,768 positions
            context_bytes=2048,
        )
        seed_results.append(bpb)
    return median(seed_results)
```

This is close in spirit to your nanoGPT work, but the agent controls the whole
research process rather than only the optimizer: architecture, data handling,
training schedule, implementation, candidate selection, and final packaging.

The 0.85 BPB reference is Al-Rfou et al.'s **235M-parameter, 64-layer T64**
character transformer. It is not a perfectly controlled head-to-head baseline.
The preprint notes that T64 trained on the first 900 MB and reports its result
on the final 50 MB, whereas Vals uses an 800/100/100 MB split. The often-quoted
**175 hours on a TPU v2** belongs to Al-Rfou et al.'s text8 evaluation, not a
reported enwik9-specific runtime.

### 3. Harness Engineering: repair a web-app judge

The agent improves the evaluation scaffold around a fixed **Gemini 3.5
Flash-Lite** judge. It receives **12 hours**, no GPU, and hosted judge calls.
Development feedback is aggregate agreement on **12 apps**, without per-item
labels.

The frozen harness is then evaluated three times on **24 held-out apps with
205 scored binary substeps**. The raw metric is agreement with gold labels.
The agent may change prompts, browser or DOM evidence collection, persistent
state, deterministic checks, and adjudication logic.

The normalized baseline is 50% agreement, but two preprint diagnostics show
why this should not be read as a practical floor: an always-pass judge scores
**59.5%**, and a source-only judge without visual browsing scores **59.3%**.
The 93.6% reference is the highest agreement between two human annotators, not
a uniform frontier-model reference.

### 4. Post-training: improve a finance agent without its private judge

The parent is **Qwen3.6-35B-A3B**, evaluated on Finance Agent v2. The agent gets
**30 hours on 16 RTX PRO 6000 96GB GPUs**, internet access, **$100** of external
inference credit, and only **27 public examples**. It does not receive the
private judge prompt or the API keys used by the evaluated model's tools.

The submission must contain a checkpoint derived from the supplied parent and
its generation settings. Vals measures the mean score across **three runs on
126 held-out questions**.

Finance Agent v2's upstream primary score is a dealbreaker-gated,
severity-weighted average of per-check scores. Failing a required dealbreaker
zeros the whole question; other checks permit partial credit. Thus the raw
number is not simply exact-answer accuracy. The reference is **Muse Spark
1.2's 59.42% on this 126-question subset**, not its 60.60% score on the full
450-question public leaderboard.

## Reference-Anchored Log Scoring

For task $t$, put every raw result on a lower-is-better scale. Compression and
LM Training already use BPB. For agreement or finance accuracy $a$, use error
$x=1-a$. Let $b_t$, $r_t$, and $u_t$ denote the baseline, reference, and
ceiling on that same scale.

The preprint first computes the fraction of the logarithmic gap closed:

$$
f_t(x)=\mathrm{clip}\left(
\frac{\log(b_t/x)}{\log(b_t/u_t)},0,1\right).
$$

It then applies a task-specific exponent that forces the reference to score
exactly 0.5:

$$
s_t(x)=f_t(x)^{\gamma_t},
\qquad
\gamma_t=\frac{\log 0.5}{\log f_t(r_t)}.
$$

Finally,

$$
\mathrm{RSI}=\frac{1}{4}\sum_{t=1}^{4}s_t(x_t).
$$

Invalid results and results below the starting baseline receive zero. The
exponents are approximately **1.123, 1.265, 1.077, and 0.316**, in task order.
A score of 0.5 therefore means the same reference-relative milestone on every
task, but not the same percentage improvement in raw units.

### The anchors on one table

| Task | 0: baseline | 0.5: reference | 0.6: milestone | 1: ceiling |
|---|---:|---:|---:|---:|
| Compression | 2.92 BPB | 1.2435 BPB | 1.070 BPB | 0.6 BPB |
| LM Training | 1.3699 BPB | 0.85 BPB | 0.789 BPB | 0.6 BPB |
| Harness | 50% | 93.6% | 95.6% | 99% |
| Post-training | 35.38% | 59.42% | 71.8% | 99% |

The 0.6 milestone corresponds to 14% smaller compression than the reference,
7% lower LM BPB, about 32% fewer harness disagreements, and about 30% fewer
post-training errors. Vals chose these margins by analogy with historical
jumps such as AlexNet and LSTM-to-Transformer progress. They are calibration
choices, not measured probabilities of an intelligence explosion.

### Numerical walkthrough: Fable 5.2

Take the LM result $x=0.8232$, with $b=1.3699$, $r=0.85$, and $u=0.6$:

$$
f(r)=\frac{\log(1.3699/0.85)}{\log(1.3699/0.6)}
=0.578098,
$$

$$
\gamma=\frac{\log 0.5}{\log 0.578098}=1.264840,
\qquad
f(x)=0.616905.
$$

Therefore $s(x)=0.616905^{1.264840}=0.542824$, displayed as **0.54**.

Repeating this for all four live Fable 5.2 results gives:

| Task | Displayed raw result | $f_t(x)$ | Reconstructed score |
|---|---:|---:|---:|
| Compression | 1.2505 BPB | 0.535917 | 0.496309 |
| LM Training | 0.8232 BPB | 0.616905 | 0.542824 |
| Harness | 78.70% agreement | 0.218127 | 0.193913 |
| Post-training | 40.13% score | 0.018315 | 0.282397 |

The mean is $(0.496309+0.542824+0.193913+0.282397)/4=0.378861$.
The model page reports **37.88%**; the tiny difference comes from reconstructing
with displayed rounded raw values.

Notice that Fable's compression result is slightly **worse** than the 1.2435
reference but rounds to a 0.50 task score. Always inspect raw results alongside
normalized scores.

## Current Results

These rankings are a snapshot of the live page and linked model pages on
**20 November 2026**, not the October preprint's 19-system table.

| Rank | System | RSI Index |
|---:|---|---:|
| 1 | Claude Fable 5.2 | 37.88% |
| 2 | Claude Opus 5.5 | 37.31% |
| 3 | Claude Fable 5.1 | 36.09% |
| 4 | Gemini 4 Neon | 34.40% |
| 5 | Claude Opus 5 | 33.02% |
| 6 | Gemini 4 Argon | 30.55% |
| 7 | GPT-6.2 Astra | 29.01% |
| 8 | GPT-6 Sol | 28.11% |

The best raw result on each task tells a clearer story than the aggregate:

| Task | Current leader | Raw result | Reference | Score |
|---|---|---:|---:|---:|
| Compression | Fable 5.2 | 1.2505 BPB | 1.2435 BPB | 0.50 |
| LM Training | Fable 5.2 | 0.8232 BPB | 0.85 BPB | 0.54 |
| Harness | GPT-6 Sol | 81.46% | 93.6% | 0.23 |
| Post-training | Opus 5.5 | 44.53% | 59.42% | 0.35 |

Four systems beat the selected LM reference: Fable 5.2, Opus 5.5, Gemini 4
Neon, and Gemini 4 Carbon. The leaders remain behind the selected references
on the other three tasks. Every listed system improved the 1.3699 BPB LM
starter, but improvement over a weak starter is not the same as advancing a
research frontier.

## What the Research Campaigns Reveal

### 1. Persistent execution matters

The preprint's Opus 4.5 compression campaign initially ended with **22 hours
remaining** at **3.1469 BPB**, worse than the 2.92 baseline. Supervisor
relaunches with the preserved workspace eventually reached **1.9280 BPB**.
A single agent session would have missed the improvement entirely.

This makes session persistence part of the measured system. It also means a
reimplementation without the supervisor is not the same protocol.

### 2. More experiments do not imply better research

In LM Training, GPT-5.6 Sol ran **114 experiments** and reached 0.9475 BPB;
Opus 5 ran **76** and reached 0.8983 BPB. In Harness Engineering, Kimi K3
evaluated **494 development apps** and reached 68.29%, while Opus 4.5 evaluated
**129** and reached 78.37%.

The bottleneck is not merely the number of trials. Agents must choose
high-leverage directions, build trustworthy measurements, and promote a
candidate only when evidence justifies it.

### 3. The strongest submissions recombine known methods

Opus 5.5's LM submission combined a byte-level Transformer with multi-byte
pattern embeddings and an auxiliary head predicting one further byte. Both
ingredients have prior literature. The preprint's overall assessment is that
agents primarily combined and tuned established techniques rather than
inventing a new general-purpose method.

That is still useful research engineering. It is weaker evidence for novel
algorithm discovery than a headline such as "beats the reference" suggests.

### 4. Post-training exposes weak feedback and brittle selection

In the October table, **15 of 19 systems scored zero** on post-training. Some
submitted a regressed checkpoint and were floored to zero; others returned the
parent. Every trained checkpoint used supervised fine-tuning, nearly all with
LoRA or QLoRA. Only Grok 4.6 used DPO, and its result fell below the parent.
No agent trained on external finance datasets.

Successful agents diagnosed parent failures and built targeted data instead of
running generic SFT. This resembles your SFT work: a clean training loop does
not help if the examples reinforce the wrong behavior or the proxy evaluation
cannot detect a regression.

### 5. Sequential gains did not compound cleanly

Vals gave Opus 5 a second, legal-domain post-training phase after its finance
improvement. The preprint reports:

| Checkpoint | Legal criteria passed | Finance score |
|---|---:|---:|
| Parent | 64.57% | 35.38% |
| After finance post-training | 72.64% | 42.20% |
| After finance, then legal | 67.29% | 38.55% |

The second phase reduced both scores relative to the first checkpoint, despite
having more in-domain legal examples than in the finance phase. This is target
model training, not researcher recursion, but it demonstrates why a sequence
of locally accepted improvements need not produce cumulative capability.

## RSI Index vs Alternatives

| Benchmark | Improved object | Researcher reused? | Scale | Main metric |
|---|---|---|---|---|
| Vals RSI Index | Four artifacts | No | 4 tasks; 12–30 h | Reference-anchored mean |
| LLM Speedrunning | Training script | No | 19 transitions | Fraction of speedup recovered |
| PostTrainBench | Target checkpoint | No | 28 settings; 10 h each | Target benchmark aggregate |
| AIDE² | Research-agent code | Yes, accepted rewrites | 100 nodes; 8 days | Concealed task grade |
| Flex-RSI | Solution and harness | Yes, accepted updates | 24 tasks; 20 main | Hidden test and rank score |

The speedrunning benchmark is a **reproduction floor**: can an agent implement
a known nanoGPT improvement? Vals asks a more open-ended question but still
uses known task families and reference results.

AIDE² closes a harness-level loop. Its main run accepted **seven rewrites**, at
steps 2, 6, 28, 39, 47, 63, and 85. However, its ignition test did not establish
that the improved researcher was better at driving another improvement run:
AIDE_47 and the human-developed outer agent had mean endpoints of **0.780 and
0.782**, respectively, with substantial seed variance.

Flex-RSI provides the cleanest controlled contrast between artifact improvement
and harness improvement. Enabling its improver raised hidden-test performance
in **six of seven matched pairs**, by **2.0–40.0 points**, with one tie. But
transferring a stronger model's evolved harness to a weaker model helped on
only **two of four tasks**. Recursion is neither automatic nor freely
transferable.

[RSI-Dojo](rsi-dojo-guide.md) differs again: it trains a controller's weights on
filtered post-training demonstrations. Its improved target models do not become
the next controller, so learning research behavior and closing a recursive
succession loop remain separate claims.

## Limitations and Failure Modes

1. **No completed recursive loop.** The researcher is fixed at the start of
   each campaign. Improved artifacts are not recursively reused as researchers.
2. **No campaign confidence intervals.** Each preprint entry is one research
   run per task. Repeated grading of a frozen LM, harness, or checkpoint
   estimates evaluation noise, not run-to-run research variability.
3. **Model and harness are confounded.** Native harnesses and reasoning
   settings differ. The benchmark does not claim that native harnesses beat a
   standardized minimal one.
4. **References are heterogeneous.** NNCP, human-human agreement, T64, and Muse
   Spark answer different calibration questions. T64 also uses a different
   enwik9 split, so crossing 0.5 is not a universal human-SOTA claim.
5. **Contamination cannot be excluded.** Compression and LM Training use public
   corpora and published methods. Recalling published work may help an agent
   approach a reference; a large gain beyond it is harder to explain this way.
6. **Four tasks omit important research abilities.** Data curation, kernel
   optimization, continued post-training, and frontier-scale infrastructure
   are not comprehensively covered. The arithmetic mean also lets strength on
   one task compensate for weakness on another.
7. **Small-scale transfer is untested.** A 30-minute byte-level training recipe
   need not transfer to frontier model development. The benchmark supplies
   evidence about bounded research campaigns, not frontier training automation.

### The 2027 extrapolation is conditional

The preprint fits index score against release date for developers with at least
five evaluated releases. It estimates slopes of **0.30 index points per year
for Anthropic** and **0.17 for OpenAI**. Its Anthropic fit crosses 0.5 in March
2027 and 0.6 in July 2027, with a **March 2027–May 2028** 95% interval for the
latter.

These are linear extrapolations over seven Anthropic releases, not predictions
derived from a scaling law of recursive improvement. The interval is conditional
on the linear trend continuing and says nothing by itself about transfer to
frontier-scale R&D.

## Practical Considerations

Use this benchmark when you want to compare complete research systems, inspect
long-horizon behavior, or test artifact verifiers. Do not use it as a model-only
leaderboard, a probability estimate for an intelligence explosion, or evidence
that improved models can already produce improved successors.

For a meaningful replication on your **single H800**, LM Training is the best
starting point:

1. Keep the 800/100/100 MB split, starter architecture, final-seed protocol,
   and 30-minute final retrains fixed.
2. Preserve a strong incumbent and screen candidates with short development
   runs, but require full-duration confirmation before promotion.
3. Repeat the **entire research campaign**, not only final artifact grading.
   Three or more independent campaigns would begin to expose agent variance.
4. Keep a held-out verifier outside the agent's control and record every change
   to the candidate, evaluator, and selection decision.
5. Treat your numbers as a replication on different hardware, not as
   leaderboard-equivalent H100 results.

For a judge-harness analogue, interleave unchanged controls with candidates so
that evaluator noise is visible. For post-training, retain the parent and a
prompt-only parent as controls, and re-evaluate earlier capabilities after every
sequential phase.

Exact replication of the other tasks is harder: post-training uses 16 GPUs and
private finance questions, while Harness Engineering uses private apps and a
hosted judge. Public Compression and LM Training are more accessible, but the
preprint still cannot rule out contamination from published solutions.

## Key Papers and Primary Sources

1. **Vals RSI Index** — [live methodology and
   leaderboard](https://www.vals.ai/benchmarks/rsi_index); [October 2026
   preprint](https://www.alphaxiv.org/abs/2609.rsi-index-autonomous-llm-rd.pdf).
2. I. J. Good, **Speculations Concerning the First Ultraintelligent Machine**,
   1965 — [primary excerpt](https://vtechworks.lib.vt.edu/items/5085379d-b24c-424e-8861-e70a47b4b2fb).
3. Jürgen Schmidhuber, **Gödel Machines**, 2003 —
   [arXiv:cs/0309048](https://arxiv.org/abs/cs/0309048).
4. Chris Lu et al., **The AI Scientist**, 2024 —
   [arXiv:2408.06292](https://arxiv.org/abs/2408.06292).
5. Hjalmar Wijk et al., **RE-Bench**, 2024 —
   [arXiv:2411.15114](https://arxiv.org/abs/2411.15114).
6. Jenny Zhang et al., **Darwin Gödel Machine**, 2025 —
   [arXiv:2505.22954](https://arxiv.org/abs/2505.22954).
7. Bingchen Zhao et al., **The Automated LLM Speedrunning Benchmark**, 2025 —
   [arXiv:2506.22419](https://arxiv.org/abs/2506.22419).
8. Bohan Lyu et al., **MLS-Bench**, 2026 —
   [arXiv:2605.08678](https://arxiv.org/abs/2605.08678).
9. Ben Rank et al., **PostTrainBench**, 2026 —
   [arXiv:2603.08640](https://arxiv.org/abs/2603.08640).
10. Dhruv Srikanth et al., **Recursive Self-Improvement of AI Research Agents**,
    2026 — [arXiv:2609.26457](https://arxiv.org/abs/2609.26457).
11. Shuzheng Gao et al., **RSI-Dojo**, 2026 —
    [arXiv:2611.03698](https://arxiv.org/abs/2611.03698).
12. Flex-RSI Team, **Flex-RSI**, 2026 —
    [arXiv:2611.08946](https://arxiv.org/abs/2611.08946).
13. Rami Al-Rfou et al., **Character-Level Language Modeling with Deeper
    Self-Attention**, 2019 —
    [arXiv:1808.04444](https://arxiv.org/abs/1808.04444).
14. Fabrice Bellard, **NNCP**, and Matt Mahoney, **Large Text Compression
    Benchmark** — [NNCP](https://bellard.org/nncp/) and [compression
    results](https://www.mattmahoney.net/dc/text.html).
15. Vals AI, **Finance Agent v2** — [methodology and
    leaderboard](https://www.vals.ai/benchmarks/fabv2).
