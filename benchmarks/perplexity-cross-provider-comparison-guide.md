# Perplexity and Cross-Entropy: What Can Be Compared Across Models?

A technical guide to what perplexity measures, when loss comparisons are meaningful across providers, and why even a careful compression score is not a general-intelligence score.

OpenAI's November 22, 2026 essay [Perplexity and Intelligence](https://openai.com/index/perplexity-and-intelligence/) ([saved original](perplexity-and-intelligence.html)) is the motivation. Jakub Pachocki reports base-model log-perplexity on a fixed held-out snapshot of OpenAI's internal codebase and proposes perplexity, together with other measurements, as a basis for shared safety limits. The essay supplies no displayed equation and does not disclose the logarithm base, tokenizer, context protocol, aggregation rule, or uncertainty intervals. Those omissions determine what an outside comparison can mean.

## Background

1. **A Mathematical Theory of Communication** (Claude Shannon, Bell Labs, 1948) — established entropy and source coding. A predictive distribution can drive a code whose expected length approaches its negative log-probability. [DOI](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x)

2. **A Neural Probabilistic Language Model** (Bengio, Ducharme, Vincent, Jauvin, 2003) — learned a distributed next-word predictor with maximum likelihood, making held-out likelihood central to neural language modeling. [JMLR](https://www.jmlr.org/papers/v3/bengio03a.html)

3. **Language Models are Unsupervised Multitask Learners** (Radford et al., OpenAI, 2019) — GPT-2 improved zero-shot language-modeling results on seven of eight benchmarks, while the report used perplexity, BPB, BPC, and accuracy in their respective native units. This was not one universally normalized leaderboard. [OpenAI report](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) · [saved PDF](language-models-unsupervised-multitask-learners.pdf)

4. **Scaling Laws for Neural Language Models** (Kaplan et al., OpenAI, 2020) — showed smooth power-law trends in held-out cross-entropy with model size, dataset size, and compute. Absolute values still belong to a particular data and evaluation protocol. [arXiv 2001.08361](https://arxiv.org/abs/2001.08361)

5. **You Should Evaluate Your Language Model on Marginal Likelihood over Tokenisations** (Cao and Rimell, 2021) — showed that scoring only one canonical segmentation can misrepresent a model; the probability of a string can instead marginalize over its tokenizations. [ACL Anthology](https://aclanthology.org/2021.emnlp-main.161/)

6. **Language Modeling Is Compression** (Delétang et al., 2023) — developed the practical equivalence between predictive models and lossless compressors. Compression is therefore a meaningful measurement in its own right, without already being a definition of intelligence. [arXiv 2309.10668](https://arxiv.org/abs/2309.10668)

7. **Paloma** (Magnusson et al., 2023; NeurIPS 2024) — built a 546-domain English-and-code fit benchmark and popularized careful BPB evaluation. Paloma explicitly calls BPB only a partial solution when vocabularies differ; its controlled baselines fix one vocabulary. [arXiv 2312.10523](https://arxiv.org/abs/2312.10523)

8. **Compression Represents Intelligence Linearly** (Huang, Zhang, Shan, He, 2024) — found strong average correlations between BPC and selected downstream scores for base models, while also showing that corpus-task alignment and contamination matter. [arXiv 2404.09937](https://arxiv.org/abs/2404.09937)

9. **Tokenizer Myths and Where To Find Them** (Rylan Schaeffer, Sanmi Koyejo, 2026 preprint) — demonstrated that BPB still depends on the tokenizer-model pair, including before training. Byte normalization supplies common units, not tokenizer independence. [arXiv 2611.04215](https://arxiv.org/abs/2611.04215)

10. **Perplexity and Intelligence** (Jakub Pachocki, OpenAI, 2026) — proposed a shared pretraining metric and disclosed OpenAI's internal loss trajectory, while explicitly listing cross-lab datasets, reasoning, actionable tasks, and the unknown entropy floor as challenges. [Official essay](https://openai.com/index/perplexity-and-intelligence/)

## What Problem Does This Metric Solve?

A next-token model produces a probability for the reference continuation. Negative log-likelihood turns that into a differentiable training objective and an additive held-out score:

```text
Fixed held-out text + fixed preceding context
                     │
                     ▼
   p_theta(actual next token | preceding text)
                     │
               -log p_theta
                     │
          sum and normalize → cross-entropy
                     │
                  exp → perplexity
```

This is attractive for three reasons:

1. **It uses the native pretraining objective.** There is no need to turn a capability into a multiple-choice question or elicit it with a clever prompt.
2. **It is continuous.** Unlike exact-match accuracy, a small probability improvement remains visible before an answer flips from wrong to right.
3. **It has a compression interpretation.** With arithmetic coding, ideal target code length is the total negative base-2 log-likelihood, plus implementation overhead.

The target is nevertheless narrow: average predictive fit to a chosen distribution under a chosen conditioning protocol. It is not a direct measurement of what a user can get the system to do.

### What OpenAI actually reported

The essay's prose gives rounded values of about 0.99 for GPT-3.5, 0.91 for GPT-4, below 0.8 for recent cybersecurity and mathematics capabilities, about 0.76 for breakthrough open-mathematics results, and 0.68–0.70 as a forecast window. The embedded chart contains more precise values:

| Date | Base model | Log-PPL | Chart association |
|---|---|---:|---|
| 2022-03 | GPT-3.5 | 0.986 | Fluent conversation |
| 2023-03 | GPT-4 | 0.913 | Thinking before answering helps |
| 2025-02 | GPT-4.5 | 0.876 | — |
| 2025-04 | GPT-4.1 / o3 | 0.902 | — |
| 2026-07 | GPT-5.6 Sol | 0.828 | Cybersecurity expert |
| 2026-09 | GPT-6 Astra | 0.783 | — |
| 2026-11 | GPT-6.2 Astra | 0.761 | Millennium Prize math annotation |
| 2026-11 | Separate internal model | 0.732 | — |

Three readings matter:

- These are **retrospective associations**, not laws that every model at 0.828 becomes a cybersecurity expert. Pachocki explicitly says the thresholds were not known in advance.
- The series is not monotone: GPT-4.1 / o3 has a higher reported loss than GPT-4.5. The chart is not a downstream-capability leaderboard.
- The 0.68–0.70 region is a forecast of unknown capabilities, not an observed result. The proposed pacing framework uses perplexity **and other measurements** and specifies no numerical stop line.

OpenAI also warns that internal-code prediction was a poor proxy for GPT-2 and GPT-3 general capability because of their training data. That footnote is the dataset problem in miniature.

## Key Terms

**Reference-text NLL**: The negative log-probability assigned to the actual supplied continuation at each position, under the actual preceding context. It is not the probability of whatever text the model chose to generate.

**Cross-entropy loss**: The average reference-text NLL per scored token. With natural logs its unit is nats per token; with base-2 logs it is bits per token.

**Perplexity (PPL)**: The exponential of mean log-loss. Equivalently, it is the geometric mean of inverse conditional token probabilities. Intuitively, it is an effective branching factor, not an accuracy percentage.

**Log-perplexity**: $\log(\mathrm{PPL})$. This equals cross-entropy only when the logarithm base and the loss unit agree. Raw perplexity and cross-entropy are not numerically interchangeable.

**Teacher forcing**: Scoring every actual target token while feeding the model the actual preceding reference tokens, rather than feeding back the model's own generations.

**Canonical tokenization**: The one segmentation produced by a tokenizer's normal encoder. A string may have other segmentations through the same vocabulary.

**BPB / BPC**: Total negative log-likelihood in bits divided by original UTF-8 bytes or by a precisely defined character count, respectively.

**Micro versus macro averaging**: Micro sums losses and denominators before dividing. Macro computes a score per document or domain and then averages those scores.

**Base versus post-trained checkpoint**: A base checkpoint measures the pretraining distribution. SFT, preference optimization, and RL produce a different distribution; a hosted assistant can additionally include hidden prompts, reasoning, and tools.

### Comparability at a glance

| Evaluation situation | Defensible quantity | What the result means |
|---|---|---|
| Same data, tokenizer, context, and checkpoint class | Token NLL and token PPL | Direct predictive-fit comparison |
| Different tokenizers, identical raw targets and context | BPB, or BPC with one character convention | Compression by each tokenizer-model pair |
| Different datasets, windows, prompts, or checkpoint classes | None of the absolute losses as a like-for-like score | A controlled re-evaluation is required |
| Public OpenAI and Anthropic chat APIs only | No direct corpus PPL/BPB | Reference NLL is unavailable |

## The Mathematics: Loss, Perplexity, and Log Bases

For documents $d$, target tokens $x_{d,t}$, and $N=\sum_d T_d$ scored tokens, define total NLL in natural-log units:

$$
S_e=-\sum_{d=1}^{D}\sum_{t=1}^{T_d}
\ln p_\theta(x_{d,t}\mid x_{d,<t})
$$

Then

$$
H_e=\frac{S_e}{N},\qquad \mathrm{PPL}=e^{H_e}
$$

$$
H_2=\frac{S_e}{N\ln 2},\qquad \mathrm{PPL}=2^{H_2}
$$

Here $H_e$ is nats per token and $H_2$ is bits per token. The conversions are

$$
H_2=\frac{H_e}{\ln 2},\qquad H_e=H_2\ln 2
$$

The chain rule connects sequence probability and token scoring:

$$
p_\theta(x_{1:T})=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t})
$$

$$
\mathrm{PPL}(x_{1:T})=p_\theta(x_{1:T})^{-1/T}
$$

### Numerical walkthrough

Suppose four target tokens receive probabilities $0.5$, $0.25$, $0.125$, and $0.5$:

```text
sequence probability = 0.5 × 0.25 × 0.125 × 0.5 = 1/128
S_e                  = -ln(1/128) = 4.852 nats
H_e                  = 4.852 / 4   = 1.213 nats/token
H_2                  = (1 + 2 + 3 + 1) / 4 = 1.75 bits/token
PPL                  = exp(1.213) = 2^1.75 = 3.364
```

The arithmetic mean of the four probabilities is irrelevant. Perplexity comes from the product of conditional probabilities, equivalently the mean of their logs.

### Why OpenAI's 0.761 is not automatically PPL 2.140

OpenAI calls its plotted quantity log-perplexity but does not state the base. The conversions are therefore conditional:

| Reported log-PPL | If natural log: $e^x$ | If base 2: $2^x$ |
|---:|---:|---:|
| 0.986 | 2.680 | 1.981 |
| 0.913 | 2.492 | 1.883 |
| 0.761 | 2.140 | 1.695 |

Natural logs are conventional for neural-network cross-entropy, but convention is not a disclosed protocol. Do not place an assumed conversion beside a measured value without labeling the assumption.

Loss differences also need matching units and tokenization. If two models share the dataset and token unit, and their natural-log losses differ by $\Delta$, then their PPL ratio is $e^\Delta$. Without those conditions, that ratio has no clean interpretation.

## Token Averaging Is Not Sequence Averaging

Consider two documents:

| Document | Scored tokens | Total NLL | NLL/token | Document PPL |
|---|---:|---:|---:|---:|
| A | 10 | 20 nats | 2.0 | 7.389 |
| B | 100 | 50 nats | 0.5 | 1.649 |

Different corpus aggregations answer different questions:

| Statistic | Calculation | Result | Weighting |
|---|---|---:|---|
| Token-weighted loss | $(20+50)/(10+100)$ | 0.636 nat/token | Long documents dominate |
| PPL of token-weighted loss | $e^{0.63636}$ | 1.890 | Same weighting as above |
| Equal-document mean loss | $(2.0+0.5)/2$ | 1.25 nats/token | Each document equal |
| PPL of mean document loss | $e^{1.25}$ | 3.490 | Geometric mean document PPL |
| Arithmetic mean document PPL | $(7.389+1.649)/2$ | 4.519 | Outliers dominate |
| Mean sequence NLL | $(20+50)/2$ | 35 nats/sequence | Length-dependent |

None of these is a bug; mixing their labels is. In particular, the arithmetic mean of document perplexities is not the perplexity of mean document loss.

Library defaults can silently make this choice. Hugging Face Evaluate returns an unweighted arithmetic mean of per-sequence perplexities, while lm-evaluation-harness computes BPB through a ratio of summed quantities. Always report the aggregation equation, not merely the library name.

## Why Tokenization Changes the Answer

Let tokenizer $\tau$ produce $N_\tau$ target tokens for the same text. Raw PPL averages over $N_\tau$, so two tokenizers create different prediction units even when the underlying string is identical.

### A constructed 12-bit example

Two systems assign the same total of 12 bits to the same 12 ASCII bytes:

| System | Tokens | Bits/token | Token PPL | BPB |
|---|---:|---:|---:|---:|
| A | 3 | 4 | $2^4=16$ | 1.0 |
| B | 6 | 2 | $2^2=4$ | 1.0 |

Calling B four times better because its token PPL is four times smaller is wrong. The common byte denominator exposes that both assigned the same code length.

### BPB and BPC

Let $S_e$ be total NLL in nats over the **scored target spans**, $B$ their original UTF-8 byte count, and $C$ their character count:

$$
\mathrm{BPB}=\frac{S_e}{B\ln 2}
$$

$$
\mathrm{BPC}=\frac{S_e}{C\ln 2}
$$

Equivalently,

$$
\mathrm{BPB}=\frac{N}{B}\log_2(\mathrm{PPL}_{token})
$$

The numerator must cover exactly the text represented by the denominator. Do not divide a full-corpus NLL by only the last window's bytes, or count prompt bytes that were context rather than targets.

A BPB difference has a useful code-length interpretation: a gap of 0.01 BPB over one billion bytes represents 10 million bits, or 1.25 MB, of idealized code length. It does **not** represent a 1% intelligence difference.

BPC requires a character convention. Unicode code points and user-perceived grapheme clusters are not interchangeable. UTF-8 also changes relative weighting: for example, `é` is one code point but two UTF-8 bytes. For multilingual evaluation, publish both per-language results and the exact encoding and character definition.

### BPB is a common unit, not a tokenizer eraser

The 2026 Schaeffer–Koyejo preprint makes the residual dependence explicit. If $L$ is total bits, then

$$
\mathrm{BPB}(\tau,p)=\frac{N_\tau}{B}\frac{L(\tau,p)}{N_\tau}
$$

For an untrained uniform model over vocabulary size $V_\tau$,

$$
\mathrm{BPB}(\tau,p_{uniform})=
\frac{N_\tau}{B}\log_2 V_\tau
$$

Changing the vocabulary changes both tokens per byte and bits per token; nothing forces cancellation. BPB can therefore favor a tokenizer before any learned parameters enter the comparison.

The correct interpretation is **model plus tokenizer as a text-coding system**. BPB is appropriate for that compression question. It does not isolate the quality of the weights, which is why Paloma prefers a fixed vocabulary for controlled comparisons.

### Canonical segmentation is another convention

For string $s$, marginal string probability sums over all allowed tokenizations $T(s)$:

$$
p(s)=\sum_{\tau\in T(s)}\prod_i
p_\theta(x_i^\tau\mid x_{<i}^\tau)
$$

Ordinary teacher-forced scoring selects only the canonical $\tau$. Canonical BPB is a well-defined score for that coding system, but it need not equal marginal text likelihood. Do not compare one model's canonical score with another's marginal score.

Normalization can also change the event being scored. Unicode normalization, case folding, whitespace replacement, and unknown-token substitution may make `decode(encode(s))` differ from `s`. If normalization is lossy, a raw-byte compressor needs either reversible side information or an explicitly shared normalized corpus. Silent normalization is not a fair comparison of the original strings.

## Context, Windows, and Scored Spans

Every token's probability depends on its prefix. A fair protocol must fix:

1. document boundaries and whether documents are concatenated;
2. BOS/EOS tokens and whether they are context or targets;
3. the raw-text prefix available to every model;
4. maximum window, stride, and which positions are masked;
5. treatment of overlong documents and final partial windows;
6. whether each target byte is scored exactly once.

The effect is not cosmetic. Hugging Face's [official tutorial](https://huggingface.co/docs/transformers/en/perplexity) reports the same GPT-2-large checkpoint on WikiText-2 at **19.44 PPL** with disjoint 1024-token windows and **16.44 PPL** with stride 512. No model changed; the available context changed.

Matching a token count across providers is not matching context in the original text. Prefer a declared raw-text prefix and scored span, then tokenize separately for each model. If full-document scoring is impossible, publish a deterministic windowing rule and verify that the target byte sets are identical.

## What Hosted APIs Can Measure

The required quantity is full-softmax $-\log p(\mathrm{reference\ token}\mid\mathrm{actual\ prefix})$ for every target position. Current public chat interfaces do not generally expose it.

| Access | Documented probability access | Consequence |
|---|---|---|
| OpenAI Chat Completions | Generated-output logprobs; 0–20 alternatives | No arbitrary prompt/reference-text scorer |
| Anthropic Messages | No logprobs or target NLL | Cannot score a fixed corpus |
| Anthropic OpenAI compatibility | `logprobs` and `top_logprobs` ignored | Compatibility does not add scoring |
| Open weights | Full logits at every position | Supports controlled teacher forcing |
| Provider-assisted private scorer | Reference NLL under agreed prefixes | Feasible if both providers cooperate |

OpenAI documents `top_logprobs` up to **20** and a **-9999.0** sentinel when a returned token is outside the top 20. The sentinel is not an exact log-probability and must never be substituted into an NLL sum. Even without truncation, generated-output logprobs answer a different question: how probable the model's own output was under its generated prefix.

These workarounds are invalid:

- **Ask the model to repeat a passage.** The passage enters the instruction context, reproduction is not guaranteed, and the measured event changes.
- **Compare each assistant's own generations.** Each model selects different text and prefixes, so the evaluation distribution is no longer shared.
- **Force the reference with bias or prefills and read output logprobs.** Generation control is not likelihood evidence for the supplied tokens; bias can also alter the distribution.
- **Use token counts as likelihoods.** A token count provides a denominator, not any $p_\theta$ value.

Anthropic's native schema exposes no reference scorer, and its compatibility table marks both logprob fields as ignored. OpenAI's standard chat schema returns probabilities only for generated output positions. Therefore an independent researcher cannot produce a direct GPT-versus-Claude base-model PPL or BPB table through these public chat APIs. The honest result is **not measurable with the available interface**, not an approximation assembled from sampled completions.

## Dataset Choice and Contamination

Cross-entropy is always loss **on a distribution**. Code, mathematics, general web text, dialogue, and multiple-choice formatting have different regularities and entropy floors. Changing a dataset version, newline convention, document length mix, or language mix can change the score without any model change.

Contamination adds a second problem: memorized evaluation text can receive high likelihood without equivalent generalization to unseen text. Lee et al. found train/test overlap affecting **over 4% of validation examples in standard datasets**; their deduplication work also reduced memorized emissions. [arXiv 2107.06499](https://arxiv.org/abs/2107.06499)

A strong cross-provider corpus should therefore be:

1. **New and private.** Commission text after all relevant training cutoffs; do not rely on public benchmark names alone.
2. **Stratified.** Include declared general-text, code, mathematics, and language domains rather than letting one predictable domain dominate.
3. **Byte-frozen.** Preserve exact UTF-8, whitespace, newlines, and document boundaries; publish hashes to the evaluator.
4. **Overlap-audited where possible.** Check exact and near duplicates against accessible training data. For closed training corpora, record that the audit is impossible rather than claiming decontamination.
5. **Used once per frozen model.** If providers receive the text, later checkpoints can train on it. Use simultaneous evaluations, no-training agreements, and fresh rotating holdouts.

A private set improves comparability but does not prove cleanliness. It also reduces public reproducibility; the evaluator should preserve immutable hashes, access logs, and enough aggregate metadata for an audit.

## Post-Training and Prompting Move a Different Axis

SFT and RL change model weights, so they can change perplexity. Their effect is not directionally guaranteed: imitation of evaluation-like text can lower its NLL, while reward optimization can improve task reward without improving generic held-out NLL. Compare base with base, or post-trained with post-trained, and label the exact recipe.

Prompting does not change weights, but it changes the conditional distribution being measured. Raw base-model PPL, PPL after a system prompt, and PPL inside a chat template are three different quantities. Hidden provider instructions make a hosted-assistant score especially difficult to equate with base loss.

Two primary results show why user-facing capability can move independently of a base-compression number:

- With unchanged `text-davinci-002` weights, adding “Let's think step by step” raised GSM8K zero-shot accuracy from **10.4% to 40.7%** in Kojima et al.'s Table 1. This is a prompting effect, not a pretraining-loss change. [arXiv 2205.11916](https://arxiv.org/abs/2205.11916)
- In InstructGPT's human evaluation on the authors' prompt distribution, raters preferred **1.3B InstructGPT outputs to 175B GPT-3 outputs**. This does not show that the smaller model has lower base PPL; it shows the practical effect of post-training. [arXiv 2203.02155](https://arxiv.org/abs/2203.02155)

OpenAI makes the same conceptual point: chain-of-thought reasoning can drastically increase capability without reducing the model's perplexity. Reasoning tokens, tools, retrieval, retries, and inference budget belong in the downstream evaluation record, not as invisible adjustments to base loss.

Even logit temperature illustrates the distinction. Transforming probabilities to $p_i^{1/\tau}$ with normalization preserves the greedy argmax for positive $\tau$ but changes NLL. A prediction score includes calibration and distributional sharpness, not just which answer is ranked first.

## Does Perplexity Measure Intelligence?

### Why it is a useful proxy

Prediction rewards compressed regularities, and many useful abilities reduce surprise: syntax, factual knowledge, code structure, and some reasoning. There is real empirical signal:

| Study | Result | Scope limit |
|---|---|---|
| Huang et al. 2024 | Average BPC versus average benchmark score: Pearson $r=-0.93$ | Overall fit uses 19 general-purpose base models; 31 models and 12 benchmarks across the full study |
| OpenAI monitorability study 2026 | Base PPL predicted most no-CoT evaluation performance | Architecture 2 showed extra planning and serial-computation gains at matched interpolated PPL |
| OpenAI essay 2026 | Lower internal-code loss coincided with major capability changes | Retrospective associations on a private dataset |

The Huang result is not a universal law. Its coding, mathematics, and knowledge fits use different aligned compression corpora; mismatch weakens the relationship, and only base models were studied. OpenAI's separate [monitorability study](https://openai.com/safety-alignment/cot-monitorability-model-lifecycle/) also reports a weaker relationship between base PPL and CoT controllability than with no-CoT capability.

### Why it cannot be the whole quantity

A concrete ranking reversal appears in DeepSeek-V3's Table 3, under one internal evaluation framework:

| Base model | Pile-test BPB, lower better | Five-shot MMLU, higher better |
|---|---:|---:|
| Llama-3.1-405B | **0.542** | 84.4 |
| DeepSeek-V3 Base | 0.548 | **87.1** |

Llama compresses slightly better; DeepSeek scores higher on MMLU. Neither number is an error. They measure different constructs. [DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437v2)

Other limitations compound:

1. **Averages hide tails.** A model can improve frequent boilerplate while failing rare but safety-critical reasoning.
2. **Memorization is not generalization.** Contaminated text can improve NLL without transferable ability.
3. **Elicitation matters.** Prompting, RL, tools, and test-time search can change task success with the same base checkpoint.
4. **Architecture and data matter at equal loss.** Equal corpus NLL does not imply identical internal algorithms or task-specific capabilities.
5. **Compression includes formatting.** Exact-text code length includes how the system renders text, which need not track answer correctness.

### The unknown floor changes the meaning of small gains

For the true data distribution $q$ and model $p$,

$$
H(q,p)=H(q)+D_{\mathrm{KL}}(q\parallel p)
$$

$H(q)$ is the irreducible entropy; $D_{\mathrm{KL}}$ is the model's excess loss. Neither is observable from a finite score alone.

Suppose two models have losses 0.70 and 0.69 in the same unit. If the unknown floor were 0.65, excess loss would fall from 0.05 to 0.04, a 20% reduction. If the floor were 0.20, excess would fall from 0.50 to 0.49, only 2%. Neither percentage is an intelligence improvement; the example simply shows why an absolute 0.01 gap has no universal meaning near an unknown floor.

**Bottom line:** perplexity measures conditional predictive compression. It can be a durable base-model progress signal and an empirical correlate of average capability. It is not a complete measure of intelligence, a universal capability threshold, or a sufficient safety case.

## A Practical Protocol for the Fairest Cross-Provider Comparison

```text
Frozen private documents ─┐
                          ├─→ Neutral reference scorer
Pinned model snapshots ───┘          │
                                     ├─→ Per-document NLL + counts
                                     │       │
                                     │       ├─→ BPB/BPC + paired intervals
                                     │       └─→ Domain breakdowns
                                     │
                                     └─→ Separate capability evaluations
```

1. **Define the claim.** Choose between base-model pretraining comparison, assistant-distribution comparison, or deployed-task performance. Do not combine them in one ranking.
2. **Pin immutable checkpoints.** Record exact model revisions, checkpoint type, evaluation date, tokenizer/config hashes, numerical precision, and quantization. Avoid moving aliases.
3. **Freeze a fresh stratified corpus.** Use the contamination controls above. Preserve raw bytes and target-span manifests; hash every document and the corpus ordering.
4. **Match context exactly.** Declare raw-text prefixes, document resets, BOS/EOS handling, window/stride, masks, and treatment of long documents. Score each target span once.
5. **Tokenize separately, score the same text.** Never feed one provider's token IDs to another model. Use each model's canonical tokenizer unless the preregistered question is marginal string likelihood.
6. **Obtain full reference NLL.** For open weights, run teacher-forced full-logit evaluation. For closed models, a neutral evaluator must call a provider-operated scorer that returns the actual reference-token log-probabilities under the agreed prefixes. Generated-output logprobs are not a substitute.
7. **Stop when scoring is unavailable.** If either provider cannot return complete reference NLL, cross-provider PPL/BPB is not measurable under the protocol. Do not impute top-k tail mass or substitute self-generated text.
8. **Report common units.** Preserve total nats, per-model scored-token counts, target UTF-8 bytes, and declared character counts. Use BPB for exact-text compression; report BPC as a sensitivity analysis where useful.
9. **Predeclare aggregation.** Publish byte-weighted corpus BPB, per-domain BPB, and an equal-domain macro view. For document or domain estimates, use paired resampling and recompute both numerator and denominator in every bootstrap replicate.
10. **Report uncertainty and effect size.** Give paired intervals for $BPB_A-BPB_B$, not just rankings. Interpret a difference as bits per byte, not as a percentage of intelligence.
11. **Run capability evaluations separately.** Fix prompts, tools, reasoning budget, sampling, and cost limits. Include fresh task suites and specific safety evaluations; report them beside, not inside, compression.
12. **Publish the audit record.** Include code, corpus hashes, scored spans, context rules, raw NLL sums, denominators, model revisions, API restrictions, and any provider attestations.

The strongest controlled study fixes vocabulary as well as text and scoring. That is feasible for models trained or instrumented for the study, as Paloma's baselines show. Existing providers generally cannot swap vocabularies, so BPB is the fairest common compression unit—but the resulting claim remains about tokenizer-model systems.

## Exactly Which Comparisons Remain Invalid

| Invalid comparison | Why it fails | Valid replacement |
|---|---|---|
| Token PPL on different datasets | Different distribution and entropy floor | Same frozen corpus and targets |
| Token PPL across tokenizers | Different prediction units | BPB/BPC with full protocol |
| Nats directly against bits | Different log bases | Convert by $\ln 2$ |
| Token-weighted against document-weighted PPL | Different estimands | Declare and match aggregation |
| Arithmetic mean PPL against PPL of mean loss | Nonlinear exponentiation | Compare identical formulas |
| Different context windows or strides | Different conditioning information | Match raw-text prefixes and masks |
| One model's canonical against another's marginal likelihood | Different probability events | Use one segmentation convention |
| OpenAI generated-text logprobs against fixed-text NLL | Different text and prefixes | Provider reference scorer |
| Repetition, forced prefill, or logit-bias scores | Prompt leakage or altered distribution | Teacher-forced reference NLL |
| Exact NLL containing -9999.0 sentinels | Missing tail probability | Full-softmax target logprobs |
| Base model against hosted assistant | Post-training and hidden context differ | Match checkpoint and prompt class |
| Public old text against fresh private text | Different contamination risk | Same time-frozen corpus for all |
| BPC using code points against graphemes | Different denominators | One declared character definition |
| BPB against raw PPL or BPC | Different units | Convert only with exact counts |
| OpenAI loss against another lab's published loss | Corpus and scoring details differ | Joint neutral evaluation |
| 0.828 as a universal cybersecurity threshold | Internal retrospective association | Direct cyber capability evaluation |
| BPB gap as an intelligence percentage | Compression is not general intelligence | Separate task and safety evidence |

Even a perfectly executed BPB comparison does not answer training-compute efficiency, inference cost, alignment, controllability, or downstream usefulness. Those require their own measurements.

## Perplexity Versus Alternatives

| Metric | Unit | Main question answered | Key dependency | Main blind spot |
|---|---|---|---|---|
| Token PPL | Effective branches/token | Canonical next-token fit | Tokenizer and corpus | Cross-tokenizer ranking |
| BPB | Bits/UTF-8 byte | Exact-text compression | Tokenizer-model pair and context | Pure model quality |
| BPC | Bits/character | Character-normalized compression | Character convention | Multilingual weighting |
| Task benchmark | Accuracy/pass rate | Elicited task capability | Prompt, tools, contamination | Average prediction quality |
| Human preference | Win rate | Product usefulness | Rater and prompt distribution | Correctness and safety |
| Targeted safety eval | Scenario success | Specific hazardous capability | Harness and threat model | General prediction ability |

These metrics are complements. A low BPB model can lose a task benchmark; a preferred assistant can have mediocre base loss; both can fail a targeted safety evaluation.

## Practical Considerations

**Use perplexity confidently for** checkpoint tracking with one tokenizer and frozen validation pipeline, scaling-law fits, controlled pretraining ablations, and exact-text compression research. Your GPT-2 124M validation loss is meaningful only beside its data split, tokenizer, context, stride, and aggregation code.

**Use BPB cautiously for** cross-tokenizer base-model compression when the full protocol is controlled. Report per-domain results and downstream tasks so that a compression ranking does not silently become an intelligence ranking.

**Do not use either as** a standalone model-selection score, a proof of emergent reasoning, a cross-lab safety threshold, or evidence that a hosted assistant is universally better. Your SFT checkpoint also needs its chat template and training stage declared; its task-tuned NLL is not interchangeable with base GPT-2 NLL.

A minimal reporting manifest is:

```text
model_snapshot:
checkpoint_type:
tokenizer_and_config_hash:
corpus_and_target_span_hash:
context_window_and_stride:
bos_eos_and_mask_rule:
normalization_and_utf8_version:
log_base:
total_nll:
scored_tokens_per_model:
target_utf8_bytes:
target_character_definition_and_count:
aggregation_and_bootstrap:
prompt_reasoning_tool_and_cost_settings:
```

The shortest defensible summary is: **same text, same context, same scoring event, explicit units, paired uncertainty, and separate capability tests. If a provider cannot return reference NLL, there is no direct perplexity comparison to make.**

## Key Papers

1. Pachocki, **Perplexity and Intelligence**, OpenAI, 2026. [Official post](https://openai.com/index/perplexity-and-intelligence/)
2. Shannon, **A Mathematical Theory of Communication**, 1948. [DOI](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x)
3. Radford et al., **Language Models are Unsupervised Multitask Learners**, 2019. [OpenAI report](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
4. Kaplan et al., **Scaling Laws for Neural Language Models**, 2020. [arXiv 2001.08361](https://arxiv.org/abs/2001.08361)
5. Cao and Rimell, **You Should Evaluate Your Language Model on Marginal Likelihood over Tokenisations**, 2021. [ACL Anthology](https://aclanthology.org/2021.emnlp-main.161/)
6. Delétang et al., **Language Modeling Is Compression**, 2023. [arXiv 2309.10668](https://arxiv.org/abs/2309.10668)
7. Magnusson et al., **Paloma: A Benchmark for Evaluating Language Model Fit**, NeurIPS 2024. [arXiv 2312.10523](https://arxiv.org/abs/2312.10523)
8. Lee et al., **Deduplicating Training Data Makes Language Models Better**, ACL 2022. [arXiv 2107.06499](https://arxiv.org/abs/2107.06499)
9. Ouyang et al., **Training Language Models to Follow Instructions with Human Feedback**, 2022. [arXiv 2203.02155](https://arxiv.org/abs/2203.02155)
10. Kojima et al., **Large Language Models are Zero-Shot Reasoners**, 2022. [arXiv 2205.11916](https://arxiv.org/abs/2205.11916)
11. Huang et al., **Compression Represents Intelligence Linearly**, 2024. [arXiv 2404.09937](https://arxiv.org/abs/2404.09937)
12. DeepSeek-AI, **DeepSeek-V3 Technical Report**, 2024. [arXiv 2412.19437](https://arxiv.org/abs/2412.19437)
13. Schaeffer and Koyejo, **Tokenizer Myths and Where To Find Them**, 2026 preprint. [arXiv 2611.04215](https://arxiv.org/abs/2611.04215)
14. Wang et al., **What affects chain-of-thought monitorability across the model lifecycle?**, OpenAI, 2026. [Official study](https://openai.com/safety-alignment/cot-monitorability-model-lifecycle/)

## Primary Implementation Documentation

1. [Hugging Face: Perplexity of fixed-length models](https://huggingface.co/docs/transformers/en/perplexity)
2. [Hugging Face Evaluate: perplexity implementation](https://github.com/huggingface/evaluate/blob/main/metrics/perplexity/perplexity.py)
3. [lm-evaluation-harness: metrics implementation](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/metrics.py)
4. [OpenAI Chat Completions API](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
5. [Anthropic Messages API](https://platform.claude.com/docs/en/api/messages/create)
6. [Anthropic OpenAI SDK compatibility](https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk)
