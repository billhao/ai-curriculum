# Perplexity, loss, and what you can compare across labs

OpenAI reports a base-model loss of 0.761 for GPT-6.2 Astra. What would you need to put your GPT-2 124M, or Anthropic's Claude, on that same chart?

## Start with one prediction from your GPT-2

Imagine scoring a passage from a held-out file. At each position you give your model the **actual preceding tokens** and record the probability it assigns to the token that really follows. This is *teacher forcing*. An unlikely prediction at position 12 costs the model points there. At position 13 it gets to see the correct token 12, just as it did in your training loop.

Suppose the model assigns three successive target tokens probabilities of 1/2, 1/4, and 1/8. The whole continuation has probability $1/64$. In base-2 logs, the surprises are 1, 2, and 3 bits. Add them and you have a six-bit description of those three outcomes, given the model and prefix. Average them and you get two bits per token. **Perplexity is four** because $2^2=4$. The model gave the observed tokens a geometric-mean probability of one in four. It could have had a vocabulary of 50,000 and chosen the right token on every greedy prediction. Perplexity alone tells you none of those ranks.

Your usual cross-entropy uses natural logarithms. For $N$ scored tokens with probabilities $p_i$, the loss is in **nats per token** and the perplexity (PPL) is

$$
L=-\frac{1}{N}\sum_{i=1}^{N}\ln p_i,\qquad
\mathrm{PPL}=e^L,\qquad e^{-L}=\left(\prod_{i=1}^{N}p_i\right)^{1/N}.
$$

The same example gives $L=\ln 4\approx1.386$. A checkpoint that lowers loss by 0.10 nats on the same targets increases their geometric-mean probability by $e^{0.10}\approx1.105$, about 10.5%. That is an exact interpretation of the scale. Translating it into math accuracy or coding success requires measurements on those tasks.

**Training loss** averages over examples used to update the weights. **Validation loss** applies the same formula to held-out examples. A lab can lower its reported training loss simply by repeating predictable text more often. Even with identical tokenizers, two unrelated training curves give you no way to tell which model would predict your next file better. Scoring both on that file holds its difficulty constant.

### Background

Claude Shannon at Bell Labs used human next-letter predictions to study how many bits English requires per letter. Frederick Jelinek and colleagues at IBM later described perplexity as the effective number of choices in speech recognition. Modern language models make the link operational. An arithmetic coder can use the probabilities from a shared predictor to encode the observed sequence close to its total negative log-probability in bits. The costs of storing the model and tokenizer are separate. Kaplan and colleagues at OpenAI then showed how predictably held-out language-model loss could fall with scale under a particular training setup. [Shannon, 1951](https://doi.org/10.1002/j.1538-7305.1951.tb01366.x) · [Jelinek et al., 1977](https://pubs.aip.org/asa/jasa/article/62/S1/S63/642598) · [Delétang et al., *Language Modeling Is Compression*](https://arxiv.org/abs/2309.10668) · [Kaplan et al., 2020](https://arxiv.org/abs/2001.08361)

## What OpenAI measured

On November 22, 2026, Jakub Pachocki published a series of losses for pretrained OpenAI models on a **fixed, held-out snapshot of OpenAI's internal codebase**. The chart gives GPT-3.5 **0.986**, GPT-4 **0.913**, GPT-5.6 Sol **0.828**, and GPT-6.2 Astra **0.761**. A separate internal model reached **0.732**. These are log-perplexities. The essay leaves the log base and full scoring protocol unspecified. Its earlier discussion of GPT-2 and GPT-3 draws on other datasets. [OpenAI essay, chart and footnotes 2–4](https://openai.com/index/perplexity-and-intelligence/)

Pachocki connects the GPT-3.5 generation with fluent dialogue, GPT-4 with productive chain-of-thought reasoning, and the model around 0.76 with progress on open math problems. The observations run in that direction historically. They do not give a formula for predicting which new capability another lab will see at 0.76. His **0.68–0.70** band is a forecast. He proposes using shared measurements alongside evidence about monitoring to guide development limits. [OpenAI essay](https://openai.com/index/perplexity-and-intelligence/)

**One other lab actually ran this test.** On November 23, Microsoft's Mustafa Suleyman reported **0.819 for MAI Base 2**, with permission to use OpenAI's evaluation set and with the same tokenizer. That puts it 0.009 below GPT-5.6 Sol on this code snapshot. The two training corpora could differ and the scores would still answer the same evaluation question. Whether the small gap generalizes to other codebases is a separate empirical question. Neither disclosure gives a Claude base score on this test. [Suleyman's report](https://x.com/mustafasuleyman/status/2124663594449318173) · [OpenAI chart](https://openai.com/index/perplexity-and-intelligence/)

## Change the tokenizer and the ranking can flip

Consider the **same 120 UTF-8 bytes** of held-out code. Suppose each model scores its own lossless tokenization with the same starting context. All numbers here are illustrative.

| Model | Tokens | Bits/token | Token PPL | Total bits | Bits/byte |
|---|---:|---:|---:|---:|---:|
| A | 60 | 2 | 4 | 120 | 1.00 |
| B | 30 | 3 | 8 | 90 | 0.75 |

Model B spends more bits on each prediction, but each prediction covers twice as much text on average. Its token PPL is double A's, yet it encodes the passage in **25% fewer bits**. For $B$ scored bytes and $N$ scored tokens, convert a natural-log loss with

$$
\mathrm{bits/byte}=\frac{N L}{B\ln 2}.
$$

Use the **actual scored tokens and bytes on that corpus**. An average like “four bytes per token” from a different corpus loses the very information the conversion needs. A Chinese passage and its English translation can have different UTF-8 lengths, so both evaluators need the same original bytes. With a tokenizer that drops accents or maps unknown spans to one symbol, reconstructing the originals also costs bits.

One subtlety matters for tight comparisons. A model might be able to emit both a single token for “cat” and the sequence “c” + “at”. Conventional scoring measures the tokenizer's chosen path. The probability of the text itself sums over all paths that decode to it under the same termination rule. Bits per byte gives the model–tokenizer pair a common unit of compression even when these probabilities differ. [Chirkova et al., ACL 2023](https://aclanthology.org/2023.acl-short.1/)

Microsoft used both approaches in practice. Separately from the shared OpenAI test, it compared MAI Base 2 with DeepSeek-V4-Pro-Base using bits per byte on its own code and commissioned STEM, math, and Q&A data. It identified comparison models available only as post-trained checkpoints, since that changes what can be inferred about their pretraining. [Microsoft AI, *Frontier Pre-training*](https://microsoft.ai/news/frontier-pre-training-at-microsoft-ai/)

## Does lower loss predict intelligence?

To predict the next line of a function, a model can exploit indentation, a naming convention, and what the function computes. All three save bits. Only the last necessarily requires following the algorithm. How much each contributes depends on the test data. At the population level, expected log loss equals the source's own uncertainty plus a penalty for predicting the wrong distribution. Even a perfect predictor of independent fair coin tosses needs one bit per toss. The analogous floor for a natural code corpus is unknown.

Three findings put bounds on how far you can take the proxy.

1. **Match the domain and the association can be strong.** Huang and colleagues studied 31 public base models and 12 downstream benchmarks. Models that used fewer bits per character on Python tended to score higher across coding tasks, with a correlation of **−0.937**. That describes variation across the tested models. It gives no fixed number of benchmark points per unit of loss on OpenAI's codebase. [Huang et al., *Compression Represents Intelligence Linearly*](https://arxiv.org/abs/2404.09937)
2. **Change the domain and transfer can fail.** In Stanford's HELM evaluation, bits per byte on The Pile poorly predicted downstream accuracy across model families. Some models had trained on The Pile, giving them an advantage on its distribution. An Anthropic 52B model that scored highly on the core tasks trailed GPT-J on the regional-English and Twitter language-modeling evaluations. [Liang et al., HELM, §§8.2 and 8.4](https://arxiv.org/pdf/2211.09110#page=54)
3. **Even matched loss leaves room for different abilities.** OpenAI compared three architecture families at equal base-model perplexity. Most of its direct-answer evaluations tracked perplexity well, yet one architecture performed materially better at multihop planning and serial computation. And if you give a frozen model more reasoning tokens at inference, it can solve problems it previously missed with exactly the same base checkpoint. [Wang et al., Fig. 3](https://openai.com/safety-alignment/cot-monitorability-model-lifecycle/)

## What can you actually compare for GPT and Claude?

**As an API customer on November 23, 2026:** neither the standard GPT-6.2 Astra/Sol endpoints nor Claude Messages documents a way to submit arbitrary text and recover every reference token's probability from the underlying pretrained checkpoint. GPT-6.2 omits logprobs. Claude's OpenAI-compatible interface ignores that field. On GPT models that do expose output logprobs, the probability of a generated reply tells you how surprising that reply was to the model. Asking the model to copy the reference lets it see the answer and changes the probability you are measuring. A top-*k* list can also omit the reference token entirely. [GPT-6.2 parameter support](https://developers.openai.com/api/docs/guides/latest-model#update-api-and-model-parameters) · [OpenAI output probabilities](https://developers.openai.com/api/docs/guides/advanced-usage) · [Claude Messages](https://platform.claude.com/docs/en/api/messages/create) · [Claude compatibility reference](https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk)

**If your question is about the assistants you use**, give both the same fresh coding problems, available tools, and a stated time or cost budget. Grade both answers against the same tests. You can make a defensible statement about their success rates under those conditions even when both base losses are unknown. A numerical GPT-versus-Claude gap in Pachocki's metric would require a new measurement.

**If the labs cooperate, or you have the weights**, you can run a shared evaluation in four steps.

1. Freeze the same held-out documents as exact bytes. Include domains you care about, specify their weights, and check for train/test overlap. Each lab can keep its training mixture private while checking its own overlap. Report per-domain results so repetitive code cannot hide a regression on prose.
2. Use named checkpoints. To compare pretraining, use base models. Give them the same raw preceding text, with fixed document boundaries and scored spans. For open weights, short passages that fit both windows are simplest. Score the first token under a documented start condition. Keep padding, unscored prompts, and wrapper tokens out of the totals.
3. Teacher-force each passage. Use raw predictive probabilities and gather the score of every true target. Add negative log-probabilities and divide **once** by the total target-token or byte count. In a sliding window, mask overlap so each target contributes once. An unweighted average of window losses overweights the short final window.
4. Use token PPL for identical tokenizations or bits per byte for differing ones. For a small difference, resample the **same documents for both models** and recompute the difference in corpus scores to obtain a paired uncertainty interval. Repeat on a second domain before generalizing. [Paloma's multi-domain evaluation](https://arxiv.org/abs/2312.10523) · [Hugging Face on context windows](https://huggingface.co/docs/transformers/en/perplexity)

**For your own training history**, compare two nanoGPT checkpoints on the same validation split with the same mask. A lower loss has the exact probability interpretation above. To see what SFT changed, score your base and SFT checkpoints twice, once on held-out prose and once on held-out Dolly-style **answer tokens** conditioned on their prompts. The two rankings can disagree, and each answers a useful question. Your hand-written DPO loss measures a preference margin relative to a reference model. Exponentiating *that* loss has no next-token perplexity interpretation.
