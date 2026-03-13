# Language Model Lineage: Transformer (2017) to GPT-3 (2020)

## Evolution Timeline

```
2017-06  Transformer (Vaswani et al.)  ──── encoder-decoder, self-attention only
           │
2018-01  ULMFiT (Howard & Ruder)  ──────── transfer learning for NLP (LSTM-based)
           │
2018-02  ELMo (Peters et al.)  ──────────── contextual word embeddings (biLSTM)
           │
2018-06  GPT-1 (Radford et al.)  ────────── decoder-only, generative pretraining
           │
2018-10  BERT (Devlin et al.)  ──────────── encoder-only, MLM + NSP, bidirectional
           │
2019-01  Transformer-XL (Dai et al.)  ───── segment recurrence + relative pos encoding
           │
2019-02  GPT-2 (Radford et al.)  ────────── scale up, zero-shot, WebText
           │
2019-06  XLNet (Yang et al.)  ───────────── permutation LM + Transformer-XL backbone
           │
2019-07  RoBERTa (Liu et al.)  ──────────── BERT done right (more data, no NSP)
           │
2019-09  ALBERT (Lan et al.)  ───────────── parameter reduction (factorized + sharing)
           │
2019-10  BART (Lewis et al.)  ───────────── denoising seq2seq (encoder-decoder)
           │
2019-10  T5 (Raffel et al.)  ────────────── text-to-text unification, C4 dataset
           │
2020-05  GPT-3 (Brown et al.)  ──────────── 175B params, in-context learning, few-shot
```

---

## Three Paradigms of Pre-trained Language Models

```
                     ┌──────────────────────────────────────────┐
                     │        Original Transformer (2017)        │
                     │       Encoder-Decoder Architecture        │
                     └─────────┬──────────┬──────────┬──────────┘
                               │          │          │
                    ┌──────────▼───┐ ┌────▼────┐ ┌───▼──────────┐
                    │  Autoencoding │ │Enc-Dec  │ │Autoregressive│
                    │  (Encoder)    │ │         │ │  (Decoder)   │
                    ├──────────────┤ ├─────────┤ ├──────────────┤
                    │ Bidirectional │ │Both     │ │ Left-to-right│
                    │ context       │ │         │ │ generation   │
                    ├──────────────┤ ├─────────┤ ├──────────────┤
                    │ BERT         │ │ T5      │ │ GPT-1/2/3    │
                    │ RoBERTa      │ │ BART    │ │ XLNet        │
                    │ ALBERT       │ │         │ │ Transformer-XL│
                    │ ELMo (biLSTM)│ │         │ │              │
                    ├──────────────┤ ├─────────┤ ├──────────────┤
                    │Best for:     │ │Best for:│ │Best for:     │
                    │NLU, classify,│ │Summarize│ │Text gen,     │
                    │NER, QA       │ │Translate│ │completion,   │
                    │(extractive)  │ │QA (gen) │ │dialogue      │
                    └──────────────┘ └─────────┘ └──────────────┘
```

---

## 1. Transformer (Vaswani et al., 2017)

| Field            | Detail                                                                |
|------------------|-----------------------------------------------------------------------|
| Paper            | "Attention Is All You Need"                                           |
| arXiv            | https://arxiv.org/abs/1706.03762                                      |
| Organization     | Google Brain / Google Research                                        |
| Authors          | Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan Gomez, Lukasz Kaiser, Illia Polosukhin |
| Conference       | NeurIPS 2017                                                          |
| Architecture     | Encoder-decoder, 6 layers each side                                   |

**Parameter Counts:**
- Base model: 65M parameters (d_model=512, d_ff=2048, h=8, d_k=64)
- Big model: 213M parameters (d_model=1024, d_ff=4096, h=16, d_k=64)

**Training Data:** WMT 2014 English-German (4.5M sentence pairs) and English-French (36M sentence pairs)

**Key Benchmarks:**
- EN-DE: 28.4 BLEU (base), improving over previous best by 2+ BLEU
- EN-FR: 41.8 BLEU (single model SOTA), trained in 3.5 days on 8 GPUs

**Key Innovation:** Replaced recurrence/convolution entirely with multi-head self-attention + positional encoding. Enabled massive parallelization. Introduced scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V

---

## 2. ULMFiT (Howard & Ruder, 2018) -- Precursor to Transfer Learning in NLP

| Field            | Detail                                                                |
|------------------|-----------------------------------------------------------------------|
| Paper            | "Universal Language Model Fine-tuning for Text Classification"        |
| arXiv            | https://arxiv.org/abs/1801.06146                                      |
| Organization     | fast.ai / NUI Galway                                                  |
| Authors          | Jeremy Howard, Sebastian Ruder                                        |
| Conference       | ACL 2018                                                              |
| Architecture     | AWD-LSTM (3-layer LSTM with dropout regularization)                   |

**Training Data:** Pretrained on Wikitext-103 (28,595 Wikipedia articles, 103M words)

**Key Benchmarks:** Reduced error by 18-24% on six text classification benchmarks (IMDb, TREC-6, AG News, DBpedia, Yelp-bi, Yelp-full)

**Key Innovation:** Demonstrated that transfer learning (pretrain LM on general corpus, fine-tune on task) works for NLP, analogous to ImageNet pretraining for CV. Three-stage approach:
1. General-domain LM pretraining
2. Target-task LM fine-tuning (with discriminative fine-tuning + slanted triangular LR)
3. Target-task classifier fine-tuning (with gradual unfreezing)

**Significance:** Conceptual blueprint for GPT and BERT. Showed that even LSTM-based LMs benefit enormously from pretrain-then-finetune. Introduced techniques (discriminative learning rates, gradual unfreezing) still relevant today.

---

## 3. ELMo (Peters et al., 2018)

| Field            | Detail                                                                |
|------------------|-----------------------------------------------------------------------|
| Paper            | "Deep Contextualized Word Representations"                            |
| arXiv            | https://arxiv.org/abs/1802.05365                                      |
| Organization     | Allen Institute for AI (AI2)                                          |
| Authors          | Matthew Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer |
| Conference       | NAACL 2018 (Best Paper Award)                                        |
| Architecture     | 2-layer bidirectional LSTM + character CNN embeddings                 |

**Parameter Count:** 93.6M parameters (original model: 2 layers, 4096 LSTM units, 512 projection)

**Training Data:**
- Standard: 1 Billion Word Benchmark (~800M tokens of news data)
- 5.5B variant: 5.5B tokens (Wikipedia 1.9B + WMT news 3.6B)

**Key Benchmarks:** New SOTA on 6 tasks: QA (SQuAD), textual entailment (SNLI), semantic role labeling, coreference resolution, NER, sentiment analysis

**Key Innovation:** Context-dependent word representations. Unlike Word2Vec/GloVe where "bank" has one vector regardless of context, ELMo produces different vectors for "river bank" vs "bank account." Representations are a learned weighted combination of all biLSTM layers (different layers capture syntax vs semantics). Character-based input handles OOV words.

**Limitation:** Still LSTM-based (sequential, not parallelizable); representations are added as features to task-specific architectures rather than fine-tuning the whole model.

---

## 4. GPT-1 (Radford et al., 2018)

| Field            | Detail                                                                |
|------------------|-----------------------------------------------------------------------|
| Paper            | "Improving Language Understanding by Generative Pre-Training"         |
| URL              | https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf |
| Organization     | OpenAI                                                                |
| Authors          | Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever       |
| Date             | June 2018                                                             |
| Architecture     | Decoder-only Transformer                                              |

**Parameter Count:** 117M (later corrected to ~124M by actual weight count)

**Architecture Details:**
- 12 Transformer decoder layers
- 768 hidden dimension (d_model)
- 12 attention heads
- 3072 feed-forward dimension (4x d_model)
- Context window: 512 tokens
- BPE tokenizer with 40,000 merges

**Training Data:** BooksCorpus (~7,000 unpublished books, ~800M words / ~5GB text)

**Key Benchmarks:** Outperformed specifically trained supervised models on 9 of 12 NLU tasks. Achieved gains on commonsense reasoning (Stories Cloze +8.9%), QA (RACE +5.7%), textual entailment, and semantic similarity.

**Key Innovation:** Combined generative pretraining (unsupervised, left-to-right LM) with discriminative fine-tuning (supervised, task-specific). First successful application of Transformer decoder for transfer learning. Showed that a single model architecture could be fine-tuned for diverse NLP tasks with minimal modification (add linear output layer).

**Distinction from BERT:** Autoregressive (left-to-right only), which means it can generate text but sees only past context during pretraining. BERT would soon show that bidirectional context helps NLU tasks.

---

## 5. BERT (Devlin et al., 2018)

| Field            | Detail                                                                |
|------------------|-----------------------------------------------------------------------|
| Paper            | "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" |
| arXiv            | https://arxiv.org/abs/1810.04805                                      |
| Organization     | Google AI Language                                                    |
| Authors          | Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova          |
| Conference       | NAACL 2019                                                            |
| Architecture     | Encoder-only Transformer                                              |

**Parameter Counts:**

| Variant    | Params | Layers | Hidden | Heads | FF dim |
|------------|--------|--------|--------|-------|--------|
| BERT-Base  | 110M   | 12     | 768    | 12    | 3072   |
| BERT-Large | 340M   | 24     | 1024   | 16    | 4096   |

**Training Data:** BooksCorpus (800M words) + English Wikipedia (2,500M words), total ~16GB of text. Trained for 1M steps with batch size 256 sequences of 512 tokens.

**Key Benchmarks:**
- GLUE: 80.5 average (7.7 point improvement over prior SOTA)
- MultiNLI: 86.7% accuracy
- SQuAD v1.1: 93.2 F1 (1.5 point improvement)
- SQuAD v2.0: 83.1 F1 (5.1 point improvement)
- New SOTA on 11 NLP tasks

**Training Objectives:**
1. **Masked Language Model (MLM):** Randomly mask 15% of input tokens, predict them. Of masked positions: 80% replaced with [MASK], 10% with random token, 10% unchanged. Enables bidirectional context.
2. **Next Sentence Prediction (NSP):** Binary classification -- given sentence A and B, predict if B actually follows A. 50% true pairs, 50% random. (Later shown to be less important by RoBERTa/ALBERT.)

**Key Innovation:** Deep bidirectional pretraining. Unlike GPT (left-to-right) or ELMo (concatenation of independently trained left-to-right and right-to-left), BERT jointly conditions on both left and right context in every layer. The MLM objective is the mechanism that enables this.

---

## 6. Transformer-XL (Dai et al., 2019)

| Field            | Detail                                                                |
|------------------|-----------------------------------------------------------------------|
| Paper            | "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" |
| arXiv            | https://arxiv.org/abs/1901.02860                                      |
| Organization     | Carnegie Mellon University / Google Brain                             |
| Authors          | Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov |
| Conference       | ACL 2019                                                              |

**Parameter Counts:**
- Standard model: ~151M
- Large model: ~257M (18 layers, d_model=1024, 16 heads)

**Training Data:** Evaluated on WikiText-103 (103M tokens), enwik8, text8, One Billion Word, Penn Treebank

**Key Benchmarks:**
- WikiText-103: 18.3 perplexity (SOTA)
- enwik8: 0.99 bpc (SOTA)
- Learns dependencies 80% longer than RNNs, 450% longer than vanilla Transformers
- 1,800x faster evaluation than vanilla Transformers

**Key Innovations:**
1. **Segment-level recurrence:** Cache hidden states from previous segments and attend to them when processing the current segment. Breaks the fixed-length context limitation without disrupting temporal coherence.
2. **Relative positional encoding:** Inject relative position info into attention scores (not absolute). Required because segment recurrence would make absolute positions meaningless across segments.

**Significance:** Directly used as the backbone for XLNet. The recurrence mechanism and relative positional encoding influenced subsequent long-context work.

---

## 7. GPT-2 (Radford et al., 2019)

| Field            | Detail                                                                |
|------------------|-----------------------------------------------------------------------|
| Paper            | "Language Models are Unsupervised Multitask Learners"                 |
| URL              | https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf |
| Organization     | OpenAI                                                                |
| Authors          | Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever |
| Date             | February 2019                                                         |
| Architecture     | Decoder-only Transformer                                              |

**Parameter Counts (corrected):**

| Variant | Params | Layers | d_model | Heads | d_head |
|---------|--------|--------|---------|-------|--------|
| Small   | 124M   | 12     | 768     | 12    | 64     |
| Medium  | 355M   | 24     | 1024    | 16    | 64     |
| Large   | 774M   | 36     | 1280    | 20    | 64     |
| XL      | 1558M  | 48     | 1600    | 25    | 64     |

Note: Originally reported as 117M/345M/762M/1.5B. Corrected counts come from actual model weights.

**Training Data:** WebText -- 40GB of text from 8M web pages scraped from outbound Reddit links with 3+ karma (before Dec 2017). Vocabulary: 50,257 BPE tokens. Context window: 1024 tokens.

**Key Benchmarks:**
- 7 of 8 language modeling datasets: zero-shot SOTA
- WikiText-103: 17.48 perplexity (zero-shot, without training on it)
- Children's Book Test (NE): 93.3% accuracy
- Lambada: 63.24% accuracy (8% improvement)
- Initially withheld release of 1.5B model due to misuse concerns (staged release)

**Key Innovation:** Demonstrated that large-enough LMs can perform tasks zero-shot without any fine-tuning. Framed NLP tasks as language modeling: e.g., summarization as "TL;DR:" continuation, translation as prompting with examples. Foreshadowed in-context learning.

---

## 8. XLNet (Yang et al., 2019)

| Field            | Detail                                                                |
|------------------|-----------------------------------------------------------------------|
| Paper            | "XLNet: Generalized Autoregressive Pretraining for Language Understanding" |
| arXiv            | https://arxiv.org/abs/1906.08237                                      |
| Organization     | Carnegie Mellon University / Google Brain                             |
| Authors          | Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le |
| Conference       | NeurIPS 2019                                                          |

**Parameter Counts:**
- XLNet-Base: ~110M (mirrors BERT-Base: 12 layers, d_model=768, 12 heads)
- XLNet-Large: ~340M (mirrors BERT-Large: 24 layers, d_model=1024, 16 heads)

**Training Data:** BooksCorpus + English Wikipedia (13GB) + Giga5 (16GB) + ClueWeb 2012-B (19GB) + Common Crawl (110GB). Total: ~158GB raw text, 32.89B subword tokens after SentencePiece tokenization.

**Key Benchmarks:**
- GLUE: ~90.5 (outperforms BERT-Large on all 9 tasks)
- SQuAD 1.1: 95.1 F1
- SQuAD 2.0: 90.6 F1
- RACE: 81.75 accuracy
- Outperforms BERT on 20 tasks, often by a large margin

**Key Innovation -- Permutation Language Modeling:**
Instead of fixed left-to-right or masked prediction, XLNet maximizes the expected log-likelihood over ALL permutations of the factorization order. For a sequence [x1, x2, x3, x4]:
- One permutation might predict x3 given {x2, x4} as context
- Another might predict x3 given {x1} as context

This achieves:
1. Bidirectional context (like BERT) without the [MASK] token discrepancy between pretraining and fine-tuning
2. Autoregressive formulation (like GPT) that respects product rule of probability
3. No independence assumption between predicted tokens (BERT's MLM assumes masked tokens are independent of each other)

Uses Transformer-XL backbone with segment recurrence and relative positional encoding for long-context modeling.

---

## 9. RoBERTa (Liu et al., 2019)

| Field            | Detail                                                                |
|------------------|-----------------------------------------------------------------------|
| Paper            | "RoBERTa: A Robustly Optimized BERT Pretraining Approach"             |
| arXiv            | https://arxiv.org/abs/1907.11692                                      |
| Organization     | Facebook AI (Meta AI)                                                 |
| Authors          | Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov |
| Date             | July 2019                                                             |

**Parameter Count:** 355M (same architecture as BERT-Large: 24 layers, d_model=1024, 16 heads)

**Training Data:** 160GB of text (10x BERT's 16GB):
- BooksCorpus + English Wikipedia (16GB, same as BERT)
- CC-News (76GB)
- OpenWebText (38GB)
- Stories (31GB)

Trained for 500K steps with batch size 8K sequences. Used 1024 V100 GPUs.

**Key Benchmarks:**
- GLUE: 88.5 (vs BERT-Large's 80.5)
- SQuAD: SOTA
- RACE: SOTA
- SuperGLUE, XNLI: SOTA

**What RoBERTa Changed vs BERT:**
1. **Removed NSP:** Next Sentence Prediction objective dropped (didn't help, sometimes hurt)
2. **Dynamic masking:** Instead of static masking (same mask every epoch), generate new mask pattern each time a sequence is fed. 10 different masks over 40 epochs.
3. **Larger batches:** 8K sequences (vs BERT's 256)
4. **More data:** 160GB (vs 16GB)
5. **Longer training:** 500K steps (vs 1M, but with larger batches = more tokens seen)
6. **Full-length sequences:** No short-sequence warming

**Key Finding:** BERT was significantly undertrained. The same architecture, with better hyperparameters and more data/compute, matches or exceeds all subsequent models (XLNet, etc.).

---

## 10. ALBERT (Lan et al., 2019)

| Field            | Detail                                                                |
|------------------|-----------------------------------------------------------------------|
| Paper            | "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations" |
| arXiv            | https://arxiv.org/abs/1909.11942                                      |
| Organization     | Google Research / Toyota Technological Institute at Chicago           |
| Authors          | Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut |
| Conference       | ICLR 2020                                                             |

**Parameter Counts:**

| Variant       | Params | Layers | Hidden | Embedding | Heads |
|---------------|--------|--------|--------|-----------|-------|
| ALBERT-Base   | 12M    | 12     | 768    | 128       | 12    |
| ALBERT-Large  | 18M    | 24     | 1024   | 128       | 16    |
| ALBERT-xLarge | 60M    | 24     | 2048   | 128       | 16    |
| ALBERT-xxLarge| 233M   | 12     | 4096   | 128       | 64    |

Compare: BERT-Large = 340M params. ALBERT-xxLarge = 233M with 18x fewer total params than BERT-Large config yet better performance.

**Training Data:** Same as BERT (BooksCorpus + Wikipedia, ~16GB)

**Key Benchmarks (v2):**
- ALBERT-xxLarge GLUE average: 90.9
- SQuAD 1.1: 94.6/89.1 (F1/EM)
- SQuAD 2.0: 89.8/86.9 (F1/EM)
- RACE: 86.5 (SOTA at time of release)

**Parameter Reduction Techniques:**
1. **Factorized embedding parameterization:** Decompose V x H embedding matrix into V x E and E x H. With E=128, H=4096: reduces from V x 4096 to V x 128 + 128 x 4096 (huge savings when V=30K).
2. **Cross-layer parameter sharing:** All Transformer layers share the same parameters (attention + FFN). Dramatically reduces parameters. Network depth maintained (12-24 layers) but weight count stays small.

**Additional Innovation -- Sentence Order Prediction (SOP):**
Replaced BERT's NSP with SOP. Positive: two consecutive segments (same as NSP). Negative: same two segments but order swapped. Forces learning inter-sentence coherence rather than the easier topic-prediction signal that NSP conflates. NSP cannot solve SOP task (random baseline), but SOP can solve NSP reasonably well (78.9%).

---

## 11. BART (Lewis et al., 2019)

| Field            | Detail                                                                |
|------------------|-----------------------------------------------------------------------|
| Paper            | "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension" |
| arXiv            | https://arxiv.org/abs/1910.13461                                      |
| Organization     | Facebook AI (Meta AI)                                                 |
| Authors          | Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, Luke Zettlemoyer |
| Conference       | ACL 2020                                                              |
| Architecture     | Encoder-decoder Transformer                                          |

**Parameter Counts:**
- BART-Base: 139M (6 encoder + 6 decoder layers, d_model=768, 16 heads)
- BART-Large: 406M (12 encoder + 12 decoder layers, d_model=1024, 16 heads)

**Training Data:** Same as RoBERTa (160GB: BooksCorpus + Wikipedia + CC-News + OpenWebText + Stories). Trained for 1M steps.

**Key Benchmarks:**
- Matches RoBERTa on GLUE and SQuAD
- New SOTA on abstractive summarization (CNN/DailyMail, XSum)
- New SOTA on dialogue response generation (ConvAI2)
- New SOTA on abstractive QA (ELI5)

**Denoising Pretraining Objectives (explored multiple):**
1. Token masking (like BERT's MLM)
2. Token deletion (tokens removed, model must decide which positions are missing)
3. Text infilling (random-length spans replaced with single [MASK], model must predict span length)
4. Sentence permutation (sentences shuffled)
5. Document rotation (token chosen at random, document rotated to start with it)

Best combination: **text infilling + sentence permutation**

**Key Innovation:** Generalizes BERT (bidirectional encoder) and GPT (left-to-right decoder) into a single denoising autoencoder framework. The encoder processes corrupted input bidirectionally; the decoder reconstructs the original left-to-right. Particularly strong for generation tasks where BERT is structurally limited.

---

## 12. T5 (Raffel et al., 2019)

| Field            | Detail                                                                |
|------------------|-----------------------------------------------------------------------|
| Paper            | "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transfer Transformer" |
| arXiv            | https://arxiv.org/abs/1910.10683                                      |
| Organization     | Google Research                                                       |
| Authors          | Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu |
| Conference       | JMLR 2020                                                             |
| Architecture     | Encoder-decoder Transformer                                          |

**Parameter Counts:**

| Variant  | Params | Encoder Layers | Decoder Layers | d_model | d_ff  | Heads |
|----------|--------|----------------|----------------|---------|-------|-------|
| T5-Small | 60M    | 6              | 6              | 512     | 2048  | 8     |
| T5-Base  | 220M   | 12             | 12             | 768     | 3072  | 12    |
| T5-Large | 770M   | 24             | 24             | 1024    | 4096  | 16    |
| T5-3B    | 3B     | 24             | 24             | 1024    | 16384 | 32    |
| T5-11B   | 11B    | 24             | 24             | 1024    | 65536 | 128   |

**Training Data -- C4 (Colossal Clean Crawled Corpus):**
- 750GB of cleaned English text from Common Crawl
- Cleaning: deduplicate, remove incomplete sentences, filter offensive/noisy content, remove code
- Pretrained on ~1 trillion tokens total

**Key Benchmarks (T5-11B):**
- GLUE: SOTA across tasks
- SuperGLUE: 90.2 (exceeded human baseline of 89.8 at time of publication)
- SQuAD: SOTA
- CNN/Daily Mail summarization: SOTA
- WMT EN-DE translation: competitive

**Key Innovation -- Text-to-Text Framework:**
Every NLP task is cast as text-in, text-out:
- Classification: "mnli premise: ... hypothesis: ..." -> "entailment"
- Summarization: "summarize: ..." -> summary text
- Translation: "translate English to German: ..." -> German text
- QA: "question: ... context: ..." -> answer text

This unified framework allows a single model, single training procedure, and single decoding process for ALL tasks. No task-specific heads or architectures needed.

**Systematic Study:** The paper is also a massive ablation study comparing pretraining objectives, architectures, data sizes, transfer approaches, and scaling. Key findings:
- Encoder-decoder slightly outperforms decoder-only at equivalent compute
- Span corruption (predicting masked spans, like a variant of MLM) works best as pretraining objective
- More data and larger models consistently help
- Multi-task pretraining + fine-tuning slightly outperforms fine-tuning alone

---

## 13. GPT-3 (Brown et al., 2020)

| Field            | Detail                                                                |
|------------------|-----------------------------------------------------------------------|
| Paper            | "Language Models are Few-Shot Learners"                               |
| arXiv            | https://arxiv.org/abs/2005.14165                                      |
| Organization     | OpenAI                                                                |
| Authors          | Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, + 27 others |
| Conference       | NeurIPS 2020                                                          |
| Architecture     | Decoder-only Transformer (same as GPT-2 with sparse attention in alternating layers) |

**All 8 Model Sizes:**

| Model Name   | Params | Layers | d_model | Heads | d_head | Batch Size |
|-------------|--------|--------|---------|-------|--------|------------|
| GPT-3 Small | 125M   | 12     | 768     | 12    | 64     | 0.5M       |
| GPT-3 Medium| 350M   | 24     | 1024    | 16    | 64     | 0.5M       |
| GPT-3 Large | 760M   | 24     | 1536    | 16    | 96     | 1M         |
| GPT-3 XL    | 1.3B   | 24     | 2048    | 24    | 128    | 1M         |
| GPT-3 2.7B  | 2.7B   | 32     | 2560    | 32    | 80     | 1M         |
| GPT-3 6.7B  | 6.7B   | 32     | 4096    | 32    | 128    | 2M         |
| GPT-3 13B   | 13B    | 40     | 5140    | 40    | 128    | 2M         |
| GPT-3 175B  | 175B   | 96     | 12288   | 96    | 128    | 3.2M       |

All models use context window n_ctx = 2048 tokens.

**Training Data:**

| Dataset         | Tokens  | Weight in Training |
|-----------------|---------|-------------------|
| Common Crawl    | 410B    | 60%               |
| WebText2        | 19B     | 22%               |
| Books1          | 12B     | 8%                |
| Books2          | 55B     | 8%                |
| Wikipedia       | 3B      | 3%                |
| **Total**       | **~499B** | --              |

Model trained on ~300B tokens (due to upsampling high-quality sources: Wikipedia and Books seen 2-3x, Common Crawl <1x). Raw filtered Common Crawl: 570GB.

**Key Benchmarks (GPT-3 175B):**
- CoQA: 85.0 F1 (few-shot), 84.0 (one-shot), 81.5 (zero-shot)
- TriviaQA: 71.2% (few-shot), 68.0% (one-shot), 64.3% (zero-shot)
- LAMBADA: 86.4% accuracy (few-shot) -- huge jump from GPT-2's 63.2%
- HellaSwag: 79.3% (few-shot)
- StoryCloze: 87.7% (few-shot)
- SuperGLUE: 71.8 (few-shot, no fine-tuning)
- Winogrande: 77.7% (few-shot)
- Translation (EN-FR): 32.6 BLEU (few-shot, no parallel training data)
- Arithmetic: Can do 2-digit addition/subtraction with high accuracy

**Key Innovation -- In-Context Learning:**
GPT-3 demonstrated that sufficiently large LMs can learn new tasks at inference time from just a few examples in the prompt, without any gradient updates:
- **Zero-shot:** Task description only ("Translate English to French:")
- **One-shot:** One example + task description
- **Few-shot:** Multiple examples (typically 10-100, limited by context window)

Performance scales smoothly with model size across all three regimes. The 175B model approaches or matches fine-tuned SOTA on some tasks purely from few-shot prompting.

**Significance:** Paradigm shift from "pretrain + fine-tune" to "pretrain + prompt." Showed that scale itself is a form of capability -- emergent abilities appear at sufficient size. Launched the era of prompt engineering and API-based NLP.

---

## Summary Comparison Table

| Model         | Year | Org      | Type     | Params        | Training Data       | Key Innovation                          |
|---------------|------|----------|----------|---------------|---------------------|-----------------------------------------|
| Transformer   | 2017 | Google   | Enc-Dec  | 65M / 213M    | WMT (parallel)      | Self-attention replaces recurrence      |
| ULMFiT        | 2018 | fast.ai  | LSTM     | ~24M (est)    | WikiText-103        | Transfer learning framework for NLP     |
| ELMo          | 2018 | AI2      | biLSTM   | 93.6M         | 1B Word Benchmark   | Contextual word representations         |
| GPT-1         | 2018 | OpenAI   | Dec-only | 117M          | BooksCorpus (5GB)   | Generative pretraining + fine-tuning    |
| BERT          | 2018 | Google   | Enc-only | 110M / 340M   | Books+Wiki (16GB)   | MLM enables deep bidirectional context  |
| Transformer-XL| 2019 | CMU/Goog | Dec-only | 151M / 257M   | WikiText-103        | Segment recurrence + relative pos enc   |
| GPT-2         | 2019 | OpenAI   | Dec-only | 124M -- 1.6B  | WebText (40GB)      | Zero-shot task transfer via LM          |
| XLNet         | 2019 | CMU/Goog | Dec-only | 110M / 340M   | Mixed (158GB)       | Permutation language modeling           |
| RoBERTa       | 2019 | Meta AI  | Enc-only | 355M          | Mixed (160GB)       | BERT was undertrained (better recipe)   |
| ALBERT        | 2019 | Google   | Enc-only | 12M -- 233M   | Books+Wiki (16GB)   | Factorized embed + cross-layer sharing  |
| BART          | 2019 | Meta AI  | Enc-Dec  | 139M / 406M   | Mixed (160GB)       | Denoising seq2seq pretraining           |
| T5            | 2019 | Google   | Enc-Dec  | 60M -- 11B    | C4 (750GB)          | Text-to-text unification of all tasks   |
| GPT-3         | 2020 | OpenAI   | Dec-only | 125M -- 175B  | Mixed (570GB)       | In-context learning / few-shot          |

---

## Key Conceptual Threads

### 1. The Pretraining Objective Spectrum

```
Autoregressive (AR)             Autoencoding (AE)            Permutation (PLM)
P(x_t | x_<t)                  P(x_mask | x_\mask)          E_z[P(x_t | x_z<t)]
─────────────────────           ──────────────────           ──────────────────
GPT-1/2/3                      BERT, RoBERTa, ALBERT        XLNet
Left-to-right only             Bidirectional but             Bidirectional +
Can generate naturally          assumes mask independence     no mask token mismatch
[MASK] not needed               [MASK] not in fine-tuning     Product rule preserved
```

### 2. Scaling Laws Progression

```
2017: 65M (Transformer)    ──►  Proof of concept
2018: 110M (BERT)          ──►  Better pretraining > more params
2019: 1.6B (GPT-2)        ──►  Scale enables zero-shot
2019: 11B (T5)             ──►  Scale + better recipe = SOTA
2020: 175B (GPT-3)         ──►  Scale enables in-context learning
                                 (100x in 3 years)
```

### 3. Data Scaling

```
2017: WMT parallel corpus         ~few GB
2018: BooksCorpus                  ~5GB
2018: Books + Wiki                 ~16GB
2019: WebText                      ~40GB
2019: Mixed (XLNet/RoBERTa)        ~160GB
2019: C4                           ~750GB
2020: GPT-3 training mix           ~570GB (filtered), ~300B tokens consumed
```
