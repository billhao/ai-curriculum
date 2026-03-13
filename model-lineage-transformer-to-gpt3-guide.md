# From Transformer to GPT-3: The Lineage of Language Models (2017–2020)

How twelve models in three years transformed NLP from task-specific architectures to general-purpose language models — tracing the ideas, objectives, and scaling insights that connect the original Transformer to GPT-3's in-context learning.

## Background

**Research lineage** — ordered by the idea each model unlocked:

1. **Attention Is All You Need** (Vaswani et al., Google, 2017) — Pure self-attention replaces recurrence. The architecture substrate everything else builds on. [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. **ULMFiT** (Howard & Ruder, fast.ai/NUI Galway, 2018) — Transfer learning works for NLP. Pretrain a language model, fine-tune on a task. The conceptual blueprint for GPT and BERT. [arxiv.org/abs/1801.06146](https://arxiv.org/abs/1801.06146)
3. **ELMo** (Peters et al., Allen AI, 2018) — Context-dependent word representations. "Bank" means different things in different sentences — embeddings should too. [arxiv.org/abs/1802.05365](https://arxiv.org/abs/1802.05365)
4. **GPT-1** (Radford et al., OpenAI, 2018) — Generative pretraining + discriminative fine-tuning on a Transformer decoder. [cdn.openai.com](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
5. **BERT** (Devlin et al., Google, 2018) — Bidirectional pretraining via masked language modeling. Dominates NLU benchmarks. [arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
6. **Transformer-XL** (Dai et al., CMU/Google, 2019) — Segment-level recurrence breaks the fixed-context bottleneck. [arxiv.org/abs/1901.02860](https://arxiv.org/abs/1901.02860)
7. **GPT-2** (Radford et al., OpenAI, 2019) — Scale up next-token prediction until zero-shot transfer emerges. [cdn.openai.com](https://cdn.openai.com/better-language-models/language-models.pdf)
8. **XLNet** (Yang et al., CMU/Google, 2019) — Permutation language modeling: bidirectional context without BERT's [MASK] mismatch. [arxiv.org/abs/1906.08237](https://arxiv.org/abs/1906.08237)
9. **RoBERTa** (Liu et al., Facebook AI, 2019) — BERT was undertrained. Same architecture, better recipe, much better results. [arxiv.org/abs/1907.11692](https://arxiv.org/abs/1907.11692)
10. **ALBERT** (Lan et al., Google/TTIC, 2019) — Parameter efficiency via factorized embeddings and cross-layer sharing. [arxiv.org/abs/1909.11942](https://arxiv.org/abs/1909.11942)
11. **BART** (Lewis et al., Facebook AI, 2019) — Denoising sequence-to-sequence: bidirectional encoder + autoregressive decoder. [arxiv.org/abs/1910.13461](https://arxiv.org/abs/1910.13461)
12. **T5** (Raffel et al., Google, 2019) — Every NLP task is text-to-text. Massive ablation study of what matters. [arxiv.org/abs/1910.10683](https://arxiv.org/abs/1910.10683)
13. **GPT-3** (Brown et al., OpenAI, 2020) — 175B parameters. In-context learning replaces fine-tuning. [arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)

## The Evolution at a Glance

```
2017  Transformer ─── encoder-decoder, self-attention replaces recurrence
        │
2018  ULMFiT ──────── "pretrain LM, fine-tune on task" works (LSTM-based)
        │
      ELMo ────────── contextual embeddings: same word, different vectors
        │
      GPT-1 ───────── Transformer decoder + generative pretraining
        │
      BERT ────────── Transformer encoder + masked LM → bidirectional
        │
2019  Transformer-XL ─ segment recurrence + relative positions → long context
        │
      GPT-2 ───────── scale decoder → zero-shot emerges
        │
      XLNet ───────── permutation LM: bidirectional + autoregressive
        │
      RoBERTa ─────── same BERT arch, better training recipe
        │
      ALBERT ──────── factorized embeddings + cross-layer sharing
        │
      BART ────────── denoising encoder-decoder
        │
      T5 ──────────── text-to-text framework + massive ablation
        │
2020  GPT-3 ───────── 175B params, in-context few-shot learning
```

## Three Architectural Paradigms

The original Transformer has both an encoder and decoder. The field split it apart and explored each piece independently before reunifying:

```
                    ┌──────────────────────────────────────┐
                    │     Original Transformer (2017)       │
                    │     Encoder-Decoder Architecture      │
                    └──────┬───────────┬───────────┬───────┘
                           │           │           │
              ┌────────────▼──┐  ┌─────▼─────┐  ┌─▼────────────┐
              │  Encoder-only  │  │ Enc-Dec   │  │ Decoder-only  │
              │ (Autoencoding) │  │           │  │(Autoregressive│
              ├───────────────┤  ├───────────┤  ├──────────────┤
              │Bidirectional   │  │Bidi enc + │  │Left-to-right │
              │context, masked │  │L-to-R dec │  │generation    │
              │prediction      │  │           │  │              │
              ├───────────────┤  ├───────────┤  ├──────────────┤
              │ BERT           │  │ T5        │  │ GPT-1/2/3    │
              │ RoBERTa        │  │ BART      │  │ Transformer-XL│
              │ ALBERT         │  │           │  │ XLNet        │
              ├───────────────┤  ├───────────┤  ├──────────────┤
              │Best at: NLU,   │  │Best at:   │  │Best at: text │
              │classification, │  │summarize, │  │generation,   │
              │extractive QA,  │  │translate, │  │completion,   │
              │NER, sentiment  │  │generative │  │zero/few-shot │
              │                │  │QA         │  │prompting     │
              └───────────────┘  └───────────┘  └──────────────┘

Pre-Transformer bridge models (LSTM-based):
  ELMo  ── feature-based contextualizer (biLSTM, not fine-tuned)
  ULMFiT ─ fine-tunable LM transfer (AWD-LSTM)
```

**Why this taxonomy matters:** Encoder-only models see the full input bidirectionally — ideal for understanding tasks. Decoder-only models generate left-to-right — ideal for text generation. Encoder-decoder models get both. The field eventually converged on decoder-only (the GPT line) once scale showed it could handle understanding tasks too.

## Model-by-Model Deep Dive

### 1. ULMFiT — The Transfer Learning Proof (Jan 2018)

**Paper:** Universal Language Model Fine-tuning for Text Classification (Howard & Ruder, fast.ai/NUI Galway, ACL 2018)

ULMFiT is not a Transformer — it uses a 3-layer AWD-LSTM. Its contribution is the **idea**, not the architecture: pretrain a language model on general text, then fine-tune it on a downstream task. Before ULMFiT, NLP trained task-specific models from scratch. Computer vision had been doing transfer learning (ImageNet pretraining) for years; ULMFiT proved it works for language too.

**Architecture:** AWD-LSTM, 3 layers, 400-dim embeddings, 1150 hidden units. Pretrained on WikiText-103 (103M words).

**Three innovations to prevent catastrophic forgetting:**

```
Standard fine-tuning:         ULMFiT fine-tuning:

Pretrained LM                 Pretrained LM
     │                             │
     ▼                        ┌────▼────┐
 Full fine-tune               │ 1. Discriminative fine-tuning:
 (same LR everywhere)        │    different LR per layer
     │                        │    (lower layers = lower LR)
     ▼                        │
 Catastrophic forgetting     │ 2. Slanted triangular LR:
 (loses pretrained           │    quick warmup, slow decay
  knowledge)                  │
                              │ 3. Gradual unfreezing:
                              │    unfreeze from top layer
                              │    down, one per epoch
                              └────▼────┘
                               Stable adaptation
```

**Discriminative fine-tuning** uses different learning rates per layer. Lower layers capture general language features (should change slowly); upper layers capture task-specific features (can change faster). With base LR η:
- Layer 1: η/9
- Layer 2: η/3
- Layer 3: η

**Benchmarks:** 18–24% error reduction on 6 text classification tasks. IMDb error: 4.6%, AG News: 5.01%, DBpedia: 0.80%.

**Why it matters for the lineage:** ULMFiT established the pretrain-then-fine-tune paradigm that GPT-1, BERT, and everything after adopted. The specific techniques (discriminative LR, gradual unfreezing) are still used in fine-tuning Transformers. Without this proof of concept, GPT-1 might not have been attempted.

---

### 2. ELMo — Context-Dependent Representations (Feb 2018)

**Paper:** Deep Contextualized Word Representations (Peters et al., Allen AI, NAACL 2018 Best Paper)

Before ELMo, word embeddings (Word2Vec, GloVe) were **static** — "bank" always got the same vector regardless of context. ELMo showed that different layers of a deep language model capture different linguistic properties, and combining them gives much richer representations.

**Architecture:**

```
Input:  "The river bank was muddy"
          │
     Character CNN          ← handles OOV words, morphology
          │
     biLSTM Layer 1         ← captures syntax (POS, dependencies)
          │
     biLSTM Layer 2         ← captures semantics (word sense, coref)
          │
     Scalar mixing:         ← learned task-specific weights
     ELMo_k = γ * Σᵢ sᵢ * hᵢ

     s = softmax weights per layer (learned)
     γ = scalar (learned)
     h = hidden states from each layer
```

- **Params:** 93.6M (2-layer biLSTM, 4096 units per direction, 512-dim projections)
- **Data:** 1 Billion Word Benchmark (~800M tokens)
- **Forward LSTM reads left-to-right, backward LSTM reads right-to-left** — then concatenated. This is shallow bidirectionality: the two directions don't interact until concatenation. BERT's key insight was that this isn't enough.

**Benchmarks:** New SOTA on 6 tasks. SQuAD F1: 80.8→85.2. SRL F1: 81.4→84.6. Up to 20% relative error reduction.

**How ELMo is used:** Unlike GPT/BERT, ELMo doesn't fine-tune. It's a **feature extractor** — you freeze ELMo, generate contextual embeddings, and feed them to your task-specific model. This is the key limitation that GPT-1 and BERT addressed.

**Why it matters:** ELMo proved that context-dependent representations are strictly better than static embeddings. It also showed that different layers encode different information — a finding that influenced BERT's design and the field's understanding of what deep networks learn.

---

### 3. GPT-1 — Generative Pretraining Meets Transformers (Jun 2018)

**Paper:** Improving Language Understanding by Generative Pre-Training (Radford et al., OpenAI)

GPT-1 married ULMFiT's insight (pretrain, then fine-tune) with the Transformer architecture. Instead of ELMo's "pretrain and extract features," GPT-1 fine-tunes the entire model end-to-end.

**Architecture:**

```
GPT-1 = Transformer decoder only (no encoder)
  12 layers, d_model=768, 12 heads, d_ff=3072
  Context: 512 tokens
  Vocab: BPE with 40K merges
  ~117M params (actual weight count ~124M)
```

Like the GPT-2 124M you trained — GPT-1 is essentially the same architecture at the same scale. The difference is what came before it: GPT-1 was the first to show this approach works.

**Training:**

```
Stage 1: Unsupervised pretraining (language modeling)
  Data: BooksCorpus (~7K unpublished books, ~800M words)
  Objective: standard next-token prediction

Stage 2: Supervised fine-tuning
  Data: task-specific labeled data
  Method: add a linear classification head on top

  Input format (clever trick):
  ┌─────────────────────────────────────────────┐
  │ Classification: [start] text [extract]       │
  │ Entailment:     [start] premise [delim]      │
  │                         hypothesis [extract]  │
  │ Similarity:     [start] text1 [delim] text2   │
  │                         [extract]             │
  │ Multiple choice: [start] context [delim]      │
  │                          answer_i [extract]   │
  │                  (one per answer, compare)     │
  └─────────────────────────────────────────────┘

  Loss = L_task + λ * L_LM   (auxiliary LM loss helps)
```

**Benchmarks:** SOTA on 9 of 12 NLU datasets. Stories Cloze +8.9%, RACE +5.7%.

**What GPT-1 got right:**
- Unsupervised pretraining on a large corpus, then fine-tune with minimal task-specific modifications
- The Transformer decoder is sufficient — no need for encoder
- Auxiliary LM loss during fine-tuning improves generalization

**What GPT-1 got wrong (that BERT fixed):**
- Unidirectional (left-to-right only) — can't attend to future tokens
- For NLU tasks like question answering or NLI, seeing the full context bidirectionally is strictly better

---

### 4. BERT — Bidirectional Pretraining (Oct 2018)

**Paper:** BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin et al., Google, NAACL 2019)

BERT's core argument: GPT-1's left-to-right constraint is unnecessarily restrictive for understanding tasks. If you're classifying sentiment or extracting answer spans, why not look at the full context?

| Variant    | Params | Layers | Hidden | Heads | FF dim |
|------------|--------|--------|--------|-------|--------|
| BERT-Base  | 110M   | 12     | 768    | 12    | 3072   |
| BERT-Large | 340M   | 24     | 1024   | 16    | 4096   |

BERT-Base was explicitly sized to match GPT-1 for fair comparison.

**Data:** BooksCorpus (800M words) + English Wikipedia (2,500M words) = ~16GB total.

**Two pretraining objectives:**

```
1. Masked Language Model (MLM):
   Input:  "The cat [MASK] on the [MASK]"
   Target: "sat", "mat"

   Masking strategy (for 15% of tokens):
     80% → [MASK]    ("The [MASK] sat on the mat")
     10% → random    ("The dog sat on the mat")
     10% → unchanged ("The cat sat on the mat")

   Why not 100% [MASK]? Because [MASK] never appears at fine-tuning time.
   The 10/10 split reduces the pretrain-finetune mismatch.

2. Next Sentence Prediction (NSP):
   Input:  [CLS] sent_A [SEP] sent_B [SEP]
   Target: IsNext / NotNext (50/50)

   Later shown to be unnecessary (RoBERTa) or harmful (ALBERT).
```

**Why MLM enables bidirectionality:**

```
GPT-1 (causal):                    BERT (MLM):
Token 4 attends to: 1, 2, 3       Token 4 attends to: 1, 2, 3, 5, 6, 7
                                   (except masked positions predict from all)

Self-attention mask:               Self-attention mask:
  1 2 3 4 5 6 7                      1 2 3 4 5 6 7
1 ■ □ □ □ □ □ □                    1 ■ ■ ■ ■ ■ ■ ■
2 ■ ■ □ □ □ □ □                    2 ■ ■ ■ ■ ■ ■ ■
3 ■ ■ ■ □ □ □ □                    3 ■ ■ ■ ■ ■ ■ ■
4 ■ ■ ■ ■ □ □ □                    4 ■ ■ ■ ■ ■ ■ ■
5 ■ ■ ■ ■ ■ □ □                    5 ■ ■ ■ ■ ■ ■ ■
6 ■ ■ ■ ■ ■ ■ □                    6 ■ ■ ■ ■ ■ ■ ■
7 ■ ■ ■ ■ ■ ■ ■                    7 ■ ■ ■ ■ ■ ■ ■

■ = can attend    □ = masked         ■ = full attention (no mask)
```

**Benchmarks:**
- GLUE: **80.5** (+7.7 points absolute improvement)
- MultiNLI: **86.7%**
- SQuAD v1.1: **93.2 F1**
- SQuAD v2.0: **83.1 F1**
- New SOTA on all 11 NLP tasks tested

**BERT's trade-off:** By masking random tokens rather than predicting left-to-right, BERT cannot generate text autoregressively. It excels at understanding but is structurally limited for generation. This is the fundamental encoder-vs-decoder trade-off that XLNet, T5, and BART each address differently.

**Two problems BERT introduced that later models fixed:**
1. **Pretrain-finetune mismatch:** [MASK] tokens exist during pretraining but never during fine-tuning. The 80/10/10 masking partially addresses this, but XLNet eliminates it entirely.
2. **Independence assumption:** BERT predicts each masked token independently. If "New" and "York" are both masked, BERT predicts each without conditioning on the other. XLNet fixes this too.

---

### 5. Transformer-XL — Breaking the Context Limit (Jan 2019)

**Paper:** Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context (Dai et al., CMU/Google Brain, ACL 2019)

The vanilla Transformer processes fixed-length segments independently. If your context window is 512 tokens and the answer depends on token 600 tokens back, you're out of luck. Transformer-XL solves this with recurrence at the hidden-state level.

**Params:** ~257M (large config: 18 layers, d_model=1024, 16 heads)

**Two innovations:**

```
1. SEGMENT-LEVEL RECURRENCE:

Vanilla Transformer:
  Segment 1: [tok 1...512]  → hidden states → discard
  Segment 2: [tok 513...1024] → hidden states → discard
  (no information flows between segments)

Transformer-XL:
  Segment 1: [tok 1...512]  → hidden states → CACHE
  Segment 2: [tok 513...1024] → attend to current + cached states

  For layer n, segment τ+1:
    h̃ₙ₋₁ = [SG(h^(τ)ₙ₋₁) ∘ h^(τ+1)ₙ₋₁]     (concat cached + current)
    qₙ = h^(τ+1)ₙ₋₁ Wq                          (query from current only)
    kₙ, vₙ = h̃ₙ₋₁ Wk, h̃ₙ₋₁ Wv                 (key/value from both)

  SG = stop gradient (cached states don't get gradients)

  Effective context length: O(N × L) where N = num layers, L = segment length
  With 18 layers and 512 segment length: up to 9,216 tokens of context

2. RELATIVE POSITIONAL ENCODING:

  Problem: if segment 2 reuses cached hidden states from segment 1,
  absolute position 1 in segment 2 ≠ absolute position 1 in segment 1.

  Solution: encode position as relative distance in the attention score.
  Instead of: Aᵢⱼ = qᵢᵀkⱼ + qᵢᵀpⱼ        (absolute)
  Use:        Aᵢⱼ = qᵢᵀkⱼ + qᵢᵀR(i-j)     (relative distance i-j)

  R(i-j) is a sinusoidal encoding of the relative distance.
```

**Benchmarks:** WikiText-103 perplexity **18.3** (SOTA). enwik8 **0.99** bpc (SOTA). Learns dependencies 80% longer than RNNs, 450% longer than vanilla Transformers. 1,800x faster at evaluation than vanilla (because of caching).

**Why it matters:** Transformer-XL's relative positional encoding became standard in modern LLMs (RoPE in LLaMA is a descendant). Its segment recurrence was directly used as XLNet's backbone. The insight that context length is a critical bottleneck led to the long-context arms race that continues today.

---

### 6. XLNet — Best of Both Worlds (Jun 2019)

**Paper:** XLNet: Generalized Autoregressive Pretraining for Language Understanding (Yang et al., CMU/Google Brain, NeurIPS 2019)

XLNet asked: can we get BERT's bidirectional context without BERT's two problems ([MASK] mismatch and independence assumption)?

- **Params:** XLNet-Base ~110M, XLNet-Large ~340M
- **Data:** 158GB (BooksCorpus+Wiki 13GB, Giga5 16GB, ClueWeb 19GB, Common Crawl 110GB)
- **Backbone:** Transformer-XL (segment recurrence + relative positions)

**The core innovation — Permutation Language Modeling:**

Standard autoregressive: always predict in order 1→2→3→4.
BERT MLM: predict masked tokens independently given all unmasked tokens.
XLNet: predict in a RANDOM order, but autoregressively (each prediction conditions on all previous predictions in that order).

```
Sequence: [x₁, x₂, x₃, x₄]

Permutation 1: predict in order 3→2→4→1
  p(x₃)                        ← no context
  p(x₂|x₃)                    ← sees x₃
  p(x₄|x₃,x₂)                ← sees x₃,x₂
  p(x₁|x₃,x₂,x₄)            ← sees all others

Permutation 2: predict in order 2→4→1→3
  p(x₂)                        ← no context
  p(x₄|x₂)                    ← sees x₂
  p(x₁|x₂,x₄)                ← sees x₂,x₄
  p(x₃|x₂,x₄,x₁)            ← sees all others

Over all permutations, every token sees every other token as context.
But within each permutation, the product rule holds — no independence assumption.
```

**Why this solves BERT's problems:**
1. **No [MASK] tokens needed** — the model always predicts actual tokens, just in different orders. No pretrain-finetune mismatch.
2. **No independence assumption** — within a permutation, each prediction conditions on all prior predictions. If both "New" and "York" need predicting, one will condition on the other.

**Two-stream self-attention** (needed to make permutation LM work):

```
Problem: when predicting x₃ in position 3, the model needs to know:
  - "I'm predicting the token at position 3" (to use positional info)
  - BUT NOT "the token at position 3 is x₃" (that's the answer!)

Solution: two hidden streams per layer:

  Content stream hᵢ: encodes both content AND position (standard)
  Query stream gᵢ:   encodes only position + context (no self-content)

  For predicting xᵢ:
    gᵢ attends to content streams hⱼ of all preceding tokens (in permutation order)
    gᵢ does NOT attend to hᵢ (would leak the answer)
    gᵢ DOES know position i (via the query)

  At fine-tuning: only the content stream is used (query stream is dropped).
```

**Benchmarks:** Outperformed BERT on 20 tasks. GLUE ~90.5. SQuAD v1.1: 95.1 F1. SQuAD v2.0: 90.6 F1. RACE: 81.75.

**The catch:** XLNet used 10x more data (158GB vs 16GB) and more compute than BERT. When RoBERTa later trained BERT's architecture with comparable data and compute, it matched or exceeded XLNet — suggesting that much of XLNet's advantage was resources, not architecture.

---

### 7. RoBERTa — BERT Done Right (Jul 2019)

**Paper:** RoBERTa: A Robustly Optimized BERT Pretraining Approach (Liu et al., Facebook AI)

RoBERTa's contribution is a controlled experiment, not a new architecture. It took BERT-Large (340M params, same architecture) and asked: what if we just train it properly?

| Change                        | BERT-Large           | RoBERTa              |
|-------------------------------|----------------------|----------------------|
| Training data                 | 16GB                 | **160GB** (10x)      |
| Batch size                    | 256 sequences        | **8K sequences**     |
| Training steps                | 1M steps             | **500K** (but bigger) |
| Total tokens seen             | ~137B                | **~2T**              |
| Masking                       | Static (same mask)   | **Dynamic** (new/ep) |
| Next sentence prediction      | Yes                  | **No** (removed)     |
| Short sequence warmup         | Yes (first 90%)      | **No** (full-length) |
| Architecture                  | 24L, 1024H, 16 heads | Same                 |
| Parameters                    | 340M                 | **355M**             |

The 355M vs 340M difference comes from a slightly larger BPE vocabulary (50K vs 30K).

**Data composition:**
- BooksCorpus + Wikipedia: 16GB (same as BERT)
- CC-News: 76GB
- OpenWebText: 38GB
- Stories: 31GB
- Total: **160GB**

**What each change contributed (ablation):**

```
BERT-Large baseline (GLUE):                    80.5
  + dynamic masking:                           +0.1  (small)
  + remove NSP:                                +0.3  (modest)
  + larger batches (8K):                       +0.7  (significant)
  + more data (160GB):                         +2.0+ (largest gain)
  + longer training:                           +1.0+
  ─────────────────────────────────────────────
  RoBERTa final:                               88.5
```

**Benchmarks:** GLUE: **88.5** (vs BERT's 80.5 — same architecture!). SOTA on SQuAD, RACE, SuperGLUE, XNLI.

**The lesson:** Before claiming architectural innovation, try training harder. RoBERTa matched XLNet (which had a genuinely novel objective) by simply scaling BERT's training recipe. This became a recurring theme: many "improvements" in 2018-2019 were confounded by different training budgets.

---

### 8. ALBERT — Efficiency Through Sharing (Sep 2019)

**Paper:** ALBERT: A Lite BERT for Self-supervised Learning of Language Representations (Lan et al., Google/TTIC, ICLR 2020)

ALBERT asked: can we get BERT-level or better performance with far fewer parameters?

| Variant         | Params | Layers | Hidden | Embedding | Heads |
|-----------------|--------|--------|--------|-----------|-------|
| BERT-Large      | 340M   | 24     | 1024   | 1024      | 16    |
| ALBERT-Base     | 12M    | 12     | 768    | 128       | 12    |
| ALBERT-Large    | 18M    | 24     | 1024   | 128       | 16    |
| ALBERT-xLarge   | 60M    | 24     | 2048   | 128       | 16    |
| ALBERT-xxLarge  | 233M   | 12     | 4096   | 128       | 64    |

ALBERT-xxLarge: 233M params, **better** than BERT-Large's 340M on every benchmark.

**Two parameter reduction techniques:**

```
1. FACTORIZED EMBEDDING PARAMETERIZATION:

   BERT: embedding matrix is V × H (vocab × hidden)
     V=30K, H=1024 → 30.7M params just for embeddings

   ALBERT: decompose into V × E then E × H
     V=30K, E=128, H=4096 → 30K×128 + 128×4096 = 4.4M params
     (was 122.9M at H=4096 → 96% reduction in embedding params)

   Why this works: embeddings capture context-independent meaning;
   hidden states capture context-dependent meaning. These live in
   different spaces — forcing them to share dimension H is wasteful.

2. CROSS-LAYER PARAMETER SHARING:

   BERT: each of 24 layers has its own attention + FFN weights
   ALBERT: ALL layers share the same weights

   Layer 1 weights = Layer 2 weights = ... = Layer 24 weights

   The input to each layer is different (residual connections ensure this),
   so the same weights produce different transformations at each depth.
   Think of it as a recurrent application of the same transformation.

   Effect on hidden state distance between layers:
   BERT:   oscillates (each layer does something different)
   ALBERT: converges smoothly (same transform applied repeatedly)
```

**SOP vs NSP — a cleaner sentence-level objective:**

```
NSP (BERT):                          SOP (ALBERT):
  Positive: [sent A] [sent B]         Positive: [sent A] [sent B]  (correct order)
  Negative: [sent A] [random sent]    Negative: [sent B] [sent A]  (reversed order)

  NSP conflates topic detection       SOP forces the model to learn
  with coherence detection.            inter-sentence coherence only.
  A random sentence is trivially       Reversed sentences share the
  off-topic — model learns topic,      same topic — model must learn
  not coherence.                       ordering/logic.

Cross-test:
  NSP model → SOP task: 52.0% (random chance)
  SOP model → NSP task: 78.9% (coherence subsumes topic)
```

**Benchmarks (ALBERT-xxLarge v2):** GLUE: **90.9**. SQuAD v1.1: 94.6 F1. SQuAD v2.0: 89.8 F1. RACE: **86.5** (SOTA).

**Caveat:** ALBERT has fewer parameters but not necessarily less compute. The shared weights still execute 12–24 times. ALBERT-xxLarge is actually **slower** than BERT-Large due to its much larger hidden dimension (4096 vs 1024). The parameter savings help with memory and communication in distributed training, not with FLOPs.

---

### 9. BART — Denoising Sequence-to-Sequence (Oct 2019)

**Paper:** BART: Denoising Sequence-to-Sequence Pre-training (Lewis et al., Facebook AI, ACL 2020)

BART combines BERT's bidirectional encoder with GPT's autoregressive decoder into a single model. The pretraining objective: corrupt the input, then reconstruct the original.

- **Params:** BART-Base 139M (6+6 layers), BART-Large 406M (12+12 layers, d=1024)
- **Data:** Same as RoBERTa (~160GB)

**Architecture:**

```
Corrupted input                    Original text
"The <mask> sat on the <mask>"  →  "The cat sat on the mat"

     ┌─────────────────────┐
     │  Bidirectional       │
     │  Encoder             │  ← sees corrupted input, full attention
     │  (like BERT)         │
     └──────────┬──────────┘
                │ cross-attention
     ┌──────────▼──────────┐
     │  Autoregressive      │
     │  Decoder             │  ← generates original text left-to-right
     │  (like GPT)          │
     └─────────────────────┘
```

**Noise functions explored (ablation):**

| Corruption Method    | How it works                                  | Effect              |
|---------------------|-----------------------------------------------|---------------------|
| Token masking       | Replace tokens with [MASK] (like BERT)        | Good for NLU        |
| Token deletion      | Delete tokens (model must figure out WHAT'S missing) | Better than masking |
| Text infilling      | Replace spans with single [MASK] (model must predict length) | Best for generation |
| Sentence permutation| Shuffle sentence order                        | Helps summarization |
| Document rotation   | Rotate document to start at random token      | Slightly harmful    |

**Best combination:** Text infilling (Poisson λ=3 span lengths) + sentence permutation.

**Why text infilling > token masking:** When you replace a 3-token span with one [MASK], the model must predict both the content AND the length. This is a harder task that forces deeper understanding.

**Benchmarks:** Matches RoBERTa on GLUE/SQuAD. New SOTA on CNN/DailyMail (+0.4 R1), XSum (+6 ROUGE), ConvAI2 dialogue, ELI5 QA.

**Why BART matters:** It showed that encoder-decoder pretraining produces a single model that's competitive for both understanding (matching BERT/RoBERTa) AND generation (much better at summarization, translation). T5 takes this further.

---

### 10. T5 — Text-to-Text Unification (Oct 2019)

**Paper:** Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transfer Transformer (Raffel et al., Google, JMLR 2020)

T5's contribution is twofold: (1) a unifying framework that casts every NLP task as text-to-text, and (2) the most thorough ablation study of its era, testing architectures, objectives, data, and scale systematically.

| Variant  | Params | Enc Layers | Dec Layers | d_model | d_ff    | Heads |
|----------|--------|------------|------------|---------|---------|-------|
| Small    | 60M    | 6          | 6          | 512     | 2,048   | 8     |
| Base     | 220M   | 12         | 12         | 768     | 3,072   | 12    |
| Large    | 770M   | 24         | 24         | 1,024   | 4,096   | 16    |
| 3B       | 3B     | 24         | 24         | 1,024   | 16,384  | 32    |
| 11B      | 11B    | 24         | 24         | 1,024   | 65,536  | 128   |

**Data — C4 (Colossal Clean Crawled Corpus):** ~750GB of cleaned English text from Common Crawl. Deduplicated, language-filtered, quality-filtered. This was a major contribution in itself — C4 became a standard pretraining corpus.

**The text-to-text framework:**

```
Every task uses the same format: text in → text out

Classification:
  Input:  "mnli premise: A man inspects the uniform. hypothesis: The man is sleeping."
  Output: "contradiction"

Summarization:
  Input:  "summarize: State authorities dispatched 120 firefighters..."
  Output: "Firefighters were dispatched to contain the blaze."

Translation:
  Input:  "translate English to German: That is good."
  Output: "Das ist gut."

Regression (STS-B):
  Input:  "stsb sentence1: The bird is bathing. sentence2: A bird is bathing."
  Output: "5.0"

Same model, same loss (cross-entropy on output tokens), same decoding.
No task-specific heads. No task-specific architectures.
```

**Pretraining objective — Span corruption:**

```
Original:  "Thank you for inviting me to your party last week"
Corrupted: "Thank you <X> me to your party <Y> week"
Target:    "<X> for inviting <Y> last <Z>"

- Replace random spans (not individual tokens) with sentinel tokens
- Target only contains the missing spans (much shorter than full input)
- Span lengths drawn from geometric distribution (mean 3 tokens)
- 15% of tokens corrupted
```

**Key findings from the ablation study:**

1. **Encoder-decoder slightly beats decoder-only at equal compute** — when both have the same number of FLOPs, encoder-decoder produces better results on NLU tasks. This is because half the parameters process the input bidirectionally.

2. **Span corruption beats individual token masking** — predicting contiguous spans is a harder, more informative task.

3. **More data always helps** — no sign of diminishing returns up to 750GB.

4. **More parameters always help** — T5-11B consistently beats T5-3B which beats T5-Large.

5. **Pretraining on domain-specific data helps for that domain** — but general pretraining is more robust.

**Benchmarks (T5-11B):** SuperGLUE: **90.2** (exceeded human baseline of 89.8). GLUE, SQuAD, CNN/DailyMail all SOTA.

---

### 11. GPT-3 — Scale Is All You Need (May 2020)

**Paper:** Language Models are Few-Shot Learners (Brown et al., OpenAI, NeurIPS 2020)

GPT-3 is not architecturally novel — it's GPT-2 scaled 100x. The contribution is demonstrating that sufficient scale produces **in-context learning**: the ability to perform tasks from a few examples in the prompt, with no gradient updates.

**Full model family:**

| Model        | Params | Layers | d_model | Heads | d_head | Batch   |
|-------------|--------|--------|---------|-------|--------|---------|
| Small       | 125M   | 12     | 768     | 12    | 64     | 0.5M    |
| Medium      | 350M   | 24     | 1,024   | 16    | 64     | 0.5M    |
| Large       | 760M   | 24     | 1,536   | 16    | 96     | 1M      |
| XL          | 1.3B   | 24     | 2,048   | 24    | 128    | 1M      |
| 2.7B        | 2.7B   | 32     | 2,560   | 32    | 80     | 1M      |
| 6.7B        | 6.7B   | 32     | 4,096   | 32    | 128    | 2M      |
| 13B         | 13B    | 40     | 5,140   | 40    | 128    | 2M      |
| 175B (davinci) | 175B | 96    | 12,288  | 96    | 128    | 3.2M    |

Context: 2048 tokens. Alternating dense and locally banded sparse attention layers.

**Training data (300B tokens consumed):**

| Dataset       | Raw tokens | Sampling weight | Epochs |
|--------------|------------|-----------------|--------|
| Common Crawl | 410B       | 60%             | 0.44   |
| WebText2     | 19B        | 22%             | 2.9    |
| Books1       | 12B        | 8%              | 1.9    |
| Books2       | 55B        | 8%              | 0.43   |
| Wikipedia    | 3B         | 3%              | 3.4    |

High-quality sources (WebText2, Books, Wikipedia) are upsampled 2-3x. Common Crawl is seen less than once. This data quality weighting was crucial.

**In-context learning:**

```
Zero-shot:    Task description → answer
              "Translate English to French: cheese →"  "fromage"

One-shot:     One example → answer
              "sea otter → loutre de mer. cheese →"    "fromage"

Few-shot:     K examples → answer (K=10-100)
              "sea otter → loutre de mer.
               peppermint → menthe poivrée.
               cheese →"                               "fromage"

No gradient updates at any point. The model "learns" from the prompt alone.
```

**How in-context learning scales with model size:**

```
Accuracy
  ▲
  │           ╱ few-shot (K=64)
  │         ╱
  │       ╱─── one-shot
  │     ╱
  │   ╱───── zero-shot
  │ ╱
  │╱
  └──────────────────────────────► Model size (log scale)
  0.1B    1B     10B     175B

Small models: few-shot ≈ zero-shot (can't use examples)
Large models: few-shot >> zero-shot (emergent ability)
```

**Selected benchmarks (175B):**

| Task          | Zero-shot | One-shot | Few-shot | SOTA (fine-tuned) |
|---------------|-----------|----------|----------|-------------------|
| LAMBADA       | 76.2%     | 72.5%    | 86.4%    | 68.0%             |
| HellaSwag     | 78.9%     | 78.1%    | 79.3%    | 85.6%             |
| TriviaQA      | 64.3%     | 68.0%    | 71.2%    | —                 |
| SuperGLUE     | —         | —        | 71.8     | 89.0              |

GPT-3 few-shot **exceeds** the previous fine-tuned SOTA on LAMBADA. On most other tasks, it's competitive but below fine-tuned models — closing the gap without any training.

**The paradigm shift:**

```
2018 (GPT-1):  Pretrain → Fine-tune per task    (need labeled data)
2019 (GPT-2):  Pretrain → Zero-shot             (works sometimes)
2020 (GPT-3):  Pretrain → Few-shot prompting     (works broadly)
```

## The Full Comparison

| Model          | Year | Type         | Params        | Data          | Objective                | Key Innovation                          |
|----------------|------|------------- |---------------|---------------|--------------------------|----------------------------------------|
| Transformer    | 2017 | Enc-Dec      | 65M / 213M    | WMT (4.5M)   | Supervised seq2seq       | Self-attention replaces recurrence      |
| ULMFiT         | 2018 | LSTM         | ~24M          | WikiText-103  | LM → fine-tune           | Transfer learning for NLP               |
| ELMo           | 2018 | biLSTM       | 94M           | 1B Word       | biLM (feature extract)   | Contextual word representations         |
| GPT-1          | 2018 | Decoder      | 117M          | BooksCorpus   | Causal LM → fine-tune    | Generative pretraining + Transformer    |
| BERT           | 2018 | Encoder      | 110M / 340M   | 16GB          | MLM + NSP               | Bidirectional pretraining               |
| Transformer-XL | 2019 | Decoder      | 257M          | WikiText-103  | Causal LM                | Segment recurrence + relative pos       |
| GPT-2          | 2019 | Decoder      | 124M–1.5B     | WebText 40GB  | Causal LM                | Scale → zero-shot emergence             |
| XLNet          | 2019 | Decoder      | 110M / 340M   | 158GB         | Permutation LM           | Bidirectional without [MASK] mismatch   |
| RoBERTa        | 2019 | Encoder      | 355M          | 160GB         | MLM (no NSP)             | Better training recipe for BERT         |
| ALBERT         | 2019 | Encoder      | 12M–233M      | 16GB          | MLM + SOP               | Factorized embeddings + weight sharing  |
| BART           | 2019 | Enc-Dec      | 139M / 406M   | 160GB         | Denoising                | Bidirectional enc + autoregressive dec  |
| T5             | 2019 | Enc-Dec      | 60M–11B       | C4 750GB      | Span corruption          | Text-to-text framework                  |
| GPT-3          | 2020 | Decoder      | 125M–175B     | 300B tokens   | Causal LM                | In-context few-shot learning via scale  |

## Five Key Insights from this Lineage

**1. Transfer learning was the first breakthrough, not architecture.**
ULMFiT proved the concept with LSTMs. GPT-1 proved it with Transformers. BERT proved it for bidirectional models. The architecture mattered less than the pretraining paradigm.

**2. Bidirectionality helps understanding; unidirectionality enables generation.**
BERT dominates NLU by seeing the full input. GPT dominates generation by predicting left-to-right. XLNet, BART, and T5 each found different ways to get both. The field eventually chose decoder-only + scale.

**3. Training recipe matters as much as architecture.**
RoBERTa proved that BERT with 10x data, bigger batches, and dynamic masking matches architecturally superior models (XLNet). Before claiming a new architecture works, try training the old one better.

**4. Scale produces emergent capabilities.**
GPT-2 showed zero-shot emergence. GPT-3 showed few-shot emergence. These capabilities don't exist at smaller scales and can't be predicted from loss curves alone. The model sizes where they appear are "emergence thresholds."

**5. The pretraining objective shapes what the model can do.**
- Causal LM (GPT): best for generation, eventually for everything at scale
- MLM (BERT): best for NLU with fine-tuning, can't generate
- Span corruption (T5): good at both, but encoder-decoder is more complex
- Permutation LM (XLNet): elegant but complex, overtaken by brute-force scaling

## Key Papers

1. Vaswani et al., "Attention Is All You Need" (2017) — [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. Howard & Ruder, "Universal Language Model Fine-tuning for Text Classification" (2018) — [arxiv.org/abs/1801.06146](https://arxiv.org/abs/1801.06146)
3. Peters et al., "Deep Contextualized Word Representations" (2018) — [arxiv.org/abs/1802.05365](https://arxiv.org/abs/1802.05365)
4. Radford et al., "Improving Language Understanding by Generative Pre-Training" (2018) — [cdn.openai.com](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
5. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers" (2018) — [arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
6. Dai et al., "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" (2019) — [arxiv.org/abs/1901.02860](https://arxiv.org/abs/1901.02860)
7. Radford et al., "Language Models are Unsupervised Multitask Learners" (2019) — [cdn.openai.com](https://cdn.openai.com/better-language-models/language-models.pdf)
8. Yang et al., "XLNet: Generalized Autoregressive Pretraining" (2019) — [arxiv.org/abs/1906.08237](https://arxiv.org/abs/1906.08237)
9. Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach" (2019) — [arxiv.org/abs/1907.11692](https://arxiv.org/abs/1907.11692)
10. Lan et al., "ALBERT: A Lite BERT for Self-supervised Learning" (2019) — [arxiv.org/abs/1909.11942](https://arxiv.org/abs/1909.11942)
11. Lewis et al., "BART: Denoising Sequence-to-Sequence Pre-training" (2019) — [arxiv.org/abs/1910.13461](https://arxiv.org/abs/1910.13461)
12. Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transfer Transformer" (2019) — [arxiv.org/abs/1910.10683](https://arxiv.org/abs/1910.10683)
13. Brown et al., "Language Models are Few-Shot Learners" (2020) — [arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
