# Long Context Mechanisms for LLMs

A guide to how language models handle long sequences — from the quadratic attention wall to million-token context windows, and the engineering that bridges the gap.

## Background

**Foundational paper**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., Google, 2017) — introduced sinusoidal positional encodings and the self-attention mechanism whose O(N²) cost defines the long-context problem.

**Research lineage** — long context capability builds on a chain of work:

1. **Sinusoidal Positional Encodings** (Vaswani et al., 2017) — Fixed sin/cos functions to inject position information. The original Transformer used 512-token sequences. Theoretically generalizes to longer sequences, but in practice fails beyond training length.

2. **Sparse Transformers** (Child et al., OpenAI, 2019) — [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509). First work to reduce attention from O(N²) to O(N√N) using sparse factorized patterns. Showed you don't need full attention over every token pair.

3. **Longformer** (Beltagy et al., AI2, 2020) — [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150). Combined sliding window local attention with task-specific global attention. Scaled to 4,096 tokens, then 16K. Introduced the idea that most tokens only need local context, with a few global "anchor" tokens.

4. **BigBird** (Zaheer et al., Google, 2020) — [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062). Proved theoretically that sparse attention with random + local + global components is a universal approximator of sequence functions. Combined three patterns: sliding window, global tokens, and random attention.

5. **RoPE / RoFormer** (Su et al., 2021) — [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864). Introduced Rotary Position Embeddings — encoding position via rotation in 2D subspaces of the embedding. Became the dominant positional encoding for modern LLMs (Llama, Mistral, Qwen, DeepSeek).

6. **ALiBi** (Press et al., 2021) — [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409). No positional embeddings at all — just add a linear distance penalty to attention scores. Demonstrated strong length extrapolation: train on 1K, test on 2K+ with no quality loss.

7. **FlashAttention** (Dao et al., Stanford, 2022) — [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135). Reduced attention memory from O(N²) to O(N) via tiling and kernel fusion, without approximation. The single most impactful systems paper for long context — made 32K+ context practical on a single GPU.

8. **Position Interpolation (PI)** (Chen et al., Meta, 2023) — [Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595). Linearly downscaled position indices to extend Llama from 2K to 32K with minimal fine-tuning. First practical method to extend a pretrained model's context.

9. **NTK-Aware Interpolation** (Reddit user "bloc97", 2023) — Adjusted the RoPE base frequency instead of linearly interpolating positions. Spread interpolation pressure across frequency dimensions — high frequencies less, low frequencies more. Required no fine-tuning.

10. **YaRN** (Peng et al., 2023) — [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071). Combined NTK scaling with attention temperature correction and dimension-specific interpolation. Extended Llama 2 from 4K to 128K with only 400 steps of fine-tuning. The method used by most open-source models (Qwen, DeepSeek, Llama 3).

11. **Ring Attention** (Liu et al., UC Berkeley, 2023) — [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889). Distributed long sequences across GPUs in a ring topology, overlapping KV block communication with computation. Enabled context scaling proportional to the number of devices — 8 GPUs = 8x context length.

12. **FlashAttention-2** (Dao, 2023) — [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691). Achieved 50-73% of theoretical peak FLOPs on A100, 2x faster than FlashAttention-1. Became the standard attention backend.

13. **FlashAttention-3** (Dao et al., 2024) — [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608). Exploited Hopper GPU features: warp specialization, asynchronous TMA, FP8 matmuls. Reached 740 TFLOPS (75% H100 utilization) in FP16, 1.2 PFLOPS in FP8.

14. **LongRoPE / LongRoPE2** (Microsoft, 2024-2025) — [LongRoPE2: Near-Lossless LLM Context Window Scaling](https://arxiv.org/abs/2502.20082). Extended LLaMA-3-8B to 128K effective context using only 10B tokens (80x fewer than Meta's approach). Key insight: insufficient training in higher RoPE dimensions causes out-of-distribution issues; evolutionary search finds optimal rescaling.

15. **Llama 4 iRoPE** (Meta, 2025) — Interleaved RoPE and NoPE (no positional encoding) layers in a 3:1 local:global ratio. Local attention uses RoPE in non-overlapping chunks; global attention has no positional encoding. Achieved 10M token context window. Inference-time temperature scaling enhances length generalization.

**The key insight**: Long context is not one problem but three — positional encoding must generalize beyond training length, attention computation must be affordable at scale, and KV cache memory must fit in GPU RAM. Modern solutions attack all three simultaneously.

## Key Terms

**Positional encoding (PE)**: A mechanism to inject sequence order information into a transformer. Without PE, attention is permutation-invariant — "the cat sat" and "sat cat the" would produce identical representations. PEs can be absolute (sinusoidal, learned) or relative (RoPE, ALiBi).

**Context window**: The maximum number of tokens a model can process in a single forward pass. Determined by the positional encoding's range and the available memory for attention computation and KV cache. Your GPT-2 124M used a 1024-token context window.

**KV cache**: During autoregressive generation, the key and value tensors from all previous tokens are cached so they don't need recomputation. Each new token only computes its own Q, K, V, then attends to the full cached K, V from all prior tokens. Cache size grows linearly with sequence length but can dominate GPU memory at long contexts.

**Needle-in-a-haystack (NIAH)**: An evaluation protocol that embeds a specific fact ("needle") at various positions within a long document ("haystack") and tests whether the model can retrieve it. Sweeps over both haystack length and needle depth. Modern models score near-perfect on simple NIAH — but the test is deceptively easy.

**Lost-in-the-middle**: The finding (Liu et al., 2023) that models perform best when relevant information is at the beginning or end of the context, but degrade when it's buried in the middle. Creates a U-shaped accuracy curve vs. position. Partially addressed by improved positional encodings but still measurable in 2026.

**Context rot**: Performance degradation as input context length increases, even when the context window isn't full. Driven by three compounding mechanisms: lost-in-the-middle effect, attention dilution, and distractor interference. More pronounced on complex tasks.

**Extrapolation problem**: The challenge of using a model at sequence lengths longer than it was trained on. Sinusoidal and learned PEs fail catastrophically. RoPE degrades gracefully but still loses quality. ALiBi was specifically designed for extrapolation.

## Why Context Length Is Hard

### The Quadratic Wall

Self-attention computes a score between every pair of tokens. For sequence length N:

```
Attention cost:

  Operations:     N² dot products per head per layer
  Memory (naive): N² floats for the attention matrix

  Sequence    Attention       Attention matrix
  length      operations      memory (fp16, 1 head)
  ─────────── ─────────────── ──────────────────────
  1K          1M              2 MB
  4K          16M             32 MB
  32K         1B              2 GB
  128K        16B             32 GB
  1M          1T              2 TB    ← impossible
```

The 1M-token attention matrix alone would require 2 TB — more than any GPU's memory. This is why FlashAttention's memory reduction (materialize only O(N)-sized tiles, never the full N² matrix) was transformative.

### KV Cache Memory Scaling

During generation, every past token's K and V vectors must be stored. The KV cache size for a single sequence is:

```
KV cache = 2 × n_layers × n_kv_heads × d_head × seq_len × bytes_per_param

Example: Llama 3.1 70B (GQA with 8 KV heads)
  n_layers = 80, n_kv_heads = 8, d_head = 128, dtype = fp16 (2 bytes)

  Seq len    KV cache per sequence
  ────────── ──────────────────────
  4K         655 MB
  32K        5.2 GB
  128K       20.9 GB
  512K       83.9 GB    ← exceeds H100 80GB
  1M         167.8 GB   ← needs 3x H100s just for KV cache
```

Now compare with your GPT-2 124M (12 layers, 12 heads, d_head=64, fp16):

```
GPT-2 124M KV cache:
  2 × 12 × 12 × 64 × seq_len × 2 bytes = 36,864 × seq_len bytes

  Seq len    KV cache
  ────────── ─────────
  1K         36 MB
  4K         144 MB     ← trivial on any GPU
  128K       4.6 GB     ← still fits on H800
```

Small models have trivial KV caches. The problem explodes with scale:

```
KV cache at 128K context (fp16):

  Model               Layers  KV heads  d_head  KV cache
  ─────────────────── ─────── ──────── ─────── ─────────
  GPT-2 124M          12      12        64      4.6 GB
  Llama 3 8B          32      8         128     4.2 GB
  Llama 3 70B         80      8         128     20.9 GB
  DeepSeek-V3 671B    61      128(MLA)  128     ~5 GB*

  * MLA compresses KV into low-rank latent — 93% smaller than full MHA
```

DeepSeek-V2/V3's MLA (Multi-head Latent Attention) is striking here — it compresses KV into a low-rank latent vector and decompresses on the fly, achieving a 671B model's KV cache that's smaller than Llama 8B's. You saw this in the MoE guide; this is the same technique viewed through the KV cache lens.

### The Three-Way Tradeoff

```
Long context requires solving all three simultaneously:

  ┌─────────────────────┐     ┌──────────────────────┐     ┌────────────────────┐
  │ Positional Encoding │     │ Attention Compute     │     │ KV Cache Memory    │
  │                     │     │                       │     │                    │
  │ Must generalize     │     │ O(N²) is too slow     │     │ Grows linearly     │
  │ beyond training     │     │ for N > 32K           │     │ but constant is    │
  │ length              │     │                       │     │ large for big      │
  │                     │     │                       │     │ models             │
  │ Solutions:          │     │ Solutions:             │     │ Solutions:         │
  │ • RoPE scaling      │     │ • FlashAttention      │     │ • GQA / MQA        │
  │ • ALiBi             │     │ • Sparse attention    │     │ • MLA              │
  │ • iRoPE             │     │ • Ring attention      │     │ • KV quantization  │
  │ • YaRN              │     │ • Linear attention    │     │ • Token eviction   │
  └─────────────────────┘     └──────────────────────┘     └────────────────────┘
```

## Positional Encodings Deep Dive

### Learned Positional Embeddings

The simplest approach: learn a separate embedding vector for each position. Used by GPT-2 (what you trained) and BERT.

```python
# From your GPT-2 training (nanoGPT):
self.wpe = nn.Embedding(block_size, n_embd)  # block_size = 1024
# Position 0 gets one learned vector, position 1 gets another, etc.
# Total: 1024 × 768 = 786,432 learned parameters
```

**Problem**: Cannot extrapolate. Position 1025 has no embedding. Extending context means adding new parameters and fine-tuning. The model has never seen these positions during training.

### Sinusoidal Positional Encodings (Vaswani et al., 2017)

The original Transformer used fixed (not learned) sin/cos functions at different frequencies:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

where:  pos = token position (0, 1, 2, ...)
        i   = dimension index (0, 1, ..., d_model/2 - 1)
        d_model = embedding dimension
```

Numerical walkthrough for d_model = 4, position 3:

```
dim 0 (i=0): sin(3 / 10000^(0/4)) = sin(3)       =  0.141
dim 1 (i=0): cos(3 / 10000^(0/4)) = cos(3)       = -0.990
dim 2 (i=1): sin(3 / 10000^(2/4)) = sin(3/100)   =  0.030
dim 3 (i=1): cos(3 / 10000^(2/4)) = cos(3/100)   =  1.000

PE(3) = [0.141, -0.990, 0.030, 1.000]
```

Low dimensions (small i) oscillate rapidly — they encode fine-grained position. High dimensions (large i) oscillate slowly — they encode coarse position. The key insight: the relative shift between PE(pos) and PE(pos+k) can be represented as a linear transformation, enabling the model to learn relative position patterns.

**Problem**: Still struggles with extrapolation in practice. The model has never seen the specific PE values for positions beyond training length.

### RoPE — Rotary Position Embeddings (Su et al., 2021)

RoPE is the dominant positional encoding in modern LLMs (Llama, Mistral, Qwen, DeepSeek, Gemma). It encodes position by rotating query and key vectors in 2D subspaces.

**Core idea**: Instead of adding position information to the embeddings, rotate them. The angle of rotation encodes position. When you compute the dot product between a rotated query and a rotated key, the result depends only on the relative distance between their positions.

**Mathematical derivation:**

Consider a pair of dimensions (q₁, q₂) from the query vector at position m. Apply a 2D rotation by angle mθ:

```
┌ q'₁ ┐   ┌ cos(mθ)  -sin(mθ) ┐ ┌ q₁ ┐
│     │ = │                    │ │    │
└ q'₂ ┘   └ sin(mθ)   cos(mθ) ┘ └ q₂ ┘
```

Same rotation for the key at position n with angle nθ. The dot product between rotated q and rotated k:

```
q'ᵀk' = (R(mθ)q)ᵀ(R(nθ)k) = qᵀR((m-n)θ)k
```

The rotation matrices compose: R(mθ)ᵀR(nθ) = R((n-m)θ). The absolute positions m, n vanish — only the relative distance (m-n) remains. This is the key property: **relative position encoding through absolute rotation**.

**Full formulation**: The d-dimensional embedding is split into d/2 pairs, each rotated at a different frequency:

```
θᵢ = base^(-2i/d)     where base = 10000, i = 0, 1, ..., d/2 - 1

For position m, pair i:
  ┌ q'₂ᵢ   ┐   ┌ cos(mθᵢ)  -sin(mθᵢ) ┐ ┌ q₂ᵢ   ┐
  │         │ = │                       │ │       │
  └ q'₂ᵢ₊₁ ┘   └ sin(mθᵢ)   cos(mθᵢ) ┘ └ q₂ᵢ₊₁ ┘
```

**Numerical walkthrough**: d=8, base=10000, position m=42, pair i=1:

```
θ₁ = 10000^(-2/8) = 10000^(-0.25) = 1/10 = 0.1

Rotation angle = m × θ₁ = 42 × 0.1 = 4.2 radians

cos(4.2) = -0.490,  sin(4.2) = -0.872

If q₂ = 1.5, q₃ = 0.8:
  q'₂ = 1.5 × (-0.490) - 0.8 × (-0.872) = -0.735 + 0.698 = -0.037
  q'₃ = 1.5 × (-0.872) + 0.8 × (-0.490) = -1.308 - 0.392 = -1.700
```

**Wavelength perspective**: Each pair i has wavelength λᵢ = 2π × base^(2i/d). This is the number of positions for one full rotation cycle:

```
d = 128, base = 10000:

  Pair i     θᵢ              Wavelength λᵢ      Interpretation
  ────────── ──────────────── ────────────────── ──────────────────
  i = 0      1.0             6.28               encodes exact position
  i = 16     0.01            628                encodes ~paragraph
  i = 32     0.0001          62,832             encodes ~chapter
  i = 63     ~0.000001       ~6.28M             encodes ~book-scale
```

Low-frequency pairs (high i) rotate slowly — they distinguish positions far apart. High-frequency pairs (low i) rotate fast — they distinguish adjacent positions. This multi-scale encoding is why RoPE works so well.

**The extrapolation problem**: When a model trained at 4K context sees position 100K, the high-frequency pairs encounter rotation angles they've never seen (but they cycle frequently, so this is OK). The low-frequency pairs, however, see rotation angles completely outside their training range — and they haven't completed even one full rotation during training. This is where RoPE scaling methods come in.

### ALiBi — Attention with Linear Biases (Press et al., 2021)

ALiBi takes a radically different approach: no positional encoding at all. Instead, it adds a linear distance penalty directly to the attention scores.

```
Standard attention:  softmax(QKᵀ / √d)
ALiBi attention:     softmax(QKᵀ / √d + m · bias_matrix)

bias_matrix[i][j] = -|i - j|    (negative distance between positions)

For head h, the slope m_h forms a geometric sequence:
  m_h = 2^(-8/H) × 2^(-8h/H)   where H = total number of heads

Example with H = 8 heads:
  Head 0: m = 2^(-1)  = 0.500    (strong recency bias)
  Head 1: m = 2^(-2)  = 0.250
  Head 2: m = 2^(-3)  = 0.125
  Head 3: m = 2^(-4)  = 0.0625
  Head 4: m = 2^(-5)  = 0.03125
  Head 5: m = 2^(-6)  = 0.01563
  Head 6: m = 2^(-7)  = 0.00781
  Head 7: m = 2^(-8)  = 0.00391  (weak recency bias — attends far)
```

For a 4-token sequence, head 0 (m=0.5), the bias matrix is:

```
        key 0   key 1   key 2   key 3
query 0 [ 0.0                        ]
query 1 [-0.5    0.0                  ]
query 2 [-1.0   -0.5    0.0           ]
query 3 [-1.5   -1.0   -0.5    0.0   ]
```

**Why ALiBi extrapolates**: The bias is a simple linear function of distance. At position 100K, the bias for attending to position 0 is just -100K × m. No rotation angles, no unseen embedding values — just arithmetic. The model learns during training how to use these distance signals, and the pattern extends naturally.

**Tradeoff**: ALiBi doesn't encode position as richly as RoPE. It only provides relative distance information and encodes an inherent recency bias. Models using ALiBi tend to have lower perplexity on short sequences but their simplicity limits fine-grained positional reasoning.

**Used by**: MPT (MosaicML), BLOOM, Falcon.

### Comparison

```
Method          Relative  Extrapolation  Parameters  Used by (2026)
                position? beyond train?  added?
──────────────  ──────── ──────────────  ──────────  ───────────────────
Learned PE      No       No             N × d       GPT-2, BERT
Sinusoidal PE   Implicit Weak           0           Original Transformer
RoPE            Yes      Moderate*      0           Llama, Mistral, Qwen,
                                                    DeepSeek, Gemma
ALiBi           Yes      Strong         0           MPT, BLOOM, Falcon
iRoPE           Yes      Very strong    0           Llama 4

* Moderate without scaling; strong with YaRN/NTK scaling
```

## RoPE Scaling / Context Extension

The central challenge: a model trained with RoPE at 4K context cannot directly handle 128K sequences. The high-frequency dimensions wrap around (fine — they're periodic), but low-frequency dimensions encounter unseen rotation angles. Three generations of solutions have emerged.

### Position Interpolation (PI) — Chen et al., Meta, 2023

The simplest fix: linearly compress all positions so they fit within the trained range.

```
Original RoPE:   θᵢ(m) = m × base^(-2i/d)
Position Interp: θᵢ(m) = (m / s) × base^(-2i/d)     where s = L'/L

Extending 4K → 128K:  s = 128K / 4K = 32
Position 128,000 → mapped to 128,000 / 32 = 4,000 (within training range)
```

**Problem**: High-frequency pairs (small i) are compressed too aggressively. These pairs had wavelengths of ~6 tokens — after 32x compression, adjacent tokens that were 1 apart are now only 1/32 apart in RoPE space. The model can't distinguish them. This hurts local attention patterns.

### NTK-Aware Interpolation (bloc97, 2023)

Key insight: instead of scaling positions (numerator), scale the base frequency (denominator). This spreads the interpolation pressure across dimensions unevenly — high frequencies are barely touched, low frequencies get most of the adjustment.

```
Original RoPE:     θᵢ = base^(-2i/d)               base = 10000
NTK-aware scaling: θᵢ = base'^(-2i/d)              base' = base × s^(d/(d-2))

Extending 4K → 128K (s=32, d=128):
  base' = 10000 × 32^(128/126) = 10000 × 33.1 = 331,131

Effect on wavelengths:
  Pair      Original λ      NTK-scaled λ    Ratio (stretch factor)
  ────────  ─────────────── ──────────────── ──────────────────────
  i = 0     6.28            6.28             1.00x (unchanged!)
  i = 16    628             1,330            2.12x
  i = 32    62,832          281,300          4.48x
  i = 63    6,283,185       208,373,000      33.2x (~= s)
```

High-frequency pairs (i=0) are barely affected — adjacent token discrimination is preserved. Low-frequency pairs (i=63) are stretched by ~s — matching the extension factor. This is much better than PI's uniform compression.

**Dynamic NTK**: Adjust s based on the current sequence length during inference:

```
s = max(1, current_length / trained_length)

At position 2K (within 4K training): s = 1    (no scaling)
At position 8K (2x beyond):          s = 2    (mild scaling)
At position 128K (32x beyond):       s = 32   (full scaling)
```

This provides a smooth transition — no scaling when within training range, gradual scaling as you exceed it.

### YaRN — Yet Another RoPE extensioN (Peng et al., 2023)

YaRN combines the best of PI and NTK with two additional innovations:

**1. Dimension-specific interpolation via ramp function**: Instead of applying one method uniformly, YaRN uses a ramp function to blend PI and NTK at different proportions across dimensions:

```
For each pair i, compute r = wavelength(i) / (trained_length * factor):

  If r < α:    use NTK scaling (preserve this frequency)
  If r > β:    use PI (interpolate this frequency)
  If α ≤ r ≤ β: blend NTK and PI (smooth transition)

  α = 1, β = 32 (typical values)

Result: high-frequency pairs → NTK (minimal change)
        mid-frequency pairs → smooth blend
        low-frequency pairs → PI (full interpolation)
```

**2. Attention temperature correction**: Long sequences shift the attention score distribution (more keys = more competition in softmax). YaRN introduces a temperature factor t to compensate:

```
softmax(QKᵀ / (√d × t))     where t = 0.1 × ln(s) + 1

At s=32: t = 0.1 × ln(32) + 1 = 0.1 × 3.47 + 1 = 1.347
```

This prevents attention from becoming too diffuse at long contexts.

**Training efficiency**: YaRN extends Llama 2 7B from 4K to 128K context with only ~400 training steps on a small dataset (~0.1% of pretraining tokens). Compare Meta's native 128K training for Llama 3.1, which used billions of tokens.

```
Context extension methods comparison:

  Method          Fine-tuning   Quality     Local       Extrapolation
                  cost          at target   resolution  beyond target
  ─────────────── ───────────── ─────────── ─────────── ─────────────
  PI              ~400 steps    Moderate    Degraded    Poor
  NTK-aware       0 (free!)     Good        Preserved   Moderate
  Dynamic NTK     0 (free!)     Good        Preserved   Good
  YaRN            ~400 steps    Best        Preserved   Good
  LongRoPE2       ~10B tokens   Near-native Preserved   Very good
  Native training Billions      Best        Best        N/A
```

### How Modern Models Extend Context

Most production models use a staged approach:

```
Phase 1: Pretrain at short context (e.g., 4K-8K)
  └─ Cheaper: O(N²) means shorter context = less compute

Phase 2: Continue pretrain at medium context (e.g., 32K-128K)
  └─ Apply YaRN/NTK scaling to RoPE
  └─ Use 1-5% of original pretraining tokens
  └─ Gradually increase sequence length during this phase

Phase 3: (Some models) Fine-tune for very long context (128K-1M+)
  └─ Use synthetic long-context data
  └─ Apply ring attention / context parallelism for training

Example — Llama 3.1 context extension:
  8K pretrain → 128K via continued pretraining with progressive scheduling
  Used ~800B tokens in the extension phase

Example — Llama 4 Scout:
  iRoPE architecture (RoPE + NoPE interleaved) designed for 10M from the start
  Trained with YaRN-based progressive context extension
```

## Flash Attention

### Why It Matters

Standard attention materializes the full N×N attention matrix in GPU HBM (high-bandwidth memory). For N=128K with fp16, that's 32 GB just for the attention scores of one head in one layer. This is the memory bottleneck.

FlashAttention never materializes the full matrix. It computes attention in tiles, keeping only O(N)-sized intermediate results in fast SRAM. The result is exact (not approximate) — same numerical output as standard attention, just computed differently.

### The Memory Hierarchy

The key insight is that GPU computation is fast but memory access is slow:

```
GPU Memory Hierarchy (H100):

  ┌─────────────────────────────────────────────┐
  │  Registers          ~20 MB   ~60 TB/s       │
  │  SRAM (per SM)      256 KB   ~33 TB/s       │  ← FlashAttention
  │  L2 Cache           50 MB    ~12 TB/s       │    works here
  │  HBM3               80 GB    3.35 TB/s      │  ← Standard attention
  │  Host RAM           ≥512 GB  ~64 GB/s       │    works here
  └─────────────────────────────────────────────┘

Standard attention:
  1. Compute S = QKᵀ         → write N²  to HBM
  2. Compute P = softmax(S)  → write N²  to HBM
  3. Compute O = PV          → write N×d to HBM
  Total HBM accesses: O(N² + Nd) reads/writes

FlashAttention:
  1. Load Q block (size B_r × d) into SRAM
  2. Stream K, V blocks (size B_c × d) through SRAM
  3. Accumulate output block in SRAM using online softmax
  4. Write final output (N × d) to HBM
  Total HBM accesses: O(N²d² / M)  where M = SRAM size
  For typical M, this is much less than O(N²)
```

### The Tiling Algorithm (Conceptual)

```
FlashAttention tiling:

  Q (N × d)          K (N × d)          V (N × d)
  ┌───────────┐      ┌───────────┐      ┌───────────┐
  │ Q block 0 │      │ K block 0 │      │ V block 0 │
  │ Q block 1 │      │ K block 1 │      │ V block 1 │
  │ Q block 2 │      │ K block 2 │      │ V block 2 │
  │    ...     │      │    ...     │      │    ...     │
  └───────────┘      └───────────┘      └───────────┘

  For each Q block (in SRAM):
    Initialize: O = 0, max_score = -inf, sum_exp = 0

    For each K, V block (streamed through SRAM):
      ┌──────────────────────────────────────────┐
      │  1. S_block = Q_block @ K_block.T        │
      │  2. Update running max and sum for        │
      │     online softmax (Milakov & Gimelshein) │
      │  3. O += softmax_block @ V_block          │
      │  (rescale previous O by max correction)   │
      └──────────────────────────────────────────┘

    Write final O block to HBM

  Never stores the full N × N attention matrix!
```

The "online softmax" trick is critical. Normal softmax requires two passes (find max, then compute exp/sum). The online algorithm maintains a running max and rescales previous results — enabling single-pass, block-by-block computation.

### Memory and Speed Impact

```
Standard attention vs FlashAttention at various sequence lengths:

  Seq len    Standard memory    FlashAttention memory    Speedup
             (attention matrix) (working memory)
  ────────── ────────────────── ──────────────────────── ────────
  4K         32 MB              ~1 MB                    1.5-2x
  32K        2 GB               ~1 MB                    2-3x
  128K       32 GB              ~1 MB                    3-5x
  512K       512 GB             ~1 MB                    —*
  1M         2 TB               ~1 MB                    —*

  * Standard attention cannot even run at these lengths
  FlashAttention memory is ~constant (depends on SRAM tile size, not N)
```

FlashAttention didn't just speed up existing workloads — it made previously impossible context lengths feasible on a single GPU. Before FlashAttention, 32K context on an 80GB GPU was challenging. After FlashAttention, 128K became routine and 1M became possible (with other optimizations).

### FlashAttention-3 (Hopper GPUs)

FlashAttention-3 exploits three features specific to NVIDIA H100/H200:

1. **Warp specialization**: Separate warps for data loading (TMA) vs. computation (Tensor Cores), running concurrently
2. **Asynchronous block-wise operations**: Interleave matmul and softmax across different data blocks — while one block does matmul, another does softmax
3. **FP8 support**: Low-precision matmul with "incoherent processing" (random signs applied to reduce quantization error) — 2.6x smaller error than naive FP8

Result: 1.5-2x faster than FlashAttention-2 on H100. On your H800, you'd see similar gains since the H800 has the same Hopper architecture.

## Ring Attention

### The Problem

Even with FlashAttention, a single GPU has finite HBM. For truly extreme context lengths (millions of tokens), the KV cache alone exceeds single-GPU memory. Ring Attention distributes the sequence across multiple GPUs.

### How It Works

```
Ring topology with 4 GPUs, sequence length 128K (32K per GPU):

  GPU 0             GPU 1             GPU 2             GPU 3
  ┌──────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐
  │ Q₀ K₀ V₀│      │ Q₁ K₁ V₁│      │ Q₂ K₂ V₂│      │ Q₃ K₃ V₃│
  │ (0-32K)  │      │ (32-64K) │      │ (64-96K) │      │ (96-128K)│
  └────┬─────┘      └────┬─────┘      └────┬─────┘      └────┬─────┘
       │                  │                  │                  │
       └──────────────────┴──────────────────┴──────────────────┘
                           Ring connection

  Step 0: Each GPU computes local attention (Qᵢ with Kᵢ, Vᵢ)
          Simultaneously: send Kᵢ,Vᵢ right, receive Kᵢ₋₁,Vᵢ₋₁ from left

  Step 1: Each GPU computes attention (Qᵢ with received K,V block)
          Simultaneously: forward the K,V block to next GPU

  Step 2: Repeat...

  Step 3: After 4 steps (= num GPUs), every Q has attended to all K,V blocks
```

The critical insight: communication of KV blocks is **fully overlapped** with attention computation. While GPU 0 computes attention on block (Q₀, K₁V₁), it simultaneously sends K₁V₁ to GPU 1 and receives K₃V₃ from GPU 3. If computation time ≥ communication time per block (true for FlashAttention), the communication is free.

### Memory and Scaling

```
Ring Attention memory scaling:

  N GPUs processing sequence of length L:
    Each GPU stores: L/N tokens of Q, K, V
    KV cache per GPU: total_cache / N
    Working memory: same as FlashAttention (tile-sized)

  Effective context length = N × single_GPU_context

  Example: 8x H800 (80GB each), Llama 3 70B:
    Single GPU max context: ~128K (20.9 GB KV cache)
    With Ring Attention:    ~1M   (each GPU holds ~128K)
```

Ring Attention was a key enabler for training models at very long contexts. Google used a variant for Gemini's 1M+ training; Meta uses context parallelism (a related technique) for Llama 4.

### Context Parallelism (Meta/NVIDIA, 2024)

An evolution of Ring Attention used in production training systems:

```
Context parallelism vs. other parallelism strategies:

  Strategy               What's split across GPUs
  ────────────────────── ──────────────────────────────
  Data parallelism       Different training examples
  Tensor parallelism     Model weights (within a layer)
  Pipeline parallelism   Model layers
  Expert parallelism     MoE experts
  Context parallelism    Sequence positions within one example

  Context parallelism is orthogonal — combine with all others.
```

Context parallelism splits the input sequence across devices and uses all-to-all or ring-based communication to exchange KV blocks. NVIDIA's implementation achieves near-linear scaling up to 128 GPUs, enabling multi-million token training.

## Sparse Attention Patterns

Full attention (every token attends to every token) is O(N²). Sparse attention reduces this by restricting which token pairs can attend to each other.

### Sliding Window Attention (Mistral, Gemma)

Each token attends only to the W nearest tokens in both directions:

```
Sliding window (W=3) attention pattern for 8 tokens:

  Token:    0  1  2  3  4  5  6  7
  0         ■  .  .  .  .  .  .  .
  1         ■  ■  .  .  .  .  .  .
  2         ■  ■  ■  .  .  .  .  .
  3         .  ■  ■  ■  .  .  .  .
  4         .  .  ■  ■  ■  .  .  .
  5         .  .  .  ■  ■  ■  .  .
  6         .  .  .  .  ■  ■  ■  .
  7         .  .  .  .  .  ■  ■  ■

  ■ = attends, . = masked out

  Complexity: O(N × W) instead of O(N²)
  For N=128K, W=4096: 128K × 4K = 512M ops vs 128K² = 16B ops (32x cheaper)
```

**Multi-layer effective receptive field**: With L layers of window size W, the effective receptive field is L × W tokens. Mistral 7B uses W=4096 with 32 layers, giving an effective receptive field of 131,072 tokens — information can propagate across the full 128K context through multiple layers, even though each individual attention operation is local.

```
Receptive field growth across layers (W=4096):

  Layer 1:   token 50,000 sees tokens [45,904 — 50,000]
  Layer 2:   token 50,000 sees tokens [41,808 — 50,000]  (via layer 1 propagation)
  ...
  Layer 32:  token 50,000 sees tokens [0 — 50,000]       (full history)
```

### Longformer: Local + Global Attention

Longformer combines sliding window attention with designated global tokens that attend to (and are attended by) all tokens:

```
Longformer attention pattern (W=2, 2 global tokens at positions 0,4):

  Token:    0  1  2  3  4  5  6  7
  0         ■  ■  ■  ■  ■  ■  ■  ■    ← global token (attends to all)
  1         ■  ■  ■  .  .  .  .  .    ← local (window) + attends to globals
  2         ■  ■  ■  ■  .  .  .  .
  3         ■  .  ■  ■  ■  .  .  .
  4         ■  ■  ■  ■  ■  ■  ■  ■    ← global token
  5         ■  .  .  .  ■  ■  ■  .
  6         ■  .  .  .  ■  ■  ■  ■
  7         ■  .  .  .  ■  .  ■  ■

  Complexity: O(N × W + N × G)  where G = number of global tokens
```

Global tokens serve as information bottlenecks — they aggregate information from the entire sequence and broadcast it back. In practice, [CLS] tokens or task-specific tokens serve as global tokens.

### BigBird: Local + Global + Random

BigBird adds random attention connections on top of Longformer's pattern:

```
BigBird = Sliding window + Global tokens + Random connections

  Theoretical result: this combination is a universal approximator
  of sequence functions, with O(N) complexity.

  In practice: each token randomly attends to R additional tokens
  beyond its local window and global connections.
```

### Dilated Attention

Instead of attending to consecutive neighbors, skip tokens at regular intervals:

```
Dilated attention (dilation=2, window=4):

  Token 6 attends to: positions 0, 2, 4, 6 (every 2nd token, 4 tokens total)

  Regular window:  [3, 4, 5, 6]           — 4 consecutive tokens
  Dilated window:  [0, 2, 4, 6]           — 4 tokens spanning 7 positions

  Receptive field: 4 tokens, but covering 2x the range
```

Multiple layers with increasing dilation (1, 2, 4, 8, ...) create an exponentially growing receptive field, similar to dilated convolutions in WaveNet. Used in some specialized architectures but less common in production LLMs.

### Llama 4's Hybrid Approach

Llama 4 Scout uses the most sophisticated sparse attention in production:

```
Llama 4 attention pattern (iRoPE):

  Layer type     Ratio    Attention scope       Positional encoding
  ────────────── ──────── ───────────────────── ────────────────────
  Local (chunked) 3/4     Non-overlapping       RoPE
                          chunks (e.g., 8K)
  Global          1/4     Full sequence         None (NoPE)

  Interleaved: [Local, Local, Local, Global, Local, Local, Local, Global, ...]

  The 1:3 global:local ratio means:
  - 75% of layers: O(chunk_size²) attention — very cheap
  - 25% of layers: O(N²) attention — expensive but infrequent

  Effective cost at 10M context with 8K chunks:
    Local layers: 10M/8K × 8K² = 10M × 8K = 80B ops
    Global layers: (10M)² / 4 = 25T ops
    Still dominated by global layers, but 4x cheaper than all-global
```

This is why Llama 4 Scout can support 10M tokens — most layers only do local attention within small chunks, and the few global layers with NoPE (no positional encoding) handle long-range dependencies without the extrapolation problem.

## KV Cache Compression

At long contexts, the KV cache often exceeds the model weights in memory. Multiple techniques compress it.

### MHA → MQA → GQA → MLA

This is the evolution of attention head sharing, which you've encountered in the MoE guide through DeepSeek-V2:

```
Multi-Head Attention (MHA) — Original Transformer, GPT-2:
  Q heads: H     K heads: H     V heads: H
  KV cache: 2 × H × d_head × seq_len

  GPT-2 124M (H=12, d=64):  KV cache = 2 × 12 × 64 × N = 1536N

Multi-Query Attention (MQA) — Shazeer 2019:
  Q heads: H     K heads: 1     V heads: 1
  All Q heads share a single K and single V

  KV cache = 2 × 1 × d_head × N = 128N    (12x smaller!)

  Tradeoff: significant quality loss from extreme sharing

Grouped-Query Attention (GQA) — Ainslie et al. 2023:
  Q heads: H     K heads: G     V heads: G     (1 < G < H)
  Groups of H/G query heads share one K,V pair

  Llama 3 70B: H=64, G=8 → 8x compression vs MHA

  ┌──────────────────────────────────────────────────┐
  │  MHA          MQA         GQA (G=2)              │
  │                                                  │
  │  Q₀→K₀V₀     Q₀→K₀V₀    Q₀─┐                   │
  │  Q₁→K₁V₁     Q₁→K₀V₀    Q₁─┼→K₀V₀             │
  │  Q₂→K₂V₂     Q₂→K₀V₀    Q₂─┐                   │
  │  Q₃→K₃V₃     Q₃→K₀V₀    Q₃─┼→K₁V₁             │
  │                                                  │
  │  4 KV pairs   1 KV pair   2 KV pairs             │
  └──────────────────────────────────────────────────┘

Multi-head Latent Attention (MLA) — DeepSeek-V2/V3:
  Compresses KV into a low-rank latent vector c_t:

  Standard: cache K_t (d_head × H) + V_t (d_head × H) per token
  MLA:      cache c_t (d_c) per token, where d_c << d_head × H

  K_t and V_t are reconstructed from c_t on the fly via:
    K_t = W_UK @ c_t      V_t = W_UV @ c_t

  DeepSeek-V2: d_c = 512, vs MHA d = 128 × 128 = 16,384
  Compression ratio: 512 / 16,384 = ~3% of MHA (97% reduction!)

  Separate RoPE handling:
    Content part: low-rank compressed (no RoPE) — cacheable
    Position part: carries RoPE — small, separate cache

  This is why DeepSeek-V3 (671B) has a smaller KV cache than Llama 8B
```

### Quantized KV Cache

Store cached K,V in lower precision than the model's compute precision:

```
KV cache quantization impact (Llama 3 70B, 128K context):

  Precision   Bytes/param   KV cache size   Quality loss
  ─────────── ──────────── ──────────────── ────────────
  FP16        2            20.9 GB          baseline
  FP8         1            10.5 GB          ~0.1% ppl
  INT4        0.5          5.2 GB           ~0.5% ppl
  INT2        0.25         2.6 GB           1-3% ppl

  KVQuant (2024): specialized per-channel quantization
    → INT4 KV cache with <0.1% perplexity degradation
    → Enables 1M context on a single GPU for 8B models
```

Key insight: KV cache values have a different distribution than model weights — they contain outlier tokens (attention sinks) with much larger magnitudes. Effective KV quantization must handle these outliers specially (per-channel scaling, outlier-aware quantization).

### Token Eviction / Pruning

Not all cached tokens are equally important. Evict low-importance tokens from the KV cache:

```
Token eviction strategies:

  StreamingLLM (Xiao et al., 2023):
    Keep: first few "sink" tokens + recent window
    Observation: first tokens accumulate disproportionate attention
    regardless of content ("attention sinks")

    ┌──────────────────────────────────────────────┐
    │ keep    evict evict evict evict ... keep keep │
    │ [sink]  [...........................] [recent]│
    │ 4 tok   (all middle tokens dropped)  W tokens│
    └──────────────────────────────────────────────┘

    Fixed memory: 4 + W tokens regardless of total length
    Quality: good for streaming/generation, poor for retrieval

  H2O — Heavy Hitter Oracle (Zhang et al., 2023):
    Track cumulative attention scores per token
    Evict tokens with lowest cumulative attention

    Token:     [The] [cat] [sat] [on] [a] [mat] [.]
    Cum. attn:  0.82  0.45  0.31  0.12 0.08 0.67 0.05
    Budget = 4: keep  keep  evict evict evict keep evict

    Dynamic: important tokens survive, unimportant ones are evicted

  SnapKV / PyramidKV (2024):
    Observation: different layers need different cache budgets
    Early layers: broad attention → need more cached tokens
    Later layers: focused attention → need fewer cached tokens

    Layer:   1    2    3   ...  30   31   32
    Budget:  4K   4K   3K       1K   512  512

    Total cache much smaller than uniform allocation
```

### Combining Techniques

Production systems stack these techniques:

```
Example: DeepSeek-V3 serving at 128K context

  MLA compression:         97% KV reduction (MLA vs MHA)
  FP8 KV quantization:     50% on top of that
  Net:                      ~1.5% of original MHA KV cache size

  Result: 671B MoE model serving 128K context with
          manageable KV cache on standard inference hardware
```

## State-of-the-Art Context Lengths (as of March 2026)

```
Model                Context    Architecture      How they achieve it
─────────────────── ────────── ───────────────── ───────────────────────────────
Llama 4 Scout       10M        iRoPE + MoE       Chunked local + global attn,
                                                  NoPE on global layers
Gemini 2.5 Pro      1M → 2M    Dense (details     Internal FlashAttn variant,
                               undisclosed)       progressive context training
GPT-5               1M         Dense (details     Likely progressive training +
                               undisclosed)       FlashAttn + context parallelism
Claude 4 Sonnet     200K → 1M  Dense (details     1M beta for high-tier users
                    (beta)     undisclosed)
Llama 3.1           128K       GQA + RoPE         YaRN scaling + continued
                                                  pretraining (800B tokens)
DeepSeek-V3         128K       MLA + MoE          MLA cache compression,
                                                  YaRN scaling
Mistral Large       128K       SWA + GQA          Sliding window + global layers
Qwen 2.5           128K       GQA + RoPE +       YaRN, progressive training
                               Dynamic NTK
Jamba 1.5           256K       SSM + Attention     Mamba (linear) layers handle
                               hybrid             long range; attention for local
```

### How They Achieve Long Context — Common Patterns

1. **RoPE extension** (YaRN/NTK/iRoPE) for positional encoding generalization
2. **FlashAttention** (or equivalent fused kernel) for memory-efficient attention
3. **Progressive training**: pretrain at short context, gradually extend
4. **Architectural choices**: GQA/MLA for KV cache compression, sliding window for compute reduction
5. **Context parallelism / Ring Attention** for training at extreme lengths
6. **Post-training long-context data**: synthetic datasets with long-range dependencies

The frontier has shifted from "can we achieve 128K?" (solved) to "can we make 1M+ actually useful?" (in progress). Raw context length is no longer the bottleneck — quality at long context is.

## Evaluation

### Needle-in-a-Haystack (NIAH)

The original test (Kamradt, 2023) embeds a single distinctive fact in a long document and checks if the model can retrieve it. Sweeps over haystack length and needle depth.

```
NIAH heatmap (typical modern model):

  Needle depth ↓    Context length →
                    4K     32K    128K    512K    1M
  Top (0%)          ■■■■■  ■■■■■  ■■■■■  ■■■■■  ■■■■■
  25%               ■■■■■  ■■■■■  ■■■■■  ■■■■□  ■■■□□
  Middle (50%)      ■■■■■  ■■■■■  ■■■■□  ■■■□□  ■■□□□
  75%               ■■■■■  ■■■■■  ■■■■■  ■■■■□  ■■■□□
  Bottom (100%)     ■■■■■  ■■■■■  ■■■■■  ■■■■■  ■■■■■

  ■ = correct retrieval, □ = failure

  Most models: near-perfect "green field" on simple NIAH
  Failures concentrate: middle depth × long context (lost-in-middle)
```

**Problem**: Simple NIAH is too easy. A model scoring 100% on NIAH may still fail at practical long-context tasks. The needle is typically a distinctive, out-of-distribution fact that's easy to pattern-match.

### RULER (NVIDIA, 2024)

RULER expands NIAH into four task categories with configurable difficulty:

```
RULER benchmark categories:

  Category             Examples                           Difficulty
  ──────────────────── ──────────────────────────────── ────────────
  Single NIAH          Retrieve one fact                  Easy
  Multi-NIAH           Retrieve 2, 4, 8 facts at once    Medium
  Multi-hop tracing    Follow a chain of references       Hard
                       across the context
  Aggregation          Count, summarize, or reason        Hardest
                       across multiple distributed facts

  Key finding: models that score 100% on single NIAH
  often drop to 50-70% on multi-NIAH and 30-50% on aggregation
```

### Lost-in-the-Middle Effect

```
Accuracy vs. information position (Liu et al., 2023):

  100% ┐■■■                                           ■■■
       │   ■■                                       ■■
   80% │     ■■                                   ■■
       │       ■■                               ■■
   60% │         ■■■                         ■■■
       │            ■■■                   ■■■
   40% │               ■■■■■■■■■■■■■■■■■■
       │
   20% │
       └──────────────────────────────────────────────
        Start           Middle                   End
        of context                               of context

  U-shaped curve: models attend well to beginning and end,
  poorly to middle positions.

  This persists in 2026, though reduced in newer models:
  - Claude 4, GPT-5: ~10-15% mid-context degradation
  - Llama 3.1: ~20-30% mid-context degradation
  - Older models: ~40-50% mid-context degradation
```

### Context Rot (Adobe, 2025)

A more nuanced view of degradation:

```
Context rot findings across 18 LLMs:

  1. Performance degrades non-monotonically — not a smooth decline
  2. Complex tasks degrade faster than simple retrieval
  3. Adding plausible distractors (semantically similar to the needle)
     causes much steeper drops than random padding
  4. Different models have different "rot profiles" — some degrade
     gradually, others cliff-edge at specific lengths

  Practical implication: a model's nominal context length (e.g., 128K)
  is NOT the length at which it works reliably. Effective context
  is typically 50-70% of nominal for complex tasks.
```

### Real-World vs. Synthetic Benchmarks

```
Benchmark type        What it tests            Correlation with
                                               real-world utility
────────────────────  ──────────────────────── ─────────────────
Simple NIAH           Basic retrieval          Low (too easy)
RULER                 Multi-skill retrieval    Medium
Lost-in-middle tests  Positional robustness    Medium
LongBench / L-Eval    Multi-document QA,       Medium-High
                      summarization, coding
InfiniteBench         Tasks requiring >100K    High
                      context for correct answer
Real user tasks       Code repos, long chats,  Highest
                      document analysis
```

The gap between synthetic benchmarks and real utility remains significant. A model scoring 95% on RULER at 128K may still produce poor summaries of 100K-token codebases, because real tasks require reasoning over context, not just retrieving from it.

## Practical Considerations

### When You Need Long Context vs. RAG

```
Decision framework:

                                   Long Context         RAG
                                   ────────────         ───
  Corpus size                      < 1M tokens          Unlimited
  Corpus changes frequently?       Expensive (re-ingest) Cheap (re-index)
  Need cross-document reasoning?   Strong                Weak
  Need exact retrieval from        Strong                Depends on
  specific location?                                     retriever
  Cost per query                   High (process all     Low (process
                                   tokens every time)    only chunks)
  Latency                          High at long context  Lower
  Implementation complexity        Low (just send it)    High (chunking,
                                                        embedding, etc.)

  Emerging winner: Hybrid approach
    1. RAG retrieves relevant chunks (narrowing 10M → 100K)
    2. Long context reasons over retrieved chunks
    → Best of both: manageable cost + good reasoning
```

### Cost Implications

```
Inference cost scales with context length:

  Typical API pricing (per 1M tokens, early 2026):

  Model              Input cost     Output cost
  ────────────────── ──────────── ────────────
  GPT-5              $2.00          $8.00
  Claude 4 Sonnet    $3.00          $15.00
  Gemini 2.5 Pro     $1.25          $10.00
  Llama 4 (self-host) ~$0.30        ~$0.30

  Cost of a single 128K-token prompt:
    GPT-5:         128K × $2.00/1M = $0.26
    Claude 4:      128K × $3.00/1M = $0.38

  Cost of the same task with RAG (retrieves 8K relevant tokens):
    GPT-5:         8K × $2.00/1M = $0.016

  16-24x cost difference between full-context and RAG approaches
```

### Degradation at Context Boundaries

```
Performance profile of a "128K context" model:

  Context     Retrieval   Reasoning   Summarization
  length      accuracy    quality     quality
  ──────────  ──────────  ──────────  ──────────────
  4K          98%         95%         95%
  16K         97%         93%         93%
  32K         95%         90%         90%
  64K         92%         85%         85%
  128K        85%         75%         78%

  Key insight: "128K context" means "can process 128K tokens,"
  not "works as well at 128K as at 4K."

  Rule of thumb: expect 10-25% quality degradation at the
  nominal context limit compared to the sweet spot (4K-16K).
```

### Practical Tips for Your H800

```
What you can run on 1x H800 80GB:

  Model              Max practical     KV cache    Weights   Free
                     context (fp16)    at max ctx  (fp16)    for batch
  ────────────────── ──────────────── ──────────── ──────── ──────────
  Llama 3 8B         128K+            4.2 GB       16 GB    ~60 GB
  Llama 3 70B (INT4) 32K              5.2 GB       35 GB    ~40 GB
  DeepSeek-V3 (MLA)  Need multi-GPU — 671B total params
  Qwen 2.5 7B        128K+            4.2 GB       14 GB    ~62 GB

  With KV cache quantization (INT4):
  Llama 3 8B          512K+           ~8 GB        16 GB    ~56 GB

  Tips:
  1. Use vLLM or SGLang — they handle KV cache management efficiently
  2. Enable FlashAttention-2/3 (default in most frameworks)
  3. For >128K: quantize KV cache to INT4 (KVQuant or KIVI)
  4. For 8B models: you can easily serve 128K on your single H800
  5. For 70B: INT4/GPTQ weights + GQA gets you 32K comfortably
```

## Key Papers

| Paper | Authors | Year | Key Contribution |
|-------|---------|------|-----------------|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Vaswani et al. (Google) | 2017 | Transformer architecture, sinusoidal PE, O(N²) attention |
| [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) | Child et al. (OpenAI) | 2019 | Sparse factorized attention patterns, O(N√N) |
| [Longformer](https://arxiv.org/abs/2004.05150) | Beltagy et al. (AI2) | 2020 | Sliding window + global attention for long documents |
| [Big Bird](https://arxiv.org/abs/2007.14062) | Zaheer et al. (Google) | 2020 | Proved sparse attention is universal approximator |
| [RoFormer](https://arxiv.org/abs/2104.09864) | Su et al. | 2021 | Rotary Position Embeddings (RoPE) |
| [ALiBi](https://arxiv.org/abs/2108.12409) | Press et al. (UW/Meta/AI2) | 2021 | Linear attention biases, train-short-test-long |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | Dao et al. (Stanford) | 2022 | IO-aware tiling, O(N) memory exact attention |
| [Lost in the Middle](https://arxiv.org/abs/2307.03172) | Liu et al. (Stanford) | 2023 | U-shaped retrieval accuracy vs. position |
| [Position Interpolation](https://arxiv.org/abs/2306.15595) | Chen et al. (Meta) | 2023 | Linear position scaling for RoPE context extension |
| [FlashAttention-2](https://arxiv.org/abs/2307.08691) | Dao (Stanford) | 2023 | 2x speedup, better parallelism on A100 |
| [YaRN](https://arxiv.org/abs/2309.00071) | Peng et al. | 2023 | Efficient RoPE extension: NTK + temperature + ramp |
| [Ring Attention](https://arxiv.org/abs/2310.01889) | Liu et al. (UC Berkeley) | 2023 | Distributed attention across GPUs in ring topology |
| [StreamingLLM](https://arxiv.org/abs/2309.17453) | Xiao et al. (MIT/Meta) | 2023 | Attention sinks + sliding window for infinite streaming |
| [GQA](https://arxiv.org/abs/2305.13245) | Ainslie et al. (Google) | 2023 | Grouped-query attention — MHA/MQA middle ground |
| [FlashAttention-3](https://arxiv.org/abs/2407.08608) | Dao et al. | 2024 | Hopper GPU optimization, FP8, 1.5-2x over FA2 |
| [RULER](https://arxiv.org/abs/2404.06654) | Hsieh et al. (NVIDIA) | 2024 | Multi-task long-context benchmark beyond NIAH |
| [KVQuant](https://arxiv.org/abs/2401.18079) | Hooper et al. (UC Berkeley) | 2024 | Per-channel KV quantization for 10M+ context |
| [LongRoPE2](https://arxiv.org/abs/2502.20082) | Microsoft | 2025 | Near-lossless context extension, 80x fewer tokens than native |
| [Context Rot](https://research.trychroma.com/context-rot) | Chroma Research | 2025 | Systematic study of long-context performance degradation |
| [Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) | Meta | 2025 | iRoPE architecture, 10M context, chunked local + global attention |
