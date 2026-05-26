# TurboQuant: Online Vector Quantization with Near-Optimal Distortion

A guide to TurboQuant — a data-oblivious vector quantization algorithm that achieves near-optimal distortion for both MSE and inner products, with direct applications to KV cache compression and nearest neighbor search.

## Background

**Originating paper**: [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (Zandieh, Daliri, Hadian, Mirrokni — Google Research / Google DeepMind / NYU, April 2025)

**Research lineage** — TurboQuant sits at the intersection of information theory and practical LLM inference:

1. **Shannon's source coding theory** (1948, 1959) — Established that any lossy compression has a minimum achievable distortion determined by the distortion-rate function. TurboQuant explicitly targets this information-theoretic bound.

2. **Zador's distortion-rate function** (1963) — Derived the limiting operational distortion-rate for fixed-rate vector quantization at high rates, closely matching Shannon's bound. Theoretical but not implementable.

3. **Lloyd-Max scalar quantizer** (Lloyd 1982, Max 1960) — Optimal scalar quantizer that minimizes MSE for a given distribution by iteratively solving a continuous 1D k-means problem. TurboQuant uses this as its core per-coordinate quantizer.

4. **Product Quantization (PQ)** (Jegou et al., 2010) — Splits vectors into subvectors, quantizes each with a separate codebook learned via k-means. Dominant approach for nearest neighbor search. Data-dependent (requires offline training), TurboQuant's main baseline for NN search.

5. **QJL (Quantized Johnson-Lindenstrauss)** (Zandieh, Daliri, Han — same group, 2024) — [arXiv:2406.03482](https://arxiv.org/abs/2406.03482). A 1-bit quantization scheme based on random projections that provides unbiased inner product estimates with zero overhead. TurboQuant uses QJL as a building block for its inner-product variant.

6. **PolarQuant** (Han, Kacham, Karbasi, Mirrokni, Zandieh, 2025) — [arXiv:2502.02617](https://arxiv.org/abs/2502.02617). Quantizes KV caches using polar coordinate transformation. Same research group. TurboQuant improves on PolarQuant's theoretical guarantees.

7. **RabitQ** (Gao et al., 2024) — [arXiv:2409.09913](https://arxiv.org/abs/2409.09913). Grid-based product quantization that projects onto the unit sphere. Online (no preprocessing), but computationally slow and theoretically suboptimal. TurboQuant's baseline for NN search.

**Why this matters**: Existing VQ methods face a fundamental trade-off — either they're data-dependent (requiring expensive offline preprocessing, unsuitable for streaming KV caches) or they achieve suboptimal distortion bounds. TurboQuant is the first online (data-oblivious) method that provably achieves near-optimal distortion rates across all bit-widths, within a constant factor (~2.7x) of the information-theoretic lower bound.

## The Core Problem

Vector quantization maps high-dimensional floating-point vectors to compact binary codes while preserving geometric structure. Formally:

```
Quantizer:      Q : R^d → {0,1}^B     (encode d-dimensional vector to B bits)
Dequantizer:    Q^{-1} : {0,1}^B → R^d  (reconstruct approximate vector)
Bit-width:      b = B/d                  (average bits per coordinate)
```

Two distortion metrics matter:

```
MSE distortion:          D_mse = E[ ||x - Q^{-1}(Q(x))||^2 ]
Inner-product distortion: D_prod = E[ |<y,x> - <y, Q^{-1}(Q(x))>|^2 ]
```

The expectation is over the randomness in the quantizer (TurboQuant uses randomized algorithms).

**Why two metrics?** MSE measures reconstruction quality — critical for model weight compression and general-purpose compression. Inner product distortion measures how well dot products are preserved — critical for attention (which is all dot products), nearest neighbor search, and cosine similarity.

A key insight of the paper: **MSE-optimal quantizers are biased for inner product estimation.** You can't optimize one metric and get the other for free. This motivates TurboQuant's two-algorithm design.

## Key Insight: Random Rotation Induces a Known Distribution

The central trick that makes TurboQuant work:

### Step 1: Random rotation

Multiply any input vector x by a random rotation matrix Pi:

```
y = Pi * x
```

where Pi is a random d x d orthogonal matrix (generated via QR decomposition of a random Gaussian matrix).

### Step 2: The rotated vector has a known coordinate distribution

If x is on the unit sphere (||x|| = 1), then Pi*x is uniformly distributed on the unit hypersphere S^{d-1}. Each coordinate of this rotated vector follows a **scaled Beta distribution**:

```
y_j ~ f_X(t) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - t^2)^{(d-3)/2}
```

for t in [-1, 1].

### Step 3: In high dimensions, this converges to Gaussian

As d grows, by the central limit theorem, each coordinate converges to:

```
y_j ~ N(0, 1/d)
```

### Step 4: Near-independence enables per-coordinate quantization

This is the deeper insight. Not only do coordinates have the same marginal distribution, but distinct coordinates are **nearly independent** in high dimensions (not just uncorrelated — actually approximately independent). This means we can quantize each coordinate separately with an optimal scalar quantizer and still achieve near-optimal vector quantization.

```
Before rotation: x could be anything — no structure to exploit
After rotation:  each coordinate is ~ Beta, nearly independent
                 → optimal SCALAR quantization ≈ optimal VECTOR quantization
```

**Why this is powerful**: Vector quantization in d dimensions requires searching over exponentially many codebook entries. TurboQuant reduces it to d independent scalar quantizations — a massive computational simplification with near-zero theoretical cost.

**Handling non-unit vectors**: For vectors where ||x|| != 1, simply store the norm ||x|| separately in floating point. Quantize x/||x||, then rescale during dequantization.

## Algorithm 1: TurboQuant_mse (MSE-Optimal)

### Setup (one-time, offline)

1. **Generate rotation matrix**: Pi in R^{d x d} via QR decomposition of random Gaussian matrix
2. **Build codebook**: Find optimal centroids c_1, c_2, ..., c_{2^b} in [-1, 1] that minimize the Lloyd-Max k-means cost for the Beta distribution f_X:

```
Cost(f_X, b) = min over centroids  sum_{i=1}^{2^b} integral |x - c_i|^2 * f_X(x) dx
```

This is solved once numerically (continuous 1D k-means) and stored. The optimal centroids partition [-1, 1] into 2^b Voronoi cells with boundaries at midpoints between consecutive centroids.

### Quantization (per vector, online)

```python
def quant_mse(x, Pi, centroids):
    y = Pi @ x                           # rotate: O(d^2) or O(d log d) with fast transforms
    idx = [nearest_centroid(y_j) for j in range(d)]  # b-bit index per coordinate
    return idx                            # total: b*d bits
```

### Dequantization

```python
def dequant_mse(idx, Pi, centroids):
    y_tilde = [centroids[idx[j]] for j in range(d)]  # look up centroids
    x_tilde = Pi.T @ y_tilde              # rotate back
    return x_tilde
```

### Numerical example (b=1, large d)

For b=1 (1 bit per coordinate), the Beta distribution is approximately N(0, 1/d), and the optimal 2 centroids are:

```
c_1 = -sqrt(2/(pi*d)),  c_2 = +sqrt(2/(pi*d))
```

Each coordinate is quantized to its sign, with reconstruction values at +/- sqrt(2/(pi*d)).

For b=2 (2 bits per coordinate, 4 centroids):

```
c = {-1.51/sqrt(d), -0.453/sqrt(d), +0.453/sqrt(d), +1.51/sqrt(d)}
```

### Distortion guarantee (Theorem 1)

For any bit-width b >= 1 and any unit vector x:

```
D_mse <= (sqrt(3) * pi / 2) * (1/4^b)  ≈  2.72 / 4^b
```

Concrete values for small bit-widths:

 b │ D_mse (upper bound) │ Info-theoretic lower bound (1/4^b)
───┼─────────────────────┼────────────────────────────────────
 1 │ 0.36                │ 0.25
 2 │ 0.117               │ 0.0625
 3 │ 0.03                │ 0.0156
 4 │ 0.009               │ 0.0039

The gap to the lower bound is at most ~2.7x, and shrinks for small b (at b=1 it's only ~1.45x).

### Why rotation preserves MSE

Since Pi is orthogonal: ||x - x_tilde||^2 = ||Pi*x - Pi*x_tilde||^2 = ||y - y_tilde||^2. The rotation doesn't change distances — it only changes which coordinates we operate on, giving us the nice Beta distribution to work with.

## The MSE-Bias Problem for Inner Products

MSE-optimal quantizers are **biased** for inner product estimation. Here's why:

For b=1 with large d, TurboQuant_mse is essentially sign quantization with reconstruction at +/- sqrt(2/(pi*d)). The dequantization map is:

```
Q_mse^{-1}(z) = sqrt(2/(pi*d)) * Pi^T * z    for z in {-1, +1}^d
```

Computing the expected inner product:

```
E[<y, Q_mse^{-1}(Q_mse(x))>] = (2/pi) * <y, x>
```

This has a **multiplicative bias of 2/pi ≈ 0.637** — inner products are systematically underestimated! The bias diminishes as b increases (at b=4, it's negligible), but at low bit-widths it's severe.

**Concrete example**: If the true inner product <y,x> = 0.5, the MSE quantizer estimates it as ~0.318. For attention scores, this systematic underestimation changes the softmax distribution, degrading model quality.

## Algorithm 2: TurboQuant_prod (Inner-Product-Optimal)

The solution is a two-stage algorithm:

### Stage 1: MSE quantization with (b-1) bits

Apply TurboQuant_mse with bit-width (b-1) instead of b. This uses one less bit per coordinate but gives good MSE reconstruction.

### Stage 2: QJL on the residual (1 bit)

Compute the residual vector r = x - Q_mse^{-1}(Q_mse(x)), then apply the 1-bit QJL transform:

```python
def quant_prod(x, Pi, centroids, S):
    # Stage 1: MSE quantize with (b-1) bits
    idx = quant_mse(x, Pi, centroids_b_minus_1)
    x_mse = dequant_mse(idx, Pi, centroids_b_minus_1)

    # Stage 2: QJL on residual
    r = x - x_mse                         # residual vector
    qjl = sign(S @ r)                     # 1-bit QJL: random projection + sign
    gamma = ||r||                          # store residual norm (scalar)
    return (idx, qjl, gamma)              # total: (b-1)*d + d + 32 = b*d + 32 bits
```

### QJL Dequantization

The QJL map and its inverse:

```
Q_qjl(r) = sign(S * r)              S is d x d random Gaussian matrix
Q_qjl^{-1}(z) = sqrt(pi/2) / d * S^T * z
```

The key property: QJL is **unbiased** for inner products:

```
E[<y, Q_qjl^{-1}(Q_qjl(r))>] = <y, r>    (exact, no bias)
```

### Full dequantization

```python
def dequant_prod(idx, qjl, gamma, Pi, centroids, S):
    x_mse = dequant_mse(idx, Pi, centroids_b_minus_1)
    x_qjl = sqrt(pi/2) / d * gamma * S.T @ qjl   # scaled QJL reconstruction
    return x_mse + x_qjl
```

### Why this is unbiased (proof sketch)

The inner product estimate decomposes as:

```
<y, x_tilde> = <y, x_mse> + <y, x_qjl>
```

Taking expectation over the QJL randomness, conditioned on x_mse:

```
E[<y, x_tilde> | x_mse] = <y, x_mse> + E[<y, x_qjl> | x_mse]
                         = <y, x_mse> + <y, r>          (QJL is unbiased)
                         = <y, x_mse> + <y, x - x_mse>  (definition of r)
                         = <y, x>                         (exact!)
```

By the law of total expectation: E[<y, x_tilde>] = <y, x>. The QJL correction on the residual exactly cancels the bias of the MSE quantizer.

### Distortion guarantee (Theorem 2)

For any bit-width b >= 1, any unit vectors x, and any y:

```
E[<y, x_tilde>] = <y, x>                              (unbiased)

D_prod <= (sqrt(3) * pi^2 * ||y||^2) / d * (1/4^b)    (variance bound)
```

Concrete values (normalized by ||y||^2/d):

 b │ D_prod * d/||y||^2 │ Lower bound (1/4^b * 1/d)
───┼────────────────────┼───────────────────────────
 1 │ 1.57/d             │ 0.25/d
 2 │ 0.56/d             │ 0.0625/d
 3 │ 0.18/d             │ 0.0156/d
 4 │ 0.047/d            │ 0.0039/d

The 1/d factor is crucial — in high dimensions, inner product distortion becomes very small.

## Information-Theoretic Lower Bounds (Theorem 3)

TurboQuant proves that **no quantizer** (online or offline, data-dependent or not) can do fundamentally better:

```
For ANY randomized quantization algorithm Q with bit-width b:

  D_mse(Q) >= 1/4^b                     (MSE lower bound)
  D_prod(Q) >= ||y||^2 / (d * 4^b)      (inner product lower bound)
```

**Proof approach**: Uses Yao's minimax principle to convert the worst-case randomized algorithm problem to an average-case deterministic algorithm problem. Then applies Shannon's Lower Bound (SLB) for uniform distribution on the unit hypersphere:

```
Shannon's Lower Bound for x uniform on S^{d-1}:
  D(B) >= 2^{-2B/d}

With B = b*d total bits:
  D_mse >= 2^{-2b} = 1/4^b
```

**Optimality gap**:

```
TurboQuant_mse upper bound:     (sqrt(3)*pi/2) / 4^b  ≈  2.72 / 4^b
Information-theoretic lower:    1 / 4^b

Gap factor: sqrt(3)*pi/2 ≈ 2.72  (constant, independent of b and d)
```

This means TurboQuant is within a factor of ~2.7 of the best ANY algorithm could ever achieve, and the gap is even smaller at low bit-widths.

## How TurboQuant Compares to Existing Methods

### Online vs Offline quantization

```
Method          │ Online? │ Accelerator-  │ Optimal      │ Unbiased inner
                │         │ friendly?     │ distortion?  │ product?
────────────────┼─────────┼───────────────┼──────────────┼────────────────
GPTQ/AWQ        │ No      │ Yes           │ No (heuristic)│ No
KIVI            │ Yes     │ Yes           │ No           │ No
PolarQuant      │ Yes     │ Yes           │ Near-optimal │ Yes
Product Quant.  │ No      │ Partial       │ No           │ No
RabitQ          │ Yes     │ No (no GPU)   │ Suboptimal   │ No
TurboQuant      │ Yes     │ Yes           │ Near-optimal │ Yes (prod variant)
```

**Online** means data-oblivious: no preprocessing, calibration, or learning on the data. The quantization map is fixed (only depends on random rotation, not on the data). This is essential for KV cache quantization where tokens arrive one at a time during generation.

### Indexing time comparison (100K vectors, 4-bit quantization)

```
                │ d=200    │ d=1536   │ d=3072
────────────────┼──────────┼──────────┼──────────
Product Quant.  │ 37.04s   │ 239.75s  │ 494.42s
RabitQ          │ 597.25s  │ 2267.59s │ 3957.19s
TurboQuant      │ 0.0007s  │ 0.0013s  │ 0.0021s
```

TurboQuant is 100,000-1,000,000x faster because it requires no k-means training or data-dependent preprocessing.

## Application 1: KV Cache Quantization

### The problem

In transformer inference, each generated token requires storing K and V projections in memory. For a model with L layers, H attention heads, and head dimension d_h:

```
KV cache size = 2 * L * H * d_h * seq_len * bytes_per_value
```

For Llama-3.1-8B with 32 layers, 8 KV heads, d_h=128, at FP16:
- 100K tokens: 2 * 32 * 8 * 128 * 100K * 2 bytes = ~12.5 GB

This is often the dominant memory consumer during long-context inference.

### How TurboQuant helps

Apply TurboQuant to each KV head's embedding independently. At 2.5 effective bits:

```
Compression ratio: 16 / 2.5 = 6.4x  →  12.5 GB → ~2 GB
```

### Outlier channel handling

Like prior work (RotateKV, QJL), TurboQuant uses a mixed-precision strategy for channels with outlier values:

```
2.5-bit setup: 32 outlier channels at 3 bits + 96 regular channels at 2 bits
  Effective: (32*3 + 96*2) / 128 = 2.5 bits/channel

3.5-bit setup: different ratio of outlier/regular channels
  Effective: 3.5 bits/channel
```

### Results: Needle-in-a-Haystack (Llama-3.1-8B-Instruct)

At 4x compression (25% of full KV cache), retrieving hidden sentences from 4K-104K tokens:

```
Method       │ Score  │ Compression │ Notes
─────────────┼────────┼─────────────┼──────────────────────────
Full Precision│ 0.997 │ 1x          │ Baseline (FP16)
TurboQuant   │ 0.997  │ 4x+         │ Identical to full precision
PolarQuant   │ 0.995  │ 4x          │ Near-identical
KIVI         │ 0.981  │ ~3x         │ Minor degradation
PyramidKV    │ 0.895  │ 4x          │ Significant degradation
SnapKV       │ 0.858  │ 4x          │ Significant degradation
```

TurboQuant achieves **quality-neutral compression** — identical retrieval performance at 4x compression.

### Results: LongBench-E (end-to-end generation quality)

```
Method          │ KV Size │ SingleQA │ MultiQA │ Summariz. │ Few-shot │ Synthetic │ Code  │ Average
────────────────┼─────────┼──────────┼─────────┼───────────┼──────────┼───────────┼───────┼────────
Full Cache      │ 16      │ 45.29    │ 45.16   │ 26.55     │ 68.38    │ 59.54     │ 46.28 │ 50.06
KIVI (3-bit)    │ 3       │ 43.38    │ 37.99   │ 27.16     │ 68.38    │ 59.50     │ 44.68 │ 48.50
PolarQuant      │ 3.9     │ 45.18    │ 44.48   │ 26.23     │ 68.25    │ 60.07     │ 45.24 │ 49.78
TurboQuant 2.5  │ 2.5     │ 44.16    │ 44.96   │ 24.80     │ 68.01    │ 59.65     │ 45.76 │ 49.44
TurboQuant 3.5  │ 3.5     │ 45.01    │ 45.31   │ 26.00     │ 68.63    │ 59.95     │ 46.17 │ 50.06
```

At 3.5 bits, TurboQuant matches full-precision performance exactly (50.06 average). At 2.5 bits (6.4x compression), it loses only 0.62 points — the best among all methods at this compression level.

**Key advantage over other methods**: TurboQuant quantizes tokens during streaming generation (including generated tokens), while KIVI and PolarQuant leave generated tokens unquantized. This gives TurboQuant better actual compression during long generation.

## Application 2: Nearest Neighbor Search

### The problem

Vector databases store millions of embeddings. Brute-force inner product search is O(n*d) per query. Product quantization compresses vectors to enable faster approximate search, but requires expensive offline codebook training.

### TurboQuant advantage

Zero indexing time — the quantization is data-independent. No codebook training needed. Just rotate and scalar-quantize.

### Recall@k results (100K vectors from DBpedia, OpenAI3 embeddings)

For d=1536, Recall@1:

```
Method         │ 2-bit  │ 4-bit
───────────────┼────────┼───────
TurboQuant     │ ~0.95  │ ~0.99
Product Quant. │ ~0.90  │ ~0.97
RabitQ         │ ~0.88  │ ~0.94
```

TurboQuant consistently outperforms both baselines across all embedding dimensions (200, 1536, 3072) and bit-widths, while requiring essentially zero indexing time.

## Computational Complexity

```
Operation       │ TurboQuant_mse          │ TurboQuant_prod
────────────────┼─────────────────────────┼──────────────────────────
Rotation (Pi*x) │ O(d^2) or O(d log d)*   │ Same
Scalar quantize │ O(d * 2^b)              │ O(d * 2^{b-1})
QJL transform   │ N/A                     │ O(d^2) or O(d log d)*
Total quantize  │ O(d^2)                  │ O(d^2)
Dequantize      │ O(d^2)                  │ O(d^2)
Storage per vec │ b*d bits + 32 bits      │ b*d bits + 64 bits
```

*With structured random rotations (e.g., randomized Hadamard transform), the rotation cost drops to O(d log d). The paper doesn't explicitly use this, but it's a natural optimization.

The O(d^2) cost is dominated by the rotation matrix multiplication. For KV cache quantization with d_h = 128 (typical head dimension), this is a small 128x128 matrix multiply — very fast on GPUs.

## Entropy Encoding (Optional Optimization)

The codebook indices produced by TurboQuant are not uniformly distributed — centroids near zero (where the Beta distribution has more mass) are selected more often. This means entropy coding can further compress without any quality loss.

For b=4: the entropy of the index distribution is ~3.8 bits (vs 4 bits nominal), giving ~5% additional compression for free. The paper chose not to implement this to keep the algorithm simple.

## Connection to Your Prior Knowledge

**Relation to your inference optimization studies**: TurboQuant directly addresses the KV cache memory bottleneck you studied. It complements techniques like GQA (which reduces the number of KV heads) by compressing each head's values more aggressively.

**Relation to quantization methods (GPTQ, AWQ, SmoothQuant)**: Those methods quantize model weights (offline, data-dependent, using calibration data). TurboQuant quantizes activations (KV cache) online — a fundamentally different problem because you can't preprocess data that doesn't exist yet.

**Relation to your scaling law knowledge**: The KV cache scaling problem is exactly Kaplan's memory scaling — cache grows linearly with context length. TurboQuant's 4-6x compression directly extends the effective context window by the same factor without quality loss.

**Information-theoretic flavor**: The lower bound proof uses Shannon's source coding theorem (rate-distortion theory) — the same framework that underpins the compression theory behind tokenization and Chinchilla scaling.

## Summary

- **Problem**: Compress high-dimensional vectors (KV cache, embeddings) while preserving distances and inner products, without expensive offline preprocessing
- **Key insight**: Random rotation maps ANY input vector to a known Beta distribution per coordinate with near-independent coordinates, enabling optimal scalar quantization to approximate optimal vector quantization
- **Two algorithms**: TurboQuant_mse (MSE-optimal, possibly biased for inner products) and TurboQuant_prod (unbiased inner products via MSE + QJL residual correction)
- **Theoretical contribution**: First online VQ algorithm provably within ~2.7x of the information-theoretic lower bound for all bit-widths. Also provides the first formal proof of these lower bounds.
- **Practical results**: Quality-neutral KV cache compression at 3.5 bits (4.5x), marginal degradation at 2.5 bits (6.4x). Outperforms Product Quantization and RabitQ for nearest neighbor search with ~1,000,000x faster indexing.
- **Design philosophy**: Data-oblivious, accelerator-friendly, theoretically grounded. Trades a small constant factor in optimality for massive gains in simplicity and speed.
