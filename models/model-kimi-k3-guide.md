# Kimi K3: Open Frontier Intelligence

How Moonshot got to 2.78T parameters and 1M context by replacing 3 of every 4 attention layers with a fixed-state linear recurrence, replacing the residual stream with attention over depth, and halving the width of the routed-expert space so 896 experts fit in the budget.

## Background

**Primary source**: *Kimi K3: Open Frontier Intelligence — Technical Report of Kimi K3*, Kimi Team, Moonshot AI. 47 pages. [arXiv 2607.24653](https://arxiv.org/abs/2607.24653) (submitted July 27, 2026) · [PDF in the release repo](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf) · [blog](https://www.kimi.com/blog/kimi-k3) · [weights](https://huggingface.co/moonshotai/Kimi-K3). Every equation, parameter count, and benchmark number below is transcribed from that PDF.

**Timeline**: API + blog announcement July 16, 2026. Open weights July 26, 2026. License is the **Kimi K3 License** — modified MIT: free to use, modify, distribute, and sell; but a MaaS operator with >$20M revenue over any consecutive 12 months needs a separate agreement, and any product with >100M MAU or >$20M monthly revenue must display "Kimi K3" in its UI. Internal use is exempt. Not OSI-approved open source, but far more permissive than a research license.

**Research lineage** — K3 is three separate Moonshot preprints bolted into one model, plus a systems paper's worth of infrastructure:

1. **Linear Transformers Are Secretly Fast Weight Programmers** (Schlag, Irie, Schmidhuber, IDSIA, 2021, [arXiv 2102.11174](https://arxiv.org/abs/2102.11174)) — the delta rule as a fast-weight memory write. `S_t = S_{t-1} + β_t k_t (v_t − S_{t-1}ᵀk_t)ᵀ`: don't just append, *correct* what's already stored at key `k_t`. Cited as [106] in K3.

2. **Parallelizing Linear Transformers with the Delta Rule over Sequence Length** (Yang et al., NeurIPS 2024, [arXiv 2406.06484](https://arxiv.org/abs/2406.06484)) — DeltaNet's chunkwise parallel form (the WY / UT transform). Made the delta rule trainable at scale instead of strictly sequential.

3. **Gated Linear Attention (GLA)** (Yang et al., ICML 2024, [arXiv 2312.06635](https://arxiv.org/abs/2312.06635)) — data-dependent decay + hardware-efficient chunked kernels. Cited as [141]; K3 inherits its secondary-tiling trick.

4. **Transformers are SSMs (Mamba-2 / SSD)** (Dao & Gu, 2024, [arXiv 2405.21060](https://arxiv.org/abs/2405.21060)) — the negative-softplus decay parameterization K3 explicitly replaces.

5. **Gated Delta Networks** (Yang, Kautz, Hatamizadeh, NVIDIA, ICLR 2025, [OpenReview r8H7xhYPwz](https://openreview.net/forum?id=r8H7xhYPwz)) — gating + delta rule combined. The direct ancestor of KDA. K3's reference list gives no arXiv ID for it.

6. **Kimi Linear: An Expressive, Efficient Attention Architecture** (Kimi Team, Oct 2025, [arXiv 2510.26692](https://arxiv.org/abs/2510.26692)) — **introduced KDA**: gated delta rule with a *channel-wise* forget gate, in a 3:1 hybrid with full attention. K3's KDA is this, with two changes (§KDA below).

7. **Attention Residuals** (Kimi Team, Mar 16, 2026, [arXiv 2603.15031](https://arxiv.org/abs/2603.15031)) — replaces the additive residual stream with **softmax attention over depth**. K3's reference [58] cites this only as "Preprint, 2026"; the arXiv ID is verified separately. [Code](https://github.com/MoonshotAI/Attention-Residuals).

8. **LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts** (Elango et al., Jan 2026, [arXiv 2601.18089](https://arxiv.org/abs/2601.18089)) — routed experts operate in a *narrow latent space*, shared experts stay full-width. K3's [32].

9. **DeepSeekMoE** ([arXiv 2401.06066](https://arxiv.org/abs/2401.06066)) and **DeepSeek-V3** ([arXiv 2412.19437](https://arxiv.org/abs/2412.19437)) — shared+routed organization and auxiliary-loss-free bias routing. K3 keeps the organization, replaces the bias update rule.

10. **DeepSeek-V2** ([arXiv 2405.04434](https://arxiv.org/abs/2405.04434)) — MLA. K3 keeps it in the 1-in-4 global layers, adds a gate, and drops RoPE from it entirely.

11. **Kimi K2** ([arXiv 2507.20534](https://arxiv.org/abs/2507.20534)) → **Kimi K2.5** ([arXiv 2602.02276](https://arxiv.org/abs/2602.02276)) → **Kimi K3**. 1.04T → 1T multimodal → 2.78T. 128K → 128K → 1M. All-MLA → all-MLA → hybrid KDA/MLA.

**The arc**: K2 was "scale MoE sparsity and make it agentic." K2.5 added vision and Agent Swarm. K3's bet is that the open ecosystem stalled at the 1T class while chasing test-time compute, so it pushes *both* axes — 2.7× the parameters **and** 8× the context **and** RL across nine domain×effort experts.

## What Problem Does K3 Solve?

The paper's framing (§1) is that scaling has two axes and open models have been scaling only one:

```
Axis 1: pre-training scale     Axis 2: test-time compute
─────────────────────────      ──────────────────────────
o1/R1 era: open models         o1/R1 era: open models
stuck at ~1T class             raced ahead (GRPO, RLVR,
(DeepSeek, GLM, MiMo,          agent swarms, effort levels)
Inkling all 1T-ish)

     ↓ K3 pushes both ↓
2.78T total / 104.2B active         RL over 3 domains × 3 effort
1M-token context                    levels → 9 experts → distilled
```

Concretely, three engineering walls stood in the way:

1. **KV cache at 1M tokens.** K2's 61 MLA layers each keep a per-token cache. At 8× the context that cache is the dominant memory cost, and prefill is quadratic. K3's answer: make 69 of 93 attention layers carry a **fixed-size recurrent state** that does not grow with sequence length at all.

2. **Residual-stream bottleneck at depth 93.** A standard pre-norm residual compresses everything the network has computed into one running vector — "a bottleneck reminiscent of RNNs over time" (§2.2). At 61 layers you tolerate it; at 93 it costs you.

3. **Routing 896 experts.** Two failure modes appear at sparsity 56 that don't at sparsity 48: the latent routed path becomes a chain of ~4 consecutive matmuls that explodes activations at 2.8T scale, and DeepSeek-V3's fixed-step bias update can't equilibrate ~10³ experts fast enough.

Net claimed result: **≈2.5× improvement in scaling efficiency over K2** (Fig. 7 — validation loss vs FLOPs on held-out OOD data, with hyperparameters re-fit per architecture).

## Key Terms

**KDA (Kimi Delta Attention)**: Delta-rule linear attention with a *channel-wise* forget gate `α_t ∈ (0,1)^{d_k}`. Fixed-size state `S ∈ R^{d_k×d_v}` per head — no growth with sequence length.

**Gated MLA**: DeepSeek's Multi-head Latent Attention plus an input-dependent full-rank output gate, with **NoPE** — no positional encoding at all on queries or keys.

**AttnRes / Block AttnRes**: Each layer emits a learned pseudo-query and does softmax attention over *previous layers' outputs* (or block summaries), instead of adding into a single residual stream.

**Stable LatentMoE**: Routed experts live in a latent space of width `ℓ = 3584 = 0.5d`; shared experts stay full-width. "Stable" = RMSNorm before up-projection + SiTU-GLU + Quantile Balancing.

**SiTU-GLU (Sigmoid Tanh Unit GLU)**: SwiGLU with both branches smoothly capped by `β·tanh(x/β)`. `β₁=4` (gate), `β₂=25` (up) → output bounded by 100.

**QB (Quantile Balancing)**: Sets each expert's routing bias to the router-score quantile matching its target load, in one shot, instead of nudging by `±γ` per step.

**Per-Head Muon**: Newton–Schulz orthogonalization applied per attention head block rather than to the whole Q/K/V projection matrix.

**MOPD (Multi-Teacher On-Policy Distillation)**: Nine RL experts (3 domains × 3 effort levels) distilled into one student with a per-token log-ratio reward.

**XTML (eXtensible Token Markup Language)**: K3's chat template — XML semantics where `<`, `>`, `/>` are replaced by three reserved special tokens `[open]`, `[sep]`, `[close]`.

**MoonEP**: Expert-parallel library that guarantees *every* rank receives exactly `S×K` tokens via dynamic redundant experts, making MoE computation shapes statically known.

**AgentENV**: Firecracker-microVM sandbox runtime with 133 ms checkpoint / 49 ms resume, used for agentic RL rollouts.

## Architecture at a Glance

Table 1 of the report, verbatim:

| | Kimi K2 | Kimi K3 | Δ |
|---|---|---|---|
| Layers | 61 | 93 | ↑52% |
| Total parameters | 1.04T | **2.78T** | ↑167% |
| Activated parameters | 32.6B | **104.2B** | ↑220% |
| Hidden dimension | 7,168 | 7,168 | = |
| Latent MoE dimension | – | 3,584 (0.5×) | – |
| MoE hidden dim per expert | 2,048 | 3,072 | ↑50% |
| Routed experts | 384 | **896** | ↑133% |
| Experts active per token | 8 | **16** | ↑100% |
| Shared experts | 1 | 2 | ↑100% |
| Attention heads | 64 | 96 | ↑50% |
| Dense (non-MoE) layers | 1 | 1 | = |
| Vocabulary | 160K | 160K | = |
| Training context length | 128K | **1M** | 8× |
| Attention mechanism | MLA | **Hybrid KDA–MLA** | – |
| Activation function | SwiGLU | **SiTU-GLU** | – |
| Attention-layer composition | 61 MLA | **69 KDA + 24 MLA** | – |
| MTP layers | 1 | 1 | = |
| ViT parameters | – | 401M | – |
| ViT layers / patch / heads | – | 27 / 14 / 12 | – |

The layer arithmetic: 23 blocks × (3 KDA + 1 Gated MLA) = 92, plus one extra Gated MLA at the very end so the **final layer always does global attention** → 93 layers, 69 KDA + 24 MLA.

```
Block n:   [KDA] → [KDA] → [KDA] → [Gated MLA]      × 23 blocks
Tail:      [Gated MLA]
Every attention layer is followed by a Stable LatentMoE FFN.
```

Sanity-check the parameter count yourself. A routed expert is a gated FFN at latent width 3584 with hidden 3072, so 3 matrices of 3584×3072:

```
per routed expert   3 × 3584 × 3072            =   33.0 M
per MoE layer       896 × 33.0 M               =   29.6 B
92 MoE layers       92 × 29.6 B                =    2.72 T   ← ~98% of the model
```

Routed experts alone account for essentially all of the 2.78T. The remaining ~60B is attention (93 layers × ~430M), shared experts (2 × 66.1M × 92 ≈ 12.2B), the latent down/up projections (51.4M × 92 ≈ 4.7B), routers, embeddings, and the 401M ViT.

Note the sparsity framing: 896/16 = **sparsity 56**, up from K2's 384/8 = 48. But *activated fraction* went **up**, 3.13% → 3.75%, because depth and attention width grew. "More sparse" here means a wider expert menu, not a cheaper forward pass.

## Kimi Delta Attention

You know softmax attention and you know the KV cache problem. KDA is the other branch of the family tree: a linear-attention recurrence whose "cache" is a fixed matrix.

### The recurrence

For one head, with `q_t, k_t ∈ R^{d_k}`, `v_t ∈ R^{d_v}`, state `S_t ∈ R^{d_k×d_v}` (Eq. 1):

```
S_t = (I − β_t k_t k_tᵀ) · Diag(α_t) · S_{t−1}  +  β_t k_t v_tᵀ
õ_t = S_tᵀ q_t
```

Read it right to left:

```
Diag(α_t) S_{t−1}          channel-wise forgetting — each of the d_k key
                           channels decays independently by its own α
(I − β_t k_t k_tᵀ) · …     delta-rule erase — remove what's currently
                           stored at key k_t, weighted by β_t
+ β_t k_t v_tᵀ             write the new value
```

`α_t ∈ (0,1)^{d_k}` is per-channel retention; `β_t ∈ (0,1)` is write strength. Set `α = 1` and you get vanilla DeltaNet; drop the `(I − βkkᵀ)` term and you get GLA/Mamba-2.

Per-head parameterization, inherited from Kimi Linear (Eq. 2):

```
q_t, k_t = L2Norm( Swish( ShortConv( W_q/k x_t ) ) )   ∈ R^{d_k}
v_t      =         Swish( ShortConv( W_v   x_t ) )     ∈ R^{d_v}
β_t      = Sigmoid( W_β x_t )                          ∈ (0,1)
z_t      = W_α^↑ W_α^↓ x_t + b_α                       ∈ R^{d_k}   (low-rank decay logits)
```

### Change 1: lower-bounded decay — and why it is really a kernel change

This is the subtle one, and it is a pure algorithm–system co-design move.

The chunkwise form (Eq. 4) rescales keys within each chunk by the **reciprocal** cumulative decay `1/Γ`, where `Γ = ∏α` is a product of numbers in (0,1). Unbounded decay ⟹ unbounded reciprocal ⟹ overflow.

```
Kimi Linear:  g_t = −e^{A} · Softplus(z_t)      ∈ (−∞, 0)
Kimi K3:      g_t = g_min · Sigmoid(e^{A} z_t)  ∈ (g_min, 0),   g_min = −5 (fixed)
              α_t = exp(g_t)
```

`A` is a learnable per-head log-scale, initialized to 0. Trace the numbers:

```
g_min = −5     →  α_{t,j} > e^{−5} ≈ 6.738 × 10⁻³
16-token tile  →  cumulative log-decay ∈ (−80, 0)
                  (16 × 5 = 80)
reciprocal     →  1/Γ < e^{80} ≈ 5.54 × 10³⁴
BF16 max       →  3.39 × 10³⁸                    ✓ ~4 orders of headroom

Unbounded case: a single channel with α = e^{−20} over 16 tokens
               →  1/Γ = e^{320} ≈ 10¹³⁹          ✗ overflows by 100 orders
```

Kimi Linear handled this by computing relative decay in log space and splitting each chunk into secondary 16-token tiles. Off-diagonal tiles then became dense Tensor Core matmuls — but the **diagonal tiles still needed explicit position-pair computation**, which was the intra-chunk bottleneck. With a bounded range, K3 makes *every* causal tile a dense matmul and deletes the position-pair path entirely.

So a one-line change to an activation function removed a whole slow kernel path. The paper notes the bounded gate is related to HGRN2 ([arXiv 2404.07904](https://arxiv.org/abs/2404.07904)), Griffin ([arXiv 2402.19427](https://arxiv.org/abs/2402.19427)), and RWKV-7 ([arXiv 2503.14456](https://arxiv.org/abs/2503.14456)).

The obvious cost: with `α > e^{−5}`, a channel can never fully forget within a tile. K3 does not report an ablation isolating what that costs on long-range retrieval.

### Change 2: full-rank output gate

Kimi Linear used a low-rank output gate. K3 goes full rank (Eq. 6), after head-wise RMSNorm:

```
y_t = W_o [ Sigmoid(W_g x_t) ⊙ RMSNorm(õ_t) ]
```

### Gated MLA, and why NoPE

The 1-in-4 global layers are MLA (compress K/V into a latent `c_t = W_c x_t`, reconstruct on the fly) with the same full-rank gate (Eq. 7):

```
y_t = W_o [ Sigmoid(W_g x_t) ⊙ õ_t ]
```

The design choice worth internalizing: **K3 applies NoPE to all MLA layers.** No RoPE, no ALiBi, nothing. The division of labour is explicit —

```
KDA layers   →  position-sensitive, recency-aware   (position comes from the decay gates)
MLA layers   →  unrestricted global content lookup  (no position at all)
```

The payoff is that context extension needs **no positional surgery**. K2 went 32K→128K by retuning YaRN. K3 goes to 1M by just training on longer sequences — there is no RoPE base frequency to rescale and no interpolation to tune. If you have ever fought YaRN/NTK scaling on a long-context fine-tune, this is the mechanism that makes that fight disappear.

One numerics detail worth stealing: K3 keeps the flash-attention output in **FP32 during training** to correct biased rounding error (following [arXiv 2510.04212](https://arxiv.org/abs/2510.04212)). That doubles the on-chip output tile, so they redesigned the kernel to overlap it with KV staging buffers instead of the query tile.

### What this buys at 1M context

```
Layer type   Count   Cache per sequence
──────────   ─────   ──────────────────────────────────
KDA            69    fixed d_k × d_v matrix per head    ← constant in T
Gated MLA      24    latent c_t per token                ← grows with T
```

74% of attention layers contribute **zero** sequence-length-dependent cache. That is the whole reason 1M is economically shippable — and it's also why the serving stack (§Infrastructure) needed a rewrite: you now have two cache types with completely different sizes and lifetimes that must be checkpointed at the *same* boundaries.

## Attention Residuals: attention over depth

This is the most conceptually novel piece, and it is a clean idea.

The observation (§2.2): a standard residual stream compresses all prior computation into a single state `h_l`, which is exactly the bottleneck RNNs have over *time*. The Transformer fixed that for sequences by replacing recurrence with attention. AttnRes applies the same fix to **depth**.

```
Standard pre-norm residual                AttnRes
──────────────────────────                ───────────────────────────────
h_{l+1} = h_l + F_l(RMSNorm(h_l))         each layer l emits a learned
                                          pseudo-query w_l and attends
one running stream                        over ALL previous layer outputs
uniform accumulation                      data-dependent selection
```

### Full form

Layer `l` has a learnable pseudo-query `q_l = w_l ∈ R^d`. Keys and values are the outputs of previous layers, with the token embedding as source 0 (Eq. 8–9):

```
k_i = v_i = h_1              if i = 0     (token embedding)
k_i = v_i = f_i(h_i)         if 1 ≤ i ≤ l−1

φ(q, k)   = exp( qᵀ · RMSNorm(k) )
α_{i→l}   = φ(q_l, k_i) / Σ_{j=0}^{l−1} φ(q_l, k_j)
h_l       = Σ_{i=0}^{l−1} α_{i→l} · v_i
```

The RMSNorm inside the kernel matters: without it, layers with large-magnitude outputs would dominate the attention weights regardless of relevance.

Cost: `O(L²d)` arithmetic — trivial at `L < 100`. The real cost is `O(Ld)` **memory**, keeping every layer output alive, plus cross-stage communication under pipeline parallelism.

### Block form, and K3's numbers

Partition `L` layers into `N` blocks of `S = L/N`. Within block `n`, sum the layer outputs into one representation `b_n = Σ_{j∈B_n} f_j(h_j)`, with `b_0 = h_1` (the embedding). Attention runs over block summaries plus the current block's running partial sum (Eq. 10):

```
V = [b_0, b_1, …, b_{n−1}]ᵀ              if i = 1  (first layer of block n)
V = [b_0, b_1, …, b_{n−1}, b_n^{i−1}]ᵀ   if i ≥ 2
```

Trace K3's actual configuration — `L = 93`, `d = 7168`, 8 blocks of 12 layers (7 full × 12 = 84, plus a 9-layer partial final block = 93; 9 sources counting the embedding):

```
Full AttnRes   O(Ld) = 93 × 7168 = 666,624 values/token → 1.33 MB/token in BF16
                                                          → ~87 GB at 64K tokens
Block AttnRes  O(Nd) =  8 × 7168 =  57,344 values/token → 114.7 KB/token in BF16
                                                          → ~7.5 GB at 64K tokens

Reduction: 93/8 = 11.6×
```

The paper reports `N ≈ 8` recovers most of the benefit across model scales. The AttnRes source paper claims ~1.25× compute advantage at <2% extra cost.

**Do not confuse the two block structures.** They are different partitions of the same 93 layers:

```
Attention hybrid block:  4 layers   (3 KDA + 1 MLA)      → 23 blocks + tail
AttnRes block:          12 layers   (arbitrary grouping)  → 8 blocks
```

The block form also bounds inference-time state, which lets the parallel inter-block result merge with the sequential intra-block partial sum via **online softmax** ([arXiv 1805.02867](https://arxiv.org/abs/1805.02867)) — the same trick FlashAttention uses, applied across depth instead of across sequence.

Prior work this resembles but isn't: DenseNet ([arXiv 1608.06993](https://arxiv.org/abs/1608.06993)) gives every layer access to previous features but by concatenation, not learned softmax routing. Value Residual Learning ([arXiv 2410.17897](https://arxiv.org/abs/2410.17897)) residualizes value vectors *inside* attention. Hyper-Connections ([arXiv 2409.19606](https://arxiv.org/abs/2409.19606)) — which DeepSeek-V4's mHC generalizes — uses multiple parallel residual tracks with a learned mixing matrix; AttnRes instead makes the mixing weights *data-dependent softmax* over a growing set of sources.

## Stable LatentMoE

You know DeepSeekMoE's shared+routed split and aux-loss-free balancing. Three things here are new.

### The latent bottleneck

In a conventional MoE, each of the `k` selected experts receives the full `d`-dimensional token. Communication and expert-weight traffic scale with `k·d`. LatentMoE separates model width from routed-expert width (Eq. 11):

```
z = W↓ x                                 ∈ R^ℓ        ℓ = 3584 = 0.5 × 7168
u = Σ_{i ∈ T_k(x)} p_i · E_i^routed(z)   ∈ R^ℓ
y = Σ_{j=1}^{N_s} E_j^shared(x)  +  W↑ RMSNorm(u)     N_s = 2 full-width shared
```

The accounting, per token, per layer:

```
                      full-width MoE       LatentMoE (ℓ = 0.5d)
all-to-all dispatch   16 × 7168 = 114,688  16 × 3584 = 57,344    2× less traffic
params per expert     3×7168×3072 = 66.1M  3×3584×3072 = 33.0M   2× smaller
896 experts / layer   59.2 B               29.6 B                → 5.4T vs 2.72T
```

That factor of 2 is exactly what makes 896 experts fit. A full-width 896-expert model at these dimensions would be a 5.4T-parameter model.

### Failure mode 1: exploding activations → RMSNorm + SiTU-GLU

The routed path is now `W↓ → gated multi-branch FFN → W↑`, "a chain of nearly four consecutive matrix multiplications." Ill-conditioned, and at 2.8T scale it blows up.

Fix (a): **RMSNorm between expert aggregation and `W↑`** (already in Eq. 11 above). `u`'s scale varies with which experts fired and their routing weights; normalize before it rejoins the full-width shared branch. Reported to improve validation loss and downstream benchmarks, not just stability.

Fix (b): **SiTU-GLU**. Both of SwiGLU's factors are unbounded, so coincident large coordinates produce outliers. K3 applies a smooth cap `softcap(x, β) = β·tanh(x/β)` to the linear factor of the Swish gate *and* independently to the up branch (Eq. 12):

```
SiTU-GLU(x) = [ β₁·tanh(W_g x / β₁) ⊙ Sigmoid(W_g x) ] ⊙ [ β₂·tanh(W_u x / β₂) ]
              β₁ = 4  (gate branch)      β₂ = 25  (up branch)
```

Two properties, both provable in one line (Appendix B). Near the origin `β·tanh(z/β) = z + O(z³/β²)`, so SiTU matches SwiGLU to first order. Globally `|tanh| < 1` and `0 < Sigmoid < 1`, so `‖SiTU-GLU(x)‖_∞ ≤ β₁β₂ = 100`.

Numerically, with gate pre-activation 50 and up pre-activation 200:

```
                gate branch                    up branch          product
SwiGLU    Swish(50) = 50·σ(50)  ≈ 50.0         200.0              10,000
SiTU-GLU  4·tanh(12.5)·σ(50)    ≈  4.0    25·tanh(8) ≈ 25.0          100

Near origin (both pre-acts = 0.5):
SwiGLU    0.5·σ(0.5) = 0.3112                  0.5               0.1556
SiTU-GLU  4·tanh(.125)·σ(0.5) = 0.3096    25·tanh(.02) = 0.4999   0.1548     (0.5% apart)
```

Why 100 and not 10,000 matters concretely: K3 does quantization-aware training with **MXFP8 activations**. FP8 E4M3 maxes out at 448. A 10,000-magnitude activation saturates; a 100-magnitude one has 4.5× headroom. The activation function and the quantization scheme were chosen together. The paper also notes the smooth cap beats hard clamping because gradients stay nonzero away from the saturation boundary.

### Failure mode 2: balancing ~10³ experts → Quantile Balancing

Routing is aux-loss-free (Eq. 13) — bias `b` steers dispatch but is excluded from the mixture weights, so it never touches the router's gradients:

```
s_i  = Sigmoid(W_r x_i)
T_i  = argtopk(s_i + b)
p_{i,j} = s_{i,j} / Σ_{r∈T_i} s_{i,r}          ← note: no b here
```

DeepSeek-V3's update is a fixed step: `b_j ← b_j + γ·sign(ℓ̄ − ℓ_j)`. With 896 experts per layer, `γ` trades slow adaptation against oscillation and neither setting works.

**QB derives the bias in closed form from a single forward pass.** The trick: route with **Top-(k+1)** instead of Top-k. The first `k` entries are the actual routes; the `(k+1)`-th biased score is the **cutoff** `α_i` — the bar an expert must clear to enter token `i`'s Top-k. That gives a token-side threshold for free, with no separate quantile pass.

With cutoffs fixed, the token count routed to expert `j` under candidate bias `b̂_j` is `Σ_i 1[s_{i,j} + b̂_j > α_i]`, monotone decreasing in `−b̂_j`. Setting it to the target load `q = mk/n` makes `−b̂_j` the `(q+1)`-th largest margin, i.e. the `(1−k/n)`-quantile (Eq. 14):

```
b̂_j^{(t+1)} ← − quantile_{1−k/n}( s_{:,j} − α^{(t)} )
b^{(t+1)}   ← b̂^{(t+1)} − mean(b̂^{(t+1)}) · 1
```

Mean-centering removes a common offset that would leave Top-k unchanged anyway. The update applies only at the **next** step, so a batch is never routed with a bias derived from itself.

**Numerical walkthrough** — the paper's Fig. 5 setting: `m = 8` tokens, `n = 4` experts, `k = 1`, so target `q = mk/n = 2`. Starting from `b = 0`, Top-2 routing gives each token's choice and its cutoff:

```
        E1    E2    E3    E4     choice   cutoff α_i
t1     .90   .40   .30   .20       E1        .40
t2     .85   .55   .25   .15       E1        .55
t3     .80   .30   .35   .10       E1        .35
t4     .70   .65   .20   .25       E1        .65
t5     .50   .75   .40   .30       E2        .50
t6     .45   .72   .55   .20       E2        .55
t7     .30   .68   .61   .35       E2        .61
t8     .25   .40   .66   .45       E3        .45
                                loads = (4, 3, 1, 0)   ← E4 is dying
```

Margins `s_{:,j} − α`, sorted descending; take the `(q+1)` = 3rd largest:

```
E1:  .50  .45  .30  .05  .00 −.10 −.30 −.20   → 3rd = .30   → b̂₁ = −.30
E2:  .25  .17  .01  .00  .00  .00 −.05 −.05   → 3rd = .01   → b̂₂ = −.01
E3:  .21  .06  .00  .00 −.10 −.10 −.30 −.45   → 3rd = .00   → b̂₃ =  .00
E4:  .00 −.20 −.20 −.25 −.25 −.35 −.40 −.40   → 3rd = −.20  → b̂₄ = +.20

mean(b̂) = −0.0275  →  b = (−.2725, +.0175, +.0275, +.2275)
```

One forward pass moved the dead expert's bias by **+0.2275** and the overheated one by **−0.2725**. DeepSeek-V3's published step size is `γ = 0.001`; the sign rule would need roughly **230 steps** to make the same move for E4. That is the entire point — the paper reports QB "equilibrates within a few update steps even for nearly 10³ experts," and needs no learning-rate-like hyperparameter.

**Why it works** (Appendix C): QB is the exact coordinate minimizer of the dual of the maximum-score balanced-assignment LP. Relax `x_{i,j} ∈ {0,1}` to `[0,1]` — exact, by bipartite b-matching integrality — dualize, and the dual objective is

```
L(α, β) = Σ_{i,j} max(0, s_{i,j} − α_i − β_j) + k·Σ_i α_i + (mk/n)·Σ_j β_j
```

Minimizing coordinate-wise in `α` gives the `(1−k/n)`-quantile along the token axis; in `β`, the same quantile along the expert axis — hence the name. The subgradient in `β_j` is exactly `(target load − observed load)`, so **DeepSeek-V3's sign rule is SignSGD on this same objective**, while QB jumps straight to the minimizer. That is a genuinely satisfying result: the heuristic everyone has been using turns out to be a crude optimizer for a problem with a closed-form solution.

Lineage: the assignment view goes back to BASE Layers (ICML 2021) and BIP ([arXiv 2502.15451](https://arxiv.org/abs/2502.15451)); K3 credits [Jianlin Su's blog post](https://spaces.ac.cn/archives/11619) (Feb 2026) for the formulation.

### Making the quantile computable at scale

Exact quantiles need `O(mn)` margins gathered across ranks and accumulation steps — millions of values. K3 uses a **histogram estimator** (Appendix D) instead, and the bookkeeping is neat:

- Histogram the **required bias** `r_{i,j} := α_i − s_{i,j}` rather than the margin (negating reverses the order, so the target becomes the `(k/n)`-quantile).
- Binning range is bounded for free: `s ∈ (0,1)` since it's a sigmoid, and `α_i` is some biased score, so `r_{i,j} ∈ [b_min − 1, b_max + 1]`. Range recomputed every step; bin width `w = (b_max − b_min + 2)/B`.
- Each rank scatter-adds into `H ∈ N^{n×B}` with **zero communication** during the forward pass. One integer all-reduce at step end.
- Because counts are additive, the pooled histogram is **exactly invariant** to how tokens are sharded — you get the true global-batch quantile, not an average of per-rank quantiles (which is a different and wrong quantity).

With `B = 1000` bins the error is bounded by the bin width, a few `10⁻³`, and communication is `nB` integers per layer per step — under 1% of the cost of exchanging raw margins. Reported result: no measurable residual load imbalance.

## Native Vision: MoonViT-V2

**MoonViT-V2 is trained from scratch with next-token prediction. No SigLIP initialization.** This is a deliberate reversal of standard practice, including K2.5's own.

The stated reason is training stability, not accuracy. Fig. 6 shows the SigLIP-initialized MoonViT-3D has persistently higher vision-tower gradient norms with frequent spikes when jointly optimized with the LLM; the from-scratch MoonViT-V2 stays flat. The secondary argument: NTP lets the language objective shape visual representations directly, rather than a contrastive loss that favours global semantics over fine-grained textual and structural cues (which is what you want for OCR and UI screenshots).

The punchline: MoonViT-V2 **matches** the SigLIP-initialized baseline across vision evals — "contrastive pre-training is unnecessary as an initialization for multimodal language models at scale."

Specs: 27 layers, 401M params, patch size 14, 12 heads, RMSNorm, **all bias terms removed** from linear and attention projections. Images and videos share parameters; attention is factorized into intra-frame spatial and inter-frame temporal passes with temporal pooling. A `2×2` pixel-shuffle before projection cuts visual tokens 4×, keeping inputs up to `3584×3584` pixels affordable inside 1M context.

## Per-Head Muon

You know Muon (orthogonalized momentum via Newton–Schulz). K3 refines it for attention projections: instead of orthogonalizing the whole `W_q`/`W_k`/`W_v` matrix, **partition the momentum matrix along the head dimension and orthogonalize each head's block separately**.

The reasoning: full-matrix orthogonalization treats all heads as one coupled block, so heads with larger gradient/momentum scale dominate the shared update direction while small-scale heads get under-normalized updates. Per-head orthogonalization equalizes update scale across heads. Bonus — Newton–Schulz on tall per-head blocks is *cheaper* than on the full matrix.

If you try Muon on your GPT-2 124M run, this is a nearly free variant to test: the split is along an axis you already have.

## Pre-Training

**Data**: four text domains (Web, Code, Math, Knowledge) plus a vision corpus (captions, interleaved docs, OCR, perception, video, visual coding). K2's rephrasing recipe carries over — style/perspective-diverse prompting, chunk-wise autoregressive generation, fidelity verification against the source. Vision coordinates are supervised in **both absolute and normalized [0,1] formats** for resolution-robust localization, and programmatic multimodal data (code paired with its rendered output: SVG, 3D assets, webpages, games, CAD) is scaled up substantially.

**Scaling law**: re-fit from scratch, since the architecture change moves the optimum. They retuned batch size, learning rate, tokens-per-parameter, and model shape. Fig. 7 is the 2.5×-efficiency claim.

**Cosine beat WSD, and the methodology is the interesting part.** Prior work reports WSD matching or beating cosine. K3 argues those comparisons are confounded: the two schedules have *substantially different* optimal peak LR and batch size even at fixed model size and token budget, so a shared hyperparameter set unfairly favours whichever it happens to suit. K3 ran an independent scaling-law search per schedule, and under each schedule's own optimum, cosine won on final loss. Worth remembering as a general pattern — a schedule comparison without per-schedule hyperparameter search proves nothing.

**Optimizer/recipe**: Per-Head Muon + K2's weight clipping, QB for load balancing, cosine LR with 1% linear warmup, weight decay 0.1 throughout.

**Context curriculum** — four stages, and the cost structure is the point:

```
Pre-training:   8K  →  64K
Cooldown:     256K  →   1M
```

The expensive long-sequence computation is concentrated in a small fraction of the total budget. Long-context data gets a dedicated cleaning pipeline (exact + fuzzy dedup, perceptual hashing over video frames, structural validation) and is upsampled, because genuinely long coherent documents are scarce.

Critically: **length alone does not confer long-range capability.** K3 synthesizes long-context data by permuting and concatenating multimodal documents and sub-tasks so the embedded tasks are *only* solvable by attending across the full window — otherwise attention degenerates into local patterns. If you have ever seen a model with a nominal 128K window fail at 40K retrieval, this is the missing training signal.

## Post-Training

Three stages: SFT → nine RL experts → distill back into one model.

```
                    ┌─ general tasks ────┐
SFT (cold start) ──→├─ general agents ───┤ × {low, high, max} effort = 9 experts
   XTML template    └─ coding agents ────┘
                              │
                              ▼
              MOPD: multi-teacher on-policy distillation → Kimi K3
```

### SFT

Trajectories synthesized by domain-specialized models from earlier Kimi generations, then multi-stage verified and human-annotated. All serialized in the XTML chat template. **MXFP4 QAT starts here**, not after.

### RL

Rather than one model per task, K3 trains one expert per (domain, effort) pair. Fig. 8 shows the load-bearing result: as RL FLOPs scale, **average assistant tool-call steps scale up** alongside scores across coding experience, general tool use, web development, agentic search, professional workflows, office deliverables, agentic chart understanding, and agentic visual puzzles. The model learns to *take more steps*, not just to answer better.

**Partial rollouts**, extended from K1.5/K2.5: sample `K` completions for each of `N` prompts, maintaining `N×K` active trajectories. Instead of waiting for all of them, the generation phase pauses as soon as a fraction `λ ∈ (0,1)` of trajectories finishes. Paused rollouts are enqueued and prioritized for resumption next iteration.

The consequence is severe: a single long-horizon trajectory spans multiple iterations, so the data is heavily stale — extreme off-policy. K3 leans on a **per-token regularization** that constrains policy updates to a local neighborhood, which is what makes the stale data survivable.

**Reasoning-Effort RL** — a budget-control mechanism, not a prompt:

```
For problem x, estimate initial budget b₀(x) from the cold-start model.
Override task reward with −1 if T(y) > τ · b₀(x).

T(y) = thinking tokens                              (general tasks)
T(y) = cumulative output tokens, reasoning + tool-call args  (agentic tasks)

Curriculum: train max-budget first (large τ, still capped to suppress
overthinking), then anneal τ down to get high- and low-effort experts.
```

τ is tuned per domain with human-in-the-loop guidance. Trajectories from all effort levels feed both SFT and MOPD.

**Agentic Generative Reward Model** for non-verifiable tasks: tournament-style group reward with binary comparisons, and the judge must follow a mandatory four-step protocol — (1) read the outcome/product/text, (2) generate a rubric, (3) score each candidate against it, (4) record scores in a scorepad. Anti-reward-hacking: given cold-start verbosity `ℓ₀` and multiplier `σ`, any candidate exceeding `σ·ℓ₀` **automatically loses** its binary comparison. Same budget-control idea as reasoning effort, applied to verbosity.

### MOPD

The per-token reward for consolidating nine teachers into one student (Eq. 15), for domain `d` and sampled effort `e`:

```
r_opd(y_t | e, x, y_<t) = clip( sg( log [ π_teacher^{(d,e)}(y_t | x, y_<t)
                                        / π_θ(y_t | e, x, y_<t) ] ),
                                −R_max, R_max )
```

`sg` is stop-gradient; `R_max` clips extreme advantages. Because this is a dense *reward* rather than a separate loss, it drops straight into the existing RL framework — so partial rollouts and every other infra optimization apply to distillation unchanged. Notably: they tried finer-grained top-k distillation objectives and found **no advantage** in convergence speed or final performance.

### Deployment-aware post-training

**MXFP4 QAT.** MoE expert weights (which dominate parameter memory) → MXFP4, activations → MXFP8; attention projections, latent MoE projections, shared experts, and routers stay higher precision. Run through the *entire* post-training stage, SFT and RL. The key consequence: during RL, **rollout and training share the same quantization scheme**, eliminating the train–inference mismatch that normally plagues quantized RL.

**Draft model from the MTP layer.** K3 pre-trains an MTP layer mirroring a backbone block. Since EAGLE-3's draft is a single decoder layer of matching structure, they fine-tune the MTP layer into an EAGLE-3 draft with the target frozen. Details worth noting:

- Draft is unrolled **7 steps** during training; past the first step it consumes its own outputs, mirroring inference.
- Draft input fuses low/mid/high features from the **1st, 4th, and final AttnRes blocks**, projected by a bias-free `W_E3` initialized as `[0 0 I]` — so at init the fused representation equals the high-level feature the MTP layer was pre-trained on, and it learns to use the other two.
- Trained with the **LK loss** ([arXiv 2602.23881](https://arxiv.org/abs/2602.23881)) — the negative log of the acceptance rate itself, not a KL surrogate (Eq. 16):

```
L_LK = −log Σ_{x∈V} min( p(x), q(x) )
```

The logic: speculative-decoding speedup is governed by `Σ min(p,q)`, and minimizing KL does not maximize that for a capacity-limited draft. Optimize the thing you actually want.

### RL environments

This section is where the report spends the most non-architectural effort, and it is the most transferable part.

**Unified white-box RL environment.** Training with one fixed harness makes the model overfit to that tool schema, system prompt, and context-management strategy. K3 represents a harness as **configurable composable modules** — tool interfaces, system prompts, context management, skills, memories, subagents — and can instantiate Kimi Code, Claude Code, Codex, OpenClaw, and Hermes, plus novel ones. During training, harness configurations are generated dynamically per task group.

**Knowledge-graph-guided task synthesis.** A self-evolving DAG built by recursive agent-driven expansion: seed nodes → an agent per node does web searches → before adding nodes it explores the existing graph to reuse equivalents and minimize duplication → edges point coarse→fine → a branch stops when the agent judges the concept atomic. Then sample nodes at varying granularity (individually or in related combinations), combine keywords with ancestor context to form web queries, retrieve real materials, and synthesize tasks.

**Kernel optimization tasks** — sourced from repos like flash-linear-attention, spanning CUDA, Triton, CuTe DSL, Gluon, ThunderKittens, TileLang, and BF16/FP8/FP4. Reward structure:

```
numerical error above threshold  →  reward 0
matching an expert implementation →  reward 0.5
approaching the hardware roofline →  reward → 1
```

Plus an explicit **hacking-detection system** penalizing CUDA graph replay, input caching, and precision reduction, extended continuously as new hacks appeared during development.

**Personal assistant tasks** — realistic mock Gmail, Notion, Slack, Canvas. Persistent evolving environments spanning multiple simulated days with dozens of interdependent events; **a single rollout may involve thousands of tool calls and millions of context tokens.** Each event has its own evaluation criterion.

**Autonomous Execution Tasks (AET)** — the agent sees only objective, context, constraints, and verification interfaces. No reference trajectories, no predefined procedures. It must decompose, select tools, plan, recover from errors, and decide when to stop. Rewards come from a verifier's assessment of the **final environment state**, not the agent's self-report. Anti-hacking: agents are isolated from verifiers; public verifiers give diagnostic feedback while **hidden verifiers** evaluate held-out scenarios; penalty-based rewards under limited submission budgets.

## Infrastructure: the co-design

The report's real thesis is that KDA, 2.8T sparsity, and 1M agentic RL each break the systems stack, and the fixes are inseparable from the architecture. Highlights an ML engineer should know about:

### KDA Context Parallelism

Standard linear-attention context parallelism computes, on each rank, the state its local tokens generate from `S = 0`, then **sums** local states across preceding ranks. That works for vanilla linear attention because the recurrence is additive. **It fails for KDA**, because KDA applies a token-dependent matrix `M_t = (I − β_t k_t k_tᵀ)Diag(α_t)` to the incoming state — the effect of a segment depends on the state entering it.

KCP decomposes each segment's effect into two locally computable pieces (Eq. 17): a cumulative transition `M^{t←1}` and a from-zero state `S̃`:

```
S^t_{[i+1]} = S̃^t_{[i+1]}  +  M^{t←1}_{[i+1]} · S^{T_i}_{[i]}
              └ local ┘        └ propagate context from preceding ranks ┘
```

Both quantities are computable from local tokens alone, before the incoming state arrives. They compose associatively, so incoming states recover via a **prefix scan** — one fixed-size all-gather, independent of sequence length. Contrast with softmax attention's ring/CP, where exchanged KV blocks grow with sequence length.

Within a device, an SM-level context-parallel planner partitions the sequence across the SMs of a single rank (pure TP leaves most SMs idle when each rank holds only a few heads), with zero cross-device communication.

Kernels: **FlashKDA**, a CUTLASS chunkwise kernel that overlaps intra-chunk computation with cross-chunk state propagation, auto-dispatched as a backend of flash-linear-attention.

### MoonEP and a provable bound

MoonEP requires every rank to receive **exactly `S×K` tokens**, using dynamic redundant experts. The theorem (Appendix E): a balanced plan always exists with at most **`E/R` redundant experts per rank** (`E` experts, `R` EP size), and this bound is essentially tight — there exist router outputs requiring `⌈E(R−1)/R²⌉`.

Check K3's numbers at `E = 896`, `R = 64`:

```
upper bound     E/R              = 896/64            = 14
tightness       ⌈E(R−1)/R²⌉      = ⌈896×63/4096⌉     = ⌈13.78⌉ = 14
```

Exactly tight. Reserving 14 redundant-expert slots per rank **guarantees** a feasible plan, so training is never interrupted. Prior work (ECHO, UltraEP) presets a redundant-expert count or per-rank token cap and must stop when no feasible plan exists — plus the cap needs manual tuning and still leaves residual imbalance.

Perfect balance cascades into three more wins:

- **Zero-copy communication**: the planner precomputes each token's destination, so tokens go directly to expert-grouped positions on remote ranks. DeepEP's worst-case copy-free buffer is `S×K×R`; MoonEP needs a fixed `S×K`.
- **Static shapes**: per-expert token counts no longer vary, so the host never syncs with the device to learn computation shapes. Eliminates per-layer MoE host synchronization.
- Residual per-expert skew *within* a rank is handled by a workload-aware GEMM scheduler tuned before launch via an analytical cost model.

### Memory and RL infrastructure

Selected items: a unified activation manager where recomputation, quantization, and offload are pluggable per-tensor storage policies declared by annotation (decoupled from model code); block-wise FP8 activation quantization plus offload; remote activation offload to *other PP ranks'* memory via the Mooncake Transfer Engine to fix 1F1B's uneven activation distribution; Pipeline ZeRO-2 gradient sharding with CPU-resident shards; and **P2P Muon orthogonalization** — each rank fetches only the shards of parameters it owns instead of all-gathering the whole parameter buffer.

For 1M agentic RL: an **external KV cache pool** in CPU DRAM with write-back (only prefixes evicted from GPU are copied, avoiding redundant copies of still-active blocks), with **KDA states offloaded and prefetched together with their MLA KV blocks** to keep lifecycles aligned. Training states go to NVMe between iterations to free DRAM. A rollout auto-throttler uses active/queued request counts and KV utilization to modulate concurrency as contexts grow. Reference-model weights are materialized into the policy's FP32 gradient-buffer storage — safe because those buffers get overwritten when real gradients arrive.

### AgentENV

Firecracker microVMs, not containers. The motivation is honest: with container runtimes they observed **kernel panics and deadlocks caused by unintended agent operations**, while also wanting agents free to mount disks, run containers, even launch VMs.

```
incremental checkpoint   133 ms   (only pages dirtied since last checkpoint)
resume                    49 ms
Pause/Resume   a paused sandbox consumes no memory or CPU — and the agent
               spends up to 98% of sandbox lifetime waiting on inference
Fork           clone exact state while the original keeps running (reward
               judging with no side effects)
Snapshot       periodic, for error recovery
memory overcommit up to 6.5× in real workloads (OverlayBD + custom ublk, P2P transport)
```

The number that conveys the scale of K3's post-training: **51,219,741 sandboxes across 1,505,678 images** were created during training and evaluation.

### KDA-aware prefix caching

The hybrid architecture breaks conventional prefix caching, and the reasoning is worth following because it generalizes to any hybrid model.

Block-hash prefix caching hashes complete physical blocks, so only block-aligned prefixes are reusable. But a KDA layer holds **one large recurrent state per sequence**, so state snapshots are only affordable at sparse boundaries — forcing a shared block size of **1024–6144 tokens**. At that granularity caching is nearly useless: requests shorter than one block are never reusable, and chunked prefill exports nothing until it crosses a full block boundary.

K3 decouples the granularities:

```
physical block   6144 tokens   ← allocation unit (sized for KDA checkpoints)
hash block        512 tokens   ← prefix-hash unit (inside MLA pages)

KDA checkpoints saved only at a sparse subset of MLA hash endpoints —
the only positions a lookup can ever reference. Checkpoints at
conversation-turn boundaries are retained; intermediate ones recycled.
```

Lookup is two-stage: match MLA physical blocks by chained hash, falling back to hash endpoints inside the first missing block; then require a KDA checkpoint at that boundary in *every* KDA cache group. The hit is the longest boundary satisfying both — a multiple of 512, never required to be a multiple of 6144. Example from Fig. 12: a request matching 2800 tokens hits at `B = 2560 = 5×512`, deep inside a 6144-token physical block, and resumes prefill from there.

Both cache types are packed into **one paged pool with identical page byte-size**, sharing a single implementation of allocation, reference counting, and eviction. A nice accidental property they call out: any type-confused access yields garbage rather than plausible data — a zero-overhead sanity check.

Three concurrency invariants, each dictated by a concrete failure mode: hit blocks are **pinned across all cache groups** before anything is allocated; blocks allocated or registered in the current scheduling step are **excluded from matching** until their GPU copies land; and evicting one group's checkpoint **atomically invalidates its siblings** (hittable in every group or none).

### Speculative decoding with a recurrent state

KDA's state updates **in place** every decoding step. If MTP verification rejects drafted tokens, the state has already advanced past the last accepted token and cannot be rolled back. Snapshotting per draft position would work but multiplies state traffic — fatal at serving batch sizes.

K3's fix: the state after any accepted draft prefix is fully determined by the **projected inputs** of the draft tokens, which are far smaller than the state. So cache only the projected inputs, rebuild accepted-token states **on-chip**, and write back only verified + bonus token states. Replayed tokens, bonus token, and the next draft window share one recurrent loop in a single fused kernel covering short convolution, input normalization, gating, the KDA recurrence, and output normalization. Verification latency grows **sub-linearly** in tokens verified. (Independently proposed as ReplaySSM by Dao AI Lab.)

### Fleet scheduling

**Cache-aware affinity**: at 1M context a typical coding request carries a 400K-token prefix but only 4K of new tokens, so a cache hit is orders of magnitude cheaper than a miss — route each session to the cluster holding its prefix cache. But that binds sessions to a cluster whose failure takes them all down. Fix: consistent hashing pins each session to **two** clusters, primary and pre-assigned secondary. The secondary holds no cache and must re-prefill on failover, but consistent hashing spreads secondary assignments uniformly, so failover work is divided across many clusters instead of concentrated on one.

**Budget-based admission control**: production traffic mixes sub-2K requests with 1M-token ones, so per-request cost spans ~3 orders of magnitude and "average request" capacity planning breaks down. Separate resource budgets per request class prevent a burst of long-context traffic from destroying TTFT for everyone else.

## Chat Template: XTML

K3 redesigned its chat template around extensibility, low alignment tax, and decoding friendliness (Appendix F). The result is XML semantics with three reserved special tokens replacing angle-bracket syntax:

```
[open]tag attr="value"[sep]  …content…  [close]tag[sep]
[end_of_msg]   ← generation stop marker
```

Every structural boundary is an explicit token, which removes tokenization ambiguity at element boundaries and simplifies grammar-constrained decoding.

**Channels** (inspired by OpenAI's Harmony format): `think` for reasoning, `response` for the user-visible answer, `tools` for tool calls. The two generation modes are selected purely by the **generation prefix** — `[open]think[sep]` vs `[open]response[sep]` — not by separate templates. K3 supports only **preserved thinking**: in thinking mode the `think` channel is always retained in history, kept even when empty, so message structure stays consistent across turns.

**Option-message placement is a KV-cache design decision**, and this is the cleverest bit:

```
global options    (tool-declare, thinking-effort)   BEFORE input messages
                  — govern the session, rarely change, so invalidating cache is fine
input options     (mid-session tool-declare)        INTERLEAVED
                  — dynamically loaded tools expand the toolset without rebuilding context
one-shot options  (tool_choice, response_format)    AFTER input messages
                  — per-request changes leave the history KV cache intact
```

**Tool calls** carry `tool` and `index` attributes; parallel calls are indexed and each result repeats the same tool/index pair. Arguments are **typed** — string arguments appear as raw text, other JSON types compactly serialized — so code is a first-class citizen rather than an escaped JSON string. (Anyone who has debugged a model emitting `"code": "def f():\\n    return \\\"x\\\""` will appreciate this.) A pure-JSON fallback block exists for undecomposable inputs; it occurs only in input tokens, never in model outputs, and **its loss is masked during training**.

**Reasoning effort** is a global option message stating the requested level **in natural language** — not a modified generation prefix, not an exposed token budget. The schema reserves four levels (low, medium, high, max); K3 supports a subset. Same for `tool_choice` and `response_format`: all option types become short natural-language instructions in context. The design principle: the pre-trained model already follows instructions well, so new options need little or no additional training. That is the "low alignment tax" claim, and it is a genuinely reusable idea.

## Evaluation

Every model evaluated at maximum reasoning effort (GPT-5.5 at "xhigh"). Kimi K3 at effort `max`, temperature 1.0, top-p 0.95 for single-step reasoning and 1.0 for agentic tasks.

Selected rows from Table 2 (**bold** = best of the six):

| Benchmark | Kimi K3 | Claude Fable 5 | GPT-5.6 Sol | Claude Opus 4.8 | GPT-5.5 | GLM-5.2 |
|---|---|---|---|---|---|---|
| GPQA Diamond | 93.5 | 92.6 | **94.1** | 91.0 | 93.5 | 91.2 |
| HLE-Full (no tool / tool) | 43.5 / 56.0 | **53.3 / 63.0** | 44.5 / 58.0 | 49.8 / 57.9 | 41.4 / 52.2 | – |
| CritPt | 23.4 | 28.6 | **32.3** | 20.9 | 27.1 | 20.9 |
| AA-LCR | **74.7** | 70.0 | 73.7 | 67.7 | 74.3 | 71.3 |
| DeepSWE | 67.5 | 70.0 | **73.0** | 59.0 | 67.0 | 46.2 |
| ProgramBench | **77.8** | 76.8 | 77.6 | 71.9 | 70.8 | 63.7 |
| Terminal-Bench 2.1 | 88.3 | 88.0 | **88.8** | 84.6 | 83.4 | 82.7 |
| FrontierSWE | 81.2 | **86.6** | 71.3 | 66.7 | 64.9 | 67.3 |
| SWE-Marathon | **42.0** | 35.0 | 39.0 | 40.0 | 14.0 | 13.0 |
| PostTrainBench | 36.6 | **41.4** | 34.6 | 34.1 | 28.4 | 34.3 |
| BrowseComp | **91.2** | 88.0 | 90.4 | 84.3 | 84.4 | – |
| DeepSearchQA (F1) | **95.0** | 94.2 | – | 93.1 | – | – |
| GDPval-AA v2 (Elo) | 1686 | **1747** | 1736 | 1593 | 1491 | 1510 |
| MCPMark-Verified | **94.5** | 87.4 | 92.9 | 76.4 | 92.9 | – |
| AutomationBench | **30.8** | 29.1 | 29.7 | 27.2 | 22.7 | 12.9 |
| Harvey Lab-AA | **94.6** | 93.6 | 87.2 | 91.1 | 86.3 | 91.0 |
| τ³-Banking | **33.4** | 26.8 | 33.0 | 27.6 | 31.3 | 26.8 |
| OSWorld 2.0 | 58.3 | **66.1** | 62.6 | 55.7 | 49.5 | – |
| OmniDocBench | **91.1** | 89.8 | 85.8 | 87.9 | 89.4 | – |
| Video-MME (w/ sub) | **90.0** | – | 89.5 | 86.0 | 89.3 | – |
| Math-Vision (no tool / Python) | 94.3 / 97.8 | 94.8 / **98.6** | **95.8** / 97.8 | 86.7 / 97.1 | 92.2 / 96.8 | – |
| ZeroBench-main pass@5 (no tool / Python) | **23.0** / 41.0 | **23.0** / **46.0** | 17.0 / 35.0 | 17.0 / 34.0 | 22.0 / 41.0 | – |

The report is unusually candid about the shape of these results.

**Where K3 leads**: long-horizon agentic execution. SWE-Marathon 42.0 is **7 points ahead of Fable 5**. BrowseComp, DeepSearchQA, MCPMark, AutomationBench, τ³-Banking, Harvey Lab-AA, SpreadsheetBench 2, ResearchRubrics — all SOTA. On in-house benchmarks the pattern sharpens: Swarm Bench 76.3 and Deep Research Bench 90.0 lead by clear margins, and on Kimi Webdev Bench blind expert judges prefer K3 over Claude Opus 4.8 by **+31.0 points overall** (+59.1 on 3D/WebGL/Shader).

**Where it trails**: research-level reasoning. HLE-Full 43.5/56.0 vs Fable 5's 53.3/63.0 — a ~10-point gap that tools don't close. CritPt 23.4 vs 32.3. The Elo-rated knowledge-work suites (GDPval-AA v2, AA-Briefcase) both go to Fable 5. Internally, K3 trails on Agent Behavior Bench (process quality, not just outcome), MIRA Bench, and 24/7 ClawBench 2.0.

**Read the footnotes.** Fable 5 results "include potential fallbacks" and GPT-5.6 Sol results "include potential cyberguards." Fable 5 hits fallbacks on 35% of SWE-Marathon tasks, and Table 3 footnotes count 13 fallbacks + 1 refusal out of 80 on Kimi Code Bench 2.0, 10 refusals out of 80 for GPT-5.6 Sol on Codex, 14 refusals for Fable 5 on Online Experience. These are self-reported comparisons under a vendor's own harness assignments; a fair reading is that the fallback/refusal accounting cuts both ways and none of it is independently audited.

One number that deserves care: BrowseComp's headline **91.2% uses context compaction triggered at 300K tokens**. With the full 1M window and *no* context management, K3 gets **90.4%** — exactly matching GPT-5.6 Sol. So "we have 1M context" and "1M context is the best way to use the model" are different claims, and the report's own numbers show it.

### Third-party (as of July 23, 2026)

| | Kimi K3 | Fable 5 | GPT-5.6 Sol | Opus 4.8 | GPT-5.5 | GLM-5.2 |
|---|---|---|---|---|---|---|
| AA Intelligence Index v4.1 (#4/580) | 57.1 | **59.9** | 58.9 | 55.7 | 55.0 | 51.1 |
| Vals Index (#2/39) | 74.7 | **75.1** | 73.1 | 70.4 | 68.0 | 65.0 |
| WebDev Arena Elo (**#1/99**) | **1,678** | 1,634 | 1,630 | 1,565 | 1,507 | 1,592 |
| Text Arena Elo (#8/200) | 1,486 | **1,507** | 1,485 | 1,484 | 1,482 | 1,469 |
| Agent Arena (#4/37) | 9.1 | **12.7** | 10.1 | 9.8 | 8.8 | 6.5 |

**First open model to top WebDev Arena.** The Text Arena rank (#8) versus WebDev (#1) is a real signal about where the post-training effort went.

### Cost efficiency

| Suite | K3 result | Cost position |
|---|---|---|
| Kimi Code Bench 2.0 | 4.0 pts behind Fable 5 | at **38%** of Fable 5's cost; K3 at *high* effort matches Opus 4.8 at *max* effort for ~⅓ the cost |
| BrowseComp | **91.2%** (best) | **$2.03/task** — half of GPT-5.6 Sol, order of magnitude below Claude at max effort |
| GDPval-AA v2 | within 50 Elo of GPT-5.6 Sol | 13% lower cost; **2.6× cheaper** than Fable 5 |
| AA-Briefcase | 2nd behind Fable 5 | ~half Fable 5's cost |

### Cyber capability

Reported as a two-tier progression, and notable because Anthropic and OpenAI frontier models refuse these tasks, making comparison infeasible — so K3 is benchmarked against GLM-5.2 only.

**Tier 1 (vulnerability discovery, defensive)**: across dozens of deployed systems — OS kernels, databases, AI services, web frameworks, blockchain, VPN — hundreds of candidate vulnerabilities; of those human-reviewed, **~70% confirmed genuine, including 16 previously unknown across six projects**. Two Linux kernel findings are described: a remotely triggerable heap out-of-bounds write introduced by an incomplete upstream fix (confirmed as a remote DoS primitive), and a Dirty-COW-class RDMA subsystem bug where an earlier fix dropped a permission check (confirmed as a deterministic local privilege-escalation primitive).

**Tier 2 (exploit development)**: 36 tasks, all verified human-solvable, estimated **~540 expert-hours total (~15 h/task)**. K3 solves **14/36 (38.9%)** vs GLM-5.2's 8/36 (22.2%) — but 10 of 14 come from the user-space track; on the 20-task kernel track neither model solves three quarters. Four recurring failure modes: can't finish the final stage of an exploit chain from primitives already obtained; poor strategy selection under mitigations (persisting with control-flow hijacking when a data-only attack is simpler); prolonged unproductive debugging loops; insufficient verification before submission.

A **joint UK AI Security Institute / NIST CAISI assessment** concurs: K3 beats GLM-5.2 (32% vs 24% on ExploitBench; 17 vs 11 steps on a 32-step simulated enterprise network that takes a human expert ~20 hours) but achieves arbitrary code execution on **0 of 41** tasks.

### Case studies

Selected, with real numbers:

- **GPU kernel optimization** (24h budget/task, four kernels, Hopper + an alternative-vendor GPGPU): AttnRes latency **283.6 ms → 114.4 ms**; DSA −55.1%; KDA −73.6%; >half of peak TFLOPS on MLA. Matched Fable 5 (with fallback), beat Opus 4.8 / GPT-5.6 Sol / GPT-5.5. The report notes an early K3 checkpoint "was already handling most of our kernel optimization work during late-stage development."
- **GPU compiler**: built [MiniTriton](https://github.com/MoonshotAI/minitriton) — Python tile DSL → warp-level MLIR → PTX codegen, plus a dual-mode tensor library with reverse-mode autograd, NN modules, and NCCL distributed primitives. Beats torch eager and `torch.compile` in geomean on its suite; from-scratch tensor-core matmul reaches ~90% of measured machine roof; trains a GPT end-to-end with gradients within fp32 rounding error (10⁻⁴) of torch autograd against an fp64 reference.
- **Chip design**: [nano-kpu](https://github.com/MoonshotAI/nano-kpu) — an inference-chip prototype for a nano model with the same architecture family (hybrid KDA + NoPE-MLA, Block AttnRes with block size 2, sigmoid MoE routing with one shared expert, INT4 group-128 weights). Single **48-hour autonomous run** with Kimi Code using open-source EDA tools and Nangate45: closes timing at 100 MHz in 4 mm², RTL-simulated decode throughput **>8,700 tokens/s**, 1.46M standard cells, 0.277 MiB SRAM.
- **Research coding**: reproduced the I–Love–Q universal relations — reviewed 20+ papers, evaluated 300+ equations of state, found inconsistencies in published formulas, wrote 3,000+ lines of Python, produced an interactive dashboard — **in ~2 hours vs 1–2 weeks** for an experienced researcher.
- **Knowledge work**: a 42-year AI-ASIC-industry research website over 120+ refinement rounds, from 87 quarterly reports and 99 PDFs (11,000+ pages) via 2,800+ web searches and 1,100+ terminal queries.

## Kimi K3 vs Alternatives

| | Kimi K3 | Kimi K2 | DeepSeek-V4-Pro | GLM-5.2 |
|---|---|---|---|---|
| Total / active | 2.78T / 104.2B | 1.04T / 32.6B | 1.6T / 49B | 744B / 40B |
| Layers | 93 | 61 | 61 | – |
| Attention | 69 KDA + 24 Gated MLA (3:1) | 61 MLA | CSA + HCA hybrid | DSA + IndexShare |
| Positional | **NoPE everywhere** | RoPE + YaRN to 128K | RoPE | RoPE |
| Routed / active experts | 896 / 16 (sparsity 56) | 384 / 8 (sparsity 48) | 384 / 6 (+1 shared) | – |
| Routed expert width | **latent 3584 (0.5d)** | full 7168 | full | full |
| Residual | **Block AttnRes** | standard | mHC (hyper-connections) | standard |
| Activation | **SiTU-GLU** | SwiGLU | SwiGLU | SwiGLU |
| Load balancing | **Quantile Balancing** | aux-loss-free sign rule | aux-loss-free + hash routing | aux-loss-free |
| Context | 1M | 128K | 1M | 1M |
| Optimizer | **Per-Head Muon** | MuonClip | Muon | – |
| Expert quantization | MXFP4 (QAT from SFT) | FP8 | FP4 | – |
| Vision | native, MoonViT-V2 from scratch | text-only | text-only | – |
| License | Kimi K3 License (modified MIT) | Modified MIT | MIT | – |

Three genuinely distinct bets, worth naming:

1. **DeepSeek-V4** attacks 1M context with two *sparse softmax* mechanisms (compress then top-k, compress harder then dense). **K3** attacks it by making 74% of layers not have a growing cache at all. Sparse-attention-with-compression vs linear-attention-with-fixed-state — the two live options.
2. **DeepSeek-V4's mHC** and **K3's AttnRes** both replace the residual stream, and both cite the Hyper-Connections lineage, but mHC uses `n=4` parallel tracks mixed by a doubly-stochastic matrix while AttnRes uses *data-dependent softmax over a growing source set*. mHC changes the width of the residual; AttnRes changes its connectivity graph.
3. Everyone runs aux-loss-free routing, but only K3 identified it as SignSGD on a solvable dual and replaced it with the closed-form solution.

## Practical Considerations

**Deployment reality.** 2.78T parameters with MXFP4 experts is roughly a 600 GB download — 8×H100-80GB is the practical floor for the native checkpoint, realistically a terabyte-class memory pool. Sparsity buys throughput, not footprint: all 896 experts must be resident somewhere. Official engine support is vLLM, SGLang, and TokenSpeed. As of July 28, 2026 there is no working llama.cpp/Ollama port — the open blockers are the per-token softmax-over-depth AttnRes op and the 896-expert MoE, and no verified community local run or Dynamic GGUF exists yet. Treat any circulating quant-size numbers as speculation.

**API.** Listed at $3.00/1M input (cache miss), $0.30/1M input (cache hit), $15.00/1M output, flat across the full 1,048,576-token window. That is ~3× the K2-era pricing and squarely Sonnet-tier — the "cheap Chinese open model" framing no longer applies. Cache economics matter a lot here: the 10× miss/hit spread is exactly what the fleet-level cache-affinity scheduling in §5.4.3 exists to exploit.

**Inference settings** (from §6.1.3, these are the paper's own recommendations): temperature 1.0; **top-p 0.95 for reasoning and knowledge tasks, top-p 1.0 for coding and agentic scenarios**.

**Use K3 when**: long-horizon agentic execution over hundreds-to-thousands of tool calls; web/frontend development (WebDev Arena #1, +31 pts blind-judged vs Opus 4.8); GPU kernel and low-level systems work (SWE-Marathon 42.0, and its self-reported kernel case studies are the strongest in the report); deep research and browsing (BrowseComp 91.2 at $2.03/task); anything where you need open weights at frontier capability.

**Don't reach for K3 when**: research-level reasoning is the bottleneck (HLE, CritPt — a real ~10-point gap to Fable 5); you need process quality rather than outcome correctness (it trails on Agent Behavior Bench); you're deploying under a terabyte of accelerator memory; or your MaaS business clears $20M/12mo, in which case read the license before you ship.

**Don't assume 1M is free.** The BrowseComp result shows context compaction at 300K *beating* the naive full-window run. The 1M window is architectural headroom, not an instruction to fill it.

**Ideas that transfer to your own scale.** Several things here are cheap to try on your 1×H800:

- **Per-Head Muon** on your GPT-2 124M run — the head-axis split already exists in your code; it's a few lines on top of a Muon implementation.
- **The bounded-decay lesson**: if you ever implement a gated linear-attention layer, bound the log-decay from below. It isn't about model quality, it's about which kernel path your tiles take.
- **The WSD-vs-cosine methodology**: whenever you compare two LR schedules, tune hyperparameters *per schedule*. K3's result is that the standard shared-hyperparameter comparison is not evidence.
- **Verbosity control as a reward-hacking guard** for your DPO/GRPO work: estimate `ℓ₀` from the cold-start model and auto-lose any comparison exceeding `σ·ℓ₀`. Length-bias in preference models is exactly the failure you fought on hh-rlhf.
- **QB** if you ever train a small MoE — it removes the balancing learning rate entirely, and the histogram estimator is ~50 lines.

## Open Questions

The report is 47 pages and still does not disclose:

1. **Training compute.** No token count, no FLOPs, no GPU count, no wall-clock, no accelerator model for pre-training. K2's report gave 15.5T tokens. K3 gives a relative scaling-efficiency curve and nothing absolute. Notably conspicuous.
2. **Ablations for the individual architecture changes.** The 2.5× figure is for the whole package — architecture + data + training recipe together. There is no table separating KDA-bounded-decay from AttnRes from LatentMoE from data improvements. The Attention Residuals and Kimi Linear preprints carry their own ablations, but nothing at 2.8T scale.
3. **What bounded decay costs.** `α > e⁻⁵` means a channel cannot fully forget within a tile. No long-range retrieval ablation against the unbounded parameterization.
4. **Post-training data volume.** Environments are described in rich qualitative detail; there are no trajectory counts, no RL step counts, no per-domain data sizes. The 51.2M sandbox figure is the only quantitative handle.
5. **Safety and alignment.** There's a substantial cyber-capability evaluation and a Faithfulness row, but no refusal/jailbreak/bias evaluation and no safety framework discussion — a striking omission for a frontier-capable open-weight release with demonstrated exploit-development ability.
6. **Independent verification.** No third-party eval of K3 beyond the cited leaderboards existed at release. The precedent worth remembering is K2's Humanity's Last Exam figure, where the self-reported number and the independently measured one differed by roughly 20 points.

## Key Papers

1. [Kimi K3: Open Frontier Intelligence](https://arxiv.org/abs/2607.24653) — Kimi Team, Moonshot AI, July 27, 2026. **Primary source for this guide.** Also at [github.com/MoonshotAI/Kimi-K3](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf).
2. [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692) — Kimi Team, Oct 2025. Origin of KDA and the 3:1 hybrid.
3. [Attention Residuals](https://arxiv.org/abs/2603.15031) — Kimi Team, Mar 2026. Depth-wise attention replacing the residual stream. [Code](https://github.com/MoonshotAI/Attention-Residuals).
4. [LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts](https://arxiv.org/abs/2601.18089) — Elango et al., Jan 2026.
5. [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534) — Kimi Team, July 2025. The 1.04T predecessor.
6. [Kimi K2.5: Visual Agentic Intelligence](https://arxiv.org/abs/2602.02276) — Kimi Team, Feb 2026. Agent Swarm, the vision pipeline K3 rebuilds.
7. [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://openreview.net/forum?id=r8H7xhYPwz) — Yang, Kautz, Hatamizadeh (NVIDIA), ICLR 2025. KDA's direct ancestor.
8. [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484) — Yang et al., NeurIPS 2024. The chunkwise/UT-transform machinery.
9. [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635) — Yang et al., ICML 2024. Secondary tiling.
10. [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174) — Schlag, Irie, Schmidhuber, ICML 2021. The delta rule.
11. [Transformers are SSMs (Mamba-2 / SSD)](https://arxiv.org/abs/2405.21060) — Dao & Gu, 2024. The decay parameterization K3 replaces.
12. [DeepSeek-V2](https://arxiv.org/abs/2405.04434) — MLA, retained in K3's global layers.
13. [DeepSeekMoE](https://arxiv.org/abs/2401.06066) / [DeepSeek-V3](https://arxiv.org/abs/2412.19437) — shared+routed organization; the sign-rule bias update QB replaces.
14. [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — Shazeer, 2020. SwiGLU, which SiTU-GLU caps.
15. [Gated Attention for Large Language Models](https://arxiv.org/abs/2505.06708) — Qiu et al., 2025. The output-gating design K3 adopts for both KDA and MLA.
16. [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) — Moonshot, Feb 2025. Baseline for Per-Head Muon.
17. [EAGLE-3](https://arxiv.org/abs/2503.01840) — Li et al., 2025. Draft-model architecture K3 fine-tunes its MTP layer into.
18. [LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding](https://arxiv.org/abs/2602.23881) — Samarin et al., 2026.
19. [Why Low-Precision Transformer Training Fails: An Analysis on Flash Attention](https://arxiv.org/abs/2510.04212) — Qiu & Yao, ICLR 2026. The FP32-attention-output fix.
20. [Microscaling Data Formats for Deep Learning](https://arxiv.org/abs/2310.10537) — Rouhani et al., 2023. MXFP4/MXFP8.
21. [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079) — Moonshot, 2024. Transfer Engine used for remote activation offload.
22. [Preliminary Assessment of Kimi K3's Cyber Capabilities](https://www.aisi.gov.uk/blog/preliminary-assessment-of-kimi-k3s-cyber-capabilities) — UK AISI + NIST CAISI, July 2026.

## Live Resources

- [Kimi K3 weights (HuggingFace)](https://huggingface.co/moonshotai/Kimi-K3) · [GitHub release repo](https://github.com/MoonshotAI/Kimi-K3) · [blog](https://www.kimi.com/blog/kimi-k3)
- [FlashKDA](https://github.com/MoonshotAI/FlashKDA) — CUTLASS chunkwise KDA kernels, dispatched from flash-linear-attention
- [MoonEP](https://github.com/MoonshotAI/MoonEP) — perfectly balanced expert-parallel communication with dynamic redundant experts
- [Attention-Residuals](https://github.com/MoonshotAI/Attention-Residuals) — reference AttnRes / Block AttnRes implementation
- [AgentENV](https://github.com/kvcache-ai/AgentENV) — Firecracker-microVM agent sandbox runtime
- [MiniTriton](https://github.com/MoonshotAI/minitriton) · [nano-kpu](https://github.com/MoonshotAI/nano-kpu) — artifacts K3 itself produced
- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) — where the KDA reference kernels live
- [Simon Willison on Kimi K3](https://simonwillison.net/2026/Jul/16/kimi-k3/)
</content>
</invoke>
