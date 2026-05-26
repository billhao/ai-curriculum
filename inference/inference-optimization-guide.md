# LLM Inference Optimization

A guide to making LLM inference fast, memory-efficient, and cost-effective — from KV caches and quantization to speculative decoding and production serving frameworks.

## Background

**Foundational context**: LLM inference is fundamentally different from training. Training is compute-bound (matrix multiplications dominate). Inference, especially autoregressive text generation, is **memory-bandwidth-bound** — each new token requires reading the entire model's weights from GPU memory, but performs relatively little computation per byte read. This insight drives every optimization in this guide.

**Research lineage** — inference optimization builds on a chain of work:

1. **Multi-Query Attention (MQA)** (Shazeer, Google, 2019) — [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150). Reduced KV cache size by sharing key-value heads across all query heads. First major architectural change targeting inference speed.

2. **FlashAttention** (Dao et al., Stanford, 2022) — [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135). Tiling-based attention that reduces HBM reads/writes from O(N²) to O(N²/M) where M is SRAM size. 2-4x speedup on GPT-2, 15% end-to-end on BERT-large.

3. **GPTQ** (Frantar et al., IST Austria, 2022) — [GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers](https://arxiv.org/abs/2210.17323). One-shot weight quantization to INT4/INT3 using approximate second-order information. First to run a 175B model on a single GPU.

4. **Speculative Decoding** (Leviathan et al., Google, 2022; Chen et al., DeepMind, 2023) — [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192). Use a small draft model to propose tokens, verify in parallel with the large model. 2-3x speedup with identical output distribution.

5. **Grouped-Query Attention (GQA)** (Ainslie et al., Google, 2023) — [GQA: Training Generalized Multi-Query Attention Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245). Middle ground between MHA and MQA — fewer KV heads, 5% of original pretraining compute to uptrain. Adopted by Llama 2/3, Mistral, Qwen, and most modern models.

6. **PagedAttention / vLLM** (Kwon et al., UC Berkeley, 2023) — [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180). Virtual-memory-inspired KV cache management. Reduced memory waste from 60-80% to under 4%. Up to 24x throughput over HuggingFace.

7. **AWQ** (Lin et al., MIT, 2023) — [AWQ: Activation-Aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978). Protects 1% salient weights using activation statistics. First to run 70B Llama-2 on mobile GPUs. Best Paper at MLSys 2024.

8. **FlashAttention-2** (Dao, 2023) — [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691). 2x over FlashAttention-1, reaching 50-73% of A100 theoretical peak (225 TFLOPs/s).

9. **Medusa** (Cai et al., 2024) — [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774). Added extra decoding heads for parallel token prediction. 2.2-3.6x speedup without a separate draft model.

10. **EAGLE** (Li et al., 2024) — [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077). Feature-level autoregression for speculative decoding. 2.7-3.5x latency speedup on LLaMA2-Chat 70B.

11. **FlashAttention-3** (Shah et al., 2024) — [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608). Exploits Hopper GPU features (warp specialization, TMA). 740 TFLOPs/s FP16, 1.2 PFLOPs/s FP8 on H100.

12. **SGLang** (Zheng et al., UC Berkeley, 2024-2026) — RadixAttention for prefix caching, zero-overhead CPU scheduler. As of v0.5.9 (Feb 2026): supports FP4/FP8/INT4, tensor/pipeline/expert parallelism, 400K+ GPUs in production.

**The shift**: From 2020-2022, inference was an afterthought — you trained a model, then served it however you could. By 2025-2026, inference optimization is a distinct engineering discipline. A well-optimized serving stack can deliver 10-50x better throughput than naive HuggingFace `model.generate()`, at equal or better quality.

## Key Terms

**Memory-bandwidth-bound**: The bottleneck is reading data from GPU HBM, not computing with it. Autoregressive decoding reads all model weights for each token but only does one matrix-vector multiply per layer. The arithmetic intensity (FLOPs per byte) is very low.

**KV cache**: Stored key and value tensors from previous tokens in the attention mechanism, so they don't need to be recomputed at each generation step. Grows linearly with sequence length and is often the dominant memory consumer during inference.

**Prefill vs decode**: Two phases of inference. Prefill processes the entire input prompt in parallel (compute-bound, like training). Decode generates tokens one at a time (memory-bandwidth-bound). Most optimizations target the decode phase.

**Time to first token (TTFT)**: Latency from request arrival to the first output token. Dominated by prefill time.

**Time per output token (TPOT)**: Latency between consecutive output tokens during decode. Dominated by memory bandwidth.

**Throughput (tokens/sec)**: Total tokens generated per second across all concurrent requests. The key metric for serving economics.

**Arithmetic intensity**: FLOPs per byte of memory access. For decode with batch size 1, a transformer layer does ~2P FLOPs (P = parameters) but reads ~2P bytes (FP16), giving arithmetic intensity of ~1 FLOP/byte. The H800 has ~2 PFLOPS compute but ~3.35 TB/s bandwidth — it needs ~600 FLOPs/byte to be compute-bound. Batching increases arithmetic intensity.

**Roofline model**: A framework for understanding whether a workload is compute-bound or memory-bound based on arithmetic intensity vs. the hardware's compute-to-bandwidth ratio.

## KV Cache

### What It Is

In your GPT-2 124M, each transformer layer computes Q, K, V from the input and runs attention. During training, all tokens are processed in parallel — no caching needed. During autoregressive generation, you generate one token at a time, and each new token attends to all previous tokens.

Without a KV cache, generating token t requires recomputing K and V for all t-1 previous tokens — O(t²) total work across a full sequence. With a KV cache, you store K and V from previous steps and only compute K, V for the new token — O(t) per step, O(t²/2) total. This is the single most basic inference optimization, and every serving system uses it.

```
Without KV Cache (recompute every step):

Step 1: compute K₁,V₁ for token 1                        → 1 KV computation
Step 2: compute K₁,V₁,K₂,V₂ for tokens 1-2              → 2 KV computations
Step 3: compute K₁,V₁,K₂,V₂,K₃,V₃ for tokens 1-3       → 3 KV computations
...
Step T: compute K₁..Kₜ, V₁..Vₜ for all tokens            → T KV computations
                                                Total: T(T+1)/2 ≈ T²/2

With KV Cache (store and reuse):

Step 1: compute K₁,V₁ → store in cache                   → 1 KV computation
Step 2: compute K₂,V₂ → append to cache, attend to K₁₋₂  → 1 KV computation
Step 3: compute K₃,V₃ → append to cache, attend to K₁₋₃  → 1 KV computation
...
Step T: compute Kₜ,Vₜ → append to cache, attend to K₁₋ₜ  → 1 KV computation
                                                Total: T
```

### Memory Analysis — Numerical Walkthrough

KV cache memory per token per layer:

```
KV cache per token = 2 × n_kv_heads × d_head × dtype_bytes

Where:
  2              = one K tensor + one V tensor
  n_kv_heads     = number of KV heads (= n_heads for MHA, < n_heads for GQA/MQA)
  d_head         = dimension per head (typically d_model / n_heads)
  dtype_bytes    = 2 for FP16/BF16, 1 for FP8/INT8
```

**Example: Llama 3.1 70B** (GQA with 8 KV heads, 128 heads total, d_head=128, 80 layers, FP16):

```
Per token per layer = 2 × 8 × 128 × 2 bytes = 4,096 bytes = 4 KB
Per token all layers = 4 KB × 80 = 320 KB
For 4096 tokens      = 320 KB × 4096 = 1.25 GB
For 128K context     = 320 KB × 131072 = 40 GB
```

At 128K context, the KV cache alone consumes 40 GB — half your H800's 80 GB. The model weights (70B × 2 bytes FP16 = 140 GB) already don't fit on one GPU, so the KV cache budget becomes critical in multi-GPU setups.

**Comparison across model sizes** (FP16, single sequence, 4K context):

```
Model             Params   Layers  KV Heads  d_head  KV Cache/tok  KV Cache 4K  Weights
────────────────  ───────  ──────  ────────  ──────  ────────────  ───────────  ───────
GPT-2 124M        124M     12      12 (MHA)  64      3.0 KB        12 MB        0.25 GB
Llama 3.1 8B      8B       32      8 (GQA)   128     64 KB         256 MB       16 GB
Llama 3.1 70B     70B      80      8 (GQA)   128     320 KB        1.25 GB      140 GB
Llama 3.1 405B    405B     126     8 (GQA)   128     504 KB        1.97 GB      810 GB
DeepSeek-V3       671B*    61      128 (MLA) 128     2.0 MB**      7.8 GB       ~350 GB*
```

`*` DeepSeek-V3 is MoE — 671B total but ~37B active params. Weight memory depends on which experts are loaded.
`**` DeepSeek-V3 uses Multi-head Latent Attention (MLA) which compresses KV cache — effective size is much smaller than this naive calculation.

### MQA, GQA, and MLA

The standard multi-head attention (MHA) in your GPT-2 gives every query head its own K and V head. This is wasteful during inference — the KV cache stores one K,V pair per head per token.

```
Multi-Head Attention (MHA):            Grouped-Query Attention (GQA):
n_heads = 8, n_kv_heads = 8           n_heads = 8, n_kv_heads = 2

Q heads: Q₁ Q₂ Q₃ Q₄ Q₅ Q₆ Q₇ Q₈    Q heads: Q₁ Q₂ Q₃ Q₄ Q₅ Q₆ Q₇ Q₈
         │  │  │  │  │  │  │  │                 ╲╲ ╱╱  ╲╲ ╱╱  ╲╲ ╱╱  ╲╲ ╱╱
KV heads: K₁ K₂ K₃ K₄ K₅ K₆ K₇ K₈    KV heads:  K₁       K₂       (shared)
         V₁ V₂ V₃ V₄ V₅ V₆ V₇ V₈              V₁       V₂

KV cache: 8 pairs per token             KV cache: 2 pairs per token (4x smaller)


Multi-Query Attention (MQA):           Multi-head Latent Attention (MLA):
n_heads = 8, n_kv_heads = 1           DeepSeek-V2/V3

Q heads: Q₁ Q₂ Q₃ Q₄ Q₅ Q₆ Q₇ Q₈    Q heads: Q₁ Q₂ Q₃ ... Q₁₂₈
         ╲  ╲  ╲  │  ╱  ╱  ╱  ╱                 ╲     │     ╱
KV heads:       K₁                     Latent:    c_KV (compressed)
               V₁                                d_c = 512 (vs d_model = 7168)

KV cache: 1 pair per token (8x smaller) KV cache: just c_KV vector (14x smaller)
```

**KV cache reduction factors**:

```
Method    KV Heads (128-head model)  Cache Reduction  Quality Impact     Used By
────────  ─────────────────────────  ───────────────  ─────────────────  ─────────────────
MHA       128                        1x (baseline)    Baseline           GPT-2, GPT-3
GQA       8                          16x              Negligible         Llama 2/3, Mistral, Qwen
MQA       1                          128x             Small degradation  Falcon, PaLM, Gemini
MLA       N/A (latent)               ~14x             Negligible         DeepSeek-V2/V3/R1
```

GQA is the current standard. It was shown to need only 5% of original pretraining compute to uptrain from an MHA checkpoint, and quality is near-identical. Llama 2 70B uses 8 KV heads with 64 query heads — an 8x reduction in KV cache.

### Paged Attention (vLLM)

The problem: KV caches are allocated per-sequence, and sequence lengths vary. Traditional systems pre-allocate a contiguous memory block for the maximum possible sequence length. If you allocate for 2048 tokens but a sequence only uses 500, you waste 75% of that memory. Across many concurrent sequences, this waste adds up — vLLM's authors measured 60-80% memory waste in existing systems.

PagedAttention borrows virtual memory concepts from operating systems:

```
Traditional KV Cache Allocation:

Sequence A (actual: 500 tokens, allocated: 2048):
┌─────────────┬──────────────────────────────────────────────┐
│ ██████ Used │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ Wasted   │
│   500 tok   │              1548 tokens wasted               │
└─────────────┴──────────────────────────────────────────────┘

PagedAttention:

Physical memory blocks (each holds 16 tokens):
┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ...
│ B0 │ │ B1 │ │ B2 │ │ B3 │ │ B4 │ │ B5 │ │Free│
└────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘

Sequence A page table:    Sequence B page table:
  Logical → Physical        Logical → Physical
  Page 0  → Block 0         Page 0  → Block 2
  Page 1  → Block 4         Page 1  → Block 1
  Page 2  → Block 5         Page 2  → Block 3

- No contiguous allocation needed
- Blocks allocated on demand as sequence grows
- Waste only in last block of each sequence (< block_size tokens)
- Memory sharing: parallel samples from same prompt share blocks (copy-on-write)
```

**Impact**: vLLM with PagedAttention achieves up to 24x throughput over HuggingFace Transformers and 3.5x over Text Generation Inference. Memory waste drops to under 4%. This is why vLLM became the default serving framework.

## Quantization

Quantization reduces the numerical precision of model weights (and optionally activations) from FP16/BF16 to lower bit-widths. It's the single most impactful optimization for making large models fit on fewer GPUs.

### Why It Works

Neural network weights are normally stored in FP16 (16 bits per parameter). Most of this precision is unnecessary for inference. The key insight: weights have a relatively narrow distribution, and the model's output is robust to small perturbations.

**Numerical walkthrough — what quantization does to a weight**:

```
Original weight (FP16):  w = 0.0273437500
                         Binary: 0 01000 1100000000 (sign=0, exp=8, mantissa)

Quantize to INT8 (symmetric, scale = max(|w|)/127):
  If scale = 0.5/127 ≈ 0.003937:
  q = round(w / scale) = round(6.94) = 7
  w_dequant = 7 × 0.003937 = 0.027559
  Error: |0.027344 - 0.027559| = 0.000215 (0.79% relative error)

Quantize to INT4 (symmetric, scale = max(|w|)/7):
  If scale = 0.5/7 ≈ 0.07143:
  q = round(w / 0.07143) = round(0.383) = 0
  w_dequant = 0 × 0.07143 = 0.0
  Error: |0.027344 - 0.0| = 0.027344 (100% relative error!)
```

This shows why naive INT4 quantization destroys small weights. Methods like GPTQ, AWQ, and GGUF use group-wise quantization (separate scale per group of 32-128 weights) to keep the scale factor tight:

```
Group-wise INT4 (group_size = 128):
  Group of 128 weights: [0.027, 0.031, -0.025, 0.029, ...]
  Local max = 0.035, scale = 0.035/7 = 0.005
  q = round(0.027 / 0.005) = round(5.4) = 5
  w_dequant = 5 × 0.005 = 0.025
  Error: |0.027 - 0.025| = 0.002 (7.4% relative error)  ← much better
```

### PTQ vs QAT

```
Post-Training Quantization (PTQ):           Quantization-Aware Training (QAT):

Trained FP16 model                          Training with fake quantization
       │                                           │
       ▼                                           ▼
  Calibrate on                              Forward: quantize weights
  small dataset (128-1024 samples)          Backward: straight-through estimator
       │                                           │
       ▼                                           ▼
  Quantize weights                          Model learns to compensate
  (one-shot, minutes to hours)              for quantization noise
       │                                           │
       ▼                                           ▼
  Quantized model                           Quantized model
  (some quality loss)                       (minimal quality loss)

Pros: Fast, no training needed              Pros: Better quality at low bits
Cons: Quality degrades at ≤4 bits           Cons: Requires full training run
Used by: GPTQ, AWQ, bitsandbytes           Used by: BitNet, some NVIDIA FP8
```

For LLMs, **PTQ dominates** — QAT requires a full training run, which is prohibitive for 70B+ models. PTQ methods have gotten good enough that INT4 quality loss is minimal for most models.

### Major Quantization Methods

**GPTQ** (Frantar & Alistarh, 2022):
- One-shot, layer-wise quantization using approximate second-order (Hessian) information
- Quantizes each weight column while compensating for error in remaining columns (like a sequential least-squares)
- Needs a small calibration dataset (~128 samples from C4 or WikiText)
- Quantizes a 175B model in ~4 GPU-hours
- INT4 with group_size=128: negligible perplexity increase on most models
- 3.25x speedup on A100, 4.5x on A6000 vs FP16

**AWQ** (Lin et al., 2023):
- Observation: 1% of weights are "salient" — they correspond to large activations
- Instead of complex Hessian-based methods, AWQ simply identifies salient channels via activation magnitudes and scales them up before quantization (equivalent transformation that preserves output)
- No backpropagation or weight reconstruction needed — just a calibration pass
- Better generalization across domains than GPTQ (less overfitting to calibration data)
- 3x+ speedup over FP16, runs 70B Llama-2 on mobile GPUs
- Generally preferred over GPTQ in 2025-2026 for its robustness

**GGUF / llama.cpp** (Gerganov, 2023-2026):
- Format designed for CPU + GPU hybrid inference
- Implements many quantization variants: Q4_0, Q4_K_M, Q5_K_M, Q6_K, Q8_0, etc.
- "K-quants" use importance-based mixed quantization — more important layers get higher precision
- Runs on consumer hardware (MacBook, desktop CPUs) with optional GPU offloading
- The go-to for local inference on laptops and desktops

**bitsandbytes** (Dettmers et al., 2022-2023):
- LLM.int8(): 8-bit quantization with vector-wise scaling + outlier handling in FP16
- NF4 (4-bit NormalFloat): Quantization levels optimized for normally-distributed weights
- Key innovation: outlier features (large-magnitude activations in ~0.1% of hidden dimensions) are computed in FP16 while the rest uses INT8 — avoids the catastrophic quality loss that pure INT8 causes
- Enables QLoRA: 4-bit quantized base model + FP16 LoRA adapters for fine-tuning
- Primarily for HuggingFace/Transformers ecosystem; not the fastest for pure inference

**FP8** (NVIDIA, 2022-2024):
- Hardware-native on H100/H800 (FP8 Tensor Cores)
- Two variants: E4M3 (4-bit exponent, 3-bit mantissa — better range) and E5M2 (5-bit exponent, 2-bit mantissa — better dynamic range)
- E4M3 for weights, E5M2 for activations/gradients is the standard recipe
- Near-zero quality loss — FP8 is close enough to FP16 that most models don't degrade
- 2x throughput over FP16 on H100/H800 (doubles the effective Tensor Core throughput)
- The default for production serving on Hopper GPUs

### Accuracy vs Memory Tradeoff

Perplexity on WikiText-2 (lower is better), approximate values for Llama 2 70B:

```
Format     Bits  Memory (70B)  Perplexity  Quality       Throughput vs FP16
─────────  ────  ────────────  ──────────  ────────────  ─────────────────
FP16       16    140 GB        3.32        Baseline      1.0x
FP8        8     70 GB         3.33        ~Identical    ~2.0x
INT8       8     70 GB         3.34        ~Identical    ~1.8x
INT4-g128  4     35 GB         3.42        Slight loss   ~3.0x
INT4-g32   4     38 GB*        3.37        Minimal loss  ~2.5x
INT3       3     26 GB         3.88        Noticeable    ~3.5x
INT2       2     18 GB         6.20+       Significant   ~4.0x
```

`*` Smaller group size = more scale factors = slightly more memory but better quality.

**Practical guidance for your H800 (80 GB)**:

```
Model            FP16      FP8       INT4-g128  Fits on 1×H800?
───────────────  ────────  ────────  ─────────  ──────────────────────────
8B (Llama 3.1)   16 GB     8 GB      4 GB       ✓ FP16 with room for KV cache
70B (Llama 3.1)  140 GB    70 GB     35 GB      ✓ FP8 tight, INT4 comfortable
70B + 128K ctx   140+40GB  70+40GB   35+40GB    ✗ FP8, ✓ INT4 (75GB total)
405B (Llama 3.1) 810 GB    405 GB    203 GB     ✗ Need 3-6× H800 even at INT4
DeepSeek-V3      ~350 GB*  ~175 GB*  ~88 GB*    ✗ Need 2-3× H800 at INT4
```

`*` MoE — need all expert weights loaded even though only ~37B active per token.

## Flash Attention

### The Problem

Standard attention computes Q·Kᵀ → softmax → ·V, materializing the full N×N attention matrix in GPU HBM. For sequence length N=4096, that's a 4096×4096 = 16M element matrix per head. For N=128K, it's 16B elements per head — this doesn't fit in HBM at all.

Even when it fits, the bottleneck is memory access, not compute. The attention matrix is written to HBM, then read back for softmax, then written again, then read for the V multiply. Each read/write to HBM is slow.

```
Standard Attention Memory Access Pattern:

GPU SRAM (fast, small ~20MB)     GPU HBM (slow, large 80GB)
┌──────────┐                     ┌──────────────────────────┐
│          │ ─── read Q,K ────── │ Q [N×d], K [N×d]         │
│ compute  │                     │                          │
│ S = QKᵀ  │ ─── write S ──────► │ S [N×N]    ← materialized│
│          │                     │                          │
│          │ ─── read S ──────── │                          │
│ softmax  │                     │                          │
│ P = σ(S) │ ─── write P ──────► │ P [N×N]    ← materialized│
│          │                     │                          │
│          │ ─── read P,V ────── │ V [N×d]                  │
│ O = PV   │                     │                          │
│          │ ─── write O ──────► │ O [N×d]                  │
└──────────┘                     └──────────────────────────┘

HBM reads/writes: O(N² + Nd) — dominated by reading/writing the N×N matrices
```

### The Tiling Algorithm (Conceptual)

FlashAttention never materializes the N×N attention matrix. Instead, it processes attention in tiles that fit in SRAM:

```
FlashAttention Tiling:

GPU SRAM (fast, ~20MB)              GPU HBM (slow, 80GB)
┌───────────────────┐               ┌──────────────────────┐
│                   │◄── load Q tile │ Q [N×d]              │
│ Q_tile [Br×d]     │               │                      │
│ K_tile [Bc×d]     │◄── load K tile│ K [N×d]              │
│ V_tile [Bc×d]     │◄── load V tile│ V [N×d]              │
│                   │               │                      │
│ S_tile = Q_t·K_tᵀ │   (in SRAM)  │  No N×N matrix       │
│ P_tile = softmax   │   (in SRAM)  │  ever written to HBM │
│ O_tile += P_t·V_t  │   (in SRAM)  │                      │
│                   │               │                      │
│ Running softmax   │               │                      │
│ statistics (m, l) │               │                      │
│                   │───write O────►│ O [N×d]              │
└───────────────────┘               └──────────────────────┘

Loop: for each Q tile, iterate over all K,V tiles
  - Compute partial attention in SRAM
  - Maintain running softmax statistics (online softmax trick)
  - Accumulate output incrementally
  - Never write the N×N matrix to HBM

HBM reads/writes: O(N²d²/M) where M = SRAM size
```

The key insight is the **online softmax trick**: you can compute softmax incrementally without seeing all values first. As you process each K tile, you update running maximum and sum statistics, rescaling previous partial results. The final output is mathematically identical to standard attention.

### IO Complexity

```
                              HBM Reads/Writes    Extra Memory
Standard attention            O(Nd + N²)           O(N²)
FlashAttention                O(N²d²/M)            O(N)
                              ▲
                              │
                              For typical d=128, M=20MB:
                              N²d²/M ≈ N² × 16384 / 20M ≈ N² × 0.0008
                              vs N² for standard
                              → ~1000x fewer HBM accesses for the attention matrix
```

FlashAttention is **exact** — not an approximation. Same output, different computation order.

### Flash Attention 1 → 2 → 3 Evolution

```
Version  Year  Key Innovation                          Hardware    Speedup    Peak Util
───────  ────  ──────────────────────────────────────  ──────────  ─────────  ─────────
FA-1     2022  IO-aware tiling, online softmax         A100        2-4x       ~50%
FA-2     2023  Better parallelism, warp partitioning,  A100        2x over    50-73%
               reduce non-matmul FLOPs                              FA-1
FA-3     2024  Warp specialization, async TMA,         H100/H800   1.5-2x     ~75% FP16
               FP8 with block quantization                         over FA-2
```

FA-1 showed the approach works. FA-2 squeezed more out of the A100 by better utilizing parallelism across thread blocks. FA-3 exploits Hopper-specific hardware: the Tensor Memory Accelerator (TMA) handles data movement asynchronously while Tensor Cores compute, and warp specialization lets different warps handle different pipeline stages simultaneously. On H100, FA-3 achieves 740 TFLOPs/s in FP16 and nearly 1.2 PFLOPs/s in FP8.

**For your H800**: FlashAttention-3 is the default — it's integrated into PyTorch's `scaled_dot_product_attention` and every serving framework. You don't call it directly; vLLM/SGLang use it automatically.

## Speculative Decoding

### The Problem

Autoregressive decoding generates one token per forward pass. Each pass reads all model weights from HBM but does minimal computation (batch size 1 = one matrix-vector multiply per layer). The GPU is vastly underutilized — sitting idle while waiting for memory reads.

You can't generate multiple tokens in parallel because each token depends on the previous one. Or can you?

### Draft Model + Verification

The key insight: a small "draft" model can propose multiple tokens cheaply, and the large "target" model can verify all of them in a single parallel forward pass (like prefill, not decode).

```
Standard Decoding (target model only):

Step 1: [The]        → target model → "cat"          1 forward pass
Step 2: [The cat]    → target model → "sat"          1 forward pass
Step 3: [The cat sat]→ target model → "on"           1 forward pass
Step 4: [The cat sat on] → target → "the"            1 forward pass
                                              Total: 4 target forward passes

Speculative Decoding:

Step 1: Draft model generates K=4 tokens quickly:
        [The] → draft → "cat sat on the"             4 cheap draft passes

Step 2: Target model verifies ALL 4 in parallel:
        [The cat sat on the] → target → verify        1 target forward pass

        Token 1 "cat":  target P("cat") ≥ draft P("cat") → ACCEPT ✓
        Token 2 "sat":  target P("sat") ≥ draft P("sat") → ACCEPT ✓
        Token 3 "on":   target P("on")  ≥ draft P("on")  → ACCEPT ✓
        Token 4 "the":  target P("the") < draft P("the") → REJECT ✗
                         resample from adjusted distribution → "a"

        Result: "cat sat on a" — 3 accepted + 1 resampled = 4 tokens
                                              Total: 1 target forward pass
                                              Speedup: ~4x (if all accepted)
```

### Acceptance Rate and Speedup

The draft model proposes K tokens. The target model verifies them using a **modified rejection sampling** scheme that guarantees the output distribution is identical to the target model's distribution.

For each proposed token, the acceptance probability is:

```
P(accept token i) = min(1, p_target(x_i) / p_draft(x_i))
```

If the draft model closely matches the target, acceptance rate α is high. Expected speedup:

```
Expected tokens per step = (1 - αᴷ⁺¹) / (1 - α)

Where K = number of draft tokens, α = average acceptance rate

Example: K=5, α=0.8:
  Expected tokens = (1 - 0.8⁶) / (1 - 0.8) = (1 - 0.262) / 0.2 = 3.69 tokens/step

If draft model is 10x cheaper than target:
  Cost per step ≈ 1 target pass + K × 0.1 target passes = 1.5 target passes
  Effective speedup ≈ 3.69 / 1.5 ≈ 2.46x
```

**Critical property**: The output distribution is mathematically identical to the target model's. Speculative decoding is not an approximation — it's a pure speed optimization.

### Medusa

Medusa avoids the separate draft model entirely. It adds lightweight "Medusa heads" (single-layer FFN + softmax) on top of the target model's last hidden state. Each head predicts a different future token position:

```
Target Model Hidden State (h_t)
       │
       ├──► Original LM Head → next token (position t+1)
       ├──► Medusa Head 1    → prediction for position t+2
       ├──► Medusa Head 2    → prediction for position t+3
       ├──► Medusa Head 3    → prediction for position t+4
       └──► Medusa Head 4    → prediction for position t+5

Tree attention: construct a tree of candidate sequences from
top-k predictions of each head, verify all paths in one forward pass
```

- **Medusa-1**: Freeze base model, train only Medusa heads. 2.2x speedup.
- **Medusa-2**: Fine-tune base model jointly with Medusa heads. 2.3-3.6x speedup.
- **Advantage**: No separate draft model to maintain, minimal extra parameters.
- **Disadvantage**: Requires training the heads (a few hours of SFT-like training).

### EAGLE

EAGLE (Li et al., 2024) takes a different approach: instead of predicting tokens, it predicts **features** (the second-to-top-layer hidden states). The observation is that feature prediction is much easier and more accurate than token prediction, because features are continuous and smooth.

```
Standard speculative:  draft model → tokens → verify tokens
EAGLE:                 feature predictor → features → LM head → tokens → verify

Feature predictor:
  Input:  h_t (current features) + embed(x_{t+1}) (token one step ahead)
  Output: h_{t+1} (predicted next features)

  The predictor is a lightweight autoregressive model over features.
  Features are more predictable than tokens (lower entropy).
```

Results on LLaMA2-Chat 70B: 2.7-3.5x latency speedup, 2x throughput improvement. EAGLE-2 (2024) added a confidence-based dynamic draft length, adapting K based on how uncertain the feature predictor is.

### When to Use Speculative Decoding

```
Scenario                           Recommendation
─────────────────────────────────  ──────────────────────────────────
Single user, low latency needed    ✓ Speculative decoding helps most
High-throughput batch serving      △ Less benefit (batching already fills GPU)
Very long outputs (>1000 tokens)   ✓ Good — decode phase dominates
Short outputs (<50 tokens)         ✗ Overhead not worth it
Draft model available              ✓ Classic speculative decoding
No draft model                     ✓ Medusa or EAGLE (train heads)
Quality must be identical          ✓ All methods preserve target distribution
```

## Continuous Batching

### Static vs Continuous Batching

Traditional (static) batching groups N requests and processes them as a batch. All requests must start and finish together. If one request generates 10 tokens and another generates 500, the short request wastes GPU cycles waiting for the long one.

```
Static Batching:

Time ──────────────────────────────────────────────────────────►

Request A (50 tokens):  ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ← idle, waiting
Request B (200 tokens): ██████████████████████████████████████████
Request C (30 tokens):  ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ← idle, waiting

                        ▲ All start together  ▲ All finish together (at B's pace)
                        GPU utilization: low — A and C waste 75-85% of their slots


Continuous Batching:

Time ──────────────────────────────────────────────────────────►

Request A (50 tokens):  ██████████
Request B (200 tokens): ██████████████████████████████████████████
Request C (30 tokens):  ██████
Request D:                        ██████████████████
Request E:                              ████████████████████

                        ▲ Requests enter and leave independently
                        ▲ New requests fill freed slots immediately
                        GPU utilization: high — slots always occupied
```

### How vLLM Implements It

vLLM's continuous batching (inspired by Orca, Yu et al., 2022) works at the **iteration level** — after each decode step, the scheduler can:

1. **Evict**: Remove finished sequences, freeing their KV cache blocks
2. **Admit**: Add waiting requests, allocating KV cache blocks from the free pool
3. **Preempt**: Swap out lower-priority sequences (page their KV cache to CPU) if memory is tight

Combined with PagedAttention, this means:
- No memory pre-allocation per max sequence length
- KV cache blocks are allocated and freed dynamically
- Memory utilization stays near 100%
- New requests don't have to wait for the entire batch to finish

**Chunked prefill**: For long prompts, vLLM splits the prefill into chunks and interleaves them with decode steps from other requests, preventing one long prompt from blocking all decode iterations.

## Tensor Parallelism vs Pipeline Parallelism

When a model doesn't fit on one GPU, you split it across multiple GPUs. Two main approaches:

```
Tensor Parallelism (TP):                  Pipeline Parallelism (PP):
Split each layer across GPUs              Split layers across GPUs

GPU 0: [Half of Layer 1-80]              GPU 0: [Layers 1-20]
GPU 1: [Half of Layer 1-80]              GPU 1: [Layers 21-40]
GPU 2: [Half of Layer 1-80]              GPU 2: [Layers 41-60]
GPU 3: [Half of Layer 1-80]              GPU 3: [Layers 61-80]

Each layer:                               Each layer:
  MatMul split across GPUs                  Full MatMul on one GPU
  AllReduce after each layer                Send activations to next GPU

Communication: AllReduce per layer        Communication: P2P send per stage
  (high bandwidth, low latency needed)      (moderate bandwidth)

Latency: ~same as 1 GPU (parallel)       Latency: higher (sequential stages)
Throughput: scales well                   Throughput: scales well with microbatching
Best for: inference (latency matters)     Best for: training, very large models
Requires: fast interconnect (NVLink)      Works with: slower interconnect (PCIe OK)
```

### Practical Serving Configurations

```
Model              Total Mem (FP16)  Recommended Setup on H800 80GB
─────────────────  ────────────────  ────────────────────────────────────────
8B (Llama 3.1)     16 GB             1× H800, FP8 or FP16
70B (Llama 3.1)    140 GB            2× H800 TP=2 (FP16) or 1× H800 (INT4)
70B + 128K ctx     180 GB            4× H800 TP=4 (FP8) — KV cache needs room
405B (Llama 3.1)   810 GB            8× H800 TP=8 (FP8) or 16× TP=8 PP=2
DeepSeek-V3        ~350 GB           4-8× H800 with expert parallelism (EP)
```

**Expert Parallelism (EP)**: For MoE models like DeepSeek-V3, experts are distributed across GPUs. Each GPU holds a subset of experts, and tokens are routed to the appropriate GPU. This is more communication-efficient than TP for MoE because only the routed expert's activations are transferred.

**Combined parallelism**: Production systems often combine TP + PP + EP. For example, DeepSeek-V3 serving might use TP=4 within a node (NVLink) and EP across nodes (InfiniBand/RoCE).

## Practical Considerations

### Decision Flowchart

```
Start: You have a model to serve
       │
       ▼
Does the model fit on 1 GPU (with KV cache room)?
       │
  ┌────┴────┐
  │ Yes     │ No
  │         │
  ▼         ▼
Use FP8     How many GPUs?
or INT4     │
if tight    ├─ 2-8 GPUs: Tensor Parallelism
            ├─ 8-32 GPUs: TP + PP
            └─ MoE model: TP + Expert Parallelism
       │
       ▼
Choose serving framework
       │
       ├─ NVIDIA GPUs, max throughput: TensorRT-LLM or SGLang
       ├─ Flexibility, easy setup: vLLM
       ├─ CPU/laptop/edge: llama.cpp (GGUF)
       └─ Structured output, complex pipelines: SGLang
       │
       ▼
Choose quantization
       │
       ├─ H800/H100: FP8 (near-zero quality loss, 2x speedup)
       ├─ Need to fit larger model: INT4 (AWQ or GPTQ)
       ├─ Consumer GPU / laptop: GGUF Q4_K_M or Q5_K_M
       └─ Fine-tuning: bitsandbytes NF4 + QLoRA
       │
       ▼
Enable speculative decoding?
       │
       ├─ Latency-critical, single-user: Yes (EAGLE or Medusa)
       ├─ High-throughput serving: Usually no (batching helps more)
       └─ No draft model available: Medusa heads or n-gram matching
```

### Memory Budget for Common Models on 1× H800 80GB

```
Component               8B FP16   8B FP8    70B INT4   70B FP8
──────────────────────  ────────  ────────  ─────────  ────────
Model weights           16 GB     8 GB      35 GB      70 GB
KV cache (4K ctx)       0.26 GB   0.13 GB   1.25 GB    0.63 GB
KV cache (32K ctx)      2 GB      1 GB      10 GB      5 GB
KV cache (128K ctx)     8 GB      4 GB      40 GB      20 GB
Activations + overhead  2 GB      2 GB      3 GB       3 GB
──────────────────────  ────────  ────────  ─────────  ────────
Total (4K ctx)          18 GB     10 GB     39 GB      74 GB
Total (32K ctx)         20 GB     11 GB     48 GB      78 GB
Total (128K ctx)        26 GB     14 GB     78 GB      93 GB ✗
Available               80 GB     80 GB     80 GB      80 GB
Headroom (32K)          60 GB     69 GB     32 GB      2 GB

Max concurrent seqs     ~30       ~63       ~3         ~1
(32K ctx, filling       (60GB ÷   (69GB ÷   (32GB ÷    (2GB ÷
 remaining memory)       2GB)      1.1GB)    10GB)      5GB)
```

This shows why quantization matters for serving economics. The 8B model at FP8 can serve ~63 concurrent 32K-context sequences on your H800, while the 70B at FP8 can barely serve 1.

## Serving Framework Comparison

```
Framework       Engine      Quantization        Speculative     Key Strengths
──────────────  ──────────  ──────────────────  ──────────────  ──────────────────────
vLLM            Python/C++  FP8, INT4 (AWQ,     EAGLE, MTP,     Easy setup, huge model
(v0.7+)         CUDA        GPTQ), GGUF,        n-gram          support, PagedAttention,
                            bitsandbytes                        de facto standard

TensorRT-LLM   C++/Python  FP4, FP8, INT4,     EAGLE, MTP,     Highest throughput on
(NVIDIA)        TensorRT    INT8 (all native)   n-gram          NVIDIA, best Hopper/
                                                                Blackwell utilization

SGLang          Python/C++  FP4, FP8, INT4,     Yes             RadixAttention (prefix
(v0.5.9)        CUDA        AWQ, GPTQ                           caching), structured
                                                                output, lowest overhead

llama.cpp       C/C++       GGUF: Q2-Q8,        Yes (draft      CPU+GPU hybrid, runs
                Metal/CUDA  K-quants, IQ         model)          anywhere, easiest local
                            (imatrix quants)                     deployment
```

### Performance Characteristics (approximate, 2025-2026 benchmarks)

```
Metric                    vLLM      TRT-LLM   SGLang    llama.cpp
────────────────────────  ────────  ────────  ────────  ──────────
Throughput (tok/s, 70B)   High      Highest   High      Low-Med
TTFT latency              Good      Best      Good      N/A*
Ease of setup             Easy      Moderate  Easy      Easiest
Model support breadth     Widest    Wide      Wide      Wide
Multi-GPU                 TP/PP/EP  TP/PP/EP  TP/PP/EP  Limited
Structured output         Basic     Basic     Best      Basic
Prefix caching            Yes       Yes       Best      No
Community/ecosystem       Largest   NVIDIA    Growing   Largest OSS
Production deployments    Most      Enterprise Growing   Local/edge
```

`*` llama.cpp is primarily for single-user local inference, not server-style TTFT measurement.

**Recommendation for your H800**:
- **Default choice**: vLLM — largest community, easiest setup, excellent performance
- **Max throughput**: TensorRT-LLM or SGLang — if you need every last token/sec
- **Complex pipelines**: SGLang — if you need structured output, prefix caching, or multi-step generation
- **Experimentation**: All three Python-based frameworks work well; switch between them easily since they share the HuggingFace model format

### Recent Developments (2025-2026)

- **FP4 quantization**: Supported by SGLang and TensorRT-LLM on Blackwell (B200/B300) GPUs. ~4x memory reduction from FP16 with surprisingly minimal quality loss on large models.
- **Disaggregated serving**: Separate prefill and decode onto different GPU pools (prefill is compute-bound, decode is memory-bound). TensorRT-LLM and SGLang both support this in beta.
- **Expert parallelism**: Critical for MoE models. SGLang reports 3.8x prefill and 4.8x decode throughput on GB200 with EP for DeepSeek-V3-class models.
- **Prefill-decode disaggregation**: SGLang and TensorRT-LLM support splitting prefill (compute-bound) and decode (memory-bound) onto different GPU types/pools for optimal resource utilization.

## Key Papers

### KV Cache and Attention

| Paper | Year | Key Contribution | Link |
|-------|------|-----------------|------|
| Fast Transformer Decoding: One Write-Head is All You Need | 2019 | Multi-Query Attention (MQA) — shared KV heads | [arxiv 1911.02150](https://arxiv.org/abs/1911.02150) |
| GQA: Training Generalized Multi-Query Attention Models | 2023 | Grouped-Query Attention — middle ground MHA↔MQA | [arxiv 2305.13245](https://arxiv.org/abs/2305.13245) |
| Efficient Memory Management for LLM Serving with PagedAttention | 2023 | Virtual-memory-inspired KV cache, vLLM | [arxiv 2309.06180](https://arxiv.org/abs/2309.06180) |
| DeepSeek-V2: A Strong, Economical, and Efficient MoE LM | 2024 | Multi-head Latent Attention (MLA) for compressed KV | [arxiv 2405.04434](https://arxiv.org/abs/2405.04434) |

### Flash Attention

| Paper | Year | Key Contribution | Link |
|-------|------|-----------------|------|
| FlashAttention: Fast and Memory-Efficient Exact Attention | 2022 | IO-aware tiling, online softmax | [arxiv 2205.14135](https://arxiv.org/abs/2205.14135) |
| FlashAttention-2: Faster Attention with Better Parallelism | 2023 | 2x over FA-1, 50-73% A100 utilization | [arxiv 2307.08691](https://arxiv.org/abs/2307.08691) |
| FlashAttention-3: Fast and Accurate Attention with Asynchrony | 2024 | Hopper features, FP8, 740 TFLOPs/s on H100 | [arxiv 2407.08608](https://arxiv.org/abs/2407.08608) |

### Quantization

| Paper | Year | Key Contribution | Link |
|-------|------|-----------------|------|
| LLM.int8(): 8-bit Matrix Multiplication for Transformers | 2022 | Outlier-aware INT8, mixed-precision decomposition | [arxiv 2208.07339](https://arxiv.org/abs/2208.07339) |
| GPTQ: Accurate Post-Training Quantization for GPTs | 2022 | One-shot INT4 via approximate Hessian, 175B on 1 GPU | [arxiv 2210.17323](https://arxiv.org/abs/2210.17323) |
| QLoRA: Efficient Finetuning of Quantized LLMs | 2023 | NF4 quantization + LoRA for 4-bit fine-tuning | [arxiv 2305.14314](https://arxiv.org/abs/2305.14314) |
| AWQ: Activation-Aware Weight Quantization | 2023 | Protect salient 1% via activation statistics, MLSys Best Paper | [arxiv 2306.00978](https://arxiv.org/abs/2306.00978) |
| FP8 Formats for Deep Learning | 2022 | E4M3/E5M2 formats, hardware-native on Hopper | [arxiv 2209.05433](https://arxiv.org/abs/2209.05433) |

### Speculative Decoding

| Paper | Year | Key Contribution | Link |
|-------|------|-----------------|------|
| Fast Inference from Transformers via Speculative Decoding | 2022 | Draft + verify with identical output distribution | [arxiv 2211.17192](https://arxiv.org/abs/2211.17192) |
| Accelerating LLM Inference with Staged Speculative Decoding | 2023 | Multi-stage draft models | [arxiv 2308.04623](https://arxiv.org/abs/2308.04623) |
| Medusa: Simple LLM Inference Acceleration | 2024 | Multiple decoding heads, tree-based verification | [arxiv 2401.10774](https://arxiv.org/abs/2401.10774) |
| EAGLE: Speculative Sampling via Feature-Level Autoregression | 2024 | Feature prediction instead of token prediction, 2.7-3.5x | [arxiv 2401.15077](https://arxiv.org/abs/2401.15077) |

### Serving Systems

| Paper / System | Year | Key Contribution | Link |
|-------|------|-----------------|------|
| Orca: A Distributed Serving System for Transformer-Based Models | 2022 | Iteration-level scheduling (continuous batching) | [OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/yu) |
| vLLM / PagedAttention | 2023 | Paged KV cache, de facto serving standard | [arxiv 2309.06180](https://arxiv.org/abs/2309.06180) |
| SGLang: Efficient Execution of Structured LM Programs | 2024 | RadixAttention, prefix caching, structured output | [arxiv 2312.07104](https://arxiv.org/abs/2312.07104) |
| TensorRT-LLM | 2023-2026 | NVIDIA's optimized engine, FP4/FP8, max NVIDIA perf | [GitHub](https://github.com/NVIDIA/TensorRT-LLM) |
| llama.cpp | 2023-2026 | CPU+GPU hybrid, GGUF format, runs anywhere | [GitHub](https://github.com/ggerganov/llama.cpp) |
