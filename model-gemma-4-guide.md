# Gemma 4: Google's Most Capable Open Models

How Google DeepMind packed Gemini 3-level intelligence into four open-weight models — from a 2B edge model that runs on a Raspberry Pi to a 31B dense model that rivals frontier reasoning systems — and why the architecture choices (dual-config attention, K=V sharing, shared KV cache, Per-Layer Embeddings) make this possible.

## Background

**Source**: [Gemma 4 Model Card](https://ai.google.dev/gemma/docs/core/model_card_4) (Google DeepMind, April 2, 2026). No standalone technical report published yet — architecture details sourced from the model card, HuggingFace blog, and community analysis.

**Research lineage** — Gemma 4 is the latest in Google's open-weight model family:

1. **Gemma 1** (Google DeepMind, 2024, [arxiv 2403.08295](https://arxiv.org/abs/2403.08295)) — First Gemma release: 2B and 7B dense models derived from Gemini 1 research. Text-only, custom license. Proved Google could distill competitive open-weight models from their proprietary line.

2. **Gemma 2** (Google DeepMind, 2024) — Scaled to 9B and 27B. Introduced sliding window attention (alternating local/global layers) and logit soft-capping. Still text-only.

3. **PaliGemma / PaliGemma 2** (Google DeepMind, 2024, [arxiv 2407.07726](https://arxiv.org/abs/2407.07726), [arxiv 2412.03555](https://arxiv.org/abs/2412.03555)) — Vision-language models built on Gemma backbones with SigLIP vision encoders. Specialized for visual understanding tasks rather than general-purpose chat.

4. **Gemma 3** (Google DeepMind, 2025, [arxiv 2503.19786](https://arxiv.org/abs/2503.19786)) — Added native vision (image + video), expanded to 1B/4B/12B/27B sizes, 128K context. First multimodal Gemma. Still used a custom "Gemma Terms of Use" license.

5. **Gemma 4** (Google DeepMind, April 2, 2026) — Four models (E2B, E4B, 26B-A4B, 31B) built from Gemini 3 research. Adds audio input, MoE variant, 256K context, and a stack of architectural innovations. First Gemma under **Apache 2.0** — no usage restrictions, no MAU limits, full commercial freedom.

**The arc**: Text-only dense → text-only with sliding window → vision-language specialized → multimodal general-purpose → multimodal + audio + MoE + Apache 2.0. Each generation inherits from the latest Gemini research while pushing the efficiency frontier for open models.

## Model Family Overview

Gemma 4 ships four models targeting two deployment tiers:

```
Edge Models (on-device)              Server Models (cloud/workstation)
┌─────────────────────────┐          ┌──────────────────────────────┐
│  E2B         E4B        │          │  26B-A4B         31B         │
│  5.1B total  8B total   │          │  25.2B total     30.7B total │
│  2.3B eff    4.5B eff   │          │  3.8B active     30.7B dense │
│  128K ctx    128K ctx   │          │  256K ctx        256K ctx    │
│  Text+Img+Audio         │          │  Text+Img+Video              │
│  Dense + PLE            │          │  MoE (128 exp)   Dense       │
└─────────────────────────┘          └──────────────────────────────┘
```

| Model | Type | Total Params | Active Params | Layers | Context | Sliding Window | Audio |
|-------|------|-------------|---------------|--------|---------|---------------|-------|
| E2B | Dense + PLE | 5.1B | 2.3B effective | 35 (28 sliding + 7 full) | 128K | 512 | Yes |
| E4B | Dense + PLE | 8B | 4.5B effective | 42 | 128K | 512 | Yes |
| 26B-A4B | MoE | 25.2B | 3.8B active | 30 (25 sliding + 5 full) | 256K | 1024 | No |
| 31B | Dense | 30.7B | 30.7B | 60 (50 sliding + 10 full) | 256K | 1024 | No |

Naming conventions:
- **E** = "Effective" parameters — the gap between total and effective comes from PLE embedding tables (large but lookup-only)
- **A4B** = "Active 4 Billion" — MoE routing activates only 3.8B of 25.2B total per token
- All models share a 262K token vocabulary
- All come in base (pretrained) and instruction-tuned (IT) variants

## Key Terms

**Per-Layer Embeddings (PLE)**: A parallel conditioning pathway used in E2B/E4B. Instead of one embedding per token shared across all layers, PLE produces a dedicated vector for each decoder layer via lightweight residual blocks. Two signal components per token: (1) a token-identity embedding lookup, and (2) a context-aware learned projection. The large PLE embedding tables explain the total vs effective parameter gap (5.1B total but only 2.3B effective for E2B).

**Dual-Config Attention**: Gemma 4's approach of using different head dimensions and KV head counts for sliding-window vs full-attention layers. Sliding layers use 256-dim heads with more KV heads; full layers use 512-dim heads with fewer KV heads. A departure from Gemma 3 which used identical configs for all layers.

**K=V Weight Sharing**: In full-attention layers, the value projection is eliminated entirely — the key tensor is cloned as the value before normalization, then K and V diverge through separate RMSNorm paths. Saves parameters and memory with minimal quality impact.

**Proportional RoPE (p-RoPE)**: Full-attention layers apply RoPE to only 25% of head dimensions (theta=1M), leaving 75% position-independent. This enables robust long-context extrapolation to 256K tokens without the quality degradation that standard RoPE suffers at extreme distances.

**Shared KV Cache**: The last N layers of each attention type (sliding/full) reuse K/V tensors from the last non-shared layer of the same type. Eliminates redundant KV projections, reducing memory footprint and compute for long-context inference.

**Logit Soft-Capping**: Final logits bounded via `tanh(x/30) * 30`, preventing extreme output values during generation. Inherited from Gemma 2.

## Architecture Deep Dive

### Hybrid Sliding Window + Full Attention

Like Gemma 2 and 3, Gemma 4 alternates between local sliding-window attention and full global attention layers. But Gemma 4 introduces **dual-config attention** — the two layer types now have fundamentally different configurations:

```
Sliding Window Layer (local)         Full Attention Layer (global)
┌────────────────────────────┐       ┌────────────────────────────┐
│ Head dim: 256              │       │ Head dim: 512              │
│ More KV heads              │       │ Fewer KV heads             │
│ Standard RoPE              │       │ p-RoPE (25% of dims)       │
│ Window: 512-1024 tokens    │       │ Attends to full context    │
│ Standard K, V projections  │       │ K=V sharing (V cloned      │
│                            │       │   from K, separate norms)  │
└────────────────────────────┘       └────────────────────────────┘
```

**Why two configs?** The sliding window layers handle local context — recent tokens, syntactic structure, immediate dependencies. They need fine-grained attention (more KV heads, smaller head dim) over a narrow window. The full attention layers handle global context — long-range dependencies, document-level coherence, cross-section reasoning. They benefit from larger head dimensions for richer per-head representations, and p-RoPE for position-invariant long-range retrieval.

### Detailed Architecture: 31B Dense

```
Component                     31B Dense
──────────────────────       ──────────
Hidden size                   5,376
Layers                        60 (50 sliding + 10 full)
Q heads                       32
KV heads (sliding)            16
KV heads (full)               4
Head dim (sliding)            256
Head dim (full)               512
FFN hidden dim                21,504 (GeGLU)
Vocabulary                    262K
Context window                256K
Sliding window size           1,024
```

### Detailed Architecture: 26B-A4B MoE

```
Component                     26B-A4B MoE
──────────────────────       ──────────
Hidden size                   2,816
Layers                        30 (25 sliding + 5 full)
Q heads                       16
KV heads (sliding)            8
KV heads (full)               2
Head dim (sliding)            256
Head dim (full)               512
Dense FFN hidden              2,112
Total experts                 128 + 1 shared
Active experts/token          8 + 1 shared = 9
Expert hidden dim             704 each
Total params                  25.2B
Active params/token           3.8B
Context window                256K
Sliding window size           1,024
```

The MoE architecture is worth examining. Each MoE layer runs a dense GeGLU FFN **in parallel** with the routed expert block. The outputs are summed and scaled by `1/√2`:

```
Token hidden state (h)
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐  ┌──────────────────────────┐
│Dense   │  │ Router                    │
│GeGLU   │  │   │                       │
│FFN     │  │   ├─► Expert 17  ─┐       │
│(2112)  │  │   ├─► Expert 42  ─┤       │
│        │  │   ├─► Expert 91  ─┤       │
│        │  │   ├─► Expert 3   ─┤ Sum   │
│        │  │   ├─► Expert 120 ─┤───►   │
│        │  │   ├─► Expert 7   ─┤       │
│        │  │   ├─► Expert 55  ─┤       │
│        │  │   └─► Expert 88  ─┘       │
│        │  │   + Shared Expert         │
└───┬────┘  └────────────┬──────────────┘
    │                    │
    └───────┬────────────┘
            │ × 1/√2
            ▼
       Output (h')
```

This parallel dense+MoE design (similar to DeepSeek-V2/V3's shared expert concept) ensures every token gets a baseline FFN computation regardless of routing decisions. The 128 routed experts provide specialization on top.

With 3.8B active parameters out of 25.2B total, the 26B-A4B achieves ~97% of the dense 31B's quality at roughly 8x less compute per token.

### K=V Weight Sharing: Numerical Walkthrough

In standard attention, a token's hidden state `h` is projected into Q, K, V separately:

```
Standard:           K=V Sharing (Gemma 4 full layers):

Q = h @ W_q         Q = h @ W_q
K = h @ W_k         K = h @ W_k
V = h @ W_v         V_raw = K.clone()      ← no W_v needed
                     K = RMSNorm_k(K)
                     V = RMSNorm_v(V_raw)   ← separate norm
```

For the 31B model's full attention layers with head_dim=512 and hidden_size=5376:
- W_v would be 5376 × 2048 = 11M parameters per layer (4 KV heads × 512 dim)
- K=V sharing eliminates W_v entirely
- Across 10 full attention layers: saves ~110M parameters

The separate RMSNorm paths allow K and V to diverge after cloning — K gets normalized for attention score computation, V gets normalized differently for value aggregation. The normalization parameters are tiny (just scale vectors), so the savings are nearly the full W_v weight.

### Per-Layer Embeddings (PLE)

PLE is used in E2B and E4B to give each decoder layer its own view of the input token:

```
Standard Embedding:                PLE (Gemma 4 E2B/E4B):

Token → Embedding table            Token → Main embedding (shared)
         │                                   │
         │ (same vector                      │
         │  for all layers)          Token → PLE table ──┐
         │                                   │           │
         ▼                                   │    ┌──────┴──────┐
    All decoder layers                       │    │ Per-layer    │
                                             │    │ residual     │
                                             │    │ blocks       │
                                             │    └──┬──┬──┬──┬─┘
                                             │       │  │  │  │
                                             ▼       ▼  ▼  ▼  ▼
                                        Layer 1  Layer 2 ... Layer N
                                        gets      gets
                                        its own   its own
                                        PLE vec   PLE vec
```

Each PLE vector combines: (1) a **token-identity** signal from a large embedding lookup (explains the total vs effective param gap), and (2) a **context-aware** projection learned to condition each layer differently.

Why PLE matters for edge models: at 2-4B effective parameters, every layer needs to punch above its weight. PLE lets each layer specialize its understanding of the same token — early layers might focus on syntax, later layers on semantics — without increasing the core hidden dimension.

### Proportional RoPE (p-RoPE) for Long Context

Standard RoPE applies positional encoding to all head dimensions. At very long distances (>100K tokens), the high-frequency components create noise that degrades attention quality.

Gemma 4's solution for full-attention layers:

```
Head dim = 512 (full attention layers)

Standard RoPE:     All 512 dims get positional encoding
                   ├──── RoPE ────────────────────────┤

p-RoPE (Gemma 4):  25% get RoPE          75% position-free
                   ├── RoPE ──┤├── no position info ──┤
                    128 dims        384 dims

theta = 1,000,000 (vs typical 10,000-500,000)
```

The 75% position-free dimensions can match content purely by semantic similarity regardless of distance — a token at position 1,000 attends as easily to position 200,000 as to position 1,001. The 25% with RoPE (at a very high theta=1M) provide enough positional signal for ordering and local structure.

This is what enables reliable 256K context. The MRCR v2 8-needle benchmark (which tests multi-hop retrieval across the full context) shows the impact:

```
Model              MRCR v2 8-needle 128K
────────────       ─────────────────────
Gemma 4 31B        66.4%
Gemma 3 27B        13.5%     ← 4.9x improvement
```

### Vision Encoder

All four models include a vision encoder for image and video input:
- E2B/E4B: ~150M parameter vision encoder
- 26B/31B: ~550M parameter vision encoder

Key innovation over Gemma 3: **configurable token budgets per image**. Instead of a fixed resolution, Gemma 4 uses a learned 2D position encoder with multidimensional RoPE that preserves original aspect ratios:

```
Token Budget    Use Case                  Tokens/Image
──────────     ────────────────────       ────────────
70             Fast OCR, thumbnails       70
140            Basic understanding        140
280            Standard (default)         280
560            High detail                560
1120           Maximum detail             1120
```

This is a practical improvement. A 256K context window at 1120 tokens/image can handle ~230 images; at 70 tokens/image, over 3600. For agentic workflows processing many screenshots or document pages, the lower budgets keep context manageable.

Video is processed as image frames at 1 fps (max 60 seconds).

### Audio Encoder (E2B/E4B Only)

The edge models include a USM-style conformer audio encoder:
- ~300-305M parameters (compressed from 681M in Gemma 3n)
- Frame duration: 40ms (down from 160ms in Gemma 3n)
- Max audio length: 30 seconds
- Supports: ASR, speech translation, audio understanding

Audio is only available on E2B and E4B — the server models (26B-A4B, 31B) do not include audio processing.

## Benchmark Results

### Reasoning and Knowledge (Instruction-Tuned)

```
Benchmark            E2B      E4B      26B-A4B    31B      Gemma 3 27B
────────────        ─────    ─────    ───────    ─────    ───────────
MMLU Pro            60.0%    69.4%     82.6%     85.2%      67.6%
AIME 2026           37.5%    42.5%     88.3%     89.2%      20.8%
GPQA Diamond        43.4%    58.6%     82.3%     84.3%      42.4%
BigBench Extra Hard 21.9%    33.1%     64.8%     74.4%      19.3%
MMMLU (multilingual)67.4%    76.6%     86.3%     88.4%      70.7%
Tau2                24.5%    42.2%     68.2%     76.9%      16.2%
```

The AIME jump is the headline number: 31B at 89.2% vs Gemma 3 27B at 20.8% — a **4.3x improvement** on competition math. This puts Gemma 4 31B in the same tier as DeepSeek R1 on mathematical reasoning, despite being a fraction of the size.

### Coding

```
Benchmark            E2B     E4B      26B-A4B    31B      Gemma 3 27B
────────────        ─────   ─────    ───────    ─────    ───────────
LiveCodeBench v6    44.0%   52.0%     77.1%     80.0%      29.1%
Codeforces ELO       633     940      1718      2150        110
```

Codeforces ELO of 2150 for the 31B is exceptional — that's roughly "Candidate Master" level, competitive with much larger models. Gemma 3 27B's ELO of 110 was essentially random.

### Vision

```
Benchmark            E2B     E4B      26B-A4B    31B      Gemma 3 27B
────────────        ─────   ─────    ───────    ─────    ───────────
MMMU Pro            44.2%   52.6%     73.8%     76.9%      49.7%
MATH-Vision         52.4%   59.5%     82.4%     85.6%      46.0%
OmniDocBench 1.5    0.290   0.181     0.149     0.131      0.365
  (lower = better)
MedXPertQA MM       23.5%   28.7%     58.1%     61.3%       —
```

### Audio (E Models Only)

```
Benchmark         E2B       E4B
────────────     ─────     ─────
CoVoST (BLEU)   33.47     35.54
FLEURS (WER)      0.09      0.08
  (lower = better)
```

### Long Context

```
Model              MRCR v2 8-needle 128K
────────────       ─────────────────────
31B                66.4%
26B-A4B            44.1%
E4B                25.4%
E2B                19.1%
Gemma 3 27B        13.5%
```

### Arena Leaderboard (April 2026)

```
Model              Arena Score    Rank (Open Models)
────────────       ──────────    ──────────────────
31B                ~1452         #3
26B-A4B            ~1441         #6
```

The 31B ranks behind GPT-OSS-120B (OpenAI) and trails slightly behind Qwen 3.5 and GLM-5 on the overall Arena leaderboard.

## Gemma 4 vs Competitors

```
                    Gemma 4     Qwen 3.5    Llama 4      DeepSeek
                    31B         27B         Scout        V3 (671B)
────────────       ─────       ────────    ──────       ──────────
Total params        30.7B       ~27B        109B (MoE)   671B (MoE)
Active params       30.7B       ~27B        ~17B         37B
Context             256K        128K        10M          128K
MMLU Pro            85.2%       ~80%        ~79%         —
AIME 2026           89.2%       ~49%        —            —
GPQA Diamond        84.3%       —           74.3%        —
LiveCodeBench       80.0%       ~43%        —            —
License             Apache 2.0  Apache 2.0  Community    MIT-ish
Modalities          T+I+V       T+I         T+I          T
Audio               No*         No          No           No
On-device variant   E2B/E4B     —           —            —

*Audio only on E2B/E4B edge models
```

Gemma 4's competitive advantages: strongest reasoning per parameter in its weight class, Apache 2.0 licensing, and a complete edge-to-server lineup. Its weaknesses: cannot match very large MoE models (DeepSeek V3, GPT-OSS-120B) on absolute capability, and the 256K context is modest compared to Llama 4 Scout's 10M.

## Gemma 3 → Gemma 4: What Changed

```
Feature                  Gemma 3                    Gemma 4
──────────────          ─────────────────────      ─────────────────────────
License                  Custom Gemma Terms         Apache 2.0
Model sizes              1B, 4B, 12B, 27B           E2B, E4B, 26B-A4B, 31B
MoE variant              None                       26B-A4B (128 experts)
Context window           128K (all)                 128K (edge) / 256K (server)
Attention config         Same head_dim all layers   Dual-config (256/512)
KV sharing (K=V)         No                         Yes (full attention layers)
Value normalization      No                         RMSNorm on values
Per-Layer Embeddings     No                         Yes (E2B/E4B)
Shared KV cache          No                         Yes (last N layers)
Vision encoder           Fixed 896×896 + Pan&Scan   Variable AR, token budgets
Audio                    Gemma 3n only (681M enc)   E2B/E4B (305M, compressed)
Sliding window pattern   5 local + 1 global         Ratio varies by model
Training data cutoff     August 2024                January 2025
Source model             Gemini 2                   Gemini 3
```

The biggest architectural shift is the dual-config attention with K=V sharing and p-RoPE. Gemma 3 used the same attention configuration everywhere. Gemma 4 recognizes that local and global attention have fundamentally different jobs and configures them accordingly.

## Training Details

Official disclosure is limited. What's known:

- **Data**: Web documents (140+ languages), code, mathematics, images, audio
- **Training cutoff**: January 2025
- **Source**: Built from Gemini 3 research — likely uses knowledge distillation from Gemini teacher models (consistent with Gemma 3's approach)
- **Post-training**: SFT with human-labeled prompt-response pairs + on-policy distillation from larger IT teachers; RLHF with composite reward functions (human feedback, code execution, math ground-truth); advanced policy optimization (BOND, WARM, WARP — same techniques used in Gemma 3)
- **Preprocessing**: CSAM filtering, sensitive data removal, content quality filtering
- **Multilingual**: 35+ languages in instruction tuning, 140+ languages in pretraining

### Thinking Mode

Gemma 4 supports configurable extended thinking (up to 4000 tokens) via a `<|think|>` tag in the system prompt:

```
System: You should think about the problem before answering. <|think|>

User: What is the integral of x² sin(x)?

Model: <|think|>
I need to use integration by parts twice...
[reasoning chain up to 4000 tokens]
</|think|>

The integral of x² sin(x) is ...
```

For multi-turn conversations, Google recommends stripping thought blocks from conversation history to save context.

### Function Calling

Native JSON tool-use support is trained from the ground up. The model can plan multi-step tool chains, handle parallel function calls, and process multimodal inputs (e.g., analyzing an image then calling an API based on what it sees).

## Practical Considerations

### Memory Requirements

```
Model        4-bit      8-bit      BF16
────────    ──────     ──────     ──────
E2B          ~5 GB       —        ~10 GB
E4B          ~5 GB       —        ~15 GB
26B-A4B     ~18 GB     ~28 GB     ~50 GB
31B         ~20 GB     ~34 GB     ~62 GB
```

The 31B fits on a single H100 80GB in BF16. Your H800 handles it comfortably. For quantized deployment, the 26B-A4B at 4-bit (~18 GB) fits on consumer GPUs (RTX 4090 24GB).

### Edge Deployment (E2B)

- Memory: <1.5 GB with 2-bit/4-bit quantization
- Raspberry Pi 5: 133 tokens/sec prefill, 7.6 tokens/sec decode
- Processes 4,000 input tokens across 2 skills in <3 seconds on GPU-accelerated devices
- Android: 4x faster inference, 60% less battery vs previous generation
- Platforms: Android (AICore), iOS, Windows, Linux, macOS (Metal), WebGPU (browser), Qualcomm IQ8 NPU, Jetson Nano/Orin

### Supported Frameworks (Day-1)

Hugging Face Transformers, vLLM, llama.cpp, MLX (Apple Silicon), Ollama, LM Studio, SGLang, Keras, TRL, transformers.js (browser), mistral.rs (Rust), ONNX, Docker, NVIDIA NIM

### Quantization Formats

GGUF (Q4/Q5/Q8 for llama.cpp), ONNX, MLX with TurboQuant, UQ/ISQ (mistral.rs), NVFP4 (forthcoming for Blackwell GPUs)

### Inference Settings

Default sampling: `temperature=1.0, top_p=0.95, top_k=64`

Input ordering: place image/audio tokens **before** text in the prompt for best results.

### When to Use Each Model

```
Use Case                                 Recommended Model
──────────────────────────────          ─────────────────
Mobile/embedded, real-time               E2B (2-4 bit)
On-device with audio understanding       E4B
Cost-efficient server, high throughput   26B-A4B (MoE)
Maximum quality, single-GPU              31B
```

The 26B-A4B is the sweet spot for most server deployments — 97% of 31B quality at ~8x less compute per token.

## Key Papers

1. Gemma Team. *Gemma: Open Models Based on Gemini Research and Technology* (2024). [arxiv 2403.08295](https://arxiv.org/abs/2403.08295)
2. Gemma Team. *Gemma 3 Technical Report* (2025). [arxiv 2503.19786](https://arxiv.org/abs/2503.19786)
3. Beyer et al. *PaliGemma: A versatile 3B VLM for transfer* (2024). [arxiv 2407.07726](https://arxiv.org/abs/2407.07726)
4. Steiner et al. *PaliGemma 2: A Family of Versatile VLMs for Transfer* (2024). [arxiv 2412.03555](https://arxiv.org/abs/2412.03555)
5. Google DeepMind. *Gemma 4 Model Card* (April 2026). [ai.google.dev/gemma/docs/core/model_card_4](https://ai.google.dev/gemma/docs/core/model_card_4)
6. HuggingFace. *Welcome Gemma 4* (April 2026). [huggingface.co/blog/gemma4](https://huggingface.co/blog/gemma4)
