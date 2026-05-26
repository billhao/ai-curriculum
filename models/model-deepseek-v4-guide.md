# DeepSeek V4: Million-Token Agentic MoE with Hybrid Sparse Attention

How DeepSeek scaled to 1.6T parameters and 1M context by replacing MLA with a CSA+HCA hybrid attention stack, swapping AdamW for Muon, moving expert weights to FP4, and folding the R1 reasoning line back into the mainline as three inference-time thinking modes.

## Background

**Release**: DeepSeek-V4 Preview, April 24, 2026. [Release note](https://api-docs.deepseek.com/news/news260424) · [V4-Pro model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) · [Tech report PDF](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/main/DeepSeek_V4.pdf) · [HF launch blog](https://huggingface.co/blog/deepseekv4). License: **MIT** (weights + code, full commercial use).

**Research lineage** — V4 sits at the end of a very tight chain:

1. **DeepSeek V2** (DeepSeek, May 2024, [arxiv 2405.04434](https://arxiv.org/abs/2405.04434)) — Introduced **MLA** (Multi-head Latent Attention): compress K/V into a low-rank latent, cache only the latent plus a small rotary component. First DeepSeek MoE at scale.

2. **DeepSeek V3** (DeepSeek, December 2024, [arxiv 2412.19437](https://arxiv.org/abs/2412.19437)) — 671B / 37B active, 14.8T tokens, MLA + DeepSeekMoE (256 routed + 1 shared, top-8), **MTP** (Multi-Token Prediction), **auxiliary-loss-free load balancing** (the `noaux_tc` scheme). FP8 training. 128K context.

3. **DeepSeek R1** (DeepSeek, January 2025, [arxiv 2501.12948](https://arxiv.org/abs/2501.12948)) — Pure-RL reasoning on V3-Base via GRPO. A **separate** model branch from the V3 chat line. (Covered in your R1 guide.)

4. **DeepSeek V3.1** (August 2025) — First **unified** Think/Non-Think API endpoint. Brought chain-of-thought into the mainline without a separate reasoner checkpoint. Still 128K, still MLA.

5. **DeepSeek V3.2-Exp** (September 2025) — Introduced **DSA** (DeepSeek Sparse Attention): learned sparse top-k over KV to cut long-context compute. A research preview.

6. **DeepSeek V3.2** (December 2025) — DSA productized. Added agentic post-training over 1,800+ environments and 85k+ instructions. Thinking-in-tool-use.

7. **DeepSeek V4** (April 24, 2026) — **1.6T total / 49B active** (Pro), **1M context**, MLA retired in favor of **CSA+HCA**, AdamW retired in favor of **Muon**, expert weights moved to **FP4**, residuals replaced with **mHC**. The R1 reasoning line folds back in as three API-level thinking modes.

**The arc**: 671B → 685B → 1.6T (2.4× scale), 128K → 1M (8× context), MLA → DSA → CSA+HCA (three attention designs in 18 months), AdamW → Muon, FP8 → FP4+FP8. Each generation retains MoE + MTP + shared-expert routing; everything around them keeps being rebuilt.

## What Problem Does V4 Solve?

V3.2 had three open problems when pushed toward 1M-token agentic workloads:

1. **KV cache dominates.** At 128K context, MLA cache is small. At 1M, even MLA's ~70KB/token blows up to **~70GB per sequence** — larger than the active weights. Every sparse-attention generation since has been a different bet on "how do we make long-context KV affordable."

2. **Reasoning + tools don't compose.** In V3.1/V3.2, `reasoning_content` was dropped at each new user turn. Agents doing multi-turn tool use lost their chain-of-thought between iterations. R1-style thinking was useful only for single-turn reasoning.

3. **AdamW optimizer state is ~2× the weights.** At 1.6T parameters in FP4, weight bytes drop to ~800GB but AdamW state (m, v in FP32) stays at ~13TB. Optimizer memory, not weights, becomes the training bottleneck.

V4 attacks all three: hybrid sparse attention to collapse KV, a serialization protocol that round-trips `reasoning_content` through tool calls, and Muon (which needs no per-parameter second moment) to shrink optimizer state.

## Model Family

```
                 DeepSeek-V4-Pro               DeepSeek-V4-Flash
                 ─────────────────             ──────────────────
Total params     1,600 B                       284 B
Active/token     49 B                          13 B
Layers           61                            43
Hidden dim       7,168                         4,096
Attn heads       128                           64
Routed experts   384 + 1 shared                256 + 1 shared
Active experts   6                             6
Context          1,048,576                     1,048,576
Vocab            129,280                       129,280
License          MIT                           MIT
FP8 weights      ~865 GB                       ~160 GB
Tensor parallel  MP=8  (e.g. 8×H200)           MP=4  (e.g. 4×H100)
```

Both ship in `-Base` (pretrain-only) and instruct variants. The instruct variants expose three modes at inference time:

| Mode         | Reasoning budget | Recommended ctx | Use case                          |
|--------------|------------------|-----------------|-----------------------------------|
| Non-Think    | 0 (skip)         | 8K+             | Chat, low-latency tools           |
| Think-High   | Medium           | 128K+           | Standard reasoning, coding        |
| Think-Max    | Uncapped         | ≥384K required  | Competition math, deep agents     |

There is **no separate `deepseek-reasoner` model anymore** — R1's successor is just `deepseek-v4-pro` with `reasoning_effort="max"`.

## Key Terms

**CSA (Compressed Sparse Attention)**: Compress the KV sequence by a factor `m=4` via softmax-gated pooling, then do *sparse top-k* attention against the compressed tokens (top-k = 512 for Flash, 1024 for Pro). Linear prefill, bounded decode cost.

**HCA (Heavily Compressed Attention)**: Same compression idea but at `m'=128`, with dense attention over the now-short compressed sequence. At 1M context, a 128× compression yields 8K effective KV tokens — cheap enough to do dense.

**mHC (Manifold-Constrained Hyper-Connections)**: Residual replacement. Instead of `x_{l+1} = x_l + f(x_l)`, mHC expands to `n_hc=4` parallel residual tracks, cross-mixes them through a **Sinkhorn-Knopp**–normalized (doubly-stochastic) coupling matrix refreshed every step, and merges at the top. ~6–7% extra training compute; improves stability and gradient flow at depth 61.

**Muon**: The optimizer shift. Muon maintains a single momentum buffer per parameter (vs AdamW's `m` and `v`), reducing optimizer state by ~2×. First used at this scale by Kimi; V4 is the largest public Muon run so far.

**Anticipatory Routing**: Load-balancing trick — routing decisions for layer `l+1` start computing from layer `l`'s hidden state while layer `l`'s MoE is still executing. Hides expert-parallel all-to-all latency.

**Hash routing (first 3 layers)**: The first 3 MoE layers route experts via a **hash of the token ID**, not a learned router. Forces early layers to see balanced expert load by construction, independent of what the router learns. Learned routing resumes at layer 4.

**DSML**: DeepSeek Markup Language. The tool-call protocol: special `|DSML|` token opens an XML-like block, arguments are annotated with `string="true"` for raw strings vs `string="false"` for JSON values.

**Quick Instructions**: A prompt-level feature letting users inject always-on system-level behavior (e.g., "always respond in Chinese"), persisted across tool turns alongside `reasoning_content`.

**DSec**: DeepSeek's Rust-based elastic compute platform (Firecracker microVMs + QEMU) used for *post-training* agentic rollouts — not pretraining infra.

## Architecture Deep-Dive

### Hybrid Attention: why CSA + HCA instead of more MLA

MLA (V2/V3) and DSA (V3.2) both answer the question "how do we make KV cheaper?" with a single mechanism. V4's bet is that **one mechanism isn't enough at 1M context** — you need two complementary ones, wired per-layer.

```
Depth 0-1 ─── HCA            (dense over 128×-compressed KV, fat global view)
Depth 2-59 ── HCA / CSA       (alternating: 4× compressed + sparse, then 128× + dense)
Depth 60 ──── MTP + sliding   (next-token-prediction head, local only)
```

Intuition:
- **CSA layers** behave like "retrieval": for each query, top-k 1024 over a 4×-compressed KV keeps **sharp detail** at bounded cost.
- **HCA layers** behave like "summary": dense attention over an 8K-token 128×-compressed KV gives **global coherence** across the full 1M window.
- Alternating them is cheaper than either alone while covering both failure modes.

KV storage per token is mixed-precision:

```
  FP8    non-RoPE K/V components
  BF16   RoPE position components   (numerics-sensitive)
  FP4    lightning indexer QK       (just for top-k selection, not attention values)
```

The net: at 1M context, V4-Pro's KV cache is **~10% of V3.2's** at equivalent context. V4-Flash is **~7%**. That's what makes 1M economically shippable.

A side component: the **lightning indexer** (64 heads, top-k 1024) is a small attention tower whose *only* job is to pick which KV blocks CSA should attend to. It's trained jointly with the main model but stored and computed in FP4 because its output is discrete indices — precision barely matters.

### MoE topology

V4-Pro's MoE is wider but shallower-per-token than V3:

```
                        V3            V4-Pro         Δ
Layers (total)          61            61             same
Routed experts          256           384            +50%
Shared experts          1             1              same
Active per token        8             6              fewer but bigger
Expert inter. dim       2,048         3,072          +50% each
Routing balance         noaux_tc      noaux_tc       same idea
First-layer routing     learned       hash (3 lyr)   NEW
Scoring function        softmax       sqrtsoftplus   NEW
```

Why 384 × top-6 instead of V3's 256 × top-8? Each active expert is now 50% wider. At equal active-param budget, fewer-but-fatter experts route less traffic through the all-to-all kernel, trading routing bandwidth for per-expert compute. This is the right trade on H100/H200 where NVLink is the scarce resource.

**sqrtsoftplus scoring**: replaces the softmax gate. Standard softmax over N=384 experts is peaky and fragile — one overconfident logit squashes the rest. `score = sqrt(softplus(logit))` is smoother, keeps small positive mass on non-top-k experts (useful for the auxiliary-loss-free balancing bias update), and is numerically stable without temperature tuning.

**Hash routing for layers 0-2**: early-layer routers in MoE training tend to collapse to 2-3 experts for weeks before auxiliary losses claw balance back. Hashing bypasses the problem: every token's expert choice in layer 0-2 is determined by `hash(token_id) mod n_experts`. Balanced by construction. Learned routing resumes at layer 3 when the residual stream has enough content for meaningful routing decisions.

### mHC residuals: what replaces `x + f(x)`

Standard residual: `x_{l+1} = x_l + block(x_l)`.

mHC (`n_hc=4` tracks):

```
x_l       ──────────────────────────────┐
                                        │
split into 4 tracks                     │
  [x_l^1, x_l^2, x_l^3, x_l^4]          │
                                        │
block processes each track              │
  [y^1, y^2, y^3, y^4]                  │
                                        │
Sinkhorn-Knopp coupling matrix P        │
  (doubly-stochastic, 4×4, 20 iters)    │
                                        │
recombined:                             │
  [z^i = Σ_j P_{ij} · y^j]              │
                                        │
x_{l+1} = mean(z) + x_l ────────────────┘
```

The Sinkhorn-Knopp step enforces that the coupling matrix `P` has all rows and columns summing to 1 — a soft doubly-stochastic constraint. This keeps the four tracks from collapsing into each other (which would turn mHC back into a scalar residual) and keeps one track from dominating (which would bottleneck gradient flow).

Cost: ~6-7% extra training FLOPs, 4× residual memory in the forward pass (offset by activation checkpointing). The claimed benefit is **deeper stable training** — at 61 layers on a wider hidden dim, vanilla residuals hit gradient instability faster.

### MTP head

MTP (Multi-Token Prediction) is retained from V3 with `depth=1` — a single extra lightweight transformer block predicts token `t+2` alongside the main head's `t+1`. During training, an auxiliary loss (`weight=0.3` during steady-state, decayed to `0.1` during LR cooldown) pushes both heads to match ground truth. At inference, the MTP head is discarded — it's purely a training-time regularizer that forces the main head to encode forward-looking structure.

## Training Recipe

### Data

- **Flash**: 32T tokens
- **Pro**: 33T tokens

Corpus weighting (from the tech report): heavy on math + code + multilingual + long documents + mid-training agentic trajectories. Agentic data is distinct from reasoning data — it includes tool-call traces with successful outcomes, mined from synthetic and human-annotated sources.

For reference, V3 was 14.8T tokens. V4 roughly 2.2× the corpus.

### Sequence-length curriculum

```
Stage        Context     Fraction of tokens   Purpose
─────        ───────     ─────────────────    ─────────────────────────
Stage 1      4K          most                 Dense knowledge, short tasks
Stage 2      16K         smaller              Document-level reasoning
Stage 3      64K         smaller              Multi-document, tool traces
Stage 4      1M          last slice           Long-context calibration
```

Crucial detail: **sparse attention (CSA) is OFF during the first 1T tokens** — training runs dense attention as warmup. This matters because the lightning indexer (which decides which blocks CSA attends to) is randomly initialized; switching to sparse too early starves the model of signal. After the dense warmup, sparse attention activates and trains through the remaining ~31T tokens.

### Optimizer — Muon (not AdamW)

Muon is the biggest training-recipe departure from V3.

**AdamW state per parameter**: `m` (FP32) + `v` (FP32) = 8 bytes / param on top of weights.
**Muon state per parameter**: single momentum buffer ≈ 4 bytes / param.

At 1.6T parameters, that difference is ~6.4TB of optimizer state saved. Combined with FP4 expert weights, V4 is the first public ~1.6T training run that can plausibly fit inside a cluster that couldn't have run V3.

Muon applies Newton-Schulz iteration to approximate the orthogonalization of the update direction. Gradient `g` → momentum `m` → orthogonalized via 5 Newton-Schulz steps → apply. The orthogonalization rotates the update into a direction with bounded singular values — in practice acts like a learning-rate-free preconditioner for 2D weight matrices.

Hyperparameters disclosed (Flash):
```
LR peak          2.7e-4
LR end           2.7e-5  (10× decay, cosine schedule)
LR warmup        2000 steps
Batch size       ramped to 75.5M tokens  (Flash) / 94.4M tokens (Pro)
MTP loss weight  0.3 steady → 0.1 during LR cooldown
Balance loss     1e-4 (auxiliary-loss-free noaux_tc bias update speed 0.001)
```

Compare to your own GPT-2 training: peak LR 6e-4, batch 0.5M tokens, AdamW β=(0.9, 0.95). V4-Pro's batch is ~200× yours, LR is 2× lower (scaling law: larger batch → slightly lower LR).

### Precision stack

```
Component                  Precision    Why
──────────                 ─────────    ────
MoE expert weights         FP4 QAT      Dominates parameter count; QAT keeps quality
Attention weights          FP8          Needs more dynamic range than experts
Embeddings / LM head       BF16         Full precision for output head
Router logits              FP32         Small but numerics-critical
KV cache (non-RoPE)        FP8          Shrink memory
KV cache (RoPE dims)       BF16         Rotary positions sensitive to precision
Lightning indexer QK       FP4          Output is discrete indices
Master weights             BF16         Standard mixed-precision
Muon momentum              BF16         (FP32 equivalent is too expensive at 1.6T)
```

**FP4 QAT detail**: expert weights use NVIDIA's `nvfp4` scheme (1 sign bit + 3 exp + microblock scaling). Critically, V4 claims **lossless FP4→FP8 dequantization** inside the FP8 GEMM framework — expert weights are stored FP4, cast to FP8 before matmul via a precomputed scale table, matmul runs in FP8. This means the FP8 kernels (DeepGEMM) don't need rewriting; FP4 is a pure storage format at training time.

### Stabilizers

**Anticipatory Routing** — start computing layer `l+1`'s routing decisions from layer `l`'s input-side activations, overlapping with layer `l`'s MoE all-to-all. Pure latency hide, no model quality change.

**SwiGLU clamping** — SwiGLU = `silu(x W_1) ⊙ (x W_3)`. At 1.6T scale, the elementwise product grows unbounded and triggers loss spikes. V4 clamps `|output| ≤ 10.0` (config: `swiglu_limit=10`). Similar trick to Gemma's logit soft-capping but applied to the MLP output.

**Training compute**: not disclosed. V3 used 2.788M H800-hours for 14.8T tokens. V4 is 2.2× tokens and ~2.4× parameters — a linear FLOP estimate puts V4 at roughly 5-6× V3's compute budget, or **~15-17M H800-hours**. This is a *guess*; DeepSeek has not published the number.

## Unified Thinking Modes

The API change:

```python
# V3.1/V3.2 — two endpoints
openai.chat.completions.create(model="deepseek-chat", ...)      # no reasoning
openai.chat.completions.create(model="deepseek-reasoner", ...)  # with reasoning

# V4 — one endpoint, three modes
openai.chat.completions.create(
    model="deepseek-v4-pro",
    extra_body={"thinking": {"type": "disabled"}},        # non-think
)
openai.chat.completions.create(
    model="deepseek-v4-pro",
    extra_body={"thinking": {"type": "enabled", "budget": "high"}},
)
openai.chat.completions.create(
    model="deepseek-v4-pro",
    extra_body={"thinking": {"type": "enabled", "budget": "max"}},
    # max requires ≥384K context allocated for the reasoning buffer
)
```

Think-Max's 384K context requirement is a hard constraint: the API returns 400 if you request `max` with a smaller `max_tokens` + prompt budget. This is because Think-Max rollouts during post-training were capped at ~200K reasoning tokens + 184K for the task — the model's output-length calibration assumes that budget.

### Reasoning persistence across tool calls

This is V4's big functional delta for agents. In V3.2:

```
turn 1: user "search X"
        → reasoning_content="...let me search..." + tool_call
turn 2: tool result
        → reasoning_content="...result says Y..." + response

turn 3: user "now do Z"
        → reasoning_content DISCARDED, model re-reasons from scratch
```

In V4:

```
turn 3: user "now do Z"
        → model RECEIVES prior reasoning_content, continues from where it left off
```

API-level enforcement: if you don't echo the prior `reasoning_content` in your message history during a tool-calling exchange, the API returns **400**. The tradeoff: your context bill goes up (reasoning tokens are often long), but multi-turn agent trajectories are coherent for the first time in DeepSeek's lineage.

The DSML tool schema:

```xml
|DSML|
<tool_call>
  <name>search</name>
  <arguments>
    <query string="true">DeepSeek V4 release</query>
    <top_k string="false">5</top_k>
  </arguments>
</tool_call>
|DSML|
```

`string="true"` means "the content is a raw string, don't try to parse it as JSON." `string="false"` means "parse this as a JSON literal" (so `5` becomes integer `5`, not string `"5"`). This replaces the JSON-inside-JSON-string encoding OpenAI and V3.2 used, which required double-escaping on every special character.

## What Changed vs V3 / V3.1 / V3.2 / R1

```
Aspect              V3             V3.1           V3.2           V4-Pro
──────              ──             ────           ────           ──────
Total params        671B           671B           685B           1,600B
Active             37B            37B            37B            49B
Context            128K           128K           128K           1M
Attention          MLA            MLA            DSA            CSA+HCA
Routed experts     256            256            256            384
Active experts     8              8              8              6
Scoring            softmax        softmax        softmax        sqrtsoftplus
Optimizer          AdamW          AdamW          AdamW          Muon
Expert precision   FP8            FP8            FP8            FP4 (QAT)
Residuals          standard       standard       standard       mHC (4-track)
Reasoning          (V3 chat)      Think/NT       Think/NT       Non/High/Max
Reasoning line     R1 separate    merged         merged         merged
Tool reasoning     —              discarded      discarded      persisted
Pretrain tokens    14.8T          ~15T           ~15T           33T
MTP depth          1              1              1              1
```

Vs **R1**: R1 used the V3-Base backbone + GRPO on correctness rewards. V4 folds that entire pipeline into post-training on a single unified backbone. The takeaway: "reasoning" is not an architecture anymore, it's a **post-training step + inference-time toggle**. No more two-product confusion.

## Benchmarks

From DeepSeek's official frontier comparison (tech report + HF model card). All V4 numbers are `V4-Pro-Max` (Think-Max mode).

| Metric               | V4-Pro Max | Claude Opus 4.6 | GPT-5.4 xHigh | Gemini 3.1 Pro High |
|----------------------|------------|-----------------|----------------|---------------------|
| MMLU-Pro             | 87.5       | 89.1            | 87.5           | **91.0**            |
| GPQA Diamond         | 90.1       | 91.3            | 93.0           | **94.3**            |
| SimpleQA-Verified    | 57.9       | 46.2            | 45.3           | **75.6**            |
| HLE                  | 37.7       | 40.0            | 39.8           | **44.4**            |
| HMMT 2026 Feb        | 95.2       | 96.2            | **97.7**       | 94.7                |
| IMOAnswerBench       | 89.8       | 75.3            | **91.4**       | 81.0                |
| LiveCodeBench        | **93.5**   | 88.8            | —              | 91.7                |
| Codeforces (rating)  | **3206**   | —               | 3168           | 3052                |
| SWE-Bench Verified   | 80.6       | **80.8**        | —              | 80.6                |
| Terminal-Bench 2.0   | 67.9       | 65.4            | **75.1**       | 68.5                |
| MRCR 1M              | 83.5       | **92.9**        | —              | 76.3                |
| CorpusQA 1M          | 62.0       | **71.7**        | —              | 53.8                |
| BrowseComp           | 83.4       | 83.7            | —              | —                   |

**Where V4 leads**: LiveCodeBench (93.5), Codeforces rating (3206). Ties: SWE-Verified.
**Where V4 trails**: world knowledge (MMLU-Pro, SimpleQA vs Gemini 3.1), long-context retrieval (MRCR 1M — Claude Opus 4.6 crushes this at 92.9 vs 83.5), HLE.

**Caveats**:
- No official AIME. DeepSeek uses HMMT 2026 Feb instead (95.2 is strong but not apples-to-apples with R1's old AIME numbers).
- No published **Qwen3 or Llama 5 head-to-head** in DeepSeek's own materials.
- MRCR 8-needle at 1M drops to 0.59 — long-context retrieval degrades past 128K even though the architecture "supports" 1M.

**Against V3.2**: V4-Pro-Base MATH-500 = 64.5 vs V3.2-Base 60.5 (base model, same eval). Modest gain on MATH, larger gains on long-context tasks where V3.2 couldn't compete at all.

## Inference

### VRAM / serving

```
Model        FP8 weight bytes    Recommended setup          MP
─────        ─────────────────   ───────────────────        ──
Flash        ~160 GB             2× H100 80GB  (FP8)        4
             ~80 GB              1× H100 80GB  (INT4 AWQ)   1
Pro          ~865 GB             8× H200 141GB (FP8)        8
             ~1.7 TB             16× H100 80GB (BF16)       8
```

**Note**: HuggingFace's UI shows "Model size 862B params" for V4-Pro because it counts FP4-expert + FP8-other bytes at bit-width, then divides. The **logical** parameter count is 1.6T. Don't confuse storage footprint with param count.

### Stacks

All three support V4 out of the box at launch:
- **vLLM** — high-throughput production serving, best for steady traffic
- **SGLang** — better for bursty and structured-output (JSON schemas, tool calls). Official V4 cookbook.
- **vLLM-Ascend** — Huawei Ascend support from day one (notable — first DeepSeek generation with that)
- **LMDeploy**, **DeepSeek-Infer** (reference demo) also supported

The FP4+FP8 mixed-precision weights ship directly from HF; no quantization step needed. For INT4 quantization of Flash down to single-GPU, use AWQ or GPTQ — community quants are already on HF within a day.

### Pricing

Official DeepSeek API pricing per 1M tokens (flagged: some pre-launch sources show different numbers; use the post-launch pricing page):

| Model     | Input (miss) | Input (cache hit) | Output   |
|-----------|--------------|-------------------|----------|
| V4-Pro    | $1.74        | $0.145            | $3.48    |
| V4-Flash  | $0.14        | $0.028            | $0.28    |

For context:
- GPT-5.4: $2.50 / $15 per 1M
- Claude Opus 4.6: $5 / $25 per 1M
- V4-Pro output is **~4.3× cheaper than Opus output** at comparable SWE-Verified.

Cache hits (prompt caching) are 12× cheaper than miss — aggressive caching matters for long-context workflows.

## Practical Considerations

**Use V4-Pro when**: coding agents (LiveCodeBench, Codeforces, SWE-Bench). Long multi-turn tool-use sessions where reasoning persistence across turns matters. Cost-sensitive workloads that would otherwise hit Opus/GPT-5.4.

**Use V4-Flash when**: low-latency chat, single-GPU deployments, high-volume non-agentic workloads. It's the best price/perf open MoE at launch.

**Think-Max when**: competition math, deep code debugging, research-grade agent trajectories. Don't use it for chat — it'll burn tokens and latency on reasoning that non-think handled fine.

**Skip V4 when**:
- World knowledge / factual QA — Gemini 3.1 Pro still wins (SimpleQA 75.6 vs V4 57.9).
- Long-context retrieval beyond 128K — MRCR 1M degrades (0.59 at 8-needle). If your actual bottleneck is "find 8 needles in 1M tokens," Claude Opus 4.6 (92.9) is the right choice.
- Deployments under 80GB single-GPU VRAM — even Flash at INT4 is tight.

**Local inference requirements**:
- `temperature=1.0, top_p=1.0` (DeepSeek's recommendation — different from the 0.7 default you'd use for other models).
- Preserve and echo `reasoning_content` across every tool-calling turn or the API returns 400.
- Think-Max demands ≥384K allocated context.

**Known limitations to plan around**:
- Long-context quality degrades above 128K. Treat 1M as "architecture supports it" not "use it blindly."
- Text-only. Multimodal is deferred; V4 is the last DeepSeek text-only flagship, probably.
- Same CCP-alignment posture as prior DeepSeek models. If that's a deployment concern, audit for your use case — V4-specific safety studies aren't out yet.

**Related to your own training work**:
- If you're running GRPO (from your R1 reproduction), V4-Pro-Base is the new RL starting point. 1.6T total but 49B active — same VRAM class as GRPO on V3-Base if you use FP4 experts.
- Muon is worth trying on your GPT-2 124M run. At that scale the AdamW→Muon savings are small (tens of MB), but the orthogonalization behavior is worth studying before the next larger run.
- V4's hash-routing trick for early MoE layers applies directly if you train a small MoE — skip the "expert collapse for the first 1B tokens" phase entirely.

## Open Questions

The tech report does not answer:

1. **Training compute**: no FLOPs, no cluster size, no wall-clock, no accelerator model. All three prior DeepSeek reports disclosed H800-hours. V4's silence is conspicuous.
2. **Post-training data mix**: "agentic data" is mentioned but not enumerated. What environments? How many trajectories? How were they filtered?
3. **Qwen3 / Llama 5 head-to-heads**: missing from DeepSeek's own materials.
4. **Multimodal roadmap**: V4 is text-only in a world where Gemini 3 and Claude 4 are natively multimodal. No signal on whether DeepSeek is bundling or staying text-focused.

## Key Papers

1. [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) — DeepSeek, May 2024. MLA origin paper.
2. [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — DeepSeek, December 2024. 671B / 37B active, MTP, noaux_tc balancing.
3. [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — DeepSeek, January 2025. Pure-RL reasoning; V4 folds this back into the mainline.
4. [DeepSeek V4 Technical Report (HF-hosted PDF)](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/main/DeepSeek_V4.pdf) — DeepSeek, April 24, 2026. **Primary source for this guide.** No arxiv landing page at time of writing.
5. [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) — Moonshot/Kimi, February 2025. The optimizer V4 adopted.
6. [Hyper-Connections](https://arxiv.org/abs/2409.19606) — origin of the n-track residual idea that mHC generalizes.
7. [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) — Alibaba, 2025. For competitive context.

## References (Live Resources)

- [V4-Pro HF model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- [V4-Flash HF model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- [DeepSeek V4 Preview release note](https://api-docs.deepseek.com/news/news260424)
- [HF launch blog](https://huggingface.co/blog/deepseekv4)
- [Thinking Mode API docs](https://api-docs.deepseek.com/guides/thinking_mode)
- [Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing)
- [Coding Agents guide](https://api-docs.deepseek.com/zh-cn/guides/coding_agents)
- [DeepEP (expert-parallel comms)](https://github.com/deepseek-ai/DeepEP)
- [DeepGEMM (FP8 kernels)](https://github.com/deepseek-ai/DeepGEMM)
- [DualPipe (pipeline parallelism)](https://github.com/deepseek-ai/DualPipe)
- [3FS (distributed filesystem)](https://github.com/deepseek-ai/3FS)
- [SGLang V4 cookbook](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4)
- [vLLM-Ascend V4 tutorial](https://docs.vllm.ai/projects/ascend/en/v0.13.0/tutorials/DeepSeek-V4.html)
- [Simon Willison's V4 post](https://simonwillison.net/2026/Apr/24/deepseek-v4/)
