# GLM-5.2: Sparse-Attention Coding Model with Cross-Layer Index Sharing

How Zhipu/Z.ai shipped a 744B/40B open-weights MoE with a usable 1M context by taking the GLM-5 backbone (MLA-256 + DeepSeek Sparse Attention + parameter-shared MTP) and adding **IndexShare** — reusing one DSA lightning-indexer top-k selection across every four attention layers to cut per-token attention FLOPs 2.9× at 1M context — plus a deeper MTP draft head that lifts speculative-decoding acceptance ~20%.

## Background

**Release**: GLM-5.2, June 13, 2026, by Zhipu AI / Z.ai (with Tsinghua University). First to GLM Coding Plan users, then open weights on HuggingFace under **MIT** (full commercial use). [Model card](https://huggingface.co/zai-org/GLM-5.2) · [FP8 weights](https://huggingface.co/zai-org/GLM-5.2-FP8) · [API docs](https://docs.z.ai/guides/llm/glm-5.2) · [GitHub](https://github.com/zai-org/GLM-5). There is **no standalone GLM-5.2 paper** — the model card cites the GLM-5 tech report for the backbone and the IndexCache paper for the index-sharing method.

**Research lineage** — GLM-5.2 sits at the intersection of two DeepSeek-originated ideas (DSA, MTP) and one new Zhipu contribution (IndexShare):

1. **DeepSeek-V2** (DeepSeek, May 2024, [arxiv 2405.04434](https://arxiv.org/abs/2405.04434)) — **MLA** (Multi-head Latent Attention): compress K/V into a low-rank latent, cache only the latent + a small RoPE component. GLM-5's attention is an MLA variant.

2. **Multi-Token Prediction** (Gloeckle et al., Meta, 2024, [arxiv 2404.19737](https://arxiv.org/abs/2404.19737)) — train the model to predict several future tokens with auxiliary heads. Improves the base model and doubles as a self-speculative draft at inference.

3. **DeepSeek-V3** (DeepSeek, Dec 2024, [arxiv 2412.19437](https://arxiv.org/abs/2412.19437)) — 671B/37B, DeepSeekMoE (256 routed + 1 shared, top-8), **MTP depth-1**, auxiliary-loss-free balancing. The MoE topology GLM-5 copies almost verbatim.

4. **DeepSeek-V3.2-Exp / V3.2** (DeepSeek, Sep–Dec 2025, [arxiv 2512.02556](https://arxiv.org/abs/2512.02556)) — **DSA** (DeepSeek Sparse Attention): a lightning indexer scores all past tokens, selects top-k=2048, and attention runs only over that subset. Introduced via continued pre-training (dense → sparse), not from scratch.

5. **GLM-4.5** (Zhipu, 2025) — the predecessor: 355B total / 32B active MoE, standard (dense) MLA, 128K context. GLM-5 doubles its size.

6. **GLM-5: from Vibe Coding to Agentic Engineering** (GLM-5 Team, Zhipu AI & Tsinghua, Feb 24 2026, [arxiv 2602.15763](https://arxiv.org/abs/2602.15763)) — **the backbone source of truth.** 744B/40B, MLA-256+Muon Split, DSA via continued pre-training, parameter-shared MTP, 28.5T-token budget. GLM-5.1 (Apr 7) and GLM-5.2 (Jun 13) are post-training + efficiency refreshes of this base.

7. **IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse** (Bai, Dong, Jiang, Lv, Du, Zeng, Tang, Li — Tsinghua & Z.ai, Mar 12 2026, [arxiv 2603.12201](https://arxiv.org/abs/2603.12201)) — **the IndexShare source of truth.** Shows DSA's indexer is the long-context bottleneck (81% of prefill time at 200K) and that its top-k selections are 70–100% redundant across adjacent layers, so most indexers can be removed.

**The arc**: GLM-4.5 (355B/32B, dense MLA, 128K) → GLM-5 (744B/40B, MLA-256, DSA, 128K→202K train) → GLM-5.2 (same base + IndexShare + deeper MTP, **1M context**). Each step keeps MoE + MTP and rebuilds the attention-efficiency story — exactly the pattern you saw in your DeepSeek-V4 guide (MLA → DSA → CSA+HCA), but GLM bets on making DSA *cheaper per layer* rather than replacing it.

## What Problem Does GLM-5.2 Solve?

DSA already fixed the headline long-context cost: dense attention is `O(L²)` per layer, and DSA's top-k selection drops the *core attention* to `O(Lk)` with k=2048. But DSA introduced a second, sneakier cost — **the indexer itself is still `O(L²)` at every layer.**

```
DSA per layer = INDEXER (score all L past tokens)  +  CORE ATTENTION (over top-k=2048)
                O(L²)  ← cheap per-FLOP but quadratic    O(Lk) ← linear, bounded

Across N≈78 layers:   indexer total = O(N·L²)   ← this is what blows up at 1M
```

The IndexCache paper profiles a 30B DSA model and finds the indexer's share of prefill time rises from 27% at 10K → **81% at 200K**, and 31% → 41% of decode time. At 1M context the indexer dominates almost everything. So "supporting 1M" on a DSA model is gated not by the core attention (already linear) but by N copies of a quadratic indexer.

GLM-5.2's answer (**IndexShare**) is the observation that you don't need a fresh indexer at every layer, because *which* tokens matter barely changes from one layer to the next.

## Key Terms

**DSA (DeepSeek Sparse Attention)**: Two-stage attention. (1) A lightweight **lightning indexer** scores every past token against the current query with a multi-head ReLU-gated dot product. (2) The top-k=2048 highest-scoring positions are selected, and the main (MLA) attention runs only over those. Trained, not heuristic — unlike fixed sliding windows.

**Lightning indexer**: GLM-5's indexer has 32 heads, head dim 128, runs in FP8, and emits a length-L score vector per query. Its only output is a *set of indices* — so precision and exactness matter little, which is why it can be cheap and later shared.

**IndexShare**: Zhipu's name for cross-layer reuse of the indexer's top-k selection. Layers are split into **Full** (run their own indexer, compute fresh top-k, cache it) and **Shared** (no indexer — inherit the nearest preceding Full layer's top-k). GLM-5.2 ships `index_topk_freq=4`: one Full indexer every 4 layers, the other 3 reuse it. 75% of indexer computations removed. (Productized form of the IndexCache paper.)

**MTP (Multi-Token Prediction)**: Extra transformer head(s) that predict tokens `t+2, t+3, …` alongside the main `t+1` head. Trains the base model to encode forward structure, and at inference serves as a **self-speculative draft** — the model proposes several tokens, then verifies them in one forward pass.

**Parameter-shared MTP**: GLM-5 predicts multiple future tokens but **shares one set of weights across all MTP iterations** (vs DeepSeek-V3's single depth-1 head). This closes the train/inference gap that hurt the 2nd-token acceptance rate, raising accept length to 2.76 (vs DeepSeek-V3.2's 2.55) at 4 speculative steps.

**MLA-256 + Muon Split**: GLM-5's attention. Plain MLA (576-dim latent) underperformed GQA-8 under the Muon optimizer; Zhipu fixed it two ways — *Muon Split* (orthogonalize per-head projection sub-matrices so heads update at different scales) to match quality, and *MLA-256* (raise V head dim 192→256, cut head count by 1/3) to cut decode cost while holding training compute constant.

**Accept length**: In speculative decoding, the average number of draft tokens accepted per verification step. Higher = fewer forward passes per token = faster decode. GLM-5.2 reports up to **+20%** over GLM-5.

## Architecture Deep-Dive

### Full spec (GLM-5 Table 10, authoritative)

```
                        GLM-4.5        GLM-5 / 5.2       note
                        ───────        ───────────       ────
Total params            355 B          744 B             ~2.1× scale
Active / token          32 B           40 B
Dense layers            3              3                 first 3 layers are dense MLP
MoE layers              89             75                fewer but wider
MTP layers              1              1
Total transformer lyrs  92             78
Hidden dim              5,120          6,144
Dense FFN dim           12,288         12,288
MoE expert FFN dim      1,536          2,048             wider experts
Attention heads         96             64
QK head dim             128            192               (128 no-RoPE + 64 RoPE)
V head dim              128            256               ← the "256" in MLA-256
Q LoRA rank             —              2,048             MLA latent (query)
KV LoRA rank            —              512               MLA latent (key/value)
Indexer heads           —             32                 DSA lightning indexer
Indexer head dim        —             128
Experts (total)         160            256
Routed / token          8              8
Shared experts          1              1
Vocab                   151,552        154,880
Context (5.2)           128 K          1,048,576
Max output (5.2)        —              131,072
HF stored size          —             ~753 B            (744B logical; 753B counts stored tensors)
```

Note the param count excludes embeddings + output layer per the report; HuggingFace's "753B" includes stored tensors. Same logical model — don't double-count.

### MoE topology — basically DeepSeek-V3's

256 routed experts + 1 shared, top-8 routing, auxiliary-loss-free balancing. This is the same recipe you studied for DeepSeek-V3 in your MoE work; GLM-5 just widens the expert FFN to 2,048 and the hidden dim to 6,144. The first 3 layers are dense MLP (no routing) — early layers route poorly before the residual stream carries enough signal, so Zhipu skips routing there entirely.

### DSA: continued pre-training, not from scratch

GLM-5 doesn't train DSA from random init. It takes the dense-MLA base at the end of mid-training and adapts it in two stages — the same "dense warm-up → sparse training" recipe DeepSeek-V3.2-Exp used:

```
Stage 1 — Indexer warm-up   1,000 steps, 14 seqs × 202,752 tokens, LR 5e-3
                            Only the indexer trains (KL-distilled against the
                            dense model's aggregated attention). Everything
                            else frozen. The indexer learns "which tokens the
                            dense model would have attended to."

Stage 2 — Sparse adaptation 20 B tokens, follows mid-training data, constant LR 1e-5
                            top-k=2048 selection turns on; whole model + indexer
                            co-train. Indexer gets distillation gradients on a
                            detached graph.
```

20B tokens is tiny (DeepSeek-V3.2 used 943.7B) yet enough to match the dense MLA model — long-context RULER@128K: 79.21 (dense) → 78.86 (DSA), essentially lossless. The claim that makes this work: **~90% of attention entries in long context are redundant**, so a top-2048 selection loses almost nothing while cutting attention compute ~1.5–2×.

Why DSA is "lossless by construction" where sliding-window / linear-attention alternatives aren't: GLM-5's ablations (Table 5) show SWA-interleave drops 30 points on RULER@128K, Gated-DeltaNet and search-based SWA still leave a 5–7 point gap — because they *discard* long-range dependencies. DSA's indexer keeps full-range reach; it only sparsifies *which* of those it computes attention over.

**RL detail worth knowing**: during reasoning RL, GLM-5 uses a *deterministic* top-k (`torch.topk`) in the indexer and **freezes the indexer**. Non-deterministic CUDA top-k caused entropy collapse within a few RL steps — the same train/inference-mismatch class of bug as MoE routing replay, which you've seen with expert collapse.

### IndexShare: the headline innovation

The IndexCache paper verifies the key premise empirically: compute the indexer's top-k set at every layer, then measure pairwise overlap. **Adjacent layers share 70–100% of their selected tokens.** So running 78 independent indexers is mostly recomputing the same answer.

IndexShare partitions the N attention layers into a binary pattern `c = c₁c₂…c_N` with each `cᵢ ∈ {Full, Shared}`:

```
                  STANDARD DSA                    INDEXSHARE (index_topk_freq=4)

Layer 1  [F] indexer → top-k → attn        [F] indexer → top-k → CACHE → attn
Layer 2  [F] indexer → top-k → attn        [S]   ·  reuse cached top-k → attn
Layer 3  [F] indexer → top-k → attn        [S]   ·  reuse cached top-k → attn
Layer 4  [F] indexer → top-k → attn        [S]   ·  reuse cached top-k → attn
Layer 5  [F] indexer → top-k → attn        [F] indexer → top-k → CACHE → attn
  ⋮              ⋮                                    ⋮
                                           1 indexer per 4 layers = 75% removed
```

The inference loop changes by exactly one conditional branch:

```
for layer ℓ in 1..N:
    if pattern[ℓ] == Full:
        I  = indexer_ℓ(X)          # O(L²) — the expensive part
        T  = topk(I, k=2048)
        cache = T                  # overwrite a single tensor buffer
    else:                          # Shared
        T  = cache                 # reuse — skip the indexer entirely
    X = sparse_attention_ℓ(X, T)   # O(Lk) — still runs every layer
    X = ffn_or_moe_ℓ(X)
```

`cache` is one index tensor, overwritten at each Full layer — **zero extra GPU memory** beyond what DSA already allocates. Core sparse attention still runs every layer; only the indexer is shared.

**Two ways to choose the pattern** (paper):
- **Training-free** — greedy search: start all-Full, flip the layer whose removal least raises LM loss on a calibration set, repeat. Finding *which* layers to keep matters far more than how many — uniform "every-4th" interleaving degrades long-context, but a searched pattern recovers it.
- **Training-aware** — multi-layer distillation: train each retained Full indexer against the *averaged* attention distribution of all Shared layers it serves. Proposition 1 in the paper proves this is gradient-equivalent to distilling toward the centroid of those layers. After retraining, the pattern-sensitivity vanishes — even uniform 1/4 interleaving matches the full-indexer baseline (and 1/4 actually *beats* DSA on AIME 2025 92.6 vs 91.0 and GPQA 78.6 vs 77.6, acting as a mild regularizer). GLM-5.2 ships the training-aware form.

### IndexShare — numerical walkthrough at 1M context

Why "2.9× per-token FLOPs"? Compare the two attention costs per layer at L = 1,048,576, k = 2048:

```
Indexer per layer   ≈ L² × (index_heads × index_head_dim)
                    = (1.05e6)² × (32 × 128)        ≈ 4.5e18 FLOP
Core attn per layer ≈ L × k × (heads × V_head_dim)
                    = 1.05e6 × 2048 × (64 × 256)    ≈ 3.5e16 FLOP

→ at 1M, the indexer is ~125× the core attention. It IS the attention budget.
```

So the attention-FLOP profile across all 78 layers is dominated by 78 indexers. Cut that to ~20 indexers (1/4) and the attention-related per-token FLOPs fall by roughly the indexer fraction → the reported **2.9×** at 1M (not a clean 4× because core attention, the shared-layer branch, and FFN/MoE FLOPs don't shrink).

That's a FLOP figure. The **measured wall-clock** speedups are smaller and context-dependent (latency includes memory traffic, kernels, MoE all-to-all):

```
IndexCache measured (30B DSA model, H100, SGLang, dp=8):
                       10K     60K     120K    200K
  Prefill speedup      1.27×   1.31×   1.51×   1.82×   (1/4 retention)
  Decode  speedup      ~1.15×  1.32×   1.40×   1.48×
On the 744B GLM-5:     ≥1.3× both prefill & decode beyond 100K context
Indexer compute removed: 75%, with negligible quality loss
```

The gains grow with context — exactly because the indexer's share of cost grows with context. At 10K it barely helps; at 1M it's the difference between shippable and not.

### MTP: parameter-shared draft head

GLM-5 keeps DeepSeek-V3's MTP idea but fixes its weakness. DeepSeek-V3 trains a single depth-1 MTP layer (predict `t+2`) but at inference wants to draft several tokens — a train/inference mismatch that drops acceptance of the later draft tokens.

```
DeepSeek-V3 MTP:   train 1 layer (predict t+2)  →  infer: draft 2 tokens   accept len 2.55
GLM-5 MTP:         share 1 weight set across 3 MTP iterations during training
                                                 →  infer: draft 3 tokens   accept len 2.76
GLM-5.2 MTP:       deeper draft (3 → 5 draft tokens), tuned                  accept len +~20%
```

Mechanically: the shared MTP block is applied recurrently — feed its own prediction back to predict the next, sharing weights across iterations so memory stays flat (1 layer's worth) while training matches the multi-step inference rollout. GLM-5.2 extends the draft horizon and reports up to 20% higher acceptance length, which at 4–5 speculative steps translates roughly linearly into decode tokens/sec.

`index_share` is also enabled for the MTP iteration in GLM-5.2's config — the draft step reuses the main model's cached top-k indices instead of running its own indexer, so MTP doesn't reintroduce the indexer cost IndexShare just removed.

## Training Recipe (from the GLM-5 report)

```
Pre-training        18 T general @ 4K  →  9 T code/reasoning @ 4K
Mid-training        1 T long code/reasoning @ 32K
                    500 B / 50 B long-context + agent data @ 128K / 200K
DSA adaptation      20 B @ 200K (dense warm-up → sparse)
                    ─────────────────────────────────────
                    Total base budget: 28.5 T tokens
```

- **Optimizer**: Muon (not AdamW), cosine decay, batch-size warmup. Pretrain LR 0→2e-4 warmup, decay to 4e-5. Mid-training 4e-5→1e-5. DSA warm-up 5e-3→2e-4; sparse adaptation constant 1e-5. (Compare your GPT-2: peak 6e-4, AdamW.)
- **INT4 QAT** in the SFT stage, with a quantization kernel that's bitwise-identical between train and inference — no post-hoc quantization drift.
- **Post-training**: SFT (with *interleaved* / *preserved* / *turn-level* thinking modes) → Reasoning RL (GRPO + IcePop, KL term removed, β=2, ε_low=0.2, ε_high=0.28, group 32) → Agentic RL (fully async, TITO gateway, 10K+ SWE + terminal envs) → General RL → on-policy cross-stage distillation to undo capability regression.

## Benchmarks (GLM-5.2 model card)

GLM-5.2 is positioned as a **coding / agentic** model, and the numbers lead there.

| Category   | Benchmark                | GLM-5.2 | context                        |
|------------|--------------------------|---------|--------------------------------|
| Reasoning  | AIME 2026                | 99.2    | near-saturated                 |
|            | GPQA-Diamond             | 91.2    |                                |
|            | HMMT Feb 2026            | 92.5    |                                |
|            | IMO-AnswerBench          | 91.0    |                                |
|            | HLE                      | 40.5    | 54.7 with tools                |
|            | CritPt                   | 16.7    | hard frontier physics          |
| Coding     | SWE-bench Pro            | 62.1    | up from GLM-5.1's 58.4         |
|            | Terminal-Bench 2.1       | 81.0    | 82.7 best harness; 5.1 was 62.0|
|            | ProgramBench             | 63.7    |                                |
|            | NL2Repo                  | 48.9    |                                |
|            | FrontierSWE              | 74.4    | long-horizon; > GPT-5.5 72.6   |
| Agentic    | MCP-Atlas (public)       | 76.8    | tool-use                       |
|            | Tool-Decathlon           | 48.2    |                                |
|            | Code Arena (Frontend)    | #2      | +29 Elo over Claude Opus 4.7   |
|            | Design Arena             | #1      | Elo 1360                       |
|            | Vending-Bench 2          | #1 open | long-horizon business sim      |

For backbone context, GLM-5-Base (Table 11): MMLU 88.3, LiveCodeBench-Base 34.4, GSM8K 87.4, MATH 56.4 — a strong base, with the big jumps coming from the agentic/coding RL stack, not the pretrain.

**Caveats**: VERY_NEW (days old at this writing); model-card numbers and configs may still shift. No separate 5.2 paper, so training-data and post-training specifics for the 5.2 delta over 5.1 aren't disclosed. CritPt 16.7 shows the frontier-science ceiling is still far off.

## GLM-5.2 vs Alternatives

| Aspect              | GLM-5.2          | DeepSeek-V3.2    | DeepSeek-V4-Pro   | GLM-4.5         |
|---------------------|------------------|------------------|-------------------|-----------------|
| Total / active      | 744B / 40B       | 685B / 37B       | 1,600B / 49B      | 355B / 32B      |
| Context             | 1 M              | 128 K            | 1 M               | 128 K           |
| Attention           | MLA-256 + DSA    | MLA + DSA        | CSA + HCA         | dense MLA       |
| Indexer cost        | **shared (1/4)** | per-layer        | per-layer (FP4)   | n/a             |
| MTP                 | shared, deep draft| depth-1         | depth-1           | depth-1         |
| Optimizer           | Muon (+ Split)   | AdamW            | Muon              | Muon            |
| License             | MIT              | MIT              | MIT               | MIT             |
| Positioning         | coding / agentic | general          | coding / agentic  | general         |

The clean contrast with DeepSeek-V4: both reach 1M, but V4 *replaces* DSA with a new CSA+HCA stack (two attention mechanisms wired per-layer) and scales to 1.6T. GLM-5.2 instead *keeps* DSA and makes its indexer cheaper via sharing — a smaller, more surgical change at roughly half the parameter count.

## Practical Considerations

**Use GLM-5.2 when**: coding agents and long-horizon agentic tasks (its strongest benchmarks — SWE-bench Pro, Terminal-Bench, FrontierSWE, frontend Arenas); 1M-context workloads where DSA's indexer would otherwise dominate cost; you want open MIT weights at ~half DeepSeek-V4's footprint. Use `reasoning_effort="max"` for hard coding/math.

**Serving**: SGLang, vLLM, Transformers, KTransformers supported at launch. FP8 weights ship directly (`zai-org/GLM-5.2-FP8`); BF16 also available. IndexShare and DSA are in the modeling code — `index_topk=2048`, `index_topk_freq=4`, `indexer_types` alternating full/shared.

**Skip / caution**: frontier science (CritPt 16.7) and pure world-knowledge tasks aren't its design center; it's a coding-first model. IndexShare's wall-clock benefit only shows up past ~100K context — at short context you're paying for machinery that barely helps.

**Relevant to your own work**:
- **Speculative decoding**: GLM-5.2's parameter-shared MTP is a clean case study of *self-speculation* — the draft model is the target model's own extra head, no separate draft net. If you implement spec-decoding on your GPT-2, the train/inference mismatch GLM fixes (single head trained, multiple tokens drafted) is exactly the trap to design around. Measure accept length, not just tokens/sec.
- **IndexShare is a `kv-cache`-free analogue of layer sharing**: you already know MoE shares *compute* across tokens; IndexShare shares a *routing decision* (which tokens to attend) across layers. The "70–100% overlap across adjacent layers" finding generalizes — if you ever build a sparse-attention model, profile cross-layer top-k overlap before paying for per-layer selection.
- **Muon + Muon Split**: if you revisit MLA at small scale, the GLM finding (plain MLA < GQA-8 under Muon, fixed by per-head orthogonalization) is a concrete gotcha to check.

## Open Questions

1. **5.2 vs 5.1 delta**: no paper isolates what changed between GLM-5.1 (Apr) and 5.2 (Jun) beyond IndexShare + deeper MTP + the 1M extension. Post-training data mix for the coding gains is undisclosed.
2. **IndexShare at 1M, measured**: the 2.9× is a FLOP figure; published wall-clock numbers top out at 200K (1.82× prefill) on a 30B model, with only "≥1.3×" stated for the 744B model. True 1M end-to-end latency isn't reported.
3. **Long-context *quality* at 1M**: "supports 1M" ≠ "retrieves well at 1M." The DSA RULER numbers are at 128K; 1M retrieval quality (needle-in-haystack past 128K) isn't in the materials.

## Key Papers

1. [GLM-5: from Vibe Coding to Agentic Engineering](https://arxiv.org/abs/2602.15763) — GLM-5 Team, Zhipu AI & Tsinghua, Feb 2026. **Backbone source of truth** (architecture Table 10, DSA continued-pretraining, parameter-shared MTP, training recipe).
2. [IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse](https://arxiv.org/abs/2603.12201) — Bai et al., Tsinghua & Z.ai, Mar 2026. **IndexShare source of truth** (Full/Shared partition, greedy + distillation methods, speedup numbers).
3. [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) — DeepSeek, 2025. DSA origin (lightning indexer, top-k=2048, dense→sparse adaptation).
4. [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — DeepSeek, Dec 2024. MoE topology + depth-1 MTP that GLM-5 builds on.
5. [Better & Faster Large Language Models via Multi-Token Prediction](https://arxiv.org/abs/2404.19737) — Gloeckle et al., Meta, 2024. The MTP objective.
6. [DeepSeek-V2](https://arxiv.org/abs/2405.04434) — DeepSeek, May 2024. MLA origin.
7. [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — Leviathan et al., ICML 2023. The speculative-decoding framework MTP plugs into.

## References (Live Resources)

- [GLM-5.2 model card (HF)](https://huggingface.co/zai-org/GLM-5.2) · [GLM-5.2-FP8](https://huggingface.co/zai-org/GLM-5.2-FP8)
- [GLM-5.2 config.json](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json) — `index_topk`, `index_topk_freq`, `indexer_types`
- [Z.ai GLM-5.2 API docs](https://docs.z.ai/guides/llm/glm-5.2)
- [GLM-5 GitHub](https://github.com/zai-org/GLM-5)
- [vLLM recipe for GLM-5.2](https://recipes.vllm.ai/zai-org/GLM-5.2)
