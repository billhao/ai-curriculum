# Do Language Models Need Sleep? Offline Recurrence for Better Online Inference

How a hybrid attention-SSM model spends extra compute *between* prediction steps — looping `N` times over a chunk of context to consolidate it into a fixed-size fast-weight state — and then throws the KV cache away, so answer-token latency stays single-pass while deep-reasoning accuracy goes up.

> Confidence: **VERY_NEW**. The paper (arXiv 2605.26099) was submitted **25 May 2026** (v2 27 May), ~3 days before this guide. Every equation, architecture, and benchmark number below was **read and transcribed verbatim from the full 15-page PDF** (not from search-engine summaries). It remains a fresh preprint: no public code, no independent replication, largest model 2B, and one setup quirk worth holding in mind (the query is placed *before* the long context in GSM-Infinite, enabling query-aware consolidation). Note also a naming collision: a *different* concurrent paper (Behrouz, Hashemi, Mirrokni) is also titled "Language Models Need Sleep" — the authors flag this in a footnote; it is RL/parameter-expansion based, not this work.

## Background

**Originating paper**: [Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference](https://arxiv.org/abs/2605.26099) (Sangyun Lee & Giulia Fanti — Carnegie Mellon; Sean McLeish & Tom Goldstein — University of Maryland; May 2026). GPU resources from Modal.

**Research lineage** — "Sleep" sits at the intersection of two long threads: (a) *fast weights* as a separate, faster-changing memory tier, and (b) *fixed-state sequence models* (SSMs / linear attention) that compress history instead of caching it. Its novelty is not a new memory cell — it reuses Gated DeltaNet — but a new *schedule*: spend offline compute looping over context before you evict it.

1. **Fast Weights** (Schmidhuber, 1992) — [Learning to Control Fast-Weight Memories](https://people.idsia.ch/~juergen/fastweights/ncfastweightsrev.html). A slow net emits context-dependent weight changes for a fast net; weights themselves act as short-term memory and variable binding. The conceptual root.

2. **Using Fast Weights to Attend to the Recent Past** (Ba, Hinton, Mnih, Leibo, Ionescu — Toronto/DeepMind, 2016) — [arXiv:1610.06258](https://arxiv.org/abs/1610.06258). Outer-product fast weights store recent memories and approximate attention without keeping copies of activations. The modern ancestor of the `S_t = α S_{t-1} + β v kᵀ` update.

3. **Transformers are RNNs / Linear Attention** (Katharopoulos et al., 2020) — [arXiv:2006.16236](https://arxiv.org/abs/2006.16236). `softmax(QKᵀ)V` → kernelized linear recurrence with a fixed-size state. O(N) instead of O(N²).

4. **Linear Transformers Are Secretly Fast Weight Programmers** (Schlag, Irie, Schmidhuber, 2021) — [arXiv:2102.11174](https://arxiv.org/abs/2102.11174). The bridge: linear attention *is* an additive outer-product fast-weight program. This is why an SSM state can be called a "fast weight."

5. **S4** (Gu, Goel, Ré — Stanford, 2021) — [arXiv:2111.00396](https://arxiv.org/abs/2111.00396). Structured state-space model; linear-time long-range modeling with a fixed state.

6. **Mamba** (Gu, Dao — CMU/Princeton, 2023) — [arXiv:2312.00752](https://arxiv.org/abs/2312.00752). Selective (input-dependent) SSM; first linear-time architecture to rival Transformers at scale.

7. **Mamba-2 / SSD** (Dao, Gu, 2024) — [arXiv:2405.21060](https://arxiv.org/abs/2405.21060). State-space duality: SSMs and attention are two views of the same thing. The `S_t = α_t S_{t-1} + β_t v_t k_tᵀ` form the Sleep paper writes is the Mamba-2 update.

8. **DeltaNet (parallelized)** (Yang et al., 2024) — [arXiv:2406.06484](https://arxiv.org/abs/2406.06484). Chunkwise-parallel delta-rule writes (error-correcting, not just additive). Makes delta-rule models trainable at scale.

9. **Gated DeltaNet (GDN)** (Yang, Kautz, Hatamizadeh — NVIDIA, 2024) — [arXiv:2412.06464](https://arxiv.org/abs/2412.06464). Adaptive decay gate + delta rule. **This is the actual SSM block Sleep uses in its experiments** (see the sibling [gated-deltanet-2-guide](../model-architecture/gated-deltanet-2-guide.md) for the full delta-rule lineage).

10. **Test-Time Training (TTT) layers** (Sun et al. — Stanford, 2024) — [arXiv:2407.04620](https://arxiv.org/abs/2407.04620). The hidden state *is* a small model, updated by gradient steps on a self-supervised loss at test time — origin of "the state is a learner." The paper's actual head-to-head is with the long-context TTT system **Tandon et al.** (2025) — [arXiv:2512.23675](https://arxiv.org/abs/2512.23675) — which replaces full attention with sliding-window attention and does *one* test-time gradient step on a subset of MLP layers per chunk. Sleep's contrast: it uses a *learned recurrent forward pass* as the update rule (not one-step gradient descent on a fixed scalar loss), and runs it `N` times.

11. **Depth-recurrent / looped models** — the "more recurrence = more reasoning depth" thread the paper draws on: Universal Transformers (Dehghani et al., 2018, [arXiv:1807.03819](https://arxiv.org/abs/1807.03819)); ACT (Graves, 2016); "A little depth goes a long way" (Merrill & Sabharwal, 2025); latent-reasoning recurrent depth (Geiping, McLeish et al., NeurIPS 2025). Most directly, **Teaching pretrained LMs to think deeper with retrofitted recurrence** (McLeish et al., 2025) — [arXiv:2511.07384](https://arxiv.org/abs/2511.07384) — by the same second author, the retrofit-recurrence-into-pretrained-models recipe Sleep reuses for Jet/Ouro. The paper's twist on all of these: recurrence is spent on **memory consolidation before eviction**, not at prediction time — so unlike looped models, it adds *zero* wake-time latency.

12. **Sleep-Time Compute** (Lin et al. — Letta, 2025) — [arXiv:2504.13171](https://arxiv.org/abs/2504.13171). Same *motto* ("think offline so inference is cheap"), totally different *level*: it precomputes query-relevant facts as **tokens/prompts** at the agent level. Sleep (this paper) consolidates into **model fast weights** at the architecture level. ~5× less test-time compute at equal accuracy; +13–18% on stateful GSM/AIME.

13. **Do Language Models Need Sleep?** (Lee, McLeish, Goldstein, Fanti — CMU/UMD, 2026) — [arXiv:2605.26099](https://arxiv.org/abs/2605.26099). **This guide.** Wraps a GDN fast-weight update in `N` offline recurrent passes over each context chunk (optimized with Muon), then evicts the KV cache.

**The one-line thesis**: A fixed-state model writes a chunk of context into its state in *one* forward pass — that's a hard cap on how much sequential computation can shape the state before the raw tokens disappear. Sleep removes the cap: loop the blocks `N` times over the same chunk ("sleep") to refine the state, *then* evict KV. The win shows up exactly where one pass isn't enough — deep, multi-step reasoning over evicted context — while answer-token latency stays single-pass.

## What Problem Does Sleep Solve?

You know the two ends of the long-context tradeoff from the long-context and GDN-2 guides:

```
                     Transformer (softmax attn)      Fixed-state SSM (Mamba/GDN)
                     ──────────────────────────      ───────────────────────────
  Memory of past     KV cache: every token kept       ONE state matrix S (fixed)
  Size at len N      O(N) — grows without bound        O(1) — constant
  Recall             exact, lossless                   lossy, superposed
  Decode cost/token  O(N)                              O(1)
  Failure mode       memory/compute blows up           state saturates / blurs
```

The Sleep paper makes a sharper observation about the SSM side. The usual story is that a fixed state fails because it runs out of **capacity** (too many associations crammed into one matrix — the GDN-2 story). Sleep isolates a *different* failure: even when capacity is fixed and adequate, the hybrid degrades as the task needs **more sequential reasoning steps**. That's a **computation bottleneck**, not a capacity bottleneck.

### The single-pass depth cap

Read context once, left to right, and the SSM recurrence threads information forward one position at a time. Within a chunk, that's *one* sweep of computation. A model with `D` layers gets ~`D` "steps" of nonlinear processing to fold the chunk into `S` before the raw KV tokens are evicted. If the answer needs more sequential steps than that — e.g. simulate a cellular automaton 32 generations forward, or chase a 16-hop graph path — one pass can't get there, and once KV is gone the information needed to finish the computation is gone too.

```
  One pass over a chunk:   tokens ─►─►─►─►  S   (then KV evicted)
                           └─ ~D layers of sequential depth, once ─┘

  Sleep = N passes:        tokens ─►─►─►─►  S⁽¹⁾
                           tokens ─►─►─►─►  S⁽²⁾   (reuse same chunk, refine S)
                              ...                   ► effective depth ≈ N·D
                           tokens ─►─►─►─►  S⁽ᴺ⁾  ──► evict KV, keep S⁽ᴺ⁾
```

Sleep buys depth by **reusing the same weights on the same chunk** `N` times. No new parameters, no longer context — just more compute, spent offline, before eviction.

### Why "offline" matters for latency

The payoff is asymmetric in *when* the compute is spent:

```
  Wake (online):   read context  ──►  answer token        ← strict latency budget
                   no CoT tokens, no prediction-time loop, no full-context scan
  Sleep (offline): N passes over each chunk as it fills    ← compute hidden here
```

The answer is still produced in a single forward pass from the consolidated state `S`. The extra `N×` compute lives at chunk-eviction boundaries during ingestion, not at answer time. So you pay for deep reasoning without paying per-token decode latency or chain-of-thought token budget. (Caveat: it preserves *answer-token* latency, not necessarily *end-to-end* latency — sleep adds pauses while ingesting.)

## Key Terms

**Wake / sleep**: Wake = normal forward inference accumulating an attention KV cache. Sleep = an offline consolidation phase triggered when the window fills: loop the blocks over the accumulated chunk, refine the SSM state, then evict KV.

**Sleep duration `N`**: Number of offline recurrent passes over a chunk. `N=1` is the ordinary hybrid (no extra sleep) — the strongest baseline. The paper sweeps `N ∈ {1, 2, 4, 6}` (some experiments {1,2,3,4}).

**Fast weights `S`**: The fixed-size SSM state matrix (`d_k × d_v`), an associative memory holding consolidated history. "Fast" = changes every step from data, vs the slow base weights `θ` that change only during training.

**Hard eviction**: At chunk boundaries, *all* attention KV for that chunk is discarded after the `N` sleep passes. The refined `S` carries forward.

**Sliding-window eviction**: Keep the newest `L−1` KV tokens; evict older ones. The alternative to hard eviction.

**Consolidation rule**: The per-token SSM update (here Gated DeltaNet) that writes context into `S`. The paper argues the *exact* rule isn't essential — the offline-recurrence *schedule* is the contribution.

**Slow weights `θ`**: The base network parameters. Trained offline by gradient descent (backprop runs through all `N` sleep passes). **Not** test-time fine-tuned — this is the key difference from TTT.

## Core Mechanism

### The two memories

```
  Attention KV (K_t, V_t)        SSM fast weight (S_t)
  ─────────────────────────      ──────────────────────────
  grows with context             fixed size  d_k × d_v
  exact recall                   lossy, superposed
  CLEARED at eviction            PERSISTS across chunks
```

Standard attention read (grows with context):
```
q_t = W_Q x_t,   k_t = W_K x_t,   v_t = W_V x_t
o_t = V_tᵀ · softmax(K_t q_t / √d)
```

SSM / fast-weight read+write (fixed size), the Mamba-2 / GDN gated form:
```
  S_t = α_t · S_{t-1}  +  β_t · v_t k_tᵀ          (write: decay old, add new)
  o_t = S_t q_t                                   (read: soft lookup)

  α_t = data-dependent forget gate
  β_t = data-dependent input gate
```
This is the paper's Eq (3), with `α_t ∈ (0,1)` a data-dependent forget gate and `β_t ∈ (0,1)` a data-dependent input gate, both computed from `x_t` (a "gated Hebbian-like outer-product rule"). In the experiments the SSM block is **Gated DeltaNet**, which adds an error-correcting delta-rule correction to this update (see the [GDN-2 guide](../model-architecture/gated-deltanet-2-guide.md) for `S_t = (I − β_t k_t k_tᵀ)α_t S_{t-1} + β_t v_t k_tᵀ`). The paper explicitly states "the specific update rule does not matter for our discussion" — the contribution is the offline-recurrence *schedule*, not the cell.

### Sleep: offline recurrence

For a chunk `c` of length `L`, with base weights `θ`:
```
  h⁽⁰⁾ = Embed(c)
  (h⁽ⁿ⁾, S⁽ⁿ⁾) = Blocks_θ(h⁽ⁿ⁻¹⁾, S⁽ⁿ⁻¹⁾)      for n = 1 … N
  ──────────────────────────────────────────────────────────
  after pass N:  discard h and KV cache;  keep S⁽ᴺ⁾
```

The same stack of blocks runs `N` times over the same chunk. Each pass reads the chunk again and updates the carried state. Training backpropagates through **all** `N` passes (so gradients flow through the recurrent depth — a source of cost and potential instability).

**Cleared vs persisted:**
```
  CLEARED                              PERSISTED
  ──────────────────────────────      ─────────────────────────────────
  all attention KV for the chunk       SSM fast weights S (across chunks,
  refined hidden features h⁽ᴺ⁾          within one sequence)
  (fast weights zero-init per          slow weights θ (trained, frozen
   new sequence)                        at test time, NOT TTT-updated)
```
In sliding-window mode the newest `L−1` KV tokens survive eviction.

### Numerical walkthrough — why a second pass changes the state

Tiny `d_k = d_v = 2`, plain gated update (ignore the delta correction for clarity). Suppose a chunk has two tokens whose write-products are `P₁ = v₁k₁ᵀ` and `P₂ = v₂k₂ᵀ`, constant gates `α = 0.5`, `β = 1` for the trace.

```
  P₁ = ⎡1 0⎤   P₂ = ⎡0 0⎤      start S = 0
       ⎣0 0⎦        ⎣0 1⎦
```

**Pass 1** (sweep both tokens, S carried from 0):
```
  after tok1:  S = 0.5·0 + P₁ = ⎡1 0⎤
                                ⎣0 0⎦
  after tok2:  S = 0.5·⎡1 0⎤ + P₂ = ⎡0.5 0⎤
                       ⎣0 0⎦         ⎣0   1⎦
```
End of pass 1: `S⁽¹⁾ = [[0.5,0],[0,1]]`. Token 1's contribution has already decayed once (0.5) by the time token 2 was written — early context is faded relative to late context. One pass bakes in this recency skew.

**Pass 2** (the sleep loop: same two tokens again, but now S starts from `S⁽¹⁾`, *not* zero):
```
  after tok1:  S = 0.5·⎡0.5 0⎤ + P₁ = ⎡1.25 0⎤
                       ⎣0   1⎦         ⎣0    0.5⎦
  after tok2:  S = 0.5·⎡1.25 0  ⎤ + P₂ = ⎡0.625 0  ⎤
                       ⎣0    0.5⎦         ⎣0     1.25⎦
```
End of pass 2: `S⁽²⁾ = [[0.625,0],[0,1.25]]`. Token 1's binding is now stronger relative to where one pass left it, because the second sweep re-injected it on top of an already-populated state. The state is no longer a function of a single left-to-right sweep — it's a *fixed-point-style refinement*. With the real nonlinear blocks (and the delta rule), each extra pass lets information from late tokens influence how early tokens are written and vice-versa — i.e. **sequential computation depth the single pass can't express**. That's the mechanism behind the depth gains below.

## Sleep Duration as a Scaling Knob

`N` is a new axis, orthogonal to params and data: hold the model and the token budget fixed, spend more *consolidation* compute. The empirical signature is consistent across all three tasks — **gains concentrate on the hard, deep examples and are ~flat on easy ones**:

```
  task difficulty ──►
  shallow  ┤ N=1 already saturates ─ extra sleep barely helps
  medium   ┤ N=2 unlocks it
  deep     ┤ needs N=4–6, and N=1 is near-random
```

This is the test-time-compute story you know from o1/BoN, but moved to **ingestion time** and **inside the state** rather than spent on output tokens. More sleep = more effective depth = more sequential reasoning steps the model can complete before KV is gone.

## Architectures Tested

```
  Task                Model                         Notes
  ──────────────────  ────────────────────────────  ─────────────────────────────
  Cellular automaton  4-layer attn-GDN hybrid, d=256  from scratch, ~5B tokens
  Depo (graph hops)   10-layer hybrid, d=512          from scratch
  GSM-Infinite        Jet-Nemotron 2B                 loops middle 14 of 28 blocks
  GSM-Infinite        Ouro 1.4B                       looped-LM + Jet SSM layers
```

The big-model runs are **fine-tuned from pretrained hybrids** (Jet-Nemotron, Ouro), not trained from scratch — Sleep is retrofitted by inserting/looping SSM blocks and continuing training, which is why it's cheap (1–2 H100 GPU-days per run; automaton runs <1 A6000 GPU-day).

## Benchmarks

All comparisons hold model, context length `L`, and answer-time compute fixed; only `N` varies. The strongest baseline is `N=1` (same architecture, no extra sleep). Accuracy ↑.

### Cellular automaton (Rule 110, t-step rollout)

Four 24-bit strings, hard eviction every `L=24`, 96 context tokens + 4 labels. Exact-match accuracy at the hardest depth `t=32`, ~5B training tokens:

```
  Sleep N    t=32 accuracy
  ───────    ─────────────
  N=1        ~10%   (near random)
  N=2        ~20%
  N=3–4      >30%
```
The cleanest demonstration: identical budget, identical model, and *only* more sleep passes lift a near-random model to 3× chance. Rule-110 rollout is the perfect probe — it's intrinsically sequential (`t` generations = `t` dependent steps), so it directly measures usable computation depth.

### Depo (k-hop graph retrieval)

Cycles up to 75 nodes, graph padded to 300 tokens, 10 QA pairs, total `T=360`, window `L=75`. Train `k∈[1,16]`, eval `k∈{1,2,4,8,16}`. Result is a **depth frontier**, not a single number:

```
  N=1   stalls at 4-hop and beyond
  N=2   stalls at 8-hop and beyond
  N=4   begins solving 16-hop
```
Each doubling of sleep pushes the solvable hop-count out — concrete evidence that `N` buys reasoning depth.

### GSM-Infinite (math, no CoT)

1,600 held-out problems, 2,000–3,300 tokens, #operations sampled from [1,8], **answer in one prediction pass (no chain-of-thought tokens)**, window `L=2000`.

```
  Model            N      6-op acc        8-op acc
  ───────────────  ─────  ──────────────  ──────────────
  Jet-Nemotron 2B  1→6    0.742 → 0.812   0.351 → 0.388
  Ouro 1.4B        1→4    0.419 → 0.615   0.210 → 0.272

  Sliding-window, Ouro 1.4B, L=512:
                   N      2-op acc
                   1→4    0.596 → 0.905    (+helps 4/6/8-op too)
```
The Ouro sliding-window 2-op jump (0.596 → 0.905) is the largest single gain — and note it's *not* a deep-reasoning case, suggesting that under aggressive eviction (`L=512`) even shallow problems become consolidation-limited, and sleep recovers them.

### Cost

```
  Throughput:  1×H200, seq 12k, FlashAttention-2 → cost ≈ inverse in N
               (training cost grows ~linearly with sleep passes)
```
Recurrent processing across windows reduces sequence-axis parallelism unless the window is large enough to saturate the GPU — so big windows amortize sleep better.

## Sleep vs Alternatives

```
                     What's reused        Updated at        Memory of      Latency at
                     for extra compute    test time?        past           answer time
─────────────────── ──────────────────── ───────────────── ────────────── ─────────────
CoT / o1            output tokens         no (just decode)  KV cache O(N)   slow (many tok)
Sleep-time (Letta)  precomputed prompts   no (prompt-level) KV cache        fast
TTT layers          gradient steps on     YES (grad on      fixed state     single-pass
                    self-sup loss          hidden state)
Sleep (this paper)  forward passes over   no (forward only, fixed state S   single-pass
                    the chunk (×N)         θ frozen)         (KV evicted)
```

The two sharpest contrasts:

- **vs TTT**: TTT updates the state by running *gradient descent* on a self-supervised objective at test time (`W_t = W_{t-1} − η∇ℓ`). Sleep runs the *learned forward update rule* repeatedly — no test-time gradients, no auxiliary loss. The "learning to consolidate" is baked into `θ` during training; at inference it's pure forward recurrence.
- **vs Sleep-Time Compute (Letta)**: same slogan, different layer. Letta anticipates queries and precomputes **text** offline (an agent/prompt technique, model-agnostic). This paper consolidates into **fast weights** (an architecture technique). They compose — you could do both.

## Practical Considerations

**When Sleep is worth it:**
- Fixed-state / hybrid models under **aggressive KV eviction**, where context won't fit and the task needs multi-step reasoning over evicted material.
- Hard latency budgets at answer time: no room for CoT tokens or prediction-time loops, but you *can* afford compute while ingesting context.
- Retrofit: insert/loop SSM blocks into a pretrained hybrid (Jet-Nemotron, Ouro) and continue-train — cheap (1–2 GPU-days).
- Large windows that saturate the GPU, so the `N×` cost amortizes.

**When it isn't:**
- Context already fits in KV — full attention recalls it exactly; sleep is wasted compute.
- Shallow tasks where `N=1` already saturates — no headroom.
- Exact-copy / high-entropy recall: a fixed state is lossy by construction ([recall-throughput tradeoff, 2402.18668](https://arxiv.org/abs/2402.18668); [copying limits, 2402.01032](https://arxiv.org/abs/2402.01032)). Sleep adds depth, not capacity.
- End-to-end latency-critical streaming: sleep pauses at eviction boundaries even though answer tokens stay fast.
- **Query-after-context** setups: if the query isn't known during ingestion, consolidation must be query-agnostic (harder). The paper's GSM-Infinite puts the question *before* the long context, enabling selective consolidation — a setup choice worth noting before you generalize the numbers.

**Experiment-design notes (from the paper's protocol):**
- Mask the loss: consolidation/sleep passes get **zero loss**; only prediction chunks compute CE.
- Sweep `N ∈ {1,2,4}` first; `N=1` is the must-beat baseline.
- Hold `L`, total context, data order, and answer-time compute fixed — otherwise gains blur into "longer context" or "slower decode."
- Report by **reasoning depth** (rollout `t`, hop `k`, #ops), never just aggregate accuracy — the whole effect is depth-conditional.
- Test both hard and sliding-window eviction. Hard eviction helped during SSM warm-up, because sliding-window attention can leave freshly-inserted SSM layers under-trained.

## Connection to Your Prior Knowledge

- **vs your GPT-2 124M**: your model caches K,V for all 1024 positions and attends over them every step. Sleep replaces the long-range memory with a fixed `S` and, instead of one ingestion pass, loops the blocks `N` times over each window before dropping KV. Training objective (next-token CE) is unchanged; you'd add a sleep loop + loss mask to the ingestion path.
- **vs the GDN-2 guide**: GDN-2 attacks the *capacity* failure of a fixed state (better erase/write gates so more associations fit). Sleep attacks the *computation* failure (more passes so deeper reasoning fits). Orthogonal — GDN-2 is literally the cell Sleep loops over; you could stack both.
- **vs test-time compute (o1/BoN, your reasoning-models study)**: same "spend compute to think harder" idea, but moved from *output tokens at decode* to *forward passes at ingestion*, and from activations to state. No verifier, no search — just recurrence depth.
- **vs TTT / your sense of "online learning"**: TTT = gradient descent at test time. Sleep = repeated forward application of a *learned* update. Cheaper and more stable per pass, but the consolidation ability is fixed by training rather than adapted per input.
- **vs distillation / DPO / GRPO**: orthogonal post-training methods. Sleep is an inference-time architecture/schedule; you'd still SFT/DPO/GRPO a Sleep model exactly as usual.

## Summary

- **Problem**: Fixed-state hybrids escape the Transformer's growing KV cache, but writing a chunk into the state in *one* pass caps how much sequential reasoning can shape it before KV is evicted — a **computation** bottleneck distinct from the **capacity** bottleneck GDN-2 targets.
- **Idea**: "Sleep" = loop the blocks `N` times over each context chunk offline, refining the SSM fast-weight state `S`, then evict the KV cache. Effective depth ≈ `N × layers`; answer-token prediction stays single-pass.
- **Mechanism**: `(h⁽ⁿ⁾, S⁽ⁿ⁾) = Blocks_θ(h⁽ⁿ⁻¹⁾, S⁽ⁿ⁻¹⁾)` for `n=1…N` over a Gated DeltaNet cell `S_t = α_t S_{t-1} + β_t v_t k_tᵀ`; backprop through all passes; `S` persists across chunks, KV and slow weights do not get test-time-updated (≠ TTT).
- **Results** (2B and below): Rule-110 t=32 ~10%→>30% (N=1→4); Depo solvable-hop frontier pushes 4→8→16 as N doubles; GSM-Infinite Jet-Nemotron 2B 6-op 0.742→0.812, Ouro 1.4B 6-op 0.419→0.615; sliding-window Ouro 2-op 0.596→0.905. Gains concentrate on deep examples; cost ≈ linear in N.
- **Lineage**: fast weights (Schmidhuber '92, Ba '16) + linear-attention-as-fast-weights (Schlag '21) + SSMs (S4/Mamba/Mamba-2) + GDN cell; cousins are TTT (test-time *gradients*) and Letta sleep-time compute (offline *prompts*). Neuroscience grounding: complementary learning systems — hippocampus replays to neocortex during sleep.
- **Status**: VERY_NEW (May 2026), no public code, no replication, ≤2B params, query-before-context setup. A clean existence proof that *more consolidation compute*, not more parameters or capacity, is a usable lever for deep reasoning under KV eviction.

## Key Papers

| Paper | Authors | Year | Contribution |
|-------|---------|------|--------------|
| [Learning to Control Fast-Weight Memories](https://people.idsia.ch/~juergen/fastweights/ncfastweightsrev.html) | Schmidhuber | 1992 | Fast weights as a separate short-term memory tier |
| [Using Fast Weights to Attend to the Recent Past](https://arxiv.org/abs/1610.06258) | Ba, Hinton et al. (Toronto/DeepMind) | 2016 | Outer-product fast weights ≈ attention to recent past |
| [Transformers are RNNs (Linear Attention)](https://arxiv.org/abs/2006.16236) | Katharopoulos et al. | 2020 | Kernelized linear attention → O(N) fixed-state recurrence |
| [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174) | Schlag, Irie, Schmidhuber | 2021 | Linear attention = additive outer-product fast weights |
| [S4](https://arxiv.org/abs/2111.00396) | Gu, Goel, Ré (Stanford) | 2021 | Structured SSM; linear-time long-range modeling |
| [Mamba](https://arxiv.org/abs/2312.00752) | Gu, Dao (CMU/Princeton) | 2023 | Selective SSM; linear-time rival to Transformers |
| [Transformers are SSMs (Mamba-2/SSD)](https://arxiv.org/abs/2405.21060) | Dao, Gu | 2024 | State-space duality; the `α S + β vkᵀ` update form |
| [DeltaNet parallelization](https://arxiv.org/abs/2406.06484) | Yang et al. | 2024 | Chunkwise-parallel delta-rule writes |
| [Gated DeltaNet](https://arxiv.org/abs/2412.06464) | Yang, Kautz, Hatamizadeh (NVIDIA) | 2024 | Decay gate + delta rule — the SSM cell Sleep loops over |
| [Test-Time Training layers](https://arxiv.org/abs/2407.04620) | Sun et al. (Stanford) | 2024 | Hidden state = model updated by test-time gradient steps (TTT origin) |
| [End-to-end TTT for long context](https://arxiv.org/abs/2512.23675) | Tandon et al. | 2025 | SWA + one test-time gradient step on MLP layers; paper's TTT contrast |
| [Retrofitted recurrence](https://arxiv.org/abs/2511.07384) | McLeish et al. | 2025 | Add depth-recurrence to pretrained LMs — the Jet/Ouro retrofit recipe |
| [Sleep-Time Compute](https://arxiv.org/abs/2504.13171) | Lin et al. (Letta) | 2025 | Offline precompute of query-relevant prompts (agent level) |
| [Do Language Models Need Sleep?](https://arxiv.org/abs/2605.26099) | Lee, McLeish, Goldstein, Fanti (CMU/UMD) | 2026 | **This guide** — N offline recurrent passes consolidate context into fast weights before KV eviction |
