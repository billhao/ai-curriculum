# Scaling Laws, Capacity, and Knowledge vs Skills

*A case study built around LFM2.5-230M (Liquid AI, released June 2026) vs your own GPT-2 124M (nanoGPT). It walks from architecture, to why a tiny model is over-trained ~4,000× past Chinchilla, to what that data actually buys — and ends on the knowledge-vs-skills distinction that ties it all together.*

---

## Background & research lineage

This guide stitches together several research threads:

- **Compute-optimal scaling** — Kaplan et al. 2020 ([2001.08361](https://arxiv.org/abs/2001.08361)) → Chinchilla / Hoffmann et al. 2022 ([2203.15556](https://arxiv.org/abs/2203.15556)) → the Epoch AI replication ([2404.10102](https://arxiv.org/abs/2404.10102)) → over-training reliability, Gadre et al. 2024 ([2403.08540](https://arxiv.org/abs/2403.08540)).
- **Inference-aware scaling** — Sardana & Frankle, *Beyond Chinchilla-Optimal* (2023).
- **Knowledge capacity** — Allen-Zhu & Li, *Physics of Language Models Part 3.3* ([2404.05405](https://arxiv.org/abs/2404.05405)).
- **Skills as circuits** — induction heads (Olsson et al. 2022), grokking (Power et al. 2022 [2201.02177]; Nanda et al. 2023 [2301.05217]), reasoning (Allen-Zhu Part 2.1 [2407.20311]).
- **Architecture** — LFM2 technical report ([2511.23404](https://arxiv.org/html/2511.23404v1)); the model card [LiquidAI/LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M).

The anchor model, **LFM2.5-230M**, is interesting precisely because it is an *extreme* design: ~230M parameters trained on **19 trillion tokens** (~82,600 tokens/param), built to run on a phone. Comparing it to your GPT-2 124M (the classic 2019 dense-transformer recipe) makes every modern design choice legible.

---

## 1. Architecture: your GPT-2 124M vs LFM2.5-230M

Your GPT-2 124M is a **pure dense pre-LN transformer**: 12 identical blocks, each full multi-head attention + GELU MLP. LFM2.5-230M is a **hybrid** that keeps only a minority of attention and replaces the rest with cheap convolution, plus every post-2019 upgrade.

| | GPT-2 124M (yours) | LFM2.5-230M |
|---|---|---|
| Total params | 124.4M | 229.7M (~1.85×) |
| Layers (sequence-mixers) | 12 | 14 = **8 conv + 6 attention** |
| Sequence mixing | 12× full MHA | 8× gated short-conv + 6× GQA |
| Hidden dim | 768 | 1024 |
| FFN | 3072, GELU (4× dense) | 2560, **SwiGLU** (gated) |
| Heads / KV heads | 12 / 12 (MHA) | 16 / 8 (**GQA**, 2:1) |
| Head dim | 64 | 64 |
| Positions | learned absolute | **RoPE** (θ=1e6) |
| Norm | LayerNorm | **RMSNorm** + QK-norm |
| Vocab | 50,257 | 65,536 |
| Context | 1,024 | 32,768 (config says 128K*) |
| Embeddings | tied | tied |
| Training tokens | ~10B (WebText) | **19T** (~1,900×) |

Where the 230M params live (note tying makes the embedding double as the LM head):

```
 Token embedding   65,536 × 1,024 = 67.1M   (29% of all params! tied → 0 extra for LM head)
 8 conv blocks     8 × 12.06M     = 96.5M
 6 GQA blocks      6 × 11.01M     = 66.1M
 ───────────────────────────────────────────
 Total ≈ 229.7M  =  embedding 67.1M + backbone 162.6M
```

### The four ideas you didn't have in GPT-2

1. **Hybrid conv/attention backbone (the headline).** Only **6 of 14 layers do attention**; the other 8 are gated depthwise causal convolutions, kernel length 3 (`in_proj 1024→3072`, gate, `out_proj 1024→1024`). Conv = cheap *local* mixing with **no KV cache**. Attention is rationed for *global* mixing. Layer order: `C C A C A C A C A C A C A C`.
2. **GQA** — 16 query heads share 8 K/V heads → half the KV cache of MHA.
3. **RoPE** (relative, extrapolates → 32K context) instead of a learned absolute table (hard-capped at 1024 in your GPT-2).
4. **SwiGLU FFN + RMSNorm** — gated FFN (3 matrices) lets the intermediate dim be *smaller* (2.5× vs your 4×) while improving quality; RMSNorm + QK-norm for stability.

---

## 2. Why LFM2.5 is faster (the latency claim)

Liquid's headline: **~2× faster decode *and* prefill than comparable transformers on CPU, smallest memory footprint in class** (e.g. LFM2-700M vs Qwen3-0.6B: 1.6–2.5× prefill, 1.4–2× decode; LFM2.5-230M hits 213 tok/s on a Galaxy S25 Ultra, 42 tok/s on a Raspberry Pi 5). The backbone was chosen by **hardware-in-the-loop NAS** optimizing latency + peak memory on Snapdragon CPUs — not FLOPs. Two regimes explain it:

**Decode (token-by-token) is memory-bandwidth-bound on edge.** Each new token re-reads weights + the full KV cache; bandwidth is the wall.

```
 Full-attention model : every layer holds a KV cache that GROWS with context
                        → per-token traffic ∝ n_layers × context_len
 LFM2.5-230M          : only 6 of 14 layers cache K/V (and GQA → 8 KV heads);
                        the 8 conv layers keep a FIXED state = last 3 tokens
                        → KV traffic ≈ 1/3, and grows far slower with context
```

KV cache ≈ **12 KB/token** vs your GPT-2's ~36 KB/token. This is why it streams on a Pi.

**Prefill (digest the prompt) is compute-bound — and the gap is even bigger here.** Attention is **O(n²·d)** in prompt length; a length-3 conv is **O(n·k·d) — linear**. Replacing 8 of 14 mixers with conv removes most of the quadratic term.

Plus an operator/CPU-fit bonus: a short depthwise conv is a regular, SIMD-friendly op, while attention's softmax over a growing KV cache has irregular, bandwidth-hungry access.

> Caveat for *your* comparison: LFM2.5-230M is *bigger* than your 124M (more weight bytes per step), so at tiny context your GPT-2 could even win on raw ms/token. The LFM win is **relative to comparable-quality models and grows with context length**.

---

## 3. Why 19T tokens? Chinchilla is the wrong law here

Chinchilla says compute-optimal ≈ **20 tokens/param**. For 230M that's ~4.6B tokens. LFM2.5 used **19T = ~4,100× more** (~82,600 tok/param). That is deliberate, and now standard for small models (Llama-3 8B used 15T ≈ 94× over).

**Chinchilla optimizes the wrong objective for this model.** It minimizes *training* loss for a *fixed training-compute budget when N is free to choose*. Both premises fail:

1. **N is fixed by deployment** — it *must* be ~230M to fit a phone and hit 200+ tok/s. Once N is locked, the only knob left is data.
2. **Inference cost dominates** — a tiny edge model is run for *trillions* of inference tokens; over-spending on training is trivially worth it. This is *Beyond Chinchilla-Optimal: Accounting for Inference* (Sardana & Frankle): charge for inference and the optimum shifts to **smaller models on far more data**.

### The "compute twin" — the argument that makes it click

Training FLOPs ≈ 6ND:

```
 6 × 230e6 × 19e12 ≈ 2.6e22 FLOPs   (LFM2.5's training bill)

 Chinchilla-optimal use of the SAME 2.6e22 budget (C = 120 N²):
   N ≈ 15B params,  D ≈ 300B tokens
```

So at **identical training compute**, Chinchilla says build a **15B model on 300B tokens**. Liquid instead spent it on **230M + 19T** — a model ~64× smaller that *fits on a phone*.

```
 Chinchilla-optimal (15B / 300B) : lower loss, but USELESS on-device
 Liquid's choice    (230M / 19T) : higher loss, but RUNS at 213 tok/s on a phone
                                   ── same training bill ──
```

It's a **constrained** optimization: minimize loss *subject to* N ≤ deployable size. When N is forced tiny, the math says "pour all your compute into data" — and you can afford to, because a small model is cheap per token (2.6e22 FLOPs is ~1/20th of the original Chinchilla-70B run). **Small model × huge data is still a small total bill.**

---

## 4. What the loss curve looks like in the over-trained tail

No one (Liquid included) has published a clean loss-vs-tokens curve anywhere near 82,600 tok/param, so the 230M@19T curve is *extrapolation*. But every published over-trained run extrapolates the same way: **loss keeps falling, smoothly, no wall — toward a floor set by parameter count.**

The governing form (Chinchilla):

```
 L(N, D) = E + A/N^α + B/D^β
           ───   ───────   ───────
        language CAPACITY    DATA term
         entropy floor       (β<1, what
        (irreducible) (FIXED  tokens buy
                       once N  down, with
                       fixed)  diminishing
                               returns)
```

With **N fixed**, `A/N^α` is frozen — data only shrinks `B/D^β`. By 19T that term is nearly exhausted, so the curve **asymptotes to `E + A/N^α`**: a floor the *parameters*, not the data, decide.

Evidence (all consistent, no saturation):

| Model | tok/param | Observation |
|---|---|---|
| Llama-3 8B/70B | ~1,875 | "continued to improve **log-linearly** up to 15T" |
| Gadre et al. | 20–640 | over-trained loss stays power-law predictable |
| MAP-Neo 7B | ~640 | loss falls *faster* than Chinchilla; fit extra −d·logD, d≈0.01–0.03 |
| SmolLM2-1.7B | ~6,500 | MMLU 29.6→48.9, GSM8K 4.3→32.6 (mostly late) |
| TinyLlama 1.1B | ~2,700 | commonsense 46.1→53.9, noisy tail |

> Two honesty caveats: (1) the famous "no saturation past 2T" line is **Llama 2's**, not TinyLlama's. (2) Big late jumps (SmolLM2 GSM8K 4→33) are mostly **curriculum / data-mix** (math/code injection, annealing), *not* raw repetition. "More data helps" is partly "more *diverse* data + better curriculum helps."

---

## 5. How reliable is that floor, `E + A/N^α`?

**Two-tier verdict: excellent *relative* predictor, rough *absolute* floor (~5–10%).**

The canonical floor was literally **mis-fit by a software bug.** The Epoch AI replication ([Besiroglu et al. 2024](https://arxiv.org/abs/2404.10102)) found — and Hoffmann *acknowledged* — that the original parametric fit (a) **averaged instead of summed** the Huber losses, so L-BFGS stopped early with wrong params, and (b) reported confidence intervals **~50× too tight** (the CI on exponent `a` was 0.454–0.455, a width needing ~600k runs; they ran ~400–500).

| | Original (2022) | Corrected (Epoch 2024) |
|---|---|---|
| Fit | 1.69 + 406.4/N^0.34 + 410.7/D^0.28 | 1.82 + 482/N^0.35 + 2085/D^0.37 |
| E (floor at N=∞) | 1.6934 | 1.8172 (**+7.3%**) |
| β | 0.2849 | 0.366 (moved most) |
| Floor at **N=230M** | **2.285 nats** | **2.413 nats** (+5.6%) |

`E` shifted 1.69→1.82 **on the same data** just from fixing the fit — so it's a **fitting nuisance parameter, not a measured "entropy of text."** `α, β` are likewise **not universal** (dataset/tokenizer-dependent).

What *is* reliable: the **relative** law. Gadre predicted over-trained loss (a model trained 32× past optimal) from experiments costing **300× less compute, to 0.7% error**.

**So:** `E + A/N^α` is a good *conceptual decomposition* and a good *relative/interpolation* predictor — but don't read the exact floor number as better than ~10%, and don't treat it as a physical constant.

---

## 6. The knowledge-capacity ceiling

How much world knowledge can 230M params hold *after 19T tokens*? Its own scorecard answers it:

```
 KNOWLEDGE / REASONING (weak)        SKILLS (strong)
 MMLU-Pro      20.25  (random 10)    IFEval     71.71   ← instruction following
 GPQA-Diamond  25.41  (random 25!)   BFCLv3     43.26   ← tool calling
                ↑ literally random    IFBench    38.40
```

vs the field (MMLU-Pro, since LFM reports it): LFM2.5-230M 20.25 → Qwen2.5-0.5B ~15.7 → Llama-3.1-8B ~48 → Llama-3.1-405B **73.3** → frontier (GPT-5.x / Gemini-3 / Claude Fable 5) **~87–91**. The 230M model closes only **~16%** of the random→405B headroom.

**Why it's a hard ceiling, not a training shortfall — the 2-bits-per-parameter law.** Allen-Zhu & Li, *Knowledge Capacity Scaling Laws* ([2404.05405](https://arxiv.org/abs/2404.05405)), measured that a well-trained transformer stores **~2 bits of factual knowledge per parameter** — a constant *independent of how much data you train on*.

```
 Factual storage = 2 bits × params
 LFM2.5-230M   2 × 230M  = 460 Mbit ≈  57.5 MB      ← hard ceiling
 Llama-3.1-8B  2 × 8B     = 16 Gbit  ≈   2 GB
 frontier 405B 2 × 405B   = 810 Gbit ≈ 101 GB       (~1,760× more room)
```

Conditions on the 2-bit figure: it's a *peak* after ~1000 exposures/fact (100 exposures → ~1 bit); holds at int8, drops to ~0.7 at int4; junk-heavy data degrades usable capacity. So **knowledge is capacity-bound (∝ params), not data-bound.** Train 230M on 19T or 190T — once the ~57 MB shelf is full, new facts have nowhere to go.

---

## 7. Knowledge vs Skills — the unifying idea

If knowledge is capacity-capped, why over-train at all? Because the data buys **skills**, and skills are a different kind of thing.

```
 KNOWLEDGE (facts)            │ SKILLS (algorithms / circuits)
 ────────────────────────────┼──────────────────────────────────
 ≈ incompressible            │ highly compressible / reused
 N facts → N×bits, no reuse  │ 1 circuit → ∞ inputs, paid once
 stored & looked up          │ written once, runs on any input
 = DATA                      │ = CODE
 priced at ~2 bits/param     │ NO known bits/param law
```

**Both consume parameters** — the difference is *amortization*. A fact is paid for once per fact; a skill (e.g. an induction head doing `[A][B]…[A]→[B]`) is paid for once and reused on every input. The **2-bit law is facts-only** (it measures memorized tuples); the reasoning papers measure skill as *generalization*, with no bit-budget.

Mechanistic backing that "skills = circuits":
- **Induction heads** (Olsson et al.): a reusable in-context copy circuit that forms in a sharp phase change (visible loss bump) and drives in-context learning.
- **Grokking** (Power 2022; Nanda 2023): models memorize first, then snap to a **compact generalizing algorithm** (modular addition becomes a Fourier-rotation circuit) — *smaller* than the lookup table it replaces; weight decay drives the compression.
- **Reasoning** (Allen-Zhu Part 2.1, [2407.20311](https://arxiv.org/abs/2407.20311)): on synthetic grade-school math, models learn genuine reasoning that *generalizes to longer problems than trained on*, **plan ahead**, and do "mental computation" — depth matters more than width.

**This is exactly why LFM2.5-230M scores IFEval 71.7 but MMLU-Pro 20.** The 57.5 MB cap is on the *facts shelf*; instruction-following and tool-formatting are cheap reusable circuits that fit fine in 230M.

### Does it really need 19T tokens for skills?

Mostly **no** — and the truth is subtler than "skills need data":

- **Small models are sample-*inefficient*** (Kaplan 2020): bigger models extract more skill per token, so a 230M model needs *many* exposures to form the same circuit. Much of the 19T is a **sample-inefficiency tax for being tiny.**
- **Skills mostly rise *smoothly*** (Schaeffer 2023, *Are Emergent Abilities a Mirage?*): apparent "sudden skills" are largely metric artifacts; on smooth metrics, skill climbs gradually with data.
- **Simple skills saturate early** — LIMA: 1,000 examples ≈ strong instruction-following *if the base is good*. **Complex skills need diverse data/curriculum** — SmolLM2 math 0%→10%→14% came from injecting math/code corpora, not generic web volume.

What the 19T actually funds:

```
 NOT facts          → capacity-capped at ~57 MB; data can't beat the param ceiling
 sample-inefficiency → tiny model needs many passes to form each circuit   ← big
 breadth coverage    → 10 languages × code × tool-formats × extraction      ← big
 circuit sharpening  → grokking-style skills need prolonged training
 marginal knowledge  → a few more facts cross the ~1000-exposure threshold
```

---

## The one-paragraph takeaway

A 230M edge model trained on 19T tokens looks absurd by Chinchilla, but Chinchilla answers the wrong question. When deployment **fixes the model size**, you optimize *quality at that size over a lifetime of inference*, and the math says pour everything into data. That data cannot buy **knowledge** (capacity is ~2 bits/param → ~57 MB, a hard wall set by parameters, which is also the `A/N^α` floor the loss curve flattens toward) — but it does buy **skills**, which are reusable circuits that amortize across all inputs and so fit even in a tiny model. The result is a model that is near-random on world knowledge yet genuinely capable at instruction-following, extraction, and tool-use: **a skills engine, not a knowledge store.** Your GPT-2 124M is the same machinery one generation back — uniform full attention, learned positions, 10B tokens — and seeing what changed (rationed attention, RoPE, SwiGLU, GQA, and 1,900× the data) is a compact tour of everything the field learned since 2019.
