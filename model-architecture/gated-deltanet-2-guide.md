# Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention

How NVIDIA split the single delta-rule gate into two independent channel-wise gates — one for erasing stale memories, one for writing new ones — and why that small change wins most on long-context retrieval.

> Confidence: **VERY_NEW**. The paper (arXiv 2605.22791) was submitted **21 May 2026**, ~5 days before this guide. Equations and benchmark numbers below are verified against the arXiv HTML, the official `NVlabs/GatedDeltaNet-2` source code, and an independent GPT-5.4 web pass — but this is a fresh technical report, not a settled peer-reviewed result. Points that conflict across sources are flagged inline.

## Background

**Originating paper**: [Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention](https://arxiv.org/abs/2605.22791) (Ali Hatamizadeh, Yejin Choi, Jan Kautz — NVIDIA, May 2026). Code: [github.com/NVlabs/GatedDeltaNet-2](https://github.com/NVlabs/GatedDeltaNet-2) (NVIDIA non-commercial license).

**Research lineage** — GDN-2 is the latest node in the "linear attention as fast-weight associative memory" line, which has been racing the Transformer on long-context efficiency:

1. **Linear Transformers** (Katharopoulos et al., Idiap/EPFL, 2020) — [Transformers are RNNs](https://arxiv.org/abs/2006.16236). Replaced `softmax(QKᵀ)V` with a kernel feature map `φ(Q)(φ(K)ᵀV)`, turning attention into a linear recurrence with a fixed-size state. O(N) instead of O(N²), but no forgetting — the state just accumulates.

2. **Fast Weight Programmers / DeltaNet** (Schlag, Irie, Schmidhuber, 2021) — [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174). Reframed the state as a "fast weight" matrix and introduced the **delta rule** (Widrow-Hoff) for writing to it: error-correct the memory instead of blindly adding. This is the conceptual root of GDN-2.

3. **GLA — Gated Linear Attention** (Yang et al., MIT/IBM, 2023) — [arXiv:2312.06635](https://arxiv.org/abs/2312.06635). Added a data-dependent gate to linear attention and gave a hardware-efficient chunkwise training kernel.

4. **Mamba** (Gu & Dao, CMU/Princeton, 2023) — [arXiv:2312.00752](https://arxiv.org/abs/2312.00752). Selective state-space model with input-dependent decay; first linear-time architecture to seriously rival Transformers at scale.

5. **DeltaNet parallelization** (Yang et al., 2024, NeurIPS) — [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484). Gave the chunkwise WY-representation algorithm that made delta-rule models trainable at scale (the delta rule is sequential by nature; this parallelizes it).

6. **Mamba-2 / SSD** (Dao & Gu, 2024, ICML) — [Transformers are SSMs](https://arxiv.org/abs/2405.21060). Showed SSMs and attention are duals; the "state-space duality" framework. The main SSM baseline GDN-2 beats.

7. **Gated DeltaNet (GDN)** (Yang, Kautz, Hatamizadeh — NVIDIA, 2024; ICLR 2025) — [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464). Added a **scalar decay gate** `α_t` on top of the delta rule — adaptive forgetting + error-correcting writes. GDN-2's direct predecessor (same NVIDIA authors).

8. **RWKV-7 "Goose"** (Peng et al., 2025) — [arXiv:2503.14456](https://arxiv.org/abs/2503.14456). Expressive dynamic state evolution; popularized **negative state-transition eigenvalues** for extra state-tracking capacity — an idea GDN-2 borrows as an option.

9. **Kimi Linear / KDA** (Moonshot AI, 2025) — [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692). Introduced **Kimi Delta Attention (KDA)**: upgrades GDN's scalar decay to a **channel-wise (diagonal) decay**. The other half of GDN-2's parentage.

10. **Mamba-3** (ICLR 2026) — [arXiv:2603.15569](https://arxiv.org/abs/2603.15569). Latest SSM; SISO and MIMO variants. The toughest SSM baseline in GDN-2's tables.

11. **Gated DeltaNet-2 (GDN-2)** (Hatamizadeh, Choi, Kautz — NVIDIA, 2026) — [arXiv:2605.22791](https://arxiv.org/abs/2605.22791). **This guide.** Takes GDN's adaptive forgetting + KDA's channel-wise decay, and removes the one thing they shared: a single scalar gate that tied "how much to erase" to "how much to write."

**The one-line thesis**: GDN and KDA used a single scalar `β_t` to control two physically distinct operations — erasing old key→value associations and writing new ones. GDN-2 splits `β_t` into two independent **channel-wise** gates (`b_t` for erase, `w_t` for write). That decoupling is the entire paper, and it wins most exactly where you'd predict: long-context retrieval, where a fixed-size memory must keep many competing associations distinct.

## What Problem Does GDN-2 Solve?

### The fixed-state bottleneck

You studied the quadratic attention wall in the long-context guide: a Transformer keeps **every** past token's K and V verbatim in the KV cache, so memory and per-token cost grow with sequence length. At 1M tokens, Llama-3-70B's KV cache alone is ~167 GB — three H100s just to hold it.

Linear attention makes the opposite bet. It compresses *all* history into a **single fixed-size state matrix** `S_t ∈ ℝ^(d_k × d_v)` and throws the per-token KV pairs away:

```
                     Transformer (softmax attn)        Linear attention (GDN-2)
                     ────────────────────────────      ──────────────────────────
  Memory of past     KV cache: every token's K,V        ONE state matrix S_t
                     kept verbatim                       (d_k × d_v), fixed size
  Size at seq len N  O(N) — grows without bound          O(1) — constant
  Read cost / token  O(N) — attend to all past           O(d_k·d_v) — constant
  Lossy?             No — exact recall of any token      Yes — history is superposed
  Decode at 1M tok   ~167 GB KV (Llama-70B)              same few hundred KB as at 1K
```

The state `S_t` is an **associative memory**: it stores key→value bindings as a sum of outer products, and you read it back by matrix-vector product. The catch is in the "Lossy?" row. A `128×128` state can hold only so many distinct associations before they start interfering — multiple key→value pairs are literally added into the same matrix. This is the central tension of every linear-attention model: **how do you write new associations and forget stale ones without corrupting the ones you still need?**

### The associative-memory view

Think of `S_t` as a key-addressed memory. To read the value bound to query `q`:

```
o_t = S_tᵀ q_t          (matrix-vector product — a "soft lookup")
```

To write a new binding k→v, the simplest rule (plain linear attention) just adds the outer product:

```
S_t = S_{t-1} + k_t v_tᵀ
```

But pure accumulation never forgets and never corrects. If key `k` was already bound to an old value `v_old`, writing `k→v_new` leaves *both* in superposition — reads at `k` return a blurry mix. As context grows, blur compounds. Two mechanisms fix this:

- **Decay / forgetting** (Mamba, GLA, GDN): multiply the old state down each step so stale content fades.
- **The delta rule** (DeltaNet): before writing, *subtract* what's currently stored at `k`, so the new write replaces rather than adds.

GDN combined both. KDA made the decay finer-grained (per-channel). GDN-2's insight is that there was still a hidden coupling left to break.

## Key Terms

**State matrix `S_t`**: The fixed-size `d_k × d_v` associative memory holding all history as superposed key→value outer products. Read via `o_t = S_tᵀ q_t`.

**Delta rule**: Widrow-Hoff error-correcting update. Writing k→v first *removes* the current read at `k`, then adds `v`. Equivalent to one step of online gradient descent on the reconstruction loss `½‖Sᵀk − v‖²` (derived below).

**Write strength / step size `β_t`**: In DeltaNet/GDN/KDA, a scalar in [0,1] (often per-head) that is the delta-rule learning rate. `β=1` is a full overwrite, `β=0` is no change. The quantity GDN-2 splits in two.

**Decay / forget gate `α_t`**: Multiplicative shrink applied to `S_{t-1}` before the write. Scalar in GDN; a per-channel vector `α_t ∈ (0,1]^(d_k)` in KDA and GDN-2 (`D_t = Diag(α_t)`).

**Erase gate `b_t`** (GDN-2): Channel-wise vector in `[0,1]^(d_k)` controlling, per **key** coordinate, how much of the old association is read out and subtracted.

**Write gate `w_t`** (GDN-2): Channel-wise vector in `[0,1]^(d_v)` controlling, per **value** coordinate, how much new content is committed.

**Channel-wise vs scalar gate**: A scalar applies one number to the whole operation; a channel-wise (diagonal) gate applies a different number to each coordinate. The jump from scalar `β_t` to vectors `b_t, w_t` is GDN-2's contribution.

**Chunkwise-parallel training**: The delta rule is inherently sequential (each state depends on the last). Training-time kernels reformulate a block of C tokens as matrix ops (the WY representation) so GPUs can parallelize within a chunk while staying recurrent across chunks.

**Hybrid model**: Interleaving linear-attention layers with a few standard-attention layers. GDN-2's hybrid uses **sliding-window attention** (2K window), not full global attention.

## The Delta-Rule Lineage: Five Recurrences

This is the heart of the guide. All five models below are the *same* basic machine — an associative memory updated each token — with progressively more expressive gating. I use one consistent convention so they line up.

**Notation** (GDN-2's orientation):
```
S_t ∈ ℝ^(d_k × d_v)   recurrent state after token t
q_t, k_t ∈ ℝ^(d_k)    query, key      v_t ∈ ℝ^(d_v)   value
o_t = S_tᵀ q_t        output readout
I                     d_k × d_k identity        ⊙   elementwise (Hadamard) product
Diag(·)               vector → diagonal matrix
β_t ∈ [0,1]           scalar delta write-strength      α_t   decay (scalar or vector)
```
> Transpose note: GDN-2 writes the erase term on the **left** (`(I − …)·S_{t−1}`) with key-outer/value-inner products `k_t v_tᵀ`. The form you may have seen for DeltaNet, `S_t = S_{t−1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ`, is the identical rule in the transposed (right-multiply, `S ∈ ℝ^(d_v×d_k)`) convention. I use GDN-2's left-multiply convention throughout so all five recurrences are directly comparable.

### (1) Plain Linear Attention — accumulate, never forget

```
S_t = S_{t-1} + k_t v_tᵀ
```
Each token adds an outer product. No decay, no error correction. The state is an ever-growing sum; old and new associations pile up and interfere. This is the Katharopoulos (2020) recurrent form.

### (2) DeltaNet — error-correcting writes

The delta rule says: don't blindly add `v`; first look at what's already stored at `k`, compute the error, and correct it. Formally, it's **one step of online gradient descent** on the per-token reconstruction loss:

```
L_t(S) = ½ ‖Sᵀ k_t − v_t‖²
∇_S L_t = k_t (Sᵀ k_t − v_t)ᵀ
S_t = S_{t-1} − β_t k_t (S_{t-1}ᵀ k_t − v_t)ᵀ
     = (I − β_t k_t k_tᵀ) S_{t-1} + β_t k_t v_tᵀ
```

The first term `(I − β_t k_t k_tᵀ)` **erases**: it subtracts `β_t ×` the current read along `k_t`. The second term `β_t k_t v_tᵀ` **writes** the new value. Notice `β_t` appears in *both* terms — this is the coupling GDN-2 will break. (Schlag 2021; scaled to large models by Yang 2024.)

### (3) Gated DeltaNet (GDN) — add scalar forgetting

```
S_t = α_t (I − β_t k_t k_tᵀ) S_{t-1} + β_t k_t v_tᵀ
```
A scalar (per-head) decay `α_t ∈ (0,1]` shrinks the whole state before the delta edit — global, uniform forgetting layered on top of targeted error correction. This is the NVIDIA GDN paper (Yang, Kautz, Hatamizadeh, ICLR 2025). `β_t` still does double duty.

### (4) Kimi Delta Attention (KDA) — channel-wise decay

```
S_t = (I − β_t k_t k_tᵀ) D_t S_{t-1} + β_t k_t v_tᵀ ,    D_t = Diag(α_t),  α_t ∈ (0,1]^(d_k)
```
Decay goes from one scalar to a **per-channel vector**: each key coordinate can fade at its own rate. Sharper memory management. But write `β_t k_t v_tᵀ` and erase `β_t k_t k_tᵀ` still share the *same scalar* `β_t`. Rewriting to make the coupling explicit:

```
S_t = (I − k_t (β_t k_t)ᵀ) D_t S_{t-1} + k_t (β_t v_t)ᵀ
                   ▲                              ▲
            same β on erase side          same β on write side
```
(Moonshot AI, Kimi Linear, 2025.)

### (5) Gated DeltaNet-2 — decouple erase from write

Replace the shared scalar `β_t` with two **independent channel-wise** gates: `b_t` on the key/erase side, `w_t` on the value/write side.

```
┌──────────────────────────────────────────────────────────────┐
│  S_t = (I − k_t (b_t ⊙ k_t)ᵀ) D_t S_{t-1} + k_t (w_t ⊙ v_t)ᵀ │
└──────────────────────────────────────────────────────────────┘

   b_t ∈ [0,1]^(d_k)   channel-wise ERASE gate (key axis)
   w_t ∈ [0,1]^(d_v)   channel-wise WRITE gate (value axis)
   D_t = Diag(α_t)     channel-wise decay  (inherited from KDA)
```

The erase decision now lives in **key space** (which key coordinates to read-and-remove), and the write decision lives in **value space** (which value coordinates to commit) — set by separate gates that the model controls independently per channel. This is the whole architecture.

**It strictly generalizes its parents** (the reductions are exact):
```
GDN-2  →  KDA   when  b_t = β_t·1_{d_k}  and  w_t = β_t·1_{d_v}   (both gates collapse to one scalar)
KDA    →  GDN   when  additionally  D_t = α_t·I                  (decay collapses to scalar)
```
So GDN-2 can never be worse than KDA in representational capacity — KDA is a point in its parameter space.

### Side-by-side

```
Model            Write term       Erase/read direction   Decay           Erase=Write
                                                                          coupled?
──────────────── ──────────────── ────────────────────── ─────────────── ───────────
Linear attn      k vᵀ             (none)                  (none)          —
DeltaNet         β k vᵀ           β k     (scalar β)      (none)          YES  (β)
Gated DeltaNet   β k vᵀ           β k     (scalar β)      α   (scalar)    YES  (β)
KDA              β k vᵀ           β k     (scalar β)      Diag(α) per-ch  YES  (β)
GDN-2            k (w⊙v)ᵀ         b⊙k    (vector b)       Diag(α) per-ch  NO   (b≠w)
```

## The Coupling Problem, and Why Decoupling Helps

### What the scalar β can't express

Look at what the delta rule does to the value stored at an addressed key. Take a unit key `‖k‖=1`, ignore decay for a moment, and read the state back at `k` after a DeltaNet/KDA update:

```
S_tᵀ k = (1 − β) (S_{t-1}ᵀ k)  +  β v
              ▲                       ▲
        keep fraction          write fraction
         of old value           of new value
```

The fraction of old content you **keep** is `(1−β)` and the fraction of new content you **write** is `β` — they're forced to sum to 1 by the same knob. You **cannot** say "throw away 90% of the stale value but only tentatively write the new one," nor "barely touch the old value but strongly imprint the new one." Erase aggressiveness and write aggressiveness are the same number. And it's a *scalar* — it can't even make that trade-off differently for different value channels.

That's fine when every token wants a clean overwrite. It's wrong when the right move is asymmetric — which, the paper argues, is exactly the long-context regime where you must surgically revise *some* associations while leaving competing ones intact.

### GDN-2's decoupled update, step by step

It's cleanest to write GDN-2's update in three stages (this is the form in the source code, equivalent to the compact equation above):

```
1.  S̄_t = D_t S_{t-1}                       apply channel-wise decay
2.  r_t = S̄_tᵀ e_t ,    e_t = b_t ⊙ k_t      READ the (erase-gated) key direction
3.  S_t = S̄_t + k_t (z_t − r_t)ᵀ ,  z_t = w_t ⊙ v_t   WRITE (write-gated) value, minus read
```

Stage 2 reads the memory along the erase-gated direction `b⊙k`; stage 3 writes the gated new content `w⊙v` and subtracts what was read. The erase gate `b` shapes *what gets pulled out*; the write gate `w` shapes *what gets put in* — separately.

### Numerical walkthrough

Tiny `d_k = d_v = 2` example so you can trace every number. State rows index key channels, columns index value channels.

```
S_{t-1} = ⎡ 2  3 ⎤      decay  α = (0.95, 0.99)  →  D = Diag(0.95, 0.99)
          ⎣ 4  5 ⎦
key       k = (1, 0)ᵀ        (addresses key-channel 1)
erase     b = (0.9, 0.1)ᵀ    (erase 90% of channel-1 reads, 10% of channel-2)
write     w = (0.2, 0.8)ᵀ    (write weakly to value-ch 1, strongly to value-ch 2)
value     v = (1, 1)ᵀ
```

Step 1 — decay:
```
S̄ = D S_{t-1} = ⎡ 0.95·2  0.95·3 ⎤ = ⎡ 1.90  2.85 ⎤
                ⎣ 0.99·4  0.99·5 ⎦   ⎣ 3.96  4.95 ⎦
```
Step 2 — read along the erase-gated key `e = b⊙k = (0.9, 0)`:
```
r = S̄ᵀ e = 0.9 · (row 1 of S̄) = 0.9·(1.90, 2.85) = (1.71, 2.565)
```
Step 3 — write `z = w⊙v = (0.2, 0.8)`, add `k (z − r)ᵀ` (only touches row 1, since k = (1,0)):
```
z − r = (0.2 − 1.71, 0.8 − 2.565) = (−1.51, −1.765)

S_t = S̄ + k(z−r)ᵀ = ⎡ 1.90−1.51  2.85−1.765 ⎤ = ⎡ 0.39  1.085 ⎤
                     ⎣ 3.96        4.95       ⎦   ⎣ 3.96  4.95  ⎦
```

Read off the new row 1, the addressed slot:
```
new row 1 = (1 − b₁)·(decayed old row 1) + (w ⊙ v)
          = 0.1·(1.90, 2.85)            + (0.2, 0.8)
          = (0.19, 0.285)               + (0.2, 0.8)   = (0.39, 1.085)  ✓
```

The point: the **erase fraction** (`b₁ = 0.9` → keep 10%) is set entirely by `b`, while the **written content** `(0.2, 0.8)` is set entirely by `w` — and `w` even writes *different amounts to different value channels*. In KDA both of those would be the single scalar `β`: keep-fraction `(1−β)` and a uniform write of `β·v`. Row 2 (key-channel 2) was only decayed — not addressed by this token, so it's preserved, not corrupted. That's the surgical, per-channel control a scalar can't give you.

## Gate Parameterization (the actual network)

How `b_t`, `w_t`, `α_t` are produced from the hidden state `x_t` (verified against `gdn2.py` in the official repo):

```
b_t = σ(W_b x_t)                              erase gate     (sigmoid → [0,1]^d_k)
w_t = σ(W_w x_t)                              write gate     (sigmoid → [0,1]^d_v)
g_t = − exp(a) ⊙ softplus(W_f x_t + δ)         log-decay
α_t = exp(g_t)                                channel decay  ((0,1]^d_k)
```

Details that matter if you implement it:

- **Decay is the Mamba-style parameterization** you'd recognize: `softplus` keeps the rate positive, `−exp(a)` scales it, `exp(·)` maps to `(0,1]`. Here `a = A_log` is a **per-head** learnable log-rate initialized to `log(Uniform(1,16))` and broadcast across the head's channels; `δ = dt_bias` is a per-channel bias. So decay is per-channel through the projection but its overall scale is per-head.
- **`W_f` is low-rank**: factored as `Linear(d_model → d_v) → Linear(d_v → d_k)` rather than a full `d_model → d_k` matrix — cheaper, and it ties the decay to the value subspace.
- **Optional negative eigenvalues** (`allow_neg_eigval`, RWKV-7 style): lift the erase gate `b_t` from `[0,1]` to `[0,2]`. This lets the state-transition matrix have negative eigenvalues, which provably increases state-tracking capacity (e.g. parity, modular counting). The write gate `w_t` stays in `[0,1]`.

## Architecture: The GDN-2 Block

A GDN-2 layer is a drop-in token mixer, structured much like its GDN predecessor (block-level details below are verified from the repo source; treat anything not in the recurrence as inherited-from-GDN where the paper's methods section is terse):

```
        x_t  (hidden state, d_model)
         │
         ├──────────┬──────────┬───────────┐
       W_q        W_k        W_v        gate/decay
         │          │          │        projections
    ShortConv   ShortConv   ShortConv    (W_b, W_w, W_f)
     (k=4,SiLU) (k=4,SiLU)  (k=4,SiLU)       │
         │          │          │             │
      L2-norm    L2-norm      —              │     b_t = σ(W_b x_t)
         │          │          │             │     w_t = σ(W_w x_t)
         q          k          v             │     α_t = exp(−e^a ⊙ softplus(W_f x_t+δ))
         └──────────┴──────────┴─────────────┘
                          │
              GDN-2 recurrence / chunk kernel
          S_t = (I − k(b⊙k)ᵀ) D_t S_{t-1} + k(w⊙v)ᵀ
                          │
                   o_t = S_tᵀ q_t
                          │
              SiLU-gated RMSNorm (FusedRMSNormSwishGate)
                          │
                       W_out
```

- **Short convolution** (depthwise causal, kernel size **4**, SiLU) on the q, k, **and** v paths — a cheap local-context mixer that's now standard in this family.
- **L2 normalization** on queries and keys per head (stabilizes the delta-rule dynamics — keeps `‖k‖` near 1, which is what makes the erase/keep-fraction interpretation hold). Values are not L2-normed.
- **Output**: SiLU-gated RMSNorm, then output projection.

### Trained 1.3B configuration

The headline models trained on 100B tokens:

```
                        Recurrent (pure)        Hybrid (+ SWA)
  ─────────────────     ─────────────────       ──────────────────────
  layers                18                      18
  d_model               2304                    2304
  heads                 18  (head_dim 128)      18  (head_dim 128)
  d_k = d_v             128                     128
  layer pattern         every layer GDN-2       2 GDN-2 mixers : 1 sliding-
                                                window-attn layer (2K window)
  params                1.30 B                  1.30 B
  train length          4K                      4K  (SWA window 2K)
  vocab                 32000                   32000
```

> **Discrepancy flag**: secondary coverage (MarkTechPost, WinBuzzer) lists "16 heads, d_model 2048." Those are the GDN-2 *module defaults* (`num_heads=16, head_dim=128`); the actually-trained 1.3B configs in `config.py` override them to **18 heads / d_model 2304 / 18 layers** (and `2304 / 18 = 128`, so head_dim 128 holds either way). I report the source-code config as authoritative.

### State size and complexity

```
  Recurrent state  = H · d_k · d_v  per layer  (independent of sequence length!)
                   = 18 · 128 · 128 = 294,912 floats/layer   (trained 18-head config)

  [The paper quotes 262,144 = 16·128·128, using the 16-head module default — same
   minor 16-vs-18 inconsistency as above. Either way it's a fixed few hundred KB,
   not a KV cache that grows with context.]

                        Training            Decoding (per token)
  ──────────────────    ─────────────────   ──────────────────────────
  GDN-2 (linear)        O(N)  chunk-parallel O(1) time, O(1) memory
  Transformer (softmax) O(N²)               O(N) time, O(N) memory (KV cache)
```

The decode story is the payoff and the mirror image of your long-context guide: where a Transformer's KV cache hit 167 GB at 1M tokens, GDN-2's state is the same size at 1M as at 1K. Throughput at long context is flat.

## Chunkwise-Parallel Training

The delta rule is sequential — `S_t` depends on `S_{t-1}` — which is death for GPU training if done token by token. GDN-2 uses the **WY-representation chunkwise** algorithm (from the DeltaNet parallelization paper, extended to channel-wise gates):

```
  Split sequence into chunks of size C = 64.

  WITHIN a chunk: express the C sequential delta updates as matrix operations
    T = tril(Ē K̄ᵀ, −1)         strictly-lower-triangular interactions
    A = (I + T)⁻¹               solved by forward substitution (T is nilpotent → exact, cheap)
    Y = A Ē,   U = A Z          materialize the chunk's effective erase/write matrices
  → one big matmul per chunk instead of C tiny sequential steps  (GPU-friendly)

  ACROSS chunks: carry the state S recurrently from one chunk to the next.

  The channel-wise decay D_t and the asymmetric erase factor (b⊙k vs k) are
  absorbed into the Ē, K̄, Z chunk matrices. Kernels are fused in Triton.
  A separate fused-recurrent kernel handles decoding / short sequences (q_len ≤ 64).
```

So training is `O(N)` and parallel within each 64-token chunk; inference is a pure `O(1)`-per-token recurrence. This "two-mode" design (chunk-parallel train, recurrent decode) is the standard recipe for the whole linear-attention family — the same trick that made Mamba and GDN trainable at scale.

**Training recipe** (from the repo): AdamW, peak LR `4e-4`, weight decay `0.1`, grad clip `1.0`, cosine schedule with 1B-token warmup, global batch 0.5M tokens, 100B tokens of FineWeb-Edu. Built on Flash-Linear-Attention + LitGPT.

## Benchmarks: 1.3B params, 100B FineWeb-Edu tokens

All models are matched on parameters **and recurrent state size**, so gains come from the update rule, not extra memory. Two settings throughout: **Recurrent** (pure linear) and **Hybrid** (+ 2K sliding-window attention). Best in each column in **bold** equivalent (marked `←`).

### Language modeling + commonsense reasoning

Perplexity ↓, accuracy ↑. "Avg" = mean over LAMBADA-acc + the 8 commonsense tasks (I verified this reconstructs every reported Avg exactly).

```
RECURRENT        Wiki   LMB    PIQA  Hella Wino  ARC-e ARC-c OBQA  SIQA  BoolQ  Avg
                 ppl↓   ppl↓                                                    ↑
Mamba-2          16.79  12.38  72.58 55.51 55.33 70.68 35.26 31.00 40.63 60.19  51.82
Gated DeltaNet   16.40  11.89  72.31 56.50 56.75 68.81 35.15 30.20 40.53 58.78  52.07
KDA              16.81  11.68  72.09 55.75 55.72 70.83 35.92 30.40 40.99 60.67  52.28
Mamba-3 SISO     16.30  12.99  72.31 55.58 56.20 70.45 34.56 31.00 41.76 55.90  51.42
Mamba-3 MIMO     16.45  11.66  72.36 56.49 55.78 72.38 38.07 30.00 40.89 57.74  52.39
GDN-2            15.90  11.41  72.80 56.84 57.85 72.43 38.23 31.60 40.58 59.54  53.11  ←

HYBRID (+SWA)    Wiki   LMB    PIQA  Hella Wino  ARC-e ARC-c OBQA  SIQA  BoolQ  Avg
Transformer      19.22  13.72  70.21 56.12 55.85 69.23 33.84 25.00 39.74 59.42  50.86
Mamba-2          17.46  11.29  71.47 57.52 56.17 70.50 34.73 29.80 40.35 59.31  51.99
Gated DeltaNet   16.00  10.82  70.06 57.50 56.83 70.41 35.15 30.60 40.97 60.00  52.25
KDA              16.01  10.66  71.06 56.89 57.77 71.59 35.07 30.00 40.53 62.03  52.68
Mamba-3 SISO     15.54  10.65  71.01 58.75 57.30 70.54 36.35 32.00 41.20 57.86  52.69
Mamba-3 MIMO     15.81  10.92  71.98 58.19 57.06 70.54 38.48 29.40 40.99 57.98  52.72
GDN-2            15.62  10.43  72.20 58.46 58.56 71.89 36.69 33.00 41.50 62.57  53.97  ←
```

GDN-2 has the lowest WikiText and LAMBADA perplexity and the best reasoning average in **both** settings. The pure full-attention Transformer (hybrid section) is the *worst* on perplexity here — a reminder that at 1.3B/100B with these recipes, the linear models are genuinely competitive, not a sacrifice. Margins on LM/reasoning are modest (~0.4–0.7 avg points), as you'd expect when the bottleneck isn't memory capacity.

### Long-context retrieval — RULER (where the gap widens)

Needle-in-a-haystack accuracy by context length. S-NIAH = single needle; MK-NIAH = multi-key needle (the interference-heavy associative-recall test). Recurrent setting:

```
                    KDA    Gated     Mamba-3    GDN-2
                           DeltaNet  MIMO
─────────────────── ────── ───────── ───────── ──────
S-NIAH-1 @8K        70.6   97.6      35.6       97.8   ←
S-NIAH-2 @4K        89.0   87.2      64.2       93.0   ←
S-NIAH-2 @8K        30.6   32.0      27.2       39.2   ←
S-NIAH-3 @2K        63.2   54.2      72.4       89.8   ←
S-NIAH-3 @4K        26.2   60.6      29.2       31.8
MK-NIAH-1 @1K       54.0   58.0      49.4       72.6   ←
MK-NIAH-1 @2K       44.2   37.0      19.2       51.4   ←
MK-NIAH-1 @4K       28.0   27.8      18.0       37.8   ←
```

This is the paper's strongest evidence. On **multi-key** retrieval (MK-NIAH) — where the fixed state must hold several distinct key→value bindings without letting them blur — GDN-2's lead is large and *grows* with context: +14.6 at 1K, +7.2 over KDA at 2K, +9.8 at 4K. Single-needle tasks (S-NIAH-1/2) it also leads; the one cell it doesn't top (S-NIAH-3 @4K) is a single-needle case where GDN's scalar happens to suffice.

### Real-world retrieval @2K

Accuracy ↑ on extractive QA / retrieval tasks:

```
RECURRENT        SWDE   SQuAD  FDA    TriQA  NQ     DROP   Avg
Mamba-2          17.24  32.38  14.53  58.35  18.91  19.60  26.84
Gated DeltaNet   17.90  32.67  18.52  59.60  20.16  19.69  28.09
KDA              22.49  35.10  14.90  58.12  19.58  21.80  28.67
Mamba-3 MIMO     16.68  36.65  17.44  59.06  19.16  21.08  28.35
GDN-2            23.65  36.75  19.98  61.37  19.64  17.87  29.88  ←

HYBRID (+SWA)    SWDE   SQuAD  FDA    TriQA  NQ     DROP   Avg
Transformer      32.21  38.67  54.78  58.09  22.49  22.18  38.07
Mamba-2          34.67  40.74  52.31  60.13  25.91  24.68  39.74
Gated DeltaNet   33.18  42.28  50.86  60.60  25.78  21.95  39.11
KDA              39.83  40.10  53.59  59.89  25.27  22.18  40.14
Mamba-3 SISO     35.30  46.42  54.95  59.54  25.91  23.96  41.01
GDN-2            41.96  44.70  54.68  62.38  26.31  23.67  42.28  ←
```

GDN-2 wins the average in both settings. (Note the hybrid jump from ~30 to ~42 across the board — that's the 2K sliding-window-attention layers earning their keep on extractive tasks, on top of GDN-2's mixer gains.)

### Ablation — which gate matters?

Retrieval recall average (recurrent), isolating each gate:

```
  Variant                               Retrieval recall avg
  ────────────────────────────────────  ────────────────────
  Write-gate only (scalar b, channel w) 28.92
  Erase-gate only (channel b, scalar w) 29.51
  Full GDN-2 (both channel-wise)        29.88   ←
```

Both channel-wise gates beat either alone, but the **erase gate contributes more** than the write gate. That's the associative-memory story made concrete: the dominant problem in a crowded fixed state is *clearing stale associations cleanly*, and per-channel erase control is what buys that.

> **MQAR note**: GDN-2 does **not** report the MQAR benchmark (the canonical synthetic associative-recall test from the Based/Zoology line). Its associative-recall evidence is the RULER MK-NIAH columns above. If you need MQAR specifically, you'd have to run it yourself — treat any MQAR claim about GDN-2 as unverified.

## Why the Gap Widens on Long Context

The benchmark pattern — small LM gains, large multi-key-retrieval gains — falls straight out of the associative-memory model:

```
  A fixed d_k×d_v state can store ~O(d_k) roughly-orthogonal key→value bindings
  before superposition noise swamps the signal. (You have 128 dimensions to
  share among every association the context demands.)

  Short context, few associations:  plenty of room, interference is low.
    → all the models do fine; the gate design barely matters → small LM gap.

  Long context, many competing keys:  the state is saturated; every write
  risks corrupting a binding you still need.
    → now HOW you erase and write is decisive.
       • scalar β  : erase-strength = write-strength, uniform across channels.
                     To write a new binding you must erase proportionally —
                     and you can't protect specific channels of the old one.
       • b_t, w_t  : erase only the key coordinates that should change; write
                     only the value coordinates that matter; leave the rest intact.
    → GDN-2 keeps competing associations distinct longer → big retrieval gap.
```

The ablation confirms the mechanism (erase gate > write gate), and the gap growing from 1K→4K on MK-NIAH confirms it's a capacity-under-interference effect, not a constant offset. The paper's framing: the decoupled gates **solve a real bottleneck** (memory management under interference) rather than just adding parameters — which is why a strictly-more-expressive-but-not-bigger model wins.

## GDN-2 vs Alternatives

```
                     State /        Decay      Erase rule        Erase=Write   Train  Decode
                     memory                                      coupled?
─────────────────── ─────────────── ────────── ───────────────── ──────────── ────── ──────
Transformer (softmx) KV cache O(N)   none       n/a (exact recall) n/a          O(N²)  O(N)
Mamba-2 (SSD)        d_state matrix  scalar/ch  none (no delta)    n/a          O(N)   O(1)
Mamba-3              d_state matrix  improved   none (no delta)    n/a          O(N)   O(1)
Gated DeltaNet       d_k×d_v matrix  scalar α   β k kᵀ            YES (scalar β)O(N)   O(1)
KDA (Kimi Linear)    d_k×d_v matrix  Diag(α)    β k kᵀ            YES (scalar β)O(N)   O(1)
GDN-2                d_k×d_v matrix  Diag(α)    k (b⊙k)ᵀ          NO (b≠w, ch)  O(N)   O(1)
```

The Mamba line has no delta rule at all — it forgets (decay) but never *error-corrects* a specific binding, which is why it lags on retrieval despite competitive LM scores. The DeltaNet line adds error correction; GDN-2 is the most expressive point on it so far. Against the Transformer, GDN-2 trades exact recall for O(1) decode memory — the classic linear-attention bargain, now with the best memory-management gates available.

## Practical Considerations

**When GDN-2 (or any linear-attention model) is the right call:**
- Long-context inference where KV-cache memory or decode throughput is the bottleneck — constant memory regardless of length. On your single H800 80GB, a linear model serves million-token contexts with a state that never grows, where a same-size Transformer would be KV-cache-bound.
- High-throughput / streaming generation, edge deployment, anything latency- or memory-sensitive at long sequence lengths.
- As the **mixer in a hybrid**: the strongest configuration here pairs GDN-2 layers with a few sliding-window-attention layers — you get linear scaling plus a little exact local recall.

**When to stay with full attention:**
- Tasks demanding exact, lossless recall of arbitrary earlier tokens (precise long-document copying, some code tasks) — a compressed state is lossy by construction. Hybrids exist precisely to hedge this.
- Short contexts where O(N²) is cheap anyway and the Transformer ecosystem (kernels, tooling, fine-tuning recipes) is more mature.

**Caveats before you build on it:**
- **VERY_NEW** (paper ~5 days old as of this writing). Results are a 1.3B/100B technical report, not yet independently reproduced or scaled to frontier sizes.
- **Non-commercial license** (NVIDIA Source Code License-NC) on the official repo.
- Untested at scale: no public >1.3B GDN-2, no instruction-tuned/RLHF checkpoints, no third-party reproductions yet.
- The architecture is a token-mixer swap, so your SFT/DPO/GRPO knowledge transfers directly — the training *objective* is unchanged; only the attention sublayer differs.

## Connection to Your Prior Knowledge

- **vs your GPT-2 124M**: your model keeps a KV cache (1536·N bytes, growing with the 1024-token context). GDN-2 would replace each attention sublayer with a fixed `S` matrix — no growing cache, O(1) decode. The token-level training loop (next-token CE) is identical; only the mixer changes.
- **vs the long-context guide**: this is the third pillar of that guide's three-way tradeoff (positional encoding / attention compute / KV memory) attacked at the root — by not keeping per-token KV at all. The 167 GB-at-1M KV figure becomes a few hundred KB.
- **vs MLA (MoE / long-context guides)**: MLA *compresses* the KV cache (low-rank latent) but still grows it with N. GDN-2 *eliminates* it — a fixed state. Different points on the lossy-compression spectrum.
- **vs Mamba/SSD**: GDN-2's decay gate is literally Mamba's `−exp(A_log)·softplus(·)` selective-decay parameterization; the delta rule is the piece Mamba lacks. If you've internalized SSM selective state, GDN-2 = SSM decay + error-correcting writes + decoupled per-channel write/erase gates.
- **vs distillation/GRPO**: orthogonal — GDN-2 is a pretraining-architecture change, not a post-training method. You'd pretrain a GDN-2 backbone, then SFT/DPO/GRPO it exactly as you did your GPT-2.

## Summary

- **Problem**: Linear attention compresses all history into a fixed-size state to escape the Transformer's O(N) KV cache — but a fixed state forces associations to share memory and interfere, worsening as context grows.
- **Predecessors**: Gated DeltaNet (scalar decay + delta rule) and KDA (channel-wise decay + delta rule) both managed memory with the delta rule, but both used a **single scalar `β_t`** to control erasing old content *and* writing new content together.
- **GDN-2's idea**: split `β_t` into two **independent channel-wise gates** — `b_t` (per-key-channel erase) and `w_t` (per-value-channel write): `S_t = (I − k_t(b_t⊙k_t)ᵀ) D_t S_{t-1} + k_t(w_t⊙v_t)ᵀ`. Strictly generalizes KDA and GDN.
- **Result** (1.3B / 100B FineWeb-Edu): best overall vs Mamba-2, GDN, KDA, and Mamba-3 on LM, reasoning, and retrieval — with the **widest margin on long-context multi-key retrieval** (MK-NIAH), exactly where keeping competing associations distinct matters most. Ablation shows the **erase gate** drives most of the gain.
- **Engineering**: O(N) chunkwise-parallel training (WY representation, Triton-fused), O(1)-per-token constant-memory decode; short-conv + L2-normed q/k; pure or hybrid-with-2K-sliding-window-attention.
- **Status**: VERY_NEW (May 2026), non-commercial license, 1.3B-only, unreproduced. A clean demonstration that *more expressive memory-management gates*, not more parameters, can be the lever for long-context linear attention.

## Key Papers

| Paper | Authors | Year | Contribution |
|-------|---------|------|--------------|
| [Transformers are RNNs](https://arxiv.org/abs/2006.16236) | Katharopoulos et al. (Idiap/EPFL) | 2020 | Linear attention: kernel feature map → O(N) recurrence with fixed state |
| [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174) | Schlag, Irie, Schmidhuber | 2021 | Fast-weight view + the delta rule for memory writes (DeltaNet root) |
| [Gated Linear Attention (GLA)](https://arxiv.org/abs/2312.06635) | Yang et al. (MIT/IBM) | 2023 | Data-dependent gating + hardware-efficient chunkwise training |
| [Mamba](https://arxiv.org/abs/2312.00752) | Gu, Dao (CMU/Princeton) | 2023 | Selective SSM; first linear-time rival to Transformers at scale |
| [DeltaNet parallelization](https://arxiv.org/abs/2406.06484) | Yang et al. | 2024 | WY-representation chunkwise algorithm — delta rule trainable at scale |
| [Transformers are SSMs (Mamba-2/SSD)](https://arxiv.org/abs/2405.21060) | Dao, Gu | 2024 | State-space duality; the SSM baseline |
| [Gated Delta Networks (GDN)](https://arxiv.org/abs/2412.06464) | Yang, Kautz, Hatamizadeh (NVIDIA) | 2024 | Scalar decay + delta rule; GDN-2's direct predecessor |
| [RWKV-7 "Goose"](https://arxiv.org/abs/2503.14456) | Peng et al. | 2025 | Dynamic state evolution; negative-eigenvalue state tracking |
| [Kimi Linear / KDA](https://arxiv.org/abs/2510.26692) | Kimi Team (Moonshot AI) | 2025 | Kimi Delta Attention: channel-wise (diagonal) decay |
| [Mamba-3](https://arxiv.org/abs/2603.15569) | (ICLR 2026) | 2026 | Latest SSM; SISO/MIMO; toughest SSM baseline |
| [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791) | Hatamizadeh, Choi, Kautz (NVIDIA) | 2026 | **This guide** — decouple erase (`b_t`) and write (`w_t`) into independent channel-wise gates |
