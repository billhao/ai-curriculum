# 01 — Knowledge Capacity & the Mechanism of Reasoning

*Covers Q1 ("how much of a frontier model is knowledge?") and Q5 ("what is the mechanism of reasoning?"). This is the scientific foundation — whether Q2/Q3 (building a small reasoning core) are even possible depends on the answers here.*

---

## Part A — How much of a model is knowledge? (Q1)

### A.1 The capacity anchor: ~2 bits per parameter

The one quantitative law in this space is **Allen-Zhu & Li, "Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws"** (arXiv:2404.05405, ICLR 2025).

- Headline: a sufficiently trained transformer stores **~2 bits of knowledge per parameter**, where "knowledge" = bits to encode `(name, attribute, value)` tuples, estimated via a bit-complexity lower bound. **[HIGH]**
- A 7B model therefore has capacity for **~14 billion knowledge-bits** — which they argue exceeds English Wikipedia + textbooks *in tuple-bit terms* (not raw text bytes). **[MEDIUM]**
- Conditions that matter (this is a *ceiling*, measured on synthetic corpora, models from scratch at 10⁶–10⁹ params — **not** an audit of frontier models):
  - **1000 exposures per fact** to reach the 2-bit ceiling; at **100 exposures** capacity drops to **~1 bit/param**. "1000 exposures" ≠ 1000 corpus passes — common web facts already recur millions of times in one pass. **[HIGH]**
  - **int8 quantization: no capacity loss. int4: >2× loss (~0.7 bit/param)** — implies real redundancy in parametric storage; not every weight-bit is a used fact-bit. **[HIGH]**
  - **MoE (32 experts): only ~1.3–1.5× capacity loss** while using ~8.8% of params per token — knowledge can be spread sparsely, not densely always-on. **[HIGH]**
  - **Junk data is expensive:** a 7:8 junk ratio cut useful capacity **20×** at 100 exposures; prepending source/domain tokens recovered it to **2×**. **[HIGH]**

**Why this *undercuts* "most params must be knowledge":** if 7B params already over-cover Wikipedia-scale facts at 2 bits each, a frontier model's hundreds of billions of params are **not** all needed for factual storage. Capacity ≠ allocation, but the law argues factual knowledge is *cheaper* than the parameter count suggests.

### A.2 Where knowledge lives — and why you can't cleanly point at it

| Finding | Paper | What it shows |
|---|---|---|
| FFN/MLP layers act as **key-value memories** | Geva et al., EMNLP 2021 (arXiv:2012.14913) | keys match input patterns, values write output distributions |
| **Knowledge neurons** | Dai et al., ACL 2022 (arXiv:2104.08696) | individual FFN neurons correlate with specific facts |
| **ROME** — rank-one edit | Meng et al., NeurIPS 2022 (arXiv:2202.05262) | a fact is causally localizable to **mid-layer MLP at the subject's last token**; editable by a rank-one weight change |
| **MEMIT** — mass editing | Meng et al., ICLR 2023 (arXiv:2210.07229) | scales to thousands of edits |

Architecturally, **MLP blocks are ~2/3 of transformer parameters** — so "knowledge lives mostly in the bulk of the weights" is a defensible *architectural* inference. **But the localization story breaks down on inspection:**

- **Localization ≠ editability** — Hase et al., NeurIPS 2023 (arXiv:2301.04213): the layer causal tracing flags as "where the fact is used" is *not* the best layer to edit. Strongest single rebuttal to clean localization. **[HIGH]**
- **Relation knowledge lives in attention too** — Wei et al., CIKM 2024 (arXiv:2409.00617): not a pure-MLP phenomenon. **[HIGH]**
- **Superposition** — Anthropic "Toy Models of Superposition" (Elhage et al., 2022): models pack more features than neurons; concepts are distributed directions. Single-neuron attribution is lossy. **[HIGH]**
- **A single direction mediates the balance** — Hong et al. (arXiv:2503.23084): one residual-stream feature toggles reasoning-vs-recall; intervening shifts behavior. If *one axis* controls the trade-off, the two are **entangled along a shared axis**, not stored in separate parameter sets. **[LOW — single-source]**

### A.3 Storage ≠ extraction ≠ manipulation (the part that matters for retrieval)

Two more Allen-Zhu results reframe what "having knowledge" even means:

- **Part 3.1, Knowledge Storage and Extraction** (arXiv:2309.14316, ICML 2024): a fact can be **stored but not extractable** — without paraphrase/diversity augmentation in pretraining, QA-style recall collapses toward zero. **[HIGH]**
- **Part 3.2, Knowledge Manipulation** (arXiv:2309.14402, ICLR 2025): **retrieval is far easier than manipulation.** Models answer "what is X?" but fail simple *comparison/classification/inverse-search* over the same facts unless CoT is present in **both** training and inference. **[HIGH]**

> **Direct implication for your Q3:** external retrieval can supply the *fact*, but Part 3.2 says the model still needs trained **manipulation** machinery to *reason over* it. Retrieval reduces the storage burden; it does not remove the reasoning-data burden. This is the mechanistic reason a "cognitive core" still needs real reasoning training, not just a search tool.

### A.4 Verdict on Q1

> "A lot of a frontier model's parameters are knowledge" is true only in the **weak** sense (frontier models obviously carry enormous parametric factual content). It is **not** supported in the **strong** sense that most parameters form a cleanly-separable factual database. **No credible paper gives an "X% knowledge / Y% reasoning" fraction.** The evidence points to shared weights, overlapping circuits, and superposed features — a **gradient, not a partition**. The "2/3 MLP = knowledge" figure is architectural inference, not a measured allocation.

---

## Part B — What is reasoning, mechanistically? (Q5)

### B.1 Reasoning ≈ extra serial computation / effective depth

The best-supported picture: **reasoning is additional computation over shared representations, often needing more serial steps.** Explicit CoT is one way to *buy* those steps by writing intermediate state into tokens. The depth/compute framing is now backed by several angles:

- **Reasoning can happen without verbalized tokens.**
  - *Implicit CoT via knowledge distillation* — Deng et al. (arXiv:2311.01460): distill the steps into hidden-state ("vertical") computation, keep much of the benefit at near-zero CoT latency.
  - *Coconut — chain of continuous thought* — Hao et al., COLM 2025 (arXiv:2412.06769): feed the last hidden state back as the next input embedding; lets the model represent multiple next-steps (search-like). **Caveat:** the win is concentrated on planning/backtracking-heavy logic (e.g. ProsQA), *not* uniformly on math like GSM8K. **[MEDIUM]**
- **Looped/recurrent depth = effective depth.**
  - *Reasoning with Latent Thoughts: On the Power of Looped Transformers* — Saunshi et al., ICLR 2025 (arXiv:2502.17416): looped models approximate deeper non-looped models and **provably simulate multiple CoT steps as latent recurrence.** **[HIGH]**
  - *Recurrent-depth latent reasoning* — Geiping et al. (arXiv:2502.05171): a recurrent block unrolled at test time reaches reasoning quality "equivalent to ~50B params" — buying reasoning by adding *compute*, not *stored knowledge*. **[MEDIUM]**

**This is the load-bearing fact for your whole program:** reasoning behaves like *iterated computation* (compressible, architecturally cheap, transferable), whereas knowledge behaves like *storage* (paid per fact). That asymmetry is what makes a small reasoning core *conceivable*.

### B.2 But latent multi-hop composition is fragile

Reasoning-as-compute has a sharp limit when it must compose *parametric* facts internally:

- **Sohee Yang et al. (DeepMind), SOCRATES benchmark** (arXiv:2411.16679): shortcut-free latent multi-hop composition is real but uneven — best models hit only **~5–6%** on year-bridge queries vs **80%+** on some country-bridge queries. **Explicit CoT vastly outperforms latent direct-answer composition.** **[HIGH]**

This separates "can emit good CoT if asked" from "can internally compose stored facts without a scratchpad" — and explains why test-time CoT/tools matter so much.

### B.3 The keystone: post-training teaches *when*, not *how*

**Venhoff et al., "Base Models Know How to Reason, Thinking Models Learn When"** (arXiv:2510.07364, Oct 2025 — *verified*):
- Steering vectors (optimized at ~37% model depth) activate latent reasoning in **base** models, recovering **up to 91% of the performance gap to thinking models — with no weight updates, while steering only 12% of tokens.**
- Conclusion: **pretraining installs the reasoning mechanisms; RLVR/post-training mostly teaches *when* to deploy them.** **[VERY_NEW but verified]**

This dovetails with the data-side evidence (see [03](03-data-for-small-reasoning-models.md)): Yue (2504.13837) — RL sharpens sampling over latent-good trajectories rather than adding capability; Gandhi (2503.01307) — the base must already exhibit the cognitive behaviors. **All three clusters independently land on the same mechanism.**

### B.4 CoT is a lossy report, not the computation

If you want to *read* the reasoning, beware:

- Turpin et al. 2023 — CoT becomes unfaithful under bias/hint manipulation.
- **Anthropic, "Reasoning Models Don't Always Say What They Think"** (arXiv:2505.05410, 2025): frontier reasoning models mention a hint they *demonstrably used* in **<20%** of cases; outcome-based RL improves faithfulness, then plateaus. **[HIGH]**
- Arcuschin et al., "CoT in the Wild Is Not Always Faithful" (arXiv:2503.08679): post-hoc rationalization on *ordinary* prompts, not just contrived probes.
- Cross-link to your [reasoning-model-interp.md](../interp/reasoning-model-interp.md): the 4-way partition of "wait" tokens (load-bearing / inflection / performative / pathological) is the fine-grained version of this — much verbalized reasoning is performative.

### B.5 The skeptic's caveat: is some "reasoning" just retrieval?

- **GSM-Symbolic** — Mirzadeh et al., ICLR 2025 (arXiv:2410.05229): math accuracy drops sharply (up to ~65% on hard variants) under irrelevant-clause insertion and number swaps — implying a chunk of "reasoning" is **pattern-matching over memorized templates.** **[HIGH]**

So the knowledge/reasoning boundary blurs from *both* sides: knowledge is computation-coupled (A.2), and some reasoning is retrieval-like (B.5). This is the deepest reason the clean factorization is an idealization.

---

## Confidence summary (Q1 + Q5)

| Claim | Confidence |
|---|---|
| ~2 bits/param capacity law exists (synthetic, ICLR'25) | HIGH (existence) / MEDIUM (frontier extrapolation) |
| Facts partly localizable in mid-layer MLPs, but not cleanly separable from attention/compute | HIGH |
| No literature estimate of "% params = knowledge vs reasoning" | HIGH |
| Reasoning ≈ added effective depth / serial compute | HIGH |
| Reasoning can occur without verbalized CoT | MEDIUM-HIGH |
| Latent multi-hop composition is fragile vs explicit CoT | HIGH |
| Post-training teaches *when* not *how* (Venhoff) | VERY_NEW, verified |
| Verbalized CoT is often unfaithful | HIGH |
| Some "reasoning" is template retrieval (GSM-Symbolic) | HIGH |

**Net:** the mechanism *supports* the hope that reasoning is separable-and-compressible, but the weights don't honor a clean knowledge/reasoning line. Continue to [02 — building the small reasoning core](02-small-reasoning-core-compression-retrieval.md).
