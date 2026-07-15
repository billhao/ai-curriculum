# Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories

How Google Research (with Cornell) replaces the static train/test split with a periodic **Wake/Sleep lifecycle**: while "awake" the model just processes context like any LLM; while "asleep" it distills its own fragile short-term (in-context) memories **upward** into a larger, slower parametric memory — then generates and rehearses its own synthetic "dreams" to self-improve, without ever touching external data again.

> **Naming collision — read this first.** There are two different, unrelated 2026 papers both asking "does a language model need sleep?" This guide covers **Behrouz, Hashemi, Javanmard, Mirrokni (Google Research + Cornell), "Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories," arXiv 2606.03979** — a continual-learning method built on RL, parameter expansion, and upward distillation. It is **not** the same as Lee, Fanti, McLeish, Goldstein (CMU/UMD), **"Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference," arXiv 2605.26099**, which loops a Gated-DeltaNet-style hybrid `N` times over a context chunk to consolidate it into a fixed-size SSM state before evicting the KV cache. That paper already has a guide at [../inference/llm-sleep-offline-recurrence-guide.md](../inference/llm-sleep-offline-recurrence-guide.md) — its own intro footnote flags this exact collision. The two share a title and a neuroscience metaphor and nothing else: one is post-training/continual-learning (this guide), the other is an inference-time architectural schedule.

> Confidence: **VERY_NEW**. arXiv 2606.03979v2, submitted 10 Jul 2026 (a version has been publicly available since **September 2025** on OpenReview, per the paper's own footnote — so the ideas are ~10 months old even though the arXiv posting is ~5 weeks old at the time of writing). Every equation, algorithm step, model size, dataset, and benchmark number below was read and transcribed **verbatim from the full 26-page PDF** (methods, all figures/tables, and the complete 144-entry reference list), not from search-engine summaries. No public code repository was found. A GPT-5.5 web pass found sparse independent discussion: one indexed OpenReview review snippet rates the submission "4: marginally below the acceptance threshold" (early signal, not consensus — treat as unresolved). Points needing extra care are flagged inline.

## Background

**Originating paper**: [Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories](https://arxiv.org/abs/2606.03979) (Ali Behrouz, Farnoosh Hashemi, Adel Javanmard — Google Research; Vahab Mirrokni — Google Research; Hashemi also Cornell). arXiv 2606.03979, v2 10 Jul 2026.

**Research lineage**:

1. **Anterograde amnesia** (Scoville & Milner, 1957) — [Loss of recent memory after bilateral hippocampal lesions](https://pmc.ncbi.nlm.nih.gov/articles/PMC497229/). The H.M. case: damage to the hippocampus leaves existing long-past memories intact but destroys the ability to convert new experience into durable memory — the founding analogy the paper uses for a static LLM (context window = "immediate present," pretrained MLP weights = "long past," nothing bridges them).
2. **Complementary Learning Systems** (McClelland, McNaughton & O'Reilly, 1995; updated by Kumaran, Hassabis & McClelland, 2016) — [Why there are complementary learning systems in the hippocampus and neocortex](https://www.semanticscholar.org/paper/2ebf18e7892e660a833152ddc6cf8f1d21a7b881). Fast hippocampal encoding + slow neocortical integration, with the hippocampus *replaying* recent experience into cortex over time. The two-speed-memory blueprint this paper's Continuum Memory System generalizes to many speeds.
3. **Sleep neuroscience** — synaptic homeostasis (Tononi & Cirelli, 2006, [Sleep function and synaptic homeostasis](https://pubmed.ncbi.nlm.nih.gov/16376591/)): slow-wave sleep downscales synaptic strength to counteract waking-hours saturation. Hippocampal replay during quiet wakefulness (Foster & Wilson, 2006, [Reverse replay of behavioural sequences](https://www.nature.com/articles/nature04587)). REM synaptic pruning (Li, Ma, Yang & Gan, 2017, [REM sleep selectively prunes and maintains new synapses](https://www.nature.com/articles/nn.4479)) — the direct inspiration for this paper's "reset the old low-rank experts after consolidation" step.
4. **Fast Weight Programmers** (Schmidhuber, 1992) — the paper's own worked example for "update frequency": a slow net emits updates for a fast net that changes every step.
5. **Elastic Weight Consolidation** (Kirkpatrick et al., 2017, PNAS) — regularization-based continual learning; the paper's continual-learning baseline (EWC).
6. **Knowledge Distillation** (Hinton, Vinyals & Dean, 2015) and **sequence-level KD** (Kim & Rush, 2016) — the classical *downward* (large→small) distillation this paper inverts. See your [distillation guide](../distillation/distillation-guide.md) for the standard direction and the temperature-softmax mechanics reused here for the semantic-similarity reward.
7. **LoRA** (Hu et al., 2022) — the low-rank adaptation mechanism used both for the new "expert" params added during consolidation and for the isolated per-dream fine-tunes during Dreaming.
8. **Titans: Learning to Memorize at Test Time** (Behrouz, Zhong & Mirrokni, 2024) — [arXiv 2501.00663](https://arxiv.org/abs/2501.00663). Neural long-term memory module that meta-learns to memorize/forget at test time; the architectural ancestor of Hope's per-block learned update rule.
9. **Nested Learning: The Illusion of Deep Learning Architectures** (Behrouz, Razaviyayn, Zhong & Mirrokni, 2025, NeurIPS 2025) — [OpenReview nbMeRvNb7A](https://openreview.net/forum?id=nbMeRvNb7A) / [arXiv 2512.24695](https://arxiv.org/abs/2512.24695). Reframes architecture + optimizer as nested associative-memory problems at different update frequencies; introduces the **Continuum Memory System (CMS)** and the **Hope** architecture that this paper's memory-consolidation mechanism is built directly on top of (Sleep reuses CMS/Hope's Definition-1 "update frequency" formalism verbatim).
10. **On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes** a.k.a. **GKD** (Agarwal et al., Google DeepMind, ICLR 2024) — [arXiv 2306.13649](https://arxiv.org/abs/2306.13649). Same paper you already know from the distillation guide's on-policy section; Sleep's Knowledge-Seeding objective is a direct extension of GKD's mixture-of-on-policy-and-off-policy formulation.
11. **Beyond Human Data: Scaling Self-Training** a.k.a. **ReST\*\*EM*\*\* (Singh et al., 2024, TMLR) — generate, filter-by-reward, SFT, repeat; the optimization loop Dreaming's self-improvement reduces to.
12. **GREATS: Online Selection of High-Quality Data for LLM Training** (Wang, Wu, Song, Mittal & Jia, NeurIPS 2024) — gradient-inner-product-based online batch selection; Dreaming's mechanism for picking which synthetic "dreams" are worth training on.
13. **SEAL: Self-Adapting Language Models** (Zweiger, Pari, Guo, Akyürek, Kim & Agrawal, 2025/2026) — [arXiv 2506.10943](https://arxiv.org/abs/2506.10943). A model generates its own SFT self-edits and is RL-rewarded by whether they improve downstream performance. Dreaming builds directly on SEAL but fixes three gaps (below); SEAL is also the strongest baseline in three of the paper's four experiment families.
14. **Cartridges** (Eyuboglu et al., 2025) — a pretrained "KV-cache adapter" compressing long context via self-study; the paper's main long-context/continual-learning baseline, and it *loses to* Sleep in every reported comparison.
15. **This paper**: **Sleep** (Behrouz, Hashemi, Javanmard & Mirrokni, 2026) — arXiv 2606.03979. Wraps Nested Learning's CMS/Hope in a Wake/Sleep lifecycle with a formal Compute→Consolidate→Update protocol for upward distillation, plus an RL-driven Dreaming stage for self-improvement.

**The one-line thesis**: an LLM's knowledge lives in only two places — the context window (fragile, evicted at the end of the session) or the pretrained weights (permanent, but frozen after training). Sleep adds a third regime: periodically pause taking new input, and spend compute distilling the fragile, currently-active short-term memory *upward* into a strictly larger set of newly-grown, more-stable parameters — then use spare cycles to rehearse self-generated synthetic data and improve further, exactly like slow-wave (NREM) and REM sleep in the human brain.

## What Problem Does It Solve?

You already know the two extremes of "memory update frequency" in a Transformer from the long-context and MoE guides — attention has effectively infinite update frequency (its "memory" is rewritten completely every context), MLP/FFN weights have zero (frozen after pretraining):

```
                    Conventional ML                       Continual Learner (Sleep)
                    ─────────────────────                 ─────────────────────────
  Lifecycle         ├── Training Time ──┤├─ Test Time ─┤   ...Wake...Sleep...Wake...Sleep...
                    (one pass, then frozen forever)         (alternates forever, no fixed end)

  Attention (KV)     update freq = ∞ (rewritten each ctx)   same — the fast, fragile layer
  MLP / FFN          update freq = 0 (frozen post-train)    replaced by a SPECTRUM (CMS)
  New knowledge      lost at context end, or needs full     absorbed into a slower, more
                     retraining / fine-tune (expensive,      stable memory tier via an
                     causes catastrophic forgetting)          explicit Sleep phase
```

The paper's anterograde-amnesia framing (Scoville & Milner 1957): H.M.'s hippocampal damage left old memories intact but destroyed the ability to *consolidate* new short-term experience into long-term memory — he re-experienced the present as perpetually new. An LLM has the identical pattern: its immediate context is like intact short-term memory, its frozen post-pretraining weights are like intact remote memory, and there is **no mechanism at all** analogous to the biological consolidation process that would bridge the two. The paper's central question: *what is that missing mechanism?*

Existing fixes are all partial:

```
  Approach                    Fixes staleness?   Avoids CF?      Cost
  ─────────────────────────   ────────────────   ─────────────   ─────────────────────
  Re-pretrain on new data     yes                n/a             extremely expensive
  Fine-tune / LoRA            yes                NO (catastrophic  cheap per update, but
                                                  forgetting, CF)   destructive over time
  In-context learning (ICL)   only within window  yes (nothing     free, but forgotten at
                                                    is overwritten)  context end — no transfer
                                                                     to durable memory
  Sleep (this paper)          yes                yes (by design)  extra compute at sleep time,
                                                                    but no destructive overwrite
```

## Key Terms

**Wake time**: the model is actively receiving external input and processing it — ordinary inference/training, no different from any LLM today.

**Sleep time**: the model receives no (or minimal) external input; it spends compute on internal computation only — consolidating memories and self-improving. Not a passive/idle state — analogous to how the brain is highly active during sleep.

**Update frequency `f_W`** (Definition 1, from Nested Learning): for a weight component `W`, the number of times it is updated per unit of time, where "unit of time" = one update-step of the *slowest* component in the system.

**Continuum Memory System (CMS)**: a chain of MLP blocks `MLP^(f1)(·),…,MLP^(fk)(·)` with strictly decreasing update frequencies `f1 ≥ f2 ≥ … ≥ fk`, generalizing the old "attention = short-term, MLP = long-term" dichotomy into a full spectrum of memory speeds.

**Online consolidation**: the ordinary end-to-end forward/backward pass through a CMS stack *during Wake* — earlier (faster) blocks already influence later (slower) blocks every forward pass. Present in any deep net; not enough for continual learning on its own (Nested Learning's finding) because it's retrieval-dependent and doesn't compress/abstract.

**Offline consolidation (Sleep, this paper's contribution)**: an explicit, separate phase where the model is not processing external input and instead runs a Compute→Consolidate→Update protocol to transfer knowledge between memory tiers using self-generated (not raw replayed) data.

**Knowledge Seeding (KS) / upward distillation**: a distillation process where one or more *smaller* models (fewer/lower-capacity active parameters) are the teacher, and a *larger* model (more active parameters) is the student — the reverse of ordinary KD's big-teacher/small-student direction.

**Self-Knowledge Seeding (SKS)**: the special case Sleep actually uses — teacher and student are the *same* model at two different capacities (some parameters inactive vs. active).

**Parameter (de)activation / expansion**: new low-rank expert weights exist in the model from the start but are masked out of the forward/backward pass until a sleep cycle explicitly activates them — mirroring the brain's fixed-but-large neuronal capacity, with new synaptic connections forming over a lifetime rather than new neurons being added.

**Synaptic pruning**: after a consolidation step, the low-rank expert parameters that were added in *past* sleep cycles are reset (zeroed / made available again) — directly modeled on REM sleep's synapse-pruning function (Li et al., 2017).

**Dreaming**: the REM-analog second stage of Sleep — the model generates and rehearses synthetic "dream" data to refine and extend its own capabilities, without external supervision.

## The Continuum Memory System: Formalizing Multi-Speed Memory

Parameterize the model as `θ = {W_1^(f1),…,W_{k1}^(f1)} ∪ {W_1^(f2),…,W_{k2}^(f2)} ∪ … ∪ {W_1^(fc),…,W_{kc}^(fc)}`, with the parameter sets sorted by decreasing update frequency `f1 ≥ f2 ≥ … ≥ fc`. Given an input sequence `x = {x_1,…,x_T}`, the CMS forward pass chains the blocks:

```
y_t = MLP^(fk)( MLP^(f_{k-1})( ⋯ MLP^(f1)(x_t) ) )                                  (Eq. 1)
```

Each block's own parameters update on their own clock. With chunk size `C^(ℓ) := max_{ℓ'} C^(ℓ') / f_ℓ` (a larger `C^(ℓ)` = longer gap between updates = "slower" block), block `ℓ`'s parameters evolve as:

```
θ_{i+1}^(fℓ) = θ_i^(fℓ) − e_{i,ℓ}          where

e_{i,ℓ} = { Σ_{t=i−C^(ℓ)}^{i} η_t^(ℓ) f(θ_t^(fℓ); x_t)     if i ≡ 0 (mod C^(ℓ))          (Eq. 2)
          { 0                                              otherwise
```

`η_t^(ℓ)` = per-block learning rate, `f(·)` = one component of an arbitrary optimizer (e.g. a gradient step). In plain terms: block `ℓ` silently *accumulates* gradient signal every step, but only actually writes it into its weights once every `C^(ℓ)` steps — i.e. each block's parameters are literally "a compressed representation of its own recent context," which is exactly why Sleep can later distill that compressed state into a neighboring block.

**Why this generalizes the old two-tier view**: setting `k=1` block with `f_W → ∞` recovers plain attention (rewritten every token — no persistence at all); setting `f_W = 0` (updates only during pretraining) recovers a standard frozen MLP. CMS is the superset — a Transformer is the special case with exactly one (zero-frequency) MLP block per layer.

### Numerical walkthrough: how many consolidations before the next tier updates?

The paper's own worked example (Section 3.2): a block with update period 1K steps feeding a block with update period 10K steps. The faster block completes **10** of its own update cycles for every 1 update cycle of the slower block — meaning 10 separate consolidation events must happen into the slower memory before the slower memory's own weights actually change. This repeated-consolidation requirement (not a single hand-off) is called out as *the* critical bottleneck that Sleep's whole machinery exists to solve efficiently.

The actual three-tier schedule used in the architecture (Figure 7) is `High-Freq FFN` period 1000 steps → `Mid-Freq FFN` period 5000 steps → `Low-Freq FFN` period 10000 steps:

```
  High-Freq FFN  (period = 1,000 steps)  ──5 consolidations──▶  Mid-Freq FFN (period = 5,000)
  Mid-Freq FFN   (period = 5,000 steps)  ──2 consolidations──▶  Low-Freq FFN (period = 10,000)
```

So within one Low-Freq update window (10,000 steps), the Mid tier consolidates into it twice, and within one Mid-Freq window the High tier consolidates into it five times — i.e. `5 × 2 = 10` High-Freq consolidation events ultimately feed into a single Low-Freq update, matching the paper's abstract 1K:10K illustration.

## The Hope Architecture

```
        x_t
         │
   ┌─────┴─────┐
   │ Sequence  │   attention block — the fastest tier, f_W = ∞ (rewritten every context)
   │  Layer    │
   └─────┬─────┘
      q,k,v (linear projections)
         │
   ┌─────▼──────────┐
   │ High-Freq FFN   │  f_W ≈ period 1,000 steps  (sparse MoE with router)
   └─────┬───────────┘
         │  consolidate ×5
   ┌─────▼──────────┐
   │ Mid-Freq FFN    │  f_W ≈ period 5,000 steps  (sparse MoE with router)
   └─────┬───────────┘
         │  consolidate ×2
   ┌─────▼──────────┐
   │ Low-Freq FFN    │  f_W ≈ period 10,000 steps (sparse MoE with router) — most persistent
   └─────────────────┘
```

Each FFN tier is a **sparse Mixture-of-Experts** with its own router `R^(fℓ)`, holding a growing set of experts `{W^(fℓ),1,…,W^(fℓ),sℓ}`. Growth happens by adding low-rank experts `A ∈ ℝ^{d×d_low}`, `B ∈ ℝ^{d_low×d}` (`d_low ≪ d`) — in the paper's experiments, **5 MLP blocks of dimension 64**, with total *active* parameter count kept unchanged from the base model (Llama-3B/8B, Qwen3-1.7B/8B, Llama-3.2-1B across experiments).

## Sleep = Memory Consolidation (NREM) + Dreaming (REM)

```
┌─────────────────────────────────── Wake ───────────────────────────────────┐
│  model receives external input, does ordinary forward/backward passes,     │
│  online consolidation happens for free (faster blocks → slower blocks,     │
│  end-to-end) — same as any deep net, not sufficient on its own for CF      │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │  (window f_W expires)
┌─────────────────────────────────── Sleep ──────────────────────────────────┐
│  NO external input. Two stages, run in sequence:                           │
│                                                                              │
│  (1) Memory Consolidation — "slow-wave sleep" analog                       │
│      Parameter Expansion → Compute → Consolidate (Knowledge Seeding) →     │
│      Update (apply base weights, prune old experts, activate new expert)   │
│                                                                              │
│  (2) Dreaming — "REM sleep" analog                                         │
│      generate synthetic dreams (with random-expert novelty injection) →    │
│      GREATS-style importance selection → per-dream isolated LoRA SFT →     │
│      ReST^EM reward loop (does the dream improve downstream performance?)  │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                   Wake (repeat)
```

## Memory Consolidation, Step by Step

### 1. Parameter Expansion

To consolidate block `ℓ*−1` (the sender, faster) into block `ℓ*` (the receiver, slower/next tier), add **one new low-rank expert** `{A^(fℓ*), B^(fℓ*)}` to receiver block `ℓ*`'s MoE expert set. This new expert is where the transferred knowledge from the sender will live. This growth happens gradually, once per sleep cycle — the model literally increases its own parameter count over its lifetime (the paper's analogy: neuroplasticity — new synaptic connections forming in a brain of fixed neuron count, not new neurons being added).

### 2. The Compute → Consolidate → Update protocol

Sleep triggers when the step count is divisible by `C^(ℓ*−1)` — i.e. right when the sender block `MLP^(fℓ*-1)` is scheduled for its own base-weight update.

```
(a) COMPUTE     Base-weight update for the sender is computed from its accumulated
                gradients (Eq. 2) → prospective new params θ_new^(fℓ*-1).
                Computed, but NOT YET applied to the running model.

(b) CONSOLIDATE Teacher  LM_θ     = model BEFORE the update: original θ^(fℓ*-1)
                                     + its already-accumulated experts.
                Student  LM_θexp  = model using the PROSPECTIVE θ_new^(fℓ*-1)
                                     (old low-rank experts reset) + the new
                                     low-rank expert {A,B} freshly added to MLP^(fℓ*).
                Optimize {A,B} via the Knowledge-Seeding objective (below) to
                minimize the teacher↔student distillation loss.

(c) UPDATE      Apply the base-weight update θ^(fℓ*-1) → θ_new^(fℓ*-1).
                Reset the old low-rank experts in MLP^(fℓ*-1) (synaptic pruning).
                Activate the new expert {A,B} in MLP^(fℓ*).
```

The ordering matters: the teacher must be frozen at the *pre-update* state so it faithfully represents "what the fast memory knew right before it was about to be overwritten" — the exact instant you'd otherwise lose that information to catastrophic forgetting.

## Knowledge Seeding: Why "Upward" Distillation Is Unusual

> **Upward Distillation (Knowledge Seeding)**: given small models `S_1(·),…,S_k(·)` as the teacher(s), transfer their knowledge to a strictly *larger*-capacity model `M(·)`.

This inverts the direction you already know from the [distillation guide](../distillation/distillation-guide.md), where a big teacher (GPT-4, DeepSeek-R1) always has *more* capacity than the small student, and the KD loss's job is to compress the teacher's knowledge into a tighter budget. Here, **Self-Knowledge Seeding (SKS)** — a smaller-capacity version of the *same* model (some params masked off) is the teacher, and a larger-capacity version (params activated) is the student. Two problems this direction creates that plain response-level SFT-on-teacher-outputs (Kim & Rush 2016) can't solve:

1. **The student has more capacity than the teacher.** Ordinary sequence-level KD trains the student purely on teacher-generated text; with a capacity-richer student, that under-uses the extra room and can even be actively sub-optimal (nothing forces the student to use its slack productively).
2. **Sleep has no access to fresh external data.** Classic KD (Hinton et al. 2015) assumes you can freely sample new inputs; during Sleep, the model can only work with what it already has in its own context/generations.

The fix: **Generalized Knowledge Distillation (GKD)** (Agarwal et al. 2024 — the same "On-Policy Distillation" paper from your distillation guide), which mixes teacher-generated data with the student's *own* on-policy generations:

```
L(θ, θ_exp) = (1−λ)·E_{(x,y)~D}[ F(LM_θ‖LM_θexp)(y|x) ]
            +    λ ·E_{x~D}[ E_{y~LM_θexp(·|x)}[ F(LM_θ‖LM_θexp)(y|x) ] ]
```

`D` is sampled from the teacher `LM_θ`; `F(·‖·)(y|x)` is any divergence between teacher and student token distributions (forward KL, reverse KL, or JSD — same menu as MiniLLM/DistiLLM in your distillation guide); `λ ∈ [0,1]` sets the on-policy student-generated fraction. Two implementation details matter: **no gradient flows through the student's own sampling** (stability + speed — you don't backprop through the sampling operation itself), and **only the newly-expanded parameters are unfrozen** — everything else in the student stays frozen, which is precisely what prevents the transferred knowledge from interfering with (i.e., catastrophically forgetting) anything already stored elsewhere.

## Learning to Imitate (LTI): Adding RL on Top of Distillation

Distillation alone leaves a gap: the student's new parameters *store* the teacher's knowledge (it's accessible via logits), but the paper observes the student still only **weakly mimics** the teacher's actual sampling behavior — it "knows" but doesn't reliably "do." The fix is an RL imitation-learning pass on top.

Given teacher-generated dreams `D_T = {d^(1),…,d^(n)}`, LTI randomly samples a **prefix** of each `d^(i)` and asks the student to complete it; the completion `d̂^(i)` is scored by a blended reward:

```
r(d̂^(i); d^(i); LM_θexp) = γ · r_sem(d̂^(i); d^(i); LM_θexp)  +  (1−γ) · r_abs(d̂^(i); d^(i); LM_θexp)     (Eq. 3)
```

`r_sem` = a **frozen** reward model scoring semantic similarity: 1 if `d̂^(i)` and `d^(i)` mean the same thing, else 0. `r_abs` = a token-level similarity based on **Levenshtein edit distance** `z(·,·)`:

```
r_abs(·) = { 1 − z(d̂^(i), d^(i)) / max{|d̂^(i)|, |d^(i)|}     if z(d̂^(i), d^(i)) ≤ z_0                     (Eq. 4)
           { 0                                                otherwise
```

`z_0` = a similarity threshold — beyond it, the attempt is scored 0 rather than a small positive credit, so partial-but-too-different completions aren't rewarded. Combining LTI with the on-policy KS objective gives the paper's full Knowledge-Seeding loss:

```
L_KS(θ, θ_exp) = E_x[ (1−α)·E_{y~LM_θexp(·|x)}[ r(y) ]  −  α·E_{y~LM_θexp(·|x)}[ D(LM_θ‖LM_θexp)(y|x) ] ]
```

`α ∈ [0,1]` trades off imitation-reward strength against distillation strength. This is what's actually optimized to fit the new expanded parameters during a consolidation step. Afterward, the **old** low-rank experts that were added in *previous* sleep periods get reset — synaptic pruning, explicitly citing Li et al. (2017)'s finding that REM sleep prunes redundant synaptic connections to keep the system efficient.

**Implementation note from the paper**: growing sparse modules by literally changing tensor dimensionality is painful in practice. Their workaround — keep the new parameters present in the model from the start, but **masked out** of forward/backward until their sleep-triggered activation. This is explicitly framed as matching the brain's actual mechanism: a large but fixed set of neurons, with new *connections* forming and old ones pruning over a lifetime, rather than new neurons being grown.

## Dreaming: The Self-Modifying (REM) Stage

Where Memory Consolidation freezes higher-frequency parameters and distills their knowledge downward, Dreaming is the stage where the model is maximally active — generating and rehearsing its own synthetic curriculum to self-modify and strengthen newly-formed connections, the way REM sleep strengthens/prunes newly formed synapses.

Dreaming builds on **SEAL** (Zweiger et al. 2025/2026) but explicitly names three gaps SEAL has that Sleep is designed to close:

1. SEAL's inner loop needs a full SFT run per self-edit → expensive → limits you to very few self-edits.
2. Naively iterating self-improvement across many sleep periods risks catastrophic forgetting.
3. SEAL only *samples* from the model's existing knowledge space; it can't synthesize genuinely novel combinations (Stickgold 2005's point about dreaming's role in exploring beyond direct experience).

Given a sampled task `(C, τ)` (`C` = context relevant to the task, `τ(·)` = a downstream performance measure), Dreaming:

```
1. GENERATE     {DREAM^(i)}_{i=1}^m ~ LM_θ(·|C),   m ≥ 1

                Novelty trick: every MoE router in the model ADDITIONALLY selects
                one RANDOM expert (on top of its normal top-k choice) while generating
                each dream — deliberately injecting irrelevant/unrelated knowledge that
                the model cannot see is irrelevant, forcing it to learn underlying
                patterns rather than just recombine what it already explicitly knows.

2. SELECT       Reject most dreams; keep the ones with real potential to help.
                Importance score (GREATS-style, gradient-based):
                    g_DR^(i) = ∇_θ L_SFT(DREAM^(i); θ)
                Keep Top-k by g_DR^(i), plus b random dreams for diversity → set D.

3. TRAIN & TEST For each DREAM^(i) ∈ D, spin up an ISOLATED copy of the model and
                LoRA-SFT it:      θ'^(i) ← SFT(θ^(i), DREAM^(i))
                Reward (SEAL-style):
                    r(DREAM^(i); τ(·); LM_θ'^(i)) = 1  if LM_θ'^(i) improves over LM_θ^(i)      (Eq. 5)
                                                    0  otherwise

4. OPTIMIZE     Use ReST^EM (Singh et al. 2024) to reinforce the generation policy
                toward dreams that scored 1 — generate, filter-by-reward, SFT, repeat.
```

The gradient-based selection is exactly the GREATS mechanism from your ML-tooling knowledge of data-selection methods: approximate each candidate's effect on validation loss via gradient inner-products, rather than actually retraining on every candidate — here repurposed to pick which *synthetic* self-generated samples are worth spending an isolated LoRA-SFT run on.

## Benchmarks

All Qwen/Llama backbones; LoRA rank 64, α=128 throughout (Table 5). Training config: LR 5e-6, effective batch 32; GRPO 500 steps, SFT 100 steps, Sleep 100 steps.

### Class-incremental learning (Figure 3) — CLINC / Banking77 / DBpedia, Llama-3B & Llama3-8B

Sleep (Hope + memory consolidation) beats ICL, EWC (Kirkpatrick 2017), and InCA (Momeni 2025, an external-learner ICL baseline) across all three datasets and both backbones. Relative to plain Hope (no explicit distillation), Sleep's gain shows converting prompt-level adaptation into durable parametric memory beats repeated in-context updates alone.

### #Sleep levels vs in-context understanding (Figure 4) — LongHealth, QASPER, MK-NIAH (RULER)

Baselines: ICL, DuoAttention (Xiao 2025), Cartridges (Eyuboglu 2025). Two consistent trends across all three tasks:
1. More consolidation stages (memory levels) → better ICL / long-context performance (more effective parameters via better abstraction/compression).
2. Raising the **lowest** frequency (making the most-persistent memory tier update faster/less stable) → *worse* performance — the most durable memory should stay the most durable.

### Continual translation of a novel language (Figure 5) — MTOB Manchu + Kalamang

Hope-1/2/3 (1–3 consolidation stages) vs ICL. Single-language setting: all Hope variants match or exceed ICL. Sequential (continual) setting: ICL reverts sharply toward pretrained behavior; Hope retains substantially more, improving **monotonically** with more consolidation stages — Hope-3 nearly recovers single-language performance. Both Cartridges and plain SFT catastrophically forget in at least one language, performing *worse than ICL*.

### BABILong (Figure 6) — needle-in-a-haystack to 10M tokens

Hope (Sleep) stays near-perfect out to 10M tokens. GPT-4 / GPT-4o-mini and Llama-8B+RAG degrade sharply beyond 128K–256K. Titans and ARMT track Hope up to ~1M tokens, then degrade sharply; Hope alone stays stable to 10M.

### Reasoning (Table 2) — Qwen3-1.7B and Qwen3-8B, AIME-24 / AIME-25 / HMMT-25 (avg@16)

```
                   Qwen3-1.7B                    Qwen3-8B
Method        AIME24  AIME25  HMMT25       AIME24  AIME25  HMMT25
────────────  ──────  ──────  ──────       ──────  ──────  ──────
Base(Instr.)   49.8    34.5    25.7         73.8    68.1    42.4
SFT            47.3    36.1    22.9         75.5    66.4    43.7
GRPO           51.0    38.6    26.1         76.4    68.1    44.9
OPSD           51.6    40.0    28.1         76.6    67.4    45.1
Sleep          53.2    40.2    29.3         79.2    69.0    46.1   ← best on every column
```

### Ablations (Qwen3-8B, math reasoning)

```
Variant                 AIME24  AIME25  HMMT25
──────────────────────  ──────  ──────  ──────
Sleep (full)             79.2    69.0    46.1
− Imitation Learning     76.8    67.9    45.0
− Semantic Reward        78.9    69.2    44.5
− w/o Expansion          78.2    67.9    44.4
OPSD (no expansion)      76.6    67.4    45.1
OPSD + Expansion         77.9    68.2    45.9
```
Every component contributes positively; removing Imitation Learning hurts the most of the three ablated pieces.

### Knowledge incorporation (Table 3) — SQuAD, no-context QA

```
Method                                    Single Passage (n=1)   Continued Pretrain (n=200)
────────────────────────────────────────  ────────────────────   ──────────────────────────
Base model                                        31.9                    31.9
Fine-tuned, no Dreaming                           33.4                    32.0
SEAL                                               46.7                    43.2
Sleep (Transformer, 2-level)                       48.1                    44.3
Sleep (Transformer, 4-level)                       48.9                    46.2   ← best
  − gradient-based selection                       47.1                    45.2
  − random expert                                  48.0                    44.7
  − Dreaming                                       35.7                    36.2
```
Removing Dreaming alone drops CPT accuracy from 46.2 to 36.2 — the single biggest ablation effect in the whole paper, underscoring that consolidation and self-improvement are both load-bearing, not just consolidation.

### Few-shot abstract reasoning (Table 4) — ARC, Llama-3.2-1B

```
Method    Success Rate
────────  ────────────
ICL           0%
TTT           10%
SEAL          72.5%
Sleep         80%      ← best
```

### Efficiency (Appendix B.5)

Per training step, SFT is **4× cheaper** than Sleep — but to reach the *same* AIME-24 / AIME-25 / HMMT-25 accuracy Sleep achieves, SFT needs **4.3× / 3.6× / 4.8×** the wall-clock time. I.e. Sleep is more expensive per step but net cheaper when you target a fixed performance bar.

## Sleep vs. Alternatives

```
Method             Mechanism                        Avoids CF?   Needs ext.  Grows params?  AIME-24 (Qwen3-8B)
                                                                   data at
                                                                   sleep time?
─────────────────  ───────────────────────────────  ───────────  ──────────  ─────────────  ───────────────────
ICL                context-window adaptation only    yes (nothing  n/a         no             73.8 (base, no adapt)
                                                       overwritten)
SFT                gradient fine-tune on new data     NO            yes         no             75.5
GRPO               RL fine-tune, verifiable reward    NO            yes         no             76.4
EWC                regularize toward old weights      partial       yes         no             (loses to Hope, Fig.3)
Cartridges         pretrained KV-cache adapter         NO (worse     no (self-   no             (loses to Hope, Fig.4/5)
                    via self-study                     than ICL      study only)
                                                        in Fig. 5)
Titans             neural long-term memory, test-time yes           n/a         no             (degrades >1M ctx,
                    learned memorization                                                        BABILong Fig.6)
SEAL               self-edit generation + RL reward   partial       no (self-   no             (Sleep beats SEAL
                                                        (Sleep beats  generated)                 on Tables 3 & 4)
                                                        it on CF)
Sleep (this paper) upward distillation + parameter    yes (by        no          yes            79.2   ← best
                    expansion + Dreaming                design)       (self-
                                                                       generated)
```

## Practical Considerations

**When Sleep-style consolidation is worth it:**
- Long-lived deployed systems that must absorb new facts/skills over many sessions without periodic expensive full retraining and without catastrophic forgetting.
- Settings where you can tolerate an explicit offline "sleep" phase (extra wall-clock, no user-facing latency during it) between bursts of active use.
- Anywhere ICL currently "leaks" — you keep re-explaining the same context every session because nothing persists.

**When it isn't (yet):**
- **VERY_NEW** — no confirmed public code, single-paper results, thin independent replication as of this writing (July 2026). One visible OpenReview review snippet is only "marginally below acceptance threshold" — treat this as an open question, not resolved.
- Adds real engineering complexity: masked/inactive parameters, MoE routers per FFN tier, an RL loop (LTI) and a second RL loop (Dreaming/ReST^EM) on top of ordinary training — meaningfully more moving parts than SFT/GRPO alone.
- Efficiency tradeoff: 4× more expensive per step than plain SFT; only pays off if you specifically need CF-resistance and can't get equivalent quality any other way (their own numbers show it *does* pay off to match target accuracy, just not on a per-step basis).
- Requires a reward model or verifiable-reward signal for both the LTI semantic reward and the Dreaming ReST^EM loop — not free for arbitrary open-ended domains.

**Failure modes flagged by the paper itself**: naive iterative self-improvement (plain SEAL-style, without the consolidation stage first) risks catastrophic forgetting (this is literally why Sleep separates consolidation from Dreaming into two stages instead of one); raising the *lowest* CMS frequency (making the most-persistent memory less stable) reliably hurts long-context retention (Figure 4) — don't make your slowest tier update too eagerly.

## Connection to Your Prior Knowledge

- **vs your GRPO tutorial**: GRPO optimizes a fixed policy against a verifiable reward with no architectural change. Sleep's Learning-to-Imitate and Dreaming stages are both RL loops in the same spirit (policy-gradient-style optimization against a reward), but layered on top of an architecture that's *itself* growing (new low-rank experts) — RL is the training signal for consolidation and self-improvement, not the whole story.
- **vs your DPO/SFT hand-written loops**: SFT and DPO both train against a fixed dataset/preference set with no persistence mechanism between rounds. Sleep's Table 2 explicitly benchmarks plain SFT and shows it *underperforms* Sleep at equal or even higher per-step cost when the objective is matching a target accuracy (Appendix B.5) — the extra machinery (expansion + distillation + RL) earns its keep specifically on continual/repeated-update workloads, not one-shot fine-tuning.
- **vs your knowledge-distillation study**: ordinary KD (Hinton temperature-softmax, response/logit/CoT distillation, DeepSeek-R1's downward distillation) always compresses a bigger teacher into a smaller student. Knowledge Seeding inverts this — smaller-self teaches larger-self — precisely because the "smaller" model here isn't weaker in the usual sense, it's the *same* model with slower-tier capacity not yet unlocked. The GKD on-policy machinery you already studied (mixing teacher and student-generated samples, avoiding train/inference distribution mismatch) is reused nearly verbatim.
- **vs your MoE study**: each CMS tier is a routed sparse MoE, and the whole consolidation mechanism *is* an MoE-growth trick — add one new low-rank expert, train only it, then keep it always-selected via the router. The Dreaming stage's "each router additionally picks one random expert" trick is a deliberate abuse of MoE routing (forcing irrelevant-expert mixing) that's the opposite of what your aux-loss-free load-balancing study optimizes *against* (there you want routing to be clean and specialized; here noise is injected on purpose for novelty).
- **vs the Sleep-offline-recurrence cousin guide**: that paper ([../inference/llm-sleep-offline-recurrence-guide.md](../inference/llm-sleep-offline-recurrence-guide.md)) buys reasoning *depth* at inference time by looping a fixed SSM architecture `N` times over a context chunk before evicting KV — no parameter growth, no RL, purely an inference-time schedule change on a frozen model. This paper buys continual-learning *durability* by growing new parameters and running two RL loops during an explicit offline phase between deployment sessions — a post-training/continual-learning method, not an inference-time trick. Orthogonal axes; nothing stops combining both.

## Summary

- **Problem**: LLM knowledge lives only in the (fragile, evicted) context window or the (permanent, frozen) pretrained weights — no mechanism bridges the two, exactly mirroring anterograde amnesia.
- **Idea**: replace static train/test with a periodic **Wake/Sleep** lifecycle. Sleep = (1) **Memory Consolidation** — Compute→Consolidate→Update protocol that upward-distills a faster CMS tier into a newly-grown low-rank expert in the next, slower tier (Knowledge Seeding = GKD on-policy distillation + RL-based Learning-to-Imitate), then prunes the old experts (synaptic pruning); (2) **Dreaming** — generate synthetic self-data with random-expert novelty injection, GREATS-style gradient-based selection, per-dream isolated LoRA-SFT, ReST^EM reward loop (built on, and fixing three gaps in, SEAL).
- **Architecture**: Hope — attention (∞ frequency) → High-Freq FFN (period 1,000) → Mid-Freq FFN (period 5,000) → Low-Freq FFN (period 10,000), each FFN tier a routed sparse MoE; 5 extra low-rank blocks of dim 64, active parameter count unchanged from base model.
- **Results**: best on every reported benchmark family — class-incremental learning (CLINC/Banking/DBpedia), #consolidation-levels vs long-context understanding (LongHealth/QASPER/MK-NIAH), continual novel-language ICL (MTOB Manchu/Kalamang), BABILong to 10M tokens, math reasoning (Qwen3-8B AIME-24 79.2 vs GRPO's 76.4 and SFT's 75.5), knowledge incorporation (SQuAD no-context CPT 46.2 vs SEAL's 43.2), few-shot ARC (80% vs SEAL's 72.5%).
- **Cost**: 4× more expensive per step than SFT, but 3.6–4.8× cheaper wall-clock to *match* a target accuracy.
- **Status**: VERY_NEW (arXiv posted June 2026, publicly on OpenReview since Sept 2025), no confirmed public code, thin independent discussion so far — treat as a promising but unreplicated research direction, not a settled result.

## Key Papers

| Paper | Authors | Year | Contribution |
|-------|---------|------|---------------|
| [Loss of recent memory after bilateral hippocampal lesions](https://pmc.ncbi.nlm.nih.gov/articles/PMC497229/) | Scoville & Milner | 1957 | H.M./anterograde amnesia — the paper's founding analogy |
| [Learning to control fast-weight memories](https://people.idsia.ch/~juergen/fastweights/ncfastweightsrev.html) | Schmidhuber | 1992 | Fast/slow weight split; the update-frequency worked example |
| [Complementary learning systems in hippocampus and neocortex](https://www.semanticscholar.org/paper/2ebf18e7892e660a833152ddc6cf8f1d21a7b881) | McClelland, McNaughton, O'Reilly | 1995 | Fast hippocampal + slow cortical memory — CMS's two-tier ancestor |
| [Reverse replay of behavioural sequences in hippocampal place cells](https://www.nature.com/articles/nature04587) | Foster & Wilson | 2006 | Awake-state hippocampal replay |
| [Sleep function and synaptic homeostasis](https://pubmed.ncbi.nlm.nih.gov/16376591/) | Tononi & Cirelli | 2006 | Slow-wave sleep downscales synaptic strength |
| [Distilling the knowledge in a neural network](https://arxiv.org/abs/1503.02531) | Hinton, Vinyals, Dean | 2015 | Classical (downward) KD — the direction Knowledge Seeding inverts |
| [Complementary learning systems theory updated](https://doi.org/10.1016/j.tics.2016.05.004) | Kumaran, Hassabis, McClelland | 2016 | Updated CLS theory |
| [REM sleep selectively prunes and maintains new synapses](https://www.nature.com/articles/nn.4479) | Li, Ma, Yang, Gan | 2017 | Direct inspiration for the paper's synaptic-pruning reset step |
| [Overcoming catastrophic forgetting in neural networks (EWC)](https://www.pnas.org/doi/10.1073/pnas.1611835114) | Kirkpatrick et al. | 2017 | Elastic Weight Consolidation — continual-learning baseline |
| [LoRA: Low-Rank Adaptation of LLMs](https://openreview.net/forum?id=nZeVKeeFYf9) | Hu et al. | 2022 | Low-rank adapters — the mechanism behind every "expert" added |
| [On-Policy Distillation of LMs / GKD](https://arxiv.org/abs/2306.13649) | Agarwal et al. (Google DeepMind) | 2024 | On-policy + off-policy mixture distillation — the Knowledge-Seeding base objective |
| [Beyond Human Data: Scaling Self-Training (ReST-EM)](https://openreview.net/forum?id=lNAyUngGfK) | Singh et al. | 2024 | Generate→filter-by-reward→SFT loop used to optimize Dreaming |
| [GREATS: Online Selection of High-Quality Data for LLM Training](https://proceedings.neurips.cc/paper_files/paper/2024/file/ed165f2ff227cf36c7e3ef88957dadd9-Paper-Conference.pdf) | Wang, Wu, Song, Mittal, Jia | 2024 | Gradient-inner-product data selection — Dreaming's dream-selection mechanism |
| [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663) | Behrouz, Zhong, Mirrokni | 2024 | Neural long-term memory module — Hope's architectural ancestor |
| [Nested Learning: The Illusion of Deep Learning Architectures](https://openreview.net/forum?id=nbMeRvNb7A) | Behrouz, Razaviyayn, Zhong, Mirrokni | 2025 | Introduces CMS + Hope; NeurIPS 2025 |
| [Cartridges](https://arxiv.org/abs/2506.06266) | Eyuboglu et al. | 2025 | Pretrained KV-cache adapter via self-study — main long-context baseline, loses to Sleep |
| [Self-Adapting Language Models (SEAL)](https://arxiv.org/abs/2506.10943) | Zweiger, Pari, Guo, Akyürek, Kim, Agrawal | 2025/2026 | Self-generated SFT edits + RL reward — Dreaming's direct base, and strongest baseline in 2 of 4 experiments |
| [Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories](https://arxiv.org/abs/2606.03979) | Behrouz, Hashemi, Javanmard, Mirrokni | 2026 | **This guide** — Wake/Sleep lifecycle, upward Knowledge Seeding, RL-driven Dreaming |
