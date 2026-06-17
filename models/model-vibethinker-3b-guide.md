# VibeThinker-3B: Frontier Verifiable Reasoning in a 3B Dense Model

How Sina Weibo took a plain **Qwen2.5-Coder-3B** base and, through post-training alone (no new architecture), pushed it to **AIME26 94.3 / 97.1 with test-time scaling** — matching DeepSeek V3.2 (671B), Kimi K2.5 (1T), GLM-5 (744B), and Gemini 3 Pro on competition math while running at 1/100–1/300 their parameter scale. The conceptual payload is the **Parametric Compression-Coverage Hypothesis**: verifiable reasoning compresses into a small reusable "reasoning core," whereas open-domain knowledge needs broad parameter coverage.

## Background

VibeThinker-3B (Sen Xu, Shixi Liu, Wei Wang, Jixin Min, Yingwei Dai, Zhibin Yin, Yirong Chen, Xin Zhou, Junlin Zhang — Sina Weibo Inc., June 15 2026, [arxiv 2606.16140](https://arxiv.org/abs/2606.16140)). 14-page technical report. [Model](https://huggingface.co/WeiboAI/VibeThinker-3B) · [GitHub](https://github.com/WeiboAI/VibeThinker).

This is a post-training *system*, not an architecture paper. Its lineage is the GRPO/verifier-RL family you already know, applied at extreme small scale:

1. **DeepSeekMath / GRPO** (DeepSeek-AI, 2024, [arxiv 2402.03300](https://arxiv.org/abs/2402.03300)) — group-relative advantage with no critic. The RL substrate. VibeThinker's MGPO is a reweighted GRPO.

2. **DeepSeek-R1** (DeepSeek-AI, 2025, [arxiv 2501.12948](https://arxiv.org/abs/2501.12948)) — pure-RL emergent reasoning, plus the *small-model distillation branch* (R1 → Qwen-1.5B/7B). VibeThinker is the spiritual successor to that branch: instead of distilling a giant teacher into a small model, it *grows* reasoning in the small model directly. It's also the large-model baseline VibeThinker positions against.

3. **DAPO** (ByteDance Seed, 2025, [arxiv 2503.14476](https://arxiv.org/abs/2503.14476)) — open-source large-scale RL system; decoupled clipping + dynamic sampling. Same thesis that *sampling distribution and RL stability* are the real levers in long-CoT RL.

4. **VibeThinker-1.5B** ("Tiny Model, Big Logic: Diversity-Driven Optimization Elicits Large-Model Reasoning Ability", Sina Weibo, 2025, [arxiv 2511.06221](https://arxiv.org/abs/2511.06221)) — **the direct predecessor and source of truth for the named methods.** Introduced the *Spectrum-to-Signal Principle (SSP)*, *Diversity-Exploring Distillation*, and *MGPO*. It proved a 1.5B model *could* produce coherent long-horizon reasoning (~$7.8K post-training cost). VibeThinker-3B asks the follow-up question: not "can it reason at all?" but "**how much parameter capacity does a small model actually need to enter the first tier?**"

The arc:

```
DeepSeekMath (GRPO)  →  DeepSeek-R1 (pure-RL + small-model distill branch)
                                    │
                                    ▼
              VibeThinker-1.5B (SSP + MGPO; "can a tiny model reason?")
                                    │
                                    ▼
              VibeThinker-3B (full post-train system; "what's the parameter
                              threshold for first-tier verifiable reasoning?")
```

Every method here is a *recombination* of things in your Knows list: MGPO ≈ GRPO with boundary-prompt selection; offline self-distillation ≈ R1's rejection-sampling + response distillation; CLR ≈ self-consistency + verifier (ORM) voting at test time; Long2Short ≈ length-penalty reward shaping. The novelty is the *system integration* at 3B and the *hypothesis* explaining why it works.

## What Problem Does VibeThinker-3B Solve?

The field's default move for hard reasoning is **scale up** — cross the parameter threshold that scaling laws say difficult math/code needs, landing you at tens-to-hundreds of billions of parameters. Small models (≤3B) are treated as deployment-efficiency fallbacks, presumed to have an inherent reasoning ceiling.

VibeThinker-3B's claim: that ceiling is an artifact of *training*, not *capacity* — at least for **verifiable** tasks. The key insight is that verifiers change the learning problem:

```
Open-ended dialogue          Verifiable reasoning (math/code/STEM)
────────────────────         ──────────────────────────────────────
reward = fuzzy / RM-judged   reward = checkable (answer match, unit
                                       tests, sandbox execution)
signal noisy, sparse         signal clean, dense, automatable
→ needs scale + RLHF         → RL + rejection filtering become very
                               powerful even at 3B
```

When the environment can reliably say "correct / incorrect," RL and rejection sampling extract far more per parameter. The bet: math/code reasoning is fundamentally **search + constraint satisfaction + error correction + multi-step composition inside a structured solution space** — a compact, reusable *procedure* — whereas "what year did X happen" requires *storing* the fact. Procedures compress; facts don't.

The empirical signal that motivates the whole paper:

```
              Verifiable reasoning          Knowledge-heavy
              (AIME26, HMMT, LCB)           (GPQA-Diamond)
              ──────────────────            ──────────────
VibeThinker-3B   94.3 / 89.3 / 80.2            70.2
DeepSeek V3.2    94.2 / 90.2 / 80.8            82.4   ← 3B matches on reasoning
GLM-5 (744B)     95.8 / 97.9 / 85.5            86.0   ← but lags ~12-16 pts
Kimi K2.5 (1T)   93.3 / 95.4 / 85.0            87.6      on knowledge
```

The 3B model *closes* the reasoning gap but *keeps* the knowledge gap. The authors read this not as a failure but as **evidence for** their hypothesis: the two capabilities have different parameter demands.

## Key Terms

**Spectrum-to-Signal Principle (SSP)**: The organizing idea. SFT builds a broad *spectrum* of diverse valid reasoning paths (don't collapse to one solution); RL then amplifies the high-value *signal* within that spectrum. The SFT stage deliberately does **not** optimize for single-path imitation — it maximizes the exploration basis for RL.

**Diversity-Exploring Distillation**: In SFT, sample *multiple* candidate reasoning traces per query (multi-path distillation) and keep them all, preserving diverse decomposition methods, derivation paths, and verification strategies. Periodically save checkpoints, score them by **Pass@K** on per-domain probe sets, pick each domain's best-diversity checkpoint as a specialist, then **merge specialists at the parameter level** into one SFT model. (Selecting on Pass@K, not lowest val-loss or Pass@1, is the whole point — it optimizes for solution-space breadth.)

**MGPO (MaxEnt-Guided Policy Optimization)**: GRPO reweighted to focus updates on prompts at the model's *capability boundary* — where rollouts are neither all-correct nor all-wrong (maximum uncertainty / entropy). Defined below.

**Long2Short Math RL**: A second math-RL stage that shifts reward from "accuracy" to "accuracy at fewer tokens" — reshapes preferences among *correct* trajectories toward shorter ones, zero-sum within each prompt group so the accuracy baseline is unchanged.

**Offline Self-Distillation**: After RL, collect verified trajectories from the Math/Code/STEM RL checkpoints, filter by a *learning-potential* score, and distill them back into one unified student via SFT — consolidating multi-domain skills into a single stable model.

**Instruct RL**: A final RL stage that converts the reasoning-heavy checkpoint into a reliable user-facing model — rule-based validators for constraints (format, ordering, item count, keywords) + rubric-based reward models for open-domain helpfulness.

**CLR (Claim-Level Reliability Assessment)**: A test-time scaling strategy. Instead of voting over whole answers, extract the *decision-critical claims* from each candidate trajectory, self-verify each claim, and aggregate by a nonlinear reliability score. The "+CLR" numbers in every table.

**Parametric Compression-Coverage Hypothesis**: The paper's thesis — foundational capabilities differ in the *structural form* of their parameter demand. **Parameter-dense** (verifiable reasoning) → compressible into a small reusable core. **Parameter-expansive** (open-domain knowledge, long-tail facts, broad semantics) → require coverage that scales with parameters.

## The Training Pipeline

Four stages over a Qwen2.5-Coder-3B base. No architecture changes — pure post-training.

```
  Qwen2.5-Coder-3B (base)
         │
         ▼
┌─────────────────────────────┐
│ 1. SFT (2-stage curriculum) │  Diversity-Exploring Distillation
│   Stage 1: broad coverage   │  (math, code, STEM, chat, instr.)
│   Stage 2: hard long-CoT    │  filter: error-rate ≥ 0.75, trace ≥ 5K tok
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ 2. Reasoning RL  (MGPO)     │  sequential: Math → Code → STEM
│   single 64K context window │  verifiable rewards per domain
│   Long2Short (math only)    │  accuracy → token-efficiency
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ 3. Offline Self-Distillation│  collect verified RL traces,
│   merge Math/Code/STEM       │  filter by learning-potential, re-SFT
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ 4. Instruct RL              │  constraint validators + rubric RMs
│                             │  → user-facing controllability
└─────────────────────────────┘
         │
         ▼
   VibeThinker-3B   (+ optional CLR test-time scaling)
```

### Stage 1 — SFT: build the spectrum

Multi-domain mixed SFT for a stable cold-start RL policy. Data pipeline: take only seed queries with reliable supervision (math with credible final answers; code with unit tests), then **query expansion** — rewrite/expand across concept composition, problem skeletons, evaluation objectives — and generate pseudo-labels by multi-sample majority voting from strong teachers. Three-level quality control: (1) n-gram filtering (kills templated degeneration + benchmark contamination), (2) LLM query-quality filtering (drops ill-posed problems), (3) trace-correctness filtering (answer verification + code-sandbox execution + LLM majority voting).

**Curriculum, two stages**:
- **Stage 1 (broad)**: full quality-filtered dataset, maximize task/reasoning diversity. Sequence packing (CoT lengths vary a lot). Global batch 128, LR 5×10⁻⁵ cosine → 8×10⁻⁸ min, 5% linear warmup, **5 epochs**.
- **Stage 2 (hard)**: init from Stage-1 checkpoint. Build a hard subset by *joint length-difficulty filtration*: discard traces shorter than 5K tokens; using VibeThinker-1.5B as a reference model, do **8 rollouts per query** and keep only problems with error rate **below 0.75** (i.e., reference gets them wrong ≥75% — genuinely hard). Same hyperparams, **2 epochs**. This forces Stage 2 onto long-horizon derivation and complex constraint satisfaction.

### Stage 2 — Reasoning RL with MGPO

The core RL algorithm. For each prompt `q`, sample `G` responses from the old policy and score with verifiable rewards. Compute the **empirical group accuracy**:

```
        1   G
p(q) =  ─   Σ   1(r_i = 1)                                    (Eq 1)
        G  i=1
```

The MGPO insight: prompts with `p(q) ≈ 0` are too hard (sparse positive signal); `p(q) ≈ 1` are saturated (nothing to learn). The useful prompts sit at intermediate correctness — the **capability boundary**, which for a binary outcome is the **maximum-entropy point p = 0.5**. So weight each prompt by how close it is to 0.5:

```
w(q) = exp( −γ · D_ME( p(q) ‖ p_0 ) ),   p_0 = 0.5,  γ > 0    (Eq 2)
```

`D_ME` measures deviation of `p(q)` from the max-entropy point 0.5. Closer to 0.5 → smaller deviation → `w(q) ≈ 1` (full weight); near 0 or 1 → large deviation → `w(q) → 0` (downweighted). This weight multiplies the standard GRPO-style clipped, group-relative objective:

```
                       1  G   1  |y_i|
J_MGPO(θ) = E_q,{y_i} [ ─  Σ  ──  Σ   min( ρ_{i,t}(θ)·w(q)·A_i,
                       G i=1 |y_i| t=1
                              clip(ρ_{i,t}(θ), 1−ε, 1+ε)·w(q)·A_i ) ]   (Eq 3)
```

where `A_i` is the group-relative advantage (same as GRPO), `ρ_{i,t}(θ)` is the token-level prob ratio new/old, `ε` the clip coefficient. **Versus the GRPO you trained**: the only structural change is the `w(q)` prompt weight — concentrate gradient on boundary-uncertainty prompts. This stabilizes updates and curbs over-optimization on already-confident tokens.

**Numerical walkthrough** (γ = 1, using binary-entropy-style deviation `D = (p−0.5)²` as illustration):

```
p(q)    deviation from 0.5     w(q)=exp(−D)     effect
────    ──────────────────     ────────────     ──────────────────
0.05    0.2025                 0.817            mostly skipped (too hard)
0.25    0.0625                 0.939            moderate
0.50    0.0                    1.000            FULL weight (boundary)
0.75    0.0625                 0.939            moderate
0.95    0.2025                 0.817            mostly skipped (saturated)
```

(The exact `D_ME` form isn't given numerically in the paper; the shape — peak at 0.5, decaying both ways — is the point.)

**RL practicalities** (where 3B differs from 1.5B):
- **On-policy only.** As the rollout engine gets optimized for throughput, the train/inference probability mismatch is amplified and can collapse RL. They run all stages strictly on-policy (citing the mismatch-praxis / off-policy-RL stabilization work).
- **Single 64K context window, no progressive expansion.** VibeThinker-1.5B benefited from gradually growing the context, but at 3B a high-truncation early stage *hurts* — the stricter SFT means fewer noisy traces to prune, so aggressive early truncation just disrupts good long-horizon behavior that can't be recovered later. So: one 64K window throughout.
- **Sequential domains**: Math RL (symbolic, long-horizon, multi-step search) → Code RL (executable-logic rigor, edge cases) → STEM RL (multidisciplinary generalization). Each checkpoint is preserved for the self-distillation stage.

### Long2Short Math RL — accuracy → efficiency

After the accuracy-first Math RL, a second pass trades verbosity for concision **without touching accuracy**. For each prompt group, keep all incorrect trajectories unchanged. For the correct set `C = {i | r_i = 1}`, define a brevity score `s_i = 1/L_i` (`L_i` = response length) and apply a *centered*, length-aware reward shift:

```
                 s_i − s̄
r'_i = r_i + λ · ──────────────  ,   i ∈ C,   λ = 0.2          (Long2Short)
                 max_{j∈C}|s_j − s̄|
```

`s̄` = mean brevity over correct trajectories. Shorter-than-average correct answers get a reward bump; longer ones get docked. Because it's centered, the shifts sum to zero within `C`:

```
  Σ_{i∈C} (r'_i − r_i) = 0
```

So the group-mean reward (the GRPO advantage baseline) is unchanged — no systematic shift in advantage estimation, the model just re-prefers concise correct paths. This is exactly length-penalty reward shaping, done in a baseline-preserving way. Concretely: two correct solutions to the same AIME problem, one 12K tokens and one 4K, now produce a *relative* preference for the 4K path even though both got reward 1.

### Stage 3 — Offline Self-Distillation

Collect trajectories from the Math/Code/STEM RL checkpoints (each a domain specialist), rejection-sample with domain verifiers to drop wrong traces, then rank the *correct* ones by **learning potential** — the length-normalized NLL of the trace under the *current student*:

```
                1  |y|
S_LP(q,y) = −  ──  Σ   log π_θstu( y_t | q, y_<t )            (Eq 4)
               |y| t=1
```

High `S_LP` = the student finds this verified trace *surprising* (poorly modeled) → high distillation value (it teaches something new). To avoid the score being dominated by sequence length or a few weird tokens: compute priorities within per-domain length buckets, exclude extremely short traces, filter extreme-high-score outliers (format errors / noise), and keep the **middle-to-high** range. Mix Math/Code/STEM and re-SFT into one unified model. This is R1-style rejection-sampling distillation, but **self**-distillation (teacher = own RL checkpoints) with an information-gain selection criterion instead of taking everything.

### Stage 4 — Instruct RL

Convert the reasoning model into a controllable assistant. Mixed instruction dataset (format-sensitive prompts, long-context instructions, alignment examples). Rewards: rule-based validators for explicit constraints (format, ordering, item count, keyword, task completion) + rubric-based RMs for open-ended prompts (helpfulness, coherence, adherence, non-redundancy). Same on-policy RL framework. The payoff is in the numbers: **IFEval 93.4** confirms that extreme reasoning optimization did *not* wreck instruction-following.

## CLR — Claim-Level Reliability Test-Time Scaling

The "+CLR" boost. Most test-time scaling (self-consistency, BoN) votes over *whole answers*. CLR votes over *claims*. Two-stage procedure:

```
1. Generate K=32 candidate trajectories per problem (same sampling as eval).
   From each trajectory k, extract M=5 decision-relevant claims.

2. The model acts as its own verifier: for each claim, attempt to
   falsify/validate → binary verdict v_{k,m} ∈ {0,1}.

   Nonlinear trajectory reliability (heavily penalizes ANY flawed claim):

              ⎛  1   M        ⎞ M
       r_k =  ⎜  ─   Σ  v_{k,m}⎟                                (Eq 5)
              ⎝  M  m=1       ⎠

3. Cluster candidate answers by equivalence; pick the answer maximizing
   reliability-weighted aggregation:

       Score(G) =   Σ      r_k                                  (Eq 6)
                 {k | y_k ∈ G}
```

The outer power `M` in Eq 5 is the teeth: a trajectory where 4/5 claims verify scores `(0.8)^5 = 0.328`; all 5 verify → `1.0`; 3/5 → `(0.6)^5 = 0.078`. One bad intermediate claim collapses the trajectory's vote. This isolates the critical logical anchors and discards noise from long verbose traces — *and* it's cheaper than re-reading the whole trace, because you only verify 5 extracted claims, not the full chain. Run the entire flow 8× and average (reported as "+CLR").

```
CLR worked example (3 candidates, all reach answer "42"):
  traj A: claims 5/5 verify → r = 1.0^5      = 1.000
  traj B: claims 4/5 verify → r = 0.8^5      = 0.328
  traj C: claims 5/5 verify → r = 1.0^5      = 1.000
  Score("42") = 1.000 + 0.328 + 1.000 = 2.328   ← wins vs any cluster
                                                   with lower summed r_k
```

Contrast with plain self-consistency, which would just count 3 votes for "42" regardless of whether the *reasoning* was sound. CLR is self-consistency where each vote is weighted by how well its chain survives self-verification.

## Results

All VibeThinker-3B evals: vLLM backend, temperature 1.0, top-p 0.95, top-k −1. Math = mean Pass@1 over 64 generations (IMO-Ans over 16); knowledge/code over 16/8 generations. Strict decontamination.

**Table 1 — vs small (<14B) and large reasoning models** (VibeThinker = no CLR):

| Model | Params | AIME25 | AIME26 | HMMT25 | BruMO25 | IMO-Ans | LCBv6 | OJBench | GPQA-D | IFEval | IFBench |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SmolLM3 | 3B | 36.7 | 41.0 | 26.0 | 49.2 | 28.7 | 29.1 | 5.2 | 41.7 | 71.2 | 27.6 |
| Qwen3-4B-Thinking | 4B | 81.3 | 79.0 | 55.5 | 77.7 | 51.6 | 55.2 | 17.9 | 65.8 | 87.4 | 52.9 |
| OpenReasoning-Nemotron | 7B | 78.2 | 80.2 | 63.5 | 78.8 | 60.6 | 64.9 | 25.9 | 61.1 | 44.0 | 31.3 |
| Phi4-Reasoning-Plus | 14B | 68.4 | 73.6 | 50.3 | 66.5 | 46.2 | 56.8 | 14.4 | 81.9 | 84.9 | 51.7 |
| GPT-OSS-20B (high) | 20B | 91.7 | 90.2 | 76.7 | 86.7 | 61.9 | 61.0 | — | 71.5 | 92.8 | 65.0 |
| GLM-4.5-Air | 106B | 83.3 | — | 69.2 | 90.0 | — | 70.7 | — | 75.0 | 86.3 | 37.6 |
| Qwen3-235B-A22B-Thinking | 235B | 92.3 | — | 83.9 | — | 70.5 | 74.1 | 32.5 | 81.1 | 87.8 | 51.2 |
| **VibeThinker-3B** | **3B** | **91.4** | **94.3** | **89.3** | **93.8** | **76.4** | **80.2** | **38.6** | **70.2** | **93.4** | **74.5** |

A 3B model leads every <14B baseline by a wide margin, beats GPT-OSS-20B on AIME26/HMMT25/LCBv6, and tops the table on LiveCodeBench v6 and OJBench. It trails only on GPQA-Diamond (knowledge).

**Table 2 — vs top-tier flagships, with CLR**:

| Model | Params | AIME25 | AIME26 | HMMT25 | BruMO25 | IMO-Ans | LCBv6 | GPQA-D |
|---|---|---|---|---|---|---|---|---|
| DeepSeek V3.2 | 671B | 93.1 | 94.2 | 90.2 | 96.7 | 78.3 | 80.8 | 82.4 |
| Kimi K2.5 | 1T | 96.1 | 93.3 | 95.4 | 98.3 | 81.8 | 85.0 | 87.6 |
| GLM-5 | 744B | 96.7 | 95.8 | 97.9 | — | 82.5 | 85.5 | 86.0 |
| Gemini 3 Pro | N/A | 96.0 | 91.7 | 97.5 | 98.3 | 83.1 | 87.4 | 91.9 |
| Claude Opus 4.5 | N/A | 92.8 | 95.1 | 92.9 | — | 78.5 | 84.8 | 87.0 |
| GPT-5 (high) | N/A | 94.6 | — | 88.3 | 91.7 | 76.0 | 84.5 | 85.7 |
| **VibeThinker-3B** | **3B** | 91.4 | 94.3 | 89.3 | 93.8 | 76.4 | 80.2 | 70.2 |
| **  + CLR** | **3B** | **96.7** | **97.1** | **95.4** | **99.2** | **80.6** | — | **72.9** |

With CLR, the 3B model's AIME26 (97.1) and BruMO25 (99.2) sit at or above every flagship listed. The persistent exception is **GPQA-Diamond** — even +CLR only reaches 72.9 vs 82–92 for the giants. That ~12-19 point knowledge gap is the hypothesis made visible.

**IMO-AnswerBench parameter efficiency** (Fig 2, 400 IMO-level problems):

```
VibeThinker-3B    3B     ████████████████  76.4  (+CLR 80.6)
DeepSeek V3.2     671B   ████████████████  78.3   (223× the params)
Kimi K2.5         1T     █████████████████ 81.8   (333×)
GLM-5             744B   █████████████████ 82.5   (248×)
```

**OOD test — recent LeetCode contests** (Apr 25–May 31 2026, fully post-cutoff, Python one-shot, 16 submissions/contest): VibeThinker-3B passes **123/128 = 96.1%**, above GPT-5.2 (95.3%), Doubao Seed 2.0 Pro, Qwen3-Max, Kimi K2.5, Claude 4.6; just under GPT-5.3-Codex (100%) and Gemini 3.1 Pro (99.2%). Fresh problems → not benchmark memorization.

## VibeThinker-3B vs Alternatives

| Dimension | VibeThinker-3B | Large reasoning LLM (e.g. GLM-5 744B) | R1-distilled small (R1→Qwen-7B) |
|---|---|---|---|
| Params | 3B dense | 100B–1T (MoE) | 1.5–7B dense |
| How reasoning is acquired | grown in-place via SSP + multi-domain RL | scale + RL | distilled from a giant teacher's traces |
| Verifiable math (AIME26) | 94.3 / 97.1 +CLR | 95.8 | far lower (R1-Qwen-7B ≈ 50s) |
| Knowledge (GPQA-D) | 70.2 / 72.9 | 86.0 | low |
| Deploy cost | single small GPU | multi-GPU / cluster | small GPU |
| Best-fit use | competitive math / code / STEM with checkable answers | general assistant, broad knowledge, agents | cheap reasoning, weaker |
| Scope warning | **not** for tool-calling / agentic / open-domain | full-spectrum | limited |

## Practical Considerations

**When VibeThinker-3B fits**: verifiable-answer domains — competition math, algorithmic coding, STEM with checkable outputs — where you want frontier-ish accuracy on a single small GPU. The 64K context preserves long derivations. CLR is a drop-in accuracy boost (no weight changes) when you can afford ~32 samples + a self-verification pass.

**When it doesn't**: the model card explicitly **discourages tool-calling, API orchestration, and autonomous coding-agent use**. The GPQA-Diamond gap is real — for open-domain knowledge, long-tail facts, or broad assistant duties, a 3B reasoning core won't substitute for a large general model. This is by design, not a bug: the Reasoning-Knowledge Decoupling view says large models remain the vehicle for knowledge breadth; small models encapsulate reasoning depth.

**Reproducing the recipe** (the transferable lessons for your own post-training):
1. **Don't collapse SFT to one solution path** — Diversity-Exploring Distillation + Pass@K checkpoint selection give RL a wider exploration basis. Single-path imitation caps the RL ceiling.
2. **MGPO over vanilla GRPO** when compute is tight — one extra `w(q)=exp(−γ·D_ME(p(q)‖0.5))` term concentrates gradient on boundary prompts; cheap, stabilizing. You could bolt this onto the GRPO loop you hand-wrote.
3. **On-policy RL** — the train/inference mismatch from an optimized rollout engine is the silent RL killer; keep it on-policy if stability matters.
4. **Long2Short as a separate stage** — chase accuracy first, *then* compress with baseline-preserving length-reward shaping. Don't co-optimize from the start.
5. **Self-distillation with a learning-potential filter** — distilling *everything* wastes the student's capacity; rank verified traces by NLL-under-student and keep the middle-to-high (surprising-but-not-noise) band.

**Caveats / open questions**: As of June 2026 there's limited independent third-party replication; some community skepticism centers on contamination/timing of AIME26 and whether competition-math scores transfer to production coding (the LeetCode OOD result is the authors' rebuttal). Competitor numbers (GLM-5, Kimi K2.5, Gemini 3 Pro, Claude Opus 4.5) are as the paper reports them.

## Key Papers

1. VibeThinker-3B — Xu et al., Sina Weibo, 2026. [arxiv 2606.16140](https://arxiv.org/abs/2606.16140)
2. VibeThinker-1.5B ("Tiny Model, Big Logic") — Xu et al., Sina Weibo, 2025. [arxiv 2511.06221](https://arxiv.org/abs/2511.06221)
3. DeepSeekMath (GRPO) — Shao et al., DeepSeek-AI, 2024. [arxiv 2402.03300](https://arxiv.org/abs/2402.03300)
4. DeepSeek-R1 — DeepSeek-AI, 2025. [arxiv 2501.12948](https://arxiv.org/abs/2501.12948)
5. DAPO — Yu et al., ByteDance Seed, 2025. [arxiv 2503.14476](https://arxiv.org/abs/2503.14476)
6. Group Sequence Policy Optimization — Zheng et al., 2025. [arxiv 2507.18071](https://arxiv.org/abs/2507.18071)
7. REINFORCE++ — Hu, 2025. [arxiv 2501 e-print]
8. OJBench (competition-level code benchmark) — Wang et al., 2025. [arxiv 2506.16395](https://arxiv.org/abs/2506.16395)
9. IMO-AnswerBench ("Towards Robust Mathematical Reasoning") — Luong et al., 2025. [arxiv 2511.01846](https://arxiv.org/abs/2511.01846)
10. LiveCodeBench — Jain et al., 2024. [arxiv 2403.07974](https://arxiv.org/abs/2403.07974)
11. GPQA (graduate-level Google-proof QA) — Rein et al., 2023. [arxiv 2311.12022](https://arxiv.org/abs/2311.12022)
