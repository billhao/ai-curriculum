# Wait-Token / Aha-Moment Analysis Pipeline

Concrete implementation plan for analyzing the reasoning chain around the "wait" / aha-moment token in a reasoning model — what features fire, which are causally load-bearing, where in the prior reasoning the trigger came from, and how to scale toward full attribution graphs.

Implements the research direction of [open-research-questions.md](open-research-questions.md) Q4 (end-to-end attribution graph of one aha moment).

Methodology stack referenced throughout: [mech-interp-lineage.md](mech-interp-lineage.md).

## Pipeline overview

```
Phase 0  Pick target model + curate wait-event corpus
   │
   ▼
Phase 1  Per-token SAE feature inventory at "wait" positions
   │
   ▼
Phase 2  Causal partition: load-bearing vs performative
   │
   ▼
Phase 3  Trigger localization via attention edges
   │       (cheapest path to a real causal chain)
   ▼
Phase 4  Strict-sense attribution graphs (CLT replacement model)
   │
   ▼
Phase 5  Cross-token stitching — research frontier (Q4)
```

## Phase 0 — Setup

- Model: DeepSeek-R1-Distill-Qwen-7B or Llama-3.1-8B distill. 7-8B fits on H800 80GB with room for activations + SAE/CLT.
- Generate ~100 CoTs on AIME-style problems; filter to cases manually verifiable as `error → wait → correction`. ~10-50 well-characterized events is more useful than thousands of noisy ones.
- Stack: TransformerLens or nnsight for hooks; SAELens for SAEs; Anthropic's open-sourced [attribution-graphs-frontend](https://github.com/anthropics/attribution-graphs-frontend) for Phase 4.

## Phase 1 — Feature inventory at "wait"

- Check Neuronpedia for pre-trained SAEs on your model; if none for R1-Distill, train one with SAELens on residual-stream activations of mid layers (typically layers 10-20 carry the most reasoning signal in an 8B).
- For each wait token, extract top-k active features across layers. Aggregate across events: which features recur? Cluster by top-activating examples.
- Replicates AIRI / Stanford / Venhoff for your stack — necessary baseline.

## Phase 2 — Causal partition

- For each candidate feature F: ablate or steer at the wait position, let the model continue, score the answer. Bucket:
  - load-bearing — ablation removes the correction, model stays committed to wrong answer
  - correlate-only — ablation doesn't change behavior (feature is downstream side-effect)
  - pathological — ablation improves the answer (wait was a derailment, not a correction)
- Output: a small set of features that genuinely cause the correction.

## Phase 3 — Trigger localization (loose-sense, highest insight-per-compute)

- For each load-bearing feature, identify the attention heads at the wait position that read it in (rank by QK attention weights × value-to-feature alignment).
- For those heads: trace back to source token positions with highest contribution. Read off what the model was "thinking" there. Hypothesis: this is where the error became detectable.
- Validate by ablating the residual stream at the source position and checking the wait still fires correctly.
- Output: a sparse causal chain `wait ← attention head ← earlier position` without full circuit tracing.

## Phase 4 — Strict-sense attribution graph (optional, big compute step)

- Train a CLT on your model per Ameisen et al.'s methods recipe — adaptation effort for R1-Distill architecture; budget several days on H800 plus non-trivial engineering.
- Build attribution graph rooted at the wait-token logit; prune to top edges.
- Versus Phase 3: full feature→feature paths grounded in a replacement model, not just attention edges.

## Phase 5 — Cross-token stitching (the Q4 frontier)

- For each load-bearing upstream position from Phase 3/4, build another attribution graph rooted there.
- Stitch via shared attention edges across positions.
- Aggregation heuristics — group consecutive positions where similar features fire — to compress the mega-graph into something readable.
- Genuinely unsolved methodologically. If Phase 4 works for you, this is where to push.

## Pitfalls

- Wait tokens are heterogeneous (load-bearing / performative / stylistic). Cluster first, analyze each cluster separately — pooling averages out signal.
- Features at the wait position mix "error detected" / "switching strategy" / "using backtracking phrasing." Disentangle via top-activating examples, not activation magnitude.
- Attribution-graph edges are causal in the replacement model, not the original. Cross-validate with real interventions on the base model.
- Compute discipline: Phase 4+5 can balloon. Confirm Phase 3 raises sharp questions before committing.

## Compute & complexity on 1×H800 80GB

| Phase | Dominant cost | Wall time | Eng. complexity | Risk |
|-------|--------------|-----------|-----------------|------|
| 0 Setup + corpus | CoT generation + manual labeling | <1 hr GPU + hrs human | Low | Low |
| 1a SAE: download (if available) | — | Minutes | Low | Low |
| 1a SAE: train (if not) | ~1B-10B activation tokens + sparse opt | 3-10 days for 5-layer band | Medium | Low (recipe mature) |
| 1b Feature inventory | Light inference on N events | <1 day | Medium | Low |
| 2 Causal ablation | ~10-30K forward continuations | 1-2 days | Medium | Low |
| 3 Attention edge tracing + validation | Activation caching + small ablations | 1-2 days | Medium | Low |
| 4a CLT training | Order of magnitude more than SAE | 2-4 weeks (estimate; uncertain) | High — adapter for R1-Distill | High (tooling, hyperparams) |
| 4b Attribution graph per event | Backward sensitivity via CLT | Minutes-hours per graph | Medium | Low (once 4a works) |
| 5 Cross-token stitching | Light compute, heavy methodology | Weeks-months | Very high | Very high (unsolved) |

## Memory budget at runtime

```
Component (7-8B model, BF16)              VRAM
─────────────────────────────────────────────────
Model weights                             ~14 GB
KV cache (5K-token CoT, batch 1)          ~3-5 GB
Activations cached for SAE/CLT analysis   ~2-4 GB
SAE state (width 64K-128K)                ~0.5-1 GB
CLT state (cross-layer, larger)           ~2-4 GB
Misc / fragmentation                      ~5 GB
─────────────────────────────────────────────────
Peak                                      ~25-30 GB
Headroom on H800 80GB                     ~50 GB free
```

H800 is comfortable for all phases. Bottleneck is wall time during training, not memory.

## Key takeaways

1. The single biggest variable is whether SAEs already exist for your model. Pre-trained Llama-3.1-8B SAEs exist on Neuronpedia; for R1-Distill you may need to train, or first try repurposing Llama-3.1-8B SAEs on R1-Distill-Llama-8B (distillation preserves residual structure partially). Training 5-layer SAE band ≈ 1-2 weeks of H800; downloading ≈ minutes.

2. Phases 0-3 are well-trodden territory. Tools mature, methodology mature, compute modest. Roughly 1 person-month with a working stack. Stanford and AIRI papers got to roughly this stage with similar resources.

3. Phase 4 (CLT) is where it gets hard. Anthropic's open-source tools target their architectures; adapting to R1-Distill is non-trivial engineering. CLT training compute estimate is uncertain — could be days or weeks depending on hyperparam search and width. Plan for 1 month minimum including engineering and tuning.

4. Phase 5 is research, not engineering. The methodology doesn't exist. Compute is cheap (lots of small attribution graphs); the bottleneck is figuring out how to stitch and compress.

## Minimum viable contribution path on 1×H800

```
   Month 1:   Phase 0 → 1 → 2 → 3
              Deliverable: characterized causal chains for
              ~20-50 wait events in R1-Distill-7B, sparse
              attention-edge attribution

              ~5-15 GPU-days of compute
              (mostly SAE training if needed)

   Month 2:   Phase 4 — only if Phase 3 raised crisp
              unanswered questions

              ~15-30 GPU-days

   Month 3+:  Phase 5 — research-grade
```

## Single-GPU vs multi-GPU comparison

Anthropic's CLT work likely uses TPU pods or multi-GPU clusters; 1×H800 will be ~5-20× slower wall-time for Phase 4 training but produce equivalent artifacts. For Phases 0-3 the gap is negligible — those are inference-bound and would be IO-bound on a multi-GPU box anyway.

## Caveats on the compute estimates

- Phase 4a CLT training time is the least well-grounded number here — extrapolated from Anthropic's published methodology, not from a known recipe at the 7-8B scale on a single H800. Worth pinning down via the methods paper or by contacting Goodfire/Anthropic before committing to that timeline.
- SAE training time scales with activation volume and width; if you go narrower (16x expansion vs 32x) it's faster but coverage drops.
- All wall-time figures assume reasonable utilization (>50%) — getting there requires usual H800 tuning (FlashAttention, BF16, batch sizing, vLLM for generation).

## Recommendation

Don't skip to Phase 4. Phase 3 (attention-edge localization) gives ~80% of the causal story at ~10% of the engineering cost, and tells you whether Phase 4 is worth doing.

## Cross-references

- [open-research-questions.md](open-research-questions.md) — Q4 motivation
- [mech-interp-lineage.md](mech-interp-lineage.md) — methodology stack (SAE/Transcoder/CLT/AG)
- [reasoning-model-interp.md](reasoning-model-interp.md) — feature-level prior work (AIRI, Stanford, Venhoff)
- [interpretability-guide.md](interpretability-guide.md) — mech interp foundations
- [../interp.md](../interp.md) — entry point
