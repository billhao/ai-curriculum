# Interpretability — Entry Point

The central index for mechanistic interpretability material in this curriculum. Read this first to orient; click through to the per-topic guides for depth.

## TL;DR

Mechanistic interpretability asks: **what algorithm is the model actually running, expressed in terms a human can audit?** The field has organized itself into a ladder of explanation levels — text, feature, component, edge, neuron, weight — each with its own tools and trade-offs. Pick the level that matches your question.

## Guides in this curriculum

All interp guides live in `interp/`:

- **[interp/interpretability-guide.md](interp/interpretability-guide.md)** — full foundations: residual stream, circuits, induction heads, superposition, SAEs, Scaling Monosemanticity, TransformerLens hands-on, open problems. Start here if you haven't done interp before.
- **[interp/natural-language-autoencoders-guide.md](interp/natural-language-autoencoders-guide.md)** — Anthropic's May 2026 NLA paper: text-bottleneck autoencoders that produce English explanations of activations. The text-level entry on the ladder.
- **[interp/knowledge-vs-intelligence.md](interp/knowledge-vs-intelligence.md)** — Can general intelligence be separated from world knowledge in LLMs? Chollet's framework, Mahowald formal/functional dissociation, MQuAKE, LLM-WikiRace, ARC-AGI-2.
- **[interp/reasoning-model-interp.md](interp/reasoning-model-interp.md)** — Mechanistic interp of R1-Zero / o1 "aha moments". What mechanisms produce backtracking; how task-general they are; the 4-way partition of "wait" tokens.
- **[interp/open-research-questions.md](interp/open-research-questions.md)** — Significant open problems on knowledge-vs-intelligence separability and reasoning mechanism. Ranked by significance, with concrete experiment shapes.

## The Interpretability Ladder

Most abstract (top) to most concrete (bottom). Each level explains the one below in different terms.

```
LEVEL                  UNIT OF EXPLANATION                EXAMPLE WORK
─────────────────      ─────────────────────────          ──────────────────────────────
Behavior               input/output pairs                 evals, benchmarks
Text / concept         English paragraph                  NLA (Anthropic 2026);
                                                          Activation Oracles (2025);
                                                          OpenAI neuron explainer (2023)
Feature                sparse latent direction            SAEs (Bricken 2023,
                                                          Templeton 2024);
                                                          Transcoders / CLT (Lindsey,
                                                          Ameisen 2025); Sparse Feature
                                                          Circuits (Marks 2024)
Component              attention head / whole MLP         IOI circuit (Wang 2022,
                                                          26 heads);
                                                          Induction heads (Olsson 2022);
                                                          Function vectors (Todd 2024)
Edge / connection      data flow between components       ACDC (Conmy 2023);
                                                          Attribution patching
                                                          (Syed 2023);
                                                          Edge Attribution Patching;
                                                          AtP* (Kramár 2024)
Neuron                 single MLP neuron / single dim     Universal Neurons
                                                          (Gurnee 2024);
                                                          "Dead, N-gram, Positional
                                                          Neurons" (Voita 2023);
                                                          Arithmetic neurons
                                                          (Stolfo 2023)
Weight                 individual W matrices / their      Mathematical Framework for
                       low-rank structure                 Transformer Circuits
                                                          (Elhage 2021); QK/OV
                                                          decomposition; Modular
                                                          addition weights (Nanda 2023)
```

## Where each level shines

- **Text level (NLA)** — *semantic summarization*. "What is the model thinking right now?" Best for audits, eval-awareness, hidden motivations. See [interp/natural-language-autoencoders-guide.md](interp/natural-language-autoencoders-guide.md).
- **Feature level (SAEs, transcoders, attribution graphs / circuit-tracer)** — *causal graph at human-readable concepts*. "Why did the model predict X?" Best for mechanism + steering. SAE foundations in [interp/interpretability-guide.md](interp/interpretability-guide.md) §Superposition and Dictionary Learning; transcoders + attribution graphs covered in this file's "Tools" section below.
- **Component level (heads + MLPs)** — *coarse algorithm*. "Which 20 heads implement task Y?" The classic IOI / induction-head pattern. Pre-SAE era but still load-bearing. See [interp/interpretability-guide.md](interp/interpretability-guide.md) §Circuits and Features and §Induction Heads Deep Dive.
- **Edge / connection level** — *which data-flow edges matter*. ACDC and attribution patching produce a pruned subgraph of the model's computational graph for a behavior. Same level as components, but answers "which connections" rather than "which nodes."
- **Neuron level** — *individual MLP dimensions*. Mostly considered a dead end since 2022 because of superposition: most neurons are polysemantic. Still useful for the **~1-5% of neurons that are clean** (Gurnee's "universal neurons" — neurons that fire for the same concept across many models, like "Python `__init__`" or "negative sentiment"). Voita's taxonomy: most are "dead" or "n-gram detectors."
- **Weight level** — *parameter-space structure*. Elhage's Mathematical Framework decomposed attention heads into QK (where to attend) and OV (what to write) matrices, often with low-rank structure that's directly interpretable. Most useful for toy models and small heads; doesn't scale to MLP weight matrices because of polysemanticity.

## Why the field consolidated at the feature level

Pre-2022: people tried neuron-level interpretability (Olah's circuits work in vision). It mostly failed for LLMs because of **superposition** — neurons are polysemantic, single neurons aren't a meaningful unit.

Post-2023: SAEs and transcoders dissolve neurons into features, which *are* (mostly) monosemantic. Features became the new unit. Everything since — attribution graphs, sparse feature circuits, NLA — sits on or above the feature substrate.

## The honest hierarchy of trust

- Component-level circuit work (IOI, induction) has the **strongest mechanistic rigor** — you can prove the circuit is faithful by ablating everything else.
- Feature-level work has the **best coverage** — SAEs/transcoders are the only thing that scales to production models.
- Text-level work (NLA) has the **best human readability** but **weakest causal grip**.

A serious audit pipeline would use NLA to flag *which activations look suspicious*, then attribution graphs / sparse feature circuits to *prove mechanistically what computation they participate in*, then component or weight-level work on the specific circuit you found if you want to be really sure.

## Tools and open-source stacks

- **[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)** — the workhorse library. Load 50+ models with clean internals, cache any activation, hook any component, run causal interventions. Foundations + worked exercises in [interp/interpretability-guide.md](interp/interpretability-guide.md) §Practical Tools.
- **[nnsight](https://nnsight.net/)** — alternative backend that works on any HuggingFace model, including 70B+. Used by circuit-tracer.
- **[circuit-tracer (decoderesearch)](https://github.com/decoderesearch/circuit-tracer)** — open implementation of Anthropic's attribution-graph method. Pre-trained cross-layer transcoders for Gemma-2/3, Llama-3.1/3.2, Qwen-3, GPT-OSS. Local web visualizer with feature interventions. Colab-friendly (15 GB).
- **[Neuronpedia](https://www.neuronpedia.org)** — hosted UI for browsing SAE features and NLA descriptions on open models. Zero-setup way to see what a feature/NLA actually outputs.
- **[NLA released checkpoints](https://huggingface.co/collections/kitft/nla-models)** — AV + AR weights for Qwen-7B, Gemma-12B/27B, Llama-3.3-70B. See [interp/natural-language-autoencoders-guide.md](interp/natural-language-autoencoders-guide.md) for the architecture and how to use them.

## Cross-references

- **SAE foundations & Scaling Monosemanticity**: [interp/interpretability-guide.md](interp/interpretability-guide.md) §Superposition and Dictionary Learning, §Scaling Monosemanticity
- **Residual stream & circuits**: [interp/interpretability-guide.md](interp/interpretability-guide.md) §The Residual Stream View, §Circuits and Features
- **Induction heads**: [interp/interpretability-guide.md](interp/interpretability-guide.md) §Induction Heads Deep Dive
- **NLA architecture & training**: [interp/natural-language-autoencoders-guide.md](interp/natural-language-autoencoders-guide.md) §Architecture Overview, §Training Procedure
- **NLA vs SAE comparison**: [interp/natural-language-autoencoders-guide.md](interp/natural-language-autoencoders-guide.md) §NLA vs SAE vs Activation Oracles

## Decision tree — which level to use

```
What is your question?
│
├─ "What is the model thinking on this specific input?"
│   └─▶ Text level (NLA)              — read out a paragraph from activations
│
├─ "Why did the model produce this specific output?"
│   └─▶ Feature level (CT/SAE)        — build attribution graph of features
│
├─ "Which mechanism inside the model implements task X?"
│   └─▶ Component level               — IOI/induction-style circuit tracing
│
├─ "Is this internal computation actually causal for the output?"
│   └─▶ Edge level (ACDC, AtP*)       — prune to minimal causal subgraph
│
├─ "Is there a clean named role for this single unit?"
│   └─▶ Neuron level (rarely useful)  — only ~5% of neurons are interpretable
│
└─ "What's the parameter-space structure of this attention head?"
    └─▶ Weight level                  — QK/OV decomposition, low-rank analysis
```

## Recommended reading order if you're new

1. [interp/interpretability-guide.md](interp/interpretability-guide.md) end-to-end — gets you fluent in residual stream, superposition, circuits, SAEs, and TransformerLens.
2. Hands-on exercises on your GPT-2 124M (§Hands-on with Your GPT-2 124M).
3. [interp/natural-language-autoencoders-guide.md](interp/natural-language-autoencoders-guide.md) — modern text-level method.
4. Play with circuit-tracer on Gemma-2-2B for attribution graphs.
5. Pick a specific behavior and try to reverse-engineer the circuit yourself.

## Recommended reading order if you already know SAEs

1. [interp/natural-language-autoencoders-guide.md](interp/natural-language-autoencoders-guide.md) §Architecture Overview, §Worked Example.
2. Ameisen 2025 "Circuit Tracing" methodology paper (linked below) for cross-layer transcoders.
3. Lindsey 2025 "On the Biology of a Large Language Model" for what attribution graphs can find.
4. Marks 2024 "Sparse Feature Circuits" — the bridge between SAE features and component-level circuits.

## Key papers (consolidated)

| Paper | Level | Year |
|-------|-------|------|
| [Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) | weight / component | 2021 |
| [In-context Learning and Induction Heads](https://arxiv.org/abs/2209.11895) | component | 2022 |
| [Toy Models of Superposition](https://arxiv.org/abs/2209.10652) | feature (foundations) | 2022 |
| [Interpretability in the Wild: IOI Circuit](https://arxiv.org/abs/2211.00593) | component | 2022 |
| [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features) | feature (SAE) | 2023 |
| [Language Models Can Explain Neurons](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html) | text (auto-interp) | 2023 |
| [ACDC: Automated Circuit Discovery](https://arxiv.org/abs/2304.14997) | edge | 2023 |
| [Attribution Patching](https://arxiv.org/abs/2304.05969) | edge | 2023 |
| [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) | feature (SAE at scale) | 2024 |
| [Universal Neurons](https://arxiv.org/abs/2401.12181) | neuron | 2024 |
| [Sparse Feature Circuits](https://arxiv.org/abs/2403.19647) | feature → component | 2024 |
| [Function Vectors](https://arxiv.org/abs/2310.15916) | component | 2024 |
| [AtP*: Attribution Patching at Scale](https://arxiv.org/abs/2403.00745) | edge | 2024 |
| [Circuit Tracing (methodology)](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) | feature (transcoders + graphs) | 2025 |
| [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) | feature (case studies) | 2025 |
| [Activation Oracles](https://alignment.anthropic.com/2025/activation-oracles/) | text (supervised) | 2025 |
| [Natural Language Autoencoders](https://transformer-circuits.pub/2026/nla/) | text (unsupervised) | 2026 |
| [Open Problems in Mechanistic Interpretability](https://arxiv.org/abs/2501.16496) | survey | 2025 |
