# Mech Interp Methodology Lineage: SAE → Transcoder → CLT → Attribution Graph

Research path from "neurons are polysemantic" to "we can build causal computational graphs of LLMs." Citations verified via web search, May 2026.

## The lineage

```
PRE-DECOMPOSITION ERA  (circuits, but not sparse-feature-based)
│
├─ Olah et al. "Circuits Thread"           Distill, 2020
├─ Elhage et al. "Mathematical Framework"  Anthropic, Dec 2021
└─ Olsson et al. "Induction Heads"         Anthropic, Mar 2022
│
▼   problem: neurons are polysemantic, superposition makes them unreadable
│
SAE ERA  (decompose activations into sparse features)
│
├─ Cunningham, Ewart, Smith, Huben, Sharkey
│    "Sparse Autoencoders Find Highly Interpretable Features"
│    arXiv 2309.08600, Sept 2023
└─ Bricken, Templeton, Batson, Chen, Jermyn et al.
     "Towards Monosemanticity: Decomposing LMs With Dictionary Learning"
     Anthropic / transformer-circuits.pub, Oct 2023
│
▼   scale up
│
Templeton et al. "Scaling Monosemanticity"  Anthropic, May 2024
                                            (SAE on Claude 3 Sonnet)
│
▼   problem: SAE features have no defined edges between layers
│         — to build a circuit you'd have to push through dense MLP weights
│
TRANSCODER ERA  (sparse approximation of the MLP function, not activations)
│
└─ Dunefsky, Chlenski, Nanda
     "Transcoders Find Interpretable LLM Feature Circuits"
     arXiv 2406.11944, June 2024 (NeurIPS 2024)
│
▼   Anthropic extension: span layers via the residual stream
│
CROSS-LAYER TRANSCODER + ATTRIBUTION-GRAPH ERA  (Anthropic, March 2025)
│
├─ Ameisen et al. "Circuit Tracing: Revealing Computational Graphs in LMs"
│    transformer-circuits.pub/2025/attribution-graphs/methods.html
│    — methodology paper: CLT training, attribution algorithm, pruning, viz
│
└─ Lindsey et al. "On the Biology of a Large Language Model"
     transformer-circuits.pub/2025/attribution-graphs/biology.html
     — applications paper: attribution graphs on Claude 3.5 Haiku
```

## What each method is

| Concept | Approximates | Sparse features live at | Gives causal edges? |
|---|---|---|---|
| SAE | SAE(activation) ≈ activation | One layer, residual stream | No — features are a dictionary, not a network |
| Transcoder | T(MLP_input) ≈ MLP_output | One MLP sublayer | Yes — encoder/decoder weights give linear feature→feature contributions through that MLP |
| CLT | Replaces every MLP with one shared sparse feature set | Each feature reads at layer L, writes to outputs of L, L+1, …, L_last | Yes — across layers, not just within one MLP |
| Attribution graph | Not a model — it's the graph built from a CLT-replaced model | Nodes = features + embeddings + error terms + logits; edges = linear effects | The graph is the edges |

## Narrative arc

1. SAEs decompose: tell you what features are present, but a feature at layer 17 has no defined relationship to a feature at layer 18. To trace causality you have to push activations through the original dense MLP — losing sparsity.
2. Transcoders replace: the sparse network is the computation (≈ MLP function). Feature-to-feature edges become literal weight products — circuit structure reads off the transcoder weights.
3. CLTs span layers: instead of one transcoder per MLP, one shared sparse feature set whose features each project into all later MLPs' outputs. Resulting circuits factor cleanly across layers.
4. Attribution graphs: pick a target node (e.g., the "wait" logit), trace backward through the linear edges of the CLT-replaced model, prune to load-bearing nodes. The output is the causal graph. Anthropic reports the replacement model matches the original output on ~50% of cases — error nodes capture the gap.

## Nuance

Attribution-graph edges are causal in the replacement model — not in the original model. Validation requires checking that interventions on real model components match graph predictions. Goodfire has been replicating circuit-tracing externally to test this; community library CLT-Forge has scaled the methodology.

## Paper links

- [Cunningham et al., Sparse Autoencoders Find Highly Interpretable Features (arXiv 2309.08600)](https://arxiv.org/abs/2309.08600)
- [Bricken et al., Towards Monosemanticity (transformer-circuits.pub)](https://transformer-circuits.pub/2023/monosemantic-features)
- [Templeton et al., Scaling Monosemanticity (transformer-circuits.pub)](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
- [Dunefsky, Chlenski, Nanda, Transcoders Find Interpretable LLM Feature Circuits (arXiv 2406.11944)](https://arxiv.org/abs/2406.11944)
- [Ameisen et al., Circuit Tracing: Revealing Computational Graphs in Language Models (Anthropic, 2025)](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)
- [Lindsey et al., On the Biology of a Large Language Model (Anthropic, 2025)](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)
- [Anthropic: Tracing the thoughts of a large language model (blog)](https://www.anthropic.com/research/tracing-thoughts-language-model)
- [Anthropic: Open-sourcing circuit-tracing tools](https://www.anthropic.com/research/open-source-circuit-tracing)
- [Goodfire: Replicating Circuit Tracing for a Simple Known Mechanism](https://www.goodfire.ai/research/replicating-circuit-tracing-for-a-simple-mechanism)

## Cross-references

- [interpretability-guide.md](interpretability-guide.md)
- [reasoning-model-interp.md](reasoning-model-interp.md)
- [natural-language-autoencoders-guide.md](natural-language-autoencoders-guide.md)
- [open-research-questions.md](open-research-questions.md)
- [../interp.md](../interp.md)
