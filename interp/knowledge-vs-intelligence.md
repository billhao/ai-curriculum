# Knowledge vs Intelligence in LLMs

Can general intelligence be cleanly separated from world knowledge in LLMs? This guide surveys the philosophical framing, the empirical evidence, and the active research tracks as of May 2026.

## The Core Question

Humans can't be intelligent with zero knowledge — physics, causality, language, social structure form the substrate on which reasoning operates. The same is true for LLMs: no system has demonstrated robust general reasoning from a model with negligible pretraining knowledge. Yet frontier models like Claude Opus 4.6 carry vastly more parametric knowledge than their reasoning capability *seems* to require. So the question is:

> **What's the minimum kind and amount of knowledge a system needs before its reasoning becomes generally useful — and can the two be cleanly separated?**

## Background — Definitional Frame

**François Chollet — *On the Measure of Intelligence* (2019).** The cleanest operational definition. Intelligence = **skill-acquisition efficiency**, not skill itself. Four decoupled variables: scope, generalization difficulty, **priors**, experience. "Unlimited priors or unlimited training data allow experimenters to 'buy' arbitrary levels of skills... in a way that masks the system's own generalization power." ARC built around a small set of Spelke-derived **Core Knowledge priors**: object cohesion, persistence, contact, agentness, elementary number, basic geometry. [arXiv:1911.01547](https://arxiv.org/abs/1911.01547)

**Cattell-Horn-Carroll psychometrics.** Fluid intelligence (Gf) = novel-problem reasoning. Crystallized intelligence (Gc) = accumulated knowledge. Maps cleanly onto ARC (Gf) vs MMLU (Gc). Important nuance: even in human psychometrics, Gf and Gc aren't independent — *fluid ability is invested into building crystallized knowledge.* The "pure fluid intelligence with no crystallized substrate" idealization isn't a thing in humans either.

**LeCun — *A Path Towards Autonomous Machine Intelligence* (2022).** Argues intelligence requires a **predictive world model** trained by self-supervised learning in representation space (JEPA, H-JEPA). Position: language is the wrong substrate; physics/video is. Implies "knowledge-free intelligence" is a category error.

**Bubeck et al. — *Sparks of AGI* (2023).** Capability-first framing. Less interested in clean decomposition than in documenting cross-domain competence. [arXiv:2303.12712](https://arxiv.org/abs/2303.12712)

**Mahowald, Ivanova, Tenenbaum, Fedorenko et al. — *Dissociating Language and Thought in LLMs* (2023).** The canonical cognitive-science-grounded framing. LLMs have strong **formal linguistic competence** (syntax, semantics) but weak **functional competence** (world reasoning, social cognition, math). [arXiv:2301.06627](https://arxiv.org/abs/2301.06627)

**Marcus — neurosymbolic position.** Long-standing claim: robust intelligence requires hybrid architecture + rich priors + sophisticated reasoning. Argues 2025 frontier systems (o3, Grok 4 with tools, Claude Code) only reach high scores via external symbolic scaffolding — accidentally vindicating the hybrid view.

## Empirical Evidence on Separability

### Local separability — mech-interp says yes

| Result | Evidence | Source |
|---|---|---|
| Factual associations localized in mid-MLPs as key-value stores | Rank-one edits replace facts without broad damage | ROME (Meng et al. 2022, [arXiv:2202.05262](https://arxiv.org/abs/2202.05262)) |
| Fact-editing scales to thousands of associations | GPT-J/NeoX with preserved fluency | MEMIT (Meng et al. 2023, [arXiv:2210.07229](https://arxiv.org/abs/2210.07229)) |
| FFN neurons correlate with specific factual expression | Ablating suppresses that fact specifically | Knowledge Neurons (Dai et al. 2021, [arXiv:2104.08696](https://arxiv.org/abs/2104.08696)) |
| Content-agnostic reasoning circuit (IOI) | 26 attention heads in 7 classes operating over arbitrary names | Wang et al. 2022, [arXiv:2211.00593](https://arxiv.org/abs/2211.00593) |
| Content-agnostic in-context pattern completion | Sharp phase transition during training | Induction Heads (Olsson et al. 2022) |
| Procedural features reused across data domains | Same "6+9" arithmetic feature in math, astronomy, financial tables; genuine two-hop reasoning ("capital of state containing Dallas" → "Texas" → "Austin") | On the Biology of an LLM (Lindsey, Marks, Batson et al. 2025, [transformer-circuits.pub](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)) |

### Smoking gun against full separability — MQuAKE

Zhong et al. 2023, [arXiv:2305.14795](https://arxiv.org/abs/2305.14795). You can edit a fact with ROME/MEMIT, the model recalls the new fact directly, **but multi-hop reasoning over the edited fact fails badly**. If facts and reasoning were truly modular, the edit would propagate; it doesn't. Local edits work, structural decoupling does not.

### Other erosion evidence

- **EWOK** (Ivanova et al. 2024, [arXiv:2405.09605](https://arxiv.org/abs/2405.09605)) — frontier models have broad deficits in basic world knowledge (physics, agents, social dynamics) relative to humans.
- **Reasoning or Reciting?** (Wu et al. 2023, [arXiv:2307.02477](https://arxiv.org/abs/2307.02477)) — strong performance drops when tasks are reparameterized to remove familiar cues.
- **Counterfactual analogical reasoning** (Lewis & Mitchell 2024, [arXiv:2402.08955](https://arxiv.org/abs/2402.08955)) — current LLMs' analogical reasoning is much narrower than benchmark scores imply.
- **ActionReasoningBench** (Handa et al. 2024, [arXiv:2406.04046](https://arxiv.org/abs/2406.04046)) — frontier LLMs struggle on action/state tracking, suggesting weak internal simulation of changing world states.
- **Jagged intelligence** (Karpathy, July 2024) — IMO problems solved but letter-counting fails. If reasoning and knowledge were truly modular, this profile would not be jagged.

### The empirical floor — TinyStories

Eldan & Li 2023, [arXiv:2305.07759](https://arxiv.org/abs/2305.07759). Sub-10M-parameter, single-transformer-block models trained on synthetic 3-4-year-old-vocabulary stories produce fluent multi-paragraph text with rudimentary reasoning. **Implication: the substrate for coherent reasoning is far smaller than 7B, but the world it can reason about is correspondingly tiny.**

### Pure-RL reasoning emergence — still atop a knowledge base

- **DeepSeek-R1-Zero** ([arXiv:2501.12948](https://arxiv.org/abs/2501.12948), 2025): RLVR on pretrained base, no SFT, elicits self-reflection / verification / backtracking. Reasoning emerges *atop* the pretrained substrate.
- **TinyZero** (Jiayi Pan et al. 2025): reproduces R1-Zero behaviors on 3B Qwen2.5 for ~$30 on Countdown task. 1.5B works, 0.5B does not. Task-specific.
- **LIMO — *Less is More for Reasoning*** (Ye et al. 2025, [arXiv:2502.03387](https://arxiv.org/abs/2502.03387)): strong reasoning from tiny curated supervised sets — but only on a rich base.
- **Open-Reasoner-Zero** (Hu et al. 2025, [arXiv:2503.24290](https://arxiv.org/abs/2503.24290)): pure-RL on pretrained base.

**Pattern across all of these:** reasoning *patterns* can be elicited from a pretrained base with minimal additional data, but nobody has shown reasoning from a model with negligible pretraining.

### Allen-Zhu — Physics of Language Models

[physics.allen-zhu.com](https://physics.allen-zhu.com/), multi-part program (ICLR 2025). Most explicit research program treating **reasoning, knowledge storage, knowledge extraction, knowledge manipulation as distinct dimensions** and probing each with controlled synthetic corpora (e.g., 20M synthetic biographies in isolation).

### Grokking — separability hint

Power et al. 2022, [arXiv:2201.02177](https://arxiv.org/abs/2201.02177). On small algorithmic datasets, models transition abruptly from memorization to generalization long after training-loss convergence. Circuit-level competition between memorizing and generalizing solutions. Suggests reasoning structure emerges as a *separate* solution under sustained regularization.

## Knowledge-Decoupled Architectures

### Retrieval (knowledge outsourced)

| Method | Result | Source |
|---|---|---|
| REALM | First end-to-end retrieval-augmented pretraining | Guu et al. 2020, [arXiv:2002.08909](https://arxiv.org/abs/2002.08909) |
| RETRO | 7.5B + 2T-token retrieval ≈ GPT-3 175B perplexity (~25× param reduction) | Borgeaud et al. 2021, [arXiv:2112.04426](https://arxiv.org/abs/2112.04426) |
| Atlas | 11B beats 540B PaLM on few-shot QA (50× less pretrain compute) | Izacard et al. 2022, [arXiv:2208.03299](https://arxiv.org/abs/2208.03299) |
| REPLUG | Treats LM as black box; +6.3% perplexity, +5.1% MMLU on frozen GPT-3 | Shi et al. 2023, [arXiv:2301.12652](https://arxiv.org/abs/2301.12652) |
| BRIGHT | Realistic reasoning-intensive retrieval benchmark | Su et al. 2024, [arXiv:2407.12883](https://arxiv.org/abs/2407.12883) |

### World models (knowledge as embedded physics, not text)

| Method | What it does | Source |
|---|---|---|
| DreamerV3 | Model-based RL across 150+ tasks, single config; first to mine Minecraft diamonds from scratch | Hafner et al. 2023, [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) |
| I-JEPA | Self-supervised image representation via joint-embedding prediction | Assran et al. 2023, [arXiv:2301.08243](https://arxiv.org/abs/2301.08243) |
| V-JEPA 2 | 1.2B video world model on >1M hours; zero-shot robot planning | Assran et al. 2025, [arXiv:2506.09985](https://arxiv.org/abs/2506.09985) |
| Genie | 11B foundation world model from 30K hours platformer video; generates playable interactive environments from one image | Bruce et al. 2024, [arXiv:2402.15391](https://arxiv.org/abs/2402.15391) |

These sidestep language entirely. They encode "knowledge" in video-statistical priors, not text.

### The Tenenbaum-Andreas line — probabilistic world models

- **From Word Models to World Models** (Wong et al. 2023, [arXiv:2306.12672](https://arxiv.org/abs/2306.12672))
- **Reasoning with LM is Planning with World Model** (Hao et al. 2023, EMNLP)
- **Modeling Open-World Cognition as On-Demand Synthesis of Probabilistic Models** (Wong et al. 2025, [arXiv:2507.12547](https://arxiv.org/abs/2507.12547))

The cutting edge connecting cognitive-science world models to LLM reasoning.

## ARC-AGI-2 — The Field's Best Direct Test

[arXiv:2505.11831](https://arxiv.org/abs/2505.11831) (May 2025). Designed to be ungameable by parametric memorization. Frontier scores on the semi-private set:

```
o3 (Medium):    3.0%
o4-mini:        2.4%
o1-pro:         0.9%
Human baseline: ~75%
```

ARC Prize 2025 organizers' verdict: *"current AI reasoning performance is tied to model knowledge"* — explicitly framing the gap as a failure to separate the two.

## LLM-WikiRace — The Threshold Hypothesis (Feb 2026)

Ziomek, Bankes, Wolf, Ramesh, Tang, Bogunovic. [arXiv:2602.16902](https://arxiv.org/abs/2602.16902) (Feb 18, 2026; v2 Feb 23, 2026). [leaderboard](https://llmwikirace.github.io).

**Central claim:** *World knowledge is necessary up to a threshold; beyond that, planning and long-horizon reasoning become the dominant bottleneck.* The clearest concrete attempt to quantify the knowledge-vs-reasoning tradeoff. **Very new — treat with caution.**

### Task

WikiRace: navigate from a source Wikipedia article to a target by following hyperlinks. The model sees outgoing links at each step and picks one. **30-step limit.** Partial graph visibility — you only see local link sets, not the full graph.

### Leaderboard (from llmwikirace.github.io)

| Model | Easy | Medium | Hard |
|---|---|---|---|
| Gemini 3.1 | 96.0% | 69.8% | 29.0% |
| Gemini 3 | 95.0% | 66.0% | 23.0% |
| Claude Opus 4.6 | 91.0% | 56.7% | 16.0% |
| GPT-5.2 | 90.0% | 50.7% | 10.0% |
| LLaMA 3 1B / Gemma 3 4B | ≈ 0% on Medium/Hard |

### Findings

- Frontier models reach **superhuman performance on Easy** but collapse on Hard (best 29%).
- Models with similar world knowledge show **significant performance gaps** → planning is the differentiator at the frontier, not knowledge.
- Common failure mode: **navigation loops** — models that loop more often perform significantly worse, indicating poor replanning after wrong turns.

## Three Positions on the Central Question

```
POSITION                       PROPONENTS                EVIDENCE
─────────────────────────      ──────────────────────    ─────────────────────────────
1. Cleanly separable           Allen-Zhu, Anthropic       Synthetic experiments isolate
   in principle                interp, Eldan/Phi line     dimensions; content-agnostic
                                                          circuits (IOI, induction);
                                                          shared procedural features

2. Partially separable in      Mainstream mech-interp,    ROME/MEMIT edits work locally;
   practice but coupled        Bau Lab                    MQuAKE shows edits don't
   at inference                                           propagate through multi-hop;
                                                          formal/functional dissociation
                                                          (Mahowald) is incomplete

3. Category error to even      LeCun, Marcus,             Reasoning is planning over a
   try separating              Tenenbaum/Spelke           world model; Spelke core
                               cognitive science          knowledge is the structural
                                                          floor; intelligence without
                                                          knowledge is incoherent
```

## Active Research Tracks

| Researcher / group | Position | Recent output |
|---|---|---|
| **François Chollet** | LLMs do local generalization via parametric memorization; need program synthesis for fluid intelligence | Co-founded **NDEA** with Mike Knoop (2025); ARC-AGI-2; ARC Prize 2025 (1,455 teams); Dwarkesh podcast |
| **Yann LeCun / Meta FAIR** | Text-only LLMs are a dead end; world models on sensory data are the path | I-JEPA → V-JEPA → V-JEPA 2 (June 2025) |
| **Sébastien Bubeck** | Curated reasoning-dense data unlocks fluid-intelligence-like behavior at small scale | Phi-4 (Jan 2025); moved to OpenAI late 2024 |
| **Mahowald, Ivanova, Tenenbaum, Fedorenko** | Formal vs functional competence dissociation in LLMs | *Dissociating Language and Thought* (2023); EWOK (2024) |
| **Yejin Choi** | Symbolic knowledge distillation, commonsense | Symbolic Knowledge Distillation (NAACL 2022); NovaCOMET (EMNLP 2023); UW → Stanford 2024 |
| **Melanie Mitchell** | Robustness, analogy, anti-shortcut diagnostics | Counterfactual analogical reasoning (2024); *On Evaluating Abstraction* (Curr Dir Psy Sci 2026); Substack |
| **Joshua Tenenbaum / Jacob Andreas / Gabriel Grand** | Probabilistic programs, on-demand world-model synthesis | *From Word Models to World Models* (2023); *Modeling Open-World Cognition* (2025) |
| **Anthropic interp** (Lindsey, Marks, Batson) | Separating fact-features from procedure-features in real circuits | Scaling Monosemanticity (2024); Biology of an LLM + Circuit Tracing (March 2025) |
| **Bau Lab** | Factual storage, model editing, memory localization | ROME, MEMIT lineage |
| **Zeyuan Allen-Zhu** | Physics of Language Models — separable dimensions of reasoning vs knowledge | physics.allen-zhu.com; ICLR 2025 |
| **Gary Marcus** | Neurosymbolic hybrid is the only path | AAAI 2025 with Vincent Belle, *The Future Is Neuro-Symbolic*; Substack |
| **Sara Hooker / Cohere For AI** | Hardware lottery; scaling/intelligence link is contingent | *The Hardware Lottery* (CACM 2021) |
| **Spelke / Gopnik / cognitive science** | Core knowledge as developmental substrate | *Core Knowledge* (2007); object permanence in newborn chicks (Wood et al. 2024) |

## The Cleanest Answer (May 2026)

- **In current frontier LLMs**: knowledge and intelligence are **not** cleanly separable. Models that score 95% on MMLU score 3% on ARC-AGI-2. Mahowald's formal/functional dissociation, MQuAKE's multi-hop failure, Mitchell's counterfactual collapse all show reasoning *appears to work* when surface cues are familiar and *collapses* when stripped.
- **In principle, partially**: mech-interp finds genuine content-agnostic circuits (IOI, induction). Allen-Zhu's controlled synthetic experiments show reasoning and knowledge are probable as separable dimensions. TinyStories shows reasoning machinery doesn't need 7B parameters.
- **There is a knowledge floor**: nobody as of May 2026 has demonstrated broad reasoning from a model with negligible pretraining. Even R1-Zero / TinyZero need a pretrained base. The Spelke / Tenenbaum view says this is structural — you need a minimum core-knowledge prior (objects, agents, number, space, causality) for reasoning to bootstrap.
- **The threshold hypothesis** (LLM-WikiRace 2026): world knowledge is necessary up to a threshold, beyond which planning capacity is the bottleneck. **If this holds up under replication, a small parametric core at ≈threshold-level knowledge + retrieval + strong reasoning training could be competitive with frontier parametric models.** The paper is very new.

### The right intuition pump

Humans have ~13 billion years of cosmic priors compressed into evolution-shaped neural architecture + ~20 years of grounded sensorimotor + linguistic experience to bootstrap reasoning. They can't reason in a vacuum either. The question isn't "can you have intelligence without knowledge" — it's:

> **"What's the *minimum, structured, bootstrap-relevant* knowledge a system needs before its reasoning becomes general?"**

Nobody has a clean answer yet, but the Spelke core-knowledge list (objects, agents, number, space, geometry, basic causality) + Tenenbaum's probabilistic-program apparatus is the leading hypothesis.

## Open Frontier (May 2026)

1. **Knowledge floor.** What's the minimum structured prior needed? Spelke core knowledge or substantially more?
2. **Threshold quantification.** Does the LLM-WikiRace claim ("knowledge necessary only up to a threshold") replicate?
3. **Multi-hop edit propagation.** Can MQuAKE be solved without retraining? Cleanly-propagating fact edits would be the strongest demonstration of separability.
4. **RLVR from minimal base.** Has anyone done RL from a *small* knowledge-light base and gotten reasoning? Not in the public record.
5. **World models as substitute substrate.** Do JEPA/Genie/Dreamer learn the abstract program synthesis ARC requires, or only physics?
6. **Jagged intelligence.** Transient artifact or permanent feature of the parametric paradigm?
7. **CHC psychometric battery on LLMs.** Apparent literature gap — no canonical 2024-2026 paper applies a Gf/Gc-aligned battery to frontier LLMs.

## Recommended Primary Reading

The five most directly relevant papers:

1. **Chollet 2019** — *On the Measure of Intelligence* ([arXiv:1911.01547](https://arxiv.org/abs/1911.01547)). Definitional foundation.
2. **Mahowald et al. 2023** — *Dissociating Language and Thought in LLMs* ([arXiv:2301.06627](https://arxiv.org/abs/2301.06627)). The formal/functional split.
3. **Zhong et al. 2023** — *MQuAKE* ([arXiv:2305.14795](https://arxiv.org/abs/2305.14795)). Strongest evidence against full modularity.
4. **Lindsey, Marks, Batson et al. 2025** — *On the Biology of an LLM* ([transformer-circuits.pub](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)). Best mech-interp evidence for partial separability.
5. **Wong et al. 2025** — *Modeling Open-World Cognition as On-Demand Synthesis* ([arXiv:2507.12547](https://arxiv.org/abs/2507.12547)). Tenenbaum-line cutting edge.

## Cross-references in this curriculum

- Mech-interp foundations: [../interp.md](../interp.md), [interpretability-guide.md](interpretability-guide.md)
- Text-level interpretability (NLA): [natural-language-autoencoders-guide.md](natural-language-autoencoders-guide.md)
- Reasoning-model interp (R1-Zero aha-moment mechanism): [reasoning-model-interp.md](reasoning-model-interp.md)
- ARC + reasoning benchmarks: [../benchmarks-guide.md](../benchmarks-guide.md)
- Reasoning model lineage: [../model-o1-guide.md](../model-o1-guide.md), [../model-deepseek-r1-guide.md](../model-deepseek-r1-guide.md)
- Test-time compute (where reasoning lives at inference): [../test-time-compute-guide.md](../test-time-compute-guide.md)
