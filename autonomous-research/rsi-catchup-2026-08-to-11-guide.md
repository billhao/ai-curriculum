# RSI Catch-Up: August to November 2026

A ranked reading guide to substantive recursive self-improvement research released between **3 August and 3 November 2026**, building on your earlier DGM, AlphaEvolve, ASI-Arch, speedrunning, and execution-grounded research studies.

**Updated 4 November:** comparison with your supplied report revealed substantial omissions in my original search. The [comparison and revised reading order](#4-november-comparison-and-revised-reading-order) below supersede the original ranking while preserving the earlier paper analyses.

The useful change is that several systems now improve the *machinery used to learn*: workflow generation, search allocation, training environments, or the model occupying multiple research roles. Results remain bounded. Some transfer beyond their development tasks; some regress elsewhere; stronger evidence that a successor is a better improver is still scarce. This is my synthesis of the studies below, rather than a claim that they establish autonomous, accelerating general intelligence.

## Recommended learning order

Rank reflects relevance to your existing knowledge and the value of the mechanism or experiment, rather than benchmark leadership. Read **1–3 first**. Entries **4–7** cover distinct new directions. Entries **8–9** are optional.

| Rank | Work | New material | Learn this first |
|---|---|---|---|
| 1 | [AIDE²](https://arxiv.org/abs/2609.26457) | September paper | Better researcher versus better improver |
| 2 | [MASS](https://arxiv.org/abs/2610.12176) | 8 October | Put learned workflows into model weights |
| 3 | [Audit](https://arxiv.org/abs/2610.27610), [False Frontiers](https://arxiv.org/abs/2609.39102), [S3Gym](https://arxiv.org/abs/2608.31100) | August–October | Distinguish progress from selection and shared errors |
| 4 | [Recursive Flint](https://huggingface.co/RecursiveSI/flint) | October model release | A concrete AI-assisted model-development output |
| 5 | [SPADE](https://arxiv.org/abs/2608.19197) | 19 August; revised 31 August | Learn the curriculum and solver together |
| 6 | [Dream-RSI](https://arxiv.org/abs/2609.14858) | 14 September; revised 6 October | Reuse experiment traces to improve search allocation |
| 7 | [RSI-Env](https://arxiv.org/abs/2611.01736) | 1 November | Let an accepted successor conduct the next round |
| 8 | [Meta^n](https://arxiv.org/abs/2608.24735) | 25 August | Layer reusable context and executable helpers |
| 9 | [RSI-Anything](https://arxiv.org/abs/2610.13912) | 9 October | Improve predictive selection under costly feedback |

Two chronology corrections matter. AIDE² was [announced in July](https://www.weco.ai/blog/first-evidence-of-recursive-self-improvement); its September technical report supplies the new detailed evidence. Recursive's earlier NanoGPT, NanoChat, and kernel results are dated **11 June** in its [official updates index](https://recursive.com/updates), despite an October header on the migrated article. Flint is the recent release; those experiments are background.

## Background: what changed since your earlier reading

Your [existing RSI map](rsi-proposer-landscape-2026.md) covers the lineage from Schmidhuber's self-rewriting machine to DGM's archive and execution-scored search. Your [execution-grounded guide](execution-grounded-auto-research-guide.md) already explains why an attractive idea needs a real implementation and outcome. Keep that foundation; the next distinction concerns *which part of the loop improves*.

```text
Current artifact A_k              Improver I_k
(model, harness, memory)          (search, training, evaluation)
          |                              |
          +---------- improve -----------+
                         |
                       A_k+1
                         |
                  independent evaluation

Additional recursive claim:
use the successor to produce I_k+1,
then test whether I_k+1 improves the SAME starting artifact better.
```

Write an update as

$$A_{k+1}=I_k(A_k;B,D),$$

where $B$ is the resource budget and $D$ the available experience. A better final artifact answers one question. A stronger test of the improver compares

$$Q(I_{k+1}(A_*;B,D))-Q(I_k(A_*;B,D)),$$

using the same starting artifact $A_*$, budget, information, and independent evaluator $Q$.

For your GPT-2 work, lowering validation loss after editing `train.py` demonstrates a better training recipe. To test whether the *research agent* improved, give both old and new agents the same untouched script and GPU allowance, then compare their new experiments. Starting the new agent with an already improved script confounds the comparison. This is the experiment to keep in mind throughout the list.

## 1. AIDE²: study the positive transfer and the inconclusive ignition test together

**Srikanth, Zhao, Xu, Wu, and Jiang; Weco AI.** [Paper](https://arxiv.org/abs/2609.26457), especially §§2–3.6 and Appendix A/D/E; PDF pp. 3–11, 24–25.

A fixed outer agent rewrites an inner research harness. The inner agent conducts experiments under resource limits; private grading evaluates its selected result. The headline eight-day run proposed 99 changes and accepted seven, raising private grade from **0.703 to 0.778**. On external ALE tasks, the final harness scores **1790 versus 1511** for the human-written harness. MLE and FML margins are smaller relative to uncertainty. Among apparent isolated-kernel improvements, the fraction classified as hacking because they crash or fail the training-transfer criterion falls from **55% to 32%**.

The learned harness combines multiple search strategies, bounded role-specific context, and recurring-error memory. The main outer improver stays fixed. In the separate §3.6 test, both arms start from the same inner AIDE47; the outer improver is either AIDE47 or AIDEhuman. Three runs per arm finish at **0.780 versus 0.782**, respectively. The authors call this ignition test inconclusive.

Study how the paper separates private task outcomes, external transfer, and improvement ability. The stronger claim is transferable harness optimization, with an unusually informative attempt to test recursion. The exact AIDE² implementation was not linked in the inspected paper or announcement; the older AIDE repository is not the same artifact.

**Question to answer after reading:** which experiment would remain positive if every proposed harness were generated by the original, unchanged outer agent?

## 2. MASS: self-generated collaboration becomes training data

**Lee, Lee, Xu, Seely, Sojoudi, Zaharia, and Tang; UC Berkeley and Sakana AI.** [Paper](https://arxiv.org/abs/2610.12176), pp. 3–9 and Appendices C/D/G, particularly pp. 19–21 and 29.

The current Qwen3.6-27B fills three roles: execute a workflow, judge its output, and improve the workflow. Selected multi-agent executions train the model with LoRA SFT. The successor then fills all three roles again. Two complete cycles therefore carry learned behavior into weights. ScienceAgentBench rises **28.8→30.4→31.7%**. Workflows produced by the newer generation win **64%** when executed by the frozen base model, although the original search changed all three roles together.

Transfer is uneven: Terminal-Bench declines **37.5→35.2%** and SWE-bench Verified **62.6→61.1%**. The MLR-Bench gain largely accompanies more completed submissions; task-level significance is weak. Appendix G supplies a revealing failure: a generated solution changes simulator movement rules, yet external judges favor it. The headline **1.2–1.6×** concerns score per output token, not general capability or total training cost.

The [released code](https://github.com/SakanaAI/mass) makes this a useful extension of your SFT work: inspect how role traces become supervised tokens and how the workflow is removed from the orchestrator's training prompt. Its newer MLR table uses **107 tasks × 3 trials**, while the PDF uses **12 × 5**; keep those results separate. Several evaluation adapters are absent, and the inspected README does not link pretrained MASS weights. First-cycle SFT alone used **6 H100s for 19 hours**, excluding search and generation.

**Question to answer:** did training create stronger coordination, teach the model to invoke coordination it already possessed, or both? The generic multi-agent prompting control in Appendix C is essential.

## 3. Evaluation package: three different ways apparent progress can mislead

### 3.1 Auditing the Product, Not the Recursion

**Chen and Zhu, Stevens Institute of Technology; Zheng, UMass Boston.** [Paper](https://arxiv.org/abs/2610.27610), Table 2, §4, Tables 4–5, and §§5.3/6.5; PDF pp. 5–15.

Its coded sample contains 114 implemented studies, including 90 that call their own system RSI. Among 26 claiming a better improver, only **four** report the combination of identical starting artifacts and matched budgets. That counts an experimental design, not four established breakthroughs. Most paper coding outside the adjudicated sample was AI-assisted without individual human review, so use the prevalence estimates cautiously.

The methodological lesson is firmer. Repeatedly keeping the highest noisy score creates an improving selection curve even when every candidate has equal true quality. With 99 equally good proposals and identically distributed, independent continuous measurement noise, including for the initial incumbent, the expected number of new records is about **4.19**. Accepted revisions alone therefore do not establish learning. Independent product evaluation, repeated complete runs, and same-start improver comparisons answer different questions.

### 3.2 False Frontiers: a separate evaluator can inherit the same wrong labels

**Chen et al.; Rutgers, independent researchers, UC San Diego, Michigan, McGill, and KFUPM.** [Paper](https://arxiv.org/abs/2609.39102), §§2–5 and Appendix A/B; PDF pp. 4–9, 16–18.

A question proposer supplies pseudo-labels; a solver learns them and then scores later proposals. Both can agree because both learned the same error. CrossFit splits by **source document**: an auxiliary solver trained on A scores questions from B, and vice versa. The main solver still trains on all admitted questions.

Across seven search benchmarks, 4B/9B average Cover-EM rises **40.0→48.8% / 42.8→51.2%** versus coupled self-evolution. False-agreement mass falls **6.1→3.0% / 8.8→3.7%**. Random question splitting and a same-source auxiliary evaluator are weaker controls. Six extra majority-vote samples help much less.

The audit uses an external LLM with roughly 86% coverage, and full CrossFit runs reserve **72–79% more H200-hours**. Fixed-bank replay supports the source-ancestry mechanism; it does not prove equal-cost downstream superiority. The practical lesson is to track the origin of evaluator training data, not merely its model name.

### 3.3 S3Gym: retain gains, rather than select the best moment

**ByteDance Seed, M-A-P, and TokenWave.AI.** [Paper](https://arxiv.org/abs/2608.31100), pp. 8–11 and 15–18; [project](https://self-developing-agents.github.io/).

Agents explore seven games without true reward, judge their own experience, then consolidate history, memory, or training examples. Independent evaluation uses disjoint seeds. In the Qwen3-8B training experiment, the PvZ score falls **23→6** at every updated checkpoint; Trust peaks at **30** but ends at **2**. Local judgment quality has essentially no measured correlation with subsequent improvement in the studied runs.

Only three evaluation episodes per checkpoint make precise rankings fragile. Its AUC+ statistic clips negative gains; a selected maximum also hides later regression. Read the trajectories and final scores. This is a concrete warning for your own SFT pipeline: a plausible self-generated target and a rising training signal do not establish that the retained policy improved.

**Use this package while reading every positive result:** identify the selection set, untouched test, starting state, actual resource use, evaluator ancestry, and final-versus-best checkpoint.

## 4. Recursive Flint: the new model artifact is worth studying

**Recursive; October 2026.** Read the [model card](https://huggingface.co/RecursiveSI/flint), [technical release](https://recursive.com/updates/introducing-recursive-flint), and [rec inference engine](https://github.com/recursive-oss/rec).

Flint is an actual Apache-2.0 coding-model release: **39.8B total / approximately 2.5B active parameters**, 49 layers, and 256 routed experts with six selected per token. The published configuration supports **262,144 tokens**. BF16 weights, configuration, chat template, and serving code are available. Developer evaluations report **54.9% Terminal-Bench 2.1** and **74.2% SWE-bench Verified**, under their stated harnesses and sampling rules.

Recursive attributes architecture, quantization, and kernel work to its Eureka research system. For your MoE and nanoGPT background, study the joint design of active computation, memory movement, quantization, and inference kernels. A useful research output need not be a new learning algorithm.

The public release does not expose a complete Eureka improvement loop or isolate its contribution from human work. It also does not show Flint making the next researcher generation better. This is concrete AI-assisted development evidence with open outputs, rather than a controlled demonstration of recursive acceleration. The rec README's local recipe targets supported Pro/Max Macs with at least **48 GB** unified memory; a small active-parameter count does not imply that all weights fit a small Mac.

## 5. SPADE: learn to create executable environments

**Liu, Yu, et al.; a multi-university collaboration including Washington, Stanford, and Northeastern.** [Paper](https://arxiv.org/abs/2608.19197), §§4–7, Tables 1–3, and Appendix C; PDF pp. 6–14, 43–46.

One shared model is both Environment Designer and Reasoning Agent. The designer writes stateful Python environments, transitions, rewards, and verification code. Solving with versus without privileged hints estimates a learning opportunity; GRPO updates both roles. The actual designer reward blends **0.4 floored, normalized hint regret** with **0.6 difficulty targeting**. Source documents supply breadth; environment memory tracks previous designs.

The 30B unweighted eight-benchmark average reaches **58.3**, versus **50.2** initially and **53.0** for the strongest fixed-environment baseline. Four columns are Reasoning Gym categories; gains concentrate in procedural reasoning. Separate tool-use gains of **5.7 BFCL points** and **13.9 ACEBench points** are relative to the base model, not those fixed-environment comparisons.

The eight evaluation benchmarks also choose the best checkpoint, so they are not untouched final tests. Equal training iterations do not match total generation and hint-rollout compute. Executable checks can still omit requirements expressed in natural language.

[Code](https://github.com/spade-rl/spade), grounding corpora, and [4B](https://huggingface.co/spade-rl/SPADE-Qwen3-4B-Games) / [30B](https://huggingface.co/spade-rl/SPADE-Qwen3-30B-A3B-Games) game weights exist. Start with a small checkpoint and inspect generated environments. Published launchers target an eight-GPU node, and release documentation does not establish exact historical-run reproduction.

**Question to answer:** what keeps the designer producing learnable new tasks rather than easy tasks, impossible tasks, or tasks whose verifier rewards the wrong behavior?

## 6. Dream-RSI: improve the search policy using past research traces

**Zheng et al.; Google, Google DeepMind, Maryland, and Virginia.** [Paper](https://arxiv.org/abs/2609.14858), §§2–4.3, Eq. 1, Table 3, and Appendix D; PDF pp. 4–10, 17, 22–27. Appendix E, pp. 27–43, contains discovered programs.

Record discovery trees containing code, outcomes, and filesystem snapshots. Replay those trees to evaluate policies choosing branches, parallelism, and stopping. A fixed coding agent rewrites policy code, and the selected policy governs the next online search. Replay reveals recorded branches; it cannot predict outcomes of untried experiments.

The clean Lasso comparison keeps the discovery backbone fixed and shares history. Gemini 3.1 Pro uses **317 versus 550 discovery calls**, with mean held-out runtime **2931 versus 3587 ms**. That average improvement comes from one of six datasets; the others slow down. Flash uses **1879 versus 3200 calls**, improving five of six datasets. Calls are not total tokens or dollars. The much larger advertised comparison changes the backbone.

The [October code release](https://github.com/google-research/dream-rsi) includes trace environments, policies, tasks, and orchestration. It excludes the harness-optimization experiment and one history ablation. The terminal-benchmark result uses the searched suite without a separate held-out suite or fixed-policy control.

For your GPT-2 experiments, the attractive idea is to reuse expensive failures when deciding *which experiment to run next*. A policy selected in replay still needs new online runs. Otherwise it may simply exploit the coverage of the saved tree.

## 7. RSI-Env: explicit successor handoff, with visible tradeoffs

**Qu et al.; Fudan, HKUST, Georgia Tech, Nanjing, Shanghai AI Laboratory, and Shanghai Innovation Institute.** [Paper](https://arxiv.org/abs/2611.01736), §§3–4, Table 2, Fig. 4, Appendix A; PDF pp. 4–9, 13–14.

An external Host Controller manages services and versioning. In the Self condition, each accepted model–harness successor conducts the next improvement round. Acceptance tests whether the system runs, not whether every capability improves.

The knowledge-directed trajectory retains a harness update, then uses that successor to construct 48 SFT examples and update weights. SimpleQA rises **82→85→88%**. This is concrete recursive continuity across two accepted updates. Meanwhile GPQA falls **65.2→53.0%** and terminal pass@3 **12.4→5.6%**. The open-general headline GPQA **65.2→73.2%** comes from one harness update with unchanged model weights.

All arms receive a 30-hour ceiling, but Self actually uses roughly 25–29 hours versus about 1–7 for fixed Teachers. This is not equal utilized compute. There are no independent complete-run replications reported, and the documented cluster has **64 A800s**. No reproduction repository was linked in the inspected paper.

Read this short, very recent paper as a handoff experiment. It sharpens the distinction between keeping a recursive process running, improving its chosen target, and maintaining general competence.

## 8. Optional: Meta^n and the value of layered context

**Kim, Lee, Jwa, and Kang; Minnesota and Seoul National University.** [Paper](https://arxiv.org/abs/2608.24735), pp. 3–9 and 11–17; [code](https://github.com/minnesotanlp/meta-n).

A fixed meta-operation adds strategic context and callable helpers around a base solver. This makes accumulated context and code explicit objects of search. Weights, evaluator, and the outer operation remain fixed.

On held-out CO-Bench, GPT-5.2 archive-best scores **0.870**, versus **0.806** for one best chain and **0.702** for OpenEvolve. Archive-best takes per-task maxima across chains, so it is not one deployable agent's score. Cross-layer-context removal is a useful ablation. However, some budget comparisons differ substantially, several tasks lack held-out splits, and the reported ARC-AGI-2 figure is marked **development set** in Table 2. More depth can lower average performance.

Study the implementation when designing an agent's reusable context and helper library. Give the stronger recursion claims less weight than the actual context mechanism and compute-controlled comparisons.

## 9. Optional: RSI-Anything and predictive feedback selection

**Cui et al.; PhAI Labs and collaborators.** [Paper](https://arxiv.org/abs/2610.13912), §§2–3 and Discussion; PDF pp. 5–16, 20–23.

A world model predicts candidate outcomes before expensive execution. Feedback can update the policy, predictor, or both; component and pair checks govern retention. The clean coding experiment holds the 7B coder fixed: hidden-test success rises **26.3% static verifier → 29.2% fixed world model → 31.2% adapted world model**, at about 3.72 test calls per task. Five laboratory-planning policy versions improve **21.80→46.01%**, only 0.69 points above the reference system.

The paper explicitly leaves sustained joint policy–world-model improvement for further evaluation. Its [reaction-optimization release](https://github.com/Gen-Verse/RSI-Anything/blob/main/tasks/reaction_optimization/README.md) clarifies that the 96.15% yield result uses offline sequential replay of measured yields in one fixed condition pool, with frozen predictions. It is not evidence of newly executed autonomous wet-lab discoveries. The immune-science therapeutic nomination awaits experimental testing.

[Public code and inference resources](https://github.com/Gen-Verse/RSI-Anything) make it useful for studying predictive selection and model-drift checks. Keep the demonstrated subloops separate from the broader proposed framework.

## What to learn or try next

1. **First two study sessions:** AIDE²'s ignition experiment, MASS's workflow-to-weight path, then the evaluation package. Write down which comparison supports each claim.
2. **Next choose a direction:** Flint for model and inference co-design; SPADE for weights and adaptive curricula; Dream-RSI for research-budget allocation. RSI-Env adds the newest successor-handoff case.
3. **A small project connected to your existing work:** retain real nanoGPT experiment traces, compare fixed and revised search policies from the same starting script, and reserve new tasks or seeds for final evaluation. Record failed runs and total API/GPU cost, not just accepted changes. This is a suggested experiment, not a result already demonstrated by these papers.

For one H800, begin with released small-model inference, code inspection, or a limited experiment-replay study. The reported MASS, SPADE, and RSI-Env training setups are substantially larger; none supplies evidence that its published full experiment fits your single-GPU setup unchanged.

## Key papers and artifacts

| Work | Version / release used | Primary reading | Artifact |
|---|---|---|---|
| AIDE² | September v1 | [2609.26457](https://arxiv.org/abs/2609.26457v1) | Exact experiment code not linked |
| MASS | October v1 | [2610.12176](https://arxiv.org/abs/2610.12176) | [SakanaAI/mass](https://github.com/SakanaAI/mass) |
| Evaluation audit | October v1 | [2610.27610](https://arxiv.org/abs/2610.27610v1) | Coding locators in PDF |
| False Frontiers | October v2 | [2609.39102v2](https://arxiv.org/abs/2609.39102v2) | No code/model link located |
| S3Gym | August v1 | [2608.31100](https://arxiv.org/abs/2608.31100) | [Project](https://self-developing-agents.github.io/) |
| Flint | October release | [Model card](https://huggingface.co/RecursiveSI/flint) | [Weights](https://huggingface.co/RecursiveSI/flint/tree/main), [engine](https://github.com/recursive-oss/rec) |
| SPADE | August v3 | [2608.19197v3](https://arxiv.org/abs/2608.19197v3) | [Code](https://github.com/spade-rl/spade), [game weights](https://huggingface.co/collections/spade-rl/spade-checkpoints) |
| Dream-RSI | October v2 | [2609.14858v2](https://arxiv.org/abs/2609.14858v2) | [Google code release](https://github.com/google-research/dream-rsi) |
| RSI-Env | November v1 | [2611.01736](https://arxiv.org/abs/2611.01736) | No reproduction link located |
| Meta^n | August v1 | [2608.24735](https://arxiv.org/abs/2608.24735) | [minnesotanlp/meta-n](https://github.com/minnesotanlp/meta-n) |
| RSI-Anything | October v1 | [2610.13912](https://arxiv.org/abs/2610.13912) | [Gen-Verse release](https://github.com/Gen-Verse/RSI-Anything) |

## 4 November comparison and revised reading order

The supplied [RSI update 2026.11.4](RSI%20update%202026.11.4.md) exposes a real coverage gap in my original research. It identifies nine relevant works I omitted, including Exo, HEMO, CARE, and two useful evaluation papers. All nine were first submitted before my 3 November cutoff; this is not explained by the one-day difference between the documents. Conversely, it omits all eleven primary works in my original guide. The two searches found largely different parts of the field.

Its discoveries deserve incorporation, but several descriptions would teach the wrong mechanism, overstate safety guarantees, or send you to unavailable artifacts. Its ten-item checklist also contains eight distinct works: ScienceCUA and SelfImprove-Bench each appear separately as paper and code. I would merge the evidence and rebuild the priorities rather than adopt either list unchanged.

For this comparison, eight newly identified papers were read in full, including their appendices. EpiAgent received a six-page methodological screening; the August Anthropic follow-up received a targeted reading of methods, results, selection procedures, and limitations. No experiments were reproduced. Artifact availability below describes what was accessible on 4 November.

### The additions that change the learning route

| Addition | First arXiv submission | What it adds | Placement |
|---|---|---|---|
| Exo | 30 October | Running agents edit their own harness | Main route, third |
| SelfImprove-Bench | 2 November | Transfer ladder and traces of actual edits | Evaluation package |
| CogRSI | 31 October | Executable semantic audits of evolved rules | Evaluation package |
| HEMO | 29 October | Search-control changes transfer to related tasks | Alongside Dream-RSI |
| CARE | 2 November | Production failures feed repeated weight updates | Main route, applied end |
| Safin-1.5 | 29 October | Released model and empirical safety-repair loop | Model branch with Flint |
| ScienceCUA | 2 November | Documentation becomes tested procedural skills | Practical skill branch |
| SPAR | 1 November | Persistent sales strategies evaluated in production | Optional application |
| EpiAgent | 29 October | Belief inference learned through self-play | Optional interaction branch |

Dates are submission dates in UTC; an arXiv identifier beginning `2611` can have an October submission date.

**Exo deserves the largest promotion.** It is an implemented runtime with [public Rust code](https://github.com/exoharness/exo), rather than the supplied report's largely theoretical architecture with “likely no code.” The agent changes tools, context handling, and other harness policies while a protected mechanism manages history and activation. Model weights remain fixed. In the three-seed AppWorld comparison, success stays at **96.9%**, while total model cost **including learning and reflection falls 43%**, from $57.36 to $32.70. The larger 61% saving counts only held-out execution. Held-out tasks are variants of the same scenario families. The default guardian checks compilation; it does not establish semantic correctness or alignment. This is useful runtime adaptation, with no controlled demonstration that later versions become better improvers. Read the architecture and AppWorld experiment together. [Paper, §§3–5](https://arxiv.org/abs/2611.00463).

**SelfImprove-Bench belongs beside Audit and S3Gym.** Its ladder separates exact replay, new instances, harder difficulty, new games, and new capability families. The supplied report later calls harder tasks G3–G4; harder difficulty is **G2**. Results are mixed rather than uniformly poor: **16/18** method–game pairs improve on unseen instances, while **8/18** regress on replay. More revealingly, the main run's **915 retained edits include no improver changes**: 87.9% concern prompt/memory, 12.1% the execution harness. An exploratory improver-evolution experiment is suggestive but narrow. Small evaluation groups, missing complete-run uncertainty, and incomplete compute accounting limit rankings. Read pp. 4–11 and the released [configuration provenance notes](https://github.com/theseus-labs-rsi/SelfImprove-Bench/blob/main/configs/main/README.md), which disclose differences from historical runs. Its contribution is a useful evaluation structure, not proof that generalization always fails. [Paper](https://arxiv.org/abs/2611.02797).

**CogRSI changes the interpretation of “auditability.”** It evolves short theory-of-mind rules, then audits their semantics using formalized worlds and executable counterexamples. This is a post-hoc audit of written rules, not the supplied report's online values auditor or chain-of-thought rebuttal network. Of **573 rules**, 286 are invalid, 44 valid, 84 ambiguous, and 159 vacuous; **86.7% invalid refers to the 330 decidable rules**, not all 573. Invalid rules can still improve answer accuracy, and a compiled rule is not necessarily the model's actual reasoning process. Expert-designed principles also outperform the evolved ones overall. Read this as a warning against equating useful explanations with valid reasoning. The advertised repository was inaccessible during this review, so “prototype code shared” was not established. [Paper, method and audit sections](https://arxiv.org/abs/2611.01113).

**HEMO is more central to RSI than its brief mention suggests.** It evolves modules controlling experience retrieval, parent selection, and operator generation. The outer proposer, scheduler, and evaluator remain fixed. For transfer, the selected harness is frozen and target memory starts empty: this tests reuse of search control, not merely reuse of successful solutions. It ranks first in **5/9** related transfer settings and second in the other four. However, equal final candidate budgets exclude up to **200 preceding harness evaluations**; example meta-training runs take tens of hours, with a reported maximum of 216.56 hours. Transfer stays within problem families. Read §§IV–V and Appendices F/H alongside Dream-RSI: both ask how to improve discovery itself, with different evidence and accounting limitations. No separate public reproduction repository was established. [Paper](https://arxiv.org/abs/2610.40023).

**CARE supplies a production learning loop.** The actual update is **on-policy self-distillation**, using the current model as a teacher given privileged diagnosis/repair information. It is not RL with an internal reward model. External inspection, component repair, and human promotion approval remain essential. Across five incremental updates, inspected Bad Case rate falls **6.42→1.97%**; simulation accuracy changes **93.0→94.1%**. The final online advantage over the incumbent is **1.73 percentage points, p=.47**, which does not establish superiority or equivalence. Changing traffic, policies, and harness repairs prevent a weights-only causal attribution. The supplied report's generic “~20% task success boost” is unsupported. Study failure attribution, teacher/student information asymmetry, and the SFT/DPO controls: an agent that immediately hands everything to humans can look deceptively good on an error metric. No CARE weight or code release was located. [Paper, Tables 6/9/10](https://arxiv.org/abs/2611.02839).

**Safin-1.5 is a real model release with a narrower safety claim.** The [public repository](https://huggingface.co/AI45Research/Safin-1.5) contains approximately 35.6B parameters and 17 weight shards. Its ReSI loop discovers attacks, searches repair recipes, trains, and promotes models through fixed empirical gates; the paper reports one posttraining round. Its optional POGB proof harness operates at inference time. Proofs depend on accepted premises and semantic bridges; they do not certify future weight updates. The [harness documentation](https://huggingface.co/AI45Research/Safin-1.5/blob/main/harness/README.md) explicitly marks semantic translation as uncertified. The empirical results also have tradeoffs: H-CoT attack success falls **82→14%**, while benign XSTest compliance falls **88.4→86.8%** against the original Safin-1. Read the repair loop, REX knowledge-removal mechanism, and proof boundary separately. The supplied report conflates these layers. [Paper, Tables 1/2/4/5 and proof appendix](https://arxiv.org/abs/2610.40751).

**ScienceCUA is valuable for practical skill engineering.** Manuals become procedural skills; execution failures motivate repairs that must succeed in a fresh session. Each evolution task starts from the same initial library, with candidates merged afterward. This is not a demonstrated sequence of globally improved generations. Grok 4.6 rises **37.47→44.70→54.59%**: the headline gain is **17.12 percentage points**, combining construction and evolution. GPT-6 Astra improves only **75.19→77.30%**. The closest evolved-skill baseline is much nearer than the vanilla model. ScienceBoard itself is earlier work; this paper audits it and adds evolution tasks. The correct [HSZD-Team repository](https://github.com/HSZD-Team/ScienceCUA) releases construction code; evolution and benchmark components are still planned. A public VM does not make the whole pipeline reproducible. Read §§3–5, Appendix I, and the stronger-model result in Appendix P. [Paper](https://arxiv.org/abs/2611.03238).

**SPAR and EpiAgent are optional branches.** [SPAR](https://arxiv.org/abs/2611.01695) uses fixed GPT-4o agents, merchant-specific episodic memory, and a shared strategy playbook in B2B sales. It is not game-session RL or offline self-play. Its live A/B test reports **15.8% relative conversion uplift**, not percentage points, without parameter updates. [EpiAgent](https://arxiv.org/abs/2610.40227), by Jin et al. rather than Rahman, trains belief inference and action planning through alternating self-play. The method screened does not evolve its optimizer or research procedure; no full empirical assessment is offered here.

### A substantial August result both documents missed

**Automated Researchers Can Mitigate Well-characterized Alignment Failures** was submitted on 28 August, with a September revision. It is the recent Anthropic follow-up to prioritize over rereading April's weak-to-strong demonstration. Five fixed researchers propose posttraining methods across ten measured failure families. The strongest methods transfer to held-out benchmarks, behavioral audits, and larger models. The research agents themselves are not updated recursively. [Paper](https://arxiv.org/abs/2608.28945).

The selection protocol matters: some later evaluations choose among leading methods using a held-out benchmark, making that split validation for that selection; the separate behavioral audit remains an additional test. Human one-shot ideas versus roughly 150 iterated automated methods is not a matched-resource comparison of research ability. In a production-checkpoint experiment, the method closes **65% of measured safety headroom**, versus 72% for the released model, using about 2,400 examples and 60 hours. That addresses the measured failures, not general alignment. The [report](https://alignment.anthropic.com/2026/automated-alignment-researchers/) and [code](https://github.com/YuehHanChen/automated_alignment_researcher) are useful; the release leaves training infrastructure to the user.

### Corrections that affect chronology or interpretation

- **AlphaEvolve:** the supplied link `2306.05685` leads to the [MT-Bench/Chatbot Arena judge paper](https://arxiv.org/abs/2306.05685). The correct [AlphaEvolve report](https://arxiv.org/abs/2506.13131) is from June 2025. Keep it as background you already know.
- **Recent publication versus recent revision:** the [Chen survey](https://arxiv.org/abs/2607.07663) began in July and was revised in September; [TTCS](https://arxiv.org/abs/2601.22628) began in January and has October revisions. TTCS can therefore be a recent-revision candidate, but not a newly introduced October method. The survey uses rule-based classification with manual correction, not individual hand-classification of all 1,250 full papers.
- **Older anchors:** Anthropic's [weak-to-strong study](https://alignment.anthropic.com/2026/automated-w2s-researcher/) belongs to April, not the supplied February timeline. [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020) is from 2024. The [XMU survey repository](https://github.com/XMUDeepLIT/Awesome-Self-Evolving-Agents) dates its survey release to February 2026, rather than the timeline's 2025.
- **“ScienceCI”:** the underlying [Anthropic article and graph](https://www.anthropic.com/institute/recursive-self-improvement) are real, but no primary system with that name was located. The article reports 76% open-ended session success in May, up 50 points over six months; a September update shows a further rise. The supplied 10→75% dates do not match. Success is judged by another model, and workloads change. Improvement across model generations does not isolate a causal RSI effect; the source itself says full RSI has not been reached.
- **Authorship and release claims:** Exo is by Krentsel et al., CogRSI by Zheng et al., and CARE by Jiang et al. The supplied first-author attributions are wrong. Exo has code; CogRSI's release was inaccessible; ScienceCUA has a partial release. These differences change which reproduction exercises are currently concrete.

### Revised merged learning order

1. **AIDE²** — retain first place for its explicit, inconclusive better-improver experiment.
2. **MASS** — retain second for workflow learning carried into weights and reused across research roles.
3. **Exo** — add runtime harness adaptation, protected mechanisms, and cost including learning.
4. **Evaluation package** — Audit → SelfImprove-Bench → S3Gym → False Frontiers → CogRSI. Together they cover causal comparisons, transfer, retention, evaluator dependence, and semantic validity.
5. **August Anthropic automated alignment researchers** — inspect whether automated research produces transferable posttraining methods, and how selection affects the evidence.
6. **SPADE** — keep the adaptive-environment and solver co-training direction.
7. **HEMO + Dream-RSI** — compare two ways to improve search control and account for the cost of learning it.
8. **CARE** — close the main route with repeated production updates and failure attribution.

Then choose a branch: **Flint + Safin-1.5** for concrete model artifacts; **ScienceCUA** for documentation and skill libraries; **RSI-Env** for successor handoff; **Meta^n, RSI-Anything, SPAR, and EpiAgent** for specialized extensions. TTCS is a revision/background option pending a full new reading.

For your own research agent, the revised list suggests three separate experiments: improve its running harness, improve the procedure that chooses its next experiments, and improve its model weights. Use a fresh task set and include learning cost when comparing each successor with its predecessor. The new papers broaden the mechanisms worth studying; they do not overturn the distinction between a better artifact and a demonstrably better improver.
