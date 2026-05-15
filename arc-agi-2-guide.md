# ARC-AGI 2

A guide to the benchmark designed to measure what current frontier LLMs are worst at — efficient few-shot abstraction on novel tasks — and why it has resisted saturation through May 2026.

## Background

**Foundational paper**: [On the Measure of Intelligence](https://arxiv.org/abs/1911.01547) (Chollet, Google, 2019). Defined intelligence operationally as *skill-acquisition efficiency over a scope of tasks, with respect to priors, experience, and generalization difficulty*. This is the thesis ARC tests.

**ARC-AGI 2 paper**: [ARC-AGI-2: A New Challenge for Frontier AI Reasoning Systems](https://arxiv.org/abs/2505.11831) (Chollet, Knoop, Kamradt, Landers, Pinkard, ARC Prize Foundation, May 2025; v2 revision Jan 2026).

**Research lineage** — how ARC evolved from a one-off 2019 dataset into a $1M+/year benchmark suite:

1. **On the Measure of Intelligence** (Chollet, 2019, [arxiv 1911.01547](https://arxiv.org/abs/1911.01547)) — Formalized "skill ≠ intelligence." Intelligence = how efficiently a system converts priors + experience into new skill, weighted by how far the new task generalizes from what was seen. Introduced ARC alongside the framework as a concrete instantiation.

2. **ARC-AGI 1 / Original ARC** (Chollet, 2019) — 1,000 public + 100 private tasks. Format: 2–5 input/output grid pairs, predict output for held-out test input(s). Designed to depend only on **Core Knowledge priors** (object persistence, goal-directedness, basic counting, geometry/topology) — no language, no world knowledge.

3. **2020 Kaggle ARC Challenge** ($20K prize) — Won by *icecuber* with a hand-built DSL + program search at 17–20%. Meta-analysis later showed that aggregating across all 2020 submissions, **49% of v1 Private Eval tasks were solved by at least one team using variations of brute-force program search**. This finding directly motivated v2.

4. **ARCathon 2022 / 2023** ($100K each) — Lab Lab AI–run contests. Top scores crept up to ~30%.

5. **ARC Prize 2024** ($1.1M, [tech report arxiv 2412.04604](https://arxiv.org/abs/2412.04604)) — Kaggle SOTA reached 53.5% (ARChitects). MindsAI's test-time training pushed unofficial scores to 55.5% on v1 Semi-Private. Industry took notice.

6. **OpenAI o3 cracks v1** (Dec 2024, [ARC Prize blog](https://arcprize.org/blog/oai-o3-pub-breakthrough)) — `o3-preview` scored 75.7% on Semi-Private v1 at ~$200/task (low-compute), and 87.5% at ~$20,000/task (172× more compute). First system to exceed Chollet's stated 85% human-comparable threshold. This was framed as "zero-to-one progress on fluid intelligence" — and also as the death knell for v1 as a frontier signal.

7. **The Surprising Effectiveness of Test-Time Training for Abstract Reasoning** (Akyürek et al., MIT, 2024, [arxiv 2411.07279](https://arxiv.org/abs/2411.07279)) — Showed that fine-tuning a base LLM on each task's few demonstration pairs at inference time dramatically improves ARC accuracy. Provided the academic foundation for what MindsAI/ARChitects had been doing empirically.

8. **ARC-AGI 2 launched** (Mar 24, 2025, [arxiv 2505.11831](https://arxiv.org/abs/2505.11831)) — Same input/output format as v1; rebuilt task distribution targeting failure modes still present in o3: symbol grounding, multi-rule composition, contextual rule application. Eval sets expanded from 100 → 120 tasks per split. **All v1 tasks ever cracked by brute-force search were excluded.** Cost-per-task elevated to a first-class reported metric.

9. **ARC Prize 2025 Kaggle results** (Nov 3, 2025 deadline) — 1,455 teams, 15,154 entries. Winner: **NVARC at 24.03%** on Private Eval, $0.20/task. Grand Prize ($700K for ≥85%) **unclaimed**. Paper prize $50K: Jolicoeur-Martineau's Tiny Recursive Model (7M params, 45% on v1, 8% on v2). [Results post](https://arcprize.org/blog/arc-prize-2025-results-analysis).

10. **Frontier model verified scores rise** (Dec 2025 – Apr 2026) — Through application-layer refinement loops and reasoning-model improvements, verified Semi-Private scores climbed: Claude Opus 4.5 (Thinking 64k) → 37.6% @ $2.20/task; GPT-5.2 Pro → 54.2%; Claude Opus 4.6 → 68.8%; Gemini 3.1 Pro → 77.1%; GPT-5.4 Pro → 83.3% (Apr 2026).

11. **ARC-AGI 3** (Mar 25, 2026, [arxiv 2603.24621](https://arxiv.org/abs/2603.24621)) — Pivot from static puzzles to 135 interactive game-like environments testing planning, memory, exploration. Humans 100%, frontier AI <1% at launch. v2 remains the active static-puzzle benchmark; v3 opens a new axis.

## What Problem Does ARC-AGI Solve?

Standard benchmarks measure **crystallized intelligence** — what a model already knows, factored through how well it can retrieve and apply that knowledge. MMLU, HumanEval, GSM8K all reward broad pretraining coverage. By 2024, frontier models had saturated most of these.

ARC tests the opposite axis: **fluid intelligence**, the ability to solve genuinely novel problems with minimal prior exposure. Each ARC task is unique, none can be "studied for," and the priors required (object permanence, counting, basic geometry) are universal across humans.

```
                          Crystallized intelligence          Fluid intelligence
                          ────────────────────────           ─────────────────────────
What it measures          Stored knowledge + retrieval        On-the-fly abstraction
                                                              and rule induction
Standard benchmark        MMLU, GPQA, HumanEval                ARC-AGI
LLM strength              Very high (saturated)                Very low (<10% on v2
                                                               for raw frontier models)
How to improve            More pretraining data                Not obvious
Chollet's term            "skill"                              "intelligence"
```

Chollet's 2019 formula, paraphrased:

```
intelligence(system) = skill-acquisition efficiency over a scope of tasks
                       given (priors, experience, generalization difficulty)
```

Crucial implication: a system that scores 95% on a benchmark using terabytes of training data and millions of dollars of compute is *not* demonstrating intelligence in this sense — it is demonstrating that priors + experience were sufficient. A system that learns a new task from 3 examples in 2 minutes *is* demonstrating intelligence, even if its absolute skill on that task is modest.

ARC operationalizes this by giving every model the same few-shot setup, holding priors constant (Core Knowledge only), and varying only the task. The benchmark cannot be cheated by memorization — every task is novel.

## Key Terms

**Task**: A single ARC puzzle. 2–5 input/output demo grid pairs plus 1–4 test inputs. The model must infer the underlying rule from demos and produce the correct output grid for each test input.

**Grid**: 2D array of cells, each holding one of 10 discrete colors (semantically just symbols 0–9). Size ranges from 1×1 to 30×30.

**Core Knowledge priors**: The innate cognitive primitives Chollet assumes humans share — object persistence, basic geometry/topology, elementary counting/number-sense, goal-directedness. ARC tasks are designed to require *only* these, no language or culturally learned facts.

**Fluid intelligence**: Ability to reason about and solve novel problems independently of stored knowledge. Tested by ARC.

**Crystallized intelligence**: Accumulated knowledge and learned skills. Tested by MMLU, etc.

**pass@2**: Native ARC metric. Each test input gets up to 2 attempts; a task is solved if either attempt is correct. Mirrors the human protocol (humans on the panel also got 2 tries).

**Semi-Private Eval**: 120-task hidden set used for verified evaluation of remote-hosted frontier models (Claude, GPT, Gemini). Tasks are not public but can be probed via API.

**Private Eval**: 120-task fully-hidden set used for the Kaggle competition. Submitted code runs offline inside the Kaggle sandbox, never sees the tasks unencrypted.

**Test-Time Training (TTT) / Test-Time Fine-Tuning (TTFT)**: At inference, fine-tune (some or all) model weights on the demo pairs of the current task before predicting the test output. Distinct from in-context learning — actual gradient steps happen at inference.

**AIRV**: MindsAI's TTT recipe — Augment + Inference + Reverse-augmentation + Vote. Generate many augmented versions of the demos, fine-tune on all, predict, invert each augmentation, majority-vote.

**Refinement loop**: Outer-loop search at inference. Generate candidate solutions (programs or output grids), score them against demos with an automatic verifier, mutate and retry. The dominant 2025–2026 frontier approach.

## ARC Task Format

Every task has the same shape:

```
Task:
  train: [(input_0, output_0), (input_1, output_1), (input_2, output_2)]   ← 2-5 demo pairs
  test:  [input_test_0]                                                     ← 1-4 test inputs

Each grid: HxW matrix of integers in {0..9}, with 1 ≤ H,W ≤ 30
```

A simple v1-style example (a "fill the rectangle" rule):

```
Demo 1:                          Demo 2:
input:        output:            input:        output:
0 0 0 0       0 0 0 0            5 0 0 0       5 5 5 5
0 1 0 0   →   0 1 1 0            0 0 0 0   →   5 5 5 5
0 0 0 0       0 0 0 0            0 0 0 0       5 5 5 5
0 0 0 0       0 0 0 0            0 0 0 5       5 5 5 5

Demo 3:                          Test:
input:        output:            input:        output:
0 0 0 2       2 2 2 2            0 0 0 0       ?
0 0 0 0   →   2 2 2 2            0 0 3 0
0 0 0 0       2 2 2 2            0 0 0 0
2 0 0 0       2 2 2 2            0 0 0 0
```

Rule to induce from demos: "If any cell is nonzero, fill the entire grid with that color." Predicted output for the test:

```
3 3 3 3
3 3 3 3
3 3 3 3
3 3 3 3
```

This is a v1-difficulty task. A 124M GPT-2–scale model cannot do this out of the box (no spatial priors, no obvious 2D tokenization). Humans solve it in seconds.

**Why this format is hard for LLMs**:
1. **Tokenization mismatch** — A grid is fundamentally 2D, but tokenizers serialize it as a 1D string. Spatial neighbors become far apart in token-space.
2. **No transfer from text** — Pretraining gives you nothing about colored grids.
3. **Few-shot from scratch** — 3 demo pairs is not enough to fit any kind of statistical model in the traditional sense.

The puzzle isn't "what's the answer" — it's "what's the rule," and rules are program-like, not pattern-like.

## Why ARC-AGI 1 Needed a Successor

Three concrete problems with v1 by late 2024:

### 1. Brute-force solvability

The 2020 Kaggle contest revealed something the original ARC team hadn't fully appreciated: many v1 tasks had short programs in a hand-built DSL, and exhaustive search over that DSL could find them. Each individual team solved ~20% of Private Eval, but *the union of solutions across teams* covered **49% of v1 Private Eval**.

This was a benchmark-design issue, not a model-capability one. A v1 score of 49% could be achieved without anything that looked like reasoning — just enumerative search.

### 2. Leaderboard leakage

The same 100 Private Eval tasks were reused unchanged across **four major competitions from 2020 to 2024**, with an estimated 10,000+ scores publicly disclosed. Teams could implicitly probe task structure from leaderboard feedback, even without ever seeing the tasks. This is a milder version of test-set contamination — the test items leak indirectly through the score signal.

### 3. Saturation at the top

When o3 hit 87.5% on v1 Semi-Private in Dec 2024 (at $20K/task), it crossed Chollet's stated 85% human-comparable threshold. v1 was no longer a frontier discriminator — it could only distinguish "can do ARC" from "cannot," not finer gradations of how well.

The combination meant v1 had stopped measuring what it was supposed to measure. v2 was the response.

## What's Different in v2

Four design changes, all targeted at known frontier-model failure modes:

```
Change                          Why                                Effect
────────────────────────────    ────────────────────────────       ────────────────────────
1. Every task is novel           Resist memorization                Forces fluid reasoning
   (no v1 carryover)             Anti-brute-force curation         Search space too big
   tasks explicitly excluded

2. Larger grids,                  More compositional cognition       Combinatorial blow-up
   more objects per grid          More bits of information per task  defeats program search
   more concepts per task

3. Anti-brute-force design        Search must combine many ops       Naive DSL enumeration
   (all v1 brute-forced tasks    to solve any single task          takes too long
   removed from candidate pool)

4. Wider difficulty spectrum      Granular signal across the          Distinguishes between
   (calibrated via 407 humans,    fluid-intelligence range            "weak reasoner" and
   515 sessions)                                                       "strong reasoner"
```

The v2 paper identifies four specific task categories that are deliberately overrepresented in v2 vs v1.

### Task Category 1: Multi-rule compositional reasoning

Multiple distinct rules apply simultaneously and must be induced together. One concrete example from the paper (task `898e7135`):

- Rule A: Crop the input to a framed sub-region defined by a colored border
- Rule B: Rescale colored objects in the cropped region by some factor
- Rule C: Place rescaled objects into holes of matching shape

You cannot solve this by sequencing — the three rules constrain each other. Identifying rule B in isolation is meaningless without rule C telling you what "matching shape" means.

```
Why it's hard for LLMs:
  Pretraining gives you "find a transformation"; v2 demands "find a transformation
  composed of three sub-transformations that only make sense together."
  Greedy chain-of-thought can find rule A, get stuck looking for rule B, and stop.
```

### Task Category 2: Multi-step compositional reasoning

Sequential rule application where state at step N depends on state at step N-1. Cannot be predicted in parallel — must actually execute the procedure.

```
Demo intuition: an input grid where colored arrows in a chain each point to the next.
The "answer" is the final position after walking N steps, where each step depends
on the orientation at the previous step.

You cannot vectorize this. The model must "run" the procedure step by step.
```

This is the closest ARC analogue to algorithmic reasoning — and exactly what o-series reasoning models *should* be good at. Their poor performance on these tasks (early o3 scored 3% on v2 vs 53% on v1) is the headline finding.

### Task Category 3: Contextual rule application

The same local transformation must be applied differently depending on global context. Example: an arrow points "in the direction the same-colored object should move," but only if there's exactly one matching object; otherwise it points to a stationary marker.

```
Why it's hard:
  In-context learning has to identify not just the rule, but also the metarule
  (which version of the rule applies). This is two-level reasoning.
```

### Task Category 4: In-context symbol definition

The task itself defines what its symbols mean. Example: a key region of the input grid shows "red rectangle with 2 holes = use red," "blue rectangle with 3 holes = use blue." Elsewhere in the grid, shapes with 2 and 3 holes must be filled with the corresponding color.

```
Why it's hard:
  Models have strong priors about what colors and shapes "mean" from pretraining
  (red is hot, etc.). v2 deliberately makes meaning *task-local* — the model must
  override its pretrained associations and read the key off the input.
```

This is the most direct test of fluid intelligence: a system has to construct a new symbol table from scratch within a single task.

## A Concrete Numerical Walkthrough

Consider a stylized in-context symbol-definition task. The input contains a "legend" region and a "puzzle" region. The legend defines: shape-with-2-holes → color 3, shape-with-3-holes → color 7. The puzzle has uncolored shapes; output should be the same shapes colored according to the legend.

```
Input (12x6 grid, separated by a column of 9s):

 8 8 0 8 8 9 0 8 8 8 0 0
 8 0 0 0 8 9 8 8 0 8 8 0
 8 8 0 8 8 9 8 0 0 0 8 0
 ─────legend──── 9 ──puzzle──

In legend: 2-hole shape uses fill color 3 (encoded elsewhere)
           3-hole shape uses fill color 7

Expected output: puzzle shapes filled with the right color per its hole-count.
```

To solve this, a system must execute (roughly):

1. Parse the grid into 2D objects (using object permanence prior).
2. Detect the separator column and split legend from puzzle.
3. For each legend object: count holes (topology prior), look up its color (some pixel adjacent).
4. Build a {hole-count → color} dictionary from the legend.
5. For each puzzle object: count holes, look up color, fill.

Five conceptual steps. An LLM trying this in chain-of-thought needs to:
- Hold the 2D structure in textual representation (every newline character matters)
- Reliably count holes (which it isn't trained to do on grids)
- Generalize the legend-key abstraction (which is task-local, not pretraining-derived)
- Output a 30+ character grid without drift

Early o3 zero-shot on tasks like this: ~3%. Why so low when o3 is otherwise an excellent reasoner? Because the symbol-grounding step is *not* something extended chain-of-thought helps with — the symbols aren't in the pretraining distribution, so longer thinking doesn't recover them.

This is what people mean when they say ARC is "orthogonal" to LLM capability: more pretraining and more reasoning compute help on most benchmarks; on ARC-AGI 2, both have limited returns until the symbol-grounding bottleneck is addressed.

## The Scoring Landscape (May 2026)

Frontier model scores on ARC-AGI 2 Semi-Private (verified), in rough chronological order of when each result was reported:

```
Model / System                        Score    $ / task     Date          Notes
────────────────────────────────────  ───────  ──────────  ───────────  ─────────────────────────
o3-mini (High)                         3.0%      —          May 2025     Paper baseline
o3 (Medium)                            3.0%      —          May 2025     Paper baseline
o4-mini (Medium)                       2.4%      —          May 2025     Paper baseline
ARChitects (2024 v1 winner)            2.5%      —          May 2025     Tuned on v1, didn't transfer
o3 (High)                              6.5%      $0.83      Jun 2025     ARC Prize verified
o4-mini (High)                         6.1%      $0.86      Jun 2025
Claude Opus 4 (Thinking 16k)           8.6%      $1.93      Jun 2025
Gemini 2.5 Pro                         3.8%      $0.81      Jun 2025
Claude Opus 4.5 (Thinking 64k)        37.6%      $2.20      Dec 2025     First serious jump
Gemini 3 Pro (baseline)               31.1%      $0.81      Dec 2025
Gemini 3 Deep Think (Preview)         45.1%     $77.16      Dec 2025
Poetiq refinement on Gemini 3 Pro     54.0%     $30         Dec 2025     Outer-loop refinement
GPT-5.2 Thinking                      52.9%      —          Dec 2025
GPT-5.2 Pro                           54.2%      —          Dec 2025
Claude Opus 4.6                       68.8%      —          Feb 2026
Claude Sonnet 4.6                     58.3%      —          Feb 2026
Gemini 3.1 Pro                        77.1%      —          Feb 2026
GPT-5.4 Pro                           83.3%      —          Apr 2026     Current verified SOTA
Human panel (average)                 60.0%     ~$17        baseline      407 participants
Human panel (at-least-2 pass@2)       100%       —          calibration   By construction
```

Kaggle 2025 Private Eval (compute-capped, no internet, L4×4 GPUs, 12hr limit):

```
Rank   Team             Private Eval     $ / task
────   ──────────────   ──────────────   ───────────
1      NVARC            24.03%           $0.20
2      ARChitects       16.53%           —
3      MindsAI          12.64%           —
4      Lonnie            6.67%           —
5      G. Barbadillo     6.53%           —
```

**Key observations**:

1. **Initial frontier-model gap was brutal** — May 2025: all major reasoning models scored under 5% on v2 while scoring 30–60% on v1. The paper notes 5% is the floor of "meaningful signal" — below that you're seeing noise plus heuristics.

2. **The verified-track curve has been steeper than expected** — Going from 3% (May 2025) → 83% (Apr 2026) in 11 months is faster than v1's 5%→88% which took 4 years (2020→2024). Better understanding of what works (refinement loops, reasoning + verifier) accelerated this.

3. **The Kaggle curve is much shallower** — From open-source, compute-capped submissions: 24% is best. The gap (83% verified vs 24% Kaggle) is the compute-and-knowledge-coverage premium. The grand prize ($700K for ≥85% in Kaggle conditions) remains uncollected.

4. **Cost matters** — Gemini 3 Deep Think at $77/task is 95× more expensive than baseline Gemini 3 Pro for ~14 percentage points of accuracy. ARC's published 2D leaderboard (accuracy × cost) is the only frontier benchmark that makes efficiency a first-class metric.

## Methodology Approaches

ARC has produced a more diverse methodology zoo than any other benchmark, precisely because no single approach dominates.

### 1. Pure LLM zero-shot

Just ask the model. Even with chain-of-thought.

- Score on v2: essentially 0%.
- Why: symbol-grounding bottleneck, 2D tokenization mismatch, no transfer from text pretraining.
- Useful as a sanity baseline only.

### 2. Reasoning models (o-series, GPT-5 thinking, Gemini Deep Think)

Extended chain-of-thought with self-verification and backtracking, trained via RL on reasoning traces. Closest analogue in your curriculum: see [test-time-compute-guide.md](test-time-compute-guide.md) and [model-o1-guide.md](model-o1-guide.md).

- Score on v2: 3–55% depending on model and effort budget.
- Why it scales: longer reasoning chains let the model try multiple rule hypotheses and reject the ones inconsistent with later demos.
- Why it caps: the model can't reason itself into a symbol grounding it doesn't have. RL on math/code reasoning doesn't transfer perfectly to grid puzzles.
- Cost profile: $0.20 to $77+ per task depending on `reasoning_effort`.

### 3. Program synthesis / DSL enumeration

Hand-build a domain-specific language of grid transformations (rotate, recolor, copy-by-pattern, etc.) and search over short programs.

- Score on v2: ~5% standalone, ~15% in ensembles.
- Why it works on v1: tasks had short DSL programs and small search spaces.
- Why it struggles on v2: anti-brute-force curation. Longer compositions = exponentially larger search. The DSL must also be much richer to express multi-rule and contextual transformations, which makes search even worse.
- Lineage: icecuber (2020 winner), Michael Hodel's `dsl` work, Ryan Greenblatt's LLM-guided Python search.

### 4. Test-Time Training (TTT) and Test-Time Fine-Tuning (TTFT)

Fine-tune the model on the demo pairs of the current task, at inference time, before predicting the test output. This is the most distinctive ARC methodology and connects directly to your training-pipeline background.

The basic loop (MindsAI's AIRV):

```
For each test task:
  1. Receive demos D = [(input_i, output_i)]                ← 2-5 pairs
  2. Generate augmented variants:                            ← rotations, color permutations,
     D_aug = augment(D)  →  500-5000 augmented pairs           reflections, transpose
  3. Fine-tune the base model on D_aug for K steps:         ← K ≈ 10-200 LoRA updates
     model_t = TTFT(base_model, D_aug)                          on a small LoRA adapter
  4. Generate N predictions per augmented version of test:
     for each aug_fn in augmentations:
       y_aug = model_t.predict(aug_fn(test_input))
       y_inv = aug_fn⁻¹(y_aug)                              ← invert the augmentation
       predictions.append(y_inv)
  5. Majority vote across predictions
  6. Discard the per-task fine-tuned weights
```

Why this works:
- Each task has its own latent rule. A single static model has to learn 120 distinct rules at once (impossible). TTFT lets the model **specialize per-task**, then throw the specialization away.
- Augmentation expands the effective training set from 3 pairs to thousands, which is enough to actually move weights without overfitting to noise.
- Inverse-augmentation voting reduces variance from any single rotation/permutation being unlucky.

Why this connects to your work:
- It's structurally similar to your GRPO setup, but at inference: a small training loop runs every time the model encounters a new task. Where GRPO trains the policy on rollouts, AIRV trains the policy on augmented demos.
- The base model is typically a SFT'd LLM (Llama, Qwen) fine-tuned on the public ARC training set with augmentations — like your SFT step on Dolly/SlimOrca, but for grid puzzles. The TTT loop is then a *second* SFT step at inference.
- LoRA is the dominant choice because full fine-tuning is too slow inside the 12-hour Kaggle window.

Results:
- MindsAI on v1 Semi-Private: 55.5% (2024).
- MindsAI on v2 Kaggle Private: 12.64% (2025, 3rd place).
- Akyürek et al.'s academic version on v1: 53%.

TTT was the dominant 2024 approach. In 2025 it was partially displaced by refinement loops, which proved more scalable.

### 5. LLM-with-search / refinement loops

The dominant 2025 frontier approach. A strong reasoning model generates candidate solutions; an automatic verifier (running the candidate program on the demos and checking output match) filters or scores them; the model mutates and retries.

The loop:

```
For each test task:
  candidates = []
  for iteration in 1..N:
    proposal = LLM.generate(demos + previous_failures)      ← reasoning model proposes
                                                              a rule or program
    score = verify(proposal, demos)                         ← run on demos, count matches
    if score == perfect:
      candidates.append(proposal)
      break
    else:
      previous_failures.append((proposal, where_it_failed))
  return best(candidates) applied to test_input
```

The verifier is the key. ARC has a perfect automatic verifier built into the task structure: a candidate rule is correct iff it reproduces all the demo outputs. This is what makes refinement loops so much more effective on ARC than on tasks without automatic verifiability.

Notable examples:
- **Poetiq** (Dec 2025): Refinement harness wrapping Gemini 3 Pro. 31% baseline → 54% with refinement, at ~$30/task.
- **Jeremy Berman's evolutionary search**: Each iteration mutates a Python solution, scores against demos, keeps the best.
- **Greenblatt's LLM-guided enumeration** (2024): Used GPT-4 to generate Python programs for each task; ran them and kept the ones that matched demos. 41% on v1 at the time.

This pattern is increasingly how frontier models score well: model + verifier + outer loop, not model alone.

### 6. Tiny recursive networks (TRM)

[Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/abs/2510.04871) (Jolicoeur-Martineau, Samsung SAIT, 2025). Won the $50K ARC Prize 2025 Paper Prize.

7M parameters. No pretraining. Recursive self-refinement loop where the network repeatedly updates its candidate output, conditioned on demos.

- ARC-AGI 1: 45%
- ARC-AGI 2: 8%

Why this matters: it's an existence proof that ARC progress isn't purely about scaling. A 7M-parameter network can beat a 670B reasoning model (zero-shot Claude on v2 is single-digits). The recursion/refinement structure matters more than parameter count for this specific task type.

Connection to your work: TRM is closer to a neural Turing machine than a transformer. It's a useful counterexample to the "just scale up" thesis and lines up with the bitter-lesson critiques of ARC as a benchmark.

### 7. Neuro-symbolic ensembles

Combine a neural module (proposal generation, perception) with a symbolic module (verified rule execution, search). The 2024 ARChitects winner was effectively this — a neural network proposed program sketches, a symbolic verifier filled in details.

- Best ensemble v1 score: 53.5% (ARChitects, 2024)
- v2: harder because the symbolic search space exploded; ARChitects scored only 2.5% on v2.

## Method Comparison

```
Approach              Best v1     Best v2     Cost on v2      Strength                 Limit
─────────────────     ────────    ────────    ──────────     ──────────────────────   ─────────────────
Pure LLM zero-shot    ~5%         ~0%         $0.01           Trivial to deploy        No abstraction
Reasoning model       88% (o3)    83% (GPT-5.4) $0.20-$77    Best raw frontier        Symbol grounding
Program synthesis     ~20%        ~5%         $0.05           Interpretable            Search explodes
Test-time training    55%         13%         $0.20 Kaggle    Adapts per task          Brittle outside DSL
                                              (capped)        Sample-efficient
Refinement loop       —           54-83%      $2-$77          Compounds with model     Cost scales fast
                                                              Best verified scores
TRM (recursive net)   45%         8%          (tiny)          Param-efficient          Domain-specific
Neuro-symbolic        53.5%       <5%         varies          Combines strengths       Engineering cost
Human panel           ~95%        60% avg     ~$17/task       Reliable baseline        Doesn't scale
```

## ARC Prize 2025: The Rules

Two tracks with very different rules:

### Kaggle track (eligible for Grand Prize)

```
Compute:         4× NVIDIA L4 GPUs (Kaggle-provided)
Time:            12 hours total wall-clock for 240 tasks
                 (120 Semi-Private + 120 Private)
Internet:        Forbidden during evaluation
Eval set:        Private Eval (120 tasks, fully hidden)
Cost/task:       ~$0.42 effective (from $50/notebook ÷ 120 tasks)
Open source:     Required to claim any prize
Grand Prize:     $700K for first team to exceed 85% on Private Eval
                 at roughly human-comparable efficiency (~$2.50/task)
Paper Prize:     $50K + $20K + $5K for top methodology papers
2025 result:     NVARC won at 24.03%; Grand Prize unclaimed
```

### Verified track (commercial models)

```
Compute:         Unlimited (provider's choice)
Time:            No fixed limit
Internet:        Allowed (API calls to remote models)
Eval set:        Semi-Private Eval (120 tasks, hidden but probeable via API)
Cost cap:        $10,000 per full run
Open source:     Not required
Prize:           Not eligible for Grand Prize
Purpose:         Public 2D leaderboard tracking accuracy vs cost-per-task for
                 frontier commercial models. Adjudicated by ARC Prize Foundation.
2025-2026:       Climbed 3% → 83% over ~12 months
```

The verified track exists because commercial frontier models can't run inside the Kaggle sandbox (closed-weight, requires API), and because measuring cost-vs-accuracy across closed systems requires a third-party adjudicator. The track explicitly does not compete for the Grand Prize — it's a public capability benchmark.

## Cost as a First-Class Metric

ARC is the only major benchmark that publishes a 2D leaderboard with accuracy on one axis and $/task on the other. The 2025 paper makes this explicit:

> "Intelligence is not just about being correct — it's about being correct efficiently. A system that requires $20,000 per task to score 88% has not demonstrated the same capability as one that scores 88% for $20."

This directly extends Chollet's 2019 framing — efficiency is part of the intelligence definition, not a separate engineering concern. The same model running at `reasoning.effort=xhigh` for 10× longer is not a more intelligent system; it's the same system using more compute, which is exactly what Chollet's framework would *factor out*.

Some practical consequences:
- Two scores can be reported for the same model at different effort levels. The benchmark cares about both endpoints of the cost curve.
- The Grand Prize requires not just 85% accuracy but ~$2.50/task efficiency.
- Frontier model providers are increasingly reporting `accuracy @ cost` rather than just `accuracy`.

## Critiques of ARC-AGI as an AGI Proxy

ARC has its critics. The most credible:

### Melanie Mitchell (Santa Fe Institute)

Mitchell sits on ARC Prize's independent academic audit panel and is the source of the most articulate critique.

1. **AGI branding is overclaim**: "I don't love the term 'AGI,' and I don't think that solving ARC is necessarily the golden ticket to achieving AGI." ARC measures one slice of intelligence (few-shot visual abstraction); AGI requires far more.

2. **Goodhart's law**: $1M+ prizes corrupt the measure. When money is on the table, teams optimize the score, not the underlying construct.

3. **Brute-force violates the spirit**: Methods that win by massive search "violate the spirit" of the benchmark. v2 partially addresses this with anti-brute-force curation, but the critique reappears at higher abstraction levels (e.g., refinement loops are also search, just at the program level).

4. **Generalization test**: Mitchell proposes evaluating whether systems can transfer ARC reasoning to variant tasks or other domains, to verify they learned reasoning vs ARC-specific patterns. Her parallel benchmark *ConceptARC* ([arxiv 2305.07141](https://arxiv.org/abs/2305.07141)) implements a version of this.

### Knowledge contamination

The 2025 paper itself acknowledges a subtler problem: modern reasoning systems can overfit a benchmark *family* through knowledge coverage, even without seeing the specific test tasks. If 10,000 ARC-style tasks circulate in pretraining corpora, a model may have effectively absorbed enough "ARC priors" to inflate fluid-intelligence-like scores via crystallized knowledge.

Mitigation: keep eval sets hidden, rotate them, and prefer fully-hidden Private Eval signals over Semi-Private API-probeable ones.

### Domain narrowness

Static colored grids are informative about abstract visual rule induction, but they are not language use, embodied control, social cognition, planning under uncertainty, or long-horizon tool use. ARC-AGI 2 measures one capability dimension well; it does not measure most of what an AGI would need to do.

The ARC team agrees — they explicitly state ARC-AGI is "a measure of capability progress, not a litmus test for AGI." ARC-AGI 3 (March 2026) addresses some of this by moving to interactive game environments testing planning and exploration.

### "Bitter Lesson" critique

The [2024 OpenReview position paper](https://openreview.net/pdf?id=GCqffUAyiu) "Position: Bitter lesson of the ARC-AGI Challenge" argues that ARC's Core Knowledge framing may be fundamentally wrong — that intelligence may emerge from scale and gradient descent in ways that don't respect Chollet's clean priors-vs-experience-vs-generalization decomposition. The counter to TRM (small recursive network beats big LLMs on the structural axis) is that maybe both are wrong and the real path is yet bigger models with the right loss.

## Connection to Your Curriculum

ARC-AGI 2 sits at the intersection of several things you've already studied:

```
You've already done:                         How it connects to ARC-AGI 2:
─────────────────────────                    ─────────────────────────────────────────
SFT (Dolly, SlimOrca)                        Base for TTFT — ARC solvers SFT on the
                                             public training set + augmentations first.

DPO on hh-rlhf                                Not directly applicable — no preference
                                             pairs in ARC.

GRPO                                          Structurally similar to TTT: outer-loop
                                             updates the model based on a reward signal.
                                             AIRV is GRPO-with-augmentation-rewards at
                                             inference time.

Test-time compute                             Reasoning models on ARC = test-time compute
                                             on grid tasks. See test-time-compute-guide.md.
                                             ARC adds: verifier-based refinement, which
                                             is structurally similar to best-of-N with
                                             a perfect process reward model.

Distillation (Hinton, R1-style)               R1-style reasoning distillation could be
                                             applied to specialize a small model on ARC
                                             demonstrations from a large solver.

Test-time compute → reasoning models          o-series and GPT-5 thinking are the largest
                                             contributors to recent v2 progress, but they
                                             cap around 50-60% without external refinement.
                                             The frontier is reasoning + refinement.
```

What you cannot get from your existing pipeline (GPT-2 on language) and would need to add to attempt ARC seriously:

1. **A 2D-aware tokenization or input scheme**. Either custom 2D positional embeddings or a tokenizer that respects grid structure.
2. **Massive grid-task augmentation**. The MindsAI recipe generates 500–5000 augmented variants per task from rotations/color-permutations/reflections.
3. **An automatic verifier**. The "run candidate rule on demos and check" loop is what makes refinement possible.
4. **A small per-task LoRA**. Full fine-tuning at inference is too slow.

If you wanted a weekend project: take your GPT-2 124M, do a fast SFT pass on the 1,000 public ARC training tasks (encoded as `<grid>` text), then implement a 50-step TTFT LoRA loop with rotation/color augmentation on each evaluation task. Expected v1 score: ~10–15%. Expected v2 score: ~1–3%. You'll feel the symbol-grounding bottleneck personally.

## Key Papers

| Paper | Authors | Year | Key Contribution |
|-------|---------|------|------------------|
| [On the Measure of Intelligence](https://arxiv.org/abs/1911.01547) | Chollet (Google) | 2019 | Intelligence = skill-acquisition efficiency; ARC framework |
| [ARC-AGI 2 Technical Report](https://arxiv.org/abs/2505.11831) | Chollet, Knoop, Kamradt, et al. (ARC Prize Foundation) | 2025 | The v2 benchmark paper; failure-mode taxonomy |
| [ARC Prize 2024 Tech Report](https://arxiv.org/abs/2412.04604) | ARC Prize Foundation | 2024 | TTT/program-synthesis methodology survey on v1 |
| [The Surprising Effectiveness of Test-Time Training](https://arxiv.org/abs/2411.07279) | Akyürek et al. (MIT) | 2024 | Academic formalization of TTT for ARC |
| [The LLM ARChitect](https://arxiv.org/abs/2412.04604) | Franzen, Disselhoff, Hartmann | 2024 | 2024 Kaggle winner methodology |
| [Less is More: Recursive Reasoning with Tiny Networks (TRM)](https://arxiv.org/abs/2510.04871) | Jolicoeur-Martineau (Samsung SAIT) | 2025 | 7M-param recursive network; 2025 paper prize |
| [ConceptARC](https://arxiv.org/abs/2305.07141) | Mitchell, Moskvichev, Steiger | 2023 | Variant benchmark probing concept grounding |
| [Position: Bitter Lesson of ARC-AGI](https://openreview.net/pdf?id=GCqffUAyiu) | Anonymous | 2024 | Scale-and-gradient-descent critique of ARC framing |
| [o3 Breakthrough Blog](https://arcprize.org/blog/oai-o3-pub-breakthrough) | ARC Prize | 2024-12 | The result that triggered v2 |
| [Analyzing o3/o4-mini with ARC-AGI](https://arcprize.org/blog/analyzing-o3-with-arc-agi) | ARC Prize | 2025 | Cost-vs-accuracy analysis |
| [ARC Prize 2025 Results](https://arcprize.org/blog/arc-prize-2025-results-analysis) | ARC Prize | 2025-12 | Kaggle results; refinement loop analysis |
| [ARC-AGI 3 Launch](https://arxiv.org/abs/2603.24621) | ARC Prize Foundation | 2026 | Successor: interactive environments |

## Where ARC-AGI 2 Fits in the Bigger Picture

```
Crystallized intelligence (saturated)            Fluid intelligence (active frontier)
                                                  
MMLU → MMLU-Pro → HLE         vs              ARC-AGI 1 (saturated by o3)
GSM8K → MATH → FrontierMath                   ARC-AGI 2 (~83% verified, 24% Kaggle, May 2026)
HumanEval → SWE-bench Pro                     ARC-AGI 3 (interactive, <1% AI at launch)
                                                  
Frontier scores: ~50-95%                      Frontier scores on v2: ~80% verified
Saturating fast                                Climbing but still gap to human ceiling
```

ARC-AGI 2's continuing role through mid-2026 is to be the cleanest single benchmark for asking: how efficiently can a system reason about genuinely novel problems from few demonstrations? The answer in May 2026 is "much better than 12 months ago, still meaningfully worse than humans on accuracy, and dramatically worse on cost." The benchmark is doing its job.
