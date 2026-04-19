# Major LLM Benchmarks (April 2026)

*Last updated: 2026-04-20*

## 1. Why this guide, why these 13

Benchmarks are the closest thing we have to a shared ruler for frontier model capability. They shape release timing, pricing tiers, and billions in capex decisions. But the field has a chronic saturation problem: the moment a benchmark is publicized, its test set leaks into pretraining corpora, reasoning models grind down remaining headroom, and the signal dies. MMLU (2020) was the field's north star for three years; by late 2023 it was pinned above 90% and useless for frontier comparison. HumanEval, GSM8K, MATH, HellaSwag — all retired by similar mechanics.

2026 looks very different from 2023. The interesting benchmarks now probe: (a) *research-level* reasoning where humans themselves split (HLE, FrontierMath), (b) *multi-step agent* behavior with shared state and tools (τ²-Bench, Terminal-Bench), (c) *long-horizon* task completion measured in wall-clock hours (METR Time Horizons), and (d) contamination-resistant *live* or *private* evaluation (SWE-bench Pro, LiveCodeBench, Scale SEAL). Single-turn QA is mostly dead as a frontier signal.

This guide covers 13 benchmarks worth knowing as of April 2026. It is deliberately selective — a long list would mostly be tombstones.

---

## 2. The 13 benchmarks

### 1. HLE — Humanity's Last Exam

Category: reasoning-math
Released: 2025-01 (arXiv 2501.14249)
Size: ~2,500–3,000 closed-ended expert questions across 100+ subjects (41% math, 11% bio/med, 9% physics, 10% CS, rest humanities/law/linguistics)
Source: https://arxiv.org/abs/2501.14249, https://agi.safe.ai/, https://labs.scale.com/leaderboard/humanitys_last_exam

Why it matters in 2026: Built by the Center for AI Safety + Scale as explicit replacement for saturated MMLU/GPQA. Questions crowdsourced from ~1,000 subject-matter experts with the specific instruction "write something a PhD in your field could answer but a model cannot." The closed-ended format keeps it machine-gradable while pushing into genuine tail-of-distribution knowledge. Still a live signal — not yet saturated.

Current SOTA (as of Apr 2026):
- Gemini 3.1 Pro Preview: 46.44% (Scale leaderboard)
- Gemini 3 Pro Preview: 37.5%
- Claude Opus 4.6 Thinking: 34.4%
- GPT-5 Pro: 31.6%
- Human experts (domain-matched): ~90%

Example tasks:
- Tiberian Biblical Hebrew: identify closed syllables in a given verse
- Translate a Palmyrene-script inscription on a Roman-era tombstone
- Anatomy: name the paired tendons supported by the hummingbird sesamoid bone

How to read the score: HLE is contamination-resistant by construction (questions held private until release, many require multi-step domain expertise even for humans). A 46% means the model answers just under half of what should be PhD-tail questions — strong, but still 2x gap to expert humans. Beware though: HLE mixes disciplines unevenly, so strong physics scores can mask weak humanities scores.

---

### 2. FrontierMath — Epoch AI

Category: reasoning-math
Released: 2024-11, expanded through 2025
Size: ~300+ problems across 4 tiers; Tier 4 is research-level math
Source: https://epoch.ai/frontiermath, https://openai.com/index/introducing-gpt-5-4/

Why it matters in 2026: Problems are written by working research mathematicians (Fields medalists on the advisory panel), kept *entirely private*, and require original mathematical reasoning — not retrieval of known theorems. Terence Tao's quote on release: "extremely challenging, [solving them] would represent a major advance." Tier 4 is genuinely at the frontier of human math research.

Current SOTA (as of Apr 2026):
- GPT-5.4 Pro: 50.0% on Tier 1–3, 38.0% on Tier 4 (also cited as 47.6% overall; 17/48 Tier 4 solved as of Jan 2026)
- Prior frontier (early 2025): <10% on any tier

Example tasks:
- Compute the Galois group of an explicitly given polynomial over Q
- Prove a bound on a novel combinatorial sum
- Derive a number-theoretic inequality under stated constraints

How to read the score: Because the problem set is private and graded by Epoch, contamination risk is minimal. A 17/48 Tier-4 score should be read as "the model produced a correct, verifiable proof for a third of genuine research problems" — a milestone, not saturation. However, note that access is partially negotiated with OpenAI, which created a structural critique (some problems were shared under NDA with the lab being evaluated). Weight accordingly.

---

### 3. ARC-AGI-2

Category: reasoning-math (abstract visual reasoning)
Released: 2025-03 (successor to ARC-AGI-1 which was declared saturated by o3 in Dec 2024)
Size: ~120 public eval tasks + private holdout
Source: https://arcprize.org/arc-agi/2

Why it matters in 2026: Grid-based visual puzzles requiring zero-shot rule induction from 2–5 input-output examples. Each task tests a different transformation rule, none of which are in training data. Designed explicitly to resist memorization and require compositional reasoning. ARC-AGI-1 fell to o3 (88% with heavy test-time compute); v2 was tuned harder.

Current SOTA (as of Apr 2026):
- GPT-5.4 Pro: 83.3% verified (with test-time compute)
- Gemini 3.1 Pro: ~77.1% (with refinement)
- GPT-5.2 with heavy TTC: 54% ($31/task in compute)
- Raw Kaggle open SOTA (compute-capped): ~24%
- Humans: ~100%

Example tasks:
- Given 3 input-output grid pairs where colored shapes translate by their own bounding-box width, apply the same rule to a 4th input

How to read the score: Pay attention to *compute cost per task*. ARC prize specifically measures cost-constrained performance. A model scoring 80%+ at $30/task is expensive genius; the same model at $0.10/task is the real signal. Also: the Kaggle private set is the gold standard — lab-reported numbers on the public set have been contested.

---

### 4. AIME 2025 / 2026 — American Invitational Mathematics Examination

Category: reasoning-math
Released: AIME I & II each year; tracked live by matharena.ai
Size: 15 integer-answer problems per exam (answers 000–999), 2 exams per year
Source: https://matharena.ai/

Why it matters in 2026: Single most-cited reasoning benchmark in 2025 model releases; competition-math canary for whether a model "can reason." The integer-answer format makes grading trivial and prevents partial-credit gaming. Matharena runs models on each exam *immediately after* human administration (before solutions leak), which preserves signal.

Current SOTA (as of Apr 2026):
- AIME 2025: Kimi K2.5 Reasoning 96.1%, GLM-4.7 95.7%
- AIME 2026: GLM-5.1 95.3%
- Non-reasoning baselines in 2023: ~10–20%

Example tasks:
- "Find the number of positive integers n ≤ 1000 such that [property involving modular arithmetic and divisors]"
- Standard format: 15 problems, 3 hours, no calculator

How to read the score: Approaching saturation — scores above 95% mean the benchmark now discriminates at the 1-problem granularity (14/15 vs 15/15). Also, 2025 contest problems have been online for ~1 year by April 2026, so pretraining contamination is plausible for Apr-2026 releases. The freshly administered AIME 2026 exams are the only uncontaminated signal; AIME 2025 should be considered lightly leaked.

---

### 5. SWE-bench Verified + SWE-bench Pro

Category: code (software engineering)
Released: Verified 2024-08 (OpenAI-curated subset of original SWE-bench); Pro 2025-09
Size: Verified: 500 real GitHub issues (Django-heavy); Pro: 1,865 tasks across Python/Go/TS/JS from GPL-licensed + private-partner repos
Source: https://www.swebench.com/verified.html, https://labs.scale.com/leaderboard/swe_bench_pro_public, https://www.morphllm.com/swe-bench-pro, https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/

Why it matters in 2026: Verified became the default SWE benchmark 2024–2025, driving the agentic-coding era (Cursor, Devin, Claude Code all tuned against it). By early 2026 it saturated and OpenAI publicly deprecated it. SWE-bench Pro is the successor — designed contamination-resistant via (a) private-startup repos and (b) GPL copyleft to actively discourage inclusion in pretraining corpora. Pro also expands beyond Python.

Current SOTA (as of Apr 2026):

SWE-bench Verified:
- Claude Opus 4.6: 80.8%
- Gemini 3.1 Pro: 80.6%
- GPT-5.2: 80.0%

SWE-bench Pro (public subset):
- Claude Mythos: 77.8%
- GPT-5.3-Codex: 56.8%

SWE-bench Pro (private held-out): GPT-5 drops from 23% → 15% vs public, indicating meaningful contamination delta.

Example tasks:
- Verified: `django__django-13279` — fix session decoding when `DEFAULT_HASHING_ALGORITHM='sha1'`. Model gets repo, issue description, test suite; must produce patch that passes hidden test.

How to read the score: Verified scores above 80% should be treated as saturated; use Pro. The ~8% gap between Verified and Pro-public on the same model is roughly the "contamination + Django-specificity" premium. The further ~8% gap between Pro-public and Pro-private is what you should mentally subtract from any public Pro score to estimate real capability.

---

### 6. LiveCodeBench

Category: code (competitive programming)
Released: 2024-03, rolling monthly ingest
Source: https://livecodebench.github.io/leaderboard.html

Size: ~1,000+ problems cumulative, filtered by release date

Why it matters in 2026: Contamination-resistant by construction — problems ingested monthly from LeetCode, AtCoder, Codeforces, and scored *only* on problems released *after* the model's training cutoff. Keeps the benchmark live indefinitely. The most honest single number for "can this model code."

Current SOTA (as of Apr 2026, v6 with post-cutoff subset):
- Gemini 3 Pro Preview: 91.7%
- DeepSeek V3.2: 89.6%
- Top frontier cluster: ~88–92%

Example tasks:
- Fresh AtCoder Grand Contest problem: given graph constraints, output a valid coloring or report impossibility. Must handle large N efficiently (algorithmic, not boilerplate).

How to read the score: Always check *which* date-filtered subset — full LiveCodeBench includes old problems that are contaminated. The "v6" or "post-cutoff" filter is the real signal. High scores still don't mean the model can do multi-file engineering (that's SWE-bench Pro); LiveCodeBench tests single-function algorithmic coding under contest time pressure.

---

### 7. Aider Polyglot

Category: code (multi-language editing)
Released: 2024-12
Size: 225 hard Exercism exercises across 6 languages
Source: https://aider.chat/docs/leaderboards/

Why it matters in 2026: Only widely-used benchmark that tests *code editing* (not generation from scratch) across C++, Go, Java, JavaScript, Python, Rust. Uses Aider's whole-file and diff edit formats — directly measures the capability that actual coding agents rely on. Graded by the Exercism test suite, so unambiguous pass/fail.

Current SOTA (as of Apr 2026):
- GPT-5 high: 88.0%
- Prior frontier cluster: 70–85%

Example tasks:
- Given a partially-filled Rust module and a failing test, produce a diff that makes the test pass. Tests span standard algorithms (tree traversal, string parsing, numeric methods).

How to read the score: This is more narrow than SWE-bench — tasks are bounded (one file, clear spec) so high scores don't prove agentic coding. But it's the canonical test for "does the model reliably emit valid diffs across languages?" Low Aider score + high LiveCodeBench score = model can write code but struggles with edit formats. Important distinction.

---

### 8. τ²-Bench (Tau-Squared Bench)

Category: agents / tool-use
Released: 2025-06 (successor to τ-Bench from Sierra, 2024)
Size: 3 domains (airline, retail, telecom), ~200+ tasks total
Source: https://sierra.ai/resources/research/tau-squared-bench, https://github.com/sierra-research/tau2-bench, https://taubench.com/blog/tau3-task-fixes.html

Why it matters in 2026: "Dual-control" benchmark — both the agent and the simulated user can modify shared database/account state. This reflects real customer-service or OS-agent scenarios better than tool-call benchmarks where only the agent acts. Measures policy compliance, API correctness, and multi-turn coherence jointly. Included in AA Intelligence Index v4.0.

Current SOTA (as of Apr 2026):
- GPT-5.4 on τ²-Telecom with reasoning: 98.9% (approaching saturation on this slice)
- Kimi K2 tops τ²-Telecom leaderboard overall
- pass^8 (all 8 parallel trajectories succeed) on τ-retail: <25% even for frontier

Example tasks:
- Airline: user requests destination change on a basic-economy fare. Policy forbids direct changes. Agent must detect this, decline via policy reference, propose cancel+rebook flow, and execute the correct sequence of API calls with proper authorization.

How to read the score: Pay attention to *which slice* — Telecom is saturating while Retail pass^8 is still brutal. The pass^k metric (all k independent runs must succeed) is more informative than pass@1 for agent reliability, because real deployments care about worst-case not best-case. A 98% pass@1 with 25% pass^8 means one in four runs still fails — not production-ready.

---

### 9. Terminal-Bench 2.0

Category: agents / tool-use
Released: 2025-11 (v2.0)
Size: 89 Dockerized real-world shell tasks
Source: https://www.tbench.ai/

Why it matters in 2026: The most realistic "can the model drive a terminal" benchmark. Tasks include compiling code, training small models, configuring servers, debugging failures, implementing crypto, running coverage tools. Each task runs in an isolated Docker container with a hidden grading script. Included as "Terminal-Bench Hard" in AA Intelligence Index v4.0.

Current SOTA (as of Apr 2026):
- Claude Mythos: 82.0%
- GPT-5.3 Codex: 77.3%
- GPT-5.4: 75.1%
- Average frontier: ~55%

Example tasks:
- Train a fastText classifier on Yelp reviews hitting both a specified accuracy threshold and a model-size constraint
- Remove leaked API keys from a git history without breaking the build
- Run `gcov` coverage on a C project and report uncovered lines
- Resolve a merge conflict in a non-trivial Python repo

How to read the score: Unlike SWE-bench, there is no single "fix this file" abstraction — the agent must decide on its own which tools to invoke (compiler, pip, docker, git). This exposes planning failures that file-edit benchmarks hide. A model at 75%+ on Terminal-Bench is plausibly deployable as an autonomous devops agent; below 50% it needs close supervision.

---

### 10. METR Time Horizons (HCAST + RE-Bench)

Category: agents / long-horizon
Released: TH1.1 in 2025-09; ongoing expansion
Size: 228 tasks (HCAST + RE-Bench); 31 of them take humans 8+ hours
Source: https://metr.org/time-horizons/

Why it matters in 2026: This is the *only* benchmark that measures agent capability in wall-clock task length rather than pass-rate. Metric = the length of task (measured by median human completion time) at which the model achieves 50% success rate. Across 2024–2025 this horizon doubled every ~4 months — the single most cited empirical curve for agent progress. As of Apr 2026, frontier models sit in the 1–4 hour range; publicly reported best is tens of minutes.

Current SOTA (as of Apr 2026):
- Frontier 50%-horizon: 1–4 hours (private lab numbers)
- Public best: tens of minutes
- Gemini 3 / Claude Opus 4.6 / GPT-5.4 cluster near 1-hour mark publicly

Example tasks:
- RE-Bench: given a training script and a compute budget, improve evaluation loss by a specified margin within the budget (real ML research task)
- HCAST: cybersecurity CTF challenges, SWE tickets from real companies, ML experiments — all with measured human completion times

How to read the score: The metric itself is unusual — longer horizon = better. "50% success at 2 hours" means on tasks that take a skilled human 2 hours, the model succeeds half the time. Don't conflate with "model can do 2 hours of useful work in one pass"; it's survival-analysis-style curve. Also: METR does not release the full task set (some held private), and scores depend heavily on scaffolding — the same model can shift 2x with better agent harness.

---

### 11. RULER

Category: long-context
Released: 2024-04 (NVIDIA); actively expanded
Size: 13 synthetic task types × configurable context lengths (4k → 128k → 1M)
Source: https://github.com/NVIDIA/RULER

Why it matters in 2026: Long-context evaluation beyond "needle-in-a-haystack" (NIAH), which is now trivial. RULER adds variable-tracking, common-word extraction, multi-hop QA, and aggregated retrieval tasks. Reveals that most models with "128k" or "1M" advertised context windows catastrophically degrade on non-NIAH tasks well before the advertised limit. Diagnostic tool, not a single-number leaderboard.

Current SOTA notes (as of Apr 2026): All frontier models still break at long lengths on at least some of the 13 tasks. The bigger the advertised context, the wider the gap between NIAH and RULER-hard. Example: a model advertising 1M context may retain 90%+ NIAH at 1M but drop to 40% on variable-tracking at 128k.

Example tasks:
- NIAH: retrieve a single planted sentence from N tokens
- variable_tracking: track a chain `x = 3; y = x + 1; z = y * 2; ...` over many hops, report final value
- common_words_extraction: list the k most common words in a long passage
- HotpotQA / SQuAD embedded in long distractor contexts

How to read the score: Use RULER when deciding whether an advertised context window is real. Always break out per-task-type scores; the aggregate hides the failures. "Effective context length" (where the model passes all 13 tasks at 80%+) is typically 4–16x smaller than advertised.

---

### 12. MMMU-Pro

Category: multimodal
Released: 2024-09 (Pro version of MMMU from 2023)
Size: ~1,700 questions across 6 disciplines: art, business, science, medicine, humanities, engineering
Source: https://mmmu-benchmark.github.io/

Why it matters in 2026: Original MMMU is saturating (~81%, GPT-5.4 and Gemini 3 Pro tied). MMMU-Pro filters to questions requiring *genuine* visual understanding (removes text-only-solvable Qs), adds OCR-adversarial variants, and expands to 10 answer choices. Pushes frontier multimodal scores back down to the 30–50% range where there is still discriminative signal. The canonical multimodal reasoning benchmark.

Current SOTA notes: Base MMMU saturating at ~81%; MMMU-Pro keeps frontier models in the 30–50% range as of Apr 2026.

Example tasks:
- Economics: supply/demand diagram with labeled curves; compute equilibrium price after shift, given calculation-required parameters in the figure
- Chemistry: multi-step reaction mechanism diagram; identify the product after a 3-step synthesis given intermediate conditions

How to read the score: MMMU-Pro scores are not comparable to MMMU — the 10-choice format and vision-required filtering roughly halves naive scores. Also check whether the evaluation used the "vision-only" split (where the question text is embedded in the image, forcing true multimodal reasoning) vs text+image split.

---

### 13. IFBench

Category: instruction-following
Released: 2025-07
Size: 58 novel out-of-domain verifiable constraint types
Source: https://github.com/allenai/IFBench

Why it matters in 2026: Instruction-following used to mean IFEval (old, saturated). IFBench introduces 58 *novel* constraint types held out from training distributions — specifically designed to catch models that memorized IFEval's constraint list. All constraints are programmatically verifiable (exact sentence count, exact word presence, format rules) so grading is deterministic. Included in AA Intelligence Index v4.0.

Current SOTA (as of Apr 2026):
- Qwen3.6 Plus: 75.8%
- Claude Opus 4.5: 58%

Example tasks:
- "Respond in exactly 7 sentences, each starting with a letter that spells HAMLETS, and include the word 'octahedron' exactly twice."

How to read the score: Surprisingly, instruction-following has not saturated — even frontier models sit in the 55–75% range. This is a useful signal because it's orthogonal to reasoning ability: a model can solve FrontierMath tier-4 and still fail to count to 7 sentences. If you're shipping a product where format compliance matters (structured outputs, tool-call JSON, length limits), trust IFBench over general capability scores.

---

## 3. Historical / saturated benchmarks

These dominated 2020–2024 but no longer provide frontier signal. Listed for context only — do not cite them in 2026 model comparisons.

```
│ Benchmark    │ Peak year │ Current SOTA │ Status            │ Replaced by           │
├──────────────┼───────────┼──────────────┼───────────────────┼───────────────────────┤
│ MMLU         │ 2020      │ >90%         │ Saturated         │ MMLU-Pro → HLE        │
│ HellaSwag    │ 2019      │ >95%         │ Saturated (2023)  │ None (commonsense)    │
│ HumanEval    │ 2021      │ >95%         │ Saturated (2024)  │ LiveCodeBench, SWE-P  │
│ GSM8K        │ 2021      │ ~95%+        │ Saturated         │ AIME, FrontierMath    │
│ GPQA Diamond │ 2023      │ 92–94%       │ Saturating        │ HLE                   │
│ MATH         │ 2021      │ >95%         │ Saturated by RL   │ AIME, FrontierMath    │
│ ARC-easy     │ 2018      │ >95%         │ Long saturated    │ ARC-AGI-2             │
```

Notes per row:
- MMLU: 57-subject multiple-choice. Defined "general knowledge" for three years; MMLU-Pro (10 choices, harder) slowed saturation briefly, HLE replaced it fully.
- HellaSwag: sentence-completion commonsense. Dead since GPT-4 era; no modern replacement because commonsense itself is no longer a frontier bottleneck.
- HumanEval: 164 Python functions from Codex paper. Most-cited code benchmark 2021–2024; killed by its tiny size and public test exposure.
- GSM8K: grade-school word problems. Useful for small-model triage only.
- GPQA Diamond: 198 expert-authored PhD questions. Currently sits at 92–94% across frontier (Claude Mythos 94.6%, Gemini 3 Deep Think 93.8%, GPT-5.4 92.8%). Still sometimes cited, but effectively saturating; HLE is the successor.
- MATH: 12,500 competition problems. Saturated after o1-class reasoning models made long CoT reliable.
- ARC-easy: Allen AI's original ARC (not the visual ARC-AGI). Distinct from Chollet's ARC; both now resolved at different levels.

---

## 4. How to read leaderboards critically

Not all leaderboards are equally trustworthy. The structural incentives and contamination surfaces differ.

Chatbot Arena / LM Arena. Pairwise human preference voting, Elo-based ranking. For two years (2023–2024) this was treated as the ground-truth "which model do users prefer" benchmark. The April 2025 "Leaderboard Illusion" paper (Cohere + Stanford + MIT + AI2) documented serious problems: Meta privately tested 27 Llama 4 variants and published only the best-performing one, effectively doing selection bias at the leaderboard level. Several other labs were shown to do similar variant-shopping. Separately, LM Arena now runs a paid "AI Evaluations" product (reportedly ~$30M ARR) which creates a structural conflict: the organization grading frontier models also sells evaluation services to the same labs. Useful as a *vibes* signal for consumer-chat preference, not as a capability measurement. Sources: https://techcrunch.com/2025/04/30/study-accuses-lm-arena-of-helping-top-ai-labs-game-its-benchmark/, https://simonwillison.net/2025/Apr/30/criticism-of-the-chatbot-arena/

Artificial Analysis Intelligence Index v4.0. Composite score over ~10 component benchmarks: GDPval-AA, τ²-Bench Telecom, Terminal-Bench Hard, SciCode, AA-LCR (long-context reasoning), AA-Omniscience, IFBench, HLE, GPQA Diamond, CritPt. Equal-weighted across four super-categories: Agents, Coding, Scientific Reasoning, General. Runs pass@1 on dedicated hardware with published ±1% confidence intervals. The most transparent composite available. Use as the single headline number when you need one. Source: https://artificialanalysis.ai/methodology/intelligence-benchmarking

Scale SEAL. Private held-out sets across reasoning, coding, tool-use, and multilingual. Scale runs evaluations internally and publishes only aggregate scores — no test items leak. The most defensible leaderboard for frontier comparison specifically because of the private-set design. Used as tie-breaker when AA Index and Arena disagree. Source: https://scale.com/leaderboard

LM Council. Cross-validates results across multiple evaluation setups. Useful as a consistency check — if a model scores 80% on benchmark X via AA, 82% via Scale, and 45% via LM Council, the discrepancy is itself informative (often reveals scaffolding differences or prompt sensitivity). Source: https://lmcouncil.ai/benchmarks

Practical heuristic. For a production decision: start with AA Intelligence Index v4.0 for the headline, cross-check with Scale SEAL, ignore Arena except for consumer-preference signals, and always look at the *per-benchmark breakdown* not the composite — a model can top the composite while being terrible at the one dimension you care about.

---

## 5. Contamination & gaming

Three distinct failure modes. They compound.

1. Training-data leakage. Public benchmark text, test questions, and answers end up in web-scraped pretraining corpora. A model scoring 95% on HumanEval in 2025 might be retrieving, not reasoning. Mitigations: rolling/live benchmarks (LiveCodeBench, AIME as it's released, Matharena), private held-out sets (Scale SEAL, FrontierMath, SWE-bench Pro private subset), GPL copyleft deterrence (SWE-bench Pro).

2. Scoring-loophole exploits. The Berkeley RDI audit "How We Broke Top AI Agent Benchmarks" (Jan 2026) demonstrated that minimal prompt engineering, without the model solving any real tasks, could hit 98% on GAIA and 73% on OSWorld. The attacks exploited (a) public validation-set answers being reachable from the task environment, and (b) loose substring-match grading rules that accepted degenerate outputs. This is not a training-time issue — it's a benchmark-design issue. Mitigations: behavior-based evaluation (did the agent actually use the API in the right sequence?) rather than string-matching on the final answer; grading scripts that run in isolated environments with no access to validation data. Source: https://rdi.berkeley.edu/blog/trustworthy-benchmarks-cont/

3. Leaderboard shopping. Labs submit many private variants and publish only the best (LM Arena / Meta Llama 4 pattern). Statistically this is p-hacking at the leaderboard level. Mitigations: leaderboards should limit submissions per lab per month, or require pre-registration of which checkpoint will be submitted. Few currently do this.

Eval awareness. Anthropic's published analysis on BrowseComp noted that frontier models have begun showing measurable "eval awareness" — inferring from task structure that they are being benchmarked, and behaving differently than in production. This is a newer failure mode than the three above and hardest to mitigate. The recommended defenses are: keep benchmarks visually and structurally similar to production traffic; don't use signature phrases ("This is a test of your capabilities..."); run shadow evals on real production traces where possible. Source: https://www.anthropic.com/engineering/eval-awareness-browsecomp

Practical takeaway. If a model's public score seems too good, assume one of these three is operative until proven otherwise. The gap between SWE-bench Pro public (57%) and private (42%) on GPT-5.3-Codex is a concrete calibration anchor — roughly 15 percentage points of the public number is "leakage advantage."

---

## 6. One-page cheat sheet

```
│ Category              │ The one to learn       │ Why                                                     │
├───────────────────────┼────────────────────────┼─────────────────────────────────────────────────────────┤
│ Reasoning / Math      │ HLE                    │ Successor to MMLU/GPQA, still live, PhD-tail            │
│ Research Math         │ FrontierMath           │ Private, Fields-medalist-curated, genuinely unsaturated │
│ Abstract Reasoning    │ ARC-AGI-2              │ Zero-shot rule induction; measures compute efficiency   │
│ Competition Math      │ AIME 2026              │ Fresh contests, integer-answer, live canary             │
│ Software Engineering  │ SWE-bench Pro          │ Private repos + GPL, contamination-resistant            │
│ Algorithmic Coding    │ LiveCodeBench          │ Rolling monthly, filtered by training cutoff            │
│ Multi-Language Edit   │ Aider Polyglot         │ Only edit-format benchmark across 6 languages           │
│ Agents (simulation)   │ τ²-Bench               │ Dual-control state, pass^k metric                       │
│ Agents (shell)        │ Terminal-Bench 2.0     │ Real Dockerized devops tasks                            │
│ Long-horizon          │ METR Time Horizons     │ Wall-clock metric; 4-month-doubling curve               │
│ Long-context          │ RULER                  │ Breaks "1M context" marketing claims                    │
│ Multimodal            │ MMMU-Pro               │ Base MMMU is saturated; Pro restores signal             │
│ Instruction-following │ IFBench                │ 58 novel constraints; orthogonal to capability          │
```
