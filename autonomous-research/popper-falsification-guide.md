# Popper: Automated Hypothesis Validation with Agentic Sequential Falsifications

How an LLM-agent system turns Karl Popper's "you can only refute, never prove" into a running algorithm — proposing falsification experiments against a free-form hypothesis, executing them as code over real data, and aggregating the evidence with a *sequential statistical test* that holds the false-discovery rate below a knob you set (α), no matter how many experiments it runs or when it decides to stop. Where Robin and AI-Scientist *generate* hypotheses, Popper is the missing other half: the rigorous *validation* engine, and it works in biology, economics, and sociology alike.

## Background

**Primary paper**: [Automated Hypothesis Validation with Agentic Sequential Falsifications](https://arxiv.org/abs/2502.09858) (Kexin Huang, Ying Jin, Ryan Li, Michael Y. Li, Emmanuel Candès, Jure Leskovec — Stanford CS / Statistics / Data Science Initiative / Harvard, Feb 2025). Code: [github.com/snap-stanford/POPPER](https://github.com/snap-stanford/POPPER).

Popper sits at the intersection of two lineages — one philosophical, one statistical — and the whole contribution is making them meet inside an agent loop.

**Lineage 1 — Popperian falsification (the philosophy of science).**
- **Karl Popper**, *The Logic of Scientific Discovery* (1959/2005). The founding claim: scientific theories can never be *verified* by evidence, only *falsified*. A theory earns credibility by surviving sincere attempts to refute it. Popper's machine implements this literally — it never tries to prove a hypothesis directly; it repeatedly tries to break the hypothesis's logical consequences and reports the hypothesis as "validated" only when those refutation attempts keep failing.
- **Thomas Kuhn**, *The Structure of Scientific Revolutions* (1962); **Imre Lakatos**, *The Methodology of Scientific Research Programmes* (1978). Kuhn stressed the sociotechnical embeddedness of science; Lakatos's "research programmes" evaluate hypotheses inside a web of *auxiliary assumptions*. The paper explicitly maps Lakatos's auxiliary assumptions onto its own "dataset relevance is a prerequisite for a valid test" check (the relevance checker, below). Also cited: van Fraassen's constructive empiricism (empirical adequacy over ontological truth — mirrors Popper's focus on *observable implications* rather than the abstract claim), and Goodman's "grue" paradox on the risk of inductive generalization.
- The reading: Popper-the-system is a 60-year-old epistemology compiled into Python. The novelty is not the philosophy, it is making falsification *automatic, scalable, and statistically honest*.

**Lineage 2 — e-values, safe testing, and sequential inference (the statistics).** This is the genuinely new machinery and the part to slow down on.
- **Neyman–Pearson (1928, 1933)** and **Fisher (1936)** — classical Type-I error control. The fixed-sample frequentist baseline Popper must respect.
- **Robbins / Ville → martingale tests** — the idea that you can test continuously and stop whenever, if your evidence statistic is a non-negative (super-)martingale.
- **Shafer**, *The language of betting* ([arXiv 1903.06991](https://arxiv.org/abs/1903.06991), 2019) and **Grünwald, de Heide & Koolen**, *Safe testing* ([arXiv 2020 ITA](https://arxiv.org/abs/1906.07801), 2020) — recast hypothesis testing as betting against the null; the wealth process is an *e-process*. This is where "any-time validity" and "optional stopping is allowed" come from.
- **Vovk & Wang**, *E-values: calibration, combination and applications* (*Annals of Statistics* 49(3), 2021) — the definitional paper for **e-values** and the **p-to-e calibrator** Popper uses verbatim. Read this one if you read only one statistics reference.
- **Wang & Ramdas**, *False discovery rate control with e-values* (*JRSS-B* 84(3), 2022) — the e-BH procedure; Popper points to it as the route to FDR control over many hypotheses (future work).

**Contemporary "AI scientist" systems Popper is positioned against (and complements):** *The AI Scientist* (Lu et al., Sakana, [2408.06292](https://arxiv.org/abs/2408.06292)) — end-to-end automation in computational ML; *AI Co-Scientist* (Gottweis et al., Google, [2502.18864](https://arxiv.org/abs/2502.18864)) — multi-agent hypothesis generation/debate; *Robin* (Ghareeb et al., FutureHouse, [2505.13400](https://arxiv.org/abs/2505.13400)) — closed-loop wet-lab biology discovery; *DiscoveryBench* (Majumder et al., [2407.01725](https://arxiv.org/abs/2407.01725)) and *TargetVal*-style benchmarks supply the test hypotheses. Every one of these is generation-heavy. Popper is the only one whose deliverable is a *statistically valid verdict* on a hypothesis it did not have to generate.

## What Problem Does Popper Solve?

The recent explosion of LLM-generated hypotheses created a validation bottleneck. LLMs produce plausible-sounding hypotheses *in volumes that make manual checking impossible*, and their plausibility is uncorrelated with truth (hallucination). The hypotheses that matter are also usually **abstract**: "Gene VAV1 regulates IL-2 production in immune tissue," "a marketing intervention increases retention," "a social policy reduces recidivism." You cannot test those statements *as stated* — there is no single dataset whose column is literally "is this hypothesis true."

So you do what a scientist does: derive **measurable implications** (concrete consequences that must hold *if* the hypothesis is true), test those, and accumulate evidence. Two failure modes lurk:

1. **The generation systems stop short of rigor.** Robin, AI-Scientist, and AI Co-Scientist are brilliant at proposing and even running experiments, but their "this looks supported" judgment is an LLM vibe, not a controlled statistical decision. Nothing stops them from declaring victory on noise.
2. **Naively testing many implications inflates false positives.** Run 20 tests, take the best p-value, and you will "validate" a false hypothesis far more than α of the time. This is the multiple-testing / optional-stopping trap, and an agent that decides *adaptively* when to stop testing makes it worse, not better.

Popper targets exactly this gap: **rigorous, scalable validation of free-form natural-language hypotheses, with a provable cap on the rate of false validations**, while still letting the agent gather evidence adaptively and stop early. The headline framing for a non-bio reader: this is a *domain-general validation layer* you can bolt onto any hypothesis generator.

```
 GENERATION  (Robin / AI-Scientist / AI Co-Scientist / a human / any LLM)
        │  emits free-form hypotheses, fast, in bulk, plausibility ≠ truth
        ▼
 ┌──────────────────────────────────────────────────────────────┐
 │  POPPER  — the validation layer this paper contributes        │
 │  derive measurable implications → falsify them → aggregate     │
 │  evidence under a SEQUENTIAL test → output ŷ ∈ {0,1} with      │
 │  P(false validation) ≤ α                                       │
 └──────────────────────────────────────────────────────────────┘
        ▼
 a statistically defensible "validated / not validated" verdict
```

## Key Terms

**Hypothesis H (free-form)**: a natural-language statement defining a relationship `r` between variables `V` under context `c`. Example: H = "Gene VAV1 regulates IL-2 production in immune tissue" → V = {VAV1, IL-2 production}, r = "regulate", c = "immune tissue." Too abstract to test directly.

**Null / alternative (main level)**: H is paired with a **main null H₀** — a whole *family* P₀ of data-generating distributions consistent with "the relationship does not hold" (e.g., "VAV1 does *not* regulate IL-2"). Validating H means *rejecting* H₀ in favor of the alternative. (Same null-vs-alternative logic as any test you know; the twist is H₀ is a family, and the data could be anything an agent can fetch or collect.)

**Measurable implication / sub-hypothesis**: a concrete, testable consequence that must hold *if* H is true. If VAV1 truly regulates IL-2, then VAV1 should be *preferentially expressed in immune tissue vs. unrelated tissue*. Each implication becomes a **falsification experiment** with its own null sub-hypothesis hᵢ⁰ ("no difference in expression") and alternative hᵢ¹. You then run a standard statistical test (t-test, Mann–Whitney U, permutation, χ²…) and get a p-value pᵢ.

**The implication assumption (Assumption 1, the load-bearing one)**: *if the main null H₀ is true, then every null sub-hypothesis hᵢ⁰ is also true.* This is what lets refuting sub-hypotheses count as evidence against H₀ with honest error control. If you test an implication that does *not* actually follow from H, you can "falsify" it by luck and wrongly validate H — which is precisely why Popper needs a relevance checker.

**p-value pᵢ**: as you know — under hᵢ⁰, P(pᵢ ≤ t) ≤ t (super-uniform). The output of one falsification experiment.

**e-value eᵢ**: a non-negative random variable with **E[eᵢ] ≤ 1 under the null**. Read it as the *payoff of a bet against the null*: a fair bet has expected value ≤ 1 if the null holds, so a large realized e-value (say 30) is strong evidence the null is false. E-values are the currency Popper aggregates because — unlike p-values — they **multiply** and survive **optional stopping**. (Mental model: an e-value is "how many times your money you'd have made betting against H₀," and you can only consistently multiply your money if H₀ is really false.)

**p-to-e calibrator**: a function turning a valid p-value into a valid e-value. Popper uses the Vovk–Wang family `eᵢ = κ · pᵢ^(κ−1)`, `κ ∈ (0,1)`, with **κ = 0.5**, i.e. **eᵢ = 0.5 / √pᵢ**. Check the property: under a uniform null, E[0.5·p^(−0.5)] = ∫₀¹ 0.5·p^(−0.5) dp = [p^0.5]₀¹ = 1. So it's a legitimate e-value. A tiny p (strong refutation) → big e; p = 1 (no evidence) → e = 0.5 < 1 (you *lose* a bit of betting capital, as you should when the implication held up).

**e-process / super-martingale**: the *sequence* of aggregated evidence {Eᵢ}. Popper aggregates by **multiplying**: `Eᵢ = ∏ₛ₌₁ⁱ eₛ`. Under H₀ this product is a non-negative **super-martingale** (E[Eᵢ | past] ≤ Eᵢ₋₁) — an *e-process*. That single property delivers everything: it stays small in expectation under the null *at every step simultaneously*, so you can peek, add experiments, and stop adaptively without breaking the guarantee.

**Sequential test / any-time validity / optional stopping**: a test you may evaluate after every new experiment and stop the instant the evidence is conclusive, with the Type-I guarantee holding *uniformly over all stopping rules*. Classical fixed-n tests do *not* allow this; sequential e-process tests do. This is the core enabling property for an *adaptive* agent.

**Type-I error**: P(validate H | H is actually false) = P(ŷ = 1) under H₀, taken over both the data randomness *and* the agent's randomness. Popper's promise: ≤ α (default α = 0.1).

**FWER / FDR**: when validating *many* hypotheses, family-wise error rate (≥1 false validation) is controlled by running Popper at level α/M (Bonferroni); false discovery rate (expected fraction of validated hypotheses that are false) by the e-BH procedure (Wang & Ramdas 2022), enabled for free because Popper already outputs a valid e-value per hypothesis. Both are flagged as natural extensions, not the paper's primary contribution.

## The Core Algorithm

Popper is a loop of three LLM agents wrapped around a sequential test. Given a hypothesis H and a Type-I budget α, each round proposes one falsification experiment, screens it for relevance, executes it to a p-value, converts that to an e-value, multiplies it into the running evidence, and checks whether the product has crossed 1/α.

```
 INPUT: hypothesis H, Type-I rate α              ┌─ running evidence  E = ∏ eᵢ ─┐
 ┌─────────┐                                      │                             │
 │   H     │   ┌──────────────┐  relevant?  ┌─────┴────────┐   pᵢ   ┌───────────┴──────────┐
 │   α     │──>│ DESIGN AGENT │────────────>│  EXECUTION   │──────> │  SEQUENTIAL ERROR    │
 └─────────┘   │ propose hᵢ⁰, │   R(h)≥r₀   │  AGENT       │        │  CONTROL             │
     ▲         │ hᵢ¹, a test  │             │ ReAct: think-│        │  eᵢ = 0.5/√pᵢ        │
     │         │ self-refine  │   no ↓ drop │ act-observe  │        │  E ← E · eᵢ          │
     │         └──────────────┘   & re-ask  │ in a Python  │        │                      │
     │                                       │ sandbox →pᵢ  │        │  E ≥ 1/α ? ──yes──> VALIDATE H (ŷ=1)
     │                                       └──────────────┘        │     │ no               (reject H₀)
     └──────────── insufficient evidence: next round i+1 ◄───────────┴─────┘
                   (until E≥1/α, or max tests Nᵗᵉˢᵗ reached → not validated)
```

| Agent | Role | How |
|-------|------|-----|
| **Experiment Design Agent** (A_design) | Propose the next falsification experiment: a sub-hypothesis (measurable implication), its null hᵢ⁰ and alternative hᵢ¹, and which data/test to use | LLM with domain knowledge; **Self-Refine** (Madaan et al.) critiques its own proposal on three axes — novelty (non-redundant with prior rounds), feasibility (answerable from available data), and logical relevance (H really implies hᵢ). Sees only dataset **schema/metadata**, never raw values — this is what keeps the e-values sequentially valid |
| **Relevance Checker** (A_rel) | Gatekeeper enforcing Assumption 1 | LLM-as-judge scoring R(h) ∈ [0,1] on a 0.1–1.0 rubric; if R(h) < r₀ (threshold), discard the experiment and ask the design agent for another. Prevents "falsify an irrelevant implication → inflate Type-I" |
| **Execution Agent** (A_exec) | Run the experiment, output a valid p-value pᵢ | **ReAct** (think→act→observe) inside a Jupyter/Python sandbox with pandas, statsmodels, scipy; inspects data, preprocesses, picks a test (t / χ² / Mann–Whitney U / permutation…), and reports pᵢ in scientific notation. Up to N_retries on execution failure |
| **Sequential Error Control** | Aggregate evidence, decide stop/continue | Calibrate pᵢ→eᵢ=0.5/√pᵢ; multiply into E=∏eₛ; if E ≥ 1/α reject H₀ (validate H); else continue |

### Pseudocode (Algorithm 1, condensed)

```
input: hypothesis H, dataset/world D, α, relevance threshold τ, max tests N_test
E ← 1                                  # aggregated evidence (e-process), E₀ = 1
F_success, F_failed ← ∅, ∅
for i = 1 … N_test:
    T ← A_design(H, D, F_success, F_failed)        # propose falsification experiment
    if A_rel(T) < τ:                               # relevance gate (Assumption 1)
        F_failed ← F_failed ∪ {T};  continue
    for j = 1 … N_retries:                          # execution with retries
        (ok, p_i) ← A_exec(T, D)
        if ok: break
    if not ok:
        F_failed ← F_failed ∪ {T};  continue
    F_success ← F_success ∪ {T}
    e_i ← 0.5 / sqrt(p_i)                           # p-to-e calibrator, κ = 0.5
    E   ← E · e_i                                   # multiply (super-martingale)
    if E ≥ 1/α:                                     # sufficient evidence
        return VALIDATED (ŷ = 1)                    # i.e. reject H₀
return NOT VALIDATED (ŷ = 0)                        # budget exhausted
```

### Why this controls Type-I error (Theorem 4, in plain steps)

The guarantee `P(ŷ = 1) ≤ α under H₀` rests on three assumptions and a one-line martingale argument:

- **Assumption 1 (Implication)**: H₀ true ⟹ every hᵢ⁰ true. (Enforced by the relevance checker.)
- **Assumption 2 (Sequential validity)**: each eᵢ satisfies E[eᵢ | data so far] ≤ 1 under hᵢ⁰. Popper engineers this by giving the design agent only *schema/metadata* of unused datasets — the choice of the next test cannot peek at the values it will be tested on, so the p-value stays conditionally valid. (For a static database this is automatic; for actively-collected data it holds because the data is gathered *after* the design step.)
- **Assumption 3 (Optional stopping)**: the stopping decision depends only on data seen so far (the running E is measurable) — a valid stopping time.

Then the proof: under H₀, `E[Eᵢ | F_{i−1}] = Eᵢ₋₁ · E[eᵢ | F_{i−1}] ≤ Eᵢ₋₁`, so {Eᵢ} is a **non-negative super-martingale** with E₀ = 1. By **Doob's optional-stopping theorem**, at *any* stopping time τ, E[E_τ] ≤ 1. By **Markov's inequality**, `P(ŷ=1) = P(E_τ ≥ 1/α) ≤ α · E[E_τ] ≤ α`. Done. The agent may run any number of experiments and stop whenever — the cap holds because multiplying e-values can only inflate your "wealth" by chance with probability ≤ α when the null is true.

### Numerical walkthrough of the sequential error control

Concrete run on H = "Gene ZAP70 regulates the production of Interleukin-2," α = 0.1, so the rejection threshold is **1/α = 10**. Calibrator: **e = 0.5/√p**. Aggregation: **E = ∏ e**. (These are the actual cumulative e-values from the paper's Table 1.)

```
Round  Falsification experiment (measurable implication)          p-value     e=0.5/√p   E=∏e
─────────────────────────────────────────────────────────────────────────────────────────────
  1    ZAP70 protein-protein interactions with IL-2 pathway        1.0        0.50       0.50
       components (mass-spec) — no enrichment found
  2    ZAP70 expression correlates with IL-2-pathway genes          8.8e-3     5.33       2.67
       across tissues (eQTL/expression)
  3    Genetic variants affecting ZAP70 also change IL-2-pathway    ~1.0       0.50       1.33   ← evidence
       activity (eQTL) — inconclusive                                                            DROPS back
  4    Rare missense variants in ZAP70 associate with immune        4.7e-4    23.06      30.78  ← E≥10
       phenotypes (GeneBASS) — strong association                                                VALIDATE
─────────────────────────────────────────────────────────────────────────────────────────────
  Decision: at round 4, E = 30.78 ≥ 10 = 1/α  →  reject H₀, hypothesis VALIDATED.
```

What to notice, because it's the whole point of using e-values instead of "take the smallest p-value":
- **Round 1's failed refutation costs you.** The implication held up (p=1.0), so e=0.5 < 1: your evidence *shrinks* to 0.5. A hypothesis that survives a sincere refutation attempt has *not* gained support — Popperian honesty, encoded numerically.
- **Round 3 is inconclusive (p≈1) and drags E from 2.67 down to 1.33.** Evidence is allowed to go *backwards*. There is no cherry-picking: you can't quietly discard the rounds that hurt you, because the product includes them all.
- **No multiplicity correction is bolted on.** With a "min p-value" rule, four tries at α=0.1 would give a false-positive rate far above 10%. Here, four tries are *already* honest: had H₀ been true, the four e-values would average ≤1 each and the product would clear 10 with probability ≤ 0.1, by construction. The agent ran an *adaptive* number of experiments (it would have stopped at any round where E hit 10) and the 10% cap still holds.
- **The threshold is interpretable.** "E ≥ 1/α" means "you turned $1 into $10 betting against the null" — a 10× return is hard to fake when the null is true.

## Results Across Six Domains

Popper is evaluated on two benchmark collections — and the cross-domain breadth is the reason a non-biology reader should care.

- **TargetVal** (deep biology): aggregates **22 tables totaling ~85 million records** from GTEx, the GWAS Catalog (MacArthur 2017), and BioGrid (Oughtred 2019). Hypotheses follow "Gene A regulates Phenotype B"; two sub-tasks **TargetVal-IL2** and **TargetVal-IFNG**; ground truth from genome-wide CRISPR screens (Schmidt 2022). 20 positive hypotheses, 50 constructed negatives (via column permutation).
- **DiscoveryBench** (Majumder 2024): spans **six domains — sociology, biology, humanities, economics, engineering, and meta-science** — yielding **86 non-null hypotheses** grounded in peer-reviewed research. This is where the domain-generality claim lives: same falsification machinery, no biology-specific tooling.

Headline numbers (α = 0.1; mean ± sd over 5 runs; default LLM Claude-Sonnet-3.5):

```
                      Type-I error (≤ 0.1 = controlled)        Power (higher = better)
Method            DiscBench  TargetVal-IL2  TargetVal-IFNG  DiscBench  IL2     IFNG
──────────────────────────────────────────────────────────────────────────────────
ReAct                0.078       0.000         0.000          0.383   0.010   0.020
Self-Refine          0.117       0.100         0.067          0.476   0.183   0.067
CodeGen (o1)         0.248✗      0.013         0.000          —       0.250   0.183
Fisher combined      0.311✗      0.264✗        0.173✗         —       —       —
LLM-Likelihood ratio 0.152✗      0.016         0.180✗         0.428   0.185   —
POPPER (ours)        0.103       0.082         0.085          0.638   0.580   0.591
──────────────────────────────────────────────────────────────────────────────────
```

- **Type-I error is controlled across all three settings** (0.103, 0.082, 0.085 — all within 1 sd of the 0.1 nominal), while most baselines either blow past it (Fisher's combined test 0.26–0.31; CodeGen-o1 0.25) or only "control" it by never rejecting anything (ReAct: Type-I 0.000 *because* power is ~0.01 — useless).
- **Highest power among methods that actually control Type-I.** vs ReAct on DiscoveryBench: **+66.5% power** (0.638 / 0.383). vs Self-Refine on TargetVal-IL2: **3.17× power** (0.580 / 0.183).
- **Ablations confirm each piece earns its place.** Removing the relevance checker (**Popper-NoReleCheck**) inflates Type-I (e.g. 0.340 on IL2, 0.300 on IFNG) — irrelevant tests break Assumption 1. Swapping e-values for Fisher's combined test loses control entirely. Swapping ReAct for plain code-gen (**Popper-CodeGen**) loses power. The whole stack is load-bearing.
- **Sensitivity:** sweeping α from 0.10→0.20 tracks the empirical Type-I error along the diagonal (calibration is honest, not coincidental); raising the max-test budget *raises power while keeping Type-I flat* — more compute → more discoveries, safely.

**vs human experts (9 PhD-level biostatisticians / computational biologists), on TargetVal-IL2:**

```
                Type-I error    Power      Time         Lines of code   # stat tests
Human experts      22.2%        66.7%      1×           1×              1×
POPPER             11.1%        66.7%      9.7× FASTER   3.6× more       2.5× more
```

Statistically indistinguishable accuracy (small n), **~10× faster** (the abstract's "reducing time by 10 folds"), with *better* Type-I control and more thorough testing. Qualitatively, Popper and humans converged on the same families of tests (permutation, Fisher's, eQTL, χ²) and the same falsification experiment types (expression enrichment, interaction networks, variant overlap).

**LLM backbone matters a lot.** o1 gave the best Type-I control (0.091 DiscoveryBench / 0.031 IL2) and best DiscoveryBench power (0.654); Sonnet-3.5 led IL2 power (0.580). Weak models break the guarantee: **Claude-Haiku-3.5 had Type-I 0.230** — it cannot reliably produce valid p-values or honor the implication structure. Rigorous reasoning + coding is a prerequisite, not a nicety.

**Where Popper still fails (error analysis over 128 failed logs, auto-labeled by o1, human-verified):** misinterpreting the context/validity of a p-value **35.9%**, ineffective test selection **28.1%**, falsification test that *breaks the implication* (Assumption-1 violation) **17.2%**, incorrect test implementation **8.6%**, failure to locate relevant data **7.0%**. **Hallucination was 0.8% (1 case), and no p-hacking was observed** — the structure successfully suppresses the failure mode you'd most fear from an LLM doing statistics.

## Limitations (stated by the authors)

- **Type-I control ≠ true discovery.** Rejecting H₀ caps the false-validation *rate*, but a validated hypothesis is not guaranteed true — in the extreme where every input hypothesis is null, *every* validation Popper emits is false, just at rate ≤ α each. Treat "validated" as "survived rigorous falsification," not "proven." (The authors note FDR/FWER extensions via Bonferroni α/M or e-BH for the many-hypotheses regime.)
- **Reasoning about p-value *validity* is the dominant error source (35.9%).** The agent can run a technically-correct test whose p-value doesn't mean what it thinks — e.g. computing an eQTL p-value for "RAB39A expression in neutrophils" and treating it as evidence that "RAB39A regulates IL-2," when the implication doesn't actually hold. Effect-size/p-value semantics remain hard.
- **The relevance checker is good but not perfect.** It agrees with human raters reasonably (Spearman ρ = 0.55, p = 5e-6; it rated 84% of tests "Strongly Relevant" vs. humans' 77%) but slightly *over*-estimates relevance, which is exactly the direction that risks Assumption-1 violations.
- **Needs a strong backbone.** Below a capability threshold (Haiku-3.5) Type-I control simply fails. The guarantee is conditional on the agent producing genuinely valid p-values.
- **Current instantiation queries static databases.** The framework is defined for live data collection and simulation too, but the demonstrated runs draw from fixed corpora; real-world experimental execution (robots, wet-lab) is described in principle, not shown.

## Popper vs. Alternatives

The cleanest way to place Popper: it is the **validation** complement to the **generation** systems. They emit hypotheses; Popper decides — with a Type-I guarantee — whether to believe them.

```
System            Primary job     Free-form    Statistical    Adaptive /      Domains
                                   hyp. input?  error control? optional stop?
──────────────────────────────────────────────────────────────────────────────────────────
POPPER            VALIDATION       ✓            ✓ (e-process,  ✓ (any-time     biology, econ,
                  (falsify+test)                 Type-I ≤ α)    valid)          sociology, +3
Robin             generation +     ✗ (it        ✗ (LLM-judge   ~ (refine on    wet-lab biology
                  wet-lab loop      generates)   + BTL rank)    results)
AI-Scientist      generation +     ✗            ✗              ~               computational ML
                  full automation
AI Co-Scientist   generation /     ✗            ✗ (debate /    ✗               biomedical
                  debate                          evolve)
DiscoveryBench    benchmark /      —            partial         ✗              6 (it supplies
(as a system)     data-driven                    (task metric)                  Popper's hyps)
──────────────────────────────────────────────────────────────────────────────────────────
```

- **Robin / AI-Scientist / AI Co-Scientist rank candidates with an LLM judge** (Robin even fits a Bradley–Terry–Luce model over pairwise preferences — the same BTL you know from RLHF reward modeling and DPO's `P(prefer A) = σ(r_A − r_B)`). That gives a *relative ordering*, not a *calibrated false-positive rate*. Popper's e-process gives the thing BTL ranking cannot: an absolute, frequentist guarantee that you validate a false hypothesis ≤ α of the time. Different statistical object — a betting-wealth e-value, not a preference logit.
- **Natural pipeline:** Robin/AI Co-Scientist *propose and rank* → Popper *adjudicates the top candidates* before anyone spends wet-lab money or makes a policy decision. Robin closes the loop in biology; Popper makes the loop's "is this real?" step honest, and does it in economics and sociology too.
- **When Popper is the wrong tool:** if the hypothesis has *no* testable implication reducible to a data query / statistical test, or if you need novel mechanism *generation* (Popper validates, it does not invent), or if your backbone LLM is weak (the guarantee evaporates).

## Practical Considerations

- **The reusable idea is the p-to-e-to-product pipeline.** Any time an agent runs *multiple adaptive tests and decides when to stop*, classical p-values give you no honest aggregate. Convert each valid p-value with a calibrator (`e = 0.5/√p` is a fine default), multiply, and reject at `1/α`. You get multiplicity *and* optional-stopping protection for free, with no Bonferroni bookkeeping. This transfers to A/B testing, monitoring, and any LLM-judge-style eval where you'd otherwise "keep testing until significant."
- **Guard the implication, not just the test.** The single most important non-obvious component is the relevance checker. Without it (ablation), Type-I error roughly *triples*. If you build something like this, budget real effort on "does H actually imply this sub-test?" — a valid statistical test of an *irrelevant* implication silently destroys the guarantee.
- **Keep the proposer blind to the values it will test on.** Sequential validity (Assumption 2) hinges on the design agent seeing only schema/metadata of unused data. Letting it peek at raw values to choose the next test is a subtle way to invalidate every downstream e-value.
- **Tune two knobs.** α (default 0.1) trades discovery rate against false validations; the max-test budget trades compute against power (more tests → more power, Type-I unaffected). Both behave monotonically and predictably.
- **Use a capable backbone and expect p-value-semantics errors.** Pick an o1-class or Sonnet-class model; below that the guarantee fails. Even with a good model, the dominant residual failure is misreading what a p-value means, so a human spot-check on validated hypotheses is worth the cost.
- **Don't over-read a "validated" verdict.** It means "survived sequential falsification at level α," which is a real, rare, and useful signal — but for acting on many validated hypotheses at once, wrap Popper in FDR control (e-BH) rather than trusting each verdict independently.

## Key Papers

1. Huang, Jin, R. Li, M. Y. Li, Candès, Leskovec. *Automated Hypothesis Validation with Agentic Sequential Falsifications.* arXiv 2502.09858 (2025). https://arxiv.org/abs/2502.09858
2. Vovk & Wang. *E-values: Calibration, combination and applications.* Annals of Statistics 49(3):1736–1754 (2021). — the e-value definition and the p-to-e calibrator Popper uses.
3. Grünwald, de Heide & Koolen. *Safe testing.* IEEE ITA (2020), arXiv 1906.07801. — any-time-valid testing as betting; the e-process / optional-stopping foundation.
4. Shafer. *The language of betting as a strategy for statistical and scientific communication.* arXiv 1903.06991 (2019). — betting interpretation of e-values.
5. Wang & Ramdas. *False discovery rate control with e-values.* JRSS-B 84(3):822–852 (2022). — e-BH; the route from Popper's per-hypothesis e-values to FDR control.
6. Popper. *The Logic of Scientific Discovery.* Hutchinson (1959) / Routledge (2005). — the falsification philosophy the system implements.
7. Madaan et al. *Self-Refine: Iterative Refinement with Self-Feedback.* NeurIPS 36 (2024). — the design agent's self-critique loop.
8. Yao et al. *ReAct: Synergizing Reasoning and Acting in Language Models.* ICLR 2023, arXiv 2210.03629. — the execution agent's think-act-observe loop.
9. Majumder et al. *DiscoveryBench: Towards Data-Driven Discovery with Large Language Models.* arXiv 2407.01725 (2024). — the six-domain hypothesis benchmark.
10. Lu et al. *The AI Scientist.* arXiv 2408.06292 (2024); Gottweis et al. *Towards an AI Co-Scientist.* arXiv 2502.18864 (2025); Ghareeb et al. *Robin.* arXiv 2505.13400 (2025). — the generation-side systems Popper complements.
