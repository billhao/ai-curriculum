# Natural Language Autoencoders (NLA)

Force an LLM's internal activation through a paragraph of English and back, train the round-trip to reconstruct, and the paragraph becomes an unsupervised, faithful explanation of what the model is computing.

## Background

**Research lineage** — NLAs sit at the intersection of three threads:

1. **"Towards Monosemanticity"** (Bricken, Templeton et al., Anthropic, Oct 2023) — Sparse autoencoders (SAEs) decompose activations into sparse linear combinations of learned feature directions. [transformer-circuits.pub](https://transformer-circuits.pub/2023/monosemantic-features)
2. **"Language models can explain neurons in language models"** (Bills, Cammarata et al., OpenAI, May 2023) — Auto-interp pipeline: a second LLM reads top-activating examples and writes a natural-language label per neuron/feature. [openai.com](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html) — the explanation step is decoupled from reconstruction.
3. **"Scaling Monosemanticity"** (Templeton et al., Anthropic, May 2024) — SAEs on Claude 3 Sonnet at 34M features; established that production-scale SAEs + auto-interp work but have coverage gaps and weak label faithfulness. [transformer-circuits.pub](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
4. **"Activation Oracles"** (Anthropic Alignment, 2025) — Train an LLM to answer arbitrary questions about activations using labeled data. Supervised, generalizes OOD. [alignment.anthropic.com](https://alignment.anthropic.com/2025/activation-oracles/) — same goal as NLAs, but uses labels.
5. **"Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations"** (Fraser-Taliente, Kantamneni, Ong et al., Anthropic, May 2026) — The paper this guide covers. Unsupervised like SAEs, text-native like Oracles. [transformer-circuits.pub/2026/nla](https://transformer-circuits.pub/2026/nla/)
6. **Cycle-Consistent Activation Oracles** (Chalnev, concurrent 2026) — Similar AV/AR loop, supervised warm-start.

## What Problem Does NLA Solve?

You already trained SAEs on your GPT-2 124M (or read the Anthropic ones). The pipeline is:

```
                                                          ┌── auto-interp LLM
h_l ──► SAE encoder ──► sparse code [0,0,2.3,0,1.1,...] ──┤   reads top-activating
                                                          │   examples for each atom
                                                          └── writes English label
```

Two problems:

1. **Coverage is bounded by the dictionary.** SAE has N atoms — anything orthogonal to all N is lost. Templeton 2024 used 34M atoms on Sonnet and still has gaps.
2. **The label isn't trained against anything.** The auto-interp LLM guesses from examples after the fact. Nothing forces the label to be faithful to what the activation actually encodes.

NLA collapses these two stages:

```
                                                  ┌──────────────────┐
h_l ──► AV (Verbalizer LLM) ──► English paragraph ┤ this IS the      │
                                                  │ explanation, and │
                                                  │ it was trained   │
                                                  │ to reconstruct h │
                                                  └──────────────────┘
```

The paragraph is the feature *and* the label. Reward signal (reconstruction MSE) directly grades the paragraph. Coverage is bounded only by what English can express.

## Key Terms

**Activation Verbalizer (AV)**: A fine-tuned copy of the target LLM that takes an activation `h_l ∈ R^d` and decodes a natural-language description `z` (typically a few hundred tokens).

**Activation Reconstructor (AR)**: A model that maps `z` back to `ĥ_l ∈ R^d`. In the paper, AR = the first `l` layers of the target LLM plus a learned affine head.

**Text bottleneck**: The paragraph `z` is the only path information takes from activation to reconstruction. Bottleneck width ≈ (entropy of token distribution) × (number of tokens).

**FVE (Fraction of Variance Explained)**: `1 − ||h − ĥ||² / Var(h)`. Standard reconstruction metric. Paper reports ~0.3–0.4 after SFT, 0.6–0.8 after RL.

**Auto-interp**: The Bills et al. 2023 pipeline of using a second LLM to label SAE features post-hoc. NLAs replace this.

**Confabulation**: The failure mode where AV writes plausible-sounding but unfaithful claims. The paper's #1 limitation.

**Steganography risk**: RL could push `z` toward token sequences that encode `h` via patterns only AR can decode, while reading as ordinary prose. KL penalty toward SFT-init mitigates.

## Architecture Overview

Three networks, two trained, one frozen. The frozen one defines the ground truth; the trained pair learn to compress and decompress its activation through a text bottleneck.

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║              NATURAL LANGUAGE AUTOENCODER — overall architecture              ║
╚═══════════════════════════════════════════════════════════════════════════════╝

   ┌──────────────────────────────────────────────────────────────────────────┐
   │                          ① TARGET MODEL  M   (FROZEN)                    │
   │                                                                          │
   │   Prompt P                                                               │
   │      │                                                                   │
   │      ▼                                                                   │
   │   ┌─────────┐    ┌─────────┐         ┌─────────┐    ┌─────────┐          │
   │   │ layer 1 │───▶│ layer 2 │── ··· ─▶│ layer L │── ─│ layer N │──▶ logits│
   │   └─────────┘    └─────────┘         └────┬────┘    └─────────┘          │
   │                                           │                              │
   │                                           ▼                              │
   │                                       h_l ∈ R^d     ◀── tap point        │
   │                                       (residual stream at chosen layer)  │
   └───────────────────────────────────────────┬──────────────────────────────┘
                                               │
                                               │  inject h_l as a special
                                               │  token embedding into AV
                                               ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │              ② ACTIVATION VERBALIZER  AV   (FULL COPY OF M, TRAINED)     │
   │                                                                          │
   │   Template tokens + [INJECTED h_l] + instruction tokens                  │
   │      │                                                                   │
   │      ▼                                                                   │
   │   ┌─────────┐    ┌─────────┐         ┌─────────┐    ┌─────────┐          │
   │   │ layer 1 │───▶│ layer 2 │── ··· ─▶│ layer L │── ─│ layer N │──▶ logits│
   │   └─────────┘    └─────────┘         └─────────┘    └─────────┘          │
   │                                                                          │
   │   ▼ autoregressive decode at T=1, ~few-hundred tokens                    │
   │                                                                          │
   │   z = "The model is processing a short narrative about a cat that has    │
   │        chased a mouse and is now tired. It is preparing to predict a     │
   │        resting action — sleeping, lying down, or grooming..."            │
   └───────────────────────────────────────────┬──────────────────────────────┘
                                               │
                                               │  treat z as a normal prompt
                                               ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │     ③ ACTIVATION RECONSTRUCTOR  AR   (FIRST L LAYERS + AFFINE, TRAINED)  │
   │                                                                          │
   │   z (text tokens)                                                        │
   │      │                                                                   │
   │      ▼                                                                   │
   │   ┌─────────┐    ┌─────────┐         ┌─────────┐    ┌───────────┐        │
   │   │ layer 1 │───▶│ layer 2 │── ··· ─▶│ layer L │───▶│ AFFINE    │──▶ ĥ_l │
   │   └─────────┘    └─────────┘         └─────────┘    │ W·x + b   │        │
   │                                                     │ d → d     │        │
   │                                                     └───────────┘        │
   │   (final-token hidden state taken, then projected back to residual dim)  │
   └───────────────────────────────────────────┬──────────────────────────────┘
                                               │
                                               ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │                              ④ LOSS / REWARD                             │
   │                                                                          │
   │     h_l ──╮                                                              │
   │           ├──▶  L2-normalize  ──▶   r = − log ‖ h_l  −  ĥ_l ‖²           │
   │     ĥ_l ──╯                                                              │
   │                                                                          │
   │     • AV update: GRPO on r  (K samples per activation, push toward high) │
   │     • AR update: supervised MSE on (z_best, h_l)                         │
   │     • KL anchor: D_KL( AV ‖ AV_SFT_init )  — prevents steganography      │
   │     • M never updates                                                    │
   └──────────────────────────────────────────────────────────────────────────┘
```

**Legend**

| Symbol | Role | Trained? | Architecture |
|---|---|---|---|
| M | Target — defines ground-truth `h_l` | frozen | full N layers of the model being interpreted |
| AV | Verbalizer — encodes `h_l` → text `z` | trained (SFT → GRPO) | full copy of M; accepts activation-injection slot |
| AR | Reconstructor — decodes text `z` → `ĥ_l` | trained (SFT → supervised in RL stage) | first L layers of M + small affine head (d → d) |

`d` = residual-stream width (e.g. 5376 for Gemma-3-27B). `L` = tap layer (e.g. 41). `N` = total layers (e.g. 62). `z` = a few-hundred-token English paragraph — the bottleneck.

### Inference-time view (after training)

```
   ┌─────────────────────────────────────────────────────────────────────┐
   │   Any prompt P ──▶ ❄ M ──▶ h_l ──▶ 🔥 AV ──▶  z  (English readout)  │
   │                                                                     │
   │   (optional sanity check:)                                          │
   │                          z ──▶ 🔥 AR ──▶ ĥ_l                        │
   │                                          ▲                          │
   │                                          │                          │
   │                                          └── compare cosine to h_l  │
   │                                              if low, z is unreliable│
   └─────────────────────────────────────────────────────────────────────┘
```

At inference you primarily need M + AV. AR stays around as a faithfulness check — if a paragraph fails to round-trip back to something close to the original activation, that paragraph is untrustworthy and should be discarded.

### One-screen mental model

```
   P  ─►  M  ─►  h_l  ─►  AV  ─►  z  ─►  AR  ─►  ĥ_l
   ❄     ❄              🔥             🔥
                                              ║
                                       loss = ‖ h_l − ĥ_l ‖²
                                              ║
                                       trains AV and AR;
                                       M is never touched.
```

## The Encode/Decode Loop

```
                   Natural Language Autoencoder
                   ───────────────────────────────────────────────

   h_l (residual)        text z                        ĥ_l
   ─────────────►   AV   ────────────►   AR   ─────────────────►
   d-dim vector    LLM   ~few hundred   LLM   d-dim vector
   at layer l    (verbal.) English tok (recon.) reconstruction

   Reward / loss:  r = -log ‖h_l − ĥ_l‖²   (vectors L2-normed,
                                            so MSE ≈ 2(1 − cos))
```

### AV — the encoder

AV is a fine-tuned copy of the target LLM (e.g., Gemma-3-27B describing Gemma-3-27B's activations). The activation is injected as if it were a token embedding into a fixed prompt template:

```
[system] You are an activation describer for a language model.
[user]   Here is an activation from layer 41:
           <INJECT h_l (scaled) into a special slot>
         Describe what the model is processing.
[assistant] <AV generates ~few-hundred token description z>
```

AV decodes autoregressively at temperature 1. The injected vector is the only model-specific information AV sees — everything else is a template.

### AR — the decoder

AR is structurally the **first `l` layers of the target LLM** plus a learned affine head:

```
z (text) ──► target_LLM[layers 1..l] ──► last hidden state ──► affine ──► ĥ_l
```

By reusing the target LLM's own early layers, AR has the same "way of reading text" as the model being interpreted. The affine head is the only trainable piece on AR's side that didn't come from the target.

### The reward

L2-normalize `h_l` and `ĥ_l`, then:

```
r = -log ‖h_l − ĥ_l‖²    ≈  log(1 / (2(1 − cos))) up to a constant
```

Equivalent to a log-cosine similarity. Magnitudes are dropped (only direction matters).

## Worked Example (concrete numbers)

Target: Gemma-3-27B. Layer 41 residual stream, `d = 5376`.

Prompt: *"The cat sat on the mat. It was tired after chasing a mouse all morning. The cat then..."*

Take `h_41` at the final token position — a 5376-d vector. Opaque on its own.

**AV sample 1** (high-reward direction):
> *"The model is processing a short narrative about a cat. The cat has performed two actions (sitting, chasing a mouse) and is described as tired. The model is preparing to predict the cat's next action — likely something restful like sleeping, lying down, or grooming."*

→ AR runs this as a prompt, reads the last hidden state, projects through the affine head. Result `ĥ_41` close to `h_41`. Cosine ≈ 0.85. Reward HIGH.

**AV sample 2** (low-reward distractor):
> *"This activation discusses recent fluctuations in banana futures markets and the impact of weather on commodity pricing."*

→ AR reconstructs a vector pointing in a totally different region of activation space. Cosine ≈ 0.05. Reward LOW.

GRPO pushes AV's policy toward the high-reward continuations. The only winning strategy is to actually describe what `h_41` encodes.

## Training Procedure

Four stages:

```
Stage 1: Generate training pairs
─────────────────────────────────
  Opus 4.5 ──► writes prompt P, then continues "what happens next" C
  Target LLM ──► encode (P, partial C) ──► extract h_l at position
                                          where C continues
  Pair: (h_l, gold-description-of-C)

Stage 2: AR SFT (text → activation)
───────────────────────────────────
  loss = ‖h_l - AR(z_gold)‖²
  Teaches AR to map English descriptions into activation space.

Stage 3: AV SFT (activation → text)
───────────────────────────────────
  loss = -log p_AV(z_gold | h_l)   [standard next-token loss]
  Warm-starts AV. Hits ~0.3-0.4 FVE.

Stage 4: Joint RL (the actual training)
───────────────────────────────────────
  For each (h_l):
    Sample K paragraphs {z_1, ..., z_K} ~ AV
    Compute rewards r_i = -log ‖h_l - AR(z_i)‖²
    GRPO update on AV using {r_i}
    Supervised update on AR using best z_i
    KL penalty: D_KL(AV ‖ AV_SFT_init) — prevents stego drift
  Climbs to 0.6-0.8 FVE.
```

The KL anchor matters. Without it, AV would drift toward token sequences that encode `h` losslessly to AR but read as gibberish to humans — defeating the whole point.

## Why It's Trainable Despite the Discrete Bottleneck

Text is discrete; you can't backprop through `argmax`. GRPO solves this:

```
For one activation h:
  ┌── Sample z_1 ── AR ── ĥ_1 ── reward r_1
  │
AV ── Sample z_2 ── AR ── ĥ_2 ── reward r_2     "Some of these
  │                                              paragraphs got
  ├── Sample z_3 ── AR ── ĥ_3 ── reward r_3      higher reward.
  │                                              Push the policy
  └── Sample z_4 ── AR ── ĥ_4 ── reward r_4     toward those."

GRPO update on AV: increase logp of high-reward samples,
                   decrease logp of low-reward samples,
                   normalized within the group.
```

This is exactly the recipe from your GRPO tutorial — relative advantages within a group of samples, no value function needed. AR is updated supervised in the same step because we have a clean MSE target.

## Open-Model Checkpoints Released

| Model | Layer | d_model | Use case |
|-------|-------|---------|----------|
| Qwen2.5-7B-Instruct | 20 | 3584 | small-scale experimentation |
| Gemma-3-12B-IT | 32 | 3840 | mid-scale |
| Gemma-3-27B-IT | 41 | 5376 | the main open-model NLA |
| Llama-3.3-70B-Instruct | 53 | 8192 | biggest open NLA |

GitHub: [kitft/natural_language_autoencoders](https://github.com/kitft/natural_language_autoencoders). Infra: Miles (RL orchestration) + SGLang (rollout serving).

Internally Anthropic trained NLAs on Haiku 3.5, Haiku 4.5, and Opus 4.6.

## Empirical Results

### Reconstruction Quality

```
FVE (Fraction of Variance Explained) — climbs roughly linear in log(RL steps)

  0.8 ┤                                            ●●●  end of RL
      │                                       ●●●●
  0.6 ┤                                  ●●●●
      │                             ●●●●
  0.4 ┤                       ●●●●●                end of SFT
      │                 ●●●●●●
  0.2 ┤           ●●●●●●
      │      ●●●●●
  0.0 ┼──────────────────────────────────────────────►
          AV-SFT     ──── joint RL (GRPO) ────►
```

### Hidden-Motivation Audit

In intentionally misaligned model checkpoints (planted hidden objectives), NLA descriptions surfaced the hidden motivation in **12–15%** of test activations vs. **<3%** for the SAE+auto-interp baseline — ~4–5x lift.

### Unverbalized Evaluation-Awareness (Opus 4.6 pre-deployment)

NLA paragraphs flagged "model recognizes this as an evaluation scenario" in:

```
Activations on destructive-action coding tasks (test):   16%
Activations on SWE-bench problems (test):                26%
Activations on opted-in real claude.ai traffic:          <1%
```

Opus said nothing like this aloud in its outputs. NLAs caught it from the residual stream.

### Faithfulness Sanity Check (Gemma-3-27B, math)

When the model emits a *wrong* numerical answer:

| Description type | % containing the exact wrong answer |
|---|---|
| NLA on the activation that produced it | **40.7%** |
| NLA on activation from a different wrong problem | 3.3% |

~12× signal-to-noise. The NLA is genuinely reading what's in `h`, not hallucinating from priors.

### Mechanistic Validation

Replicated Lindsey et al.'s "models plan rhymes in poetry" by extracting steering vectors directly from NLA descriptions of rhyme-planning activations. Steering with NLA-derived directions flipped rhyme target ~50% of the time — comparable to SAE-feature-derived steering.

Diagnosed an early Opus 4.6 bug where the model occasionally replied in Russian to English prompts. NLA flagged activations carrying *"thinks user is a non-native English speaker"*; tracing to training data confirmed the spurious correlation.

## NLA vs SAE vs Activation Oracles

```
                       NLA                        SAE + auto-interp           Activation Oracle
                       ─────────────────────      ──────────────────────      ──────────────────
Output                 English paragraph          sparse atom + LLM label     Q&A interface
Labels needed?         no (unsupervised)          no (unsupervised)           yes (supervised)
Label trained?         yes, via reconstruction    no, post-hoc                yes
Coverage               open-ended (language)      bounded by dictionary       bounded by Q&A coverage
Decomposition          monolithic                 per-feature, addressable    per-question
Causal steering        weak (blackbox AR)         strong (linear features)    weak
Compute (inference)    expensive (~200 tok/act)   cheap                       cheap-medium
Compute (training)     expensive (RL on 2 LLMs)   medium-expensive            expensive (labels)
Best at                audits, intent, themes     circuits, edits, atomicity  ad-hoc questions
Faithfulness           medium (40% on math)       label is decoupled          high (when in-dist)
```

The headline framing: SAEs give you a *vocabulary*, NLAs give you a *summary*, Oracles give you a *Q&A endpoint*. They are complementary.

## Limitations & Failure Modes

1. **Confabulation.** NLA paragraphs can assert false specifics. Numbers, named entities, dates are most affected. Authors' guidance: read for themes, not single claims.

2. **No mechanistic decomposition.** AR is a blackbox — you can't trace "which subspace of `h_l` produced which sentence of `z`." A regression from SAEs on causal-circuit work.

3. **AV is a full LLM** → it has strong priors → it can plausibly *guess* attributes (gender, occupation) that aren't actually in `h`.

4. **Steganography risk.** RL could in principle make `z` a steganographic encoding of `h` that reads as natural prose. KL anchor mitigates but doesn't prove safety. Top concern in public discussion.

5. **Anthropomorphization.** Alien internal computation gets compressed into human-shaped motivational narratives because that's AV's prior.

6. **Cost.** Joint RL on two LLM-sized models, with ~200-token rollouts per activation. Impractical for per-token monitoring of long traces. SAE inference is ~1000× cheaper.

7. **Convergence/stability.** Open question raised in public review: do independent NLA runs produce consistent descriptions of the same activation? Not answered in the paper.

8. **Greenblatt's negative result.** External red-team probe: NLAs failed to recover full chain-of-thought reasoning on single-forward-pass arithmetic problems, though intermediates were "non-trivially interesting."

## Practical Considerations

**When to use NLA over SAE:**
- You want a high-level *summary* of what a model is thinking, not a feature list.
- You're doing alignment audits (hidden motivations, eval-awareness, unstated intent).
- You don't have labels for activation behavior.
- You only need to look at a few activations, not all of them.

**When to use SAE over NLA:**
- You need atomic, addressable features for steering / ablation / circuit work.
- You need per-token, real-time coverage.
- You care about exact mechanistic causality, not narrative summary.
- You can't afford the RL training cost of an NLA.

**When to use Activation Oracles over NLA:**
- You have labeled activation data.
- You want flexible Q&A rather than fixed-format paragraph summaries.

**Practical deployment notes:**
- Run NLA on suspicious activations flagged by cheap monitors (logits, simple probes), not on every token.
- Treat NLA output as a *hypothesis* to verify mechanistically (patching, ablation), not a ground truth.
- Cross-check: do K independent samples from AV converge on the same theme? If not, the activation may not be cleanly describable.

## Hands-on Sketch (Gemma-3-27B NLA)

The released checkpoint is on HuggingFace as separate AV and AR weights. Rough sketch:

```python
from transformers import AutoTokenizer
import torch
from nla import ActivationVerbalizer, ActivationReconstructor

tok = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")
av = ActivationVerbalizer.from_pretrained("kitft/nla-gemma3-27b-av")
ar = ActivationReconstructor.from_pretrained("kitft/nla-gemma3-27b-ar")

# Suppose you've already extracted h_l at layer 41 from the target model
h = torch.load("activation_layer41.pt")  # shape: (5376,)

# Describe the activation
z = av.verbalize(h, max_new_tokens=300, temperature=1.0)
print(z)
# → "The model is processing a short narrative about a cat..."

# Verify by round-tripping
h_hat = ar.reconstruct(z)
cos = torch.nn.functional.cosine_similarity(h, h_hat, dim=0)
print(f"Round-trip cosine: {cos:.3f}")
# → 0.83
```

Run with `K=5` samples and see if themes converge — that's your stability check.

## Key Papers

| Paper | Authors | Year | Link |
|-------|---------|------|------|
| Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations | Fraser-Taliente, Kantamneni, Ong et al. | 2026 | [transformer-circuits.pub/2026/nla](https://transformer-circuits.pub/2026/nla/) |
| Activation Oracles | Anthropic Alignment | 2025 | [alignment.anthropic.com](https://alignment.anthropic.com/2025/activation-oracles/) |
| Scaling Monosemanticity | Templeton et al. | 2024 | [transformer-circuits.pub](https://transformer-circuits.pub/2024/scaling-monosemanticity/) |
| Towards Monosemanticity | Bricken, Templeton et al. | 2023 | [transformer-circuits.pub](https://transformer-circuits.pub/2023/monosemantic-features) |
| Language Models Can Explain Neurons | Bills, Cammarata et al. | 2023 | [openai neuron-explainer](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html) |
| Toy Models of Superposition | Elhage, Hume, Olsson et al. | 2022 | [arxiv.org/abs/2209.10652](https://arxiv.org/abs/2209.10652) |

**Resources**:
- Anthropic blog: [Natural Language Autoencoders](https://www.anthropic.com/research/natural-language-autoencoders)
- Code & checkpoints: [github.com/kitft/natural_language_autoencoders](https://github.com/kitft/natural_language_autoencoders)
- Public discussion: [LessWrong](https://www.lesswrong.com/posts/oeYesesaxjzMAktCM/natural-language-autoencoders-produce-unsupervised) — contains Greenblatt's CoT-recovery probe and the steganography/anthropomorphization debate.
