# Interpretability & Mechanistic Understanding of LLMs

A step-by-step guide to understanding what happens inside transformer language models — from individual neurons to full circuits.

## Background

**Research lineage** — Mechanistic interpretability evolved from Anthropic's Transformer Circuits thread:

1. **"A Mathematical Framework for Transformer Circuits"** (Elhage, Nanda, Olsson et al., Anthropic, Dec 2021) — Established the mathematical foundation for reverse-engineering transformers. Introduced the residual stream view: attention heads and MLPs read from and write to a shared communication channel. Showed how attention heads can be understood as performing linear operations on the residual stream, and how they compose across layers via "virtual attention heads."

2. **"In-context Learning and Induction Heads"** (Olsson, Elhage et al., Anthropic, Sep 2022) — Discovered that a specific circuit (induction heads) is responsible for in-context learning. Showed a sharp phase transition during training where induction heads form and in-context learning ability suddenly appears. Six independent lines of evidence argue induction heads are the mechanistic source of general in-context learning.

3. **"Toy Models of Superposition"** (Elhage, Hume, Olsson et al., Anthropic, Sep 2022) — Formalized why interpretability is hard: neural networks represent more features than they have dimensions by encoding them in overlapping directions (superposition). Used small ReLU networks on synthetic data to map out when superposition occurs and why. Laid the theoretical groundwork for sparse autoencoders.

4. **"Towards Monosemanticity: Decomposing Language Models with Dictionary Learning"** (Bricken, Templeton, Conerly, Nanda et al., Anthropic, Oct 2023) — Applied sparse autoencoders (SAEs) to a 1-layer 512-neuron transformer and extracted interpretable, monosemantic features. Demonstrated that dictionary learning can decompose polysemantic neurons into clean, understandable units. Proved the concept works in practice.

5. **"Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet"** (Templeton et al., Anthropic, May 2024) — Scaled SAEs to a production model (Claude 3 Sonnet), extracting 34 million features from the middle-layer residual stream. Found features ranging from concrete concepts ("Golden Gate Bridge") to abstract ones ("code bugs", "deceptive behavior"). Demonstrated feature steering — clamping a feature to 10x its max value changes model behavior.

**Parallel work**:
- **"Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small"** (Wang, Variengien et al., Nov 2022) — The largest end-to-end circuit reverse-engineering effort at the time. Found 26 attention heads in 7 classes that implement indirect object identification.
- **"Locating and Editing Factual Associations in GPT"** (Meng et al., 2022) — Introduced causal tracing (activation patching) for locating factual knowledge in transformers.

## What Problem Does Interpretability Solve?

You've trained models — you know the pipeline. You define an architecture, run gradient descent, and get weights that produce useful outputs. But you have no idea **why** any particular output was produced. The model is a black box.

This matters for several concrete reasons:

**Safety**: If a model learns a deceptive strategy during RLHF (producing outputs that look aligned while pursuing a hidden objective), how would you detect it? You can't just look at outputs — a sufficiently capable model could game evaluations. You need to inspect the internal computation.

**Debugging**: Your SFT model sometimes generates toxic outputs. Is it a data problem? A specific attention head that learned the wrong pattern? Without interpretability tools, you're guessing.

**Alignment verification**: After DPO training, did the model actually learn your intended preferences, or did it find a shortcut (e.g., "longer responses score higher")? Mechanistic analysis can distinguish genuine learning from Goodhart-style gaming.

**Trust**: As models are deployed in high-stakes domains (medicine, law, infrastructure), "it usually works" isn't sufficient. We need to understand failure modes mechanistically, not just statistically.

The core tension: neural networks are **not designed** to be interpretable. Unlike traditional software where you can read the source code, a transformer's "algorithm" is distributed across billions of floating-point parameters. Mechanistic interpretability is the project of reverse-engineering these parameters back into human-understandable algorithms.

## Key Terms

**Residual stream**: The main communication channel in a transformer. A vector (dimension `d_model`, e.g., 768 for GPT-2 124M) that flows through all layers. Attention heads and MLPs read from it and write back to it via addition. Think of it as a shared whiteboard that every component can read and update.

**Feature**: A property of the input that the model represents internally. Examples: "this token is a proper noun", "the text is about science fiction", "the current token follows an opening parenthesis." Features are represented as directions in activation space, not necessarily aligned with individual neurons.

**Superposition**: The phenomenon where a model represents more features than it has dimensions, by encoding features as nearly-orthogonal directions. A 768-dimensional residual stream might encode thousands of features by accepting small amounts of interference between them. This is why individual neurons are hard to interpret.

**Polysemanticity**: A single neuron activates for multiple unrelated concepts. Example: a neuron fires for both "academic citations" and "cat breeds." This is a consequence of superposition — the neuron participates in multiple feature directions.

**Monosemanticity**: A neuron or feature that represents a single, interpretable concept. The goal of dictionary learning is to decompose polysemantic neurons into monosemantic features.

**Circuit**: A subgraph of the model's computational graph that implements a specific behavior. A circuit consists of specific attention heads and MLP neurons across specific layers, connected by how they read from and write to the residual stream. Analogous to a function or subroutine in traditional code.

**Attention head**: One "head" in a multi-head attention layer. Each head independently computes what to attend to (via QK circuit) and what information to move (via OV circuit). GPT-2 124M has 12 layers x 12 heads = 144 attention heads.

**Induction head**: A specific type of attention head that completes repeated patterns. Given the sequence `[A][B]...[A]`, an induction head predicts `[B]`. This is the primary mechanism for in-context learning.

**Sparse autoencoder (SAE)**: A neural network trained to decompose model activations into a sparse set of interpretable features. The encoder maps a `d_model`-dimensional activation to a much larger (e.g., 16x-64x) hidden space via ReLU, producing a sparse vector. The decoder reconstructs the original activation. The dictionary columns (decoder weights) represent individual features.

**Probing / Linear probes**: Training a simple linear classifier on a model's internal activations to test whether specific information is represented. Example: train a probe on layer 6 activations to predict "is this token part of a French word?" High accuracy means that information is linearly accessible at that layer.

**Activation patching** (aka causal tracing): A causal intervention technique. Run the model on a clean input and a corrupted input, then patch specific activations from the clean run into the corrupted run. If the output recovers, that activation is causally important. Three forward passes: clean (cache activations), corrupted (wrong answer), patched (restore one component to test its importance).

**Logit lens**: A technique that applies the model's final unembedding matrix to intermediate layer activations, converting hidden states into vocabulary distributions. Reveals what the model "would predict" at each layer — lets you watch a prediction form layer by layer.

**Tuned lens**: An improved version of the logit lens. Trains a small affine transformation per layer (rather than reusing the final unembedding directly), producing more reliable intermediate predictions. Addresses the logit lens's failure on some model families.

## The Residual Stream View

The standard way to think about a transformer is "layer by layer" — input goes through layer 1, then layer 2, etc. The mechanistic interpretability view is different: the **residual stream** is the primary object, and layers are components that read from and write to it.

```
Token embeddings + Position embeddings
              │
              ▼
     ┌────────────────────┐
     │   Residual Stream   │  ← d_model-dimensional vector (768 for GPT-2)
     │   (shared channel)  │
     └────────┬───────────┘
              │
    ┌─────────┼─────────┐
    │         │         │
    ▼         │         │
 ┌──────┐    │         │
 │Attn 0│────┘         │   read from stream, compute, write back (addition)
 └──────┘    │         │
    ┌────────┼─────────┐
    │         │         │
    ▼         │         │
 ┌──────┐    │         │
 │MLP 0 │────┘         │   same: read from stream, write back
 └──────┘    │         │
    ┌────────┼─────────┐
    │         │         │
    ▼         │         │
 ┌──────┐    │         │
 │Attn 1│────┘         │
 └──────┘    │         │
    ┌────────┼─────────┐
    │         │         │
    ▼         │         │
 ┌──────┐    │         │
 │MLP 1 │────┘         │
 └──────┘    │         │
              │         │
             ...       ...  (12 layers for GPT-2 124M)
              │         │
    ┌─────────┼─────────┐
    │         │         │
    ▼         │         │
 ┌──────┐    │         │
 │Attn11│────┘         │
 └──────┘    │         │
    ┌────────┼─────────┐
    │         │         │
    ▼         │         │
 ┌──────┐    │         │
 │MLP 11│────┘         │
 └──────┘              │
              │         │
              ▼         │
     ┌────────────────────┐
     │    Unembedding      │  final residual stream → logits
     └────────────────────┘
```

The key insight: **the residual stream at any point is the sum of the original embeddings plus all previous attention and MLP outputs**:

```
residual_stream[after layer L] = embed + pos_embed
                                + attn_0_out + mlp_0_out
                                + attn_1_out + mlp_1_out
                                + ...
                                + attn_L_out + mlp_L_out
```

This is just addition. Every component writes a vector that gets added to the stream. The final logits are a linear function (unembedding matrix) of this sum. Because everything combines linearly at the residual stream level, you can analyze each component's contribution independently.

**Why this matters**: You can ask "which attention head contributed the most to predicting token X?" by looking at the dot product of each head's output with the unembedding direction for token X. This is the foundation of logit attribution and direct logit attribution.

**For your GPT-2 124M**: The residual stream is 768-dimensional. At each position in the sequence, there are 24 components writing to it (12 attention layers + 12 MLP layers), plus the initial embedding. That's 25 vectors being summed. Each attention layer has 12 heads, so there are actually 144 + 12 + 1 = 157 individual write operations contributing to each position's final residual stream.

## Circuits and Features

### What Is a "Circuit"?

A circuit is a subset of the model's components (specific attention heads and MLP neurons across specific layers) that together implement a specific, identifiable behavior. It's the mechanistic interpretability analogue of a function in code.

The key properties of a good circuit explanation:
- **Faithful**: The circuit reproduces the behavior when run in isolation
- **Complete**: No important components are missing
- **Minimal**: No unnecessary components are included

### Concrete Example: Indirect Object Identification (IOI)

The IOI circuit (Wang et al., 2022) is the most thoroughly reverse-engineered circuit in a language model. It handles sentences like:

```
"When Mary and John went to the store, John gave a drink to ___"
→ Model predicts "Mary" (the indirect object — the name that isn't the subject)
```

The circuit in GPT-2 small uses 26 attention heads across 7 functional classes:

```
Input: "When Mary and John went to the store, John gave a drink to"

Layer 0-1:  Duplicate Token Heads
            └─ Detect that "John" appears twice

Layer 2-4:  Previous Token Heads
            └─ Attend to the token before each name

Layer 5-6:  S-Inhibition Heads
            └─ Suppress the repeated name ("John") from the prediction

Layer 7-8:  Name Mover Heads
            └─ Copy the non-suppressed name ("Mary") to the output

Layer 9-11: Backup Name Mover Heads
            └─ Redundant copies of name movers (robustness)

            ┌──────────────────────────────────────────────────────┐
Final       │ Output logits boosted for "Mary", suppressed for    │
position:   │ "John" → model predicts "Mary"                      │
            └──────────────────────────────────────────────────────┘
```

This is a complete algorithm, discovered by tracing backward from the logits using activation patching. Each class of head has a specific, understandable role.

### How Circuits Compose Across Layers

Circuits work because earlier components write information into the residual stream that later components read. The composition happens through the residual stream:

```
Layer 0:  Head writes "token X is a duplicate" into residual stream
            │
            │  (information travels via residual stream)
            ▼
Layer 5:  Head reads "token X is a duplicate" → suppresses X in output
            │
            │  (suppression signal in residual stream)
            ▼
Layer 7:  Head reads suppression signal → copies non-suppressed name
```

There are three types of composition between attention heads:
- **Q-composition**: Head B uses head A's output to compute its queries (what to look for)
- **K-composition**: Head B uses head A's output to compute its keys (what to be found as)
- **V-composition**: Head B uses head A's output to compute its values (what information to move)

Induction heads (next section) are the canonical example of K-composition.

## Induction Heads Deep Dive

### What They Are

Induction heads implement a simple but powerful algorithm: **pattern completion of repeated sequences**. Given a sequence like `[A][B]...[A]`, an induction head predicts `[B]`. This is the core mechanism behind in-context learning.

Example:
```
Prompt: "The cat sat on the mat. The cat sat on the"
                                                   ↑
Induction head recognizes "The cat sat on the" appeared before,
followed by "mat" → predicts "mat"
```

### The Two-Head Mechanism

Induction heads require **two** attention heads working together across layers. Neither head alone can do the job.

**Step 1 — Previous Token Head (early layer, e.g., layer 0-1)**:

This head always attends to the previous token position. Its job is to write information about "what came before me" into the residual stream.

```
Position:  ... "The"  "cat"  "sat"  "on"  "the"  "mat"  ...
                 ↑      │
                 └──────┘  Previous token head at "cat" attends to "The"
                           Writes: "I was preceded by 'The'" into residual stream
```

After this head runs, each position's residual stream contains information about the previous token — shifted by one position.

**Step 2 — Induction Head (later layer, e.g., layer 5-6)**:

This head uses K-composition with the previous token head. It asks: "where in the context did a token appear that was preceded by the same token as my current input?"

```
Current position: second "the" (position 10)
Query: "Find a position where the previous token was 'the'"
                                                    ↓
Found: "mat" (position 6), because position 6 was preceded by "the" (position 5)
                                                    ↓
Attend to position 6 → copy "mat" → predict "mat"
```

The full mechanism:

```
Text:     ... the   cat   sat   on   the   mat   .   The   cat   sat   on   the   ???
Position: ... 0     1     2     3    4     5     6   7     8     9     10   11    12

Step 1 (Layer 0-1): Previous Token Head
  At pos 5 ("mat"):  attends to pos 4 ("the") → writes "preceded by 'the'" at pos 5
  At pos 12 ("???"): looking for pattern completion

Step 2 (Layer 5-6): Induction Head
  At pos 12: query = "current token is 'the', find where 'the' was followed by something"
  Keys at pos 5 contain "preceded by 'the'" (from Step 1)
  → pos 12 attends to pos 5 ("mat")
  → copies "mat" to output → predicts "mat"
```

### Why Induction Heads Matter

Olsson et al. (2022) found that:

1. **Phase transition**: Induction heads form at a specific point during training, coinciding with a sudden drop in loss on repeated sequences. This is visible as a "bump" in the training loss curve.

2. **Universality**: Induction heads appear in every transformer model they examined, from 2-layer attention-only models to full-scale production models.

3. **In-context learning = induction**: The evidence strongly suggests induction heads are the primary mechanism for in-context learning — the ability of transformers to improve at a task within a single forward pass by using examples in the prompt.

4. **Generalization**: While the simplest induction heads do literal token matching (`[A][B]...[A]→[B]`), more sophisticated versions do fuzzy/abstract matching — matching on semantic similarity rather than exact tokens. This is likely how few-shot prompting works.

### How to Find Them in Your GPT-2 124M

You can detect induction heads by checking their attention pattern on repeated random sequences:

```python
import torch
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2-small")

# Create a sequence with a repeated random segment
seq_len = 50
random_tokens = torch.randint(0, model.cfg.d_vocab, (1, seq_len))
repeated_tokens = torch.cat([random_tokens, random_tokens], dim=1)  # [A B C ... A B C ...]

# Run model, cache all activations
_, cache = model.run_with_cache(repeated_tokens)

# Score each head: does it attend from position i (in second half)
# to position i-seq_len+1 (one after the matching token in first half)?
# This is the "induction stripe" pattern.
induction_scores = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)
for layer in range(model.cfg.n_layers):
    attn_pattern = cache["pattern", layer][0]  # shape: (n_heads, seq, seq)
    for head in range(model.cfg.n_heads):
        # For positions in second half, check attention to (pos - seq_len + 1)
        score = 0.0
        count = 0
        for pos in range(seq_len + 1, 2 * seq_len):
            target = pos - seq_len + 1  # position after the matching token
            score += attn_pattern[head, pos, target].item()
            count += 1
        induction_scores[layer, head] = score / count

# Print top induction heads
print("Top induction heads (layer, head, score):")
flat_scores = induction_scores.flatten()
top_indices = torch.topk(flat_scores, k=10).indices
for idx in top_indices:
    layer = idx // model.cfg.n_heads
    head = idx % model.cfg.n_heads
    print(f"  L{layer}H{head}: {induction_scores[layer, head]:.3f}")
```

In GPT-2 small, you'll typically find strong induction heads at layers 5-6. Scores above 0.4 indicate a clear induction head.

## Superposition and Dictionary Learning

### The Superposition Hypothesis

Your GPT-2 124M has a 768-dimensional residual stream. If each feature required its own dimension (its own neuron), the model could represent at most 768 features. But natural language has far more than 768 concepts the model needs to track.

The superposition hypothesis: **models encode many more features than they have dimensions by representing features as nearly-orthogonal directions in activation space, tolerating small amounts of interference.**

In 768 dimensions, you can fit thousands of nearly-orthogonal vectors. If features are **sparse** (only a few active at any time), the interference between them is manageable — most features are inactive, so their interference doesn't cause problems in practice.

```
768-dimensional space, 3 features shown:

Aligned with neurons (no superposition):
  Feature A = [1, 0, 0, 0, ...]     ← uses neuron 0
  Feature B = [0, 1, 0, 0, ...]     ← uses neuron 1
  Feature C = [0, 0, 1, 0, ...]     ← uses neuron 2
  Max features = 768

In superposition:
  Feature A = [0.8, 0.3, -0.1, 0.5, ...]   ← spread across many neurons
  Feature B = [0.2, -0.7, 0.4, 0.3, ...]   ← different direction, nearly orthogonal
  Feature C = [-0.1, 0.5, 0.7, -0.2, ...]  ← yet another direction
  ...
  Feature N = [0.4, -0.2, 0.1, 0.8, ...]   ← N >> 768
```

Elhage et al. (2022) showed with toy models that superposition follows a **phase transition**: when features are dense (frequently active), the model stores them in dedicated dimensions. When features are sparse (rarely active), the model packs them into superposition because the rare interference is worth the extra capacity.

### Why Superposition Makes Interpretation Hard

If each neuron corresponded to one feature, you could just inspect individual neurons. But in superposition, a single neuron participates in many features (polysemanticity), and each feature is spread across many neurons. Looking at individual neurons gives you an incoherent mix of concepts.

```
Neuron 42 fires for:         (polysemantic — participates in multiple features)
  - Academic citations
  - Cat breeds
  - Semicolons in code

"Academic citation" feature:  (distributed — spread across many neurons)
  Neuron 42: +0.3
  Neuron 107: +0.7
  Neuron 203: -0.2
  Neuron 418: +0.5
  ... (many more)
```

### Sparse Autoencoders: The Solution

If the model internally represents features as sparse linear combinations of directions in activation space, we can learn those directions with a **sparse autoencoder (SAE)**. The SAE's job is to find a dictionary of feature directions that explain the model's activations.

**Architecture**:

```
Model activation x ∈ R^d_model (e.g., 768 for GPT-2)
         │
         ▼
┌─────────────────────────────┐
│  Encoder: h = ReLU(W_enc·x + b_enc)    h ∈ R^d_hidden (e.g., 768 × 16 = 12288)
│                                          h is SPARSE (mostly zeros due to ReLU + L1)
│                                                                                     │
│  Decoder: x̂ = W_dec·h + b_dec          x̂ ∈ R^d_model (reconstructs x)             │
└─────────────────────────────┘

Loss = ||x - x̂||² + λ·||h||₁
        ↑              ↑
    reconstruction    sparsity penalty (encourages most of h to be zero)
```

The hidden dimension is much larger than the input (expansion factor of 16x-64x). The L1 penalty on `h` ensures only a few features are active for any given input — typically 50-200 out of 12,000+.

**What the SAE learns**:
- Each **column of W_dec** (decoder weight matrix) represents one feature direction
- Each **element of h** represents the activation strength of that feature
- The columns of W_dec form the "dictionary" — a set of monosemantic feature directions

**Training**: SAEs are trained **after** the base model, on cached activations from the model's residual stream (or MLP outputs, or attention outputs). The base model is not modified.

### What Monosemantic Features Look Like

From Anthropic's "Towards Monosemanticity" (1-layer model) and "Scaling Monosemanticity" (Claude 3 Sonnet):

```
Feature examples from Claude 3 Sonnet (34M features extracted):

Feature 1:  "Golden Gate Bridge"
  Activates on: text mentioning Golden Gate Bridge, San Francisco landmarks, suspension bridges
  When amplified: Claude claims to BE the Golden Gate Bridge

Feature 2:  "Code bugs / errors"
  Activates on: buggy code, error messages, debugging discussions
  When amplified: Claude becomes paranoid about bugs in everything

Feature 3:  "Deceptive behavior"
  Activates on: text about lying, manipulation, hidden motives
  Safety-relevant: could be monitored to detect if model is reasoning about deception

Feature 4:  "Academic citations"
  Activates on: text with paper references, citation formats, bibliography
  Clean, monosemantic: doesn't also fire for unrelated concepts

Feature 5:  "Python f-strings"
  Activates on: f"..." syntax, string formatting in Python
  Highly specific, language-level feature
```

Each feature has a clear interpretation, activates consistently for related inputs, and doesn't spuriously activate for unrelated content. This is the monosemanticity goal.

## Practical Tools

### TransformerLens

**What it does**: A library by Neel Nanda (formerly Anthropic, now Google DeepMind) that loads 50+ open-source language models and exposes all internal activations. You can cache any activation, hook into any component, and run causal interventions.

**Key capabilities**:
- Load models with clean, interpretable internal structure
- Cache all intermediate activations in a single forward pass
- Add hook functions that modify activations during forward passes
- Decompose outputs by component (which head contributed what to the logits)

```python
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2-small")

# Run model and cache everything
logits, cache = model.run_with_cache("The capital of France is")

# Access any intermediate activation
residual_L6 = cache["resid_post", 6]        # residual stream after layer 6
attn_pattern_L5H3 = cache["pattern", 5][:, 3]  # attention pattern of L5H3
mlp_out_L8 = cache["mlp_out", 8]            # MLP output at layer 8

# Decompose logits by component
logit_attr = cache.logit_attrs(logits, tokens=model.to_tokens("Paris"))
# Returns contribution of each component to the "Paris" logit
```

### Activation Patching

The main technique for finding which components matter for a specific behavior. Three forward passes:

```
Pass 1 (Clean):    "The Eiffel Tower is in" → "Paris"     (cache activations)
Pass 2 (Corrupt):  "The Colosseum is in"    → "Rome"      (wrong answer for original)
Pass 3 (Patched):  "The Colosseum is in"    → ???         (patch one component from clean)

If patching L8H6's output restores "Paris" → L8H6 is causally responsible
for encoding "Eiffel Tower → Paris" knowledge
```

```python
from transformer_lens import patching

# Patch each head's output and measure effect on correct logit
patching_results = patching.get_act_patch_attn_head_out_all_pos(
    model,
    corrupted_tokens,
    cache,             # clean cache
    metric_fn,         # e.g., logit difference between "Paris" and "Rome"
)
# Returns: (n_layers, n_heads, seq_len) tensor of patching effects
```

### Logit Lens / Tuned Lens

**Logit lens**: Apply the model's unembedding matrix to intermediate residual stream states. Shows what the model "would predict" at each layer.

```python
# Manual logit lens
for layer in range(model.cfg.n_layers):
    residual = cache["resid_post", layer][0, -1]  # last token position
    logits = model.unembed(model.ln_final(residual))
    top_token = model.to_string(logits.argmax())
    print(f"Layer {layer:2d}: top prediction = {top_token}")

# Output for "The capital of France is":
# Layer  0: top prediction = " the"
# Layer  1: top prediction = " the"
# Layer  2: top prediction = " France"
# ...
# Layer  8: top prediction = " Paris"      ← prediction forms here
# Layer  9: top prediction = " Paris"
# Layer 10: top prediction = " Paris"
# Layer 11: top prediction = " Paris"
```

**Tuned lens**: Trains a small affine probe per layer for more accurate intermediate predictions. Especially useful for models where the logit lens is unreliable (non-GPT-2 architectures).

```bash
pip install tuned-lens
tuned-lens train --model gpt2 --dataset wikitext
tuned-lens eval --model gpt2
```

### Attention Visualization

Visualize which tokens each head attends to:

```python
import circuitsvis as cv

# Visualize attention patterns for a specific layer
tokens = model.to_str_tokens("The cat sat on the mat")
attn_pattern = cache["pattern", 5][0]  # layer 5, shape: (n_heads, seq, seq)
cv.attention.attention_patterns(tokens=tokens, attention=attn_pattern)
```

Look for:
- **Previous token pattern**: head attends to position i-1 (previous token head)
- **Induction stripe**: diagonal pattern shifted by the repeated sequence length
- **Positional pattern**: head always attends to position 0 or last position
- **Semantic pattern**: head attends to specific token types (names, verbs, etc.)

## Hands-on with Your GPT-2 124M

### Exercise 1: Visualize Attention Patterns

```python
from transformer_lens import HookedTransformer
import torch

model = HookedTransformer.from_pretrained("gpt2-small")

text = "When Mary and John went to the store, John gave a bottle of milk to"
tokens = model.to_str_tokens(text)
logits, cache = model.run_with_cache(text)

# Check what the model predicts
print("Top prediction:", model.to_string(logits[0, -1].argmax()))

# Look at attention patterns for each layer
for layer in range(12):
    pattern = cache["pattern", layer][0]  # (12 heads, seq_len, seq_len)
    # For each head, show where the last token attends most
    for head in range(12):
        last_token_attn = pattern[head, -1]  # attention from last position
        top_pos = last_token_attn.argmax().item()
        if last_token_attn[top_pos] > 0.3:  # strong attention
            print(f"L{layer}H{head}: last token attends to '{tokens[top_pos]}' "
                  f"(pos {top_pos}, weight {last_token_attn[top_pos]:.2f})")
```

### Exercise 2: Find Induction Heads

Use the code from the Induction Heads section above. After finding the top induction heads, verify them by checking their attention pattern on natural text with repeated phrases:

```python
# Verify a suspected induction head (e.g., L5H1)
text = "I like cats. I like cats."
_, cache = model.run_with_cache(text)
tokens = model.to_str_tokens(text)
pattern = cache["pattern", 5][0, 1]  # layer 5, head 1

print("Attention pattern for L5H1:")
for q_pos in range(len(tokens)):
    top_k = pattern[q_pos].topk(3)
    top_strs = [(tokens[i], f"{v:.2f}") for v, i in zip(top_k.values, top_k.indices)]
    print(f"  '{tokens[q_pos]}' attends to: {top_strs}")
```

### Exercise 3: Logit Lens on Your Model

Track how a prediction builds up layer by layer:

```python
text = "The capital of France is"
logits, cache = model.run_with_cache(text)
last_pos = -1

print("Layer-by-layer prediction (logit lens):")
for layer in range(12):
    residual = cache["resid_post", layer][0, last_pos]
    layer_logits = model.unembed(model.ln_final(residual))
    probs = torch.softmax(layer_logits, dim=-1)
    top5 = probs.topk(5)
    top5_strs = [f"{model.to_string(idx)}({p:.2%})" for p, idx in zip(top5.values, top5.indices)]
    print(f"  Layer {layer:2d}: {', '.join(top5_strs)}")
```

### Exercise 4: Compare Activations Before/After SFT

If you have both a base GPT-2 checkpoint and your SFT checkpoint:

```python
base_model = HookedTransformer.from_pretrained("gpt2-small")
sft_model = HookedTransformer.from_pretrained("path/to/your/sft/checkpoint")

text = "What is the meaning of life?"
_, base_cache = base_model.run_with_cache(text)
_, sft_cache = sft_model.run_with_cache(text)

# Compare residual streams at each layer
for layer in range(12):
    base_resid = base_cache["resid_post", layer][0]
    sft_resid = sft_cache["resid_post", layer][0]
    cos_sim = torch.cosine_similarity(base_resid, sft_resid, dim=-1).mean()
    l2_diff = (base_resid - sft_resid).norm(dim=-1).mean()
    print(f"Layer {layer:2d}: cos_sim={cos_sim:.4f}, L2_diff={l2_diff:.4f}")

# Early layers should be similar (cos_sim close to 1.0)
# Later layers may diverge more (SFT modifies high-level representations)
```

### Exercise 5: Direct Logit Attribution

Which components contribute most to the model's prediction?

```python
text = "The Eiffel Tower is in"
logits, cache = model.run_with_cache(text)
answer_token = model.to_single_token(" Paris")

# Get the unembedding direction for "Paris"
paris_dir = model.unembed.W_U[:, answer_token]  # (d_model,)

# Attribute logits to each attention head
print("Attention head contributions to 'Paris' logit:")
for layer in range(12):
    for head in range(12):
        head_out = cache["result", layer][0, -1, head]  # (d_model,)
        contribution = head_out @ paris_dir
        if abs(contribution.item()) > 0.5:
            print(f"  L{layer}H{head}: {contribution.item():+.2f}")
```

## Scaling Monosemanticity

Anthropic's "Scaling Monosemanticity" (Templeton et al., May 2024) was a landmark result: applying sparse autoencoders to Claude 3 Sonnet, a production-scale model, and extracting 34 million interpretable features.

### Key Results

**Scale**: Trained SAEs with up to 34 million features on the middle-layer residual stream of Claude 3 Sonnet. Previous work was limited to toy models or 1-layer networks.

**Feature diversity**: Found features spanning multiple levels of abstraction:

```
Concrete:    "Golden Gate Bridge", "DNA double helix", "Python list comprehension"
Abstract:    "mathematical proof structure", "sycophantic tone", "uncertainty"
Multilingual: Features that activate for the same concept across languages
Multimodal:  Features that activate for both text and images of the same concept
```

**Safety-relevant features**: Discovered features related to:
- Deceptive behavior and hidden motives
- Bias and discrimination
- Dangerous information (weapons, malware)
- Sycophancy (telling users what they want to hear)

These features could potentially be monitored during deployment to detect concerning model behavior.

**Feature steering ("Golden Gate Claude")**: By clamping the Golden Gate Bridge feature to 10x its maximum activation during inference, Anthropic created a version of Claude that was obsessed with the Golden Gate Bridge — bringing it up in every conversation, claiming to be the bridge, etc. Anthropic released this publicly for a limited time as a demonstration.

This proves features are **causally meaningful**, not just correlations. Modifying a feature's activation directly changes the model's behavior in the expected direction.

### Limitations

- SAEs have significant reconstruction error — they don't capture everything
- Training SAEs at scale is extremely expensive (comparable to training the model itself)
- 34 million features is still potentially far fewer than the model actually uses
- We don't know how features compose — individual features are interpretable, but their interactions during computation are not yet understood
- Feature steering is imprecise — clamping one feature has unintended side effects

## Open Problems

Mechanistic interpretability has made real progress, but fundamental challenges remain. A comprehensive survey (Bereska & Rieck, "Open Problems in Mechanistic Interpretability", Jan 2025) catalogs the field's biggest unsolved problems:

**No rigorous definition of "feature"**: The field talks about features constantly, but there's no formal, agreed-upon definition. When is a direction in activation space a "real" feature vs. an artifact of the analysis method? Toy models use ground-truth features, but real models don't come with labels.

**Superposition remains unsolved**: SAEs are the best tool we have, but they have high reconstruction error, are expensive to train, and may miss features that aren't well-approximated by sparse linear combinations. The relationship between SAE features and the model's "true" features (if such a thing exists) is unclear.

**Circuits don't scale**: The IOI circuit (26 heads) took months of expert effort to find. Modern models have thousands of layers and heads. No one has reverse-engineered a circuit in a model larger than GPT-2 small. Automated circuit discovery methods exist but produce noisy results.

**Computation in superposition**: We understand how models store features in superposition. We don't understand how they compute on superposed representations. If feature A and feature B share dimensions, how does an MLP layer operate on A without corrupting B?

**Interpretability illusions**: Convincing-seeming interpretations can be wrong. A feature might appear monosemantic on the examples you check but be polysemantic on a broader distribution. Validation is hard — there's no ground truth for what a real model "should" be representing.

**Connecting features to behavior**: Even if we can identify individual features, connecting them to end-to-end model behavior (e.g., "why did the model output this specific sentence?") requires understanding feature interactions across all layers — a combinatorial explosion.

**Scaling to frontier models**: Most interpretability research is done on GPT-2 (124M-1.5B parameters). Frontier models are 100-1000x larger. It's unclear whether current techniques will work at that scale, or if qualitatively different approaches are needed.

**Shifting expert outlook**: Neel Nanda (mechanistic interpretability lead at Google DeepMind) publicly updated his views in September 2025, stating that "the most ambitious vision of mechanistic interpretability I once dreamed of is probably dead" — while noting that more targeted, practical applications (medium-risk, medium-reward) remain promising. MIT Technology Review still named the field a "breakthrough technology for 2026."

## Key Papers

| Paper | Authors | Year | Link |
|-------|---------|------|------|
| A Mathematical Framework for Transformer Circuits | Elhage, Nanda, Olsson et al. | 2021 | [transformer-circuits.pub](https://transformer-circuits.pub/2021/framework/index.html) |
| In-context Learning and Induction Heads | Olsson, Elhage et al. | 2022 | [arxiv.org/abs/2209.11895](https://arxiv.org/abs/2209.11895) |
| Toy Models of Superposition | Elhage, Hume, Olsson et al. | 2022 | [arxiv.org/abs/2209.10652](https://arxiv.org/abs/2209.10652) |
| Interpretability in the Wild: IOI Circuit | Wang, Variengien et al. | 2022 | [arxiv.org/abs/2211.00593](https://arxiv.org/abs/2211.00593) |
| Locating and Editing Factual Associations in GPT | Meng, Bau, Mitchell, Finn | 2022 | [arxiv.org/abs/2202.05262](https://arxiv.org/abs/2202.05262) |
| Towards Monosemanticity | Bricken, Templeton et al. | 2023 | [transformer-circuits.pub](https://transformer-circuits.pub/2023/monosemantic-features) |
| Sparse Autoencoders Find Interpretable Features | Cunningham, Ewart et al. | 2023 | [arxiv.org/abs/2309.08600](https://arxiv.org/abs/2309.08600) |
| Scaling Monosemanticity | Templeton et al. | 2024 | [transformer-circuits.pub](https://transformer-circuits.pub/2024/scaling-monosemanticity/) |
| Eliciting Latent Predictions with the Tuned Lens | Belrose, Furman et al. | 2023 | [arxiv.org/abs/2303.08112](https://arxiv.org/abs/2303.08112) |
| Scaling and Evaluating Sparse Autoencoders | Gao, Dupré la Tour et al. (OpenAI) | 2024 | [cdn.openai.com](https://cdn.openai.com/papers/sparse-autoencoders.pdf) |
| Open Problems in Mechanistic Interpretability | Bereska, Rieck | 2025 | [arxiv.org/abs/2501.16496](https://arxiv.org/abs/2501.16496) |

**Getting started**: Install TransformerLens (`pip install transformer-lens`), load GPT-2 small, and work through the exercises in Section 9. The [TransformerLens documentation](https://transformerlensorg.github.io/TransformerLens/) includes a comprehensive demo notebook that covers most of the techniques described here.
