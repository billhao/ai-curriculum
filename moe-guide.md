# Mixture of Experts (MoE) for LLMs

A guide to understanding MoE architectures — how they decouple model capacity from compute cost, and why they power the most efficient frontier models today.

## Background

**Original concept**: [Adaptive Mixtures of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf) (Jacobs, Jordan, Nowlan, Hinton, 1991) — proposed the idea of multiple "expert" networks, each specializing in different regions of the input space, combined by a gating network. This was a general neural network technique, not specific to language models.

**Research lineage** — MoE in modern LLMs builds on a chain of work:

1. **Sparsely-Gated MoE** (Shazeer et al., Google, 2017) — [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538). The breakthrough paper that made MoE practical for deep learning at scale. Introduced sparsity (only top-k experts activated per token), a learnable gating network, and load balancing via noise injection. Scaled to 137B parameters with only modest compute overhead. Authors include Noam Shazeer, Geoffrey Hinton, and Jeff Dean.

2. **GShard** (Lepikhin et al., Google, 2020) — [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668). Scaled MoE to 600B parameters for multilingual translation across 2048 TPUs. Introduced top-2 routing, expert capacity limits, and the auxiliary load-balancing loss that became standard. First demonstration of MoE working at truly massive scale.

3. **Switch Transformer** (Fedus et al., Google, 2021) — [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961). Simplified MoE by switching to top-1 routing (one expert per token). Scaled to 1.6T parameters. Showed that simpler routing works — reducing routing computation while maintaining quality. Achieved 7x speedup over T5-Base with 64 experts.

4. **Mixtral 8x7B** (Mistral AI, 2023) — [Mixtral of Experts](https://arxiv.org/abs/2401.04088). First high-quality open-source MoE LLM. 8 experts per layer, top-2 routing. 47B total parameters, 13B active per token. Matched or exceeded LLaMA 2 70B (a dense model 5x larger in active compute). Proved MoE works for general-purpose LLMs, not just translation.

5. **DeepSeekMoE / DeepSeek-V2 / V3** (DeepSeek, 2024) — Introduced fine-grained experts and shared experts, pushing MoE efficiency further. DeepSeek-V3's auxiliary-loss-free load balancing became a key innovation. These models demonstrated that MoE can match frontier dense models at a fraction of the training cost.

**The key contribution of MoE**: You can scale model capacity (total parameters) without proportionally scaling compute per token. A 671B parameter model can run inference using only 37B parameters per token — getting the knowledge capacity of the larger model at the compute cost of the smaller one.

## What Problem Does MoE Solve?

Dense transformers (like the GPT-2 you trained) activate every parameter for every token. If you want more capacity, you add more parameters, and every token pays the full compute cost.

```
Dense Model Scaling:

  Parameters:   1B ─────── 7B ─────── 70B ──────── 700B
  Compute/token: 1x ─────── 7x ─────── 70x ──────── 700x

  Linear relationship: more capacity = more compute per token
```

This is wasteful. Not every token needs every parameter. The word "the" doesn't require the same processing as a complex math expression. Different tokens benefit from different types of computation.

MoE breaks this coupling:

```
MoE Scaling:

  Total params:  47B ────── 236B ────── 671B
  Active params: 13B ────── 21B ─────── 37B
  Compute/token: ~13x ───── ~21x ────── ~37x

  Sublinear: 14x more capacity, only 3x more compute
```

The insight: replace each dense FFN layer with multiple smaller FFN "experts," and let a router decide which experts to use for each token. Most experts stay idle for any given token — **sparse activation**.

## Key Terms

**Expert**: A standard FFN (feed-forward network) sub-layer — typically two linear transformations with a nonlinearity. In a dense transformer, there's one FFN per layer. In MoE, there are N FFNs (experts) per layer. Each expert has the same architecture but different learned weights, so each specializes in different aspects of the input.

**Router / Gating network**: A small learned linear layer that takes a token's hidden state and produces a score for each expert. These scores determine which experts process the token. It's just a matrix multiply: `scores = h @ W_gate` where `W_gate` has shape `(d_model, num_experts)`.

**Top-k routing**: Only the k highest-scoring experts are activated for each token. k is typically 1 (Switch Transformer) or 2 (GShard, Mixtral). The remaining experts are skipped entirely — their compute is never performed.

**Expert capacity**: The maximum number of tokens an expert can process in one batch. Calculated as `(total_tokens / num_experts) * capacity_factor`. Tokens routed to an already-full expert are dropped (passed through via residual connection without expert processing). The `capacity_factor` is typically 1.0-1.5.

**Load balancing loss / Auxiliary loss**: An additional loss term added to the training objective that penalizes uneven expert utilization. Without it, the router tends to collapse — sending most tokens to a few "favorite" experts while others go unused.

**Expert collapse**: A failure mode where the router learns to always select the same few experts, effectively reducing the MoE layer to a smaller dense layer. The unused experts never receive gradients and never improve, creating a self-reinforcing cycle.

**Sparse activation**: The defining property of MoE — only a fraction of total parameters participate in each forward pass. A model with 671B total parameters might activate only 37B per token.

## Architecture Deep Dive

### Where MoE Fits in the Transformer

MoE replaces the FFN layers, not the attention layers. Each transformer block still has a standard multi-head attention layer — only the FFN is swapped:

```
Dense Transformer Block:              MoE Transformer Block:

┌──────────────────────┐              ┌──────────────────────────────┐
│   Input (hidden h)   │              │      Input (hidden h)        │
├──────────────────────┤              ├──────────────────────────────┤
│  Multi-Head Attention│              │   Multi-Head Attention       │
│  (unchanged)         │              │   (unchanged)                │
├──────────────────────┤              ├──────────────────────────────┤
│  LayerNorm + Residual│              │   LayerNorm + Residual       │
├──────────────────────┤              ├──────────────────────────────┤
│                      │              │ ┌──────┐                     │
│   FFN                │              │ │Router│──► expert scores    │
│   (all tokens use    │              │ └──┬───┘                     │
│    same weights)     │              │    │    ┌────┐ ┌────┐ ┌────┐│
│                      │              │    ├───►│ E1 │ │ E2 │ │ E3 ││
│                      │              │    │    └────┘ └────┘ └────┘│
│                      │              │    │    ┌────┐ ┌────┐ ┌────┐│
│                      │              │    └───►│ E4 │ │ E5 │ │ E6 ││
│                      │              │         └────┘ └────┘ └────┘│
│                      │              │   (only top-k activated)     │
├──────────────────────┤              ├──────────────────────────────┤
│  LayerNorm + Residual│              │   LayerNorm + Residual       │
├──────────────────────┤              ├──────────────────────────────┤
│       Output         │              │         Output               │
└──────────────────────┘              └──────────────────────────────┘
```

Attention is global — every token attends to every other token — so MoE doesn't apply there. The FFN operates independently per token, making it a natural fit for conditional computation: different tokens can use different FFN weights.

### Token-Level Routing

Routing happens independently for each token position. In a batch of sequences, every token gets its own expert assignment:

```
Sequence: "The cat sat on the mat"
           ─── ─── ─── ── ─── ───
Token:      1   2   3   4   5   6

Router assignments (top-2, 8 experts):
  Token 1 "The"  → Expert 3, Expert 7
  Token 2 "cat"  → Expert 1, Expert 5
  Token 3 "sat"  → Expert 2, Expert 5
  Token 4 "on"   → Expert 3, Expert 7
  Token 5 "the"  → Expert 3, Expert 7
  Token 6 "mat"  → Expert 1, Expert 4

Notice: Expert 3 gets 3 tokens, Expert 6 gets 0 tokens
        → This imbalance is what load balancing addresses
```

### Data Flow Through an MoE Layer

```
                      token hidden state h
                             │
                             ▼
                     ┌───────────────┐
                     │    Router     │  h @ W_gate → scores (N experts)
                     │  (linear +   │  softmax → probabilities
                     │   softmax)   │  top-k → selected experts + weights
                     └───────┬───────┘
                             │
                    ┌────────┴────────┐
                    │  top-k select   │
                    └────┬───────┬────┘
                         │       │
                    ┌────▼──┐ ┌──▼────┐
                    │Expert │ │Expert │  (only these 2 run)
                    │  i    │ │  j    │
                    └────┬──┘ └──┬────┘
                         │       │
                    ┌────▼───────▼────┐
                    │ Weighted sum:   │  g_i * E_i(h) + g_j * E_j(h)
                    │ using gate      │
                    │ probabilities   │
                    └────────┬────────┘
                             │
                             ▼
                        MoE output
                    (added to residual)
```

The output is a weighted combination of the selected experts' outputs, where the weights are the gate probabilities (after top-k selection and renormalization).

## The Router / Gating Mechanism

### Mathematical Formulation

Given a token's hidden state `h` (shape: `d_model`), the router computes:

```
Step 1: Compute raw scores
  s = h · W_gate          W_gate shape: (d_model, N)
                          s shape: (N,)  — one score per expert

Step 2: Softmax to get probabilities
  p_i = exp(s_i) / Σ_j exp(s_j)    for i = 1..N

Step 3: Select top-k experts
  Selected = top-k(p)    — indices of k highest probabilities

Step 4: Renormalize selected gate weights
  g_i = p_i / Σ_{j ∈ Selected} p_j    (weights sum to 1)

Step 5: Compute output
  output = Σ_{i ∈ Selected} g_i · Expert_i(h)
```

### Numerical Walkthrough

Let's trace one token through an MoE layer with 4 experts and top-2 routing.

**Setup**:
```
d_model = 768
num_experts = 4
top_k = 2

h = [0.5, -0.3, 0.8, ...]   (768-dim hidden state for one token)
W_gate = 768 × 4 matrix      (router parameters — learned during training)
```

**Step 1 — Router scores**:
```
s = h @ W_gate = [2.1, 0.3, -0.5, 1.8]     (4 raw scores)
```

**Step 2 — Softmax**:
```
exp(s) = [exp(2.1), exp(0.3), exp(-0.5), exp(1.8)]
       = [8.17, 1.35, 0.61, 6.05]

sum = 8.17 + 1.35 + 0.61 + 6.05 = 16.18

p = [8.17/16.18, 1.35/16.18, 0.61/16.18, 6.05/16.18]
  = [0.505, 0.083, 0.038, 0.374]
```

Expert 0 gets probability 0.505, Expert 3 gets 0.374. The router is most confident about these two.

**Step 3 — Top-2 selection**:
```
Selected experts: {0, 3}     (highest two probabilities)
Selected probs:   {0.505, 0.374}
```

Experts 1 and 2 are not activated — their FFN computation is skipped entirely.

**Step 4 — Renormalize gate weights**:
```
sum_selected = 0.505 + 0.374 = 0.879

g_0 = 0.505 / 0.879 = 0.574
g_3 = 0.374 / 0.879 = 0.426
```

These weights sum to 1.0 and determine how much each expert's output contributes.

**Step 5 — Compute expert outputs and combine**:
```
E_0(h) = FFN_0(h) = [1.2, -0.4, 0.9, ...]    (768-dim)
E_3(h) = FFN_3(h) = [0.3, 0.7, -0.2, ...]    (768-dim)

output = 0.574 * E_0(h) + 0.426 * E_3(h)
       = 0.574 * [1.2, -0.4, 0.9, ...] + 0.426 * [0.3, 0.7, -0.2, ...]
       = [0.689+0.128, -0.230+0.298, 0.517-0.085, ...]
       = [0.817, 0.068, 0.432, ...]            (768-dim)
```

This output replaces what a single dense FFN would have produced.

**Parameter count comparison** (for this one layer):
```
Dense FFN:   768 × 3072 × 2 = 4.7M parameters   (all activated)
MoE (4 exp): 4 × 4.7M = 18.8M total              (9.4M activated = 2 experts)
MoE (4 exp): + 768 × 4 = 3K router params         (negligible)
```

4x more total parameters, but only 2x the compute per token (top-2). The remaining 2 experts are free capacity that other tokens can use.

## Load Balancing

### Why Naive Routing Fails

Without intervention, the router converges to a degenerate solution: it sends most tokens to just 1-2 "favorite" experts. This is a positive feedback loop:

```
Expert 3 slightly better initially
          │
          ▼
Router sends more tokens to Expert 3
          │
          ▼
Expert 3 gets more gradient updates → improves faster
          │
          ▼
Router sends even MORE tokens to Expert 3
          │
          ▼
Other experts starved of gradients → never improve
          │
          ▼
Expert collapse: only Expert 3 is used
Model effectively becomes a smaller dense model
```

This wastes all the extra parameters. You paid for 8 experts but only 1 is doing any work.

### The Load Balancing Auxiliary Loss

The Switch Transformer introduced the standard auxiliary loss. The idea: penalize the router when some experts get too many tokens and others get too few.

**Formulation** (from Switch Transformer):

```
L_aux = α · N · Σ_{i=1}^{N} f_i · P_i
```

Where:
- `N` = number of experts
- `f_i` = fraction of tokens actually routed to expert i in this batch
- `P_i` = average router probability assigned to expert i across all tokens
- `α` = scaling coefficient (typically 0.01)

**Numerical example** with 4 experts processing 8 tokens:

```
Token routing decisions:  [E0, E0, E0, E0, E1, E1, E2, E3]

f (fraction dispatched):
  f_0 = 4/8 = 0.500    (Expert 0 got 4 tokens — overloaded)
  f_1 = 2/8 = 0.250
  f_2 = 1/8 = 0.125
  f_3 = 1/8 = 0.125

P (avg router probability):
  P_0 = avg of router probs for Expert 0 across all 8 tokens = 0.45
  P_1 = 0.25
  P_2 = 0.18
  P_3 = 0.12

L_aux = 0.01 × 4 × (0.500×0.45 + 0.250×0.25 + 0.125×0.18 + 0.125×0.12)
      = 0.04 × (0.225 + 0.0625 + 0.0225 + 0.015)
      = 0.04 × 0.325
      = 0.013
```

**Why this formula works**: The product `f_i · P_i` is large when an expert both receives many tokens (high `f_i`) AND has high router confidence (high `P_i`). Minimizing this product pushes toward uniform distribution. The minimum of `Σ f_i · P_i` subject to `Σ f_i = 1` and `Σ P_i = 1` occurs when `f_i = P_i = 1/N` for all i — perfect balance.

**Perfectly balanced case**:
```
f = [0.25, 0.25, 0.25, 0.25]
P = [0.25, 0.25, 0.25, 0.25]

L_aux = 0.01 × 4 × 4 × (0.25 × 0.25)
      = 0.04 × 0.25
      = 0.01
```

**Total training loss**:
```
L_total = L_task + L_aux
        = cross_entropy_loss + α · N · Σ f_i · P_i
```

The auxiliary loss is differentiable through the router probabilities `P_i` (not through the discrete routing decisions `f_i`, which are treated as constants). This gradient signal adjusts the router weights to spread tokens more evenly.

### Trade-off: α Selection

```
α too small (10^-5):  Experts collapse, load severely imbalanced
α = 10^-2 (typical):  Good balance, minimal impact on task loss
α too large (10^-1):  Load is balanced but task performance degrades
                      (balancing loss dominates the gradient signal)
```

## Training Challenges

### 1. Expert Collapse

The most common failure mode. Despite the auxiliary loss, experts can still collapse if:
- The auxiliary loss coefficient α is too small
- Learning rate is too high early in training (router commits too quickly)
- Too many experts relative to data diversity

**Symptoms**: A few experts receive >50% of tokens. Unused experts have near-zero gradient norms.

**Mitigations**: Auxiliary loss, expert dropout (randomly disable experts during training to force the router to diversify), careful learning rate warmup.

### 2. Training Instability

MoE models are more unstable than dense models, especially at scale. The router's discrete decisions create non-smooth optimization landscapes. Small changes in router weights can cause large shifts in which experts see which tokens.

**Symptoms**: Loss spikes, NaN gradients, sudden performance drops.

**Mitigations**: Lower learning rate, longer warmup, bfloat16 precision (Switch Transformer showed this helps), gradient clipping.

### 3. Communication Overhead in Distributed Training

In multi-GPU setups, experts are typically distributed across devices (expert parallelism). Each token must be routed to the correct device, processed, and the result sent back. This all-to-all communication can become the bottleneck.

```
Expert Parallelism across 4 GPUs:

  GPU 0          GPU 1          GPU 2          GPU 3
┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐
│Expert 0│    │Expert 1│    │Expert 2│    │Expert 3│
│Expert 4│    │Expert 5│    │Expert 6│    │Expert 7│
└────┬───┘    └────┬───┘    └────┬───┘    └────┬───┘
     │             │             │             │
     └─────────────┴─────────────┴─────────────┘
                        │
              All-to-all communication:
              Tokens dispatched to correct GPU,
              results gathered back
```

With 256 experts across many GPUs, this communication cost is substantial. Load imbalance makes it worse — if one GPU's experts get more tokens, all other GPUs wait for it.

### 4. Dropped Tokens

When an expert exceeds its capacity, overflow tokens are dropped (passed through via residual connection without expert processing). This means some tokens get degraded processing.

**Mitigations**: Higher capacity factor (wastes some compute on padding), auxiliary loss to balance load, or allowing dynamic capacity (more complex implementation).

## DeepSeek MoE Innovations

DeepSeek introduced several innovations across their MoE papers that significantly improved the architecture.

### DeepSeekMoE (Jan 2024): Fine-Grained Expert Segmentation

**Paper**: [DeepSeekMoE: Towards Ultimate Expert Specialization](https://arxiv.org/abs/2401.06066)

**Problem**: Standard MoE uses a small number of large experts (e.g., Mixtral: 8 experts, activate 2). Each expert is a full-size FFN, so it handles a broad mix of knowledge — limiting specialization.

**Solution**: Instead of N large experts activating K, split each expert into m smaller pieces — creating mN fine-grained experts and activating mK of them. Total compute stays the same, but the combination is more flexible.

```
Standard MoE (Mixtral-style):           DeepSeekMoE:
8 experts, activate 2                   64 experts, activate 8
                                        (each expert 1/8 the size)

┌──────┐ ┌──────┐                       ┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐
│      │ │      │  2 large              │  ││  ││  ││  ││  ││  │
│ E_1  │ │ E_2  │  experts              │E1││E5││E9││17││32││48│
│      │ │      │  activated            │  ││  ││  ││  ││  ││  │
└──────┘ └──────┘                       └──┘└──┘└──┘└──┘└──┘└──┘
                                        + 2 more = 8 active experts

Combinations: C(8,2) = 28               Combinations: C(64,8) = 4B+
```

More fine-grained experts = exponentially more possible combinations = more flexible specialization. Each small expert can learn a narrow, specific skill.

### DeepSeekMoE: Shared Expert Isolation

**Problem**: In standard MoE, some knowledge (common syntax, frequent patterns) is needed by all tokens but gets redundantly encoded across multiple routed experts.

**Solution**: Designate K_s experts as **shared experts** that process every token, always. The remaining experts are routed normally. Shared experts capture universal knowledge; routed experts specialize.

```
Standard MoE:                          DeepSeekMoE with Shared Experts:

All experts routed:                    ┌─────────────────────────────┐
                                       │ Shared Experts (always on)  │
┌────┐┌────┐┌────┐┌────┐              │ Process ALL tokens          │
│ E1 ││ E2 ││ E3 ││ E4 │              │ Common knowledge: syntax,   │
└────┘└────┘└────┘└────┘              │ function words, patterns    │
(router picks top-k)                   └────────────┬────────────────┘
                                                    │
                                       ┌────────────▼────────────────┐
                                       │ Routed Experts (top-k)      │
                                       │ Specialized knowledge:      │
                                       │ domain-specific, rare       │
                                       └─────────────────────────────┘

                                       output = shared_out + routed_out
```

### DeepSeek-V2: Architecture at Scale

**Paper**: [DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model](https://arxiv.org/abs/2405.04434)

Applied the DeepSeekMoE design at scale:

```
DeepSeek-V2 Specifications:
─────────────────────────────────────────────
Total parameters:        236B
Active parameters/token: 21B
Layers:                  60
Per MoE layer:           2 shared experts + 160 routed experts
Activated routed:        6 per token
─────────────────────────────────────────────
```

236B total but only 21B active per token — a 11x sparsity ratio. Combined with Multi-head Latent Attention (MLA) for KV cache compression, this made inference dramatically cheaper than comparable dense models.

### DeepSeek-V3: Auxiliary-Loss-Free Load Balancing

**Paper**: [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)

**The problem with auxiliary loss**: The load-balancing loss in Switch Transformer / GShard introduces "interference gradients" — gradients that push toward balance but conflict with the task loss gradient. This hurts model quality. Larger α = better balance but worse performance.

**DeepSeek-V3's solution**: Eliminate the auxiliary loss entirely. Instead, add a learnable **bias term** to each expert's routing score:

```
Standard routing:                  Loss-Free Balancing:

s_i = h · w_i                     s_i = h · w_i + b_i
                                              ▲
                                              │
                                   b_i adjusted dynamically:
                                   - Expert overloaded → decrease b_i
                                   - Expert underloaded → increase b_i
```

The bias `b_i` is NOT a learned parameter updated by gradient descent. It's adjusted by a simple heuristic rule after each training step based on actual load statistics. This means:
- No interference gradients (bias adjustment is outside the computation graph)
- Load balance is maintained by direct intervention
- Task loss gradients are uncontaminated

```
DeepSeek-V3 Specifications:
─────────────────────────────────────────────
Total parameters:        671B
Active parameters/token: 37B
Layers:                  61
Per MoE layer:           1 shared expert + 256 routed experts
Activated routed:        8 per token
Training data:           14.8T tokens
Training cost:           2.788M H800 GPU hours
─────────────────────────────────────────────
```

671B parameters, 37B active — an 18x sparsity ratio. Trained for a fraction of the cost of comparable dense models and achieved frontier performance.

## MoE vs Dense Models

| Aspect | Dense Model | MoE Model |
|--------|-------------|-----------|
| Parameters activated | All | Top-k experts only (5-20% of total) |
| Capacity scaling | Linear with compute | Sublinear — can scale capacity cheaply |
| Training FLOPs | Proportional to params | Proportional to active params |
| Memory (inference) | Load all params | Load ALL params (all experts must be in memory) |
| Memory (training) | Model + optimizer states | Same but for all experts + communication buffers |
| Training stability | More stable | Less stable (routing dynamics, load balancing) |
| Implementation | Standard | Complex (routing, load balancing, expert parallelism) |
| Communication cost | Standard (data/tensor parallel) | Higher (all-to-all for expert parallelism) |
| Example | LLaMA 70B: 70B active | Mixtral 8x7B: 47B total, 13B active |
| Example | GPT-4 (rumored dense) | DeepSeek-V3: 671B total, 37B active |

## Practical Considerations

### Memory: All Experts Must Be Loaded

A common misconception: "MoE only activates 2 of 8 experts, so it needs 1/4 the memory." **Wrong.** All expert weights must reside in memory (GPU VRAM or CPU RAM) because any token could be routed to any expert. The memory footprint is determined by total parameters, not active parameters.

```
Mixtral 8x7B memory footprint:
  47B parameters × 2 bytes (bfloat16) ≈ 94 GB
  Not: 13B × 2 = 26 GB

DeepSeek-V3 memory footprint:
  671B parameters × 2 bytes (bfloat16) ≈ 1.3 TB
  Requires many GPUs just for weight storage
```

MoE saves **compute** (FLOPs per token), not **memory**. This is why MoE models are harder to deploy — you need enough aggregate GPU memory for all experts even though most are idle at any moment.

### Inference Efficiency

MoE shines for **throughput** (tokens per second per dollar) but the latency story is more nuanced:

- **Throughput**: Excellent. Less compute per token means faster processing. Mixtral 8x7B matches LLaMA 70B quality at ~2.5x less compute per token.
- **Latency**: Depends on setup. If all experts fit in GPU memory, latency is similar to a dense model of the active parameter count. If experts are split across GPUs (expert parallelism), communication overhead adds latency.
- **Batch size sensitivity**: At very small batch sizes, the overhead of routing + sparse expert activation can outweigh the compute savings. MoE benefits increase with batch size.

### When MoE Makes Sense

**Good fit**:
- You want frontier model quality at lower training cost
- You have enough aggregate GPU memory for all experts
- High-throughput serving (many concurrent requests)
- Training budget is the bottleneck, not serving infrastructure

**Not ideal**:
- Edge deployment with limited memory (a phone can't hold 47B parameters)
- Single-GPU inference where all experts must fit in one device
- Very small models (overhead of routing isn't worth it below ~1B active params)
- When implementation simplicity matters more than efficiency

### MoE in Practice: Who Uses It

```
Model                    Total    Active   Experts   Top-k  Released
───────────────────────  ───────  ───────  ────────  ─────  ────────
Qwen3.5-397B-A17B        397B     17B      512+1s    10     2026-02
Qwen3.5-122B-A10B        122B     10B      256       10     2026-02
Qwen3.5-35B-A3B          35B      3B       256       10     2026-02
Kimi K2.5 (Moonshot)     1T       32B      384       8      2026-01
Qwen3-Next-80B-A3B       80B      3B       512+1s    10     2025-09
DeepSeek-V3.1            671B     37B      256+1s    8      2025-08
GPT-OSS-120B (OpenAI)    117B     5.1B     128       4      2025-08
Kimi K2 (Moonshot)       1T       32B      384       8      2025-07
DeepSeek-R1-0528         671B     37B      256+1s    8      2025-05
Qwen3-235B-A22B          235B     22B      128       8      2025-04
Qwen3-30B-A3B            30B      3B       128       8      2025-04
Llama 4 Maverick (Meta)  400B     17B      128+1s    1      2025-04
Llama 4 Scout (Meta)     109B     17B      16        1      2025-04
DeepSeek-R1              671B     37B      256+1s    8      2025-01
MiniMax-Text-01          456B     45.9B    32        2      2025-01
DeepSeek-V3              671B     37B      256+1s    8      2024-12
Hunyuan-Large (Tencent)  389B     52B      16+1s     1      2024-11
OLMoE (Ai2)              6.9B     1.3B     64        8      2024-09
Jamba 1.5 Large (AI21)   398B     94B      16        2      2024-08
Jamba 1.5 Mini (AI21)    52B      12B      16        2      2024-08
Phi-3.5-MoE (MS)         42B      6.6B     16        2      2024-08
Qwen2-MoE                57B      14B      64        8      2024-06
DeepSeek-V2              236B     21B      160+2s    6      2024-06
Snowflake Arctic         480B     17B      128       2      2024-04
Grok-1 (xAI)             314B     ~86B     8         2      2024-03
DBRX (Databricks)        132B     36B      16        4      2024-03
Jamba (AI21)             52B      12B      16        2      2024-03
Mixtral 8x22B            141B     39B      8         2      2024-01
Mixtral 8x7B             47B      13B      8         2      2023-12
```

`+2s` / `+1s` = shared experts (always active, in addition to routed experts).
Kimi K2/K2.5 are Moonshot AI's trillion-parameter open-weight models (384 experts, 8 active per token).
Qwen3.5 uses hybrid Gated DeltaNet + Gated Attention architecture (3:1 ratio).
GPT-OSS-120B is OpenAI's first open-weight MoE (Apache 2.0, runs on single 80GB GPU via MXFP4).
DeepSeek-R1/R1-0528 share the same MoE architecture as DeepSeek-V3 (fine-tuned via RL for reasoning).
Llama 4 uses top-1 routed expert + 1 shared expert.

## Key Papers

1. **Adaptive Mixtures of Local Experts** — Jacobs, Jordan, Nowlan, Hinton (1991). The original MoE concept. [Link](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf)

2. **Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer** — Shazeer et al. (2017). Made MoE practical for deep learning. Introduced sparsity and trainable gating. [arxiv:1701.06538](https://arxiv.org/abs/1701.06538)

3. **GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding** — Lepikhin et al. (2020). Scaled MoE to 600B parameters with top-2 routing and auxiliary loss. [arxiv:2006.16668](https://arxiv.org/abs/2006.16668)

4. **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity** — Fedus, Zoph, Shazeer (2021). Simplified to top-1 routing. Standard load balancing loss formula. [arxiv:2101.03961](https://arxiv.org/abs/2101.03961)

5. **Mixtral of Experts** — Mistral AI (2024). Open-source 8x7B MoE. Proved MoE works for general-purpose LLMs. [arxiv:2401.04088](https://arxiv.org/abs/2401.04088)

6. **DeepSeekMoE: Towards Ultimate Expert Specialization** — DeepSeek (2024). Fine-grained experts + shared experts. [arxiv:2401.06066](https://arxiv.org/abs/2401.06066)

7. **DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model** — DeepSeek (2024). 236B/21B model with MLA + DeepSeekMoE at scale. [arxiv:2405.04434](https://arxiv.org/abs/2405.04434)

8. **Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts** — DeepSeek (2024). Bias-based load balancing without auxiliary loss interference. [arxiv:2408.15664](https://arxiv.org/abs/2408.15664)

9. **DeepSeek-V3 Technical Report** — DeepSeek (2024). 671B/37B model applying all innovations at scale. [arxiv:2412.19437](https://arxiv.org/abs/2412.19437)
