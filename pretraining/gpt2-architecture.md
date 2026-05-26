# GPT-2 124M Architecture

```
  Input token IDs (B, T)
          │
          ▼
  ┌───────────────┐
  │  wte: Embed   │ token embeddings    (50304, 768)
  │  (50304, 768) │────────────────────────────────────┐
  └───────┬───────┘                                    │
          │                                            │
          ▼                                            │
  ┌───────────────┐                                    │
  │  wpe: Embed   │ position embeddings (1024, 768)    │
  │  (1024, 768)  │                                    │
  └───────┬───────┘                                    │
          │                                            │
          ▼                                            │
       [  +  ] ◄── sum token + position embeddings     │
          │                                            │
          │    ┌───────────────────────────────────┐   │
          │    │         Block (x12)                │   │
          │    │                                    │   │
          ▼    │                                    │   │
  ┌─────────┐  │                                    │   │
  │ LN_1    │  │  LayerNorm                         │   │
  └────┬────┘  │                                    │   │
       │       │                                    │   │
       ▼       │                                    │   │
  ┌─────────┐  │  ┌──────────────────────────────┐  │   │
  │  Attn   │──┼──│ c_attn: (768 → 2304)         │  │   │
  │         │  │  │   split → Q, K, V (768 each) │  │   │
  │         │  │  │ reshape → (B, 12, T, 64)     │  │   │
  │         │  │  │ scaled_dot_product_attention │  │   │
  │         │  │  │ c_proj: (768 → 768)          │  │   │
  └────┬────┘  │  └──────────────────────────────┘  │   │
       │       │                                    │   │
  ───► + ◄─────┼── residual connection              │   │
       │       │                                    │   │
       ▼       │                                    │   │
  ┌─────────┐  │                                    │   │
  │ LN_2    │  │  LayerNorm                         │   │
  └────┬────┘  │                                    │   │
       │       │                                    │   │
       ▼       │                                    │   │
  ┌─────────┐  │  ┌──────────────────────────────┐  │   │
  │  MLP    │──┼──│ c_fc:   (768 → 3072)  4x up  │  │   │
  │         │  │  │ GELU activation              │  │   │
  │         │  │  │ c_proj: (3072 → 768)  4x down│  │   │
  └────┬────┘  │  └──────────────────────────────┘  │   │
       │       │                                    │   │
  ───► + ◄─────┼── residual connection              │   │
       │       │                                    │   │
       │       └────────────────────────────────────┘   │
       │                                                │
       ▼                                                │
  ┌─────────┐                                           │
  │  ln_f   │  final LayerNorm                          │
  └────┬────┘                                           │
       │                                                │
       ▼                                                │
  ┌─────────┐                                           │
  │ lm_head │  (768 → 50304)  shares weights with wte  ◄┘
  └────┬────┘
       │
       ▼
  Logits (B, T, 50304)
       │
       ▼
  Cross-Entropy Loss (with optional SFT mask)


  Config: n_layer=12  n_head=12  n_embd=768  block_size=1024
  Params: 124M unique (wte/lm_head shared)
```
