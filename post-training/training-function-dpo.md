The DPO training function follows the same structure as SFT.
Changes from SFT are marked with `← DPO`.


- Preps
    - Set training parameters like learning rate, β, etc          ← β is new
    - Create tokenizer
    - Create model (policy π_θ) from SFT checkpoint               ← not random init
    - Create reference model (π_ref) = frozen copy of π_θ         ← DPO: second model
    - Create optimizer like AdamW (only for π_θ)
    - Create train/val dataloaders (preference pairs dataset)      ← DPO: chosen + rejected per sample
- [Optional] Precompute π_ref log-probs for all training data     ← DPO: optimization
- Training loop - # steps/epochs
    - Training
        - Optimizer.zero_grad()
        - Micro-step for gradient accumulation
            - Get batch of (prompt, chosen, rejected) from train dataloader
            - Forward pass π_θ on prompt+chosen  → log_p_chosen_θ     ← DPO: 4 forward passes
            - Forward pass π_θ on prompt+rejected → log_p_rejected_θ       instead of 1
            - Forward pass π_ref on prompt+chosen  → log_p_chosen_ref
            - Forward pass π_ref on prompt+rejected → log_p_rejected_ref
              (or look up precomputed π_ref values)
            - Compute DPO loss:                                        ← DPO: replaces cross-entropy
                chosen_logr  = log_p_chosen_θ - log_p_chosen_ref
                rejected_logr = log_p_rejected_θ - log_p_rejected_ref
                margin = β * (chosen_logr - rejected_logr)
                loss = -log(sigmoid(margin))
            - loss.backward()                                          (only π_θ gets gradients)
            - Accumulate loss
        - Clip gradient norm
        - Update LR according to schedule
        - Optimizer.step()
    - Validation (every some steps)
        - Same 4 forward passes on val preference pairs
        - Compute DPO loss on val set (torch.no_grad())
        - Compute rewards/margins and rewards/accuracies              ← DPO: new metrics
    - Sample (every some steps)
        - Same as SFT (autoregressive generation from π_θ)
    - Checkpointing (every some steps)
        - Save only π_θ (π_ref never changes)


## Key Differences from SFT

| | SFT | DPO |
|--|-----|-----|
| **Model init** | Random or pretrained | SFT checkpoint |
| **Models in memory** | 1 | 2 (π_θ + frozen π_ref) |
| **Data format** | (prompt, response) | (prompt, chosen, rejected) |
| **Forward passes per sample** | 1 | 4 (or 2 if π_ref precomputed) |
| **Loss** | Cross-entropy on target tokens | `-log(σ(β × margin))` on preference pair |
| **What gets gradients** | The model | Only π_θ, never π_ref |
| **Key metrics** | loss, token accuracy | loss, rewards/margins, rewards/accuracies |
| **Epochs** | 1-3 | Usually 1 |
| **Learning rate** | ~1e-5 | ~5e-6 (lower) |

The overall loop structure is identical. The difference is **what goes into the model** (preference pairs instead of single responses) and **what comes out** (log-prob ratios instead of cross-entropy loss).
