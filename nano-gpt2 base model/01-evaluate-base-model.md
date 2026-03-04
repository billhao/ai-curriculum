# Step 1: Evaluate Your Base GPT-2 124M Model

Establish a baseline before any fine-tuning. You'll compare against these numbers after SFT and DPO.

---

## Prerequisites

```bash
# Create a virtual environment (using your existing ~/projects/.venv or new one)
source ~/projects/.venv/bin/activate

# Install lm-evaluation-harness
uv pip install lm-eval

# Install additional dependencies
uv pip install torch transformers datasets accelerate
```

---

## Option A: Evaluate a Hugging Face Model

If your GPT-2 124M is saved in Hugging Face format:

```bash
# Basic evaluation on key benchmarks
lm_eval --model hf \
    --model_args pretrained=/path/to/your/gpt2-124m \
    --tasks hellaswag,arc_easy,arc_challenge,piqa,winogrande \
    --batch_size 16 \
    --device cuda:0 \
    --output_path ./eval_results/base_model
```

---

## Option B: Evaluate a Custom Model (nanoGPT style)

If your model is in nanoGPT format, you'll need to wrap it first.

### Step 1: Create a model wrapper

Create a file `eval_wrapper.py`:

```python
"""
Wrapper to evaluate nanoGPT model with lm-evaluation-harness
"""
import torch
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer

def load_nanogpt_to_hf(checkpoint_path, output_path):
    """Convert nanoGPT checkpoint to HuggingFace format"""

    # Load your nanoGPT checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Get model config from checkpoint
    model_args = checkpoint.get('model_args', {})

    # Create HF config (adjust these to match your model)
    config = GPT2Config(
        vocab_size=50257,  # GPT-2 default
        n_positions=1024,   # context length
        n_embd=768,         # embedding dimension
        n_layer=12,         # number of layers
        n_head=12,          # number of attention heads
    )

    # Create HF model and load weights
    model = GPT2LMHeadModel(config)

    # Map nanoGPT weights to HF format
    # This depends on your exact checkpoint structure
    state_dict = checkpoint['model']

    # Remove 'module.' prefix if using DDP
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Load weights (may need key remapping depending on your nanoGPT version)
    model.load_state_dict(state_dict, strict=False)

    # Save in HF format
    model.save_pretrained(output_path)

    # Also save tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.save_pretrained(output_path)

    print(f"Model saved to {output_path}")
    return output_path

if __name__ == "__main__":
    import sys
    checkpoint_path = sys.argv[1]  # e.g., "./out/ckpt.pt"
    output_path = sys.argv[2]      # e.g., "./gpt2-124m-hf"
    load_nanogpt_to_hf(checkpoint_path, output_path)
```

### Step 2: Convert and evaluate

```bash
# Convert your checkpoint
python eval_wrapper.py ./out/ckpt.pt ./gpt2-124m-hf

# Now evaluate
lm_eval --model hf \
    --model_args pretrained=./gpt2-124m-hf \
    --tasks hellaswag,arc_easy,arc_challenge,piqa,winogrande \
    --batch_size 16 \
    --device cuda:0 \
    --output_path ./eval_results/base_model
```

---

## Recommended Benchmarks

| Benchmark | What it measures | Expected GPT-2 124M |
|-----------|------------------|---------------------|
| **hellaswag** | Commonsense reasoning | ~29-31% |
| **arc_easy** | Grade-school science (easy) | ~43-45% |
| **arc_challenge** | Grade-school science (hard) | ~21-23% |
| **piqa** | Physical intuition | ~62-65% |
| **winogrande** | Coreference resolution | ~52-54% |
| **truthfulqa_mc** | Factual accuracy | ~25-30% |

Run all at once:

```bash
lm_eval --model hf \
    --model_args pretrained=./gpt2-124m-hf \
    --tasks hellaswag,arc_easy,arc_challenge,piqa,winogrande,truthfulqa_mc2 \
    --batch_size 16 \
    --device cuda:0 \
    --output_path ./eval_results/base_model \
    --log_samples
```

---

## Measure Perplexity

Perplexity on held-out data is a key metric. Lower is better.

Create `measure_perplexity.py`:

```python
"""
Measure perplexity on WikiText-2 test set
"""
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm
import math

def calculate_perplexity(model_path, device="cuda:0"):
    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model.eval()

    # Load WikiText-2 test set
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Concatenate all text
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")

    # Calculate perplexity with sliding window
    max_length = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0

    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc

        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100  # Mask already-seen tokens

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc

        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./gpt2-124m-hf"

    ppl = calculate_perplexity(model_path)
    print(f"\n{'='*50}")
    print(f"Perplexity on WikiText-2: {ppl:.2f}")
    print(f"{'='*50}")

    # Reference: GPT-2 124M should get ~29-35 perplexity on WikiText-2
```

Run it:

```bash
python measure_perplexity.py ./gpt2-124m-hf
```

**Expected:** GPT-2 124M typically gets ~29-35 perplexity on WikiText-2.

---

## Record Your Baseline

Create a file to track your results across training stages:

```bash
# Create results tracking file
cat > eval_results/baseline.json << 'EOF'
{
  "model": "GPT-2 124M (nanoGPT)",
  "stage": "base_model",
  "date": "2025-01-10",
  "perplexity_wikitext2": null,
  "benchmarks": {
    "hellaswag": null,
    "arc_easy": null,
    "arc_challenge": null,
    "piqa": null,
    "winogrande": null,
    "truthfulqa_mc2": null
  },
  "notes": "Trained from scratch following Karpathy GPT-2 video"
}
EOF
```

Fill in the values after running evaluations.

---

## Quick Test: Generate Some Text

Before benchmarks, do a quick sanity check:

```python
"""
Quick generation test
"""
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("./gpt2-124m-hf")
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-124m-hf")

prompts = [
    "The meaning of life is",
    "In a shocking turn of events,",
    "The best way to learn programming is",
    "Once upon a time,",
]

for prompt in prompts:
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=50,
        num_return_sequences=1,
        temperature=0.8,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    print(f"\nPrompt: {prompt}")
    print(f"Output: {tokenizer.decode(output[0], skip_special_tokens=True)}")
    print("-" * 50)
```

---

## Summary Checklist

- [ ] Install lm-evaluation-harness
- [ ] Convert model to HuggingFace format (if needed)
- [ ] Run benchmark suite (hellaswag, arc, piqa, winogrande, truthfulqa)
- [ ] Measure perplexity on WikiText-2
- [ ] Quick generation test
- [ ] Record baseline numbers

---

## Next Step

Once you have your baseline numbers, proceed to **Step 2: Supervised Fine-Tuning (SFT)**.

Your model currently just predicts the next token. After SFT, it will actually answer questions and follow instructions.
