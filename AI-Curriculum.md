TODO
*   Production inference pipeline. what's processed in parallel and sequentially. explain prefill and decode steps

---
### PyTorch & Deep Learning Fundamentals

Understanding neural networks and PyTorch is the foundation for everything else. You need to grasp backpropagation, gradient descent, and tensor operations before diving into transformers.

**Projects:**

*   ✅[Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI)
    *   Quiz
*   Complete [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) playlist - implement each exercise

**Reading:**

*   [Backpropagation lecture notes](http://cs231n.github.io/optimization-2/)

---
### Tokenization

Tokenization converts text to numbers that models can process. The choice of tokenizer and vocabulary size significantly impacts model performance, training efficiency, and multilingual capabilities.

**Projects:**

*   Train a BPE tokenizer from scratch using [minbpe](https://github.com/karpathy/minbpe)
*   Compare tokenization of the same text across GPT-2, Llama, and multilingual tokenizers
*   Analyze how vocabulary size affects your GPT-2 124M's performance

**Reading:**

*   [SentencePiece paper](https://arxiv.org/abs/1808.06226)
*   [BPE paper (Neural Machine Translation)](https://arxiv.org/abs/1508.07909)
*   [Tokenizer comparison blog](https://huggingface.co/docs/transformers/tokenizer_summary)

---
### ✅ Transformer Architecture Deep Dive

Transformers are the architecture behind all modern LLMs. Understanding attention mechanisms, positional encoding, and the encoder-decoder structure is essential for working with or modifying these models.

**Projects:**
*   ✅ Implement a transformer from scratch following [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
*   Complete [BuildAnLLM](https://github.com/jammastergirish/BuildAnLLM)

**Reading:**
*   ✅ [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
*   ✅ [Transformer论文逐段精读](https://www.youtube.com/watch?v=nzqlFIcCSWQ)
*    [Introduction to Transformers w/ Andrej Karpathy](https://www.youtube.com/watch?v=XfpMkf4rD6E)
*   [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

---

### ✅ Training & Pre-training

Pre-training is where models learn general language understanding from massive datasets. Understanding scaling laws and training dynamics helps you make informed decisions about compute, data, and model size tradeoffs.

**Projects:**

*   ✅[Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU&t=1900s)
    *   Started 12/31/25
    *   Finished 1/8/26
*   Continue pre-training your GPT-2 124M on domain-specific data (e.g., code, scientific papers)
*   Experiment with learning rate schedules: cosine decay vs linear warmup, different warmup ratios

**Datasets:**

*   [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) - high-quality educational web data
*   [The Pile](https://pile.eleuther.ai/) - diverse 800GB dataset
*   [RedPajama](https://github.com/togethercomputer/RedPajama-Data) - open reproduction of LLaMA training data

**Reading:**

*   [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
*   [Training Compute-Optimal LLMs (Chinchilla)](https://arxiv.org/abs/2203.15556)
*   [Llama 3 Technical Report](https://arxiv.org/abs/2407.21783)

---

### Parameter-Efficient Fine-tuning

Full fine-tuning requires updating all model weights, which is expensive. Parameter-efficient methods like LoRA and QLoRA let you adapt large models by training only a small number of additional parameters.

**Projects:**

*   Apply LoRA to your GPT-2 124M: experiment with rank (r=8, 16, 32) and target modules
*   Compare training speed and memory usage: full fine-tuning vs LoRA vs QLoRA on same task
*   Fine-tune Llama 3.1 8B with QLoRA on a custom dataset (leveraging your H100s)

**Reading:**

*   [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
*   [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)

---

### Post-training Pipeline

Post-training transforms a base model into an assistant. The pipeline is: **SFT** (teach instruction-following) → **Preference Optimization** (align with human preferences) → optionally **RLHF** (reinforce with reward model). This is what makes the difference between GPT-3 and ChatGPT.

**Projects:**

_Step 1: Supervised Fine-Tuning (SFT)_

*   SFT your GPT-2 124M on instruction data using [TRL's SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
*   Format: `<|user|>question<|assistant|>answer` - teach it to follow instructions

_Step 2: Preference Optimization_

*   Create preference pairs from your SFT model's outputs (chosen vs rejected)
*   Apply DPO using [TRL library](https://huggingface.co/docs/trl) to align the SFT model
*   Compare before/after on safety and helpfulness prompts

_Step 3: RLHF (Optional - more complex)_

*   Train a reward model on preference data
*   Use PPO to optimize against the reward model

**Datasets:**

*   [Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) - good starter for SFT
*   [OpenAssistant (oasst1)](https://huggingface.co/datasets/OpenAssistant/oasst1) - 160k conversation trees
*   [SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca) - 500k high-quality instructions
*   [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) - preference data for DPO
*   [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) - 64k with GPT-4 preference labels

**Reading:**

*   [Training language models to follow instructions (InstructGPT)](https://arxiv.org/abs/2203.02155)
*   [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290)
*   [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
*   [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)

---

### Evaluation & Benchmarks

You can't improve what you can't measure. Evaluation tells you if your training actually worked and how your model compares to others.

**Projects:**

*   Evaluate your GPT-2 124M (base, SFT, DPO versions) using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
*   Track perplexity on held-out data throughout training
*   Run benchmarks: HellaSwag (commonsense), ARC (reasoning), TruthfulQA (factuality)
*   Create a simple eval suite for your specific use case

**Reading:**

*   [MMLU benchmark paper](https://arxiv.org/abs/2009.03300)
*   [HumanEval (code)](https://arxiv.org/abs/2107.03374)
*   [Holistic Evaluation of Language Models (HELM)](https://arxiv.org/abs/2211.09110)

---

### Interpretability & Mechanistic Understanding

Interpretability helps us understand what's happening inside neural networks. As models become more powerful, understanding their internal representations and circuits becomes critical for safety and debugging.

**Projects:**

*   Use [TransformerLens](https://github.com/neelnanda-io/TransformerLens) to probe your GPT-2 124M's internals
*   Visualize attention patterns on specific prompts - what is your model attending to?
*   Reproduce induction head finding in your model
*   Compare activations before/after SFT and DPO - what changed?

**Reading:**

*   [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
*   [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
*   [Toward Monosemanticity (Dictionary Learning)](https://transformer-circuits.pub/2023/monosemantic-features/)
*   [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/)

---

### Inference & Deployment

Efficient inference is essential for practical applications. Quantization and optimized serving can reduce costs by 10-100x while maintaining quality, making deployment feasible.

**Projects:**

*   Quantize your post-trained GPT-2 124M to GGUF format and run with [llama.cpp](https://github.com/ggerganov/llama.cpp)
*   Set up a local inference server using [vLLM](https://github.com/vllm-project/vllm)
*   Benchmark: measure tokens/second at FP16, INT8, INT4 quantization levels
*   Deploy with a simple chat interface (Gradio or similar)

**Reading:**

*   [FlashAttention: Fast and Memory-Efficient Attention](https://arxiv.org/abs/2205.14135)
*   [LLM.int8(): 8-bit Matrix Multiplication](https://arxiv.org/abs/2208.07339)
*   [FlashAttention-2](https://arxiv.org/abs/2307.08691)

---

### Advanced Architectures

Modern LLM applications go beyond simple text generation. RAG grounds models in external knowledge, tool use extends their capabilities, and architectures like MoE enable scaling efficiency.

**Projects:**

*   Build a RAG system: your GPT-2 124M + vector database (ChromaDB or FAISS) + retriever
*   Implement tool-use / function calling for your model
*   (Advanced) Experiment with Mixture of Experts on a larger model

**Datasets:**

*   [MS MARCO](https://microsoft.github.io/msmarco/) - passage retrieval benchmark
*   [HotpotQA](https://hotpotqa.github.io/) - multi-hop reasoning with supporting facts
*   [Natural Questions](https://ai.google.com/research/NaturalQuestions) - real Google queries

**Reading:**

*   [Mixture of Experts (Switch Transformer)](https://arxiv.org/abs/2101.03961)
*   [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)
*   [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)
*   [Toolformer](https://arxiv.org/abs/2302.04761)
*   [DeepSeek-V2: MoE architecture](https://arxiv.org/abs/2405.04434)

---

### Distributed Training (Optional)

With 4-8 H100s, you can train much larger models. Distributed training techniques let you scale beyond single-GPU limits.

**Projects:**

*   Train a larger model (1B+ parameters) using PyTorch FSDP
*   Experiment with DeepSpeed ZeRO stages on your cluster
*   Profile memory usage and throughput across different parallelism strategies

**Reading:**

*   [PyTorch FSDP tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
*   [DeepSpeed ZeRO paper](https://arxiv.org/abs/1910.02054)
*   [Megatron-LM (tensor parallelism)](https://arxiv.org/abs/1909.08053)

---

### Capstone Project

Putting it all together demonstrates mastery and creates a portfolio piece. This end-to-end project exercises every skill from the curriculum.

**The Journey of Your GPT-2 124M:**

1.  ✅ Pre-train from scratch (done)
2.  Continue pre-training on domain data
3.  SFT on instruction dataset
4.  Align with DPO
5.  Evaluate on benchmarks
6.  Quantize and deploy
7.  Add RAG for knowledge grounding
8.  Interpretability analysis: what did each stage change?

---

### Key Technical Reports

Essential reading for understanding state-of-the-art models.

*   [Llama 3 Technical Report](https://arxiv.org/abs/2407.21783)
*   [Qwen 2.5 Technical Report](https://arxiv.org/abs/2412.15115)
*   [DeepSeek-V2](https://arxiv.org/abs/2405.04434)
*   [Mistral 7B](https://arxiv.org/abs/2310.06825)

---

### Conferences

*   NeurIPS 2025
*   ICML 2025
*   ICLR 2025
*   ACL 2025