# Reading Backlog

## Interpretability
- interp
  - Mechanistic interpretability entry point: circuits, induction heads, SAEs/dictionary learning; index to per-topic guides.

## Models
- model-gemma-4-guide
  - Gemma 4: Google's open models (2B–31B); dual-config attention, K=V sharing, shared KV cache, Per-Layer Embeddings.
- model-kimi-k2.5
  - Kimi K2.5 technical deep-dive: architecture and capabilities, for a nanoGPT background.
- model-deepseek-v4-guide
  - DeepSeek V4: 1.6T-param agentic MoE, 1M context via CSA+HCA hybrid attention, Muon optimizer, FP4 experts, R1 thinking modes.

## Model Architecture
- gated-deltanet-2-guide
  - NVIDIA linear-attention layer; decouples erase/write into channel-wise gates; beats Mamba-2/3, Gated DeltaNet, KDA on long-context retrieval.
- long-context-guide
  - Long sequences: quadratic wall, RoPE scaling (YaRN/NTK), FlashAttention, ring/sparse attention, KV-cache compression (MLA/GQA).

## Inference
- inference-optimization-guide
  - Fast, memory-efficient inference: KV caches, quantization, speculative decoding, production serving frameworks (vLLM/SGLang).
- llm-sleep-offline-recurrence-guide
  - "Sleep": N offline recurrent passes consolidate context into SSM fast weights before KV eviction; buys reasoning depth at fixed answer-token latency.
- turboquant-guide
  - Online vector quantization with near-optimal distortion; KV-cache compression and nearest-neighbor search.

## Multimodal
- multimodal-vlm-guide
  - Vision-language models: image tokenization, vision encoders, projection layers, contrastive pretraining, visual instruction tuning (GPT-4o, Gemini, LLaVA).

## Agents
- agents-guide
  - LLM agents: lineage, architectures, memory, planning, multi-agent systems, code agents, evaluation, training, failure modes, patterns.

## Benchmarks
- benchmarks-guide
  - Major 2026 LLM benchmarks: reasoning, code, agents, long-context, multimodal; SOTA scores and contamination caveats.
- arc-agi-2-guide
  - ARC-AGI 2 benchmark: efficient few-shot abstraction on novel tasks; why it resists saturation through 2026.
