# Reading Backlog

## Interpretability
- interp
  - Mechanistic interpretability entry point: circuits, induction heads, SAEs/dictionary learning; index to per-topic guides.

## Knowledge-Vs-Intelligence
- physics-of-language-models-guide
  - Allen-Zhu/Li controlled science-of-LLMs: knowledge storage vs reasoning skills, 2-bits/param capacity, iGSM hidden planning, Canon layers; ICML 2024 tutorial.

## Models
- model-gemma-4-guide
  - Gemma 4: Google's open models (2B–31B); dual-config attention, K=V sharing, shared KV cache, Per-Layer Embeddings.
- model-kimi-k2.5
  - Kimi K2.5 technical deep-dive: architecture and capabilities, for a nanoGPT background.
- model-kimi-k3-guide
  - Kimi K3: 2.78T/104B MoE, 1M context; hybrid KDA linear attention, Attention Residuals over depth, 896-expert LatentMoE.
- model-deepseek-v4-guide
  - DeepSeek V4: 1.6T-param agentic MoE, 1M context via CSA+HCA hybrid attention, Muon optimizer, FP4 experts, R1 thinking modes.
- model-glm-5.2-guide
  - GLM-5.2: 744B/40B coding MoE; DSA + IndexShare (share lightning-indexer top-k every 4 layers, 2.9× FLOP cut at 1M), deeper MTP.
- model-vibethinker-3b-guide
  - VibeThinker-3B: 3B dense matches 671B–1T on verifiable math via SSP/MGPO/Long2Short/self-distill + CLR; Compression-Coverage Hypothesis.

## Model Architecture
- gated-deltanet-2-guide
  - NVIDIA linear-attention layer; decouples erase/write into channel-wise gates; beats Mamba-2/3, Gated DeltaNet, KDA on long-context retrieval.
- sleep-paradigm-hope-guide
  - Wake/Sleep lifecycle for continual learning: upward Knowledge Seeding distillation + RL Dreaming on Nested Learning's Hope/CMS architecture.
- long-context-guide
  - Long sequences: quadratic wall, RoPE scaling (YaRN/NTK), FlashAttention, ring/sparse attention, KV-cache compression (MLA/GQA).
- flash-attention-4-guide
  - FlashAttention 4: latest exact-attention kernel; Blackwell-optimized IO-aware tiling and scheduling.

## Pretraining
- optimizers-adam-muon-guide
  - From AdamW to Muon: momentum, adaptive LR, and Muon's orthogonalized momentum (Newton-Schulz) for matrix params; used in Kimi K2/DeepSeek V4.

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
- qwen-audio-3-tts-vs-qwen3-tts-guide
  - Head-to-head TTS comparison: open Qwen3-TTS (Apache, 12Hz codec, self-host) vs hosted Qwen-Audio-3.0-TTS (Speech Arena #1); benchmarks, pricing, TCO.

## Agents
- agents-guide
  - LLM agents: lineage, architectures, memory, planning, multi-agent systems, code agents, evaluation, training, failure modes, patterns.

## Autonomous-Research
- robin-guide
  - Robin (FutureHouse): closed-loop multi-agent biology; 2-phase hypothesis gen, BTL-ranked LLM judge, Finch consensus analysis; ripasudil/ABCA1 for dry AMD.
- robin-architecture
  - Code-level walkthrough of the Robin repo: module map, orchestration flow, choix-based BTL ranking, code-vs-paper differences.
- ai-scientist-guide
  - Sakana's The AI Scientist: fully autonomous ML research loop (idea→code via Aider→paper→automated reviewer); ~$15/paper; reviewer near-human.
- popper-falsification-guide
  - Popper (Stanford): domain-general agentic hypothesis validation; sequential e-values/e-processes give any-time Type-I error control.
- ai-co-scientist-guide
  - Google AI Co-Scientist: 6 specialist agents + Supervisor; generate-debate-evolve with online Elo tournament; 3 wet-lab validations.
- data-to-paper-guide
  - data-to-paper (Technion): stepwise multi-agent data→hypothesis→code→full paper with verifiable claim-to-data provenance chains.
- landscape-2025-2026
  - Ranked map of 2025-2026 autonomous-research work: end-to-end systems, hypothesis gen/validation, self-driving labs, benchmarks, critiques.
- nanogpt-speedrun-agent-followups-2026
  - Intology NanoGPT-Bench + Prime Intellect auto-nanogpt (2026): frontier coding agents on the speedrun as open-ended discovery; <10% human progress.
- execution-grounded-auto-research-guide
  - Stanford's constructive answer to the Ideation-Execution Gap: automated executor as reward; evolutionary search beats baselines, RL mode-collapses.
- rsi-proposer-landscape-2026
  - RSI through the proposer lens: what makes LLM hypothesis-generation work, 2026 systems, diversity collapse, co-evolved evaluators, idea-level benchmarks.
- novelty-evaluation-research
  - Quantitative novelty/impact metrics: combinatorial, semantic, disruption, Bayesian surprise; DiscoveryBench/HypoBench/NovBench; Goodhart-resistant closed-loop RSI evaluation.

## Benchmarks
- benchmarks-guide
  - Major 2026 LLM benchmarks: reasoning, code, agents, long-context, multimodal; SOTA scores and contamination caveats.
- arc-agi-2-guide
  - ARC-AGI 2 benchmark: efficient few-shot abstraction on novel tasks; why it resists saturation through 2026.
