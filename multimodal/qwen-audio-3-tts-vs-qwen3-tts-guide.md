# Qwen-Audio-3.0-TTS vs Qwen3-TTS

Six months after Alibaba open-sourced the Qwen3-TTS weights, Tongyi Lab shipped a closed, hosted-only successor (Qwen-Audio-3.0-TTS) that tops the human-preference Speech Arena — this guide quantifies exactly what you gain by paying for the API and what you give up versus self-hosting the open model on your own Mac/H800.

> Confidence: **VERY_NEW** for the hosted model. Qwen-Audio-3.0-TTS launched **2026-07-20** (~6 days before this guide); its internals are undisclosed, and its Speech Arena Elo, pricing, and free-quota terms are live-updating and drift day to day. The open Qwen3-TTS side is anchored to its **arXiv technical report (2601.15621)** and GitHub, and is solid. Every number below is traced to a primary source; where sources disagree (notably pricing and the exact Elo), the conflict is flagged inline and the primary/official value is preferred.

## The Core Question: How Much Better?

Start with the honest headline: **there is no primary, apples-to-apples benchmark between the two lines.** They are evaluated on different test sets, one is open with a full technical report and the other is a hosted API with only system-level disclosure. So "how much better" has to be answered capability-by-capability, and some of it cannot be quantified from primary sources at all. Here is the numbers-first verdict:

```
Dimension              Qwen3-TTS (open)            Qwen-Audio-3.0-TTS (hosted)   Verified?
─────────────────────  ──────────────────────────  ────────────────────────────  ─────────
Release / license      2026-01-22, Apache-2.0      2026-07-20, closed/API-only    yes
Weights                Downloadable (HF/GH)         None (Model Studio only)       yes
Flagship size          1.7B (also 0.6B)             Undisclosed                    yes/—
Human-pref (AA Elo)    Not on leaderboard           Plus #1, ≈1234–1236            partial*
Multiling. WER/CER     Seed-TTS 0.77 zh / 1.24 en   16-lang avg 3.87(F)/3.96(P)    yes**
Speaker sim (SIM/SS)   Best-in-class, 10 langs      Plus 82.75 / Flash 80.44 (16)  yes**
First-packet latency   97 ms (0.6B) / 101 ms (1.7B) Flash "300 ms-level"           yes***
Languages              10                           16 (+6)                        yes
Chinese dialects       "multiple" (unspecified)     20 dialect regions             yes
Inline control tags    Voice-design + NL control    86 inline tags                 yes
Voice cloning          3 s reference                3 s + noisy/reverb-robust      yes
Output sample rate     24 kHz                       up to 48 kHz                   yes
Cost                   Self-host (own H800/Mac)     Plus $20/1M, Flash $15/1M      yes****
```

```
*    No primary Speech Arena Elo exists for the OPEN Qwen3-TTS checkpoints, so
     the Elo gap cannot be measured. A secondary "Qwen3-TTS-Flash ≈929" figure
     circulates but is NOT primary-verified — treat any "≈300 Elo gap" claim as
     unconfirmed.
**   These WER/CER and SIM numbers are on DIFFERENT test sets and language sets.
     Qwen3-TTS's 1.24 (Seed-TTS, English only) and 3.0's 3.96 (16-language
     aggregate) are NOT comparable — see the Benchmarks section for why.
***  Open-model latency is measured on the authors' internal vLLM engine;
     the hosted "300 ms" is end-to-end and includes WebSocket + network setup.
**** Official Alibaba Cloud Model Studio rate. Launch press and the Artificial
     Analysis leaderboard listed Plus at ≈$27.59/1M — discrepancy flagged below.
```

**Bottom line.** Qwen-Audio-3.0-TTS-Plus is the top human-preference hosted TTS in the world as of late July 2026, and it adds real, verified capability over the open line: +6 languages, 20 Chinese dialects, 86 expressive inline tags, noisy-reference cloning, and 48 kHz audio. But "better" here means *better on naturalness as judged by blind listeners, plus broader coverage and richer control* — not "lower WER on the benchmarks the open model was tuned for" (it isn't measured that way) and not "lower latency" (the open 12Hz model actually emits its first packet in ~100 ms locally). And you pay for it in per-character billing, undisclosed internals, and the loss of self-hosting. The rest of this guide unpacks each of those trade-offs with the underlying numbers.

## Background

Research lineage — this is a fast-moving line inside Alibaba's speech group (FunAudioLLM / Tongyi Lab), built on top of the CosyVoice codec-LM tradition:

1. **CosyVoice / CosyVoice 2 / 3** (Du et al., Alibaba, 2024–2025) — [CosyVoice 3](https://arxiv.org/abs/2505.17589). The direct ancestor: scalable, multilingual, zero-shot TTS built on *supervised semantic speech tokens* + an LM + a flow-matching decoder. Established the "discrete-token LM front-end, continuous decoder back-end" recipe that both Qwen TTS lines inherit. CosyVoice 3 is the strongest external baseline in the Qwen3-TTS tables (it holds the best Seed-TTS Chinese WER, 0.71).

2. **Mimi / Moshi** (Défossez et al., Kyutai, 2024) — [Moshi](https://arxiv.org/abs/2410.00037). Introduced the **semantic–acoustic disentangled** low-frame-rate codec (12.5 Hz, one semantic codebook + RVQ acoustic codebooks) that Qwen-TTS-Tokenizer-12Hz explicitly builds on.

3. **Qwen3-TTS** (Qwen Team, Alibaba, Jan 2026) — [arXiv:2601.15621](https://arxiv.org/abs/2601.15621), [github.com/QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS). The **first open** TTS model in the Qwen series. Two tokenizers (25 Hz single-codebook + 12 Hz multi-codebook), Qwen3 LM backbone, dual-track AR generation, Apache-2.0. This is the model you already run locally via mlx-audio.

4. **Qwen-Audio-3.0-TTS** (Tongyi Lab, Jul 2026) — [official page](https://funaudiollm.github.io/qwen-audio-3.0-tts/), [Tongyi Lab blog](https://tongyilab.substack.com/p/qwen-audio-30-tts-more-multilingual), [announcement](https://x.com/Alibaba_Qwen/status/2080270065547809133). The **hosted, closed** successor. Flash/Plus tiers, LM + flow-matching, five-stage training, 16 languages, 86 tags. **This guide's subject.** Note the naming: it is branded "Qwen-Audio-3.0" (aligning with the Qwen-Audio understanding line), not "Qwen4-TTS" — but functionally it is the next-generation TTS system after Qwen3-TTS.

5. **TTS evaluation context** — the yardsticks used below:
   - **Seed-TTS-eval** (Anastassiou et al., ByteDance, 2024) — [Seed-TTS](https://arxiv.org/abs/2406.02430). The standard zero-shot WER/SIM test set (Chinese `test-zh` + English `test-en`) that Qwen3-TTS reports.
   - **Artificial Analysis Speech Arena** ([leaderboard](https://artificialanalysis.ai/text-to-speech/leaderboard/provider-voice)) — a blind, pairwise human-preference arena that produces an **Elo** ranking of *hosted provider voices*. This is where Qwen-Audio-3.0-TTS-Plus sits at #1. Because it ranks provider APIs, the open Qwen3-TTS checkpoints are not on it.
   - **InstructTTSEval** (Huang et al., 2025) — controllability benchmark (voice design + target-speaker editing) that Qwen3-TTS reports.

**The arc**: CosyVoice (semantic-token LM + flow-matching) → Mimi (disentangled 12.5 Hz codec) → Qwen3-TTS (open, dual codec, AR LM + lightweight ConvNet decoder) → Qwen-Audio-3.0-TTS (hosted, LM + flow-matching re-introduced for the quality tier, five-stage RL-heavy training).

## Key Terms

These are TTS-specific and outside your stated background (transformers/SFT/DPO/MoE) — define them once and the tables below read cleanly.

**Codec / speech tokenizer**: The module that turns a continuous waveform into a sequence of **discrete tokens** (like text tokens, but for audio) and back. Modern TTS is "text LM predicts audio tokens, then a decoder turns tokens into waveform." Analogous to how your VLM work turned images into visual tokens — here it's audio into acoustic tokens.

**Codec frame rate (the "12 Hz" / "25 Hz" in the model names)**: How many token-frames represent one second of audio. Qwen-TTS-Tokenizer-**12Hz** emits **12.5 frames/sec** (one frame ≈ 80 ms of audio); the **25Hz** tokenizer emits 25 frames/sec (40 ms/frame). Lower frame rate = *fewer tokens the LM must autoregress through per second of speech* = cheaper, lower-latency decoding, but each token must carry more information (hence multiple codebooks per frame). This is the single most important architectural lever in both models.

**Codebook / RVQ (residual vector quantization)**: A frame of audio is too rich for one discrete symbol, so it's quantized with *several* codebooks stacked: codebook 1 captures coarse/semantic content, codebooks 2…N (the "residual" quantizers) successively refine acoustic detail. Qwen-TTS-Tokenizer-12Hz uses **16 codebooks** (1 semantic + 15 RVQ acoustic), each of size 2048. "Multi-codebook" is why a 12.5 Hz frame rate can still reconstruct high-fidelity speech.

**AR (autoregressive) TTS vs flow-matching TTS**: Two ways to generate the audio.
- **AR**: an LM predicts audio tokens left-to-right, exactly like your GPT-2 predicts text tokens. Naturally streaming (emit as you go), but each token is sequential.
- **Flow-matching (FM)**: a continuous-generative decoder (a cousin of diffusion) that maps noise → mel-spectrogram in a few solver steps, conditioned on the tokens. Higher fidelity/naturalness, but not inherently streaming and adds latency. Qwen3-TTS's *open 12Hz flagship* deliberately **avoids** flow-matching (uses a lightweight causal ConvNet decoder) for low latency; Qwen-Audio-3.0-TTS **re-introduces flow-matching** as its quality back-end.

**First-packet latency**: Wall-clock time from "send text" to "first chunk of audio comes back." The number that decides whether a voice agent feels responsive. 100 ms feels instant; 300 ms is noticeable but usable; >500 ms feels laggy.

**RTF (real-time factor)** = compute_time / audio_duration. RTF < 1 means faster than real-time (you can generate 1 s of speech in less than 1 s of compute). Qwen3-TTS-12Hz-1.7B runs at **RTF 0.31** at low load — ~3× faster than real-time. RTF governs *throughput* (how many concurrent streams one GPU sustains), distinct from first-packet latency (responsiveness).

**WER / CER for TTS**: You synthesize speech from a text, run an ASR model on the synthesized audio, and measure Word (or Character, for Chinese) Error Rate of the transcription against the original text. It measures **content consistency / intelligibility** — did the TTS say the right words clearly? Lower is better. It does *not* measure naturalness (a robotic-but-clear voice scores well).

**Speaker similarity (SIM / SS)**: Cosine similarity between a speaker-verification embedding of the reference voice and of the generated voice. Measures **how well the clone preserves the target timbre**. 0–1 scale in the Qwen3-TTS paper (0.95 is excellent); Qwen-Audio-3.0-TTS reports it on a 0–100 scale ("SS 82.75").

**Streaming / chunked decoding**: Emitting audio incrementally as text arrives, instead of waiting for the full utterance. Both models support it; the *codec's causality* (can it decode a frame using only past context?) determines how low first-packet latency can go. The 12 Hz tokenizer is fully causal → immediate emission; the 25 Hz DiT decoder needs a look-ahead window → higher latency.

**Voice cloning vs voice design**: *Cloning* = "make it sound like **this** reference clip" (3 s of audio in). *Voice design* = "make it sound like **this description**" ("an epic, trustworthy voice for a documentary trailer" — text in, novel voice out, no reference). Different capabilities; Qwen3-TTS ships separate `-CustomVoice` (clone) and `-VoiceDesign` checkpoints.

**Inline control tags**: In-text markup like `[excited]`, `[whispers]`, `[laughing]`, `[clears throat]` that steer emotion/style or inject non-verbal sounds mid-utterance. Qwen-Audio-3.0-TTS exposes **86** of them.

## Architecture: What's Under the Hood

The asymmetry here is the whole story: **Qwen3-TTS is fully documented; Qwen-Audio-3.0-TTS discloses only a system-level sketch.**

### Qwen3-TTS (open, fully disclosed)

Backbone is the **Qwen3 LM family** used as a dual-track autoregressive generator. Text tokens (standard Qwen tokenizer) and speech tokens (Qwen-TTS-Tokenizer) are concatenated along the channel axis; on receiving a text token the model predicts the corresponding acoustic tokens, which a codec decoder turns into waveform. A learnable speaker encoder is trained jointly with the backbone for identity control. Two tokenizer variants define two sub-lines:

```
                            Qwen3-TTS-12Hz  (the open flagship you run)      Qwen3-TTS-25Hz
                            ──────────────────────────────────────────────  ─────────────────────────
  Tokenizer                 12.5 Hz, multi-codebook (Mimi-style)             25 Hz, single-codebook
  Codebooks                 16  (1 semantic [WavLM teacher] + 15 RVQ acou.)  1 (codebook size 32768)
  Semantic teacher          WavLM                                            Qwen2-Audio encoder
  Decoder (token→waveform)  Lightweight CAUSAL ConvNet  (NO diffusion/FM)    DiT + Flow-Matching + BigVGAN
  Streaming                 Fully causal → immediate first-packet            Chunked (block look-ahead)
  LM residual-codebook gen  MTP (Multi-Token Prediction) module              Linear head → chunk-wise DiT
  First-packet (1.7B, c=1)  101 ms                                           150 ms
  Best at                   Low-latency streaming, high SIM                  Long-form stability (Table 10)
```

Pipeline for the 12 Hz flagship:

```
 text  ──►  Qwen3 LM  ──►  zeroth codec token ──► MTP module ──► 15 residual codec tokens
   ▲          │  (predicts frame's coarse token, then all residuals in one step)
 speaker      │
 embedding ───┘                            │
                                           ▼
                            Streaming causal ConvNet detokenizer ──► 24 kHz waveform
                            (no DiT, no flow-matching → 101 ms first packet)
```

Model family (Table 1 of the report): `{12Hz, 25Hz} × {0.6B, 1.7B} × {Base, CustomVoice, VoiceDesign/VoiceEditing}`. Trained on **5M+ hours across 10 languages**, three-stage pretraining (general → high-quality continual-pretrain → long-context to 32,768 tokens), then post-trained with **DPO** (human-preference pairs — the same algorithm you hand-implemented on hh-rlhf) + **GSPO** (rule-based rewards, a GRPO relative) + lightweight speaker fine-tuning. The 12 Hz tokenizer itself is SOTA on reconstruction (Table 4: PESQ-WB 3.21, STOI 0.96, UTMOS 4.16, SIM 0.95 — beating Mimi and FireRedTTS-2 at the same 12.5 Hz / 16-codebook budget).

### Qwen-Audio-3.0-TTS (hosted, system-level disclosure only)

What the official page *does* tell us:

```
  Tokenizer      12.5 Hz low-frame-rate supervised speech tokenizer
                 (same frame-rate philosophy as Qwen3-TTS-12Hz — keeps AR decode cheap)
  Generation     LM + FLOW-MATCHING (FM) hybrid  ← the quality back-end is re-introduced
  Training       FIVE-STAGE progressive pipeline:
                   1. independent pretraining of LM and FM
                   2. joint training with high-quality data annealing
                   3. LM reinforcement learning
                   4. FM robustness training
                   5. FM reinforcement learning
  Long-form      one-pass synthesis up to 3 minutes
  Output         vocoder super-resolution to up to 48 kHz
  Serving        bidirectional WebSocket streaming; PCM/WAV/MP3/Opus
```

What it **does NOT** disclose (and you must not guess): parameter count, whether Flash and Plus are different-size models or the same model at different solver/step budgets, codebook count, exact decoder topology, where flow-matching sits relative to the LM, context length, and per-model RTF. Treat any spec beyond the box above as unverified.

```
 text + [tags] + style prompt ──► LM (12.5 Hz tokens) ──► Flow-Matching decoder ──► SR vocoder ──► 48 kHz
                                        ▲                        ▲
                                   LM RL stage             FM robustness + FM RL stages
                                   (stage 3)               (stages 4–5) → noisy-reference robustness
```

**The one architectural takeaway**: the open flagship dropped flow-matching to win latency; the hosted model brings flow-matching *back* and wraps the whole LM+FM stack in a heavy multi-stage RL pipeline (stages 3–5 are all RL/robustness). That is almost certainly where the naturalness edge that tops Speech Arena comes from — and it is also why Plus is slow (see throughput below): a flow-matching back-end plus super-resolution is not cheap.

## Benchmarks: The Actual Gap

This is where you must be most careful. **The two models publish numbers on different test sets, and comparing them directly is a trap.**

### The apples-to-oranges warning (read this before the tables)

Qwen3-TTS's headline "1.24" is Seed-TTS `test-en` — **English only**, a clean-reference zero-shot set. Qwen-Audio-3.0-TTS's "3.87 / 3.96" is a **16-language aggregate** WER/CER that includes hard languages (Arabic, Thai, Tagalog, Vietnamese) the open model doesn't even support. A 16-language average being numerically higher than an English-only score tells you **nothing** about which model is more intelligible on a matched language. There is no primary head-to-head. So the tables below are presented *within each model*, not across them.

### Qwen3-TTS — Seed-TTS zero-shot WER (primary, arXiv Table 5)

```
Model                       test-zh (CER↓)   test-en (WER↓)
──────────────────────────  ──────────────   ──────────────
Seed-TTS                    1.12             2.25
F5-TTS                      1.56             1.83
MiniMax-Speech              0.83             1.65
CosyVoice 3                 0.71  ← best zh  1.45
Qwen3-TTS-12Hz-0.6B-Base    0.92             1.32
Qwen3-TTS-12Hz-1.7B-Base    0.77             1.24  ← best en
```

The open 1.7B flagship holds the **best English WER (1.24)** of any system in the table and is second only to CosyVoice 3 on Chinese. Its multilingual table (10 languages vs MiniMax + ElevenLabs) further shows it holds the **best speaker similarity in all 10 languages**, and its cross-lingual table shows a **~66% error reduction on Chinese→Korean vs CosyVoice 3 (4.82 vs 14.4)**. This is a genuinely strong open model — that matters for the verdict, because "the hosted one is better" does not mean "the open one is weak."

### Qwen-Audio-3.0-TTS — official numbers (primary, funaudiollm/Tongyi)

```
Metric                      Flash        Plus         Note
──────────────────────────  ───────────  ───────────  ─────────────────────────────
WER/CER, 16-lang average    3.87 ← best  3.96         best WER/CER in 10 of 16 langs
Speaker similarity (0–100)  80.44        82.75 ← #1    Plus ranks 1st across all 16
Speech Arena Elo (Plus)     —            ≈1234–1236   #1 provider voice, blind human pref
Throughput (Plus)           —            ≈16 chars/s  slow (see cost section)
First-packet (Flash)        ~300 ms      —            end-to-end, incl. network
```

### Speech Arena — the one metric where "better" is real and measured

```
Rank  Provider voice                 Elo      Samples   Price/1M chars
────  ─────────────────────────────  ───────  ────────  ──────────────
 1    Qwen-Audio-3.0-TTS-Plus (Ali)  ≈1234    ~1.5k     $27.6*
 2    Simba 3.2 (SpeechifyAI)        ≈1230    ~1.5k     $10.0
 3    Gemini 3.1 Flash TTS (Google)  ≈1215    ~2.9k     $18.3
 4    Sonic 3.5 (Cartesia)           ≈1208    ~2.3k     $49.0
 5    StepAudio 2.5 TTS (StepFun)    ≈1199    ~1.2k     $85.0
```

```
*  The leaderboard normalizes Plus to ≈$27.6/1M; the official Alibaba Model
   Studio rate is $20/1M (see Cost section). Elo is LIVE and drifts: at launch
   (2026-07-20) Plus was reported at 1236 over 1305 votes, edging Simba 3.2 by
   2 Elo; the current model page shows ≈1234.41 over ~1517 samples. A 2–4 Elo
   lead over Simba is a STATISTICAL TIE (overlapping confidence intervals) — the
   honest reading is "Plus is in a 3-way tie for #1," not "decisively best."
```

**How much better, quantified as far as primary sources allow:**
- **Human preference**: Plus is #1 (Elo ≈1234), a *statistical tie* with Simba 3.2, clearly ahead of Gemini/Cartesia. This is the strongest verified "better."
- **vs the OPEN Qwen3-TTS specifically**: **unquantifiable from primary sources** — Qwen3-TTS has no Speech Arena Elo. A secondary "Qwen3-TTS-Flash ≈929 Elo" figure circulates (which would imply a ~300-Elo gap ≈ 85% preference for Plus), but it is not primary-verified and may reference an API snapshot, not the open checkpoints. **Do not cite the 85% figure as fact.**
- **Latency**: the open 12Hz model is actually *faster* on first packet locally (97–101 ms measured vs Flash's ~300 ms end-to-end) — though the 300 ms includes network/WebSocket setup, so it's not a clean architectural comparison.

## Capability Differences

Where the hosted model's advantages are concrete and verified:

```
Capability            Qwen3-TTS (open)              Qwen-Audio-3.0-TTS (hosted)      Delta
────────────────────  ────────────────────────────  ───────────────────────────────  ──────────────
Languages             10                            16                                +6 (Ar, Id, Ms,
                      (zh,en,ja,ko,de,fr,ru,pt,      (adds Arabic, Indonesian,         Tl, Th, Vi)
                       es,it)                         Malay, Tagalog, Thai, Vietnamese)
Chinese dialects      "multiple" (unspecified)      20 dialect regions                clear win
Expressive control    NL voice-design prompts +     86 inline tags (emotion +         much richer
                      thinking-pattern instructions  non-verbal sounds) + NL control   inline surface
Voice cloning         3 s reference (+transcript)   3 s ref, robust to NOISY /         robustness win
                                                     reverberant references
Voice design          Dedicated -VoiceDesign ckpt   NL style control (hosted)         open = editable
Long-form             10+ min (25Hz best, WER 1.5)  one-pass up to 3 min              open longer
Output sample rate    24 kHz                        up to 48 kHz (SR vocoder)          fidelity win
Streaming             Local, fully causal           Hosted bidirectional WebSocket    tie (diff modes)
```

Two caveats on control: (1) the 86 tags split into *control tags* (emotion/style, e.g. `[excited]`) and *rich-language tags* (non-verbal, e.g. `[laughing]`), and per the Model Studio docs the emotion/rich-language tags work **only in unidirectional streaming mode** — a real API constraint. (2) The open model's "control" is different in kind: it's editable and fine-tunable (you own the weights), whereas the hosted tags are a fixed menu you can't extend.

## Cost & Deployment

### Pricing (primary source, with a flagged discrepancy)

The task brief and the Artificial Analysis leaderboard both cite **≈$27.59/1M chars** for Plus. The GPT deep-dive traced the **official Alibaba Cloud Model Studio pricing page**, which lists:

```
Model                       Per 10k chars   Per 1M chars   Free quota
──────────────────────────  ─────────────   ────────────   ────────────────────────
qwen-audio-3.0-tts-plus     $0.20           $20            10,000 chars, 90 days
qwen-audio-3.0-tts-flash    $0.15           $15            10,000 chars, 90 days
```

**Discrepancy**: official Model Studio = **$20/1M (Plus), $15/1M (Flash)**; Artificial Analysis + launch press = **≈$27.59/1M (Plus)**. The $20/$15 split is corroborated by third-party aggregators (OpenRouter lists $20/$15). The $27.59 figure is likely AA's normalized/region-adjusted or a launch-snapshot number. **Prefer the official $20/$15 for any real budgeting**, and treat $27.59 as an upper-bound press figure. (Both are "roughly a third of ElevenLabs/MiniMax," which is the qualitative claim everyone agrees on.)

### TCO crossover: self-host Qwen3-TTS vs pay for 3.0

The open model has **zero per-character cost** — you pay for GPU time instead. The crossover is a utilization question. Rough throughput math for the 12Hz-1.7B flagship on your single H800:

```
  From arXiv Table 2: RTF 0.463 at concurrency 6  (6 streams, each ~2× real-time)
  → GPU produces ≈ 6 / 0.463 ≈ 13 audio-seconds of speech per wall-second
  → speech is ≈ 15 characters / audio-second (English, ~150 wpm)
  → ≈ 13 × 15 ≈ 195 chars/second ≈ 0.7M chars/hour  at high utilization
```

```
                        Cost model                         Effective $/1M chars
  ────────────────────  ─────────────────────────────────  ────────────────────
  API — Flash           purely variable, $15/1M            $15.00
  API — Plus            purely variable, $20/1M            $20.00
  Self-host (rented     $2/hr H800 ÷ 0.7M chars/hr         ≈ $2.86   (if GPU is
   H800, high util.)    at high utilization                          kept busy)
  Self-host (YOUR       marginal electricity only          ≈ $0      (sunk GPU;
   owned H800)          (~$0.3/hr)                                    ~$0.43/1M)
  Self-host (Mac,       marginal electricity, 0.6B via     ≈ $0
   mlx-audio)           your existing runbook
```

**Crossover in volume** (renting an H800 at ~$2/hr, running 24/7 = ~$1,460/month):
```
  $1,460/month of GPU  ÷  $15/1M (Flash)  ≈  97M chars/month  (≈ 3.2M chars/day)
```
Above **~97M chars/month** (~3.2M/day) of steady volume, renting a GPU to self-host the open model is cheaper than the Flash API — *provided you keep the GPU busy*. Below that, the API wins because you'd be paying for idle GPU. But you **own** your H800: at the margin it's just electricity (~$0.4/1M chars), so at any steady production volume self-hosting is dramatically cheaper — the real cost is your ops time, not dollars. On your Mac (mlx-audio, 0.6B) the marginal cost is ~$0, ideal for local/offline/dev use where the 0.6B quality and 10 languages suffice.

The catch the math hides: you're not buying the same thing. $20/1M for Plus buys the **#1-ranked naturalness, 16 languages, 48 kHz, 86 tags, and noisy-reference robustness** — capabilities the open 1.7B does not fully match. TCO only decides it when the open model's quality is *good enough* for your use case.

## When To Use Which

Decision guidance mapped to your actual setup (local Mac mlx-audio runbook + 1×H800):

**Use the open Qwen3-TTS (self-host) when:**
1. You need **offline / on-prem / privacy** — reference voices or scripts that must not leave your machine (a hosted API means shipping audio to Alibaba Cloud). Your Mac mlx-audio runbook already covers this for the 0.6B.
2. You want **Apache-2.0 rights**: redistribute, fine-tune (the repo ships single-speaker fine-tuning), or embed in a product.
3. You need the **lowest first-packet latency locally** (97–101 ms) for a real-time voice agent on your own hardware.
4. **High steady volume** (≳3M chars/day) where GPU amortization on your H800 crushes per-character billing.
5. Your languages are within its **10** and you want best-in-class **speaker-cloning fidelity** (its SIM is the strongest in the paper's tables).
6. **Long-form** narration (use the 25Hz-1.7B variant — it wins the >10-min stability benchmark, WER 1.52 zh / 1.23 en).

**Pay for Qwen-Audio-3.0-TTS (API) when:**
1. You need the **top human-judged naturalness** available (Speech Arena #1) and can't invest in tuning.
2. You need **languages/dialects beyond the open 10** — Arabic, Thai, Vietnamese, Tagalog, Indonesian, Malay, or the 20 Chinese dialect regions.
3. You need **86-tag expressive control** or **48 kHz** output out of the box.
4. Your reference audio is **noisy/reverberant** — the FM robustness + RL stages target exactly this.
5. **Low/bursty volume** where you'd rather not run a GPU, and the free 10k-char/90-day quota + $15–20/1M variable cost beats ops overhead.
6. You want a hosted **WebSocket streaming** endpoint with no infrastructure.

**Pragmatic hybrid**: keep your Mac/H800 Qwen3-TTS for offline dev, privacy-sensitive jobs, high-volume batch, and English/Chinese cloning; reach for the 3.0 API for the long tail of languages, top-quality customer-facing narration, and noisy-reference cloning. They are complements, not strict substitutes.

## Key Papers / Sources

1. **Qwen3-TTS Technical Report** — Qwen Team, Alibaba (Jan 2026). Full architecture, two tokenizers, dual-track AR + MTP, all WER/SIM/latency/RTF tables. Apache-2.0. [arXiv:2601.15621](https://arxiv.org/abs/2601.15621) · [github.com/QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
2. **Qwen-Audio-3.0-TTS official page** — Tongyi Lab / FunAudioLLM (Jul 2026). WER/CER 3.87/3.96, SS 82.75/80.44, five-stage LM+FM training, 16 langs, 20 dialects. [funaudiollm.github.io/qwen-audio-3.0-tts](https://funaudiollm.github.io/qwen-audio-3.0-tts/)
3. **Qwen-Audio-3.0-TTS release blog** — Tongyi Lab (Jul 2026). Flash/Plus tiers, 300 ms latency, tag control, noisy-reference cloning. [tongyilab.substack.com](https://tongyilab.substack.com/p/qwen-audio-30-tts-more-multilingual) · [announcement](https://x.com/Alibaba_Qwen/status/2080270065547809133)
4. **Alibaba Cloud Model Studio — TTS docs & pricing** — model IDs, tag constraints (streaming-only), official $20/$15 per-1M pricing, free quota. [qwen-tts docs](https://www.alibabacloud.com/help/en/model-studio/qwen-tts) · [realtime-tts guide](https://www.alibabacloud.com/help/en/model-studio/realtime-tts-user-guide) · [pricing](https://www.alibabacloud.com/help/en/model-studio/model-pricing)
5. **Artificial Analysis Speech Arena** — blind human-preference Elo leaderboard; Plus #1 ≈1234–1236. [leaderboard](https://artificialanalysis.ai/text-to-speech/leaderboard/provider-voice) · [Plus model page](https://artificialanalysis.ai/text-to-speech/models/qwen-audio-3-0-tts-plus)
6. **CosyVoice 3** — Du et al., Alibaba (2025). The semantic-token-LM + flow-matching ancestor; strongest external baseline. [arXiv:2505.17589](https://arxiv.org/abs/2505.17589)
7. **Moshi / Mimi** — Défossez et al., Kyutai (2024). The 12.5 Hz disentangled codec Qwen-TTS-Tokenizer-12Hz builds on. [arXiv:2410.00037](https://arxiv.org/abs/2410.00037)
8. **Seed-TTS** — Anastassiou et al., ByteDance (2024). The Seed-TTS-eval WER/SIM benchmark. [arXiv:2406.02430](https://arxiv.org/abs/2406.02430)
9. **Coverage** — MarkTechPost, The Decoder, digitalapplied (Jul 2026) for cross-checks on tiers, throughput, and price-war context.

## Summary

- **What changed**: open, self-hostable Qwen3-TTS (Jan 2026, Apache-2.0, 1.7B/0.6B, AR LM + causal-ConvNet 12 Hz codec, no flow-matching for latency) → closed, hosted-only Qwen-Audio-3.0-TTS (Jul 2026, Flash/Plus, LM + flow-matching re-introduced, five-stage RL-heavy training).
- **How much better, verified**: Plus is **#1 on the blind human-preference Speech Arena** (Elo ≈1234, a statistical tie with Simba 3.2), adds **+6 languages, 20 dialects, 86 inline tags, noisy-reference cloning, 48 kHz**. All concrete and traced to primary sources.
- **How much better, NOT verifiable**: there is **no primary Elo for the open model** and **no matched WER benchmark** between the two — so the raw quality gap over Qwen3-TTS specifically is unmeasured. The "3.96 vs 1.24 WER" and "≈300 Elo / 85% preference" comparisons that circulate are **apples-to-oranges or unverified** — flagged, not used.
- **Where the open model still wins**: lower local first-packet latency (~100 ms vs ~300 ms), best-in-class speaker-cloning SIM, long-form stability (25 Hz variant), and near-zero marginal cost on hardware you already own.
- **Cost**: official Model Studio **$20/1M (Plus), $15/1M (Flash)** — not the $27.59 the leaderboard/press cite (discrepancy flagged). Self-hosting on your owned H800 costs ~$0.4/1M at the margin; renting breaks even with Flash above ~97M chars/month.
- **Verdict**: pay for 3.0 when you need top naturalness, broad language/dialect coverage, expressive tags, or noisy-reference robustness without ops; keep self-hosting Qwen3-TTS (your Mac mlx-audio 0.6B, your H800 1.7B) for privacy, fine-tuning, lowest local latency, and high-volume batch. Complements, not substitutes.
