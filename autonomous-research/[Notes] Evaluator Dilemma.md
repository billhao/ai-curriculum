# RSI with a Fixed Evaluator

2026-11-27

## The evaluator sets the practical ceiling

Today I was thinking about the architecture and workflow of recursive self-improvement (RSI), and researching recent work in autonomous research. I came across *The Evaluator Dilemma*, which I understood to be MIT work examining evaluators in eight RSI systems.

The study groups evaluators into three architectures:

1. A fixed evaluator outside the self-improving loop.
2. An evaluator that co-evolves with the model or proposer.
3. A hybrid, with one part co-evolving and another part fixed outside the loop.

My main takeaway is that the evaluator's ability to distinguish good proposals from bad ones places a practical ceiling on self-improvement. Without a reliable evaluation signal, the system cannot tell whether a proposed change actually helps.

My reading also suggests that a strong evaluator outside the loop can be more useful than a co-evolving evaluator inside it. If the evaluator is no stronger at judging proposals than the proposer is at producing them, evolving both together may not provide a better signal.

## Start with a strong, fixed evaluator

For my own system, a fixed evaluator seems like the better first step for two reasons:

1. It is simpler. I do not need to build a component that also evolves the evaluator.
2. It may provide more room for improvement if it can reliably judge proposals beyond the proposer's current ability.

I could use a frontier model such as GPT Astra, Claude, or Gemini as the evaluator, while improving an open-source model such as Qwen. My working assumption is that these frontier models would be stronger evaluators for the tasks I care about, at least initially.

## The connection to benchmark saturation

This reminds me of how benchmarks change as frontier models improve. Once a benchmark is saturated, it stops distinguishing meaningfully between stronger models. Harder benchmarks restore a useful signal and make further progress visible.

I see a similar relationship in RSI: improving models need evaluators that can still recognize improvement. A saturated benchmark limits what we can measure; an inadequate evaluator can also limit which improvements the system selects and learns from.

In both cases, models improve while benchmarks or evaluators judge the results. The difference is that frontier-model development still involves substantial human labor, whereas the ambition of RSI is to make the improvement process increasingly autonomous.

## The next objective

Fully autonomous RSI may be the ultimate goal, but I do not think it needs to be the next objective. A more practical next step is partial RSI: keep the evaluator fixed while allowing the rest of the system to operate autonomously and improve.

In particular, I want the proposer to get better at identifying useful questions and proposing improvements to itself. That ability is another key part of the system.

For now, I would leave evaluator evolution alone. I would use a fixed frontier-model evaluator to guide the loop, and keep a separate held-out evaluation set outside the loop to measure whether successive versions actually become better at proposing useful self-improvements.

The immediate question is whether the proposer can improve under a strong, stable evaluation signal. Frontier models seem like a reasonable starting point for providing that signal.
