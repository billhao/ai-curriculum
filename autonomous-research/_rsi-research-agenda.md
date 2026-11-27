This doc describes the high level overall research agenda for RSI (recursive self improvement for AI)

## Overall problem
Mₜ = model weights + harness

Ultimate problem: Improve AI (Mₜ)'s <u>ability to identify impactful problems and propose novel improvements</u> on itself

Directions (high priority): pretraining, post-training
More directions: harness, prompt, memory, scaling etc

When using AI for RSI, directions and overall problem needs to be specified by human


## Suggested research workflow
1. identify problems
2. research literature
3. propose improvements (theories / hypothesis)
   3.1 the proposer should also state what would be learned from each plausible answer before spending compute on it (ref: After42 evaluation)
4. verify with exp on small models
5. verify with larger models if result is positive
5. save results → publish to literature (internal repo/archive)

Note, my own ~/projects/autonomous-science-lab or Allen AI's Asta may be a good infra for some of this research workflow. Other frameworks exist too

## Input
* Overall problem
* Agent instructions
* Literature (or online search)
