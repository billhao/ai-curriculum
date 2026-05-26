# ai-curriculum — project instructions

## _todo.md (reading backlog) maintenance
_todo.md tracks guides created but not yet finished by the user, grouped by topic in suggested reading order; line format: - <slug> — <description ≤20 words>, where <slug> is the guide filename without .md.
On creating any new guide or educational doc (e.g. via /edu), smart-insert one line for it under the best-matching topic heading at the position matching suggested reading order; add a new topic heading if none fits; keep descriptions ≤20 words.
When the user says they finished a guide, remove its line from _todo.md; drop the topic heading if it becomes empty.
Topics mirror AI-Curriculum.md (e.g. Models; Long Context & Efficiency; Multimodal; Agents; Evaluation & Benchmarks; Interpretability).
Only track likely-unread guides; exclude topics already in the user's Knows list (global CLAUDE.md User Background).
Commit and push after every _todo.md change.
