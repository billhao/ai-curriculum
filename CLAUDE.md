# ai-curriculum — project instructions

## Repo organization
Teaching guides live in topic subdirs: models, model-architecture, pretraining, sft, post-training, distillation, reasoning, inference, benchmarks, multimodal, agents, interp, knowledge-vs-intelligence; reference docs in reference/. Index/meta files stay at root: AI-Curriculum.md, _todo.md, CLAUDE.md, conferences.md, interp.md. A guide's path is <topic-dir>/<slug>.md.

## _todo.md (reading backlog) maintenance
_todo.md tracks guides created but not yet finished by the user, grouped by topic in suggested reading order. Each entry is two lines: a bullet with the bare slug (filename without .md), then a 2-space-indented sub-bullet with a ≤20-word description —
- <slug>
  - <description ≤20 words>
Topic headings mirror the topic subdirs (Title-Cased), so heading + slug give the path <topic-dir>/<slug>.md.
On creating any new guide or educational doc (e.g. via /edu), save it into the matching topic subdir and smart-insert its two-line entry under the matching topic heading at the position matching suggested reading order; add a new heading (and dir) if none fits.
When the user says they finished a guide, remove its two-line entry; drop the topic heading if it becomes empty.
Only track likely-unread guides; exclude topics already in the user's Knows list (global CLAUDE.md User Background).
Commit and push after every change.
