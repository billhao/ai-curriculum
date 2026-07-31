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

## autonomous-research/people.md (living people & groups map) maintenance
people.md maps research groups and people in RSI / AI-for-science. Structure: Research Groups (each major group gets its own ## section with a Key people table — name, known for, previous affiliations, graduate education (school, advisor) — plus a Major papers/systems bullet list; every other institution goes in the Other research groups table: institution, key people, key papers), then Major academic lineage (advisor→student edges), then Appendix ranked by paper importance (paper, author & affiliation pairs).
Update people.md in the same pass whenever research touches a new paper, system, group, or person in this area: add or extend the group section, add the people, insert the paper into the appendix at its rank. Don't defer it to a later cleanup.
Affiliations must come from a printed source — arXiv HTML author block (curl arxiv.org/html/<id>v1, div class ltx_authors) or PDF page 1 — never from recall or from where you believe someone currently works; keep bare acronyms bare (SII, GAIR, DeepMind) and flag any expansion as inference; mark a paper whose title page can't be fetched as inferred, inline.
Leave previous-affiliation and graduate-education cells blank when unknown — never guess an advisor, school, or past employer; the user runs those searches separately.
No scores, tiers, rubrics, or structural-findings/country-split sections in people.md — ranking lives only in the appendix, per paper. Analysis of that kind belongs in the topic guides instead.
Commit and push after every change.
