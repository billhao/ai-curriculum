---
name: mark-read
description: Mark a guide, repository file, URL, or described content as read by Hao. Use when the user says /mark-read, finished reading, or read this.
allowed-tools: Bash, Read, Edit, AskUserQuestion
user-invocable: true
---
# Mark Read
Record exactly what Hao read in reading-history.md and remove a uniquely identified guide from _todo.md. Never treat reading a source as proof that a guide citing it was read.
## 1. Start fresh and identify the input
- Work at the ai-curriculum Git root and run git pull --rebase before resolving the item.
- $ARGUMENTS may be a repository file, URL, exact backlog slug, or text description. If empty, ask for the item; do not guess.
- Complete all resolution and validation before editing either reading record. If resolution fails or remains ambiguous, ask and write nothing.
## 2. Resolve the intended item
- Existing file: accept absolute or repository-relative paths. Resolve symlinks and require the resolved path to be inside this repository.
- Markdown guide: use its filename stem as the slug. Find exact slug matches in _todo.md; a backlog match must be unique and correspond to the resolved file. Other Markdown documents can be logged without a backlog match.
- Other repository file: log only that PDF, HTML, or other file. Never infer that any related guide was read.
- URL: log only the external content. Never remove a backlog entry merely because a guide cites that URL, even if the citation is unique.
- Exact unique backlog slug: resolve it to one existing Markdown guide and proceed without confirmation.
- Text description: search backlog slugs/descriptions, guide titles, filenames, and links. If exactly one local guide is intended but the match is fuzzy, ask the user to confirm that guide before changing the backlog. If it is clearly external or free-form content, log the supplied description without backlog changes. If the intended target is unclear or multiple local guides remain plausible, ask and write nothing.
- Reject malformed backlog structure, duplicate exact slugs, and unresolved paths before either file is written.
## 3. Check history and duplicates
- Read reading-history.md before editing. Identity is the normalized repository path, canonical URL, or description with trimmed/collapsed whitespace and case folded.
- For URL identity only, lowercase scheme/host, remove fragments and common tracking parameters such as utm_*, fbclid, and gclid, and map arXiv abs/pdf/html forms of the same paper ID to one identity. Preserve the submitted URL for display.
- For a URL label, use a description supplied with the input or a unique existing link title; otherwise use the URL itself.
- Never add a duplicate history entry or rewrite an existing entry’s label. Repeating a guide may still remove its currently present backlog entry.
## 4. Update reading-history.md
- Get the local date with date +%F. Keep date headings newest first. Add a missing heading immediately below # Reading History; otherwise append the entry at the end of that date’s section.
- Use - Guide: [slug](topic/slug.md) for guides and - File: [name](topic/file.pdf) for other repository files. Make paths repository-relative and wrap link destinations containing spaces in angle brackets.
- Use - URL: [label](submitted-url) for external links and - Description: supplied text for descriptions.
## 5. Update the backlog when applicable
- For a confirmed guide with an exact backlog entry, remove precisely its - slug line and immediate two-space-indented description line.
- Remove its ## Topic heading only when no entries remain under it. Preserve every other entry, description, heading, and their order.
- If the guide is absent from the backlog, leave _todo.md unchanged and report that no backlog entry existed.
## 6. Commit and report
- Do not update AI-Curriculum.md, the global Knows list, or autonomous-research/people.md as part of marking an item read.
- If neither file changed, make no commit and report that it was already recorded or nothing needed changing.
- After real changes, stage ALL repository changes including unrelated WIP. Commit as hao <billhao@gmail.com> with a message describing the item and ending with Co-Authored-By: Claude Code <noreply@anthropic.com>.
- Run git pull --rebase before pushing, rebase and retry a normal push if the remote advances, and never force-push.
- Report the logged item, history date, exact backlog removal or absence, duplicate status, and commit/push result.
