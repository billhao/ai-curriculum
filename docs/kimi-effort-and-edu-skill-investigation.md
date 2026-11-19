# Kimi effort and `/edu` skill investigation

Date: 2026-11-19

## Current symptoms

- `/effort` reports that `CLAUDE_CODE_EFFORT_LEVEL=high` overrides the session; clearing it would let medium take over.
- `/edu` is absent from the available skills in the Kimi session.

## Findings

### 1. The Kimi launcher forces high effort

`/Users/hao/projects/common/scripts/code-claude:41` contains:

```sh
export CLAUDE_CODE_EFFORT_LEVEL=high
```

This is inside the `-kimi` branch, so it does not affect normal Claude sessions. The environment variable overrides `/Users/hao/.claude-kimi/settings.json`, whose `modelSettings.k3.1-pro.effortLevel` is already `medium`.

Proposed fix: change only that launcher line to `export CLAUDE_CODE_EFFORT_LEVEL=medium`. No settings.json change is needed.

### 2. The isolated Kimi profile does not discover shared skills

The launcher sets `CLAUDE_CONFIG_DIR="$HOME/.claude-kimi"` at `scripts/code-claude:32`.

The normal Claude profile has this symlink:

```text
/Users/hao/.claude/skills -> /Users/hao/projects/common/claude/skills
```

The Kimi profile has no `/Users/hao/.claude-kimi/skills` directory or symlink. Consequently, its session loads bundled skills but not the shared user skills.

The educational skill itself exists and has valid invocation metadata at `/Users/hao/projects/common/claude/skills/edu/SKILL.md`: its name is `edu` and `user-invocable` is `true`. The skill file is not the problem.

Proposed fix: create the equivalent Kimi-profile symlink:

```sh
ln -s /Users/hao/projects/common/claude/skills /Users/hao/.claude-kimi/skills
```

This uses the normal per-profile skill-discovery path. Do not restore automatic `--add-dir` flags: the Kimi branch deliberately omits them because Claude Code 2.1.327 repeatedly asks to trust added directories.

## Verification for the follow-up session

1. Start a new Kimi session and run `/effort`; it should report medium without a high environment override.
2. Confirm `/edu` appears in the skill list and can be invoked. Avoid running it with a real topic merely to test discovery.
3. Confirm normal Claude configuration remains unchanged.

## State left by this investigation

No launcher, settings, or symlink changes were applied. This document is the only intended file change; the actual fix is deferred to another session.
