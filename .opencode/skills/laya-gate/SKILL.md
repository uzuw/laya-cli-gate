---
name: laya-gate
description: Gate terminal commands through the local finetuned laya model before running them. Use when routing a shell command to the right tool (git/docker/node/k8s/shell), checking if a command is safe to run without human review, or injecting a custom tool list for fast command classification.
---

# Laya Gate

Fast local pre-execution gate for terminal commands: **tool routing + safety verdict** in ms per command (after one ~1.6GB weight load per process). Model: finetuned head in `laya/laya-cli/` — tool acc 0.995, safe acc 1.000 @ t=0.5, ECE 0.015 — plus a deterministic deny net over known-dangerous verbs (pipe-to-shell, -delete, unpublish/publish, reset --hard, delete, prune, rm, …; `--dry-run` exempts). Deny forces `safe: false`; end-to-end 12/12 on unseen probes (see `laya/FINETUNE.md`).

## Use

Run with the repo venv (laya is installed there, not system-wide). Batch all commands in ONE call — weights load once per process:

```bash
laya/.venv/bin/python laya/.opencode/skills/laya-gate/gate.py "<cmd1>" "<cmd2>" [...]
```

One JSON line per command:

```json
{"cmd": "git push origin main", "tool": "git", "tool_conf": 1.0, "safe_p": 0.13, "safe": false, "deny": "git-push", "ms": 42}
```

- `deny` names the matched deny rule, or `null` when the verdict is purely the model's.

- `safe: false` → ask the human before running. Never override with a higher `--threshold` to force a pass.
- `tool_conf < 0.5` → the command fits none of the tools; treat as `shell` and re-check safety.

## Inject custom tools

Pass your agent's tool list as JSON (names + one-line descriptions). Keep ≤10 tools; the head was finetuned on 5, accuracy degrades past ~11 options:

```bash
gate.py "restart the api" --tools '{"deploy":"ship builds to prod","logs":"read service logs","shell":"anything else"}'
```

Custom names are zero-shot (the head was finetuned on the default 5), so treat `tool_conf < 0.7` on custom sets as uncertain.

## Rules

- Operating threshold is `--threshold 0.5` (default; verified 1.000 @0.4–0.8 on train, ECE 0.015). Never raise the threshold to force a pass.
- First call per session is slow (weight load); subsequent questions in the same process are ms. That is why batching matters.
- The gate is a classifier, not a sandbox: `safe: true` never excuses running an irreversible command you do not understand.
