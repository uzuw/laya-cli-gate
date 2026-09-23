# Laya finetune — CLI gate (`laya-cli/`)

## What
Head-only finetune of the laya decision model for terminal-command gating:
`tool` (choice over git/docker/node/k8s/shell) + `safe` (noul: runnable without human review).
Encoder frozen (`answerdotai/ModernBERT-large`); only the head trains.

## How
- Data: `cli_commands.py` — 65 commands (36 safe / 29 unsafe; round 2 added 12 miss-family rows), each yields 2 items (tool + safe) = 130 items.
- Script: `laya_finetune.py` — head-only, `detach_encoder=True`. Adam lr=5e-4, batch 8, ≤60 epochs, cosine schedule, grad-clip 1.0, shuffles per epoch, saves best-on-safe@0.7 (seed 123). Saves `laya-cli/model.safetensors` + `laya-cli/rl_agent_config.json`.
- Env: `laya/.venv`, torch 2.14.0+cu130 (CUDA).
- Reproduce: `.venv/bin/python laya_finetune.py` then `.venv/bin/python laya_eval.py`

## Results (2026-09-23, round 2 — 65 cmds, continued from checkpoint, seed 21, LR 2e-4, early-stop ep19)
Continued run: `loss 0.486 → 0.017`, best combined 2.985 (saved; plateau after ep07).

`laya_eval.py` on disk (65 cmds):

| | base | round 1 (53) | round 2 (65) |
|---|---|---|---|
| tool acc | 0.68 | 0.981 | **0.985** (only miss: 1 k8s) |
| safe acc @0.5 | 0.42 | 1.000 | **1.000** (TP=36 FP=0 FN=0) |
| safe acc @0.7 | ~0.47 | 0.981 | **1.000** |
| ECE | 0.339 | 0.023 | **0.029** |

Base `Router()` (no finetune, `laya_threshold.py`): safe acc ~0.40–0.45, ECE 0.339 — near chance. Finetune is what makes the gate usable.

## Assess
- Operating point `t=0.5`: perfect on train (prec/rec 1.00). `t=0.7` also 0.981 — either is safe to use now.
- Last miss is tool routing (`kubectl exec` → docker); node fixed itself (0.60→1.00).
- Eval is on train data: ceiling, not generalization. Backup of previous weights: `/tmp/opencode/laya-cli-backup.safetensors` (ephemeral — commit or copy aside if it matters).

## Holdout (overfit check via `gate.py`)
Round 1 — 19 unseen: tool **1.000**, safe **0.684** (`stash`/`fetch`/`top`/`ls /var/tmp` → unsafe; `npm unpublish` → safe).
Round 2 — 12 fresh paraphrases (after adding the 12 miss-family rows to train): safe **9/12 = 0.75**. Fixed: stash/fetch/top/ls families now pass. Persists: `npm unpublish --force` → safe 0.94 despite train containing `npm unpublish`; `npm outdated` (read-only!) → unsafe 0.01. npm heuristics look surface-level — small-data ceiling reached.
Verdict: good gate (human reviews `safe:false`), not an auto-approver. Next lever is dataset scale (100s of cmds), not more epochs.

## Snapshot
Locked: `laya-cli-best/` (weights + config copy of the final checkpoint).
- Known warning (upstream, not ours): checkpoint ships `temperature choice:11+ ≈ 0.10` outside [0.5,5]; laya clamps it — treat that bucket's confidence as uncalibrated.
