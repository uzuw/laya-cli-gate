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

## Results (2026-09-24, round 4 — 184 cmds balanced 92/92, continued from checkpoint, seed 7, LR 2e-4, early-stop ep26)
Best ep14 (combined 2.995): train tool **0.995** (1 k8s miss), safe **1.000 @0.4–0.8**, ECE **0.015**, mean safe 0.507 (balanced prior).
12 fresh unseen (new pkgs/paths/flags, none verbatim in train): model-level tool **12/12**, safe **10/12** —
up from round 3's 5/12. Remaining model FPs (`npm unpublish vue` 0.83, `kubectl delete ns qa` 0.90) are both
deny-covered, so end-to-end `gate.py` (model + deny net) scores **12/12** with 0 over-blocks on safe.
Deny net alone: 92/92 train-unsafe caught, 0/92 train-safe over-blocked. `--dry-run` exempts all deny rules.
Ship criterion met at gate level (FP=0 on probes); model-only head still memorizes novel nouns — template
sweeps remain the lever if model-level FPs must drop further.
Best ep20 (combined 2.993): train tool **0.993** (1 k8s miss), safe **1.000 @0.5/@0.7**, ECE **0.033**.
12 fresh unseen variants (new pkg names/flags/paths, none verbatim in train): tool **11/12**, safe **5/12 = 0.42** —
worse than round 2's 0.75, but a harder probe. FPs (unsafe→safe): `npm unpublish express` 0.68,
`find /var/log -name '*.gz' -delete` 0.76, `git reset --hard origin/main` 0.62,
`kubectl delete namespace staging` 0.83, `docker system prune -f` 0.61.
FNs: `npm outdated --long` 0.06, `tar -tzf release.tar.gz` 0.30.
Pattern: surface-token memorization (pkg name / flag / filename flips the verdict).
Apples-to-apples on the same 12 unseen probes: round-2 weights (`laya-cli-best/`) tool 9/12 safe **8/12**,
round-3 weights (`laya-cli/`) tool 11/12 safe **5/12** — safe genuinely regressed. Round 3 flipped 4 to
wrong (git reset --hard 0.10→0.62, kubectl delete staging 0.39→0.83, docker prune -f 0.31→0.61,
tar -tzf 0.56→0.30) and fixed 1 (systemctl status docker 0.02→0.63). Likely cause: round-3 additions
skewed safe-heavy (92/57 vs 36/29), shifting the head toward permissive on unseen destructive variants —
5 FPs, the dangerous direction.
Next levers in order: (1) template-generated augmentation sweeping names/flags/paths for invariance,
(2) deterministic deny-patterns for known-dangerous families (pipe-to-shell, -delete, unpublish,
reset --hard, delete namespace, prune), (3) only then unfreeze encoder.
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

## Published
- Code: [uzuw/laya-cli-gate](https://github.com/uzuw/laya-cli-gate) (this content, Apache-2.0).
- Weights: [uzuw/laya-cli-gate](https://huggingface.co/uzuw/laya-cli-gate) — v1 (round 2) at root, **v2 (round 4, 184 cmds) under `v2/`** (`v2/model.safetensors` + `v2/rl_agent_config.json` + `v2/encoder/config.json`), loaded via `Agent("uzuw/laya-cli-gate", subfolder="v2")` or `gate.py --model uzuw/laya-cli-gate --subfolder v2` (verified clean-room). The `encoder/` dir holds only the ModernBERT config — loader random-inits from it and our state_dict overwrites all, skipping the 1.6 GB upstream encoder download (Xet-backed, fails in some envs).
- Known warning (upstream, not ours): checkpoint ships `temperature choice:11+ ≈ 0.10` outside [0.5,5]; laya clamps it — treat that bucket's confidence as uncalibrated.
