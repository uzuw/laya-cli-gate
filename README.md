# laya-cli-gate

Head-only finetune of [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya)
(Apache-2.0) as a **pre-execution safety gate + tool router for terminal commands**.
Local, fast (~26 ms/cmd on GPU after one weight load), no API.

- **tool**: routes a command to git / docker / node / k8s / shell — acc **0.985**
- **safe**: "runnable without human review?" — acc **1.000** @ t=0.5, ECE **0.029**

Weights: `<HF-username>/laya-cli-gate` on the Hub (1.6 GB, `safetensors`) — coming shortly;
code here reproduces them from the base checkpoint.

## Quick use (agent skill)

```
.opencode/skills/laya-gate/gate.py "<cmd>" [...] [--tools '{"name":"desc"}'] [--model <dir|HF-id>]
```

Needs the `laya` package (`pip install laya`) and weights: `--model <you>/laya-cli-gate`
(HF Hub, once published), a local `laya-cli/` dir from running `laya_finetune.py`, or
`Agent("<you>/laya-cli-gate")` in Python. Batch commands in one call —
1.6 GB weights load once per process. `safe:false` → ask the human. Full protocol:
`.opencode/skills/laya-gate/SKILL.md`.

## Reproduce

```bash
python laya_finetune.py   # head-only, encoder frozen, saves best-on-combined to laya-cli/
python laya_eval.py       # tool + safe @ thresholds + ECE + confusion
```

Data: `cli_commands.py` — 65 terminal commands (36 safe / 29 unsafe) across
git, shell, docker, k8s, node. Details, training log, and honest limits:
[`FINETUNE.md`](FINETUNE.md).

## Honest limits

Eval is on train data (ceiling). Holdout on 31 unseen commands: tool ~1.0,
safe ~0.7 — fail-closed on novel read-only commands, one known FP
(`npm unpublish --force` reads safe). Good gate with human review of
`safe:false`; not an auto-approver. See FINETUNE.md.

## License

Apache-2.0. Base model © Convai Innovations; encoder `answerdotai/ModernBERT-large`.
