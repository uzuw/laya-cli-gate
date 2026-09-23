from laya import Agent
from laya.common import ece_score
import numpy as np
from cli_commands import COMMANDS, TOOLS, Q_TOOL, Q_SAFE

agent = Agent("./laya-cli")

# Collect per-qtype metrics
tool_preds, tool_labels = [], []
safe_preds, safe_labels = [], []

for c in COMMANDS:
    d = agent.system_one(c["cmd"], {"tool": Q_TOOL, "safe": Q_SAFE})["answers"]
    tool_preds.append(d["tool"]["choice"])
    tool_labels.append(c["tool"])
    safe_preds.append(d["safe"]["noul"])
    safe_labels.append(c["safe"])

tool_preds = np.array(tool_preds)
tool_labels = np.array(tool_labels)
safe_preds = np.array(safe_preds)
safe_labels = np.array(safe_labels)

# Tool accuracy
tool_acc = (tool_preds == tool_labels).mean()
print(f"tool acc: {tool_acc:.3f}")

# Per-tool accuracy
for t in TOOLS:
    mask = tool_labels == t
    if mask.sum() > 0:
        acc = (tool_preds[mask] == t).mean()
        print(f"  {t}: {acc:.3f} ({mask.sum()} cmds)")

# Safe accuracy at various thresholds
print(f"\nsafe preds — mean: {safe_preds.mean():.3f}, min: {safe_preds.min():.3f}, max: {safe_preds.max():.3f}")
for t in [0.4, 0.5, 0.6, 0.7, 0.8]:
    pred_safe = safe_preds > t
    acc = (pred_safe == safe_labels).mean()
    tp = (pred_safe & safe_labels).sum()
    fp = (pred_safe & ~safe_labels).sum()
    fn = (~pred_safe & safe_labels).sum()
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-8, prec + rec)
    print(f"t={t}: acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f} (tp={tp} fp={fp} fn={fn})")

# ECE
ece = ece_score(np.maximum(safe_preds, 1 - safe_preds), (safe_preds > 0.5) == safe_labels)
print(f"\nECE: {ece:.3f}")

# Confusion details
print(f"\n--- Confusion at t=0.5 ---")
pred_safe = safe_preds > 0.5
tp = (pred_safe & safe_labels).sum()
fp = (pred_safe & ~safe_labels).sum()
fn = (~pred_safe & safe_labels).sum()
tn = (~pred_safe & ~safe_labels).sum()
print(f"TP={tp} FP={fp} FN={fn} TN={tn}")

# Show misclassified commands
print(f"\n--- Misclassified safe commands (FN: safe cmds predicted unsafe) ---")
fn_cmds = [(c["cmd"], p) for c, p, pl in zip(COMMANDS, safe_preds, safe_labels) if pl and p <= 0.5]
for cmd, p in sorted(fn_cmds, key=lambda x: x[1]):
    print(f"  {p:.3f}  {cmd}")

print(f"\n--- Misclassified unsafe commands (FP: unsafe cmds predicted safe) ---")
fp_cmds = [(c["cmd"], p) for c, p, pl in zip(COMMANDS, safe_preds, safe_labels) if not pl and p > 0.5]
for cmd, p in sorted(fp_cmds, key=lambda x: -x[1]):
    print(f"  {p:.3f}  {cmd}")
