import json, os, math
import torch
import torch.nn.functional as F
from laya import Agent, QTYPES
from laya.common import build_sequence, collate_items, ece_score
from cli_commands import COMMANDS, TOOLS, Q_TOOL, Q_SAFE
import numpy as np

EPOCHS = 30
BATCH_SIZE = 8
LR = 2e-4  # continue from checkpoint, lower LR
SEED = 21  # round 2 (65 cmds)
PATIENCE = 12

torch.manual_seed(SEED)
np.random.seed(SEED)
agent = Agent("laya-cli")  # ponytail: continue, don't restart from base
agent.model.train()

max_len = agent.cfg.get("max_len", 512)
head_max_len = agent.cfg.get("head_max_len", 192)

def make_item(cmd, qdef, label):
    q = Agent._to_internal(qdef)
    seq, markers = build_sequence(agent.tok, cmd, q, max_len, head_max_len)
    return {"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]], "label": label}

data = []
for c in COMMANDS:
    data.append(make_item(c["cmd"], Q_TOOL, TOOLS.index(c["tool"])))
    data.append(make_item(c["cmd"], Q_SAFE, 1 if c["safe"] else 0))
print(f"Total items: {len(data)}")

head_params = [p for n, p in agent.model.named_parameters() if not n.startswith("encoder.")]
opt = torch.optim.Adam(head_params, lr=LR)

total_steps = EPOCHS * math.ceil(len(data) / BATCH_SIZE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

def train_epoch():
    agent.model.train()
    total_loss, n = 0.0, 0
    perm = torch.randperm(len(data))
    shuffled = [data[i] for i in perm]
    for i in range(0, len(shuffled), BATCH_SIZE):
        batch = shuffled[i:i+BATCH_SIZE]
        b = collate_items([batch], agent.tok.pad_token_id)
        dev = agent.device
        b = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
        logits, _ = agent.model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"], detach_encoder=True)
        loss = F.cross_entropy(logits, b["label"])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head_params, 1.0)
        opt.step()
        scheduler.step()
        total_loss += loss.item()
        n += 1
    return total_loss / n

def eval_full():
    agent.model.eval()
    safe_preds, safe_labels, tool_preds, tool_labels = [], [], [], []
    with torch.no_grad():
        for c in COMMANDS:
            d = agent.system_one(c["cmd"], {"tool": Q_TOOL, "safe": Q_SAFE})["answers"]
            tool_preds.append(d["tool"]["choice"]); tool_labels.append(c["tool"])
            safe_preds.append(d["safe"]["noul"]); safe_labels.append(c["safe"])
    safe_preds = np.array(safe_preds); safe_labels = np.array(safe_labels)
    tool_acc = np.mean(np.array(tool_preds) == np.array(tool_labels))
    acc_05 = ((safe_preds > 0.5) == safe_labels).mean()
    acc_07 = ((safe_preds > 0.7) == safe_labels).mean()
    ece = ece_score(np.maximum(safe_preds, 1 - safe_preds), (safe_preds > 0.5) == safe_labels)
    return acc_05, acc_07, ece, tool_acc

# ponytail: snapshot in laya-cli-best/ guards us — save best-seen combined
best_comb = -1; trigger = 0
for ep in range(EPOCHS):
    loss = train_epoch()
    a05, a07, ece, tool = eval_full()
    lr = opt.param_groups[0]['lr']
    marker = ""
    if tool + a05 + a07 > best_comb:
        best_comb = tool + a05 + a07; trigger = 0
        os.makedirs("laya-cli", exist_ok=True)
        json.dump(agent.cfg, open("laya-cli/rl_agent_config.json", "w"))
        from safetensors.torch import save_file
        save_file(agent.model.state_dict(), "laya-cli/model.safetensors")
        marker = " ★"
    else:
        trigger += 1
        if trigger >= PATIENCE:
            print(f"Early stop at ep {ep}!")
            break
    print(f"ep{ep:02d} loss={loss:.3f} @0.5={a05:.3f} @0.7={a07:.3f} ECE={ece:.3f} tool={tool:.3f}{marker}", flush=True)

print(f"\nDone. Best combined={best_comb:.3f}")
