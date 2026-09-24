#!/usr/bin/env python3
"""laya-gate: route shell commands to a tool + safety verdict in ms. See SKILL.md.
Run with the repo venv: laya/.venv/bin/python gate.py "<cmd>" [...]
Batch commands in one call — model weights (1.6GB) load once per process."""
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
LAYA = HERE.parents[3]  # .../laya (skill lives at laya/.opencode/skills/laya-gate/)
sys.path.insert(0, str(LAYA))

DEFAULT_TOOLS = {  # ponytail: same 5 the head was finetuned on (cli_commands.py)
    "git": "git version control",
    "docker": "docker containers",
    "node": "npm/node javascript",
    "k8s": "kubectl kubernetes",
    "shell": "shell/system commands",
}
Q_SAFE = {"type": "noul", "instructions": "Is this command safe to run without human review?"}

# ponytail: deterministic net for known-dangerous verbs — model memorizes nouns, regex catches verbs.
# Checked before the model verdict; a hit forces safe=false (human reviews). --dry-run exempts all.
DENY = [
    ("pipe-to-shell", re.compile(r"\|\s*(sudo\s+)?(sh|bash|zsh|fish|dash)\b")),
    ("process-sub-shell", re.compile(r"<\(\s*(curl|wget|ssh)\b")),
    ("find-delete", re.compile(r"\bfind\b.*-delete\b")),
    ("rm", re.compile(r"(^|[;&|])\s*(sudo\s+)?rm\b")),
    ("npm-unpublish", re.compile(r"\bnpm\s+unpublish\b")),
    ("npm-publish", re.compile(r"\bnpm\s+publish\b")),
    ("npm-install", re.compile(r"\bnpm\s+(install|ci|update|uninstall)\b")),
    ("npm-deprecate", re.compile(r"\bnpm\s+deprecate\b")),
    ("tar-extract", re.compile(r"\btar\s+-[a-zA-Z]*x")),
    ("sys-pkg-mutate", re.compile(r"\bpacman\s+-S|\bapt(-get)?\s+(install|upgrade|dist-upgrade|remove|purge)\b|\b(dnf|yum)\s+(install|upgrade|update|remove)\b|\bbrew\s+(install|upgrade|uninstall)\b")),
    ("npx-exec", re.compile(r"\b(npx|npm\s+exec)\b")),
    ("docker-pull", re.compile(r"\bdocker\s+pull\b")),
    ("docker-destroy", re.compile(r"\bdocker\s+(rm|rmi)\b|\bdocker\s+\w+\s+prune\b|\bdocker\s+volume\s+rm\b")),
    ("docker-lifecycle", re.compile(r"\bdocker\s+(stop|restart|kill|pause)\b")),
    ("git-reset-hard", re.compile(r"\bgit\s+reset\s+--hard\b")),
    ("git-checkout-discard", re.compile(r"\bgit\s+checkout\s+--\s")),
    ("git-clean-force", re.compile(r"\bgit\s+clean\s+-[a-zA-Z]*f")),
    ("git-push-force", re.compile(r"\bgit\s+push\b.*--force")),
    ("git-push", re.compile(r"\bgit\s+push\b")),
    ("git-branch-delete", re.compile(r"\bgit\s+branch\s+-D\b")),
    ("kubectl-delete", re.compile(r"\bkubectl\s+delete\b")),
    ("kubectl-drain", re.compile(r"\bkubectl\s+drain\b")),
    ("kubectl-mutate", re.compile(r"\bkubectl\s+(apply|scale|cordon|uncordon)\b|\bkubectl\s+rollout\s+restart\b")),
    ("dd-to-dev", re.compile(r"\bdd\b.*of=/dev/")),
    ("mkfs", re.compile(r"\bmkfs\b")),
    ("chmod-777", re.compile(r"\bchmod\b.*777")),
    ("power", re.compile(r"^\s*(shutdown|reboot|poweroff|halt)\b")),
    ("kill", re.compile(r"(^|[;&|])\s*(sudo\s+)?(kill|killall|pkill)\b")),
    ("fork-bomb", re.compile(r":\(\)\s*\{")),
    ("systemctl-mutate", re.compile(r"\bsystemctl\s+(start|stop|restart|reload|enable|disable|mask)\b")),
]


def denied(cmd):
    if "--dry-run" in cmd:  # ponytail: rehearsals are read-only by construction
        return None
    return next((name for name, rx in DENY if rx.search(cmd)), None)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("cmds", nargs="+", help="commands to gate (quote each)")
    p.add_argument("--tools", default=None, help='JSON {"name": "description"}; default: git/docker/node/k8s/shell')
    p.add_argument("--model", default=str(LAYA / "laya-cli"))
    p.add_argument("--subfolder", default=None, help='HF subfolder, e.g. "v2" for Agent("uzuw/laya-cli-gate", subfolder="v2")')
    p.add_argument("--threshold", type=float, default=0.5)  # ponytail: operating point from FINETUNE.md
    a = p.parse_args()
    tools = json.loads(a.tools) if a.tools else DEFAULT_TOOLS

    from laya import Agent
    agent = Agent(a.model, subfolder=a.subfolder)
    q = {"tool": {"type": "choice", "instructions": "Which tool owns this command?",
                  "criteria": {t: tools[t] for t in tools}}, "safe": Q_SAFE}
    for c in a.cmds:
        t0 = time.time()
        d = agent.system_one(c, q)["answers"]
        p_safe = d["safe"]["noul"]
        deny = denied(c)
        print(json.dumps({"cmd": c, "tool": d["tool"]["choice"],
                          "tool_conf": round(d["tool"]["confidence"], 2),
                          "safe_p": round(p_safe, 2),
                          "safe": bool(False if deny else p_safe > a.threshold),
                          "deny": deny,
                          "ms": round((time.time() - t0) * 1000)}), flush=True)


if __name__ == "__main__":
    main()
