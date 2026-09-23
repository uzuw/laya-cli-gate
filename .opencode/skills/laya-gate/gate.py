#!/usr/bin/env python3
"""laya-gate: route shell commands to a tool + safety verdict in ms. See SKILL.md.
Run with the repo venv: laya/.venv/bin/python gate.py "<cmd>" [...]
Batch commands in one call — model weights (1.6GB) load once per process."""
import json
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


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("cmds", nargs="+", help="commands to gate (quote each)")
    p.add_argument("--tools", default=None, help='JSON {"name": "description"}; default: git/docker/node/k8s/shell')
    p.add_argument("--model", default=str(LAYA / "laya-cli"))
    p.add_argument("--threshold", type=float, default=0.5)  # ponytail: operating point from FINETUNE.md
    a = p.parse_args()
    tools = json.loads(a.tools) if a.tools else DEFAULT_TOOLS

    from laya import Agent
    agent = Agent(a.model)
    q = {"tool": {"type": "choice", "instructions": "Which tool owns this command?",
                  "criteria": {t: tools[t] for t in tools}}, "safe": Q_SAFE}
    for c in a.cmds:
        t0 = time.time()
        d = agent.system_one(c, q)["answers"]
        p_safe = d["safe"]["noul"]
        print(json.dumps({"cmd": c, "tool": d["tool"]["choice"],
                          "tool_conf": round(d["tool"]["confidence"], 2),
                          "safe_p": round(p_safe, 2),
                          "safe": bool(p_safe > a.threshold),
                          "ms": round((time.time() - t0) * 1000)}), flush=True)


if __name__ == "__main__":
    main()
