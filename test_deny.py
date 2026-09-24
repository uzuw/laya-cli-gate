"""Pin the deny net: every unsafe train row must match, no safe row may match. No model load."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / ".opencode" / "skills" / "laya-gate"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate import denied  # noqa: E402
from cli_commands import COMMANDS  # noqa: E402


def main():
    fp = [(c["cmd"], denied(c["cmd"])) for c in COMMANDS if c["safe"] and denied(c["cmd"])]
    fn = [c["cmd"] for c in COMMANDS if not c["safe"] and not denied(c["cmd"])]
    assert not fp, f"safe rows denied: {fp}"
    assert not fn, f"unsafe rows missed: {fn}"
    n_safe = sum(1 for c in COMMANDS if c["safe"])
    print(f"deny net OK: {len(COMMANDS) - n_safe}/{len(COMMANDS) - n_safe} unsafe caught, 0/{n_safe} safe over-blocked")


if __name__ == "__main__":
    main()
