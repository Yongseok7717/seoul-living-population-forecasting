"""Shared helper for the base-run drivers: run one training script of src/ as a subprocess, skip it if its result exists."""
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / "src"; sys.path.insert(0, str(SRC))
import config as C   # noqa: E402

RES = C.BASE / "results"; LOGS = C.BASE / "logs"
RES.mkdir(parents=True, exist_ok=True); LOGS.mkdir(parents=True, exist_ok=True)


def run(script, name, *argv, dry=False):
    """python src/<script> --name <name> <argv>; returns the result json (or None on failure). Resumable."""
    f = RES / f"{name}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    cmd = [sys.executable, str(SRC / script), "--name", name] + [str(a) for a in argv]
    print(time.strftime("%H:%M"), "RUN", " ".join(cmd[1:]), flush=True)
    if dry:
        return None
    env = dict(os.environ, PYTHONIOENCODING="utf-8"); env.pop("RD_SPLITS", None)
    with open(LOGS / f"{name}.log", "w", encoding="utf-8") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(SRC), env=env)
    if rc != 0 or not f.exists():
        print("FAILED", name, "rc", rc, "(see", LOGS / f"{name}.log", ")", flush=True); return None
    return json.loads(f.read_text(encoding="utf-8"))
