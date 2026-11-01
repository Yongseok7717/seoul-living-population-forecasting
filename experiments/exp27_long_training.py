"""exp27: long-training (grokking) check for M2.  Two runs of the M2 recipe (FiLM-MLP-3L, cluster x calendar conditioning, seed 0,
fixed protocol) for 1000 epochs without early stopping, one-cycle schedule over the full budget, weight decay 0, 1e-4 and 0.1 (the last one a grokking-strength decay).
Usage: python exp27_long_training.py wd0 | wd1e-4   (one process per run; both can share the GPU).
Per-epoch history goes to results/exp27_long_training/logs/<name>.log (JSON lines) and the final JSON next to it."""
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / "src"; OUT = ROOT / "results" / "exp27_long_training"; CK = OUT / "ckpt"; LOGS = OUT / "logs"
for d in (OUT, CK, LOGS):
    d.mkdir(parents=True, exist_ok=True)
RUNS = {"wd0": 0.0, "wd1e-4": 1e-4, "wd0.1": 0.1}
key = sys.argv[1]; wd = RUNS[key]; name = f"exp27_m2_1000ep_{key}"
env = dict(os.environ, LP_RESULTS=str(OUT), LP_CKPT=str(CK), PYTHONIOENCODING="utf-8"); env.pop("RD_SPLITS", None)
cmd = [sys.executable, str(SRC / "train_cap.py"), "--name", name, "--arch", "mlp:512,256,128", "--cond", "func_x_cal", "--epochs", "1000", "--patience", "100000",
       "--no_flat_stop", "--anneal", "0", "--lr", "0.003", "--wd", str(wd), "--seed", "0"]
print(time.strftime("%m-%d %H:%M"), "RUN", name, flush=True)
with open(LOGS / f"{name}.log", "w", encoding="utf-8") as log:
    rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(SRC), env=env)
print(time.strftime("%m-%d %H:%M"), "DONE", name, "rc", rc, flush=True)
