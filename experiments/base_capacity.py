"""Task 16 capacity / optimisation driver: runs every stage sequentially (one training process at a time), skips runs whose result
json already exists (resumable), selects by validation wMAPE only, and writes the four CSVs.

    python run_task16cap.py [--stages A,B,C,D,E,S] [--dry]

Stage A  width/depth sweep A0..A5 (AdamW 1e-3 / wd 1e-4, <= 24 epochs, patience 5)               -> task16_capacity.csv
Stage B  lr {3e-4, 1e-3, 3e-3} @ wd 1e-4, then wd {0, 1e-5} @ best lr, for A0 and A_best (<= 48 epochs, patience 7) -> task16_optimizer.csv
Stage C  from the best Stage-B checkpoint: L-BFGS (5 outer) and AdamW-long with the same wall-clock budget                 -> task16_optimizer.csv
Stage D  tiny-set interpolation check (256 windows x 64 nodes, 168-512-512-256-128, wd 0)
Stage E  Transformer+FiLM T0/T1/T2 with the best AdamW settings of A_best (<= 24 epochs, patience 5)                        -> task16_transformer.csv
Stage S  seeds 1, 2 for reference A0, A_best and best Transformer                                                            -> task16_multiseed.csv
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / "src"; sys.path.insert(0, str(SRC))
import config as C   # noqa: E402
RES = C.BASE / "results"; LOGS = C.BASE / "logs"; RES.mkdir(parents=True, exist_ok=True); LOGS.mkdir(parents=True, exist_ok=True)
ARCH = {"A0": "mlp:128,64", "A1": "mlp:256,64", "A2": "mlp:256,128", "A3": "mlp:256,128,64", "A4": "mlp:256,256,128", "A5": "mlp:512,256,128"}
TFM = {"T0": "tfm:64,2,4,128,0.1", "T1": "tfm:128,2,4,256,0.1", "T2": "tfm:128,4,4,256,0.1"}
DRY = False


def run(name, *argv):
    f = RES / f"{name}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    cmd = [sys.executable, str(SRC / "train_cap.py"), "--name", name] + [str(a) for a in argv]
    print(time.strftime("%H:%M"), "RUN", " ".join(cmd[2:]), flush=True)
    if DRY:
        return None
    with open(LOGS / f"{name}.log", "w", encoding="utf-8") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(SRC))
    if rc != 0 or not f.exists():
        print("FAILED", name, "rc", rc, flush=True); return None
    return json.loads(f.read_text(encoding="utf-8"))


def row(r, stage, label, optimizer):
    if r is None:
        return None
    m = r.get("models", {}).get("model", {}); v, t = m.get("val", {}), m.get("test", {}); h = r.get("history", [])
    tl = [x["train_loss"] for x in h if "train_loss" in x]
    return {"stage": stage, "model": label, "name": r["name"], "architecture": r["arch"], "parameters": r["params"], "optimizer": optimizer, "lr": r["args"]["lr"], "weight_decay": r["args"]["wd"], "seed": r["args"]["seed"],
            "train_loss": (tl[-1] if tl else np.nan), "train_loss_eval": r.get("train_loss_eval", np.nan), "val_loss": v.get("huber_loss", np.nan), "test_loss": t.get("huber_loss", np.nan),
            "val_wMAPE": v.get("X_wmape", np.nan) * 100, "test_wMAPE": t.get("X_wmape", np.nan) * 100, "resid_explained": 1 - t.get("R_wmape", np.nan), "top1_error": t.get("R_rel_err_top1pct", np.nan),
            "extreme_recall": t.get("extreme_recall_top1", np.nan), "F1": t.get("extreme_f1_top1", np.nan), "peak_mag_err": t.get("peak_magnitude_rel_err", np.nan), "spatial_jaccard": t.get("top5pct_spatial_jaccard", np.nan),
            "epochs": r.get("epochs_run", np.nan), "best_epoch": r.get("best_epoch", np.nan), "runtime_min": r.get("train_minutes", r.get("minutes", np.nan)), "stop_reason": r.get("stop_reason", "")}


def save(rows, fname):
    rows = [x for x in rows if x]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows); df.to_csv(RES / fname, index=False, encoding="utf-8-sig"); print(df[["model", "architecture", "parameters", "optimizer", "val_wMAPE", "test_wMAPE", "best_epoch", "runtime_min"]].round(3).to_string(), flush=True); return df


def main():
    global DRY
    ap = argparse.ArgumentParser(); ap.add_argument("--stages", default="A,B,C,D,E,S"); ap.add_argument("--dry", action="store_true"); a = ap.parse_args(); DRY = a.dry; st = a.stages.split(",")
    # ---- Stage A ----
    A = {}
    for tag, arch in ARCH.items():
        A[tag] = run(f"t16c_{tag}", "--arch", arch, "--epochs", 24, "--patience", 5, "--lr", 1e-3, "--wd", 1e-4)
    capA = save([row(A[t], "A", t, "AdamW") for t in ARCH], "task16_capacity.csv") if any(A.values()) else None
    if "A" in st and not all(A.values()):
        print("Stage A incomplete");
    valA = {t: A[t]["models"]["model"]["val"]["X_wmape"] for t in ARCH if A[t]}
    best_tag = min(valA, key=valA.get) if valA else "A0"; print("A_best by val wMAPE:", best_tag, {t: round(v * 100, 3) for t, v in valA.items()}, flush=True)
    if not any(s in st for s in "BCES"):
        return
    # ---- Stage B ----
    opt_rows = []; bestB = {}
    for tag in dict.fromkeys(["A0", best_tag]):
        arch = ARCH[tag]; runs = {}
        for lr in (3e-4, 1e-3, 3e-3):
            runs[(lr, 1e-4)] = run(f"t16c_B_{tag}_lr{lr:g}_wd0.0001", "--arch", arch, "--epochs", 48, "--patience", 7, "--lr", lr, "--wd", 1e-4)
        ok = {k: v for k, v in runs.items() if v}
        if not ok:
            continue
        lr_best = min(ok, key=lambda k: ok[k]["models"]["model"]["val"]["X_wmape"])[0]
        for wd in (0.0, 1e-5):
            runs[(lr_best, wd)] = run(f"t16c_B_{tag}_lr{lr_best:g}_wd{wd:g}", "--arch", arch, "--epochs", 48, "--patience", 7, "--lr", lr_best, "--wd", wd)
        ok = {k: v for k, v in runs.items() if v}; kb = min(ok, key=lambda k: ok[k]["models"]["model"]["val"]["X_wmape"]); bestB[tag] = (kb, ok[kb])
        print(f"Stage B {tag}: best lr {kb[0]:g} wd {kb[1]:g} val {ok[kb]['models']['model']['val']['X_wmape'] * 100:.3f}", flush=True)
        opt_rows += [row(v, "B", f"{tag} lr={k[0]:g} wd={k[1]:g}", "AdamW") for k, v in runs.items()]
    save(opt_rows, "task16_optimizer.csv")
    if not any(s in st for s in "CES"):
        return
    # ---- Stage C ----
    for tag, (kb, rb) in bestB.items():
        arch = ARCH[tag]; init = rb["name"]; mins = [h["min"] for h in rb["history"] if h.get("stage") == "adamw"]; ep_min = float(np.median(np.diff(mins))) if len(mins) > 1 else 3.0
        lb = run(f"t16c_C3_{tag}_lbfgs", "--arch", arch, "--mode", "lbfgs", "--init", init, "--wd", kb[1], "--lr", kb[0])
        opt_rows.append(row(rb, "C", f"{tag} C1 AdamW (best B)", "AdamW"))
        if lb:
            budget = lb["train_minutes"]; n_ep = max(1, int(round(budget / max(ep_min, 0.5))))
            print(f"Stage C {tag}: L-BFGS {budget:.1f} min ~ {n_ep} AdamW epochs (epoch {ep_min:.1f} min)", flush=True)
            lg = run(f"t16c_C2_{tag}_long", "--arch", arch, "--mode", "long", "--init", init, "--epochs", n_ep, "--patience", 99, "--lr", kb[0] * 0.1, "--wd", kb[1])
            opt_rows.append(row(lb, "C", f"{tag} C3 AdamW->L-BFGS", "AdamW->L-BFGS")); opt_rows.append(row(lg, "C", f"{tag} C2 AdamW-long (+{n_ep} ep)", "AdamW-long"))
    save(opt_rows, "task16_optimizer.csv")
    # ---- Stage D ----
    if "D" in st:
        run("t16c_D_tiny", "--arch", "mlp:512,512,256,128", "--mode", "tiny", "--tiny_windows", 256, "--tiny_nodes", 64, "--tiny_steps", 3000, "--tiny_outer", 60, "--lr", 1e-3)
    if not any(s in st for s in "ES"):
        return
    # ---- Stage E ----
    kb = bestB.get(best_tag, ((1e-3, 1e-4), None))[0]; Tr = {}; erows = []
    # the MLP's best lr can be unstable for the Transformer (T0 @ 3e-3: val worsens after epoch 1) -> small lr check on T0, then T1/T2 at the chosen lr
    T0 = {kb[0]: run("t16c_T0", "--arch", TFM["T0"], "--epochs", 24, "--patience", 5, "--lr", kb[0], "--wd", kb[1])}
    for lr in (1e-3, 3e-4):
        if lr != kb[0]:
            T0[lr] = run(f"t16c_T0_lr{lr:g}", "--arch", TFM["T0"], "--epochs", 24, "--patience", 5, "--lr", lr, "--wd", kb[1])
    okT = {lr: r for lr, r in T0.items() if r}; lrT = min(okT, key=lambda lr: okT[lr]["models"]["model"]["val"]["X_wmape"]) if okT else kb[0]
    print("Transformer lr by T0 val wMAPE:", {f"{lr:g}": round(r["models"]["model"]["val"]["X_wmape"] * 100, 3) for lr, r in okT.items()}, "->", lrT, flush=True)
    erows += [row(r, "E", f"T0 lr={lr:g}", "AdamW") for lr, r in T0.items() if lr != lrT]
    Tr["T0"] = okT.get(lrT)
    for tag in ("T1", "T2"):
        Tr[tag] = run(f"t16c_{tag}", "--arch", TFM[tag], "--epochs", 24, "--patience", 5, "--lr", lrT, "--wd", kb[1])
    # ablation: the Task-14 head's persistence anchor decay(h) * z_t is what breaks sequence backbones (earlier Task 16: Tfm 6.49) -> same models without it
    Tr["T0-np"] = run("t16c_T0_np", "--arch", TFM["T0"], "--no_persist", "--epochs", 24, "--patience", 5, "--lr", lrT, "--wd", kb[1])
    # T0-np is worse still -> the remaining difference to the earlier backbone study (Tfm+FiLM 5.50) is the input convention: per-step calendar tokens
    Tr["T0-cal"] = run("t16c_T0_cal", "--arch", TFM["T0"], "--step_cal", "--epochs", 24, "--patience", 5, "--lr", lrT, "--wd", kb[1])
    # T0-cal 6.28: still far from the earlier study's 5.50 -> remove the future-log-B branch (and the anchor) to replicate that configuration
    Tr["T0-cal-nofb"] = run("t16c_T0_cal_nofb", "--arch", TFM["T0"], "--step_cal", "--no_future_b", "--epochs", 24, "--patience", 5, "--lr", lrT, "--wd", kb[1])
    Tr["T0-cal-nofb-np"] = run("t16c_T0_cal_nofb_np", "--arch", TFM["T0"], "--step_cal", "--no_future_b", "--no_persist", "--epochs", 24, "--patience", 5, "--lr", lrT, "--wd", kb[1])
    v0 = {t_: r_["models"]["model"]["val"]["X_wmape"] for t_, r_ in Tr.items() if t_.startswith("T0") and r_}; b0 = min(v0, key=v0.get)
    print("best T0 variant by val:", b0, {t_: round(v * 100, 3) for t_, v in v0.items()}, flush=True)
    if b0 != "T0" and v0[b0] * 100 < 5.3:                      # only scale the Transformer further if some variant is near the earlier study's level
        suf = b0[2:]; flags = (["--step_cal"] if "-cal" in suf else []) + (["--no_future_b"] if "-nofb" in suf else []) + (["--no_persist"] if "-np" in suf else [])
        for tag in ("T1", "T2"):
            Tr[tag + suf] = run(f"t16c_{tag}{suf.replace('-', '_')}", "--arch", TFM[tag], *flags, "--epochs", 24, "--patience", 5, "--lr", lrT, "--wd", kb[1])
    save([row(Tr[t], "E", t, "AdamW") for t in Tr] + erows, "task16_transformer.csv")
    valT = {t: Tr[t]["models"]["model"]["val"]["X_wmape"] for t in Tr if Tr[t]}; tbest = min(valT, key=valT.get) if valT else None
    if "S" not in st:
        return
    # ---- multi-seed ----
    ms = []
    cands = [("reference A0", ARCH["A0"], bestB.get("A0", ((1e-3, 1e-4), None))[0], 48, 7, bestB.get("A0", (None, None))[1])]
    if best_tag != "A0":
        cands.append((f"best MLP {best_tag}", ARCH[best_tag], kb, 48, 7, bestB[best_tag][1]))
    if tbest:
        base = tbest.split("-")[0]; cands.append((f"best Transformer {tbest}", TFM[base] + ("|np" if "-np" in tbest else "") + ("|cal" if "-cal" in tbest else "") + ("|nofb" if "-nofb" in tbest else ""), (Tr[tbest]["args"]["lr"], Tr[tbest]["args"]["wd"]), 24, 5, Tr[tbest]))
    for label, arch, (lr, wd), ep, pat, r0 in cands:
        rs = [r0] if r0 else []
        for sd in (1, 2):
            extra = (["--no_persist"] if "|np" in arch else []) + (["--step_cal"] if "|cal" in arch else []) + (["--no_future_b"] if "|nofb" in arch else [])
            rs.append(run(f"t16c_S_{label.split()[-1]}_s{sd}", "--arch", arch.split("|")[0], *extra, "--epochs", ep, "--patience", pat, "--lr", lr, "--wd", wd, "--seed", sd))
        ms += [row(r, "S", label, "AdamW") for r in rs if r]
    df = save(ms, "task16_multiseed.csv")
    agg = df.groupby("model")[["val_wMAPE", "test_wMAPE", "top1_error", "extreme_recall", "F1", "train_loss_eval", "val_loss", "test_loss"]].agg(["mean", "std", "count"]); print(agg.round(3).to_string())
    agg.to_csv(RES / "task16_multiseed_summary.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
