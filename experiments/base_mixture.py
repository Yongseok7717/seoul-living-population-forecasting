"""Task 16 Stage F driver: optimised binary Normal/Extreme MoE.  Run after base_capacity.py; runs, one process at a time (resumable: existing result json = skip):

  F0   reference MoE          : ref FiLM-MLP normal + ref MLP extreme + ref calibrated gate            (moef_F0)
  F1a  optimised normal (as is): best Task 16 single model checkpoint (full loss) + ref extreme + ref gate  (moef_F1a)
  F1b  optimised normal (masked): A_best retrained with the masked normal loss (t16c_N_<A>) + ref E + ref gate (moef_F1b)
  F2   optimised extreme        : E0/E1/E2 masked-extreme experts (t16c_E0..2), selected by val MAE_E (tie: val top-1%) (moef_F2)
  F3   extreme + L-BFGS / AdamW-long from the best E checkpoint, selected by val MAE_E                     (moef_F3)
  F4   gate refit with frozen experts: refit_bce (retrained wBCE) and refit_mix (mixture Huber fine-tune)  (moef_F4bce, moef_F4mix)
Outputs task16_stageF.csv (+ final comparison rows for the single FiLM-MLP and Transformer+FiLM).
"""
import json, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / "src"; sys.path.insert(0, str(SRC))
import config as C   # noqa: E402
RES = C.BASE / "results"; LOGS = C.BASE / "logs"; RES.mkdir(parents=True, exist_ok=True); LOGS.mkdir(parents=True, exist_ok=True)
ARCH = {"A0": "mlp:128,64", "A1": "mlp:256,64", "A2": "mlp:256,128", "A3": "mlp:256,128,64", "A4": "mlp:256,256,128", "A5": "mlp:512,256,128"}
EARCH = {"E0": "mlp:128,64", "E1": "mlp:256,128,64", "E2": "mlp:256,256,128,64"}


def run(script, name, *argv):
    f = RES / f"{name}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    cmd = [sys.executable, str(SRC / script), "--name", name] + [str(a) for a in argv]; print(time.strftime("%H:%M"), "RUN", " ".join(cmd[2:]), flush=True)
    with open(LOGS / f"{name}.log", "w", encoding="utf-8") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(SRC))
    if rc != 0 or not f.exists():
        print("FAILED", name, rc, flush=True); return None
    return json.loads(f.read_text(encoding="utf-8"))


def best_stageB(tag):
    opt = pd.read_csv(RES / "task16_optimizer.csv", encoding="utf-8-sig"); d = opt[(opt.stage == "B") & opt.model.str.startswith(tag + " ")]
    r = d.sort_values("val_wMAPE").iloc[0]; return r["name"], float(r.lr), float(r.weight_decay)


def erow(r, label):
    if r is None:
        return None
    m = r["models"]["model"]; return {"expert": label, "name": r["name"], "arch": r["arch"], "params": r["params"], "optimizer": r["mode"], "train_masked_eval": r.get("train_loss_eval"),
                                     **{f"{s}_{k}": m[s][k] for s in ("val", "test") for k in ("MAE_E", "MAE_N", "R_rel_err_top1pct", "extreme_recall_top1", "peak_magnitude_rel_err", "huber_masked")}, "best_epoch": r.get("best_epoch"), "minutes": r.get("train_minutes")}


def mrow(r, label, which="soft_platt"):
    if r is None:
        return None
    out = {"model": label, "name": r["name"], "normal": r.get("normal", ""), "extreme": r.get("extreme", ""), "gate": r.get("gate", ""), "mixture": which}
    for s in ("val", "test"):
        m = r["models"][which][s]
        out.update({f"{s}_wMAPE": m["X_wmape"] * 100, f"{s}_resid_explained": 1 - m["R_wmape"], f"{s}_MAE_N": m["MAE_N"], f"{s}_MAE_E": m["MAE_E"], f"{s}_top1": m["R_rel_err_top1pct"], f"{s}_recall": m["extreme_recall_top1"], f"{s}_F1": m["extreme_f1_top1"], f"{s}_peak_mag": m["peak_magnitude_rel_err"], f"{s}_jaccard": m["top5pct_spatial_jaccard"]})
    return out


def single_row(r, label):
    if r is None:
        return None
    out = {"model": label, "name": r["name"], "normal": r["arch"], "extreme": "–", "gate": "–", "mixture": "single"}
    for s in ("val", "test"):
        m = r["models"]["model"][s]
        out.update({f"{s}_wMAPE": m["X_wmape"] * 100, f"{s}_resid_explained": 1 - m["R_wmape"], f"{s}_MAE_N": m["MAE_N"], f"{s}_MAE_E": m["MAE_E"], f"{s}_top1": m["R_rel_err_top1pct"], f"{s}_recall": m["extreme_recall_top1"], f"{s}_F1": m["extreme_f1_top1"], f"{s}_peak_mag": m["peak_magnitude_rel_err"], f"{s}_jaccard": m["top5pct_spatial_jaccard"]})
    return out


def main():
    cap = pd.read_csv(RES / "task16_capacity.csv", encoding="utf-8-sig"); best_tag = cap.sort_values("val_wMAPE").model.iloc[0]; archN = ARCH[best_tag]
    nameB, lr, wd = best_stageB(best_tag); print("A_best", best_tag, archN, "best Stage-B run", nameB, lr, wd, flush=True)
    rows = []
    # F0, F1a
    rows.append(mrow(run("train_moe_f.py", "moef_F0"), "F0 reference MoE"))
    rows.append(mrow(run("train_moe_f.py", "moef_F1a", "--normal", f"ckpt:{nameB}"), f"F1a optimised normal ({best_tag}, full-loss ckpt) + ref E + ref gate"))
    # F1b: masked normal expert with the optimised architecture / settings
    rN = run("train_cap.py", f"t16c_N_{best_tag}", "--arch", archN, "--loss_mask", "normal", "--epochs", 48, "--patience", 7, "--lr", lr, "--wd", wd)
    rows.append(mrow(run("train_moe_f.py", "moef_F1b", "--normal", f"ckpt:t16c_N_{best_tag}"), f"F1b optimised normal ({best_tag}, masked) + ref E + ref gate"))
    # normal expert for F2-F4: the better of F1a (full-loss checkpoint) and F1b (masked retrain) by VALIDATION soft_platt wMAPE
    cand = {f"ckpt:{nameB}": RES / "moef_F1a.json", f"ckpt:t16c_N_{best_tag}": RES / "moef_F1b.json"}
    cand = {k: json.loads(v.read_text(encoding="utf-8"))["models"]["soft_platt"]["val"]["X_wmape"] for k, v in cand.items() if v.exists()}
    normal_spec = min(cand, key=cand.get) if cand else (f"ckpt:t16c_N_{best_tag}" if rN else f"ckpt:{nameB}")
    print("normal expert for F2-F4 by val soft_platt wMAPE:", {k: round(v * 100, 3) for k, v in cand.items()}, "->", normal_spec, flush=True)
    # F2: extreme experts
    E = {t: run("train_cap.py", f"t16c_E_{t}", "--arch", a, "--loss_mask", "extreme", "--epochs", 48, "--patience", 7, "--lr", lr, "--wd", wd) for t, a in EARCH.items()}
    erows = [erow(E[t], t) for t in EARCH]
    ok = {t: r for t, r in E.items() if r}
    eb = min(ok, key=lambda t: (round(ok[t]["models"]["model"]["val"]["MAE_E"], 1), ok[t]["models"]["model"]["val"]["R_rel_err_top1pct"])); print("E_best by val MAE_E:", eb, {t: round(r["models"]["model"]["val"]["MAE_E"], 1) for t, r in ok.items()}, flush=True)
    rows.append(mrow(run("train_moe_f.py", "moef_F2", "--normal", normal_spec, "--extreme", f"ckpt:t16c_E_{eb}"), f"F2 optimised normal + optimised extreme {eb} + ref gate"))
    # F3: L-BFGS vs AdamW-long on the best extreme expert
    mins = [h["min"] for h in ok[eb]["history"] if h.get("stage") == "adamw"]; ep_min = float(np.median(np.diff(mins))) if len(mins) > 1 else 3.0
    lb = run("train_cap.py", f"t16c_E_{eb}_lbfgs", "--arch", EARCH[eb], "--loss_mask", "extreme", "--mode", "lbfgs", "--init", f"t16c_E_{eb}", "--lr", lr, "--wd", wd)
    n_ep = max(1, int(round(lb["train_minutes"] / max(ep_min, 0.5)))) if lb else 3
    lg = run("train_cap.py", f"t16c_E_{eb}_long", "--arch", EARCH[eb], "--loss_mask", "extreme", "--mode", "long", "--init", f"t16c_E_{eb}", "--epochs", n_ep, "--patience", 99, "--lr", lr * 0.1, "--wd", wd)
    erows += [erow(lb, f"{eb} AdamW->L-BFGS"), erow(lg, f"{eb} AdamW-long (+{n_ep} ep)")]
    pd.DataFrame([x for x in erows if x]).to_csv(RES / "task16_stageF_experts.csv", index=False, encoding="utf-8-sig")
    candE = {f"t16c_E_{eb}": ok[eb]}
    if lb:
        candE[f"t16c_E_{eb}_lbfgs"] = lb
    if lg:
        candE[f"t16c_E_{eb}_long"] = lg
    e3 = min(candE, key=lambda n: candE[n]["models"]["model"]["val"]["MAE_E"]); print("F3 extreme choice by val MAE_E:", e3, flush=True)
    rows.append(mrow(run("train_moe_f.py", "moef_F3", "--normal", normal_spec, "--extreme", f"ckpt:{e3}"), f"F3 optimised normal + extreme after refinement ({e3.split('_', 2)[-1]}) + ref gate"))
    # F4: gate refit with frozen experts
    rows.append(mrow(run("train_moe_f.py", "moef_F4bce", "--normal", normal_spec, "--extreme", f"ckpt:{e3}", "--gate", "refit_bce", "--gate_epochs", 12, "--gate_patience", 4), "F4 refitted gate (wBCE retrained) + optimised experts"))
    rows.append(mrow(run("train_moe_f.py", "moef_F4mix", "--normal", normal_spec, "--extreme", f"ckpt:{e3}", "--gate", "refit_mix", "--gate_epochs", 6, "--gate_patience", 3), "F4 refitted gate (mixture Huber fine-tune) + optimised experts"))
    rows.append(mrow(json.loads((RES / "moef_F4mix.json").read_text(encoding="utf-8")) if (RES / "moef_F4mix.json").exists() else None, "F4 mixture-tuned gate, raw p (no Platt)", "soft"))
    # singles for the final comparison
    rows.append(single_row(json.loads((RES / f"{nameB}.json").read_text(encoding="utf-8")), f"single optimised FiLM-MLP ({best_tag}, Stage B best)"))
    tf = RES / "task16_transformer.csv"
    if tf.exists():
        t = pd.read_csv(tf, encoding="utf-8-sig").sort_values("val_wMAPE").iloc[0]; rows.append(single_row(json.loads((RES / t["name"]).with_suffix(".json").read_text(encoding="utf-8")), f"single optimised Transformer+FiLM ({t.model})"))
    df = pd.DataFrame([x for x in rows if x]); df.to_csv(RES / "task16_stageF.csv", index=False, encoding="utf-8-sig")
    print(df[["model", "val_wMAPE", "test_wMAPE", "test_MAE_N", "test_MAE_E", "test_top1", "test_recall", "test_F1"]].round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
