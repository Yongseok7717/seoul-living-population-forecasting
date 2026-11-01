"""Base runs 1/5 - reference models R0-R4 (main results table; Appendix B, first series) with src/train_residual.py.

    R0  r_zero                        baseline only, X_hat = B
    R1  r_persist                     residual persistence, z_hat(t+h) = z(t)
    R2  t6_linear_k24                 linear autoregression on the residual, 24-hour history, squared loss
    R3  r_k24_none                    GRU, 64 units, 24-hour history, squared loss
    R4  r_k24_<structure>             R3 with graph- and hypergraph-based propagation: rook adjacency, functional kNN, hard / soft /
                                      weekday-weekend-dynamic functional hypergraphs, facility hypergraph and three combinations
        r_k72_{none,func_knn,func_dynamic}   the same with a 72-hour history (control runs of Appendix B)

    python experiments/base_references.py [--dry]

Resumable: a run whose results/base/results/<name>.json exists is skipped.
"""
import argparse
from _runner import run

STRUCTURES = ["spatial", "func_knn", "func_hard", "func_soft", "func_dynamic", "facility", "spatial,func_soft", "spatial,facility", "spatial,func_soft,facility"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); dry = ap.parse_args().dry
    run("train_residual.py", "r_zero", "--baseline_only", "--k", 24, "--eval_stride", 12, dry=dry)
    run("train_residual.py", "r_persist", "--persistence", "--k", 24, "--eval_stride", 12, dry=dry)
    run("train_residual.py", "t6_linear_k24", "--k", 24, "--relations", "none", "--backbone", "linear", "--node_sample", 2000, "--epochs", 6, "--batch", 8, dry=dry)
    run("train_residual.py", "r_k24_none", "--k", 24, "--relations", "none", "--epochs", 6, dry=dry)
    for rel in STRUCTURES:
        run("train_residual.py", "r_k24_" + rel.replace(",", "+"), "--k", 24, "--relations", rel, "--epochs", 6, dry=dry)
    for rel in ("none", "func_knn", "func_dynamic"):
        run("train_residual.py", f"r_k72_{rel}", "--k", 72, "--relations", rel, "--epochs", 6, dry=dry)


if __name__ == "__main__":
    main()
