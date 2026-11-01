"""Base runs 3/5 - reference mixture of experts and gate variants with src/train_moe.py.

    moe_k72_q95_efilm_mlpN        reference mixture (FiLM-MLP normal and MLP extreme experts, class-weighted gate); its cached experts and
                                  gate are the 'ref' components of src/train_moe_f.py (base_mixture.py)
    moe_k72_q95                   same with a linear normal expert: control run of the gate-prior series
    moe_k72_q95_p{cal,calrain,calvenue,all}
                                  gate with calendar, rainfall and scheduled-event prior channels (Section 5.5)
    moe_k72_q95_r{geo,mob,ext,trans,thyper,mobloc}, moe_k72_q95_rattn{,_cal,_func}
                                  gate with propagated neighbor histories or attention over neighbors (Appendix B, second series)

    python experiments/base_gate_variants.py [--dry]
"""
import argparse
from _runner import run

COMMON = ["--k", 72, "--event_def", "q95", "--extreme", "mlp", "--extreme_rel", "none"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); dry = ap.parse_args().dry
    run("train_moe.py", "moe_k72_q95_efilm_mlpN", *COMMON, "--normal", "mlp", "--gate_losses", "wbce", "--expert_cond", "film", dry=dry)
    run("train_moe.py", "moe_k72_q95", *COMMON, "--gate_losses", "wbce,focal2", dry=dry)
    for tag, prior in (("pcal", "cal"), ("pcalrain", "cal,rain"), ("pcalvenue", "cal,venue"), ("pall", "cal,rain,venue")):
        run("train_moe.py", f"moe_k72_q95_{tag}", *COMMON, "--gate_losses", "wbce,focal2", "--prior", prior, dry=dry)
    for tag, rel in (("rgeo", "geo"), ("rmob", "mob_lag"), ("rext", "mob_ext"), ("rtrans", "trans"), ("rthyper", "trans_hyper"), ("rmobloc", "mob_lag_local")):
        run("train_moe.py", f"moe_k72_q95_{tag}", *COMMON, "--gate_losses", "wbce", "--gate_rel", rel, dry=dry)
    for tag, cond in (("rattn", None), ("rattn_cal", "cal"), ("rattn_func", "func")):
        run("train_moe.py", f"moe_k72_q95_{tag}", *COMMON, "--gate_losses", "wbce", "--gate_attn", "mob_lag_local", *(["--attn_cond", cond] if cond else []), dry=dry)


if __name__ == "__main__":
    main()
