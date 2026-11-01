"""Base runs 2/5 - conditioning study on the compact encoder (M1 of the main results table; Section 5.4) with src/train_func.py.

    tf_k72_global            compact encoder, no conditioning
    tf_k72_func_x_cal        cluster, calendar and their interaction concatenated to the input
    tf_k72_film_func_x_cal   the same information through feature-wise linear modulation (FiLM) = M1

    python experiments/base_conditioning_m1.py [--dry]
"""
import argparse
from _runner import run


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); dry = ap.parse_args().dry
    for model in ("global", "func_x_cal", "film_func_x_cal"):
        run("train_func.py", f"tf_k72_{model}", "--model", model, "--k", 72, dry=dry)


if __name__ == "__main__":
    main()
