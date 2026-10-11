"""Run the paper's three scenarios with our own implementations.

Usage:
    python run_experiments.py                  # full scale: T=1e6, 100 trials
    python run_experiments.py --T 100000 --trials 20   # quick check
Outputs (in results/): final_regret.csv, curves.npz
"""
import argparse, os, time
import numpy as np
import pandas as pd
from bandits import run_trials

SCENARIOS = {            # arm means, listed as (p1, p2); optimal arm = larger p
    "A": (0.80, 0.90),
    "B": (0.895, 0.90),
    "C": (0.89, 0.895),
}
# (name, algo id, param1, param2, param3)   -- hyperparameters from paper Table 1
CONFIGS = (
    [(f"ETC m={m}", 0, m, 0, 0) for m in (10, 100, 1000, 10_000, 100_000)]
    + [(f"eps-greedy e={e}", 1, e, 0, 0) for e in (0.5, 0.1, 0.05, 0.01, 0.005)]
    + [("UCB", 2, 0, 0, 0), ("UCB-Tuned", 3, 0, 0, 0),
       ("UCB-V (th=1,c=1,b=1)", 4, 1.0, 1.0, 1.0)]
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=1_000_000)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # log-spaced checkpoints for regret curves, always ending exactly at T
    ck = np.unique(np.round(np.logspace(1, np.log10(a.T), 60)).astype(np.int64))
    ck[-1] = a.T

    rows, curves = [], {}
    t0 = time.time()
    for sc, p in SCENARIOS.items():
        gap = abs(p[1] - p[0])
        # order 'asc' = (p1<p2) as in the paper; 'desc' = permuted arm indices
        for order, means in (("asc", p), ("desc", p[::-1])):
            arr = np.array(means, dtype=np.float64)
            for name, algo, p1, p2, p3 in CONFIGS:
                res = run_trials(algo, arr, a.T, ck, a.trials, a.seed,
                                 float(p1), float(p2), float(p3))
                final = res[:, -1]
                rows.append(dict(scenario=sc, order=order, algorithm=name,
                                 mean_regret=final.mean(), std_regret=final.std(),
                                 subopt_ratio=final.mean() / gap / a.T,
                                 T=a.T, trials=a.trials))
                curves[f"{sc}|{order}|{name}"] = res.mean(axis=0)
                print(f"{sc} {order:4s} {name:24s} regret={final.mean():10.2f}"
                      f"  ({time.time()-t0:5.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(f"{a.out}/final_regret.csv", index=False)
    np.savez(f"{a.out}/curves.npz", checkpoints=ck, **curves)


if __name__ == "__main__":
    main()
