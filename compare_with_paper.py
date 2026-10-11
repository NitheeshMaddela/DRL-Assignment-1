"""Compare our reproduced regrets with the paper's Tables 3-5 and make plots.

Usage: python compare_with_paper.py   (after run_experiments.py)
Outputs (results/): comparison.csv, fig_compare.png, fig_curves.png
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Colours match the presentation theme
BROWN, TERRA, SAGE, INK = "#8C7A66", "#B85042", "#6E8B7A", "#3A2E28"
plt.rcParams.update({"font.family": ["Carlito", "DejaVu Sans"],
                     "font.size": 11, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": BROWN})

# z = diff / (sqrt(2)*SE): the paper's own 100-run mean carries similar noise.
# 'consistent' = within 0.5% (deterministic ETC) or |z| <= 2.
# Average regret at T=1e6 from the paper (Tables 3-5), arm order p1 < p2.
PAPER = {
    "A": {"ETC m=100": 10.10, "ETC m=1000": 100.00, "UCB": 238.17,
          "eps-greedy e=0.005": 422.99, "ETC m=10000": 1000.00,
          "eps-greedy e=0.5": 25002.20, "UCB-Tuned": 30.67,
          "UCB-V (th=1,c=1,b=1)": 109.93},
    "B": {"ETC m=10000": 150.66, "eps-greedy e=0.05": 361.97, "UCB": 1127.17,
          "eps-greedy e=0.5": 1264.52, "eps-greedy e=0.005": 1428.70,
          "UCB-Tuned": 212.21, "UCB-V (th=1,c=1,b=1)": 346.58},
    "C": {"ETC m=10000": 170.53, "eps-greedy e=0.05": 301.55,
          "ETC m=100000": 500.00, "UCB": 1172.57, "eps-greedy e=0.5": 1269.05,
          "eps-greedy e=0.005": 1762.59, "ETC m=10": 2150.00,
          "UCB-Tuned": 226.09, "UCB-V (th=1,c=1,b=1)": 367.65},
}
GAP = {"A": 0.1, "B": 0.005, "C": 0.005}
# Paper's suboptimal-pull ratios, used for the internal consistency check
PAPER_RATIO = {("A", "UCB"): 0.00238, ("B", "UCB"): 0.22543, ("C", "UCB"): 0.23451,
               ("A", "UCB-Tuned"): 0.00031, ("B", "UCB-Tuned"): 0.04242,
               ("C", "UCB-Tuned"): 0.04522}

ours = pd.read_csv("results/final_regret.csv")
asc = ours[ours.order == "asc"].set_index(["scenario", "algorithm"])
T = int(ours["T"].iloc[0]); trials = int(ours["trials"].iloc[0])

rows = []
for sc, d in PAPER.items():
    for alg, pv in d.items():
        r = asc.loc[(sc, alg)]
        se = r.std_regret / np.sqrt(trials)
        rows.append(dict(scenario=sc, algorithm=alg, paper=pv,
                         ours=round(r.mean_regret, 2),
                         diff_pct=round(100 * (r.mean_regret - pv) / pv, 1),
                         ours_std_err=round(se, 2),
                         z=round((r.mean_regret - pv) / (np.sqrt(2) * se), 2) if se > 1e-6 else 0.0,
                         consistent=bool(abs(r.mean_regret - pv) <= 0.005 * pv
                                         or abs(r.mean_regret - pv) <= 2 * np.sqrt(2) * se)))
cmp = pd.DataFrame(rows)
cmp.to_csv("results/comparison.csv", index=False)
print(cmp.to_string(index=False))

print("\nPaper-table consistency: regret == gap * ratio * T")
for (sc, alg), ratio in PAPER_RATIO.items():
    print(f"  {sc} {alg:10s} {GAP[sc]*ratio*T:9.2f} vs table {PAPER[sc][alg]:9.2f}")

# ---- figure 1: paper vs ours, the three UCB-family methods --------------
algs = ["UCB", "UCB-Tuned", "UCB-V (th=1,c=1,b=1)"]
fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), sharey=False)
for ax, sc in zip(axs, "ABC"):
    x = np.arange(len(algs)); w = 0.38
    pv = [PAPER[sc][a] for a in algs]
    ov = [asc.loc[(sc, a)].mean_regret for a in algs]
    ax.bar(x - w/2, pv, w, label="Paper", color=BROWN)
    ax.bar(x + w/2, ov, w, label="Our reproduction", color=TERRA)
    for xi, (a, b) in enumerate(zip(pv, ov)):
        ax.text(xi - w/2, a, f"{a:.0f}", ha="center", va="bottom", fontsize=9)
        ax.text(xi + w/2, b, f"{b:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(["UCB", "UCB-Tuned", "UCB-V"])
    ax.set_title(f"Scenario {sc}"); ax.spines[["top", "right"]].set_visible(False)
axs[0].set_ylabel(f"Avg. regret at T={T:,}"); axs[0].legend(frameon=False)
plt.tight_layout(); plt.savefig("results/fig_compare.png", dpi=200); plt.close()

# ---- figure 2: regret curves (log-log) ----------------------------------
cur = np.load("results/curves.npz"); ck = cur["checkpoints"]
fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), sharey=False)
col = {"UCB": BROWN, "UCB-Tuned": TERRA, "UCB-V (th=1,c=1,b=1)": SAGE}
for ax, sc in zip(axs, "ABC"):
    for a in algs:
        ax.plot(ck, cur[f"{sc}|asc|{a}"], label=a.split(" ")[0], color=col[a], lw=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(f"Scenario {sc}"); ax.set_xlabel("step t")
    ax.spines[["top", "right"]].set_visible(False)
axs[0].set_ylabel("Avg. cumulative regret"); axs[0].legend(frameon=False)
plt.tight_layout(); plt.savefig("results/fig_curves.png", dpi=200); plt.close()
