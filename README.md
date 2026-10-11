# Reproducing "A Framework for Fair Evaluation of Variance-Aware Bandit Algorithms"

DRL Assignment 1, Group 26 (BITS Pilani WILP): Supratik Banerjee (2025AG05412), Venkata Sai Nitheesh Maddela (2025AG05497), Shrinithi RA. (2025AG05918), Divya C. (2025AG05778).

Paper: Wolf (2025), arXiv:2510.27001. Authors' code: https://github.com/eelisee/bandit_playground (tag `v1.0-SKILL2025`).

This repo is an independent re-implementation of five of the paper's eight algorithms
(ETC, epsilon-greedy, UCB, UCB-Tuned, UCB-V) and of its three Bernoulli scenarios, used to check the paper's reported regrets.

| Scenario | arm means (p1, p2) | note |
|---|---|---|
| A | 0.80, 0.90 | large gap |
| B | 0.895, 0.90 | micro-gap, low variance |
| C | 0.89, 0.895 | micro-gap, higher variance |

Protocol (as in the paper): T = 1,000,000 steps, 100 independent trials, hyperparameters from the paper's Table 1,
metric = average pseudo-regret. Each scenario is run with both arm orders to rule out tie-breaking effects.

## Run

```bash
pip install -r requirements.txt
python run_experiments.py            # full scale, about 2 minutes on 2 cores -> results/final_regret.csv, curves.npz
python compare_with_paper.py         # -> results/comparison.csv, fig_compare.png, fig_curves.png
# quick smoke test: python run_experiments.py --T 20000 --trials 4
# (compare_with_paper.py is only meaningful for the full-scale run, since the paper values are at T = 1,000,000)
```

## Files

- `bandits.py`: numba-compiled algorithms and the trial runner.
- `run_experiments.py`: scenarios, hyperparameters, seeds (base seed 42, trial i uses 42+i), CSV and curve output.
- `compare_with_paper.py`: paper Tables 3-5 values, z-score comparison, plots.
- `results/`: outputs of the full-scale run used in the slides.

## Result

24 cells compared against the paper (UCB, UCB-Tuned, UCB-V, ETC, epsilon-greedy). 22 agree within sampling noise
(z = diff / (sqrt(2) * standard error), |z| <= 2, or within 0.5% for deterministic ETC). The exceptions are UCB (+5.2%)
and UCB-Tuned (+15.6%) in scenario A (z about 2.2). The ordering UCB-Tuned < UCB-V < UCB holds in all scenarios.

Not reproduced here: EUCBV, PAC-UCB and UCB-Improved (use the authors' repo or its released data for those).
