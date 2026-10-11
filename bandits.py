"""Bandit algorithms (numba-compiled) used to reproduce the evaluation in
Wolf (2025), "A Framework for Fair Evaluation of Variance-Aware Bandit
Algorithms", arXiv:2510.27001.

Implemented here from scratch (not copied from the authors' repo):
    ETC, epsilon-greedy, UCB, UCB-Tuned, UCB-V

Every function simulates ONE run of T steps on a Bernoulli bandit and returns
the pseudo-regret at the requested checkpoints, where
    pseudo-regret(t) = sum over steps of (mu_star - mu_{arm pulled}).
Conventions follow the paper's setup: initialisation pulls each arm once,
ties are broken by lowest arm index (which is why the paper also evaluates all
arm-index permutations).
"""
import numpy as np
from numba import njit, prange

ALGOS = ("ETC", "EGREEDY", "UCB", "UCB_TUNED", "UCB_V")


@njit(cache=True)
def run_once(algo, p, T, checkpoints, seed, param1, param2, param3):
    """Simulate one run. algo: 0=ETC(m=param1) 1=eps-greedy(eps=param1)
    2=UCB 3=UCB-Tuned 4=UCB-V(theta,c,b = param1,param2,param3).
    Returns regret at each checkpoint (array, len(checkpoints))."""
    np.random.seed(seed)
    K = p.shape[0]
    best = p.max()
    n = np.zeros(K)          # pull counts
    s = np.zeros(K)          # reward sums (Bernoulli: sum of squares == sum)
    regret = 0.0
    out = np.zeros(checkpoints.shape[0])
    ci = 0
    for t in range(1, T + 1):
        arm = 0
        # --- arm selection -------------------------------------------------
        if algo == 0:                                   # ETC
            if t <= param1 * K:
                arm = (t - 1) % K
            else:
                bestq = -1.0
                for k in range(K):
                    q = s[k] / n[k]
                    if q > bestq:
                        bestq = q
                        arm = k
        elif algo == 1:                                 # epsilon-greedy
            if np.random.random() < param1:
                arm = int(np.random.random() * K)
            else:
                bestq = -1.0
                for k in range(K):
                    q = s[k] / n[k] if n[k] > 0 else 0.0
                    if q > bestq:
                        bestq = q
                        arm = k
        else:                                           # UCB family
            unpulled = -1
            for k in range(K):
                if n[k] == 0:
                    unpulled = k
                    break
            if unpulled >= 0:
                arm = unpulled
            else:
                bestu = -1e300
                for k in range(K):
                    mean = s[k] / n[k]
                    if algo == 2:                       # UCB1
                        u = mean + np.sqrt(2.0 * np.log(t - 1) / n[k])
                    elif algo == 3:                     # UCB-Tuned (Auer 2002)
                        var = s[k] / n[k] - mean * mean
                        vk = var + np.sqrt(2.0 * np.log(t) / n[k])
                        u = mean + np.sqrt(np.log(t) / n[k] * min(0.25, vk))
                    else:                               # UCB-V (Audibert 2009)
                        var = s[k] / n[k] - mean * mean
                        e = param1 * np.log(t)          # theta * log t
                        u = mean + np.sqrt(2.0 * var * e / n[k]) \
                            + param2 * 3.0 * param3 * e / n[k]
                    if u > bestu:
                        bestu = u
                        arm = k
        # --- environment ---------------------------------------------------
        r = 1.0 if np.random.random() < p[arm] else 0.0
        n[arm] += 1.0
        s[arm] += r
        regret += best - p[arm]
        if ci < checkpoints.shape[0] and t == checkpoints[ci]:
            out[ci] = regret
            ci += 1
    return out


@njit(cache=True, parallel=True)
def run_trials(algo, p, T, checkpoints, n_trials, base_seed, param1, param2, param3):
    """Run n_trials independent runs (parallel). Returns (n_trials, n_ckpt)."""
    res = np.zeros((n_trials, checkpoints.shape[0]))
    for i in prange(n_trials):
        res[i] = run_once(algo, p, T, checkpoints, base_seed + i,
                          param1, param2, param3)
    return res
