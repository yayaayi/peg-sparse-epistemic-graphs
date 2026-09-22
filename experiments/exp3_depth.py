"""
exp3_depth.py -- Experience 3 : Profondeur epistemique adaptative (Section 7.3)

Implemente la regle d'arret de la Definition 6-7 : mises a jour bayesiennes
sequentielles de la croyance b_i^(d)(theta_j) sur l'intention latente de j,
a partir de l'historique de la paire (i,j) uniquement, jusqu'a ce que la
surprise residuelle S_res^{ij}(t,d) = D_KL(b^(d+1) || b^(d)) tombe sous eta_d.

Teste la Conjecture 1 (decroissance geometrique) : regression log-lineaire
de la surprise residuelle contre d.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def kl_bernoulli(p, q, eps=1e-9):
    p = min(max(p, eps), 1 - eps)
    q = min(max(q, eps), 1 - eps)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def belief_update_sequence(observations, prior_alpha=1.0, prior_beta=1.0,
                             d_max=12):
    """
    Mise a jour bayesienne sequentielle d'une croyance Beta sur le taux de
    deviation theta_j, a partir des `d` observations les plus recentes de la
    paire (i,j). Retourne la sequence de surprises residuelles S_res(t,d).
    """
    residuals = []
    beliefs = []
    alpha, beta = prior_alpha, prior_beta
    b_prev = alpha / (alpha + beta)
    for d in range(1, min(d_max, len(observations)) + 1):
        o = observations[-d]
        alpha += o
        beta += (1 - o)
        b_curr = alpha / (alpha + beta)
        s_res = kl_bernoulli(b_curr, b_prev)
        residuals.append(s_res)
        beliefs.append(b_curr)
        b_prev = b_curr
    return residuals, beliefs


def run(n_pairs=80, obs_per_pair=12, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    all_residuals = []

    for _ in range(n_pairs):
        true_theta = rng.uniform(0.05, 0.6)
        obs = (rng.random(obs_per_pair) < true_theta).astype(int)
        residuals, _ = belief_update_sequence(obs, d_max=obs_per_pair)
        all_residuals.append(residuals)

    max_d = max(len(r) for r in all_residuals)
    mean_by_d = []
    for d in range(max_d):
        vals = [r[d] for r in all_residuals if len(r) > d]
        mean_by_d.append(np.mean(vals))

    d_axis = np.arange(1, len(mean_by_d) + 1)
    valid = np.array(mean_by_d) > 1e-9
    beta_hat = None
    if valid.sum() >= 2:
        slope, intercept = np.polyfit(d_axis[valid], np.log(np.array(mean_by_d)[valid]), 1)
        beta_hat = np.exp(slope)
        if verbose:
            print(f"Decroissance geometrique estimee : beta_hat = {beta_hat:.3f} "
                  f"(Conjecture 1 attend 0 < beta < 1)")
    if verbose:
        for d, m in zip(d_axis, mean_by_d):
            print(f"  d={d:2d}  surprise residuelle moyenne = {m:.4f}")
    return beta_hat


def run_multi_seed(n_seeds=5, **kwargs):
    """Boucle sur n_seeds graines (0..n_seeds-1) -- reproduit la statistique
    agregee du papier (Section 7.3, Conjecture 1 : N=5 graines)."""
    betas = []
    for s in range(n_seeds):
        b = run(seed=s, verbose=False, **kwargs)
        if b is not None:
            betas.append(b)
    betas = np.array(betas)
    print(f"=== Agrege sur {n_seeds} graines (seeds=0..{n_seeds-1}) ===")
    print(f"beta_hat : {betas.mean():.3f} +/- {betas.std():.3f}")
    return betas


if __name__ == "__main__":
    run_multi_seed(n_seeds=5)
