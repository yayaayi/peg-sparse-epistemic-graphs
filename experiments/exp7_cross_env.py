"""
exp7_cross_env.py -- Experience 7 : Validation croisee sur un second
environnement (Section 7.7 du papier)

Reproduit les Experiences 1, 2 et 4 sur l'environnement continu (telemetrie
gaussienne) plutot que binaire, avec un modele de reference et une surprise
recalcules en consequence (vraisemblance gaussienne negative). Verifie que
les resultats les plus solides du document se generalisent.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.stats import spearmanr
from env_continuous import ContinuousPopulation, GaussianReactiveModel, calibrate_continuous
from env import SmoothedSurprise
from peg_algorithm import candidate_sets, peg_step


def calibrate_tau_continuous(mu_hat, sigma, n_calib=100, T_calib=400, window=20,
                               F_target=1.0, n_target=100, seed=1):
    model = GaussianReactiveModel(mu_hat, sigma)
    smoother = SmoothedSurprise(n_calib, window=window)
    pop = ContinuousPopulation(n_calib, mu_compliant=mu_hat, sigma=sigma, seed=seed)
    vals = []
    for t in range(T_calib):
        x, _ = pop.step()
        for j in range(n_calib):
            s = model.surprise(x[j])
            smoother.push_raw(j, s)
            if t >= window:
                vals.append(smoother.smoothed(j))
    q = 1 - F_target / n_target
    return float(np.quantile(vals, min(max(q, 0), 0.999999)))


def sub_exp1_scalability(mu_hat, sigma, tau, ns=(10, 20, 50, 100, 200), seed=42):
    print("-- Sous-experience 1 (regimes de largeur, environnement continu) --")
    for mode, label in [("bounded_degree", "degre borne"),
                         ("constant_density", "densite constante")]:
        costs = []
        for n in ns:
            model = GaussianReactiveModel(mu_hat, sigma)
            smoother = SmoothedSurprise(n, window=20)
            pop = ContinuousPopulation(n, mu_compliant=mu_hat, sigma=sigma, seed=seed)
            cand = candidate_sets(n, k_max=8, mode=mode, density=0.30, seed=seed + 1)
            agents = list(range(n))
            edge_counts = []
            for t in range(300):
                x, _ = pop.step()
                for j in agents:
                    smoother.push_raw(j, model.surprise(x[j]))
                if t >= 60:
                    result = peg_step(t, agents, model, smoother, cand, tau=tau,
                                       rho_min=0.5, w=20)
                    edge_counts.append(len(result["E"]))
            costs.append(np.mean(edge_counts))
        slope, _ = np.polyfit(np.log(ns), np.log(np.maximum(costs, 1e-6)), 1)
        print(f"  {label:20s} pente log-log = {slope:.3f}")


def sub_exp2_coalition(mu_hat, sigma, tau, n=60, seed=7):
    print("-- Sous-experience 2 (detection de coalition, environnement continu) --")
    from itertools import combinations
    model = GaussianReactiveModel(mu_hat, sigma)
    smoother = SmoothedSurprise(n, window=25)
    pop = ContinuousPopulation(n, mu_compliant=mu_hat, sigma=sigma, mu_adv=3.5,
                                 frac_adv=0.10, frac_coalition=0.10,
                                 coalition_sync_prob=0.35, seed=seed)
    cand = candidate_sets(n, k_max=8, mode="bounded_degree", seed=seed + 1)
    agents = list(range(n))
    coalition_idx = set(np.where(pop.types == 2)[0].tolist())
    true_pairs = set(combinations(sorted(coalition_idx), 2))

    tp = fp = fn = 0
    for t in range(400):
        x, _ = pop.step()
        for j in agents:
            smoother.push_raw(j, model.surprise(x[j]))
        if t < 60:
            continue
        result = peg_step(t, agents, model, smoother, cand, tau=tau,
                           rho_min=0.15, w=25)
        detected = result["C"]
        tp += len(detected & true_pairs)
        fp += len(detected - true_pairs)
        fn += len(true_pairs - detected)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    print(f"  precision={precision:.3f}  rappel={recall:.3f}  F1={f1:.3f}")


def sub_exp4_alignment(mu_hat, sigma, n=100, frac_adv=0.15, seed=0):
    print("-- Sous-experience 4 (alignement activation-contribution, environnement continu) --")
    model = GaussianReactiveModel(mu_hat, sigma)
    smoother = SmoothedSurprise(n, window=20)
    pop = ContinuousPopulation(n, mu_compliant=mu_hat, sigma=sigma, mu_adv=3.0,
                                 frac_adv=frac_adv, seed=seed)
    cand = candidate_sets(n, k_max=8, mode="bounded_degree", seed=seed + 1)
    agents = list(range(n))
    adversaries = set(np.where(pop.types == 1)[0].tolist())

    surprises, contributions = [], []
    for t in range(400):
        x, _ = pop.step()
        for j in agents:
            smoother.push_raw(j, model.surprise(x[j]))
        if t < 60:
            continue
        for i in agents:
            for j in cand[i]:
                s_bar = smoother.smoothed(j)
                is_adv = j in adversaries
                du = (0.9 * 10 + 0.1 * -3 - -3) if is_adv else (0.1 * -50 - 0.0)
                surprises.append(s_bar)
                contributions.append(du)
    rho, pval = spearmanr(surprises, contributions)
    print(f"  rho de Spearman = {rho:.3f}  p={pval:.2e}")


def run():
    mu_hat, sigma_hat = calibrate_continuous(seed=1)
    print(f"mu_hat={mu_hat:.4f}  sigma_hat={sigma_hat:.4f}")
    tau = calibrate_tau_continuous(mu_hat, sigma_hat, seed=1)
    print(f"tau (environnement continu) = {tau:.4f}\n")

    sub_exp1_scalability(mu_hat, sigma_hat, tau)
    sub_exp2_coalition(mu_hat, sigma_hat, tau)
    sub_exp4_alignment(mu_hat, sigma_hat)


if __name__ == "__main__":
    run()
