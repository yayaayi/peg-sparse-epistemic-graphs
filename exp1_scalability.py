"""
exp1_scalability.py -- Experience 1 : Regimes de largeur (Section 7.1 du papier)

Reproduit le protocole : n in {10,20,50,100,200}, deux structures de candidats
par agent (degre borne k_max=8 vs densite constante c=0.30), seuil tau
calibre par quantile sur population purement compliante (F_cible=1).

Critere de refutation (defini a priori) : pente de regression log-log du
cout moyen par pas significativement superieure a 1 en regime a degre borne.

Sortie attendue (papier) : pente ~0.95 (degre borne, proche de la valeur
lineaire attendue ~1) et ~1.83 (densite constante, proche de la valeur
quadratique attendue ~2).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from env import Population, ReactiveModel, SmoothedSurprise, calibrate_threshold, calibrate_smoothed_threshold
from peg_algorithm import candidate_sets, peg_step


def run_one(n, mode, tau, p0_hat, window=20, T=400, warmup=60, seed=0):
    pop = Population(n, p_compliant=p0_hat, seed=seed)
    model = ReactiveModel(n, p0_hat)
    smoother = SmoothedSurprise(n, window=window)
    cand = candidate_sets(n, k_max=8, mode=mode, density=0.30, seed=seed + 100)
    agents = list(range(n))

    costs = []
    for t in range(T):
        o, _ = pop.step()
        for j in agents:
            s = model.surprise(j, o[j])
            smoother.push_raw(j, s)
        if t >= warmup:
            result = peg_step(t, agents, model, smoother, cand, tau=tau,
                               rho_min=0.30, w=window)
            costs.append(len(result["E"]))
    return np.mean(costs)


def main():
    ns = [10, 20, 50, 100, 200]
    _, _, p0_hat = calibrate_threshold(n_calib=100, seed=1)
    tau, _ = calibrate_smoothed_threshold(p0_hat, n_calib=100, window=20,
                                            F_target=1.0, n_target=100)
    print(f"p0_hat={p0_hat:.4f}  tau={tau:.4f}")

    for mode, label in [("bounded_degree", "degre borne"),
                         ("constant_density", "densite constante")]:
        costs = []
        for n in ns:
            c = run_one(n, mode, tau, p0_hat, seed=42)
            costs.append(c)
            print(f"  {label:20s} n={n:4d}  cout moyen={c:.2f}")
        log_n = np.log(ns)
        log_c = np.log(np.maximum(costs, 1e-6))
        slope, intercept = np.polyfit(log_n, log_c, 1)
        print(f"{label} : pente log-log = {slope:.3f}\n")


if __name__ == "__main__":
    main()
