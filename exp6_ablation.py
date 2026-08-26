"""
exp6_ablation.py -- Experience 6 : Etude d'ablation (Section 7.6 du papier)

Ablation (a) : compare le rappel de detection des membres de coalition avec
activation par surprise seule (E(t) restreint aux Etapes 1-2, sans Etape 3b)
contre PEG complet (surprise + correlation).

Reproduit le resultat central de cette experience : la correlation est
decisive pour les adversaires en coalition (rappel quasi nul via la surprise
seule, tres superieur avec PEG complet), sans effet sur les adversaires
isoles (deja bien captes par la surprise seule).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from env import Population, ReactiveModel, SmoothedSurprise, calibrate_threshold, calibrate_smoothed_threshold
from peg_algorithm import candidate_sets, peg_step


def run(n=80, frac_adv=0.10, frac_coalition=0.10, seed=0,
        coalition_sync_prob=0.35, p_adv=0.55):
    window, T, warmup = 25, 400, 60

    _, _, p0_hat = calibrate_threshold(n_calib=100, seed=1)
    tau, _ = calibrate_smoothed_threshold(p0_hat, n_calib=100, window=20,
                                            F_target=1.0, n_target=100)

    pop = Population(n, p_compliant=p0_hat, p_adv=p_adv, frac_adv=frac_adv,
                      frac_coalition=frac_coalition,
                      coalition_sync_prob=coalition_sync_prob, seed=seed)
    model = ReactiveModel(n, p0_hat)
    smoother = SmoothedSurprise(n, window=window)
    cand = candidate_sets(n, k_max=8, mode="bounded_degree", seed=seed + 1)
    agents = list(range(n))

    isolated = set(np.where(pop.types == 1)[0].tolist())
    coalition = set(np.where(pop.types == 2)[0].tolist())

    hits_surprise_only = {"isolated": 0, "coalition": 0}
    hits_full = {"isolated": 0, "coalition": 0}
    n_eval = 0

    for t in range(T):
        o, _ = pop.step()
        for j in agents:
            s = model.surprise(j, o[j])
            smoother.push_raw(j, s)
        if t < warmup:
            continue
        n_eval += 1

        result = peg_step(t, agents, model, smoother, cand, tau=tau,
                           rho_min=0.30, w=window)
        flagged_full = set(j for pair in result["E"] for j in pair)
        flagged_surprise_only = result["V"]  # Etapes 1-2 seules, sans coalition

        for label, group in [("isolated", isolated), ("coalition", coalition)]:
            if group & flagged_surprise_only:
                hits_surprise_only[label] += 1
            if group & flagged_full:
                hits_full[label] += 1

    print(f"n={n}  {len(isolated)} adversaires isoles, {len(coalition)} en coalition\n")
    for label in ["isolated", "coalition"]:
        r_so = hits_surprise_only[label] / n_eval
        r_full = hits_full[label] / n_eval
        print(f"  {label:10s} : rappel surprise seule = {r_so:.3f}   "
              f"rappel PEG complet = {r_full:.3f}")


if __name__ == "__main__":
    run()
