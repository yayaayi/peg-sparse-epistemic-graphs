"""
exp5_baseline.py -- Experience 5 : Comparaison a une baseline exhaustive
(Section 7.5 du papier)

Compare PEG (activation par surprise, seules les paires actives recoivent une
inference bayesienne complete) a une baseline exhaustive (toutes les paires
candidates recoivent systematiquement l'inference complete, sans filtrage).

Critere de refutation (defini a priori) : PEG doit reduire le cout sans
degrader excessivement l'utilite decisionnelle -- verdict attendu : compromis
explicite (Pareto), pas de domination.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from env import Population, ReactiveModel, SmoothedSurprise, calibrate_threshold, calibrate_smoothed_threshold
from peg_algorithm import candidate_sets, peg_step


def decision_utility(active_pairs, all_candidate_pairs, true_adversaries,
                       reward_tp=10, penalty_fp=-50, penalty_fn=-3):
    """
    Fonction d'utilite asymetrique (Section 6.5, Metriques) : recompense une
    detection correcte, penalise une fausse accusation, penalise un
    adversaire manque. `active_pairs` = paires ayant recu une inference
    (PEG : E(t) ; baseline : toutes les paires candidates).
    """
    flagged_agents = set(j for pair in active_pairs for j in pair)
    utility = 0.0
    n_pairs = 0
    for (i, j) in all_candidate_pairs:
        n_pairs += 1
        j_is_adv = j in true_adversaries
        j_flagged = j in flagged_agents
        if j_is_adv and j_flagged:
            utility += reward_tp
        elif (not j_is_adv) and j_flagged:
            utility += penalty_fp
        elif j_is_adv and (not j_flagged):
            utility += penalty_fn
    return utility / max(n_pairs, 1)


def run(ns=(20, 50, 100, 150, 200), frac_adv=0.15, seeds=range(5)):
    _, _, p0_hat = calibrate_threshold(n_calib=100, seed=1)
    tau, _ = calibrate_smoothed_threshold(p0_hat, n_calib=100, window=20,
                                            F_target=1.0, n_target=100)
    window, T, warmup = 20, 300, 60

    print(f"{'n':>5} {'cout PEG':>10} {'cout exhaustif':>15} "
          f"{'utilite PEG':>12} {'utilite exhaustif':>18}")

    for n in ns:
        peg_costs, exh_costs, peg_utils, exh_utils = [], [], [], []
        for seed in seeds:
            pop = Population(n, p_compliant=p0_hat, p_adv=0.30,
                              frac_adv=frac_adv, seed=seed)
            model = ReactiveModel(n, p0_hat)
            smoother = SmoothedSurprise(n, window=window)
            cand = candidate_sets(n, k_max=8, mode="bounded_degree", seed=seed + 1)
            agents = list(range(n))
            true_adv = set(np.where(pop.types == 1)[0].tolist())

            all_candidate_pairs = set()
            for i in agents:
                for j in cand[i]:
                    all_candidate_pairs.add((i, j))

            for t in range(T):
                o, _ = pop.step()
                for j in agents:
                    s = model.surprise(j, o[j])
                    smoother.push_raw(j, s)
                if t < warmup:
                    continue
                result = peg_step(t, agents, model, smoother, cand,
                                   tau=tau, rho_min=0.30, w=window)
                peg_costs.append(len(result["E"]))
                exh_costs.append(len(all_candidate_pairs))
                peg_utils.append(decision_utility(result["E"], all_candidate_pairs, true_adv))
                exh_utils.append(decision_utility(all_candidate_pairs, all_candidate_pairs, true_adv))

        print(f"{n:5d} {np.mean(peg_costs):10.1f} {np.mean(exh_costs):15.1f} "
              f"{np.mean(peg_utils):12.3f} {np.mean(exh_utils):18.3f}")


if __name__ == "__main__":
    run()
