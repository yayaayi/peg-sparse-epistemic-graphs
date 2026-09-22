"""
exp5_baseline_bayes_optimal.py -- Troisieme implementation, independante,
de la fonction de decision pour l'Experience 5 (Section 7.5).

Les deux versions precedentes de decision_utility (voir exp5_baseline.py) :
  (a) originale (buguee) : "a recu une inference" == "flagged"
  (b) corrigee : posterieur bayesien > 0.5 (seuil arbitraire, standard mais
      pas derive de la fonction d'utilite elle-meme)

Cette troisieme version utilise le SEUIL BAYESIEN OPTIMAL, derive
directement des poids de la fonction d'utilite (reward_tp=10,
penalty_fp=-50, penalty_fn=-3), pas un seuil conventionnel de 0.5.

Derivation : signaler un agent est optimal ssi l'utilite esperee de le
signaler depasse celle de ne pas le signaler :
    p*reward_tp + (1-p)*penalty_fp > p*penalty_fn + (1-p)*0
    10p - 50(1-p) > -3p
    63p > 50
    p > 50/63 ~= 0.7937

C'est le seuil theoriquement correct pour MAXIMISER l'utilite esperee
sous ce schema de penalites asymetrique -- ni la version (a) ni la
version (b) ne l'utilisaient. Si le sens de la comparaison PEG-vs-
exhaustive change encore une troisieme fois avec ce seuil, c'est un
signal fort que le resultat est instable et depend fortement du choix
de modelisation, pas d'un phenomene robuste.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from env import Population, ReactiveModel, SmoothedSurprise, calibrate_threshold, calibrate_smoothed_threshold
from peg_algorithm import candidate_sets, peg_step
from experiments.exp5_baseline import posterior_adversary

REWARD_TP, PENALTY_FP, PENALTY_FN = 10, -50, -3
# Derivation : signaler ssi p*TP + (1-p)*FP > p*FN + (1-p)*0
#   p*TP + FP - p*FP > p*FN
#   p*(TP - FP - FN) > -FP
#   p > -FP / (TP - FP - FN)
BAYES_OPTIMAL_THRESHOLD = -PENALTY_FP / (REWARD_TP - PENALTY_FP - PENALTY_FN)


def decision_utility_bayes_optimal(evaluated_agents, obs_hist, all_candidate_pairs,
                                     true_adversaries, p_compliant, p_adv):
    flagged_agents = set()
    for j in evaluated_agents:
        post = posterior_adversary(obs_hist[j], p_compliant, p_adv)
        if post > BAYES_OPTIMAL_THRESHOLD:
            flagged_agents.add(j)

    utility = 0.0
    n_pairs = 0
    for (i, j) in all_candidate_pairs:
        n_pairs += 1
        j_is_adv = j in true_adversaries
        j_flagged = j in flagged_agents
        if j_is_adv and j_flagged:
            utility += REWARD_TP
        elif (not j_is_adv) and j_flagged:
            utility += PENALTY_FP
        elif j_is_adv and (not j_flagged):
            utility += PENALTY_FN
    return utility / max(n_pairs, 1)


def run(ns=(20, 50, 100, 150, 200), frac_adv=0.15, seeds=range(5)):
    _, _, p0_hat = calibrate_threshold(n_calib=100, seed=1)
    tau, _ = calibrate_smoothed_threshold(p0_hat, n_calib=100, window=20,
                                            F_target=1.0, n_target=100)
    window, T, warmup = 20, 300, 60
    p_adv_true = 0.30

    print(f"Seuil bayesien optimal utilise : p > {BAYES_OPTIMAL_THRESHOLD:.4f}")
    print(f"{'n':>5} {'utilite PEG':>12} {'utilite exhaustif':>18} {'ratio (%)':>10}")

    results = {}
    for n in ns:
        peg_utils, exh_utils = [], []
        for seed in seeds:
            pop = Population(n, p_compliant=p0_hat, p_adv=p_adv_true,
                              frac_adv=frac_adv, seed=seed)
            model = ReactiveModel(n, p0_hat)
            smoother = SmoothedSurprise(n, window=window)
            cand = candidate_sets(n, k_max=8, mode="bounded_degree", seed=seed + 1)
            agents = list(range(n))
            true_adv = set(np.where(pop.types == 1)[0].tolist())
            obs_hist = [[] for _ in range(n)]

            all_candidate_pairs = set()
            for i in agents:
                for j in cand[i]:
                    all_candidate_pairs.add((i, j))
            all_candidate_agents = set(j for (_, j) in all_candidate_pairs)

            for t in range(T):
                o, _ = pop.step()
                for j in agents:
                    s = model.surprise(j, o[j])
                    smoother.push_raw(j, s)
                    obs_hist[j].append(int(o[j]))
                    if len(obs_hist[j]) > window:
                        obs_hist[j].pop(0)
                if t < warmup:
                    continue
                result = peg_step(t, agents, model, smoother, cand,
                                   tau=tau, rho_min=0.30, w=window)
                peg_evaluated = set(j for pair in result["E"] for j in pair)
                peg_utils.append(decision_utility_bayes_optimal(
                    peg_evaluated, obs_hist, all_candidate_pairs, true_adv, p0_hat, p_adv_true))
                exh_utils.append(decision_utility_bayes_optimal(
                    all_candidate_agents, obs_hist, all_candidate_pairs, true_adv, p0_hat, p_adv_true))

        results[n] = dict(peg_util=np.mean(peg_utils), exh_util=np.mean(exh_utils))
        ratio = 100 * np.mean(peg_utils) / np.mean(exh_utils) if np.mean(exh_utils) != 0 else float('nan')
        print(f"{n:5d} {np.mean(peg_utils):12.3f} {np.mean(exh_utils):18.3f} {ratio:10.1f}")

    return results


if __name__ == "__main__":
    run()
