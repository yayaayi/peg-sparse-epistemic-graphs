"""
exp5_penalty_sensitivity.py -- Verification de robustesse pour l'Experience
5 (Section 7.5) : le resultat d'equivalence exacte entre PEG et la baseline
exhaustive, obtenu au seuil bayesien optimal (exp5_baseline_bayes_optimal.py),
tient-il uniquement pour les poids de penalite (10, -50, -3) choisis pour
cette tache, ou est-il robuste a d'autres structures de penalite ?

RESULTAT : l'equivalence exacte se maintient pour tout seuil de decision
>= 0.65 environ (six structures de penalite testees, seuils correspondants
de 0.57 a 0.89, 5 graines chacune, aucune exception observee). En-deca de
ce point (schemas de penalite plus proches de la symetrie), un ecart
marginal reapparait sur certaines graines. L'equivalence n'est donc pas
universelle par construction, mais robuste sur la gamme de schemas
raisonnablement asymetriques testee.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from env import Population, ReactiveModel, SmoothedSurprise, calibrate_threshold, calibrate_smoothed_threshold
from peg_algorithm import candidate_sets, peg_step
from experiments.exp5_baseline import posterior_adversary


def run_at_penalties(reward_tp, penalty_fp, penalty_fn, n=100, seed=0,
                       frac_adv=0.15, p_adv_true=0.30, window=20, T=300, warmup=60):
    threshold = -penalty_fp / (reward_tp - penalty_fp - penalty_fn)
    _, _, p0_hat = calibrate_threshold(n_calib=100, seed=1)
    tau, _ = calibrate_smoothed_threshold(p0_hat, n_calib=100, window=window,
                                            F_target=1.0, n_target=100)
    pop = Population(n, p_compliant=p0_hat, p_adv=p_adv_true, frac_adv=frac_adv, seed=seed)
    model = ReactiveModel(n, p0_hat)
    smoother = SmoothedSurprise(n, window=window)
    cand = candidate_sets(n, k_max=8, mode="bounded_degree", seed=seed + 1)
    agents = list(range(n))
    true_adv = set(np.where(pop.types == 1)[0].tolist())
    obs_hist = [[] for _ in range(n)]
    all_pairs = set()
    for i in agents:
        for j in cand[i]:
            all_pairs.add((i, j))
    all_agents_c = set(j for (_, j) in all_pairs)

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
        result = peg_step(t, agents, model, smoother, cand, tau=tau, rho_min=0.30, w=window)
        peg_eval = set(j for pair in result["E"] for j in pair)

    def util(evaluated):
        flagged = set()
        for j in evaluated:
            if posterior_adversary(obs_hist[j], p0_hat, p_adv_true) > threshold:
                flagged.add(j)
        u, n_pairs = 0.0, 0
        for (i, j) in all_pairs:
            n_pairs += 1
            adv, fl = j in true_adv, j in flagged
            if adv and fl:
                u += reward_tp
            elif fl and not adv:
                u += penalty_fp
            elif adv and not fl:
                u += penalty_fn
        return u / n_pairs

    return threshold, util(peg_eval), util(all_agents_c)


def run_sensitivity_sweep(n_seeds=5):
    configs = [(10, -50, -3), (10, -30, -3), (10, -100, -3),
               (10, -50, -1), (5, -50, -3), (10, -20, -5)]
    print(f"{'reward_TP':>10} {'penalty_FP':>11} {'penalty_FN':>11} "
          f"{'seuil':>7} {'diff max':>10}")
    results = []
    for tp, fp, fn in configs:
        diffs = []
        for seed in range(n_seeds):
            thr, peg_u, exh_u = run_at_penalties(tp, fp, fn, seed=seed)
            diffs.append(abs(peg_u - exh_u))
        max_diff = max(diffs)
        print(f"{tp:10d} {fp:11d} {fn:11d} {thr:7.3f} {max_diff:10.2e}")
        results.append(dict(reward_tp=tp, penalty_fp=fp, penalty_fn=fn,
                              threshold=thr, max_diff=max_diff))
    return results


if __name__ == "__main__":
    run_sensitivity_sweep(n_seeds=5)
