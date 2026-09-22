"""
exp11_full.py -- Experience 10/11 complete : decision conditionnee aux
fenetres de rafale, avec les METRIQUES COMPLETES decrites dans le papier
(Section 7.11) : rappel coalition, rappel isole, taux de faux positifs sur
les compliants, et utilite -- pas seulement la comparaison d'utilite
simplifiee de exp11_burst_decision.py, qui ne couvrait pas ces metriques.

Reutilise la meme structure de population (compliant / adversaire isole /
coalition) que exp6_ablation.py, et compare deux regles de decision sur la
MEME population :
  (a) decision par taux moyen sur toute la fenetre (modele original, comme
      en Section 7.8) ;
  (b) decision conditionnee aux fenetres de rafale detectees par
      correlation (Section 3.3), specifiquement pendant les instants ou
      l'activite agregee des agents actifs depasse un multiple de sa
      valeur typique.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from env import Population, ReactiveModel, SmoothedSurprise, calibrate_threshold, calibrate_smoothed_threshold
from peg_algorithm import candidate_sets, peg_step


def decision_utility(flagged, true_adv, all_agents, reward_tp=10, penalty_fp=-50, penalty_fn=-3):
    u = 0.0
    for j in all_agents:
        j_adv = j in true_adv
        j_flag = j in flagged
        if j_adv and j_flag:
            u += reward_tp
        elif (not j_adv) and j_flag:
            u += penalty_fp
        elif j_adv and (not j_flag):
            u += penalty_fn
    return u / len(all_agents)


def run(n=80, frac_adv=0.10, frac_coalition=0.10, seed=0,
         coalition_sync_prob=0.15, p_adv=0.55, decision_threshold_frac=1.5):
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
    true_adv_isolated = set(np.where(pop.types == 1)[0].tolist())
    true_coalition = set(np.where(pop.types == 2)[0].tolist())
    true_adv_all = true_adv_isolated | true_coalition
    compliant = set(np.where(pop.types == 0)[0].tolist())

    obs_hist = {j: [] for j in agents}
    activity_hist = []

    for t in range(T):
        o, _ = pop.step()
        for j in agents:
            obs_hist[j].append(int(o[j]))
            s = model.surprise(j, o[j])
            smoother.push_raw(j, s)
        activity_hist.append(o.mean())
        if t < warmup:
            continue

    obs_arr = {j: np.array(obs_hist[j][warmup:]) for j in agents}
    activity_arr = np.array(activity_hist[warmup:])
    typical_activity = np.median(activity_arr)
    burst_mask = activity_arr > decision_threshold_frac * max(typical_activity, 1e-6)

    p_mid = (p0_hat + p_adv) / 2

    # (a) decision par taux moyen sur toute la fenetre
    flagged_mean = set(j for j in agents if obs_arr[j].mean() > p_mid)

    # (b) decision conditionnee aux fenetres de rafale detectees
    flagged_burst = set()
    if burst_mask.sum() > 0:
        for j in agents:
            rate_in_burst = obs_arr[j][burst_mask].mean()
            if rate_in_burst > p_mid:
                flagged_burst.add(j)

    def recall(flagged, group):
        if not group:
            return float("nan")
        return len(flagged & group) / len(group)

    def fpr(flagged, group):
        if not group:
            return 0.0
        return len(flagged & group) / len(group)

    return dict(
        recall_coalition_mean=recall(flagged_mean, true_coalition),
        recall_coalition_burst=recall(flagged_burst, true_coalition),
        recall_isolated_mean=recall(flagged_mean, true_adv_isolated),
        recall_isolated_burst=recall(flagged_burst, true_adv_isolated),
        fpr_mean=fpr(flagged_mean, compliant),
        fpr_burst=fpr(flagged_burst, compliant),
        utility_mean=decision_utility(flagged_mean, true_adv_all, agents),
        utility_burst=decision_utility(flagged_burst, true_adv_all, agents),
    )


def run_multi_seed(n_seeds=10, **kwargs):
    keys = ["recall_coalition_mean", "recall_coalition_burst",
            "recall_isolated_mean", "recall_isolated_burst",
            "fpr_mean", "fpr_burst", "utility_mean", "utility_burst"]
    agg = {k: [] for k in keys}
    for s in range(n_seeds):
        r = run(seed=s, **kwargs)
        for k in keys:
            agg[k].append(r[k])
    print(f"=== Experience 11 complete -- {n_seeds} graines ===")
    for k in keys:
        vals = np.array(agg[k])
        print(f"  {k:24s}: {np.nanmean(vals)*100 if 'utility' not in k else np.nanmean(vals):.3f} "
              f"+/- {np.nanstd(vals)*100 if 'utility' not in k else np.nanstd(vals):.3f}"
              f"{'%' if 'utility' not in k else ''}")
    from scipy.stats import ttest_rel
    t_stat, p_val = ttest_rel(agg["utility_burst"], agg["utility_mean"])
    diff = np.array(agg["utility_burst"]) - np.array(agg["utility_mean"])
    d_cohen = diff.mean() / diff.std() if diff.std() > 0 else float("nan")
    print(f"  t={t_stat:.3f}  p={p_val:.6f}  d de Cohen={d_cohen:.2f}")
    return agg


if __name__ == "__main__":
    run_multi_seed(n_seeds=10)
