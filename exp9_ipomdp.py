"""
exp9_ipomdp.py -- Experience 9 : Comparaison a une inference bayesienne
jointe exacte, representative d'I-POMDP (Section 7.9 du papier)

Implemente une inference bayesienne JOINTE et EXACTE sur l'espace des types
d'agents (compliant / adversaire), avec un avantage donne a cette baseline :
modele generatif exact connu (contrairement au seuil de PEG, calibre
empiriquement). Coût en principe exponentiel (2^n hypotheses), rendu
praticable ici par une forme close specifique a ce modele generatif
(les types sont independants a priori, donc la vraisemblance jointe se
factorise -- voir note dans le code).

Protocole corrige (tel que documente dans le papier, Section 7.9) : le
nombre ABSOLU de membres de coalition disponibles pour la correlation doit
etre suffisant (au moins 2-3), independamment de la taille de la population.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from itertools import combinations
from env import Population, ReactiveModel, SmoothedSurprise, calibrate_threshold, calibrate_smoothed_threshold
from peg_algorithm import candidate_sets, peg_step


def joint_exact_inference(obs_window_per_agent, p_compliant, p_adv, prior_adv=0.15):
    """
    Inference bayesienne EXACTE sur le type de chaque agent, a partir de sa
    fenetre d'observations. Comme le prior est independant entre agents et
    que la vraisemblance de chaque agent ne depend que de ses PROPRES
    observations (pas d'interaction dans ce modele generatif), la loi
    posterieure jointe se factorise : P(type_1,...,type_n | obs) =
    prod_j P(type_j | obs_j). C'est cette factorisation -- specifique a ce
    modele generatif exact connu -- qui rend le calcul praticable malgre le
    nombre d'hypotheses en principe exponentiel (2^n). Une methode qui ne
    connaitrait pas cette factorisation devrait enumerer explicitement.
    """
    posteriors = {}
    for j, obs in obs_window_per_agent.items():
        obs = np.asarray(obs)
        if len(obs) == 0:
            posteriors[j] = prior_adv
            continue
        log_lik_compliant = np.sum(obs * np.log(p_compliant) + (1 - obs) * np.log(1 - p_compliant))
        log_lik_adv = np.sum(obs * np.log(p_adv) + (1 - obs) * np.log(1 - p_adv))
        log_post_compliant = log_lik_compliant + np.log(1 - prior_adv)
        log_post_adv = log_lik_adv + np.log(prior_adv)
        m = max(log_post_compliant, log_post_adv)
        post_adv = np.exp(log_post_adv - m) / (np.exp(log_post_adv - m) + np.exp(log_post_compliant - m))
        posteriors[j] = post_adv
    return posteriors


def enumeration_cost_estimate(n):
    """Cout combinatoire (nombre d'hypotheses) d'une enumeration NAIVE sans
    factorisation, a titre d'illustration (Section 7.9) -- pas execute."""
    return 2 ** n


def run(ns=(10, 20, 50, 100), n_coalition_members=4, frac_coalition_of_pop=None,
        window=20, T=300, warmup=60, seed=0, decision_threshold=0.5):
    _, _, p0_hat = calibrate_threshold(n_calib=100, seed=1)
    tau, _ = calibrate_smoothed_threshold(p0_hat, n_calib=100, window=window,
                                            F_target=1.0, n_target=100)
    p_adv = 0.30

    print(f"{'n':>5} {'utilite PEG':>12} {'utilite I-POMDP':>16} "
          f"{'cout enum. naive (2^n)':>24}")

    for n in ns:
        # nombre ABSOLU de membres de coalition fixe (protocole corrige,
        # Section 7.9), pas une fraction de n -- c'est le point precis que le
        # papier montre comme determinant pour la comparaison
        n_coal = min(n_coalition_members, n // 4)
        frac_coalition = n_coal / n

        pop = Population(n, p_compliant=p0_hat, p_adv=p_adv,
                          frac_coalition=frac_coalition, coalition_sync_prob=0.35,
                          seed=seed)
        model = ReactiveModel(n, p0_hat)
        smoother = SmoothedSurprise(n, window=window)
        cand = candidate_sets(n, k_max=min(8, n - 1), mode="bounded_degree", seed=seed + 1)
        agents = list(range(n))
        true_adv = set(np.where(pop.types != 0)[0].tolist())

        obs_hist = {j: [] for j in agents}
        peg_utils, ipomdp_utils = [], []

        for t in range(T):
            o, _ = pop.step()
            for j in agents:
                obs_hist[j].append(o[j])
                if len(obs_hist[j]) > window:
                    obs_hist[j].pop(0)
                s = model.surprise(j, o[j])
                smoother.push_raw(j, s)
            if t < warmup:
                continue

            result = peg_step(t, agents, model, smoother, cand, tau=tau,
                               rho_min=0.30, w=window)
            flagged_peg = result["V"]
            y_true = [1 if j in true_adv else 0 for j in agents]
            y_pred_peg = [1 if j in flagged_peg else 0 for j in agents]

            posteriors = joint_exact_inference(obs_hist, p0_hat, p_adv)
            y_pred_ipomdp = [1 if posteriors[j] > decision_threshold else 0 for j in agents]

            def utility(y_true, y_pred, reward_tp=10, penalty_fp=-50, penalty_fn=-3):
                u = 0.0
                for yt, yp in zip(y_true, y_pred):
                    if yt == 1 and yp == 1:
                        u += reward_tp
                    elif yt == 0 and yp == 1:
                        u += penalty_fp
                    elif yt == 1 and yp == 0:
                        u += penalty_fn
                return u / len(y_true)

            peg_utils.append(utility(y_true, y_pred_peg))
            ipomdp_utils.append(utility(y_true, y_pred_ipomdp))

        print(f"{n:5d} {np.mean(peg_utils):12.3f} {np.mean(ipomdp_utils):16.3f} "
              f"{enumeration_cost_estimate(n):24.3e}")


if __name__ == "__main__":
    run()
