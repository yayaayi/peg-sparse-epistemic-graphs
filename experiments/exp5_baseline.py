"""
exp5_baseline.py -- Experience 5 : Comparaison a une baseline exhaustive
(Section 7.5 du papier)

Compare PEG (activation par surprise, seules les paires actives recoivent une
inference bayesienne complete) a une baseline exhaustive (toutes les paires
candidates recoivent systematiquement l'inference complete, sans filtrage).

RESOLU (voir README.md, Note 2). Deux bugs/choix arbitraires successifs ont
ete corriges dans l'ordre :
  1. La toute premiere version confondait "a recu une inference" avec "a ete
     classe adversaire", penalisant injustement l'exhaustive (PEG semblait
     dominer massivement sur les deux axes).
  2. Une premiere correction a introduit un vrai posterieur bayesien mais
     avec un seuil de decision ARBITRAIRE (0.5), ce qui donnait PEG
     legerement SUPERIEUR a l'exhaustive -- un resultat different du (1)
     mais tout aussi peu fonde, puisque rien ne justifiait ce seuil precis.
  3. Cette version utilise desormais le seuil BAYES-OPTIMAL, derive
     analytiquement de la structure de recompense/penalite elle-meme
     (reward_tp, penalty_fp, penalty_fn), et non choisi a la main. Sous ce
     seuil, PEG et l'exhaustive obtiennent une utilite IDENTIQUE a chaque
     taille de population testee (verifie sur 5 graines) : aucun agent
     compliant ne depasse jamais ce seuil plus conservateur, donc evaluer
     tout le monde (exhaustif) ou seulement les suspects pre-filtres (PEG)
     ne change rien a la decision finale -- seulement au cout de calcul.
     C'est le resultat theoriquement le mieux fonde des trois versions
     successives, et celui desormais rapporte dans le papier.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from env import Population, ReactiveModel, SmoothedSurprise, calibrate_threshold, calibrate_smoothed_threshold
from peg_algorithm import candidate_sets, peg_step


def posterior_adversary(obs_window, p_compliant, p_adv, prior_adv=0.15):
    """Posterieur bayesien P(adversaire | fenetre d'observations), meme
    logique que joint_exact_inference (exp9_ipomdp.py) mais pour un seul
    agent evalue isolement -- c'est exactement l'inference qu'un mecanisme
    de decision bayesien complet appliquerait a CE candidat, qu'il ait ete
    filtre par PEG ou inclus par l'exhaustive."""
    obs = np.asarray(obs_window)
    if len(obs) == 0:
        return prior_adv
    log_lik_compliant = np.sum(obs * np.log(p_compliant) + (1 - obs) * np.log(1 - p_compliant))
    log_lik_adv = np.sum(obs * np.log(p_adv) + (1 - obs) * np.log(1 - p_adv))
    log_post_compliant = log_lik_compliant + np.log(1 - prior_adv)
    log_post_adv = log_lik_adv + np.log(prior_adv)
    m = max(log_post_compliant, log_post_adv)
    return np.exp(log_post_adv - m) / (np.exp(log_post_adv - m) + np.exp(log_post_compliant - m))


def bayes_optimal_threshold(reward_tp=10, penalty_fp=-50, penalty_fn=-3):
    """Seuil de decision derive analytiquement de la structure d'utilite
    elle-meme (minimisation du risque bayesien), plutot qu'un seuil
    arbitraire choisi a la main. Flag si E[utilite|flag] > E[utilite|non
    flag] :
        P*reward_tp + (1-P)*penalty_fp > P*penalty_fn + (1-P)*0
    ce qui se resout en P > -penalty_fp / (reward_tp - penalty_fn - penalty_fp).
    Pour (10, -50, -3) : P > 50/63 ~= 0.794 -- tres different du seuil
    arbitraire de 0.5 utilise par defaut ci-dessous."""
    return -penalty_fp / (reward_tp - penalty_fn - penalty_fp)


def decision_utility(evaluated_agents, obs_hist, all_candidate_pairs, true_adversaries,
                       p_compliant, p_adv, reward_tp=10, penalty_fp=-50, penalty_fn=-3,
                       threshold=None):
    """
    Fonction d'utilite asymetrique (Section 6.5, Metriques) : recompense une
    detection correcte, penalise une fausse accusation, penalise un
    adversaire manque. `evaluated_agents` = agents recevant une inference
    (PEG : agents dans E(t) ; baseline : tous les candidats) -- mais un agent
    evalue n'est desormais "flagged" QUE si son posterieur bayesien depasse
    `threshold`, pas simplement parce qu'il a ete evalue. Par defaut,
    `threshold` est calcule comme le seuil Bayes-optimal derive de
    (reward_tp, penalty_fp, penalty_fn) -- pas une valeur arbitraire.
    """
    if threshold is None:
        threshold = bayes_optimal_threshold(reward_tp, penalty_fp, penalty_fn)
    flagged_agents = set()
    for j in evaluated_agents:
        post = posterior_adversary(obs_hist[j], p_compliant, p_adv)
        if post > threshold:
            flagged_agents.add(j)

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
    p_adv_true = 0.30

    print(f"{'n':>5} {'cout PEG':>10} {'cout exhaustif':>15} "
          f"{'utilite PEG':>12} {'utilite exhaustif':>18}")

    results = {}
    for n in ns:
        peg_costs, exh_costs, peg_utils, exh_utils = [], [], [], []
        for seed in seeds:
            pop = Population(n, p_compliant=p0_hat, p_adv=p_adv_true,
                              frac_adv=frac_adv, seed=seed)
            model = ReactiveModel(n, p0_hat)
            smoother = SmoothedSurprise(n, window=window)
            cand = candidate_sets(n, k_max=8, mode="bounded_degree", seed=seed + 1)
            agents = list(range(n))
            true_adv = set(np.where(pop.types == 1)[0].tolist())

            # historique BRUT des observations (0/1), separe de la surprise,
            # necessaire pour le posterieur bayesien
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
                peg_costs.append(len(result["E"]))
                exh_costs.append(len(all_candidate_pairs))
                peg_utils.append(decision_utility(peg_evaluated, obs_hist, all_candidate_pairs,
                                                    true_adv, p0_hat, p_adv_true))
                exh_utils.append(decision_utility(all_candidate_agents, obs_hist, all_candidate_pairs,
                                                    true_adv, p0_hat, p_adv_true))

        results[n] = dict(peg_cost=np.mean(peg_costs), exh_cost=np.mean(exh_costs),
                            peg_util=np.mean(peg_utils), exh_util=np.mean(exh_utils))
        print(f"{n:5d} {np.mean(peg_costs):10.1f} {np.mean(exh_costs):15.1f} "
              f"{np.mean(peg_utils):12.3f} {np.mean(exh_utils):18.3f}")

    return results


if __name__ == "__main__":
    run()
