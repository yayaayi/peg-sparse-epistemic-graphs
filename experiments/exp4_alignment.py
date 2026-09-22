"""
exp4_alignment.py -- Experience 4 : Alignement activation-contribution
(Conjecture 2, Section 7.4 du papier)

Teste si la surprise lissee S_barre_i^j(t) est correlee a la contribution
marginale reelle de la relation (i,j) a la decision :

    Delta U(i,j) = U(a | e_ij) - U(a | vide)

ou U est la fonction d'utilite decisionnelle (Section 6.5). Correlation de
Spearman entre surprise et contribution marginale sur l'ensemble des paires
(actives et non actives).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.stats import spearmanr
from env import Population, ReactiveModel, SmoothedSurprise, calibrate_threshold, calibrate_smoothed_threshold
from peg_algorithm import candidate_sets, peg_step


def marginal_contribution(j, o_j, is_adversary, reward_tp=10, penalty_fp=-50,
                            penalty_fn=-3, detection_prob_given_edge=0.9):
    """
    Contribution marginale attendue de l'ajout de l'arete (i,j) a la decision :
    approxime U(a|e_ij) - U(a|vide) par l'esperance de gain d'information sur
    le type de j, ponderee par la structure de cout asymetrique (Section 6.5).
    Sert de proxy simule pour Delta U(i,j) dans cette reconstruction.
    """
    if is_adversary:
        # avec l'arete : detection probable (a detection_prob_given_edge) ;
        # sans l'arete : jamais detecte (utilite de reference = penalite fn)
        u_with = detection_prob_given_edge * reward_tp + (1 - detection_prob_given_edge) * penalty_fn
        u_without = penalty_fn
    else:
        # avec l'arete : risque de fausse accusation ; sans l'arete : rien
        false_accuse_prob = 1 - detection_prob_given_edge
        u_with = false_accuse_prob * penalty_fp
        u_without = 0.0
    return u_with - u_without


def run(n=100, frac_adv=0.15, T=400, warmup=60, seed=0):
    window = 20
    _, _, p0_hat = calibrate_threshold(n_calib=100, seed=1)
    tau, _ = calibrate_smoothed_threshold(p0_hat, n_calib=100, window=window,
                                            F_target=1.0, n_target=100)

    pop = Population(n, p_compliant=p0_hat, p_adv=0.30, frac_adv=frac_adv, seed=seed)
    model = ReactiveModel(n, p0_hat)
    smoother = SmoothedSurprise(n, window=window)
    cand = candidate_sets(n, k_max=8, mode="bounded_degree", seed=seed + 1)
    agents = list(range(n))
    adversaries = set(np.where(pop.types == 1)[0].tolist())

    surprises, contributions = [], []

    for t in range(T):
        o, _ = pop.step()
        for j in agents:
            s = model.surprise(j, o[j])
            smoother.push_raw(j, s)
        if t < warmup:
            continue

        for i in agents:
            for j in cand[i]:
                s_bar = smoother.smoothed(j)
                du = marginal_contribution(j, o[j], j in adversaries)
                surprises.append(s_bar)
                contributions.append(du)

    rho, pval = spearmanr(surprises, contributions)
    print(f"n={n}  paires evaluees={len(surprises)}")
    print(f"Correlation de Spearman (surprise, contribution marginale) : "
          f"rho={rho:.3f}  p={pval:.2e}")

    # verdict par rapport au critere de refutation defini a priori
    zero_surprise_mask = np.array(surprises) <= np.percentile(surprises, 15)
    contrib_at_low_surprise = np.array(contributions)[zero_surprise_mask]
    print(f"Contribution marginale moyenne au 15e percentile de surprise "
          f"le plus bas : {contrib_at_low_surprise.mean():.3f}")
    return rho, pval, contrib_at_low_surprise.mean()


def run_multi_seed(n_seeds=5, **kwargs):
    """Boucle sur n_seeds graines independantes -- reproduit la statistique
    inter-graines du papier (Section 7.4, Conjecture 2 : N=5 graines,
    rho=0,617+/-0,005 rapporte)."""
    rhos, low_surprise_contribs = [], []
    for s in range(n_seeds):
        print(f"\n--- graine {s} ---")
        rho, _, low_contrib = run(seed=s, **kwargs)
        rhos.append(rho)
        low_surprise_contribs.append(low_contrib)
    rhos = np.array(rhos)
    print(f"\n=== Agrege sur {n_seeds} graines (seeds=0..{n_seeds-1}) ===")
    print(f"rho de Spearman : {rhos.mean():.3f} +/- {rhos.std():.3f}")
    print(f"Contribution marginale (bas percentile), toutes graines : "
          f"{np.mean(low_surprise_contribs):.3f}")
    return rhos


if __name__ == "__main__":
    run_multi_seed(n_seeds=5)
