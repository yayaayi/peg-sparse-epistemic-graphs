"""
exp11_burst_decision.py -- Experience 11 : Correction par modele de croyance
conscient des regimes (Section 7.11 du papier)

Un adversaire de coalition ne devie que par rafales synchronisees rares : sa
surprise lissee moyenne sur toute la fenetre reste proche du comportement
conforme, ce qui degrade la decision finale meme quand l'activation
fonctionne (constat de l'Experience 8, Section 7.8). Ce script implemente la
correction qui resout ce probleme pour le cas SYNCHRONISE : conditionner la
decision aux fenetres de rafale detectees plutot que de moyenner sur toute
la periode.

Principe : la detection de coalition par correlation (Definition 5) indique
deja A QUELS INSTANTS une coordination synchronisee s'est produite (les pas
ou l'activite agregee des agents actifs depasse un seuil). La decision
n'utilise alors que les observations tombant dans ces fenetres, au lieu de
la fenetre complete.

Teste egalement la robustesse a la desynchronisation (chaque membre devie
avec un decalage aleatoire autour de l'instant commun) -- reproduit
l'effondrement documente dans le papier au-dela d'un decalage modeste.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from env import Population, ReactiveModel, SmoothedSurprise, calibrate_threshold, calibrate_smoothed_threshold


def simulate_coalition_bursts(n_coalition=8, T=400, sync_prob=0.15,
                                desync_spread=0, p_burst=0.85, p_base=0.05,
                                seed=0):
    """
    Simule une coalition dont chaque membre devie fortement uniquement
    pendant les fenetres de rafale synchronisee, avec un decalage aleatoire
    optionnel de +/- desync_spread pas autour de l'instant commun.
    """
    rng = np.random.default_rng(seed)
    burst_steps = rng.random(T) < sync_prob
    obs = np.zeros((n_coalition, T), dtype=int)
    for m in range(n_coalition):
        offset = rng.integers(-desync_spread, desync_spread + 1) if desync_spread > 0 else 0
        member_burst = np.zeros(T, dtype=bool)
        for t in range(T):
            src_t = t - offset
            if 0 <= src_t < T and burst_steps[src_t]:
                member_burst[t] = True
        p = np.where(member_burst, p_burst, p_base)
        obs[m] = (rng.random(T) < p).astype(int)
    return obs, burst_steps


def decide_windowed_mean(obs_member, p_base, p_burst, reward_tp=10,
                           penalty_fp=-50, penalty_fn=-3, prior_adv=0.5):
    """Decision de reference : moyenne sur toute la fenetre (modele a taux
    constant, comme dans l'Experience 8)."""
    rate = obs_member.mean()
    p_mid = (p_base + p_burst) / 2
    return 1 if rate > p_mid else 0


def decide_burst_conditioned(obs_member, burst_mask, p_base, p_burst):
    """Decision corrigee : n'utilise que les observations tombant dans les
    fenetres de rafale detectees par correlation."""
    if burst_mask.sum() == 0:
        return 0
    rate_in_burst = obs_member[burst_mask].mean()
    p_mid = (p_base + p_burst) / 2
    return 1 if rate_in_burst > p_mid else 0


def detect_burst_windows(obs, threshold_frac=0.5):
    """Detection des fenetres de rafale par activite agregee (proxy simplifie
    du mecanisme de correlation de coalition, Definition 5, applique dans le
    temps plutot qu'entre paires) : un pas est declare "en rafale" si une
    fraction suffisante des membres devient active simultanement."""
    activity_frac = obs.mean(axis=0)
    return activity_frac > threshold_frac


def utility_of_decision(decisions, reward_tp=10, penalty_fp=-50, penalty_fn=-3,
                          true_label=1):
    u = 0.0
    for d in decisions:
        if true_label == 1 and d == 1:
            u += reward_tp
        elif true_label == 1 and d == 0:
            u += penalty_fn
    return u / len(decisions)


def run_synchronized(n_coalition=8, T=400, n_reps=10, p_base=0.05, p_burst=0.85):
    print("-- Cas synchronise --")
    utils_mean, utils_burst = [], []
    for rep in range(n_reps):
        obs, true_burst_steps = simulate_coalition_bursts(
            n_coalition, T, desync_spread=0, p_base=p_base, p_burst=p_burst, seed=rep)
        detected_bursts = detect_burst_windows(obs)

        decisions_mean = [decide_windowed_mean(obs[m], p_base, p_burst) for m in range(n_coalition)]
        decisions_burst = [decide_burst_conditioned(obs[m], detected_bursts, p_base, p_burst)
                            for m in range(n_coalition)]

        utils_mean.append(utility_of_decision(decisions_mean))
        utils_burst.append(utility_of_decision(decisions_burst))

    print(f"Utilite (decision moyenne sur toute la fenetre) : "
          f"{np.mean(utils_mean):.3f} +/- {np.std(utils_mean):.3f}")
    print(f"Utilite (decision conditionnee aux rafales)      : "
          f"{np.mean(utils_burst):.3f} +/- {np.std(utils_burst):.3f}")
    if np.std(utils_mean) == 0 and np.std(utils_burst) == 0:
        print("Ecart parfaitement separe sur ce protocole synthetique "
              "(variance nulle entre repetitions) : test t et d de Cohen "
              "non definis, la separation est deterministe ici plutot que "
              "statistique -- voir README.md.\n")
    else:
        from scipy.stats import ttest_rel
        t_stat, p_val = ttest_rel(utils_burst, utils_mean)
        diff = np.array(utils_burst) - np.array(utils_mean)
        d_cohen = diff.mean() / diff.std() if diff.std() > 0 else float("nan")
        print(f"t={t_stat:.3f}  p={p_val:.6f}  d approx.={d_cohen:.2f}\n")


def run_desync_robustness(n_coalition=8, T=400, n_reps=10, p_base=0.05,
                            p_burst=0.85, spreads=(0, 1, 2, 3, 5)):
    print("-- Robustesse a la desynchronisation --")
    for spread in spreads:
        recalls = []
        for rep in range(n_reps):
            obs, _ = simulate_coalition_bursts(n_coalition, T, desync_spread=spread,
                                                 p_base=p_base, p_burst=p_burst, seed=rep + 100)
            detected_bursts = detect_burst_windows(obs)
            decisions = [decide_burst_conditioned(obs[m], detected_bursts, p_base, p_burst)
                         for m in range(n_coalition)]
            recalls.append(np.mean(decisions))
        print(f"  decalage +/-{spread:2d} pas : rappel moyen = {np.mean(recalls):.3f}")


if __name__ == "__main__":
    run_synchronized()
    run_desync_robustness()
