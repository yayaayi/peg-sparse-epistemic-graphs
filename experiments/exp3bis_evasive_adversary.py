"""
exp3bis_evasive_adversary.py -- Experience 3bis : adversaire evasif sous
regime non stationnaire (Section 7.3bis du papier).

Ce script n'existait pas dans le depot ; construit a partir du protocole
decrit dans le papier pour (a) reproduire le resultat de base (profondeur
adaptative significativement moins bonne que profondeur fixe sous regime
non stationnaire), puis (b) tester le correctif de detection de rupture
propose lors d'une revue externe (test de Page-Hinkley sur la surprise
brute, reinitialisation de la croyance a la detection d'une bascule).

PROTOCOLE (tel que decrit dans le papier) : 100 agents, 30% evasifs dont
le taux de deviation suit une chaine de Markov cachee a 2 etats (bas/haut),
probabilite de bascule 0.15 par pas ; comparaison profondeur fixe d=1
contre profondeur adaptative (Definition 7, d_max=10) ; 8 graines, test
apparie.

Semantique de la profondeur (Definitions 6-7) : la croyance b_i^(d) sur le
taux de deviation de j est mise a jour sequentiellement a partir des d
observations les plus recentes de la paire (i,j) ; la profondeur adaptative
augmente d tant que le gain d'evidence residuelle (divergence KL entre
b^(d+1) et b^(d)) depasse un seuil tau_d.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.stats import beta as beta_dist


class HiddenMarkovAgent:
    """Agent dont le taux de deviation bascule entre un etat bas et un etat
    haut selon une chaine de Markov cachee (probabilite de bascule fixe par
    pas), simulant un adversaire evasif changeant de mode operatoire."""

    def __init__(self, p_low, p_high, switch_prob, seed):
        self.p_low = p_low
        self.p_high = p_high
        self.switch_prob = switch_prob
        self.rng = np.random.default_rng(seed)
        self.state_high = self.rng.random() < 0.5

    def step(self):
        if self.rng.random() < self.switch_prob:
            self.state_high = not self.state_high
        p = self.p_high if self.state_high else self.p_low
        return int(self.rng.random() < p), self.state_high


def belief_at_depth(obs_window, p_low, p_high, prior_high=0.5):
    """Croyance bayesienne P(etat haut | d observations les plus recentes),
    modele beta-bernoulli simplifie a deux hypotheses (p_low, p_high)."""
    obs = np.asarray(obs_window)
    if len(obs) == 0:
        return prior_high
    log_lik_high = np.sum(obs * np.log(p_high) + (1 - obs) * np.log(1 - p_high))
    log_lik_low = np.sum(obs * np.log(p_low) + (1 - obs) * np.log(1 - p_low))
    log_post_high = log_lik_high + np.log(prior_high)
    log_post_low = log_lik_low + np.log(1 - prior_high)
    m = max(log_post_high, log_post_low)
    return np.exp(log_post_high - m) / (np.exp(log_post_high - m) + np.exp(log_post_low - m))


def kl_bernoulli(p, q, eps=1e-9):
    p, q = np.clip(p, eps, 1 - eps), np.clip(q, eps, 1 - eps)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def page_hinkley_detect(values, delta=0.01, threshold=3.0):
    """Test de Page-Hinkley sequentiel sur une serie de surprise brute :
    retourne l'indice du dernier point de rupture detecte, ou None."""
    values = np.asarray(values)
    if len(values) < 3:
        return None
    m_t = 0.0
    min_m = 0.0
    mean_est = values[0]
    last_change = None
    for t in range(1, len(values)):
        mean_est = mean_est + (values[t] - mean_est) / (t + 1)
        m_t += values[t] - mean_est - delta
        min_m = min(min_m, m_t)
        if (m_t - min_m) > threshold:
            last_change = t
            m_t, min_m = 0.0, 0.0
    return last_change


def decision_utility(flagged, true_evasive, all_agents, reward_tp=10, penalty_fp=-50, penalty_fn=-3):
    u = 0.0
    for j in all_agents:
        is_ev, is_fl = j in true_evasive, j in flagged
        if is_ev and is_fl:
            u += reward_tp
        elif is_fl and not is_ev:
            u += penalty_fp
        elif is_ev and not is_fl:
            u += penalty_fn
    return u / len(all_agents)


def run(n=100, frac_evasive=0.30, p_low=0.10, p_high=0.55, switch_prob=0.15,
         T=300, warmup=20, d_max=10, tau_d=0.02, seed=0, use_changepoint_fix=False,
         base_window=5):
    """base_window : nombre d'observations par palier de profondeur (d=1
    correspond a base_window observations, d=2 a 2*base_window, etc.).

    TACHE : contrairement a une premiere tentative qui demandait "cet agent
    est-il de type evasif" (une question ou plus de donnees aide toujours,
    puisque le taux moyen a long terme des evasifs differe de celui des
    compliants quel que soit le regime courant), cette version demande "cet
    agent est-il EN CE MOMENT dans son etat de deviation haute" -- une
    classification de l'etat courant, pas du type de l'agent. C'est cette
    tache qui rend la dilution par de vieilles donnees d'un regime passe
    reellement trompeuse, conformement a l'interpretation donnee dans le
    papier (Section 7.3bis)."""
    rng_types = np.random.default_rng(seed)
    is_evasive = rng_types.random(n) < frac_evasive
    agents_obj = []
    for j in range(n):
        if is_evasive[j]:
            agents_obj.append(HiddenMarkovAgent(p_low, p_high, switch_prob, seed=seed * 1000 + j))
        else:
            agents_obj.append(HiddenMarkovAgent(p_low, p_low, 0.0, seed=seed * 1000 + j))

    obs_hist = [[] for _ in range(n)]
    true_state_hist = [[] for _ in range(n)]
    for t in range(T):
        for j in range(n):
            o, state_high = agents_obj[j].step()
            obs_hist[j].append(o)
            true_state_hist[j].append(state_high)

    all_agents = list(range(n))
    # cible de decision : l'agent est-il EN CE MOMENT (dernier pas) dans son
    # etat de deviation haute -- vrai uniquement pour les evasifs actuellement
    # en phase haute, jamais pour les compliants (jamais en phase haute)
    true_high_now = set(j for j in all_agents if true_state_hist[j][-1])

    # --- profondeur fixe d=1 : un seul palier (base_window observations) ---
    flagged_fixed = set()
    for j in all_agents:
        belief = belief_at_depth(obs_hist[j][-base_window:], p_low, p_high)
        if belief > 0.5:
            flagged_fixed.add(j)

    # --- profondeur adaptative (Definition 7) ---
    flagged_adaptive = set()
    for j in all_agents:
        window = obs_hist[j][warmup:] if not use_changepoint_fix else obs_hist[j]
        if use_changepoint_fix:
            cp = page_hinkley_detect(window)
            if cp is not None:
                window = window[cp:]
        d = 1
        prev_belief = belief_at_depth(window[-base_window:], p_low, p_high)
        while d < d_max:
            next_belief = belief_at_depth(window[-((d + 1) * base_window):], p_low, p_high)
            residual = kl_bernoulli(next_belief, prev_belief)
            if residual <= tau_d:
                break
            prev_belief = next_belief
            d += 1
        final_belief = belief_at_depth(window[-(d * base_window):], p_low, p_high)
        if final_belief > 0.5:
            flagged_adaptive.add(j)

    u_fixed = decision_utility(flagged_fixed, true_high_now, all_agents)
    u_adaptive = decision_utility(flagged_adaptive, true_high_now, all_agents)
    return u_fixed, u_adaptive


def run_multi_seed(n_seeds=8, use_changepoint_fix=False, **kwargs):
    fixed, adaptive = [], []
    for s in range(n_seeds):
        uf, ua = run(seed=s, use_changepoint_fix=use_changepoint_fix, **kwargs)
        fixed.append(uf)
        adaptive.append(ua)
    fixed, adaptive = np.array(fixed), np.array(adaptive)
    from scipy.stats import ttest_rel
    t_stat, p_val = ttest_rel(adaptive, fixed)
    label = "AVEC correction (detection de rupture)" if use_changepoint_fix else "SANS correction (originale)"
    print(f"=== Experience 3bis -- {label} -- {n_seeds} graines ===")
    print(f"Profondeur fixe (d=1)      : {fixed.mean():.3f} +/- {fixed.std():.3f}")
    print(f"Profondeur adaptative      : {adaptive.mean():.3f} +/- {adaptive.std():.3f}")
    print(f"t={t_stat:.3f}  p={p_val:.4f}")
    return fixed, adaptive, p_val


if __name__ == "__main__":
    print("Reproduction du resultat original (sans correction) :")
    run_multi_seed(n_seeds=8, use_changepoint_fix=False)
    print()
    print("Avec le correctif de detection de rupture (Page-Hinkley) :")
    run_multi_seed(n_seeds=8, use_changepoint_fix=True)
