"""
peg_algorithm.py -- Algorithme 1 (Section 4.3 du papier)

Implementation fidele au pseudo-code :

  1. Surprise (O(n))       : une fois par agent observe, pas par paire
  2. Activation (O(n))     : V(t) = agents individuellement surprenants
  3. Largeur (O(n.k_max))  : E(t) restreint aux candidats de chaque agent
  3b. Coalition (O(|V(t)|^2.w)) : correlation restreinte aux agents deja actifs
  4. Profondeur (O(|E(t)|.d_max)) : mise a jour bayesienne sequentielle,
                                     uniquement sur les aretes actives

Respecte l'invariant central du papier : la profondeur n'est jamais calculee
pour une arete non activee (Invariant 1, Section 4.4).
"""

import numpy as np
from itertools import combinations


def candidate_sets(n, k_max=8, mode="bounded_degree", density=0.30, seed=0):
    """
    Construit Cand(i) pour chaque agent i (Section 4.2).

    mode="bounded_degree"   : k_max candidats fixes par agent (regime lineaire)
    mode="constant_density" : fraction `density` des autres agents (regime O(n^2))
    """
    rng = np.random.default_rng(seed)
    cand = {}
    for i in range(n):
        others = [j for j in range(n) if j != i]
        if mode == "bounded_degree":
            k = min(k_max, len(others))
            cand[i] = set(rng.choice(others, size=k, replace=False))
        elif mode == "constant_density":
            k = max(1, int(round(density * len(others))))
            cand[i] = set(rng.choice(others, size=k, replace=False))
        else:
            raise ValueError(f"mode inconnu : {mode}")
    return cand


def peg_step(t, agents, model, smoother, cand, tau, rho_min, w,
             active_edges_prev=None, hysteresis_horizon=0,
             compute_depth=False, d_max=10, eta=0.05, belief_update_fn=None):
    """
    Une iteration de l'Algorithme 1.

    Parametres
    ----------
    agents : liste des indices d'agents [0..n-1]
    model  : instance de env.ReactiveModel
    smoother : instance de env.SmoothedSurprise (deja mise a jour avec les
               observations du pas courant via smoother.push_raw)
    cand   : dict agent -> ensemble de candidats (sortie de candidate_sets)
    tau    : seuil d'activation (Definition 4bis)
    rho_min : seuil de correlation de coalition (Definition 5)
    w      : taille de la fenetre glissante
    active_edges_prev, hysteresis_horizon : mecanisme d'hysteresis (Section 3.4),
               ignores si hysteresis_horizon == 0
    compute_depth : si True, calcule la profondeur (Etape 4) pour chaque arete
               active -- necessite belief_update_fn(i, j, d) -> (b_d, b_d1, D_KL)
    d_max, eta : parametres de la Definition 7

    Retourne
    -------
    dict avec cles : 'V' (agents actifs), 'E' (aretes actives),
    'C' (paires en coalition suspectee), 'depth' (profondeur par arete, si
    compute_depth), 'rho' (dict des correlations calculees)
    """
    n = len(agents)

    # --- Etape 1+2 : surprise individuelle et activation (O(n)) ---
    V = set(j for j in agents if smoother.smoothed(j) > tau)

    # --- Etape 3a : largeur, restreinte aux candidats ET actifs (O(n.k_max)) ---
    E = set()
    for i in agents:
        for j in cand[i]:
            if j in V:
                E.add((i, j))

    # --- Etape 3b : coalition (Definition 5), restreinte a V(t) (O(|V|^2.w)) ---
    # Conforme a la borne de cout du papier (Section 4.5). Voir README.md,
    # section "Ecarts connus par rapport au papier" pour la discussion du
    # compromis rappel/cout que ce choix implique pour les coalitions a tres
    # faible taux d'activite moyenne.
    C = set()
    rho_values = {}
    V_list = sorted(V)
    for j, k in combinations(V_list, 2):
        sj = smoother.raw_window(j)
        sk = smoother.raw_window(k)
        if len(sj) >= 2 and len(sk) >= 2 and len(sj) == len(sk):
            if np.std(sj) > 0 and np.std(sk) > 0:
                rho = np.corrcoef(sj, sk)[0, 1]
            else:
                rho = 0.0
        else:
            rho = 0.0
        rho_values[(j, k)] = rho
        if rho > rho_min:
            C.add((j, k))
            E.add((j, k))

    # --- Hysteresis (Section 3.4) ---
    if hysteresis_horizon > 0 and active_edges_prev is not None:
        for edge, since in active_edges_prev.items():
            if t - since < hysteresis_horizon:
                E.add(edge)

    # --- Etape 4 : profondeur, restreinte aux aretes actives (O(|E|.d_max)) ---
    depth = {}
    beliefs = {}
    if compute_depth and belief_update_fn is not None:
        for (i, j) in E:
            d = 0
            b_d = None
            while d < d_max:
                b_d, b_d1, s_res = belief_update_fn(i, j, d)
                if s_res < eta:
                    break
                d += 1
                b_d = b_d1
            depth[(i, j)] = d
            beliefs[(i, j)] = b_d

    return {"V": V, "E": E, "C": C, "depth": depth, "beliefs": beliefs,
            "rho": rho_values}
