"""
exp2_quater_group_detection.py -- Tentative de detection de coalition par
GROUPE plutot que par paire isolee (Section 7.2, exploratoire).

MOTIVATION : le mecanisme actuel (Definition 5) traite chaque paire (j,k)
independamment. Un groupe de 5 membres coalises produit 10 paires evaluees
separement -- l'information que ces 10 paires appartiennent au MEME groupe
n'est jamais exploitee. Cette tentative teste si l'utiliser ameliore le F1.

METHODE (tractable, pas de recherche combinatoire sur tous les sous-
ensembles) :
  1. Construire un graphe permissif a partir des correlations par paire
     (seuil bas, rho_seed) -- memes calculs que le mecanisme existant.
  2. En extraire les composantes connexes : des groupes CANDIDATS, obtenus
     en O(n^2) comme le mecanisme actuel, pas en cherchant tous les
     sous-ensembles possibles (qui serait exponentiel).
  3. Pour chaque groupe candidat de taille >= 2, calculer une statistique
     JOINTE par membre : correlation de la surprise brute de cet agent
     avec la moyenne ("consensus") des autres membres du groupe (leave-
     one-out) -- une information que la paire isolee n'a pas.
  4. Classer un agent comme membre de coalition si cette statistique de
     consensus depasse un second seuil, rho_consensus.

Comparaison au mecanisme pairwise existant (F1=0.353+/-0.044) sur le F1
obtenu au niveau des PAIRES (memes candidats, meme population, seuils
optimises independamment pour une comparaison equitable).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from itertools import combinations
from env import Population, ReactiveModel, SmoothedSurprise, calibrate_threshold, calibrate_smoothed_threshold


def connected_components(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for (a, b) in edges:
        union(a, b)

    groups = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)
    return [g for g in groups.values() if len(g) >= 2]


def run(n=60, frac_adv=0.08, frac_coalition=0.12, window=25, T=400,
         warmup=60, seed=0, coalition_sync_prob=0.35, p_adv=0.55,
         rho_seed=0.25, rho_consensus_values=None):
    if rho_consensus_values is None:
        rho_consensus_values = np.arange(0.10, 0.55, 0.05)

    _, _, p0_hat = calibrate_threshold(n_calib=100, seed=1)
    pop = Population(n, p_compliant=p0_hat, p_adv=p_adv, frac_adv=frac_adv,
                       frac_coalition=frac_coalition,
                       coalition_sync_prob=coalition_sync_prob, seed=seed)
    model = ReactiveModel(n, p0_hat)
    agents = list(range(n))
    coalition_idx = set(np.where(pop.types == 2)[0].tolist())
    true_pairs = set(combinations(sorted(coalition_idx), 2))

    raw_surprise = {j: [] for j in agents}
    for t in range(T):
        o, _ = pop.step()
        for j in agents:
            raw_surprise[j].append(model.surprise(j, o[j]))
    raw_arr = {j: np.array(raw_surprise[j][warmup:]) for j in agents}

    # Etape 1-2 : graphe permissif -> composantes connexes (groupes candidats)
    edges = []
    pairwise_rho = {}
    for i, j in combinations(agents, 2):
        rho = np.corrcoef(raw_arr[i], raw_arr[j])[0, 1]
        if np.isnan(rho):
            rho = 0.0
        pairwise_rho[(i, j)] = rho
        if rho > rho_seed:
            edges.append((i, j))
    groups = connected_components(n, edges)

    # Etape 3 : statistique de consensus par membre, a l'interieur de
    # chaque groupe candidat
    consensus_score = {}
    for group in groups:
        for j in group:
            others = [k for k in group if k != j]
            if not others:
                continue
            group_mean_signal = np.mean([raw_arr[k] for k in others], axis=0)
            r = np.corrcoef(raw_arr[j], group_mean_signal)[0, 1]
            consensus_score[j] = 0.0 if np.isnan(r) else r

    # Etape 4 : evaluation au niveau des PAIRES pour comparaison directe
    # avec le mecanisme pairwise existant -- une paire est predite
    # "coalition" si les DEUX membres depassent le seuil de consensus ET
    # appartiennent au meme groupe candidat
    best_f1, best_precision, best_recall, best_rho = 0, 0, 0, None
    group_of = {}
    for gi, group in enumerate(groups):
        for j in group:
            group_of[j] = gi

    for rho_c in rho_consensus_values:
        tp = fp = fn = 0
        for i, j in combinations(agents, 2):
            same_group = group_of.get(i) is not None and group_of.get(i) == group_of.get(j)
            both_high = (consensus_score.get(i, -1) > rho_c and
                          consensus_score.get(j, -1) > rho_c)
            pred = same_group and both_high
            is_true = (i, j) in true_pairs
            if pred and is_true:
                tp += 1
            elif pred and not is_true:
                fp += 1
            elif (not pred) and is_true:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        if f1 > best_f1:
            best_f1, best_precision, best_recall, best_rho = f1, precision, recall, rho_c

    return dict(f1=best_f1, precision=best_precision, recall=best_recall,
                 rho_consensus=best_rho, n_groups=len(groups))


def run_multi_seed(n_seeds=5, **kwargs):
    f1s, precisions, recalls = [], [], []
    for s in range(n_seeds):
        r = run(seed=s, **kwargs)
        f1s.append(r["f1"]); precisions.append(r["precision"]); recalls.append(r["recall"])
    f1s, precisions, recalls = np.array(f1s), np.array(precisions), np.array(recalls)
    print(f"=== Detection de coalition par GROUPE -- {n_seeds} graines ===")
    print(f"F1 optimal    : {f1s.mean():.3f} +/- {f1s.std():.3f}")
    print(f"Precision     : {precisions.mean():.3f} +/- {precisions.std():.3f}")
    print(f"Rappel        : {recalls.mean():.3f} +/- {recalls.std():.3f}")
    print(f"(reference, mecanisme pairwise original : F1=0.353+/-0.044)")
    return f1s, precisions, recalls


if __name__ == "__main__":
    run_multi_seed(n_seeds=5)
