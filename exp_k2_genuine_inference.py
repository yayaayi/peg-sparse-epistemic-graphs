"""
exp_k2_genuine_inference.py -- Version genuine (fondee sur une vraie
inference de vraisemblance, pas une heuristique de dispersion ad hoc) du
signal de second ordre contre l'adversaire a quota.

Utilise k_order_inference.py, qui n'accede JAMAIS au type reel de l'agent
dans sa logique de decision (verifie par test AST inclus dans ce module).
Le type reel n'est utilise ICI, dans ce script d'evaluation, qu'APRES coup,
pour calculer le rappel et le taux de faux positifs -- jamais comme entree
du mecanisme de detection lui-meme.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from env_recursive import CamouflagingPopulation
from k_order_inference import adaptive_k_order


def run(n=80, frac_naif=0.10, frac_camouflant=0.10, p_adv=0.135, window=20,
        T=1200, warmup=200, tau_order=2.0, n_recent_blocks=10, seed=0):
    pop = CamouflagingPopulation(n, p_adv=p_adv, window=window, frac_naif=frac_naif,
                                   frac_camouflant=frac_camouflant, seed=seed)
    hist = [[] for _ in range(n)]
    for t in range(T):
        o, types = pop.step()
        for j in range(n):
            hist[j].append(o[j])

    p_population = np.mean([np.mean(hist[j][warmup:]) for j in range(n)])

    naif_idx = set(np.where(types == 1)[0].tolist())
    camo_idx = set(np.where(types == 2)[0].tolist())
    compliant_idx = set(np.where(types == 0)[0].tolist())

    k_alloue = {}
    suspect_k2 = {}
    for j in range(n):
        o = np.array(hist[j][warmup:])
        k, suspect = adaptive_k_order(o, window=window, tau_order=tau_order,
                                        n_recent_blocks=n_recent_blocks,
                                        p_population=p_population)
        k_alloue[j] = k
        suspect_k2[j] = suspect

    def rate(idx_set, pred_dict):
        vals = [pred_dict[j] for j in idx_set]
        return np.mean(vals) if vals else 0.0

    recall_k1_naif = rate(naif_idx, {j: (k_alloue[j] >= 1) for j in range(n)})
    recall_k1_camo = rate(camo_idx, {j: (k_alloue[j] >= 1) for j in range(n)})
    recall_k2_naif = rate(naif_idx, suspect_k2)
    recall_k2_camo = rate(camo_idx, suspect_k2)
    fpr_k1 = rate(compliant_idx, {j: (k_alloue[j] >= 1) for j in range(n)})
    fpr_k2 = rate(compliant_idx, suspect_k2)
    promotion_rate = np.mean([k_alloue[j] == 2 for j in range(n)])

    return {
        "recall_k1_naif": recall_k1_naif, "recall_k1_camo": recall_k1_camo,
        "recall_k2_naif": recall_k2_naif, "recall_k2_camo": recall_k2_camo,
        "fpr_k1": fpr_k1, "fpr_k2": fpr_k2, "promotion_rate": promotion_rate,
    }


if __name__ == "__main__":
    results = [run(seed=s) for s in range(20)]
    print("=== Inference genuine par vraisemblance -- N=20 graines ===\n")
    for key in ["recall_k1_naif", "recall_k1_camo", "recall_k2_naif", "recall_k2_camo",
                "fpr_k1", "fpr_k2", "promotion_rate"]:
        vals = [r[key] for r in results]
        print(f"{key:20s} : {np.mean(vals)*100:5.1f}% +/- {np.std(vals)*100:4.1f}%  "
              f"(min={np.min(vals)*100:.1f}%, max={np.max(vals)*100:.1f}%)")
