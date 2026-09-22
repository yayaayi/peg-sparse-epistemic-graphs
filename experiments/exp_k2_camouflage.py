"""
exp_k2_camouflage.py -- Test reel d'un signal de second ordre (k=2) contre un
adversaire en camouflage actif que le seuillage de premier ordre (k=1) est
structurellement incapable de detecter par construction (le camouflage est
precisement concu pour rester sous tau).

Ceci N'EST PAS une reproduction de l'"Experience 12" du document soumis par
l'utilisateur (qui rapportait des chiffres jamais calcules). C'est une
experience originale, executee ici pour la premiere fois, dont les resultats
-- quels qu'ils soient -- sont rapportes tels quels.

Protocole :
  1. Population avec agents compliants, adversaires naifs (taux fixe,
     Bernoulli), et adversaires camouflants (quota par fenetre -- voir
     env_recursive.py pour la justification du choix de ce mecanisme).
  2. Detection k=1 (existante) : V(t) = {j : S_barre^j(t) > tau}.
  3. Signal k=2 (nouveau, teste ici) : pour chaque agent, variance du taux de
     deviation entre fenetres successives, sur les H dernieres fenetres.
     Hypothese : les agents a quota ont une variance anormalement BASSE par
     rapport a ce qu'un processus Bernoulli au meme taux produirait.
  4. Detection adaptative : k=1 SEUL, puis k=1 relaye par k=2 pour les agents
     dont la variance inter-fenetres est en dessous d'un seuil calibre sur la
     population compliante (pas de fuite d'information sur le camouflage).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from env_recursive import CamouflagingPopulation


def calibrate_tau_and_variance_threshold(p_compliant=0.05, window=20, n_calib=100,
                                            T_calib=800, F_target=1.0, n_target=100,
                                            n_history_blocks=5, seed=1):
    """Calibre tau (comme dans le reste du document) ET le seuil de variance
    inter-fenetres, exclusivement sur une population purement compliante --
    aucune information sur le camouflage n'entre dans cette calibration."""
    pop = CamouflagingPopulation(n_calib, p_compliant=p_compliant, window=window, seed=seed)
    hist = [[] for _ in range(n_calib)]
    smoothed_vals = []
    for t in range(T_calib):
        o, _ = pop.step()
        for j in range(n_calib):
            hist[j].append(o[j])
            if len(hist[j]) >= window:
                recent = hist[j][-window:]
                s = [-np.log(p_compliant) if x else -np.log(1-p_compliant) for x in recent]
                smoothed_vals.append(np.mean(s))
    q = 1 - F_target / n_target
    tau = np.quantile(smoothed_vals, min(max(q, 0), 0.999999))

    # calibration du RATIO variance observee / variance binomiale attendue au
    # taux propre de l'agent (dispersion index) -- conditionner sur le taux
    # propre de chaque agent elimine le confondant "faible taux => faible
    # variance naturelle" qui invalidait un seuil de variance absolu
    dispersion_ratios = []
    for j in range(n_calib):
        o = np.array(hist[j])
        n_blocks = len(o) // window
        if n_blocks < n_history_blocks:
            continue
        rates = np.array([o[b*window:(b+1)*window].mean() for b in range(n_blocks)])
        for start in range(0, len(rates) - n_history_blocks + 1):
            block = rates[start:start+n_history_blocks]
            p_hat = block.mean()
            expected_var = max(p_hat * (1 - p_hat) / window, 1e-6)
            observed_var = block.var()
            dispersion_ratios.append(observed_var / expected_var)
    ratio_threshold = np.quantile(dispersion_ratios, 0.05)
    return tau, ratio_threshold


def run(n=80, frac_naif=0.10, frac_camouflant=0.10, p_adv=0.135, window=20,
        T=1200, warmup=100, n_history_blocks=10, seed=0):
    tau, ratio_threshold = calibrate_tau_and_variance_threshold(
        window=window, n_history_blocks=n_history_blocks, seed=1)
    print(f"tau calibre = {tau:.4f}  |  seuil de dispersion (5e percentile, H0 compliant) = {ratio_threshold:.4f}")

    pop = CamouflagingPopulation(n, p_adv=p_adv, window=window, frac_naif=frac_naif,
                                   frac_camouflant=frac_camouflant, seed=seed)
    hist = [[] for _ in range(n)]

    for t in range(T):
        o, types = pop.step()
        for j in range(n):
            hist[j].append(o[j])

    naif_idx = set(np.where(types == 1)[0].tolist())
    camo_idx = set(np.where(types == 2)[0].tolist())
    compliant_idx = set(np.where(types == 0)[0].tolist())

    # --- Verification prealable : le camouflage reste-t-il bien sous tau au sens k=1 ? ---
    mean_smoothed_camo = []
    mean_smoothed_naif = []
    for j in range(n):
        o = np.array(hist[j][warmup:])
        smoothed_series = [o[i-window+1:i+1].mean() for i in range(window-1, len(o))]
        s_bar_ref = np.mean([-np.log(0.05) if x else -np.log(0.95) for x in
                               np.repeat(smoothed_series, 1)])  # placeholder, recalcule ci-dessous
    # calcul propre de la surprise lissee (pas juste le taux brut) pour comparer a tau
    p0_hat = 0.05
    def surprise_smoothed_series(o_arr):
        s = np.array([-np.log(p0_hat) if x else -np.log(1-p0_hat) for x in o_arr])
        return np.array([s[i-window+1:i+1].mean() for i in range(window-1, len(s))])

    recall_k1 = {"naif": [], "camo": []}
    fpr_k1 = []
    recall_k2_added = {"naif": [], "camo": []}
    fpr_k2_added = []

    for j in range(n):
        o = np.array(hist[j])
        s_bar_series = surprise_smoothed_series(o)
        if len(s_bar_series) < warmup:
            continue
        s_bar_eval = s_bar_series[warmup:]
        flagged_k1 = (s_bar_eval > tau)
        frac_flagged_k1 = flagged_k1.mean()

        # signal k=2 : ratio dispersion observee / attendue (Binomiale au taux
        # propre de l'agent), calcule sur des BLOCS NON CHEVAUCHANTS de taille
        # `window` (coherent avec la calibration -- un taux glissant recalcule
        # a chaque pas lisserait artificiellement la variance)
        o_eval = o[warmup:]
        n_blocks = len(o_eval) // window
        block_rates = np.array([o_eval[b*window:(b+1)*window].mean() for b in range(n_blocks)])
        block_flagged_k1 = np.array([
            surprise_smoothed_series(o_eval[b*window:(b+1)*window])[-1] > tau
            if len(o_eval[b*window:(b+1)*window]) >= window else False
            for b in range(n_blocks)
        ])
        dispersion_series = []
        for i in range(n_blocks):
            block = block_rates[max(0,i-n_history_blocks+1):i+1]
            if len(block) < 2:
                dispersion_series.append(np.inf)
                continue
            p_hat = block.mean()
            expected_var = max(p_hat * (1 - p_hat) / window, 1e-6)
            dispersion_series.append(block.var() / expected_var)
        dispersion_series = np.array(dispersion_series)
        flagged_k2_extra = (dispersion_series < ratio_threshold) & (~block_flagged_k1)
        frac_flagged_k1 = block_flagged_k1.mean()
        frac_flagged_either = (block_flagged_k1 | flagged_k2_extra).mean()

        if j in naif_idx:
            recall_k1["naif"].append(frac_flagged_k1)
            recall_k2_added["naif"].append(frac_flagged_either)
        elif j in camo_idx:
            recall_k1["camo"].append(frac_flagged_k1)
            recall_k2_added["camo"].append(frac_flagged_either)
        else:
            fpr_k1.append(frac_flagged_k1)
            fpr_k2_added.append(frac_flagged_either)

    print(f"\nPopulation : n={n}, {len(naif_idx)} adversaires naifs, {len(camo_idx)} camouflants\n")
    print(f"{'':20s} {'Rappel k=1 seul':>18s} {'Rappel k=1+k=2':>18s}")
    print(f"{'Adversaire naif':20s} {np.mean(recall_k1['naif']):18.3f} {np.mean(recall_k2_added['naif']):18.3f}")
    print(f"{'Adversaire camouflant':20s} {np.mean(recall_k1['camo']):18.3f} {np.mean(recall_k2_added['camo']):18.3f}")
    print(f"\nFaux positifs (compliants) : k=1 seul = {np.mean(fpr_k1):.3f}  |  k=1+k=2 = {np.mean(fpr_k2_added):.3f}")


if __name__ == "__main__":
    run()
