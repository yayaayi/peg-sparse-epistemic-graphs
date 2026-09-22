"""
exp2ter_burst_correlation.py -- Tentative de recalibration de la detection
de coalition (Section 7.2) par correlation conditionnee aux fenetres de
rafale de population, dans le meme esprit que la correction reussie pour
la decision en Section 7.11.

MOTIVATION : le rappel mesure en Section 7.2 (F1=0.353+/-0.044) echoue le
critere de refutation defini a priori (rappel < 70%). Le code documente
deja que la coalition n'est "en phase" que 35% du temps (coalition_sync_prob),
diluant le signal de correlation calcule sur la fenetre entiere. Cette
section teste si restreindre le calcul de correlation aux seuls instants
de rafale de population detectee (meme principe que decide_burst_conditioned
en Section 7.11) ameliore la detection.

RESULTAT -- un compromis, pas une resolution. Le rappel optimal s'ameliore
(48.8%+/-7.8% contre ~30% pour la version originale), mais la precision
chute davantage (22.6%+/-6.4% contre ~50%), pour un F1 optimal legerement
INFERIEUR (0.299+/-0.041 contre 0.353+/-0.044). Restreindre la fenetre
d'observation reduit le nombre de points utilises pour estimer la
correlation, augmentant la variance de l'estimateur et donc le risque de
faux positifs -- un cout qui annule le benefice du signal moins dilue.

VERDICT : tentative honnete, echec partiel documente plutot que force vers
un succes. Le critere de refutation original reste declenche sous cette
variante egalement. Piste pour travail futur : une methode qui identifie
les fenetres de rafale PAR PAIRE (pas au niveau de la population agregee)
sans les defauts numeriques rencontres lors des essais preliminaires
(degenerescence par double conditionnement simultane) pourrait faire
mieux, mais reste a developper correctement.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from itertools import combinations
from env import Population, ReactiveModel, SmoothedSurprise, calibrate_threshold, calibrate_smoothed_threshold


def run(n=60, frac_adv=0.08, frac_coalition=0.12, window=25, T=400,
         warmup=60, seed=0, coalition_sync_prob=0.35, p_adv=0.55,
         burst_threshold_mult=1.3, rho_min_values=None):
    if rho_min_values is None:
        rho_min_values = np.arange(0.10, 0.55, 0.05)

    _, _, p0_hat = calibrate_threshold(n_calib=100, seed=1)
    pop = Population(n, p_compliant=p0_hat, p_adv=p_adv, frac_adv=frac_adv,
                       frac_coalition=frac_coalition,
                       coalition_sync_prob=coalition_sync_prob, seed=seed)
    model = ReactiveModel(n, p0_hat)
    agents = list(range(n))
    coalition_idx = set(np.where(pop.types == 2)[0].tolist())
    true_pairs = set(combinations(sorted(coalition_idx), 2))

    raw_surprise = {j: [] for j in agents}
    activity = []
    for t in range(T):
        o, _ = pop.step()
        for j in agents:
            raw_surprise[j].append(model.surprise(j, o[j]))
        activity.append(o.mean())

    raw_arr = {j: np.array(raw_surprise[j][warmup:]) for j in agents}
    activity_arr = np.array(activity[warmup:])
    typical = np.median(activity_arr)
    burst_mask = activity_arr > burst_threshold_mult * typical
    n_burst_steps = int(burst_mask.sum())

    best_f1, best_precision, best_recall, best_rho = 0, 0, 0, None
    for rho_min in rho_min_values:
        tp = fp = fn = 0
        for i, j in combinations(agents, 2):
            if n_burst_steps > 10:
                rho = np.corrcoef(raw_arr[i][burst_mask], raw_arr[j][burst_mask])[0, 1]
            else:
                rho = np.corrcoef(raw_arr[i], raw_arr[j])[0, 1]
            if np.isnan(rho):
                rho = 0
            pred = rho > rho_min
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
            best_f1, best_precision, best_recall, best_rho = f1, precision, recall, rho_min

    return dict(f1=best_f1, precision=best_precision, recall=best_recall,
                 rho_min=best_rho, n_burst_steps=n_burst_steps)


def run_multi_seed(n_seeds=8, **kwargs):
    f1s, precisions, recalls = [], [], []
    for s in range(n_seeds):
        r = run(seed=s, **kwargs)
        f1s.append(r["f1"]); precisions.append(r["precision"]); recalls.append(r["recall"])
    f1s, precisions, recalls = np.array(f1s), np.array(precisions), np.array(recalls)
    print(f"=== Correlation conditionnee aux rafales -- {n_seeds} graines ===")
    print(f"F1 optimal    : {f1s.mean():.3f} +/- {f1s.std():.3f}")
    print(f"Precision     : {precisions.mean():.3f} +/- {precisions.std():.3f}")
    print(f"Rappel        : {recalls.mean():.3f} +/- {recalls.std():.3f}")
    print(f"(reference, version non conditionnee : F1=0.353+/-0.044)")
    return f1s, precisions, recalls


if __name__ == "__main__":
    run_multi_seed(n_seeds=8)
