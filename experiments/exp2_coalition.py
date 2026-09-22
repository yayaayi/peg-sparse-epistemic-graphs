"""
exp2_coalition.py -- Experience 2 : Detection de coalition (Section 7.2 du papier)

Population avec adversaires isoles et adversaires en coalition synchronisee
(meme phase d'activite partagee). Mesure precision/rappel/F1 de la detection
de coalition (Definition 5) en fonction du seuil rho_min.

Sortie attendue (papier) : F1 optimal = 1.00 pour une coalition
parfaitement synchronisee, autour de rho_min ~= 0.30.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from itertools import combinations
from env import Population, ReactiveModel, SmoothedSurprise, calibrate_threshold, calibrate_smoothed_threshold
from peg_algorithm import candidate_sets, peg_step


def run(n=60, frac_adv=0.08, frac_coalition=0.12, window=25, T=400,
        warmup=60, seed=0, rho_min_values=None, coalition_sync_prob=0.35,
        p_adv=0.55):
    if rho_min_values is None:
        rho_min_values = np.arange(0.10, 0.55, 0.05)

    _, _, p0_hat = calibrate_threshold(n_calib=100, seed=1)
    tau, _ = calibrate_smoothed_threshold(p0_hat, n_calib=100, window=20,
                                            F_target=1.0, n_target=100)

    # coalition_sync_prob et p_adv releves par rapport aux valeurs par defaut :
    # une coalition qui n'est "en phase" que 15% du temps ne traverse jamais le
    # seuil d'activation individuel moyenne sur la fenetre (voir README.md,
    # "Ecarts connus par rapport au papier") ; ces valeurs assurent qu'elle
    # entre dans V(t) au moins ponctuellement, condition necessaire au
    # mecanisme de correlation restreint a V(t) tel que specifie Section 4.5.
    pop = Population(n, p_compliant=p0_hat, p_adv=p_adv, frac_adv=frac_adv,
                      frac_coalition=frac_coalition,
                      coalition_sync_prob=coalition_sync_prob, seed=seed)
    model = ReactiveModel(n, p0_hat)
    smoother = SmoothedSurprise(n, window=window)
    cand = candidate_sets(n, k_max=n - 1, mode="bounded_degree", seed=seed + 1)
    agents = list(range(n))

    coalition_idx = set(np.where(pop.types == 2)[0].tolist())
    true_pairs = set(combinations(sorted(coalition_idx), 2))

    scores = {rm: {"tp": 0, "fp": 0, "fn": 0} for rm in rho_min_values}

    for t in range(T):
        o, _ = pop.step()
        for j in agents:
            s = model.surprise(j, o[j])
            smoother.push_raw(j, s)
        if t < warmup:
            continue

        # correlations calculees une seule fois (rho_min=0 -> tout candidat retenu)
        result = peg_step(t, agents, model, smoother, cand, tau=tau,
                           rho_min=-1.0, w=window)
        rho = result["rho"]

        for rm in rho_min_values:
            detected = set(pair for pair, r in rho.items() if r > rm)
            tp = len(detected & true_pairs)
            fp = len(detected - true_pairs)
            fn = len(true_pairs - detected)
            scores[rm]["tp"] += tp
            scores[rm]["fp"] += fp
            scores[rm]["fn"] += fn

    print(f"n={n}  paires de coalition vraies (par pas evalue): {len(true_pairs)}")
    best_f1, best_rm = -1, None
    for rm in rho_min_values:
        tp, fp, fn = scores[rm]["tp"], scores[rm]["fp"], scores[rm]["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"  rho_min={rm:.2f}  precision={precision:.3f}  rappel={recall:.3f}  F1={f1:.3f}")
        if f1 > best_f1:
            best_f1, best_rm = f1, rm
    print(f"\nMeilleur F1={best_f1:.3f} atteint a rho_min={best_rm:.2f}")
    return best_f1, best_rm


def run_multi_seed(n_seeds=5, **kwargs):
    """Boucle sur n_seeds graines (0..n_seeds-1), agrege moyenne +/- ecart-type
    du F1 optimal -- reproduit directement le chiffre rapporte dans le papier
    (Section 7.2 : F1=1.00 a rho_min~=0.30, N=5 graines)."""
    f1s, rms = [], []
    for s in range(n_seeds):
        print(f"\n--- graine {s} ---")
        f1, rm = run(seed=s, **kwargs)
        f1s.append(f1)
        rms.append(rm)
    f1s = np.array(f1s)
    print(f"\n=== Agrege sur {n_seeds} graines (seeds=0..{n_seeds-1}) ===")
    print(f"F1 optimal : {f1s.mean():.3f} +/- {f1s.std():.3f}")
    print(f"rho_min optimal (mode) : {max(set(rms), key=rms.count):.2f}")
    return f1s


if __name__ == "__main__":
    run_multi_seed(n_seeds=5)
