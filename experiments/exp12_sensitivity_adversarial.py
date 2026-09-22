"""
exp12_sensitivity_adversarial.py -- Experience 12 : Sensibilite aux
hyperparametres et robustesse adversariale (Section 7.12 du papier)

Deux volets :
  (i)  balayage de tau (via F_cible), rho_min et w -- verifie l'absence de
       discontinuite inquietante dans le comportement du mecanisme ;
  (ii) scenario adversarial de contournement strategique -- un adversaire
       connaissant tau et le modele de reference P_1 ajuste sa deviation
       pour viser une surprise juste sous le seuil.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from env import Population, ReactiveModel, SmoothedSurprise, calibrate_threshold, calibrate_smoothed_threshold
from peg_algorithm import candidate_sets, peg_step


def sensitivity_sweep(n=80, frac_adv=0.15, window=20, T=300, warmup=60, seed=0):
    print("-- Sensibilite aux hyperparametres --\n")
    _, _, p0_hat = calibrate_threshold(n_calib=100, seed=1)

    print("(i) Variation de F_cible (donc de tau) :")
    for F_cible in [0.5, 1.0, 2.0, 5.0, 10.0]:
        tau, _ = calibrate_smoothed_threshold(p0_hat, n_calib=100, window=window,
                                                F_target=F_cible, n_target=100)
        n_active, fpr_list = [], []
        pop = Population(n, p_compliant=p0_hat, p_adv=0.30, frac_adv=frac_adv, seed=seed)
        model = ReactiveModel(n, p0_hat)
        smoother = SmoothedSurprise(n, window=window)
        cand = candidate_sets(n, k_max=8, seed=seed + 1)
        agents = list(range(n))
        adversaries = set(np.where(pop.types == 1)[0].tolist())
        for t in range(T):
            o, _ = pop.step()
            for j in agents:
                smoother.push_raw(j, model.surprise(j, o[j]))
            if t < warmup:
                continue
            result = peg_step(t, agents, model, smoother, cand, tau=tau,
                               rho_min=0.30, w=window)
            n_active.append(len(result["E"]))
            compliant = set(agents) - adversaries
            flagged_compliant = result["V"] & compliant
            fpr_list.append(len(flagged_compliant) / max(len(compliant), 1))
        print(f"  F_cible={F_cible:5.1f}  tau={tau:.3f}  "
              f"aretes actives moy.={np.mean(n_active):.1f}  FPR moy.={np.mean(fpr_list):.3f}")

    print("\n(ii) Variation de rho_min (plage large) :")
    tau, _ = calibrate_smoothed_threshold(p0_hat, n_calib=100, window=window,
                                            F_target=1.0, n_target=100)
    for rho_min in [0.10, 0.20, 0.30, 0.40, 0.50]:
        pop = Population(n, p_compliant=p0_hat, p_adv=0.30, frac_adv=frac_adv,
                          frac_coalition=0.10, coalition_sync_prob=0.35, seed=seed)
        model = ReactiveModel(n, p0_hat)
        smoother = SmoothedSurprise(n, window=window)
        cand = candidate_sets(n, k_max=8, seed=seed + 1)
        agents = list(range(n))
        f1_list = []
        for t in range(T):
            o, _ = pop.step()
            for j in agents:
                smoother.push_raw(j, model.surprise(j, o[j]))
            if t < warmup:
                continue
            result = peg_step(t, agents, model, smoother, cand, tau=tau,
                               rho_min=rho_min, w=window)
            f1_list.append(len(result["C"]))
        print(f"  rho_min={rho_min:.2f}  paires de coalition moy. detectees/pas={np.mean(f1_list):.2f}")

    print("\n(iii) Variation de la fenetre w :")
    for w in [10, 15, 20, 30, 40]:
        tau_w, _ = calibrate_smoothed_threshold(p0_hat, n_calib=100, window=w,
                                                   F_target=1.0, n_target=100)
        pop = Population(n, p_compliant=p0_hat, p_adv=0.30, frac_adv=frac_adv, seed=seed)
        model = ReactiveModel(n, p0_hat)
        smoother = SmoothedSurprise(n, window=w)
        cand = candidate_sets(n, k_max=8, seed=seed + 1)
        agents = list(range(n))
        n_active = []
        for t in range(T):
            o, _ = pop.step()
            for j in agents:
                smoother.push_raw(j, model.surprise(j, o[j]))
            if t < max(warmup, w):
                continue
            result = peg_step(t, agents, model, smoother, cand, tau=tau_w,
                               rho_min=0.30, w=w)
            n_active.append(len(result["E"]))
        print(f"  w={w:3d}  tau={tau_w:.3f}  aretes actives moy.={np.mean(n_active):.1f}")


def adversarial_evasion(n=80, frac_adv=0.15, window=20, T=300, warmup=60, seed=0):
    print("\n-- Scenario adversarial : contournement strategique du seuil --\n")
    _, _, p0_hat = calibrate_threshold(n_calib=100, seed=1)
    tau, _ = calibrate_smoothed_threshold(p0_hat, n_calib=100, window=window,
                                            F_target=1.0, n_target=100)

    # un adversaire informe vise une surprise lissee EGALE a `margin * tau`
    for margin in [0.5, 0.7, 0.9, 1.0, 1.05]:
        # cherche p_adv tel que la surprise lissee attendue = margin * tau
        # E[S] = -p*log(p0_hat) - (1-p)*log(1-p0_hat), resolu numeriquement
        target_s = margin * tau
        p_grid = np.linspace(0.001, 0.999, 2000)
        s_grid = -(p_grid * np.log(p0_hat) + (1 - p_grid) * np.log(1 - p0_hat))
        p_adv = p_grid[np.argmin(np.abs(s_grid - target_s))]

        pop = Population(n, p_compliant=p0_hat, p_adv=p_adv, frac_adv=frac_adv, seed=seed)
        model = ReactiveModel(n, p0_hat)
        smoother = SmoothedSurprise(n, window=window)
        cand = candidate_sets(n, k_max=8, seed=seed + 1)
        agents = list(range(n))
        adversaries = set(np.where(pop.types == 1)[0].tolist())

        recalls = []
        for t in range(T):
            o, _ = pop.step()
            for j in agents:
                smoother.push_raw(j, model.surprise(j, o[j]))
            if t < warmup:
                continue
            result = peg_step(t, agents, model, smoother, cand, tau=tau,
                               rho_min=0.30, w=window)
            flagged_adv = result["V"] & adversaries
            recalls.append(len(flagged_adv) / max(len(adversaries), 1))

        print(f"  marge={margin:.2f}  p_adv vise={p_adv:.3f}  "
              f"rappel de detection={np.mean(recalls):.3f}")


if __name__ == "__main__":
    sensitivity_sweep()
    adversarial_evasion()
