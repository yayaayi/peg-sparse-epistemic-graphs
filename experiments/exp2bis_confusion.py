"""
exp2bis_confusion.py -- Experience 2bis : test de confusion coalition vs.
contexte partage (Section 7.2bis du papier)

Ce test n'existait pas dans le depot avant cette campagne de verification ;
ses chiffres provenaient jusqu'ici des simulations originales non
recouvrees. Ce script l'implemente honnetement a partir de la description
du protocole dans le papier :

Population sans aucune vraie coalition (adversaires ISOLES uniquement,
type=1, jamais type=2). Deux conditions :
  (a) sans choc de contexte : chaque adversaire isole devie au taux p_adv
      fixe, independamment des autres -- aucune cause commune.
  (b) avec choc de contexte partage : a certains pas, une cause EXOGENE
      commune (ex. une meme perturbation environnementale) eleve
      SIMULTANEMENT le taux de deviation de TOUS les adversaires isoles a
      la fois -- mais chacun tire ensuite sa propre deviation
      INDEPENDAMMENT etant donne ce taux commun. Il n'y a aucune
      coordination reelle (aucune variable de phase partagee decidee entre
      eux, contrairement au mecanisme de coalition de Population type=2) :
      c'est une correlation induite par une cause commune, pas une
      strategie coordonnee.

Toute detection positive de "coalition" ici est un faux positif par
construction, puisqu'aucune vraie coalition n'existe dans ce protocole.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from env import ReactiveModel, SmoothedSurprise, calibrate_threshold, calibrate_smoothed_threshold
from peg_algorithm import candidate_sets, peg_step


class SharedContextPopulation:
    """Adversaires isoles uniquement (jamais de vraie coalition). Sous choc
    de contexte, une cause EXOGENE commune (pas une decision partagee entre
    agents) eleve simultanement le taux de tous les adversaires isoles."""

    def __init__(self, n, p_compliant, p_adv_base, p_adv_shock, frac_adv,
                  shock_prob, with_shock, seed=0):
        self.n = n
        self.p_compliant = p_compliant
        self.p_adv_base = p_adv_base
        self.p_adv_shock = p_adv_shock
        self.shock_prob = shock_prob
        self.with_shock = with_shock
        self.rng = np.random.default_rng(seed)
        n_adv = int(round(n * frac_adv))
        self.types = np.array([0] * (n - n_adv) + [1] * n_adv)
        self.rng.shuffle(self.types)

    def step(self):
        o = np.zeros(self.n, dtype=int)
        compliant_mask = self.types == 0
        adv_mask = self.types == 1
        o[compliant_mask] = (self.rng.random(compliant_mask.sum()) < self.p_compliant).astype(int)

        if self.with_shock:
            # cause EXOGENE commune (ex. panne reseau partagee, alerte
            # generale) -- affecte le taux de TOUS les adversaires isoles
            # simultanement, mais chaque tirage individuel reste independant
            in_shock = self.rng.random() < self.shock_prob
            p = self.p_adv_shock if in_shock else self.p_adv_base
        else:
            p = self.p_adv_base
        o[adv_mask] = (self.rng.random(adv_mask.sum()) < p).astype(int)
        return o, self.types


def run(n=60, frac_adv=0.15, window=25, T=400, warmup=60, rho_min=0.30,
         p_adv_base=0.15, p_adv_shock=0.55, shock_prob=0.20, seed=0):
    _, _, p0_hat = calibrate_threshold(n_calib=100, seed=1)
    tau, _ = calibrate_smoothed_threshold(p0_hat, n_calib=100, window=window,
                                            F_target=1.0, n_target=100)

    results = {}
    for with_shock, label in [(False, "sans_choc"), (True, "avec_choc")]:
        pop = SharedContextPopulation(n, p0_hat, p_adv_base, p_adv_shock, frac_adv,
                                        shock_prob, with_shock, seed=seed)
        model = ReactiveModel(n, p0_hat)
        smoother = SmoothedSurprise(n, window=window)
        cand = candidate_sets(n, k_max=8, mode="bounded_degree", seed=seed + 1)
        agents = list(range(n))
        adv_ids = set(np.where(pop.types == 1)[0])
        n_adv = len(adv_ids)
        flagged_counts = []  # fraction d'adversaires isoles signales, PAR PAS

        for t in range(T):
            o, _ = pop.step()
            for j in agents:
                s = model.surprise(j, o[j])
                smoother.push_raw(j, s)
            if t < warmup:
                continue
            result = peg_step(t, agents, model, smoother, cand, tau=tau,
                               rho_min=rho_min, w=window)
            coalition_pairs = result["C"]
            flagged_this_step = set()
            for (j, k) in coalition_pairs:
                # toute paire signalee ici est un FAUX positif par
                # construction (aucune vraie coalition dans ce protocole)
                if j in adv_ids:
                    flagged_this_step.add(j)
                if k in adv_ids:
                    flagged_this_step.add(k)
            flagged_counts.append(len(flagged_this_step) / max(n_adv, 1))

        # taux de faux positifs = fraction moyenne, PAR PAS, des adversaires
        # isoles signales a tort comme membres d'une coalition -- borne
        # dans [0,1], directement comparable au chiffre du papier
        fpr = float(np.mean(flagged_counts))
        results[label] = dict(fpr=fpr, n_adv=n_adv)
    return results


def run_multi_seed(n_seeds=8, **kwargs):
    """8 graines, comme specifie dans le protocole du papier pour 7.2bis."""
    fpr_no_shock, fpr_shock = [], []
    for s in range(n_seeds):
        r = run(seed=s, **kwargs)
        fpr_no_shock.append(r["sans_choc"]["fpr"])
        fpr_shock.append(r["avec_choc"]["fpr"])
    fpr_no_shock = np.array(fpr_no_shock)
    fpr_shock = np.array(fpr_shock)
    print(f"Taux de faux positifs (adversaires isoles signales a tort comme "
          f"coalition), sans choc de contexte : "
          f"{fpr_no_shock.mean()*100:.1f}% +/- {fpr_no_shock.std()*100:.1f}%")
    print(f"Taux de faux positifs, avec choc de contexte : "
          f"{fpr_shock.mean()*100:.1f}% +/- {fpr_shock.std()*100:.1f}%")
    return fpr_no_shock, fpr_shock


if __name__ == "__main__":
    run_multi_seed(n_seeds=8)
