"""
env_recursive.py -- Population avec adversaire en camouflage actif, pour tester
si un signal de second ordre (k=2) peut detecter ce qu'un seuillage de premier
ordre (k=1) manque structurellement.

Hypothese testee (pas supposee vraie a priori) : un adversaire "naif" qui vise
une marge sous tau via un taux de deviation Bernoulli FIXE (Experience 12
existante) a la MEME variance de surprise par fenetre qu'un agent conforme au
meme taux moyen -- un simple seuillage sur la moyenne ne peut donc pas les
distinguer par construction, mais un test de variance ne le peut pas non plus,
puisque leur variance EST la meme.

Un adversaire qui REGULE ACTIVEMENT le nombre de deviations par fenetre
(quota deterministe plutot que tirage Bernoulli independant a chaque pas)
produit en revanche une variance par fenetre structurellement plus BASSE que
n'importe quel processus i.i.d. au meme taux moyen. C'est cette signature
statistique -- pas la moyenne seule -- qu'un mecanisme de second ordre
pourrait exploiter. Le test decidera si c'est effectivement detectable dans
la plage de parametres consideree.
"""

import numpy as np


class CamouflagingPopulation:
    """
    Population avec trois types d'agents :
      0 = compliant (Bernoulli(p_compliant) a chaque pas, non strategique)
      1 = adversaire naif (Bernoulli(p_adv) a chaque pas, taux fixe choisi
          pour cibler une marge sous tau -- IDENTIQUE en distribution a un
          agent conforme dont le taux serait p_adv ; c'est le meme modele
          que l'Experience 12 existante)
      2 = adversaire camouflant a quota (au sein de chaque fenetre de w pas,
          le nombre de deviations est fixe a round(p_adv * w), reparti aussi
          uniformement que possible dans la fenetre plutot que tire
          independamment a chaque pas -- variance par fenetre structurellement
          plus faible qu'un processus Bernoulli au meme taux moyen)
    """

    def __init__(self, n, p_compliant=0.05, p_adv=0.18, window=20,
                 frac_naif=0.0, frac_camouflant=0.0, seed=0):
        self.n = n
        self.p_compliant = p_compliant
        self.p_adv = p_adv
        self.window = window
        self.rng = np.random.default_rng(seed)

        n_naif = int(round(n * frac_naif))
        n_camo = int(round(n * frac_camouflant))
        n_compliant = n - n_naif - n_camo
        assert n_compliant >= 0

        self.types = np.array([0] * n_compliant + [1] * n_naif + [2] * n_camo)
        self.rng.shuffle(self.types)

        self.quota_per_window = max(0, round(p_adv * window))
        self.block_schedule = {}
        self.t = 0

    def _get_quota_deviation(self, j, t):
        block_idx = t // self.window
        pos_in_block = t % self.window
        key = (j, block_idx)
        if key not in self.block_schedule:
            positions = self.rng.choice(self.window, size=self.quota_per_window,
                                          replace=False)
            self.block_schedule[key] = set(positions.tolist())
        return 1 if pos_in_block in self.block_schedule[key] else 0

    def step(self):
        o = np.zeros(self.n, dtype=int)
        for j in range(self.n):
            if self.types[j] == 0:
                o[j] = 1 if self.rng.random() < self.p_compliant else 0
            elif self.types[j] == 1:
                o[j] = 1 if self.rng.random() < self.p_adv else 0
            else:
                o[j] = self._get_quota_deviation(j, self.t)
        self.t += 1
        return o, self.types
