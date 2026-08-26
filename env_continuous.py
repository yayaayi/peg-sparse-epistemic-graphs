"""
env_continuous.py -- Variante continue de l'environnement (Section 7.7 du papier)

Environnement a observations continues (telemetrie gaussienne, typique d'un
contexte de surveillance IoT/flotte), utilise pour la validation croisee.
Remplace la vraisemblance de Bernoulli par une vraisemblance gaussienne
negative, conformement au protocole decrit Section 6.1.
"""

import numpy as np


class ContinuousPopulation:
    """
    Meme structure de types que env.Population (compliant / isole / coalition),
    mais chaque agent emet une mesure continue x ~ N(mu, sigma^2) au lieu d'un
    signal binaire. Les agents deviants ont une moyenne decalee.
    """

    def __init__(self, n, mu_compliant=0.0, sigma=1.0, mu_adv=2.5,
                 frac_adv=0.0, frac_coalition=0.0, coalition_sync_prob=0.15,
                 seed=0):
        self.n = n
        self.mu_compliant = mu_compliant
        self.sigma = sigma
        self.mu_adv = mu_adv
        self.coalition_sync_prob = coalition_sync_prob
        self.rng = np.random.default_rng(seed)

        n_adv = int(round(n * frac_adv))
        n_coal = int(round(n * frac_coalition))
        n_compliant = n - n_adv - n_coal
        assert n_compliant >= 0

        self.types = np.array([0] * n_compliant + [1] * n_adv + [2] * n_coal)
        self.rng.shuffle(self.types)
        self.t = 0

    def step(self):
        x = np.zeros(self.n)
        compliant_mask = self.types == 0
        adv_mask = self.types == 1
        coal_mask = self.types == 2

        x[compliant_mask] = self.rng.normal(self.mu_compliant, self.sigma,
                                              compliant_mask.sum())
        x[adv_mask] = self.rng.normal(self.mu_adv, self.sigma, adv_mask.sum())

        if coal_mask.sum() > 0:
            in_phase = self.rng.random() < self.coalition_sync_prob
            mu = self.mu_adv if in_phase else self.mu_compliant
            x[coal_mask] = self.rng.normal(mu, self.sigma, coal_mask.sum())

        self.t += 1
        return x, self.types


class GaussianReactiveModel:
    """Modele de reference P_1 gaussien : N(mu_compliant, sigma^2)."""

    def __init__(self, mu_compliant, sigma):
        self.mu = mu_compliant
        self.sigma = sigma

    def surprise(self, x):
        """S(t) = -log P_1(x) sous le modele gaussien de reference."""
        return 0.5 * np.log(2 * np.pi * self.sigma ** 2) + \
               ((x - self.mu) ** 2) / (2 * self.sigma ** 2)


def calibrate_continuous(n_calib=100, T_calib=500, mu_compliant=0.0, sigma=1.0,
                           seed=1):
    pop = ContinuousPopulation(n_calib, mu_compliant=mu_compliant, sigma=sigma,
                                 seed=seed)
    xs = []
    for _ in range(T_calib):
        x, _ = pop.step()
        xs.extend(x.tolist())
    mu_hat = float(np.mean(xs))
    sigma_hat = float(np.std(xs))
    return mu_hat, sigma_hat
