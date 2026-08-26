"""
env.py -- Modele de population et calcul de la surprise (Section 3.1-3.2 du papier)

Implemente :
  - Population    : agents compliants / adversaires isoles / adversaires en coalition
                     (Section 6.2 -- Protocole experimental, Agents)
  - ReactiveModel  : modele de reference P_1, calibre sur une population purement
                     compliante (Convention 3.0, Definition 3)
  - SmoothedSurprise : surprise brute S_i^j(t) et surprise lissee S_barre_i^j(t)
                     sur une fenetre glissante w (Definition 3)
  - calibrate_threshold : calibration de tau par quantile corrige pour la
                     multiplicite (Definition 4bis)

Toutes les grandeurs sont calculees exclusivement a partir de l'historique
d'observations comportementales, conformement a la Convention 3.0 du papier.
"""

import numpy as np


class Population:
    """
    Population d'agents A = {a_1, ..., a_n}.

    Chaque agent emet a chaque pas un signal binaire de conformite/deviation,
    tire d'une loi de Bernoulli dont le parametre depend de son type :
      - compliant       : p = p0 (taux de deviation "normal")
      - adversaire isole : p = p_adv (deviation individuelle, sans coordination)
      - coalition        : p varie selon une phase de synchronisation partagee
                            (voir `coalition_sync_prob`)

    Correspond a l'environnement binaire (Section 6.1, Protocole experimental).
    """

    def __init__(self, n, p_compliant=0.05, p_adv=0.30, frac_adv=0.0,
                 frac_coalition=0.0, coalition_sync_prob=0.15, seed=0):
        self.n = n
        self.p_compliant = p_compliant
        self.p_adv = p_adv
        self.coalition_sync_prob = coalition_sync_prob
        self.rng = np.random.default_rng(seed)

        n_adv = int(round(n * frac_adv))
        n_coal = int(round(n * frac_coalition))
        n_compliant = n - n_adv - n_coal
        assert n_compliant >= 0, "frac_adv + frac_coalition ne peut pas depasser 1"

        # types : 0 = compliant, 1 = adversaire isole, 2 = coalition
        self.types = np.array(
            [0] * n_compliant + [1] * n_adv + [2] * n_coal
        )
        self.rng.shuffle(self.types)
        self.t = 0

    def step(self):
        """Un pas de simulation : renvoie le vecteur d'observations o[j] in {0,1}."""
        o = np.zeros(self.n, dtype=int)
        compliant_mask = self.types == 0
        adv_mask = self.types == 1
        coal_mask = self.types == 2

        o[compliant_mask] = (
            self.rng.random(compliant_mask.sum()) < self.p_compliant
        ).astype(int)
        o[adv_mask] = (
            self.rng.random(adv_mask.sum()) < self.p_adv
        ).astype(int)

        if coal_mask.sum() > 0:
            # phase de synchronisation partagee : un seul tirage pour toute la coalition
            in_phase = self.rng.random() < self.coalition_sync_prob
            p_coal = self.p_adv if in_phase else self.p_compliant
            o[coal_mask] = (
                self.rng.random(coal_mask.sum()) < p_coal
            ).astype(int)

        self.t += 1
        return o, self.types


class ReactiveModel:
    """
    Modele de reference P_1 (Definition 3). Calibre sur une population purement
    compliante -- jamais mis a jour a partir d'un comportement deviant
    (Convention 3.0 : la construction du graphe precede tout calcul de profondeur).
    """

    def __init__(self, n, p0_hat, eps=1e-6):
        self.n = n
        # p0_hat borne loin de 0 et 1, comme l'exige le Theoreme 3
        self.p0_hat = min(max(p0_hat, eps), 1 - eps)

    def surprise(self, j, o_j):
        """S^j(t) = -log P_1(o_{j,t} | H_{t-1}) -- Definition 3."""
        p = self.p0_hat if o_j == 1 else (1 - self.p0_hat)
        return -np.log(p)


class SmoothedSurprise:
    """
    Surprise lissee sur fenetre glissante w (Definition 3, critere d'activation).
    Maintient egalement l'historique brut necessaire a la correlation de
    coalition (Definition 5), qui opere sur la surprise BRUTE, pas lissee.
    """

    def __init__(self, n, window):
        self.n = n
        self.window = window
        self.raw_hist = [[] for _ in range(n)]

    def push_raw(self, j, s):
        self.raw_hist[j].append(s)
        if len(self.raw_hist[j]) > self.window:
            self.raw_hist[j].pop(0)

    def smoothed(self, j):
        """S_barre^j(t) : moyenne des `window` dernieres surprises brutes."""
        h = self.raw_hist[j]
        return sum(h) / len(h) if h else 0.0

    def raw_window(self, j):
        return np.array(self.raw_hist[j])


def calibrate_threshold(n_calib=200, F_target=1.0, n_target=100, T_calib=500,
                          p_compliant=0.05, seed=1):
    """
    Estime p0_hat sur une population purement compliante (n_calib agents,
    T_calib pas), puis calcule le seuil brut tau = -log(1-p0_hat) correspondant
    a une deviation isolee. Retourne (tau_brut, p0_hat_estime, p0_hat_estime).

    Utilise par calibrate_smoothed_threshold pour la vraie calibration par
    quantile (Definition 4bis).
    """
    pop = Population(n_calib, p_compliant=p_compliant, seed=seed)
    total_dev = 0
    total_obs = 0
    for _ in range(T_calib):
        o, _ = pop.step()
        total_dev += o.sum()
        total_obs += len(o)
    p0_hat = total_dev / total_obs
    tau_brut = -np.log(1 - p0_hat)
    return tau_brut, n_target, p0_hat


def calibrate_smoothed_threshold(p0_hat, n_calib=200, T_calib=500, F_target=1.0,
                                   n_target=100, window=20, seed=1):
    """
    Calibration operationnelle de tau par quantile corrige pour la multiplicite
    (Definition 4bis) :

        tau = Q_compliant(1 - F_target / n_target)

    ou Q_compliant est la fonction quantile de la surprise LISSEE sous
    comportement conforme.
    """
    model = ReactiveModel(n_calib, p0_hat)
    smoother = SmoothedSurprise(n_calib, window=window)
    pop = Population(n_calib, p_compliant=p0_hat, seed=seed)
    smoothed_vals = []
    for t in range(T_calib):
        o, _ = pop.step()
        for j in range(n_calib):
            s = model.surprise(j, o[j])
            smoother.push_raw(j, s)
            if t >= window:
                smoothed_vals.append(smoother.smoothed(j))
    smoothed_vals = np.array(smoothed_vals)
    q = 1 - F_target / n_target
    q = min(max(q, 0.0), 0.999999)
    tau = np.quantile(smoothed_vals, q)
    return tau, smoothed_vals
