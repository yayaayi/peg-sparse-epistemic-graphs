"""
exp8_tomnet.py -- Experience 8 : Comparaison a un modele d'opposant implicite
de type ToMnet (Section 7.8 du papier)

Important : ce script implemente un CLASSIFIEUR COMPORTEMENTAL appris sur des
caracteristiques de fenetre (taux de deviation, variance, plus longue rafale,
autocorrelation lag-1), pas une reproduction de l'architecture ToMnet
originale (Rabinowitz et al., 2018). C'est exactement la distinction que le
papier maintient explicitement (Section 6.4, Baselines) : un representant de
la famille "Opponent Modeling / apprentissage implicite", pas ToMnet
lui-meme.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
from env import Population, ReactiveModel, SmoothedSurprise, calibrate_threshold, calibrate_smoothed_threshold
from peg_algorithm import candidate_sets, peg_step


def behavioral_features(obs_window):
    """
    Caracteristiques comportementales de fenetre (Section 6.4) :
    taux de deviation, variance, plus longue rafale, autocorrelation lag-1.
    """
    obs = np.asarray(obs_window)
    if len(obs) < 2:
        return np.array([0.0, 0.0, 0.0, 0.0])
    rate = obs.mean()
    var = obs.var()
    # plus longue rafale de 1 consecutifs
    max_burst, cur = 0, 0
    for v in obs:
        cur = cur + 1 if v == 1 else 0
        max_burst = max(max_burst, cur)
    burst_norm = max_burst / len(obs)
    if obs.std() > 0:
        autocorr = np.corrcoef(obs[:-1], obs[1:])[0, 1]
        if np.isnan(autocorr):
            autocorr = 0.0
    else:
        autocorr = 0.0
    return np.array([rate, var, burst_norm, autocorr])


class LogisticClassifier:
    """Regression logistique minimale (descente de gradient), sans dependance
    a scikit-learn, pour rester dans le perimetre `requirements.txt`."""

    def __init__(self, n_features, lr=0.1, n_iter=500, l2=1e-3):
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        mu, sigma = X.mean(0), X.std(0) + 1e-8
        self.mu, self.sigma = mu, sigma
        Xn = (X - mu) / sigma
        n = len(y)
        for _ in range(self.n_iter):
            z = Xn @ self.w + self.b
            p = self._sigmoid(z)
            grad_w = Xn.T @ (p - y) / n + self.l2 * self.w
            grad_b = (p - y).mean()
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

    def predict_proba(self, X):
        Xn = (np.asarray(X) - self.mu) / self.sigma
        return self._sigmoid(Xn @ self.w + self.b)


def utility_from_predictions(y_true, y_pred, reward_tp=10, penalty_fp=-50,
                               penalty_fn=-3):
    u = 0.0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            u += reward_tp
        elif yt == 0 and yp == 1:
            u += penalty_fp
        elif yt == 1 and yp == 0:
            u += penalty_fn
    return u / len(y_true)


def run(n_train=100, n_test=100, frac_adv=0.15, window=20, T=400, warmup=60,
        seed=0, decision_threshold=0.5):
    _, _, p0_hat = calibrate_threshold(n_calib=100, seed=1)
    tau, _ = calibrate_smoothed_threshold(p0_hat, n_calib=100, window=window,
                                            F_target=1.0, n_target=100)

    # --- entrainement du classifieur implicite sur une population distincte ---
    pop_train = Population(n_train, p_compliant=p0_hat, p_adv=0.30,
                             frac_adv=frac_adv, seed=seed)
    obs_hist = [[] for _ in range(n_train)]
    for t in range(T):
        o, _ = pop_train.step()
        for j in range(n_train):
            obs_hist[j].append(o[j])

    X_train = np.array([behavioral_features(obs_hist[j][-window:]) for j in range(n_train)])
    y_train = (pop_train.types == 1).astype(int)

    clf = LogisticClassifier(n_features=4)
    clf.fit(X_train, y_train)

    # calibration du seuil de decision sur un jeu de validation distinct
    probs_train = clf.predict_proba(X_train)
    best_thr, best_u = 0.5, -np.inf
    for thr in np.arange(0.1, 0.95, 0.05):
        u = utility_from_predictions(y_train, (probs_train > thr).astype(int))
        if u > best_u:
            best_u, best_thr = u, thr

    # --- evaluation sur population de test jamais vue ---
    pop_test = Population(n_test, p_compliant=p0_hat, p_adv=0.30,
                            frac_adv=frac_adv, seed=seed + 999)
    model = ReactiveModel(n_test, p0_hat)
    smoother = SmoothedSurprise(n_test, window=window)
    cand = candidate_sets(n_test, k_max=8, mode="bounded_degree", seed=seed + 2)
    agents = list(range(n_test))
    adversaries_test = set(np.where(pop_test.types == 1)[0].tolist())

    obs_test = [[] for _ in range(n_test)]
    peg_utils, implicit_utils = [], []
    for t in range(T):
        o, _ = pop_test.step()
        for j in agents:
            obs_test[j].append(o[j])
            s = model.surprise(j, o[j])
            smoother.push_raw(j, s)
        if t < warmup:
            continue

        result = peg_step(t, agents, model, smoother, cand, tau=tau,
                           rho_min=0.30, w=window)
        flagged_peg = result["V"]
        y_true = [(1 if j in adversaries_test else 0) for j in agents]
        y_pred_peg = [(1 if j in flagged_peg else 0) for j in agents]
        peg_utils.append(utility_from_predictions(y_true, y_pred_peg))

        X_t = np.array([behavioral_features(obs_test[j][-window:]) for j in agents])
        probs_t = clf.predict_proba(X_t)
        y_pred_implicit = (probs_t > best_thr).astype(int)
        implicit_utils.append(utility_from_predictions(y_true, y_pred_implicit))

    print(f"n={n_test}  seuil de decision implicite calibre = {best_thr:.2f}")
    print(f"Utilite moyenne PEG           : {np.mean(peg_utils):.3f} +/- {np.std(peg_utils):.3f}")
    print(f"Utilite moyenne modele implicite : {np.mean(implicit_utils):.3f} +/- {np.std(implicit_utils):.3f}")
    return np.mean(peg_utils), np.mean(implicit_utils)


def run_multi_seed(n_seeds=5, **kwargs):
    """Boucle sur n_seeds graines -- le papier ne precise pas de compte de
    graines specifique pour cette experience (Section 7.8) ; 5 est la borne
    basse de la valeur par defaut de la Section 6 ("typiquement 5 a 10
    graines")."""
    peg_means, implicit_means = [], []
    for s in range(n_seeds):
        print(f"\n--- graine {s} ---")
        peg_u, imp_u = run(seed=s, **kwargs)
        peg_means.append(peg_u)
        implicit_means.append(imp_u)
    peg_means = np.array(peg_means)
    implicit_means = np.array(implicit_means)
    print(f"\n=== Agrege sur {n_seeds} graines (seeds=0..{n_seeds-1}) ===")
    print(f"PEG              : {peg_means.mean():.3f} +/- {peg_means.std():.3f}")
    print(f"Modele implicite : {implicit_means.mean():.3f} +/- {implicit_means.std():.3f}")
    if n_seeds > 1:
        from scipy.stats import ttest_rel
        t_stat, p_val = ttest_rel(peg_means, implicit_means)
        print(f"t={t_stat:.3f}  p={p_val:.3f} (test apparie, memes graines)")
    return peg_means, implicit_means


if __name__ == "__main__":
    run_multi_seed(n_seeds=5)
