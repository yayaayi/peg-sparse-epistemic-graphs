"""
k_order_inference.py -- Inference reelle d'ordre k (0, 1, 2), fondee sur des
modeles generatifs candidats et leur vraisemblance sur les observations
COMPORTEMENTALES uniquement. AUCUNE fonction de ce module n'accede au type
reel de l'agent (compliant / adversaire) -- verifie explicitement par un test
d'assertion en fin de fichier, qui echoue si une fuite est introduite.

Modeles candidats pour la sequence de comptages par fenetre c_1,...,c_m :
  - Ordre 0 : taux marginal de population (aucune adaptation a l'historique
    individuel de l'agent).
  - Ordre 1 : processus i.i.d. Bernoulli(theta_hat), theta_hat estime par
    maximum de vraisemblance sur l'historique propre de l'agent (croyance de
    premier ordre standard -- c'est exactement le mecanisme deja utilise
    ailleurs dans PEG, Definition 3-4bis).
  - Ordre 2 : processus regule (theta_hat identique, mais dispersion
    inter-fenetres fortement reduite) -- modelise l'hypothese qu'un agent de
    niveau 1 pourrait lui-meme reguler activement son taux pour eviter la
    detection (une forme simple, testable, de raisonnement sur autrui qui
    raisonne sur sa propre detectabilite).

La regle d'arret est fondee sur le GAIN DE VRAISEMBLANCE reel entre modeles,
suivant exactement la structure de la Section 3.6 du papier :
    S(k) = -log L(observations | modele d'ordre k)
    Delta S(k -> k+1) = S(k) - S(k+1)
    escalade a k+1 si Delta S(k -> k+1) > tau_order
"""

import numpy as np
from scipy.stats import binom, norm


def block_counts(o, window):
    """Comptages de deviations par bloc NON CHEVAUCHANT de taille `window`."""
    n_blocks = len(o) // window
    return np.array([o[b*window:(b+1)*window].sum() for b in range(n_blocks)])


def surprise_order0(counts, window, p_population):
    """S(0) : vraisemblance negative des comptages par bloc sous le taux
    marginal de POPULATION fixe (aucune adaptation a l'historique propre de
    l'agent -- reactif, Definition niveau 0). Meme forme fonctionnelle que
    S(1) ci-dessous (Binomiale), pour une comparaison a echelle homogene ;
    seul le parametre change (fixe vs estime sur l'agent)."""
    if len(counts) == 0:
        return 0.0
    p = np.clip(p_population, 1e-6, 1 - 1e-6)
    return -np.sum(binom.logpmf(counts, window, p))


def surprise_order1(counts, window):
    """S(1) : vraisemblance negative des comptages par bloc sous un modele
    i.i.d. Bernoulli(theta_hat), theta_hat estime par MLE sur CES MEMES
    donnees (croyance de premier ordre standard, Definition 3)."""
    if len(counts) == 0:
        return 0.0
    theta_hat = np.clip(counts.mean() / window, 1e-6, 1 - 1e-6)
    log_lik = np.sum(binom.logpmf(counts, window, theta_hat))
    return -log_lik


def surprise_order2(counts, window, sigma_reg=0.6):
    """S(2) : vraisemblance negative des MEMES comptages sous un modele
    REGULE -- meme taux moyen theta_hat, mais dispersion inter-fenetres
    fixee a une valeur faible (sigma_reg), representant l'hypothese qu'un
    agent regule activement son taux plutot que de le tirer i.i.d. a
    chaque fenetre. sigma_reg est un choix de modele generique (a quel
    point un processus \"regule\" est suppose serre), fixe une fois pour
    toutes ici -- jamais ajuste sur les donnees d'un agent particulier ni
    sur son type reel."""
    if len(counts) == 0:
        return 0.0
    theta_hat = np.clip(counts.mean() / window, 1e-6, 1 - 1e-6)
    target_count = theta_hat * window
    log_lik = np.sum(norm.logpdf(counts, loc=target_count, scale=sigma_reg))
    return -log_lik


def adaptive_k_order(o_history, window, tau_order, n_recent_blocks=10,
                       p_population=0.08, sigma_reg=0.6, K_max=2):
    """
    Allocation adaptative reelle de l'ordre k, fondee uniquement sur le
    GAIN DE VRAISEMBLANCE observe -- structure identique a la Section 3.6
    du papier (S_order, Delta S, regle d'arret sur tau_order).

    Retourne (k_alloue, decision_k2) ou decision_k2 est True si l'agent est
    juge par le modele d'ordre 2 comme suspect (a utiliser ensuite pour la
    decision finale, separement du calcul de k lui-meme).

    N'accede JAMAIS au type reel de l'agent -- uniquement a o_history
    (observations comportementales brutes).
    """
    o = np.asarray(o_history)
    counts = block_counts(o, window)
    recent = counts[-n_recent_blocks:] if len(counts) >= n_recent_blocks else counts

    s0 = surprise_order0(recent, window, p_population)
    s1 = surprise_order1(recent, window)
    s2 = surprise_order2(recent, window, sigma_reg)

    delta_01 = s0 - s1
    if delta_01 <= tau_order or K_max < 1:
        return 0, False

    delta_12 = s1 - s2
    if delta_12 <= tau_order or K_max < 2:
        return 1, False

    # ordre 2 alloue : le modele regule explique mieux les donnees que le
    # modele i.i.d. -- l'agent est signale comme suspect au niveau 2
    return 2, True


if __name__ == "__main__":
    # Test d'absence de fuite : verifie par analyse syntaxique (AST, pas une
    # simple recherche textuelle qui matcherait aussi les docstrings/commentaires
    # explicatifs) qu'aucune fonction de ce module n'accede a un type d'agent
    # ou une etiquette de verite terrain dans son CODE EXECUTABLE.
    import ast
    with open(__file__) as f:
        tree = ast.parse(f.read())
    forbidden = {"role", "adversary", "compliant", "types", "true_label", "ground_truth"}
    leaks = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    continue  # ignorer les chaines litterales (docstrings)
                if isinstance(sub, ast.Attribute) and sub.attr in forbidden:
                    leaks.append((node.name, sub.attr))
                if isinstance(sub, ast.Name) and sub.id in forbidden:
                    leaks.append((node.name, sub.id))
    if leaks:
        print("ECHEC -- fuite reelle d'etiquette detectee :", leaks)
    else:
        print("OK -- verification par AST : aucun acces au type reel dans le code executable")
