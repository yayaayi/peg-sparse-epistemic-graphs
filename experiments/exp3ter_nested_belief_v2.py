"""
exp3ter_nested_belief_v2.py -- "Scenario de suspicion", ancre sur
l'adversaire informe de la Section 7.12 (Section 7.12ter, exploratoire).

CORRECTION DE CONCEPTION par rapport a la version 1 (exp3ter_nested_belief.py) :
la v1 utilisait un basculement aleatoire de la verite terrain, creant une
dissociation trop faible entre premier et second ordre (PEG-1 ratait a
peine plus sur les cas de second ordre : 0.821 vs 0.820 sur le premier).

CETTE VERSION cree une dissociation structurelle forte, ancree sur un
phenomene deja documente dans le papier (Section 7.12 : un adversaire
informe de tau ajuste sa deviation pour rester juste sous le seuil).

MECANISME : certains agents adversaires "deviennent conscients" d'etre
suspectes (leur propre proxy de risque cumule depasse un seuil) et
basculent alors d'un taux de deviation eleve (p_high, non evasif) vers un
taux evasif juste sous tau -- exactement le mecanisme de la Section 7.12,
mais applique comme une TRANSITION plutot qu'un ciblage constant depuis
le debut.

LA CIBLE DE SECOND ORDRE (B_i^{(2)}(theta_i | j) tel que reformule) :
i doit inferer si le taux BAS actuellement observe chez j reflete (a) une
conformite genuine, ou (b) une evasion consciente apres une periode a
taux eleve. Ces deux cas sont \\textbf{indiscernables au premier ordre}
sur la seule fenetre recente (le taux courant est bas dans les deux cas)
-- seule la structure temporelle complete (y a-t-il eu une transition
haut->bas ?) permet de les distinguer. C'est la propriete de
dissociation qui manquait a la v1.

QUATRE CATEGORIES DE SCENARIOS :
  1. Compliant genuine (jamais de deviation elevee)          -- pas de mentalisation requise
  2. Adversaire non conscient (deviation elevee constante)    -- premier ordre suffit (facile a detecter)
  3. Adversaire evasif (transition haut -> bas, VRAI second ordre) -- second ordre necessaire
  4. Compliant avec un pic transitoire de bruit (imite une transition) -- test de FAUX second ordre
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from env import calibrate_threshold, calibrate_smoothed_threshold


def generate_agent_series(category, T, tau_margin_evasive, p_low, p_high, rng):
    """Genere la sequence d'observations binaires d'un agent selon sa
    categorie, et retourne aussi l'instant de transition (None si aucune)."""
    transition_t = None
    if category == "compliant":
        rate_seq = [p_low] * T
    elif category == "adv_unaware":
        rate_seq = [p_high] * T
    elif category == "adv_evasive":
        transition_t = rng.integers(T // 3, 2 * T // 3)
        rate_seq = [p_high] * transition_t + [tau_margin_evasive] * (T - transition_t)
    elif category == "compliant_noise_spike":
        transition_t = rng.integers(T // 3, 2 * T // 3)
        spike_len = max(2, T // 10)
        rate_seq = [p_low] * T
        for k in range(transition_t, min(transition_t + spike_len, T)):
            rate_seq[k] = p_high
    else:
        raise ValueError(category)

    obs = [1 if rng.random() < r else 0 for r in rate_seq]
    return obs, transition_t


def decide_first_order(obs, window):
    """Premier ordre : taux moyen sur la fenetre recente uniquement."""
    recent = obs[-window:]
    return np.mean(recent) if recent else 0.0


def decide_second_order(obs, window_short, window_long):
    """Second ordre : compare le taux recent (court terme) au taux sur un
    horizon plus long -- signal de transition brut, conserve pour
    reference/comparaison avec le detecteur ameliore ci-dessous."""
    recent = obs[-window_short:]
    longer = obs[-window_long:] if len(obs) >= window_long else obs
    rate_recent = np.mean(recent) if recent else 0.0
    rate_longer = np.mean(longer) if longer else 0.0
    transition_signal = max(0.0, rate_longer - rate_recent)
    return rate_recent, transition_signal


def _bernoulli_loglik(k, n, eps=1e-6):
    """Log-vraisemblance d'une sequence binaire de n essais, k succes,
    sous le taux MLE k/n (borne pour eviter log(0))."""
    if n == 0:
        return 0.0
    p = np.clip(k / n, eps, 1 - eps)
    return k * np.log(p) + (n - k) * np.log(1 - p)


def decide_second_order_v2(obs, window_long, min_segment):
    """Second ordre ameliore : test de rupture par rapport de
    vraisemblance (a la Page-Hinkley/CUSUM), pas une simple difference de
    taux entre deux fenetres fixes. Pour chaque point de coupure candidat
    s dans la fenetre longue, compare la vraisemblance d'un modele a UNE
    phase (taux constant) contre un modele a DEUX phases (taux avant/
    apres s). Le signal retenu est le MEILLEUR rapport de vraisemblance
    sur tous les points de coupure -- une transition SOUTENUE (evasion
    reelle) produit un rapport eleve car les deux segments sont longs et
    bien separes ; un simple pic BREF de bruit ne peut expliquer qu'un
    segment court, donc le meilleur rapport reste faible, meme si la
    difference brute de taux est similaire."""
    window = obs[-window_long:] if len(obs) >= window_long else obs
    n = len(window)
    if n < 2 * min_segment:
        rate_recent = np.mean(window) if window else 0.0
        return rate_recent, 0.0

    k_total = sum(window)
    ll_one_phase = _bernoulli_loglik(k_total, n)

    best_lr = 0.0
    k_before = 0
    for s in range(1, n):
        k_before += window[s - 1]
        n_before, n_after = s, n - s
        if n_before < min_segment or n_after < min_segment:
            continue
        k_after = k_total - k_before
        ll_two_phase = _bernoulli_loglik(k_before, n_before) + _bernoulli_loglik(k_after, n_after)
        lr = 2 * (ll_two_phase - ll_one_phase)
        # ne retenir qu'une rupture HAUT -> BAS (taux qui diminue), la
        # signature d'une evasion consciente, pas une hausse quelconque
        if n_before > 0 and n_after > 0 and (k_before / n_before) > (k_after / n_after):
            best_lr = max(best_lr, lr)

    rate_recent_short = np.mean(window[-min_segment:]) if window else 0.0
    return rate_recent_short, best_lr


def run_condition(n_scenarios, T, tau, p_low, p_high, window_fo, window_short,
                    window_long, tau_d, tau_d_lr, seed, config):
    rng = np.random.default_rng(seed)
    categories = ["compliant", "adv_unaware", "adv_evasive", "compliant_noise_spike"]
    weights = [0.35, 0.25, 0.25, 0.15]

    results_by_cat = {c: {"correct": [], "depth": []} for c in categories}
    false_escalations, n_fo_only_cases = 0, 0  # categories 1,2,4 = pas besoin reel de 2nd ordre
    true_escalations, n_so_cases = 0, 0        # categorie 3 = besoin reel

    for s in range(n_scenarios):
        cat = rng.choice(categories, p=weights)
        obs, transition_t = generate_agent_series(cat, T, tau, p_low, p_high, rng)

        # verite terrain de la tache : l'agent est-il ACTUELLEMENT "a risque"
        # (deviation elevee non deguisee OU evasion consciente en cours) ?
        is_currently_risky = cat in ("adv_unaware", "adv_evasive")
        needs_second_order = (cat == "adv_evasive")  # seul ce cas est indiscernable au 1er ordre

        rate_fo = decide_first_order(obs, window_fo)
        rate_recent, transition_signal = decide_second_order(obs, window_short, window_long)
        rate_recent_v2, lr_signal = decide_second_order_v2(obs, window_long, min_segment=window_short)

        if config == "baseline":
            pred, depth = False, 0
        elif config == "peg1":
            pred, depth = (rate_fo > (p_low + p_high) / 2), 1
        elif config == "peg2":
            pred = (rate_recent > (p_low + p_high) / 2) or (transition_signal > tau_d)
            depth = 2
        elif config == "peg2_v2":
            pred = (rate_recent_v2 > (p_low + p_high) / 2) or (lr_signal > tau_d_lr)
            depth = 2
        elif config == "adaptive":
            pred_fo = rate_fo > (p_low + p_high) / 2
            if transition_signal > tau_d:
                pred, depth = True, 2
            else:
                pred, depth = pred_fo, 1
        elif config == "adaptive_v2":
            pred_fo = rate_fo > (p_low + p_high) / 2
            if lr_signal > tau_d_lr:
                pred, depth = True, 2
            else:
                pred, depth = pred_fo, 1
        else:
            raise ValueError(config)

        correct = (pred == is_currently_risky)
        results_by_cat[cat]["correct"].append(correct)
        results_by_cat[cat]["depth"].append(depth)

        if needs_second_order:
            n_so_cases += 1
            if depth == 2:
                true_escalations += 1
        else:
            n_fo_only_cases += 1
            if depth == 2:
                false_escalations += 1

    out = {}
    for c in categories:
        out[f"acc_{c}"] = np.mean(results_by_cat[c]["correct"]) if results_by_cat[c]["correct"] else float('nan')
        out[f"depth_{c}"] = np.mean(results_by_cat[c]["depth"]) if results_by_cat[c]["depth"] else float('nan')
    out["fpr_escalation"] = false_escalations / max(n_fo_only_cases, 1)
    out["tpr_escalation"] = true_escalations / max(n_so_cases, 1)
    return out


def run_multi_seed(n_seeds=10, n_scenarios=800, T=40, tau=0.30, p_low=0.10,
                     p_high=0.55, window_fo=10, window_short=6, window_long=25,
                     tau_d=0.15, tau_d_lr=3.0):
    configs = ["baseline", "peg1", "peg2", "peg2_v2", "adaptive", "adaptive_v2"]
    all_results = {c: [] for c in configs}

    for seed in range(n_seeds):
        for c in configs:
            r = run_condition(n_scenarios, T, tau, p_low, p_high, window_fo,
                                window_short, window_long, tau_d, tau_d_lr,
                                seed=seed * 1000 + hash(c) % 997, config=c)
            all_results[c].append(r)

    print(f"{'Config':<12} {'Acc_compliant':>14} {'Acc_unaware':>13} {'Acc_EVASIF(SO)':>16} {'Acc_faux_pic':>13} {'FPR_esc':>9} {'TPR_esc':>9}")
    for c in configs:
        rs = all_results[c]
        def m(key):
            vals = [r[key] for r in rs]
            return np.mean(vals), np.std(vals)
        ac, _ = m("acc_compliant")
        au, _ = m("acc_adv_unaware")
        ae, ae_std = m("acc_adv_evasive")
        an, an_std = m("acc_compliant_noise_spike")
        fe, _ = m("fpr_escalation")
        te, _ = m("tpr_escalation")
        print(f"{c:<12} {ac:14.3f} {au:13.3f} {ae:9.3f}+/-{ae_std:.3f} {an:9.3f}+/-{an_std:.3f} {fe:9.3f} {te:9.3f}")
    return all_results


if __name__ == "__main__":
    run_multi_seed(n_seeds=10)
