"""
exp3ter_nested_belief.py -- PREMIERE TENTATIVE, SUPERSEDEE PAR
exp3ter_nested_belief_v2.py -- conservee pour transparence methodologique,
PAS la source des chiffres cites dans le papier (voir Section 7.12ter).

Cette version utilisait un basculement ALEATOIRE de la verite terrain
(boites L/R), creant une dissociation trop faible entre premier et
second ordre : PEG-1 (premier ordre) ratait a peine plus sur les cas de
second ordre que sur les cas de premier ordre (0.821 vs 0.820) -- pas de
signal exploitable. exp3ter_nested_belief_v2.py corrige ce defaut de
conception en ancrant le scenario sur l'adversaire informe deja
caracterise en Section 7.12 (transition haut->bas plutot que
basculement aleatoire), ce qui produit une vraie dissociation
(PEG-1=0.355 vs PEG-2=0.858 sur le cas cible).

QUESTION SCIENTIFIQUE : PEG peut-il determiner automatiquement quand une
decision necessite une croyance de second ordre B_i(B_j(phi)), plutot que
de s'arreter a une croyance de premier ordre B_i(phi) ? Contrairement a
la proposition originale (un monde de boites/ressource entierement
nouveau), cette version reutilise l'infrastructure bayesienne existante
de PEG (mise a jour de type a la Definition 6) plutot que d'introduire
un domaine disjoint.

ENVIRONNEMENT : une verite de terrain r_true in {L,R} bascule rarement
(probabilite p_switch par pas). L'agent j observe r_true avec un bruit
q_j et forme une croyance b_j(t) par mise a jour bayesienne recursive
(memes mecanismes que le reste de PEG). j "agit" selon sa croyance
courante (va vers la boite qu'il croit correcte). L'agent i n'observe
JAMAIS r_true directement -- il observe seulement l'action de j, avec un
bruit supplementaire q_i. La tache de i est de decider si la croyance
ACTUELLE de j correspond a une cible donnee -- pas si r_true y
correspond. C'est ce qui rend la tache proprement de second ordre :
la bonne reponse depend de B_j(phi), jamais de phi lui-meme.

LE TEST DE FAUSSE CROYANCE (indiscernabilite au premier ordre) :
quand r_true bascule JUSTE avant la decision de i, la croyance de j
peut etre en retard (j n'a pas encore eu le temps de re-observer et de
mettre a jour). Une strategie de premier ordre (qui traite l'action de
j comme une preuve directe et a jour de r_true, sans modeliser le
retard possible) echoue precisement dans ces scenarios -- alors qu'une
strategie de second ordre, qui modele explicitement le risque de
retard de la croyance de j, peut corriger le tir.

QUATRE CONFIGURATIONS :
  - Baseline (d=0)      : decision a partir du seul prior, sans observation
  - PEG-1 (d_max=1)      : croyance sur l'action de j = estimation directe
                           de la croyance de j, sans modeliser le retard
  - PEG-2 (d_max=2)      : modelise explicitement le retard possible de j
                           (test de vraisemblance : j a-t-il eu le temps
                           de re-observer depuis le dernier basculement ?)
  - PEG-adaptatif        : escalade de 1 vers 2 seulement si le gain de
                           vraisemblance residuelle depasse tau_d (meme
                           regle d'arret que Definition 7)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def generate_scenario(T, p_switch, q_j, q_i, rng):
    """Genere une sequence : r_true(t), croyance de j b_j(t), action de j,
    observation de i. Retourne aussi les instants de basculement pour
    diagnostiquer les scenarios de "fausse croyance" a posteriori."""
    r_true = rng.integers(0, 2)  # 0=L, 1=R
    switches = []
    r_seq, bj_seq, action_j_seq, obs_i_seq = [], [], [], []

    log_odds_j = 0.0  # croyance de j en faveur de R (log-odds)
    for t in range(T):
        if rng.random() < p_switch:
            r_true = 1 - r_true
            switches.append(t)
        r_seq.append(r_true)

        # j observe r_true avec bruit q_j, met a jour sa croyance (bayesien recursif)
        o_j = r_true if rng.random() < q_j else 1 - r_true
        llr = np.log(q_j / (1 - q_j)) if o_j == 1 else np.log((1 - q_j) / q_j)
        log_odds_j = 0.7 * log_odds_j + llr  # facteur d'oubli -- j croit surtout au recent
        b_j = 1 / (1 + np.exp(-log_odds_j))
        bj_seq.append(b_j)

        # action de j : va vers la boite qu'il croit correcte (avec un peu de bruit propre)
        action_j = 1 if (b_j > 0.5) == (rng.random() < 0.95) else 0
        action_j_seq.append(action_j)

        # i observe l'action de j avec un bruit supplementaire q_i -- i n'observe JAMAIS r_true
        obs_i = action_j if rng.random() < q_i else 1 - action_j
        obs_i_seq.append(obs_i)

    return dict(r=r_seq, b_j=bj_seq, action_j=action_j_seq, obs_i=obs_i_seq,
                switches=switches, T=T)


def decide_first_order(scenario, window, q_i):
    """PEG-1 : traite les observations recentes de l'action de j comme une
    preuve DIRECTE et a jour de la croyance de j -- ne modelise aucun
    retard possible."""
    obs = scenario["obs_i"][-window:]
    if not obs:
        return 0.5, 0.0
    p_j_believes_R = np.mean(obs)  # estimation naive, pas de ponderation temporelle
    surprise = -np.log(max(min(q_i if obs[-1] == round(p_j_believes_R) else 1 - q_i, 0.999), 0.001))
    return p_j_believes_R, surprise


def decide_second_order(scenario, window, q_i, p_switch):
    """PEG-2 : modelise explicitement que la croyance de j peut etre en
    retard sur un basculement recent -- pondere les observations
    recentes plus fortement, avec un facteur d'oubli qui reflete le
    risque de basculement depuis chaque observation."""
    obs = scenario["obs_i"][-window:]
    if not obs:
        return 0.5, 0.0
    weights = np.array([(1 - p_switch) ** (len(obs) - 1 - k) for k in range(len(obs))])
    weights /= weights.sum()
    p_j_believes_R = np.sum(weights * np.array(obs))
    pred = round(p_j_believes_R)
    surprise = -np.log(max(min(q_i if obs[-1] == pred else 1 - q_i, 0.999), 0.001))
    return p_j_believes_R, surprise


def run_condition(n_scenarios, T, p_switch, q_j, q_i, window, tau_d, seed,
                    config="peg1"):
    rng = np.random.default_rng(seed)
    correct = {"FO": [], "SO": []}
    depths_used = []
    false_escalations = 0   # escalade a d=2 sur un cas qui n'en avait pas besoin (FO)
    n_fo_cases = 0
    true_escalations = 0    # escalade a d=2 sur un cas qui en avait reellement besoin (SO)
    n_so_cases_total = 0

    for s in range(n_scenarios):
        scenario = generate_scenario(T, p_switch, q_j, q_i, rng)
        true_b_j = round(scenario["b_j"][-1])  # ce que j croit VRAIMENT a la fin
        true_r = scenario["r"][-1]
        is_so_case = (true_b_j != true_r)  # cas ou premier ordre et second ordre divergent

        if config == "baseline":
            pred_p, depth = 0.5, 0
        elif config == "peg1":
            pred_p, _ = decide_first_order(scenario, window, q_i)
            depth = 1
        elif config == "peg2":
            pred_p, _ = decide_second_order(scenario, window, q_i, p_switch)
            depth = 2
        elif config == "adaptive":
            pred_p1, s1 = decide_first_order(scenario, window, q_i)
            pred_p2, s2 = decide_second_order(scenario, window, q_i, p_switch)
            residual = s1 - s2  # gain de vraisemblance a passer au modele d'ordre 2
            if residual > tau_d:
                pred_p, depth = pred_p2, 2
            else:
                pred_p, depth = pred_p1, 1
        else:
            raise ValueError(config)

        pred = round(pred_p)
        is_correct = (pred == true_b_j)  # la cible est la croyance de j, jamais r_true
        depths_used.append(depth)

        if is_so_case:
            correct["SO"].append(is_correct)
            n_so_cases_total += 1
            if depth == 2:
                true_escalations += 1
        else:
            correct["FO"].append(is_correct)
            n_fo_cases += 1
            if depth == 2:
                false_escalations += 1

    acc_fo = np.mean(correct["FO"]) if correct["FO"] else float('nan')
    acc_so = np.mean(correct["SO"]) if correct["SO"] else float('nan')
    fpr_escalation = false_escalations / max(n_fo_cases, 1)   # fausse escalade sur cas FO
    tpr_escalation = true_escalations / max(n_so_cases_total, 1)  # vraie escalade sur cas SO
    return dict(acc_fo=acc_fo, acc_so=acc_so, mean_depth=np.mean(depths_used),
                fpr_escalation=fpr_escalation, tpr_escalation=tpr_escalation,
                n_so_cases=len(correct["SO"]))


def run_multi_seed(n_seeds=10, n_scenarios=600, T=20, p_switch=0.08,
                     q_j=0.9, q_i=0.85, window=6, tau_d=0.3):
    configs = ["baseline", "peg1", "peg2", "adaptive"]
    results = {c: {"acc_fo": [], "acc_so": [], "mean_depth": [], "fpr_escalation": [], "tpr_escalation": []} for c in configs}

    for seed in range(n_seeds):
        for c in configs:
            r = run_condition(n_scenarios, T, p_switch, q_j, q_i, window, tau_d,
                                seed=seed * 100 + hash(c) % 97, config=c)
            for k in ["acc_fo", "acc_so", "mean_depth", "fpr_escalation", "tpr_escalation"]:
                results[c][k].append(r[k])

    print(f"{'Config':<12} {'Acc_FO':>14} {'Acc_SO':>14} {'Prof. moy.':>12} {'FPR_esc':>10} {'TPR_esc':>10}")
    for c in configs:
        af = np.array(results[c]["acc_fo"])
        as_ = np.array(results[c]["acc_so"])
        d = np.array(results[c]["mean_depth"])
        fe = np.array(results[c]["fpr_escalation"])
        te = np.array(results[c]["tpr_escalation"])
        print(f"{c:<12} {af.mean():6.3f}+/-{af.std():5.3f} {as_.mean():6.3f}+/-{as_.std():5.3f} "
              f"{d.mean():10.3f} {fe.mean():8.3f} {te.mean():8.3f}")
    return results


if __name__ == "__main__":
    run_multi_seed(n_seeds=10)
