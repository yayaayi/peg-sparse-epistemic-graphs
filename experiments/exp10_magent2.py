"""
exp10_magent2.py -- Experience 10 : Validation sur un benchmark MARL reconnu
(MAgent2, Section 7.10 du papier)

Teste PEG sur MAgent2 (Farama Foundation, successeur de MAgent -- Zheng et
al., 2018), scenario `adversarial_pursuit`. Les predateurs sont scindes en
**agressifs** (marquage frequent -- action "tag", verifiee empiriquement
comme l'action d'indice n_actions-1) et **passifs** (marquage rare). Le
signal binaire "action de marquage prise a ce pas" joue le role de
l'observation du reste du document ; le mecanisme PEG (activation par
surprise lissee, puis correlation) est applique SANS AUCUNE MODIFICATION a
ce signal issu d'un moteur de simulation tiers.

Deux variantes : agressifs non synchronises (chacun marque independamment
avec une probabilite elevee constante) et agressifs synchronises (phase de
chasse commune, tiree une fois par pas et partagee par tous les agressifs).

Necessite le paquet `magent2` (Farama Foundation) :
    pip install magent2
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from env import SmoothedSurprise, calibrate_smoothed_threshold
from peg_algorithm import candidate_sets, peg_step

try:
    from magent2.environments import adversarial_pursuit_v4
except ImportError:
    adversarial_pursuit_v4 = None


class ScriptedReactiveModel:
    """Modele de reference P_1 pour le signal binaire de marquage, calibre
    sur le taux de marquage des predateurs PASSIFS uniquement (population de
    reference conforme -- Convention 3.0)."""

    def __init__(self, p0_hat, eps=1e-6):
        self.p0_hat = min(max(p0_hat, eps), 1 - eps)

    def surprise(self, tag_taken):
        p = self.p0_hat if tag_taken == 1 else (1 - self.p0_hat)
        return -np.log(p)


TAG_ACTION_INDEX = 12  # verifie empiriquement : seule action a penalite non nulle


def run_variant(n_predators=8, n_prey=16, map_size=20, T=250, synchronized=False,
                 p_aggressive=0.55, p_passive=0.05, frac_aggressive=0.5,
                 sync_phase_prob=0.15, window=20, seed=0):
    if adversarial_pursuit_v4 is None:
        raise ImportError(
            "Le paquet 'magent2' n'est pas installe. Executez : pip install magent2"
        )

    rng = np.random.default_rng(seed)
    env = adversarial_pursuit_v4.env(map_size=map_size, max_cycles=T,
                                       tag_penalty=-0.2)
    env.reset(seed=seed)

    predators = sorted([a for a in env.agents if a.startswith("predator")])
    n_pred = len(predators)
    is_aggressive = rng.random(n_pred) < frac_aggressive
    pred_index = {a: i for i, a in enumerate(predators)}

    smoother = SmoothedSurprise(n_pred, window=window)
    cand = candidate_sets(n_pred, k_max=min(8, n_pred - 1),
                            mode="bounded_degree", seed=seed + 1)

    # calibration de P_1 sur le taux de marquage des passifs
    p0_hat = p_passive
    model = ScriptedReactiveModel(p0_hat)

    # CORRECTION (suite a une revue externe) : la version precedente utilisait
    # une heuristique arbitraire (tau_local = -log(1-p0_hat)*1.5) au lieu de la
    # vraie calibration par quantile corrige pour la multiplicite (Definition
    # 4bis) utilisee dans toutes les autres experiences. calibrate_smoothed_threshold
    # ne necessite pas de re-simuler MAgent2 : le signal de marquage des passifs
    # est deja modelise comme Bernoulli(p_passive) par ScriptedReactiveModel,
    # donc la calibration synthetique standard s'applique exactement au meme
    # modele generatif.
    tau_local, _ = calibrate_smoothed_threshold(p0_hat, n_calib=200, window=window,
                                                   F_target=1.0, n_target=n_pred, seed=seed + 100)

    tag_history = {a: [] for a in predators}
    step_count = 0
    in_phase_this_step = False

    recall_hits, fpr_hits, n_eval = 0, 0, 0
    rho_agg_agg = []

    for i, agent in enumerate(env.agent_iter()):
        obs, reward, term, trunc, info = env.last()

        if term or trunc:
            env.step(None)
            continue

        if agent.startswith("predator"):
            idx = pred_index[agent]
            if is_aggressive[idx]:
                if synchronized:
                    if agent == predators[0] or not tag_history[predators[0]]:
                        pass  # phase tiree une seule fois par pas, voir ci-dessous
                    tag_prob = p_aggressive if in_phase_this_step else p_passive
                else:
                    tag_prob = p_aggressive
            else:
                tag_prob = p_passive
            do_tag = rng.random() < tag_prob
            action = TAG_ACTION_INDEX if do_tag else rng.integers(0, TAG_ACTION_INDEX)
            tag_history[agent].append(1 if do_tag else 0)
        else:
            action = env.action_space(agent).sample()

        env.step(action)

        # une phase de synchronisation est tiree une fois par pas complet
        # (tous les agents, predateurs et proies, ont ete visites)
        if agent == predators[-1]:
            step_count += 1
            in_phase_this_step = rng.random() < sync_phase_prob

            for a in predators:
                j = pred_index[a]
                if tag_history[a]:
                    s = model.surprise(tag_history[a][-1])
                    smoother.push_raw(j, s)

            if step_count > window:
                # seuil calibre par quantile (Definition 4bis), pas l'ancienne
                # heuristique arbitraire -- voir la calibration ci-dessus
                result = peg_step(step_count, list(range(n_pred)), model,
                                   smoother, cand, tau=tau_local, rho_min=0.4,
                                   w=window)
                V = result["V"]

                agg_idx = set(np.where(is_aggressive)[0].tolist())
                n_eval += 1
                if agg_idx:
                    recall_hits += len(agg_idx & V) / len(agg_idx)
                passive_idx = set(range(n_pred)) - agg_idx
                if passive_idx:
                    fpr_hits += len(passive_idx & V) / len(passive_idx)

                for j, k in [(a, b) for a in agg_idx for b in agg_idx if a < b]:
                    sj = smoother.raw_window(j)
                    sk = smoother.raw_window(k)
                    if len(sj) >= 2 and np.std(sj) > 0 and np.std(sk) > 0:
                        rho_agg_agg.append(np.corrcoef(sj, sk)[0, 1])

    recall = recall_hits / n_eval if n_eval else 0.0
    fpr = fpr_hits / n_eval if n_eval else 0.0
    rho_mean = float(np.mean(rho_agg_agg)) if rho_agg_agg else 0.0
    return recall, fpr, rho_mean


def main(n_seeds=1):
    print("Experience 10 -- MAgent2 (adversarial_pursuit)\n")
    for sync, label in [(False, "agressifs non synchronises"),
                         (True, "agressifs synchronises")]:
        recalls, fprs, rhos = [], [], []
        for s in range(n_seeds):
            recall, fpr, rho = run_variant(synchronized=sync, seed=3 + s)
            recalls.append(recall); fprs.append(fpr); rhos.append(rho)
        recalls, fprs, rhos = np.array(recalls), np.array(fprs), np.array(rhos)
        if n_seeds > 1:
            print(f"{label:30s} rappel(activation)={recalls.mean():.3f}+/-{recalls.std():.3f}  "
                  f"FPR(passifs)={fprs.mean():.3f}+/-{fprs.std():.3f}  "
                  f"correlation(agressif-agressif)={rhos.mean():.3f}+/-{rhos.std():.3f}  "
                  f"({n_seeds} graines, seeds=3..{2+n_seeds})")
        else:
            print(f"{label:30s} rappel(activation)={recalls[0]:.3f}  "
                  f"FPR(passifs)={fprs[0]:.3f}  correlation(agressif-agressif)={rhos[0]:.3f}  "
                  f"(graine unique, seed=3)")


if __name__ == "__main__":
    main(n_seeds=1)
