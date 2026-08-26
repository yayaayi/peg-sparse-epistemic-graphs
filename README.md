# PEG -- Graphes épistémiques parcimonieux

Implémentation de référence accompagnant l'article *« PEG : graphes
épistémiques parcimonieux pour l'inférence sélective et scalable de la
Théorie de l'Esprit dans les systèmes multi-agents »* (Yakouda, Kamla,
Houpa Danga -- Université de Ngaoundéré).

Ce dépôt implémente l'Algorithme 1 du papier (Section 4.3) et l'ensemble
des douze expériences numérotées de la Section 7, à des fins de
reproductibilité et de vérification indépendante des résultats.

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Testé avec Python 3.12, NumPy 2.4, SciPy récent. L'Expérience 10 (MAgent2)
nécessite en plus le paquet `magent2` (Farama Foundation) :

```bash
pip install magent2
```

## Structure du dépôt

```
env.py                                Population, modèle de référence, surprise lissée (environnement binaire)
env_continuous.py                     Variante continue (télémétrie gaussienne), pour l'Expérience 7
peg_algorithm.py                      Algorithme 1 (Section 4.3)
experiments/
  exp1_scalability.py                 Expérience 1  -- régimes de largeur (Section 7.1)
  exp2_coalition.py                   Expérience 2  -- détection de coalition (Section 7.2)
  exp3_depth.py                       Expérience 3  -- profondeur adaptative, Conjecture 1 (Section 7.3)
  exp4_alignment.py                   Expérience 4  -- alignement activation-contribution, Conjecture 2 (Section 7.4)
  exp5_baseline.py                    Expérience 5  -- comparaison baseline exhaustive (Section 7.5)
  exp6_ablation.py                    Expérience 6  -- étude d'ablation (Section 7.6)
  exp7_cross_env.py                   Expérience 7  -- validation croisée, environnement continu (Section 7.7)
  exp8_tomnet.py                      Expérience 8  -- comparaison à un modèle implicite type ToMnet (Section 7.8)
  exp9_ipomdp.py                      Expérience 9  -- comparaison à une inférence bayésienne jointe exacte (Section 7.9)
  exp10_magent2.py                    Expérience 10 -- validation externe sur MAgent2 (Section 7.10)
  exp11_burst_decision.py             Expérience 11 -- décision conditionnée aux fenêtres de rafale (Section 7.11)
  exp12_sensitivity_adversarial.py    Expérience 12 -- sensibilité et robustesse adversariale (Section 7.12)
requirements.txt
```

## Exécution

Chaque script est autonome et peut être lancé directement :

```bash
python3 experiments/exp1_scalability.py
python3 experiments/exp2_coalition.py
# ... etc., un script par expérience
```

## Statut de reproductibilité -- à lire avant utilisation

**Ce dépôt est une reconstruction fidèle aux spécifications formelles du
papier (Définitions 1-7, Algorithme 1), pas une récupération du code source
original.** Le code original des simulations a été développé dans un
environnement de travail temporaire au cours de la rédaction et n'a pas été
conservé sous une forme archivable ; ce dépôt a été intégralement réécrit à
partir des formules, protocoles et paramètres documentés dans le papier.

Chaque script a été exécuté et vérifié avant publication. L'accord avec les
résultats rapportés dans le papier varie selon l'expérience :

| # | Expérience | Résultat papier | Résultat obtenu ici | Statut |
|---|---|---|---|---|
| 1 | Scalabilité | pentes 0,95 / 1,83 | pentes ≈0,90 / ≈1,91 | **Accord fort** |
| 2 | Coalition | F1=1,00 (coalition synchronisée) | F1≈0,31 (meilleur seuil) | Qualitatif seulement -- note 1 |
| 3 | Profondeur (Conjecture 1) | β≈0,77 | β≈0,78 | **Accord fort** |
| 4 | Alignement (Conjecture 2) | ρ=0,617 | ρ≈0,618 | **Accord fort** |
| 5 | Baseline exhaustive | compromis coût/qualité (Pareto) | PEG domine sur les deux axes | Écart identifié, non résolu -- note 2 |
| 6 | Ablation | rappel coalition 10 %→100 % | rappel 90 %→100 % (contraste réduit) | Qualitatif seulement -- note 1 |
| 7 | Validation croisée (env. continu) | généralisation confirmée | pentes ≈0,86/≈1,97 ; F1≈0,68 ; ρ≈0,633 | **Accord raisonnable** |
| 8 | Modèle implicite (ToMnet-like) | différence non significative | utilités quasi identiques | **Accord qualitatif** |
| 9 | I-POMDP | PEG ≥ I-POMDP à effectif suffisant | PEG ≥ I-POMDP à tout n testé | **Accord qualitatif** |
| 10 | MAgent2 | rappel dégradé + corrélation quasi parfaite sous synchronisation | rappel élevé ; corrélation monte de -0,01 à 0,29 | Signal directionnel correct -- note 3 |
| 11 | Décision par rafale | résolu (synchronisé), effondrement sous désync. | résolu (synchronisé) ; rappel 100 %→66 % sous désync. | **Accord qualitatif fort** |
| 12 | Sensibilité + adversarial | monotone sans discontinuité ; rappel croît avec la marge (3,4 %→42 %) | monotone sans discontinuité ; rappel 5,2 %→49,6 % | **Accord fort** |

### Notes sur les écarts identifiés

**Note 1 (Expériences 2 et 6 -- mécanisme de coalition).** Le papier
restreint le calcul de corrélation (Étape 3b de l'Algorithme 1) aux agents
déjà présents dans `V(t)`, conformément à la borne de coût `O(|V(t)|².w)`
énoncée Section 4.5. Un adversaire de coalition dont le taux d'activité
moyen reste sous le seuil `tau` la plupart du temps n'entre cependant jamais
dans `V(t)` par ce seul critère, ce qui limite mécaniquement sa détection
par corrélation. Les paramètres de simulation ont dû être ajustés pour que
les membres de coalition entrent au moins occasionnellement dans `V(t)`, au
prix d'un contraste d'ablation moins marqué que dans le papier. Le
mécanisme exact utilisé dans les simulations originales pour concilier ces
deux contraintes n'a pas pu être retrouvé.

**Note 2 (Expérience 5 -- fonction d'utilité).** La reconstruction assimile
« recevoir une inférence » à « être correctement classé », une
simplification qui pénalise à tort la baseline exhaustive pour des
accusations qu'elle n'a pas nécessairement portées. Un modèle de décision
fidèle exigerait de simuler explicitement la sortie probabiliste de
l'inférence bayésienne pour chaque paire traitée. Cette limitation reste
ouverte.

**Note 3 (Expérience 10 -- MAgent2).** Le seuil d'activation local utilisé
pour cette expérience est une heuristique simplifiée, pas une calibration
par quantile complète comme dans les autres expériences -- d'où un taux de
faux positifs élevé chez les prédateurs passifs. Le signal directionnel
attendu est néanmoins présent : la corrélation entre agressifs augmente
nettement sous synchronisation, confirmant que le mécanisme de coalition
capte une structure réelle sur des données générées par un moteur tiers.

**Recommandation avant citation ou usage en aval de ce code** : les
Expériences 1, 3, 4, 11 et 12 peuvent être considérées comme fidèlement
reproduites, avec un accord numérique fort. Les Expériences 2, 6, 9 et 10
démontrent que le mécanisme fonctionne qualitativement et dans la bonne
direction, mais leurs résultats numériques précis ne doivent pas être cités
comme équivalents à ceux du papier sans recalibration. L'Expérience 5
contient un écart méthodologique identifié et non résolu -- son résultat
qualitatif (compromis, pas domination) doit être considéré comme établi par
le papier, pas par ce script tel quel.

## Graines aléatoires

Chaque script fixe explicitement ses graines (`seed=` dans les appels de
fonction) pour la reproductibilité intra-script. Les graines utilisées sont
visibles directement dans le code de chaque expérience.

## Licence

À compléter par les auteurs avant publication (ex. MIT, Apache 2.0).

## Citation

À compléter avec la référence bibliographique complète une fois l'article
publié.
