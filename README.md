# PEG -- Sparse Epistemic Graphs

Reference implementation accompanying the paper *"PEG: Sparse Epistemic
Graphs for Selective and Scalable Theory of Mind Inference in Multi-Agent
Systems"* (Yakouda, Kamla, Houpa Danga -- University of Ngaoundéré).

This repository implements Algorithm 1 from the paper (Section 4.3) and all
twelve numbered experiments from Section 7, for the purpose of
reproducibility and independent verification of the reported results.

## Installation

First, check which command your system uses for Python 3. Some systems use
`python3`, others (especially Windows) use `python`. Check with:

```bash
python3 --version
```

or, if that fails:

```bash
python --version
```

Either way, make sure it reports **Python 3.x** (e.g. `Python 3.11.2`), not
Python 2.

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Windows -- Command Prompt (cmd.exe)

```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Windows -- PowerShell

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell refuses to run the activation script with a message about
execution policies, run this once first (in the same PowerShell window),
then retry:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Once the virtual environment is active, your prompt should start with
`(venv)`. From then on, use `python` (not `python3`) to run the scripts
below, on every platform, as long as the virtual environment stays active.

Tested with Python 3.12, NumPy 2.4, and a recent SciPy. Experiment 10
(MAgent2) additionally requires the `magent2` package (Farama Foundation),
which is **not** included in `requirements.txt` to keep the base install
light:

```bash
pip install magent2
```

## Repository structure

```
env.py                                Population, reference model, smoothed surprise (binary environment)
env_continuous.py                     Continuous variant (Gaussian telemetry), used by Experiment 7
peg_algorithm.py                      Algorithm 1 (Section 4.3)
experiments/
  exp1_scalability.py                 Experiment 1  -- breadth regimes (Section 7.1)
  exp2_coalition.py                   Experiment 2  -- coalition detection (Section 7.2)
  exp3_depth.py                       Experiment 3  -- adaptive depth, Conjecture 1 (Section 7.3)
  exp4_alignment.py                   Experiment 4  -- activation-contribution alignment, Conjecture 2 (Section 7.4)
  exp5_baseline.py                    Experiment 5  -- exhaustive baseline comparison (Section 7.5)
  exp6_ablation.py                    Experiment 6  -- ablation study (Section 7.6)
  exp7_cross_env.py                   Experiment 7  -- cross-validation, continuous environment (Section 7.7)
  exp8_tomnet.py                      Experiment 8  -- comparison to an implicit ToMnet-like model (Section 7.8)
  exp9_ipomdp.py                      Experiment 9  -- comparison to exact joint Bayesian inference (Section 7.9)
  exp10_magent2.py                    Experiment 10 -- external validation on MAgent2 (Section 7.10)
  exp11_burst_decision.py             Experiment 11 -- burst-window-conditioned decision (Section 7.11)
  exp12_sensitivity_adversarial.py    Experiment 12 -- hyperparameter sensitivity and adversarial robustness (Section 7.12)
requirements.txt
```

## Running the experiments

Each script is self-contained and can be run directly, from inside the
`peg-sparse-epistemic-graphs` folder, with the virtual environment active:

```bash
python experiments/exp1_scalability.py
python experiments/exp2_coalition.py
python experiments/exp3_depth.py
python experiments/exp4_alignment.py
python experiments/exp5_baseline.py
python experiments/exp6_ablation.py
python experiments/exp7_cross_env.py
python experiments/exp8_tomnet.py
python experiments/exp9_ipomdp.py
python experiments/exp10_magent2.py   # requires: pip install magent2
python experiments/exp11_burst_decision.py
python experiments/exp12_sensitivity_adversarial.py
```

(On macOS/Linux, `python3` also works if `python` is not mapped to Python 3
on your system.)

## Reproducibility status -- please read before use

**This repository is a faithful reconstruction from the paper's formal
specification (Definitions 1-7, Algorithm 1), not a recovery of the
original source code.** The original simulation code was developed in a
temporary working environment during the writing process and was not
preserved in archivable form; this repository was rewritten from scratch,
based on the formulas, protocols, and parameters documented in the paper.

Every script has been executed and checked before publication. Agreement
with the results reported in the paper varies by experiment:

| # | Experiment | Paper result | Result obtained here | Status |
|---|---|---|---|---|
| 1 | Scalability | slopes 0.95 / 1.83 | slopes ≈0.90 / ≈1.91 | **Strong agreement** |
| 2 | Coalition | F1=1.00 (synchronized coalition) | F1≈0.31 (best threshold) | Qualitative only -- note 1 |
| 3 | Depth (Conjecture 1) | β≈0.77 | β≈0.78 | **Strong agreement** |
| 4 | Alignment (Conjecture 2) | ρ=0.617 | ρ≈0.618 | **Strong agreement** |
| 5 | Exhaustive baseline | cost/quality trade-off (Pareto) | PEG dominates on both axes | Identified gap, unresolved -- note 2 |
| 6 | Ablation | coalition recall 10%→100% | recall 90%→100% (weaker contrast) | Qualitative only -- note 1 |
| 7 | Cross-validation (continuous env.) | generalization confirmed | slopes ≈0.86/≈1.97; F1≈0.68; ρ≈0.633 | **Reasonable agreement** |
| 8 | Implicit model (ToMnet-like) | non-significant difference | near-identical utilities | **Qualitative agreement** |
| 9 | I-POMDP | PEG ≥ I-POMDP at sufficient scale | PEG ≥ I-POMDP at every tested n | **Qualitative agreement** |
| 10 | MAgent2 | degraded recall + near-perfect correlation under synchronization | high recall; correlation rises from -0.01 to 0.29 | Directionally correct signal, FPR calibration rough -- note 3 |
| 11 | Burst-conditioned decision | solved (synchronized), collapses under desync. | solved (synchronized); recall 100%→66% under increasing desync. | **Strong qualitative agreement** |
| 12 | Sensitivity + adversarial | monotone, no discontinuity; recall grows with margin (3.4%→42%) | monotone, no discontinuity; recall 5.2%→49.6% | **Strong agreement** |

### Notes on identified gaps

**Note 1 (Experiments 2 and 6 -- coalition mechanism).** The paper restricts
the correlation computation (Step 3b of Algorithm 1) to agents already
present in `V(t)`, consistent with the `O(|V(t)|².w)` cost bound stated in
Section 4.5. A coalition member whose average activity rate stays under
`tau` most of the time therefore never enters `V(t)` through that
criterion alone, which mechanically limits its detectability via
correlation. Simulation parameters had to be adjusted so that coalition
members enter `V(t)` at least occasionally, at the cost of a less dramatic
ablation contrast than in the paper. The exact mechanism used in the
original simulations to reconcile these two constraints could not be
recovered.

**Note 2 (Experiment 5 -- utility function).** The reconstruction equates
"received an inference" with "was correctly classified", a simplification
that unfairly penalizes the exhaustive baseline for accusations it may not
actually have made. A faithful decision model would need to explicitly
simulate the probabilistic output of the Bayesian inference for each
processed pair. This limitation remains open.

**Note 3 (Experiment 10 -- MAgent2).** The local activation threshold used
in this experiment is a simplified heuristic, not a full quantile
calibration as in the other experiments -- hence the elevated false-positive
rate among passive predators. The expected directional signal is present
nonetheless: correlation between aggressive predators rises sharply under
synchronization, confirming that the coalition mechanism picks up a real
structure in data generated by a third-party engine.

**Recommendation before citing or building on this code**: Experiments 1,
3, 4, 11, and 12 can be considered faithfully reproduced, with strong
numerical agreement. Experiments 2, 6, 9, and 10 demonstrate that the
mechanism works qualitatively and in the right direction, but their precise
numerical results should not be cited as equivalent to the paper's without
recalibration. Experiment 5 contains an identified, unresolved
methodological gap -- its qualitative finding (trade-off, not domination)
should be considered established by the paper itself, not by this script
as written.

## Random seeds

Every script fixes its random seeds explicitly (`seed=` in function calls)
for intra-script reproducibility. The seeds used are visible directly in
each experiment's source code.


