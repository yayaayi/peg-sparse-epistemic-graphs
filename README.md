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

**Note 1 (Experiments 2, 6, and 9 -- coalition mechanism).** The paper
restricts the correlation computation (Step 3b of Algorithm 1) to agents
already present in `V(t)`, consistent with the `O(|V(t)|².w)` cost bound
stated in Section 4.5. A coalition member whose average activity rate
stays under `tau` most of the time therefore never enters `V(t)` through
that criterion alone, which mechanically limits its detectability via
correlation. Simulation parameters had to be adjusted so that coalition
members enter `V(t)` at least occasionally, at the cost of a less dramatic
ablation contrast than in the paper. The exact mechanism used in the
original simulations to reconcile these two constraints could not be
recovered. **This also affects Experiment 9's "corrected protocol"
headline comparison (paper: PEG 3.600±0.354 vs I-POMDP 2.983±1.244 at
n=18)**: `run_headline_n18()` in `exp9_ipomdp.py`, run across 10 seeds,
gives utilities close to zero for both methods (not the paper's reported
values) -- verified while preparing this archive, not previously checked
against multiple seeds. The qualitative finding this experiment is meant
to support (no statistically significant difference between PEG and exact
joint inference once enough coalition members are present) is still what
the reconstruction shows, but the specific reported numbers should not be
treated as reproduced.

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

**Note 4 (Experiment 1 -- seed variance not previously characterized).**
The paper reports single-seed log-log slopes (0.95 bounded-degree, 1.83
constant-density, seed=42). Averaged over 10 seeds while preparing this
archive: bounded-degree slope 1.076±0.358, constant-density 2.087±0.329.
Both regimes remain qualitatively correct (sub-quadratic vs
super-linear-approaching-quadratic), but the single-seed point estimates
carry more sampling variance than a bare point value suggests -- some
individual seeds give a bounded-degree slope above 1, which would fail the
paper's own pre-registered refutation criterion if that seed had been the
one reported. `exp1_scalability.py`'s `main()` now accepts `n_seeds` to
reproduce this averaged view directly.

**Note 5 (Experiment 8 -- direction of the comparison).** The paper reports
PEG numerically above the implicit ToMnet-like model. Averaged over 5
seeds while preparing this archive: PEG 0.760±0.062 vs implicit model
0.799±0.139 -- numerically the other way round, though the paired t-test
remains non-significant (p=0.486) either way, consistent with the paper's
own qualitative conclusion ("not statistically significant at this
sample size"). The direction of a non-significant comparison should not
be over-interpreted in either version.

**Recommendation before citing or building on this code**: Experiments 3,
4, 6, and 11 can be considered faithfully reproduced, with strong
numerical agreement across seeds. Experiment 1's qualitative regime
(sub-linear vs super-linear cost growth) holds, but its exact reported
slopes should be read as one representative seed among several rather
than a stable population mean (Note 4). Experiments 2, 10, and 12
demonstrate that the mechanism works qualitatively and in the right
direction, but their precise numerical results should not be cited as
equivalent to the paper's without recalibration. Experiments 8 and 9
reproduce the paper's qualitative conclusion (no statistically
significant difference from the comparison baseline) but not its specific
reported numbers or, for Experiment 8, even the reported direction (Notes
1 and 5) -- cite the qualitative finding, not the numbers. Experiment 5
contains an identified, unresolved methodological gap -- its qualitative
finding (trade-off, not domination) should be considered established by
the paper itself, not by this script as written.

**Exact seeds.** `seeds_config.json` at the repository root documents,
script by script, exactly which seed values are used by default and how
to reproduce the paper's reported statistics (or, where noted above, how
this reconstruction's own aggregated statistics differ from them).

## Experiment 7.12bis (exploratory, genuine likelihood-based order allocation)

`k_order_inference.py` implements a real generative-model comparison for
order allocation (levels 0/1/2), following exactly the paper's stopping-rule
structure (S_order, ΔS, τ_order): level 0 uses population-marginal rate,
level 1 uses a per-agent MLE-estimated Bernoulli rate, level 2 tests whether
that same rate is better explained by a *regulated* process (suppressed
inter-window variance) than an unregulated one. **No function in this module
ever reads an agent's true type** -- this is verified mechanically by an
AST-based static-analysis test included in the module (run
`python3 k_order_inference.py` directly to see it pass), not just asserted
in a docstring. `experiments/exp_k2_genuine_inference.py` runs the full
evaluation; ground truth is used only in that evaluation script, after the
fact, to compute recall/FPR -- never inside the detection logic itself.

Result (20 seeds): the first escalation gate (order 0→1) is **too
permissive** -- 21.6% ± 4.3% of compliant agents get escalated past level 0,
a real limitation, reported as such. The second gate (order 1→2, regulated
vs. unregulated model comparison) is sharp: 100.0% ± 0.0% recall on the
quota-camouflaging adversary (zero variance across all 20 seeds), 1.2% ±
3.8% on the naive fixed-rate adversary, 0.5% ± 0.7% false-positive rate on
compliant agents. This supersedes the earlier ad hoc dispersion-ratio
version (`exp_k2_camouflage.py`, kept in this repo for reference) which had
a notably higher false-positive rate (~6.7-10.9%) for comparable recall.

**Scope.** Same caveat as before: this validates a specific two-hypothesis
model comparison against a specific adversary model. It is not a general
k-order recursive ToM framework or a comparison to full recursive Bayesian
inference.

## Random seeds

Every script fixes its random seeds explicitly (`seed=` in function calls)
for intra-script reproducibility. The seeds used are visible directly in
each experiment's source code.

## License

To be completed by the authors before publication (e.g. MIT, Apache 2.0).

## Citation

To be completed with the full bibliographic reference once the paper is
published.
