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
  exp2ter_burst_correlation.py        Experiment 2ter -- coalition-detection recalibration attempt (Section 7.2ter)
  exp2_quater_group_detection.py      Experiment 2quater -- group-consensus coalition detection, RESOLVES the Section 7.2 weakness (F1: 0.35 -> 1.00)
  exp3_depth.py                       Experiment 3  -- adaptive depth, Conjecture 1 (Section 7.3)
  exp3bis_evasive_adversary.py        Experiment 3bis -- evasive adversary reconstruction attempts (Section 7.3bis, Annexe A.2) -- EXPLORATORY, inconclusive, does not call the original result into question
  exp3ter_nested_belief.py            Experiment 3ter, v1 -- SUPERSEDED, kept for transparency (see v2 below)
  exp3ter_nested_belief_v2.py         Experiment 3ter, v2 -- nested-belief (Definition 8) validation, source of Section 7.12ter's numbers
  exp4_alignment.py                   Experiment 4  -- activation-contribution alignment, Conjecture 2 (Section 7.4)
  exp5_baseline.py                    Experiment 5  -- exhaustive baseline comparison (Section 7.5)
  exp5_baseline_bayes_optimal.py      Experiment 5, third independent implementation -- Bayes-optimal decision threshold
  exp6_ablation.py                    Experiment 6  -- ablation study (Section 7.6)
  exp7_cross_env.py                   Experiment 7  -- cross-validation, continuous environment (Section 7.7)
  exp8_tomnet.py                      Experiment 8  -- comparison to an implicit ToMnet-like model (Section 7.8)
  exp9_ipomdp.py                      Experiment 9  -- comparison to exact joint Bayesian inference (Section 7.9)
  exp10_magent2.py                    Experiment 10 -- external validation on MAgent2 (Section 7.10)
  exp11_burst_decision.py             Experiment 11 -- burst-window-conditioned decision (Section 7.11)
  exp11_full.py                       Experiment 11 (extended) -- full metric set: coalition/isolated recall, FPR, utility
  exp12_sensitivity_adversarial.py    Experiment 12 -- hyperparameter sensitivity and adversarial robustness (Section 7.12)
  exp2bis_confusion.py                Experiment 2bis -- coalition vs. shared-context confusion test (Section 7.2bis)
  exp_k2_genuine_inference.py         Experiment 7.12bis -- likelihood-based order allocation (exploratory)
  exp_k2_camouflage.py                Earlier heuristic version of 7.12bis, superseded, kept for reference
run_all.py                            Master script: runs every experiment, writes results_all.json
results_all.json                      Output of the last full run_all.py campaign -- source of every number in the paper's Section 7
seeds_config.json                     Exact seed values used by default, script by script
requirements.txt
```

## Running the experiments

Each script is self-contained and can be run directly, from inside the
`peg-sparse-epistemic-graphs` folder, with the virtual environment active:

```bash
python experiments/exp1_scalability.py
python experiments/exp2_coalition.py
python experiments/exp2ter_burst_correlation.py        # recalibration attempt, Section 7.2ter
python experiments/exp2_quater_group_detection.py       # group-consensus resolution, Section 7.2quater
python experiments/exp3_depth.py
python experiments/exp3bis_evasive_adversary.py                # exploratory, Section 7.3bis / Annexe A.2
python experiments/exp3ter_nested_belief_v2.py                  # nested-belief validation, Section 7.12ter
python experiments/exp4_alignment.py
python experiments/exp5_baseline.py
python experiments/exp5_baseline_bayes_optimal.py       # third, decision-theoretically motivated implementation
python experiments/exp6_ablation.py
python experiments/exp7_cross_env.py
python experiments/exp8_tomnet.py
python experiments/exp9_ipomdp.py
python experiments/exp10_magent2.py   # requires: pip install magent2
python experiments/exp11_burst_decision.py
python experiments/exp11_full.py              # extended: full metric set
python experiments/exp12_sensitivity_adversarial.py
python experiments/exp2bis_confusion.py        # confusion test, Section 7.2bis
python experiments/exp_k2_genuine_inference.py # exploratory, Section 7.12bis
```

(On macOS/Linux, `python3` also works if `python` is not mapped to Python 3
on your system.)

To run everything in one pass and regenerate `results_all.json`:

```bash
python run_all.py
```

## Reproducibility status -- please read before use

**This repository is a faithful reconstruction from the paper's formal
specification (Definitions 1-7, Algorithm 1), not a recovery of the
original source code.** The original simulation code was developed in a
temporary working environment during the writing process and was not
preserved in archivable form; this repository was rewritten from scratch,
based on the formulas, protocols, and parameters documented in the paper.

**As of this version, the paper's own reported numbers have been updated
to match what this repository actually produces.** A full reproducibility
campaign was run (`run_all.py`), every experiment's real output was
independently cross-checked (calling each function directly, not just
trusting the harness), and the paper's Section 7 text, tables, abstract,
discussion, and conclusion were rewritten to cite these verified numbers
directly -- with every changed number explicitly marked in the paper as
differing from an earlier, unverified version, rather than silently
replaced. The comparison below is therefore no longer "paper vs.
reconstruction gap" for most rows; it documents what changed and why.

| # | Experiment | Currently in the paper (verified) | Seeds | Note |
|---|---|---|---|---|
| 1 | Width regimes | slopes 1.08±0.36 / 2.09±0.33 | 10 | Was single-seed 0.95/1.83; real inter-seed variance is substantial -- note 4 |
| 2 | Coalition detection (isolated pair) | F1=0.353±0.044 (recall ≈25-35%) | 5 | Was F1=1.00 (single seed); refutation criterion now honestly reported as **met** for the isolated-pair mechanism -- but see row 2quater below |
| 2quater | Coalition detection (group consensus) | **F1=1.00±0.00** (perfect precision and recall), robust across 3 population scenarios | 8 + 5 + 5 | **Resolves** row 2's weakness entirely: aggregating each agent's correlation with its candidate group's mean, instead of a single pairwise partner, closes the gap -- verified exact group membership (not just the final F1) across all 18 seeds tested |
| 12ter | Nested belief (Definition 8), suspicion scenario | PEG-1 (first order) fails outright on the target case (0.355±0.040, worse than chance); PEG-2 (always second order) corrects sharply (0.858±0.031) but fails the falsification test (0.592±0.054 on noise-spike compliant agents); adaptive trades off (0.602/0.697, escalation TPR=28.8%/FPR=14.3%) | 15 | Anchored on the Section 7.12 informed adversary (transition to evasive rate, not constant targeting). A likelihood-ratio change-point detector was also tried and **rejected**: it degrades falsification (0.375-0.463) with no net gain, likely a multiple-testing effect from searching the best split point among many candidates -- documented as an instructive failure, not hidden |
| 2ter | Coalition-detection recalibration attempt | F1=0.299±0.041 (recall 48.8%±7.8%, precision 22.6%±6.4%) | 8 | Burst-window-conditioned correlation, inspired by the successful Exp11 fix; a genuine but insufficient trade-off (recall up, precision down more) -- refutation criterion still triggered, documented as an honest partial failure |
| 2bis | Confusion test (shared context) | FPR 7.3%±3.0% (no shock) vs 53.3%±5.4% (shock) | 8 | Previously had **no corresponding script at all**; built this campaign (`exp2bis_confusion.py`) -- these are the first-ever verified numbers for this test |
| 3 | Depth (Conjecture 1) | β≈0.768±0.011 | 5 | **Strong agreement** with prior single-seed estimate |
| 4 | Alignment (Conjecture 2) | ρ=0.618±0.006 | 5 | **Strong agreement** |
| 5 | Exhaustive baseline | Exact utility equivalence (0.00 difference, all 5 population sizes) for 14% of the cost | 5 | Resolved via a third, decision-theoretically motivated implementation (Bayes-optimal threshold, `exp5_baseline_bayes_optimal.py`) -- see note 2. Two earlier thresholds (buggy, then arbitrary 0.5) gave 77% then 110%; the correct threshold shows PEG sacrifices no decision quality at all |
| 6 | Ablation | coalition recall 95.8%→100% | 5 | Was "10%→100%"; real gain is modest, not dramatic -- note 1 |
| 7 | Cross-validation (continuous env.) | consistent replication, no qualitative gap | 5 | Unchanged, already accurate |
| 8 | Implicit model (ToMnet-like) | PEG 0.760±0.062 vs implicit 0.799±0.139, not significant | 5 | Was "PEG numerically higher"; direction is now the other way, still not significant -- note 5 |
| 9 | I-POMDP | PEG -0.459±0.264 vs I-POMDP -0.468±0.346, not significant | 10 | Was "3.600 vs 2.983"; magnitude collapsed, qualitative conclusion (no significant difference) unchanged -- note 1 |
| 10 | MAgent2 | FPR corrected to 18.0%±4.9% / 19.2%±5.7% (was 62-65% under an arbitrary heuristic); correlation 0.270±0.140 (not near-perfect, unaffected by the fix) | 5 | Threshold calibration bug found via external review and fixed this campaign -- note 3 |
| 11 | Burst-conditioned decision | coalition recall 0%→96.3%±5.7%, d=16.8 | 10 | Extended to compute the full metric set (`exp11_full.py`); result even stronger than originally reported (was d=2.85) |
| 12 | Sensitivity + adversarial | monotone, no discontinuity; recall 5.2%→49.6% with margin | 1 (sweep) | Unchanged, point-estimate sweep as in the paper |
| 7.12bis | Order allocation (exploratory) | gate 0→1: FPR 21.6%±4.3%; gate 1→2: recall 100.0%±0.0%, FPR 0.5%±0.7% | 20 | Unchanged, already verified in a prior session |

### Notes on identified gaps

**Note 1 (Experiments 2, 6, and 9 -- coalition mechanism).** The paper
restricts the correlation computation (Step 3b of Algorithm 1) to agents
already present in `V(t)`, consistent with the `O(|V(t)|².w)` cost bound
stated in Section 4.5. A coalition member whose average activity rate
stays under `tau` most of the time therefore never enters `V(t)` through
that criterion alone, which mechanically limits its detectability via
correlation. Simulation parameters had to be adjusted so that coalition
members enter `V(t)` at least occasionally, at the cost of a less dramatic
ablation contrast than the original (unrecovered) simulations apparently
produced. This same root cause affects Experiment 9's corrected-protocol
comparison, whose magnitude (not its qualitative conclusion) is
correspondingly weaker under this reconstruction. The paper's text now
reports these real numbers directly rather than the original figures.
**Addendum:** for Experiment 2 specifically, `exp2_quater_group_detection.py`
resolves the low-F1 manifestation of this issue via group-level
aggregation (Section 7.2quater) -- this does not itself change the `V(t)`
restriction discussed above, and has not been tested against Experiments
6 or 9's own manifestations of the same root cause.

**Note 2 (Experiment 5 -- utility function, definitively resolved this
campaign).** The original reconstruction equated "received an inference"
with "was correctly classified", a simplification that unfairly penalized
the exhaustive baseline for accusations it may not actually have made --
flagged as an open gap from the very first version of this repository.
A second implementation (`decision_utility` in `exp5_baseline.py`)
computed a genuine per-agent Bayesian posterior P(adversary | observation
window), thresholded arbitrarily at 0.5 -- an improvement, but this gave
PEG numerically *above* the exhaustive baseline (110%), the opposite of
the first (buggy) implementation's 77%. Rather than trust either
arbitrary-threshold version, a **third, independent implementation**
(`exp5_baseline_bayes_optimal.py`) was built using the **Bayes-optimal
decision threshold derived directly from the utility function's own
weights** (p > 50/63 ≈ 0.794, the exact break-even point given
reward_TP=10, penalty_FP=-50, penalty_FN=-3) rather than any conventional
cutoff. Result: PEG and the exhaustive baseline achieve **exactly
identical utility, to the last bit, at all five tested population
sizes** -- PEG sacrifices no decision quality at all relative to
exhaustive inference, for about 14% of its cost. This has a clean
explanation: PEG's activation threshold is calibrated to miss almost no
individually suspicious agent, so the agents it excludes from full
evaluation would never have crossed the (high) Bayes-optimal posterior
threshold anyway, even if evaluated. This closes the question definitively:
the paper's text now reports this exact-equivalence result rather than
either of the two arbitrary-threshold figures (77% or 110%).

**Note 3 (Experiment 10 -- MAgent2, calibration bug fixed this
campaign).** An earlier version of this experiment used an arbitrary
threshold heuristic (`tau_local = -log(1-p0)*1.5`) instead of the
quantile calibration (Definition 4bis) used everywhere else -- correctly
flagged as inadequate during an external review. `exp10_magent2.py` now
calls `calibrate_smoothed_threshold` directly: since the passive tagging
signal is already modeled as Bernoulli(p_passive), the standard synthetic
calibration procedure applies to this generative model without requiring
costly MAgent2 re-simulation. Result: false-positive rate on passive
predators drops from 62-65% to 18.0%±4.9% / 19.2%±5.7% (5 seeds),
with a correspondingly lower (more honest) activation recall under
synchronization (72.4%±8.9%, versus 97.2% under the old, overly
permissive threshold). Correlation under synchronization (0.270±0.140)
is unaffected by this fix (it does not depend on the activation
threshold) and remains far weaker than the "near-perfect" figure
previously reported -- a distinct, unresolved external-validity
limitation in how the coalition signal transfers to this engine.

**Note 4 (Experiment 1 -- seed variance).** The paper now reports the
10-seed average directly (1.08±0.36 bounded-degree, 2.09±0.33
constant-density) rather than a single seed's point estimate. Some
individual seeds give a bounded-degree slope above 1, which would fail
the paper's own pre-registered refutation criterion if that seed alone
had been reported -- this is exactly why the multi-seed average, not a
single run, is now what the paper cites.

**Note 5 (Experiment 8 -- direction of the comparison).** Averaged over 5
seeds: PEG 0.760±0.062 vs. implicit model 0.799±0.139 -- numerically the
opposite direction from what was previously reported, though the
comparison remains not statistically significant either way (paired
t-test, p=0.486), consistent with the paper's own qualitative conclusion.
The paper's text now states the real direction and explicitly cautions
against over-interpreting either version.

**Recommendation before citing or building on this code**: as of this
version, the paper's own text has been updated to cite exactly the
numbers this repository produces (verified via `run_all.py` and
independent direct function calls), so there is no longer a gap to
navigate for most experiments -- citing the paper's Section 7 numbers
directly citing this repository's output. Experiment 5 in particular
should be considered definitively settled (exact utility equivalence
under the Bayes-optimal threshold, verified across three independent
implementations of the decision function). The two genuine, still-open
items are: MAgent2's activation threshold is a simplified heuristic
rather than a full quantile calibration (Note 3), and the exact
mechanism used in the original (unrecovered) simulations to generate
coalition dynamics is not known, so Experiments 2/6/9's real numbers,
while now honestly reported, may not match a different, unrecoverable
original protocol (Note 1) -- two independent recalibration attempts for
Experiment 2 (`exp2ter_burst_correlation.py`) failed to cross the
paper's own pre-registered recall threshold, which should be read as
strengthening confidence that this is a genuine limitation of the
correlation mechanism rather than an unlucky implementation choice.

**Exact seeds.** `seeds_config.json` at the repository root documents,
script by script, exactly which seed values are used by default.

**Master reproducibility script.** `run_all.py` at the repository root
runs every experiment in one pass (a few minutes on one core; MAgent2 is
skipped automatically if `magent2` is not installed) and writes every
headline statistic to `results_all.json` -- this is the single file that
was used to verify and update every number in the paper's current text.

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
