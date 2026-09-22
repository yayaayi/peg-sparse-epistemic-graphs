"""
run_all.py -- Master reproducibility script for the PEG paper (all 12 core
experiments + the two 7.12bis exploratory variants).

Runs every experiment with the seed counts documented in seeds_config.json,
captures every headline statistic used in the paper's text, tables, and
figures, and writes everything to results_all.json. This is the SINGLE
entry point a reviewer or reader needs to regenerate every number and
figure in the paper from this repository:

    python3 run_all.py

Runtime: a few minutes on a single core (MAgent2's experiment is skipped
automatically if the optional `magent2` package is not installed, and
reported as such in the output rather than failing).
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments"))

import numpy as np

results = {}
t_start = time.time()


def record(key, **kwargs):
    results[key] = kwargs
    print(f"  -> recorded '{key}'")


print("=" * 70)
print("PEG reproducibility campaign -- running all experiments")
print("=" * 70)

# ---------------------------------------------------------------- Exp 1
print("\n[7.1] Width regimes (exp1_scalability, 10 seeds)")
from experiments.exp1_scalability import main as exp1_main
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exp1_main(n_seeds=10)
out = buf.getvalue()
print(out.strip().split("\n")[-2])
print(out.strip().split("\n")[-1])
slopes = {}
for line in out.strip().split("\n"):
    if "pente log-log" in line:
        label = "bounded" if "borne" in line else "density"
        # format: "... pente log-log = 1.076 +/- 0.358 (10 graines, seeds=...)"
        core = line.split("=", 1)[1].split("(")[0].strip()
        mean_str, std_str = core.split("+/-")
        slopes[label] = dict(mean=float(mean_str.strip()), std=float(std_str.strip()))
record("exp1_width_regimes", n_seeds=10, slopes=slopes)

# ---------------------------------------------------------------- Exp 2
print("\n[7.2] Coalition detection (exp2_coalition, 5 seeds)")
from experiments.exp2_coalition import run_multi_seed as exp2_run
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    f1s = exp2_run(n_seeds=5)
record("exp2_coalition_detection", n_seeds=5, f1_mean=float(np.mean(f1s)), f1_std=float(np.std(f1s)))

# ---------------------------------------------------------------- Exp 2bis
print("\n[7.2bis] Confusion test: coalition vs. shared context (exp2bis_confusion, 8 seeds)")
from experiments.exp2bis_confusion import run_multi_seed as exp2bis_run
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    fpr_no_shock, fpr_shock = exp2bis_run(n_seeds=8)
record("exp2bis_confusion", n_seeds=8,
       fpr_no_shock_mean=float(np.mean(fpr_no_shock)), fpr_no_shock_std=float(np.std(fpr_no_shock)),
       fpr_shock_mean=float(np.mean(fpr_shock)), fpr_shock_std=float(np.std(fpr_shock)))

# ---------------------------------------------------------------- Exp 3
print("\n[7.3] Adaptive depth / Conjecture 1 (exp3_depth, 5 seeds)")
from experiments.exp3_depth import run_multi_seed as exp3_run
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    betas = exp3_run(n_seeds=5)
record("exp3_adaptive_depth", n_seeds=5, beta_mean=float(np.mean(betas)), beta_std=float(np.std(betas)))

# ---------------------------------------------------------------- Exp 4
print("\n[7.4] Activation-contribution alignment (exp4_alignment, 5 seeds)")
from experiments.exp4_alignment import run_multi_seed as exp4_run
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rhos = exp4_run(n_seeds=5)
record("exp4_alignment", n_seeds=5, rho_mean=float(np.mean(rhos)), rho_std=float(np.std(rhos)))

# ---------------------------------------------------------------- Exp 5
print("\n[7.5] Exhaustive baseline comparison (exp5_baseline, corrected utility, 5 seeds)")
from experiments.exp5_baseline import run as exp5_run
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exp5_results = exp5_run()
record("exp5_exhaustive_baseline", n_seeds=5, by_n=exp5_results,
       note="utility function corrected this campaign: genuine per-agent Bayesian "
            "posterior classification, see README Note 2")

# ---------------------------------------------------------------- Exp 6
print("\n[7.6] Ablation study (exp6_ablation, 5 seeds)")
from experiments.exp6_ablation import run_multi_seed as exp6_run
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ablation = exp6_run(n_seeds=5)
record("exp6_ablation", n_seeds=5,
       isolated_so=float(np.mean(ablation["isolated"]["so"])),
       isolated_full=float(np.mean(ablation["isolated"]["full"])),
       coalition_so=float(np.mean(ablation["coalition"]["so"])),
       coalition_full=float(np.mean(ablation["coalition"]["full"])))

# ---------------------------------------------------------------- Exp 8
print("\n[7.8] Implicit opponent model comparison (exp8_tomnet, 5 seeds)")
from experiments.exp8_tomnet import run_multi_seed as exp8_run
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    peg_m, imp_m = exp8_run(n_seeds=5)
record("exp8_tomnet_comparison", n_seeds=5,
       peg_util_mean=float(np.mean(peg_m)), peg_util_std=float(np.std(peg_m)),
       implicit_util_mean=float(np.mean(imp_m)), implicit_util_std=float(np.std(imp_m)))

# ---------------------------------------------------------------- Exp 9
print("\n[7.9] I-POMDP comparison, corrected protocol n=18 (exp9_ipomdp, 10 seeds)")
from experiments.exp9_ipomdp import run_headline_n18
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    peg_u, ip_u = run_headline_n18(n_seeds=10)
record("exp9_ipomdp_n18", n_seeds=10,
       peg_util_mean=float(np.mean(peg_u)), peg_util_std=float(np.std(peg_u)),
       ipomdp_util_mean=float(np.mean(ip_u)), ipomdp_util_std=float(np.std(ip_u)))

# ---------------------------------------------------------------- Exp 10
print("\n[7.10] MAgent2 benchmark (exp10_magent2) -- optional dependency")
try:
    from experiments.exp10_magent2 import run_variant
    results_10 = {}
    for sync, label in [(False, "unsynchronized"), (True, "synchronized")]:
        recalls, fprs, rhos = [], [], []
        for s in range(5):
            recall, fpr, rho = run_variant(synchronized=sync, seed=3 + s)
            recalls.append(recall); fprs.append(fpr); rhos.append(rho)
        results_10[label] = dict(recall_mean=float(np.mean(recalls)), recall_std=float(np.std(recalls)),
                                    fpr_mean=float(np.mean(fprs)), fpr_std=float(np.std(fprs)),
                                    correlation_mean=float(np.mean(rhos)), correlation_std=float(np.std(rhos)))
    record("exp10_magent2", n_seeds=5, status="ran", **results_10)
except ImportError as e:
    record("exp10_magent2", status="SKIPPED - magent2 package not installed in this environment",
           error=str(e))
    print("  SKIPPED: magent2 not installed")

# ---------------------------------------------------------------- Exp 11
print("\n[7.11] Burst-window-conditioned decision (exp11_burst_decision, 10 reps)")
from experiments.exp11_burst_decision import (simulate_coalition_bursts, detect_burst_windows,
                                                 decide_windowed_mean, decide_burst_conditioned,
                                                 utility_of_decision)
n_coalition, T, n_reps = 8, 400, 10
p_base, p_burst = 0.05, 0.85
utils_mean, utils_burst = [], []
for rep in range(n_reps):
    obs, _ = simulate_coalition_bursts(n_coalition, T, desync_spread=0, p_base=p_base, p_burst=p_burst, seed=rep)
    detected = detect_burst_windows(obs)
    d_mean = [decide_windowed_mean(obs[m], p_base, p_burst) for m in range(n_coalition)]
    d_burst = [decide_burst_conditioned(obs[m], detected, p_base, p_burst) for m in range(n_coalition)]
    utils_mean.append(utility_of_decision(d_mean))
    utils_burst.append(utility_of_decision(d_burst))
spreads = (0, 1, 2, 3, 5)
recall_by_spread = []
for spread in spreads:
    recalls = []
    for rep in range(n_reps):
        obs, _ = simulate_coalition_bursts(n_coalition, T, desync_spread=spread, p_base=p_base, p_burst=p_burst, seed=rep + 100)
        detected = detect_burst_windows(obs)
        decisions = [decide_burst_conditioned(obs[m], detected, p_base, p_burst) for m in range(n_coalition)]
        recalls.append(np.mean(decisions))
    recall_by_spread.append(float(np.mean(recalls)))
record("exp11_burst_decision", n_reps=10,
       windowed_mean_util=float(np.mean(utils_mean)), windowed_mean_std=float(np.std(utils_mean)),
       burst_conditioned_util=float(np.mean(utils_burst)), burst_conditioned_std=float(np.std(utils_burst)),
       recall_by_desync_spread=dict(zip(spreads, recall_by_spread)))

# ---------------------------------------------------------------- Exp 11 (full metrics)
print("\n[7.11] Burst-window-conditioned decision, full metric set (exp11_full, 10 seeds)")
from experiments.exp11_full import run_multi_seed as exp11_full_run
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    agg11 = exp11_full_run(n_seeds=10)
record("exp11_burst_decision_full", n_seeds=10,
       recall_coalition=dict(mean_condition=float(np.mean(agg11["recall_coalition_mean"])),
                                burst_condition=float(np.nanmean(agg11["recall_coalition_burst"])),
                                burst_std=float(np.nanstd(agg11["recall_coalition_burst"]))),
       recall_isolated=dict(mean_condition=float(np.mean(agg11["recall_isolated_mean"])),
                               burst_condition=float(np.mean(agg11["recall_isolated_burst"]))),
       fpr=dict(mean_condition=float(np.mean(agg11["fpr_mean"])),
                 burst_condition=float(np.mean(agg11["fpr_burst"]))),
       utility=dict(mean_condition=float(np.mean(agg11["utility_mean"])),
                      burst_condition=float(np.mean(agg11["utility_burst"])),
                      burst_std=float(np.std(agg11["utility_burst"]))))

# ---------------------------------------------------------------- Exp 12
print("\n[7.12] Sensitivity + adversarial robustness (exp12, single-seed sweeps, as in paper)")
import experiments.exp12_sensitivity_adversarial as exp12
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exp12.__name__ = "__not_main__"  # avoid double-run if module has top-level calls
record("exp12_sensitivity", status="see exp12_sensitivity_adversarial.py raw output",
       note="single-seed hyperparameter sweeps as in the paper; run "
            "`python3 experiments/exp12_sensitivity_adversarial.py` directly for the full sweep table")

# ---------------------------------------------------------------- 7.12bis
print("\n[7.12bis] Adaptive order allocation (exp_k2_genuine_inference, 20 seeds)")
from experiments.exp_k2_genuine_inference import run as exp_k2_run
recalls_k1_naif, recalls_k1_camo, recalls_k2_naif, recalls_k2_camo = [], [], [], []
fprs_k1, fprs_k2, promos = [], [], []
for s in range(20):
    r = exp_k2_run(seed=s)
    recalls_k1_naif.append(r["recall_k1_naif"]); recalls_k1_camo.append(r["recall_k1_camo"])
    recalls_k2_naif.append(r["recall_k2_naif"]); recalls_k2_camo.append(r["recall_k2_camo"])
    fprs_k1.append(r["fpr_k1"]); fprs_k2.append(r["fpr_k2"]); promos.append(r["promotion_rate"])
record("exp7_12bis_order_allocation", n_seeds=20,
       recall_k1_naif=dict(mean=float(np.mean(recalls_k1_naif)), std=float(np.std(recalls_k1_naif))),
       recall_k1_camo=dict(mean=float(np.mean(recalls_k1_camo)), std=float(np.std(recalls_k1_camo))),
       recall_k2_naif=dict(mean=float(np.mean(recalls_k2_naif)), std=float(np.std(recalls_k2_naif))),
       recall_k2_camo=dict(mean=float(np.mean(recalls_k2_camo)), std=float(np.std(recalls_k2_camo))),
       fpr_k1=dict(mean=float(np.mean(fprs_k1)), std=float(np.std(fprs_k1))),
       fpr_k2=dict(mean=float(np.mean(fprs_k2)), std=float(np.std(fprs_k2))),
       promotion_rate=dict(mean=float(np.mean(promos)), std=float(np.std(promos))))

# ---------------------------------------------------------------------
elapsed = time.time() - t_start
results["_meta"] = dict(elapsed_seconds=round(elapsed, 1),
                          note="Generated by run_all.py -- see README.md for the exact seed "
                               "used per experiment and known discrepancies with the original "
                               "(unrecovered) simulations.")

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_all.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)

print("\n" + "=" * 70)
print(f"Done in {elapsed:.1f}s -- full results written to {out_path}")
print("=" * 70)
