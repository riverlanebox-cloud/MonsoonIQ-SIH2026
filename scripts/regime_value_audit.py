"""
MonsoonIQ regime-value audit.

Answers, with numbers, the questions a verification-literate judge asks about a
regime-conditional post-processor:

  1. How accurate is the regime classifier on held-out years, and why?
  2. How much of the corrected forecast's skill comes from the correction, and
     how much from simply being told the regime?
  3. How accurate must the regime classifier be before regime-conditioning
     stops beating a single regime-agnostic LightGBM?
  4. What does the regime-stratified heavy-rainfall table look like once you add
     event counts and block-bootstrap confidence intervals?
  5. Is the headline improvement over raw NWP still significant when the
     bootstrap respects the day-to-day correlation of the sample?

Outputs
  artifacts/metrics/regime_value_audit.json   machine-readable results
  artifacts/plots/regime_value_curve.png      MoE vs global-LGB crossover plot

Run:  PYTHONPATH=. python scripts/regime_value_audit.py
"""

import json
import os

import numpy as np
import pandas as pd

from src.regime.ml_classifier import MLRegimeClassifier
from src.correction.mixture_of_experts import MonsoonIQMixtureOfExperts
from src.verification.metrics import (
    compute_continuous_metrics,
    compute_contingency_table,
    compute_dichotomous_metrics,
)

DATA = "data/synthetic/district_daily.parquet"
METRICS_OUT = "artifacts/metrics/regime_value_audit.json"
PLOT_OUT = "artifacts/plots/regime_value_curve.png"
TEST_YEARS = [2022, 2023]
HEAVY = 64.5
VERY_HEAVY = 115.6
BOOT = 400
SEED = 42


def dich(pred, obs, thr=HEAVY):
    return compute_dichotomous_metrics(compute_contingency_table(pred, obs, thr))


def csi(pred, obs, thr=HEAVY):
    return dich(pred, obs, thr)["csi"]


def normalise(p):
    return p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)


def day_groups(days):
    """Pre-group row indices by calendar day for block resampling."""
    order = np.argsort(days, kind="stable")
    sorted_days = days[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_days[1:] != sorted_days[:-1]])
    return np.split(order, boundaries)


def day_block_indices(groups, rng):
    """Resample whole calendar days, so the bootstrap respects day-to-day correlation."""
    chosen = rng.randint(0, len(groups), size=len(groups))
    return np.concatenate([groups[i] for i in chosen])


def main():
    rng = np.random.RandomState(SEED)
    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    os.makedirs(os.path.dirname(PLOT_OUT), exist_ok=True)

    df = pd.read_parquet(DATA)
    test = df[df["year"].isin(TEST_YEARS)].reset_index(drop=True)
    y = test["obs_rain_mean"].to_numpy(float)
    y_max = test["obs_rain_max"].to_numpy(float)
    true_regime = test["regime"].to_numpy(int)
    n = len(test)
    days = test["date"].to_numpy()

    clf = MLRegimeClassifier()
    clf.load()
    moe = MonsoonIQMixtureOfExperts.load()

    # ------------------------------------------------------------------ 1
    p_clf = clf.predict_proba(test)
    pred_regime = np.array([clf.classes_[i] for i in np.argmax(p_clf, axis=1)])
    accuracy = float((pred_regime == true_regime).mean())
    per_class = {}
    for r in range(1, 8):
        m = true_regime == r
        if m.sum():
            per_class[clf.REGIME_NAMES[r]] = {
                "n": int(m.sum()),
                "recall": round(float((pred_regime[m] == r).mean()), 4),
                "mean_posterior_on_truth": round(float(p_clf[m, r - 1].mean()), 4),
            }

    # ------------------------------------------------------------------ 2
    # All four systems, plus the per-expert forecasts, computed once.
    base = moe.predict_all_systems(test, nwp_col="raw_nwp_d1",
                                   regime_probs=np.ones((n, 7)) / 7.0)
    expert_preds = base["expert_breakdown"]          # (n, 7) one column per regime expert
    onehot_true = np.zeros((n, 7))
    onehot_true[np.arange(n), true_regime - 1] = 1.0
    uniform = np.ones((n, 7)) / 7.0

    def blend(weights):
        return np.maximum(0.0, np.sum(weights * expert_preds, axis=1))

    systems = {
        "raw_nwp": base["raw_nwp"],
        "global_qm": base["global_qm"],
        "global_lgb": base["global_lgb"],
        "mo_e_oracle_regime": blend(onehot_true),
        "mo_e_classifier": blend(p_clf),
        "mo_e_uniform_weights": blend(uniform),
    }
    overall = {}
    for name, pred in systems.items():
        d = dich(pred, y_max)
        overall[name] = {
            "rmse": compute_continuous_metrics(pred, y)["rmse"],
            "bias": compute_continuous_metrics(pred, y)["bias"],
            "csi": d["csi"], "ets": d["ets"], "pod": d["pod"], "far": d["far"],
            "hits": d["hits"], "false_alarms": d["false_alarms"], "misses": d["misses"],
        }

    # ---------------------------------------------------------- 2b, 3
    # Simulate a regime classifier of a given accuracy: flip the true label with
    # probability (1 - acc), then hand the MoE a smoothed distribution around it
    # (this is how a real, imperfect classifier presents itself downstream).
    def simulate_classifier(acc, smoothing=0.02):
        w = np.zeros((n, 7))
        correct = rng.rand(n) < acc
        chosen = np.where(correct, true_regime, rng.randint(1, 8, size=n))
        w[np.arange(n), chosen - 1] = 1.0 - smoothing
        w += smoothing / 7.0
        return normalise(w)

    sweep_acc = [1.00, 0.97, 0.95, 0.90, 0.85, 0.80, 0.75, 0.65, 0.50]
    sweep = []
    for acc in sweep_acc:
        pred = blend(simulate_classifier(acc))
        sweep.append({
            "regime_classifier_accuracy": acc,
            "mo_e_csi": csi(pred, y_max),
            "mo_e_rmse": compute_continuous_metrics(pred, y)["rmse"],
            "global_lgb_csi": overall["global_lgb"]["csi"],
            "beats_global_lgb": bool(csi(pred, y_max) > overall["global_lgb"]["csi"]),
        })

    # ------------------------------------------------------------------ 5
    def block_ci(pred_a, pred_b, obs_use, metric, groups, iters=BOOT):
        diffs = np.empty(iters)
        for i in range(iters):
            idx = day_block_indices(groups, rng)
            diffs[i] = metric(pred_a[idx], obs_use[idx]) - metric(pred_b[idx], obs_use[idx])
        return {
            "delta": round(float(metric(pred_a, obs_use) - metric(pred_b, obs_use)), 4),
            "ci95_low": round(float(np.percentile(diffs, 2.5)), 4),
            "ci95_high": round(float(np.percentile(diffs, 97.5)), 4),
            "p_positive": round(float((diffs > 0).mean()), 3),
        }

    def rmse_metric(f, o):
        return compute_continuous_metrics(f, o)["rmse"]

    def csi_metric(f, o):
        return csi(f, o)

    full_groups = day_groups(days)
    comparisons = {
        "mo_e_classifier_minus_raw_nwp_heavy_csi": block_ci(
            systems["mo_e_classifier"], systems["raw_nwp"], y_max, csi_metric, full_groups),
        "mo_e_classifier_minus_global_lgb_heavy_csi": block_ci(
            systems["mo_e_classifier"], systems["global_lgb"], y_max, csi_metric, full_groups),
        "mo_e_classifier_minus_global_lgb_rmse": block_ci(
            systems["mo_e_classifier"], systems["global_lgb"], y, rmse_metric, full_groups),
        "mo_e_classifier_minus_uniform_weights_heavy_csi": block_ci(
            systems["mo_e_classifier"], systems["mo_e_uniform_weights"], y_max, csi_metric,
            full_groups),
    }

    # ------------------------------------------------------------------ 4
    by_regime = {}
    for r in range(1, 8):
        m = true_regime == r
        if m.sum() == 0:
            continue
        events = int((y_max[m] >= HEAVY).sum())
        entry = {
            "name": clf.REGIME_NAMES[r],
            "district_days": int(m.sum()),
            "heavy_events": events,
            "very_heavy_events": int((y_max[m] >= VERY_HEAVY).sum()),
            "scoreable": events >= 10,
        }
        for name, pred in [("raw_nwp", systems["raw_nwp"]),
                           ("global_lgb", systems["global_lgb"]),
                           ("mo_e", systems["mo_e_classifier"])]:
            entry[name] = {k: dich(pred[m], y_max[m])[k]
                           for k in ("pod", "far", "csi", "ets")}
        entry["mo_e_minus_raw_nwp_csi"] = block_ci(
            systems["mo_e_classifier"][m], systems["raw_nwp"][m], y_max[m], csi_metric,
            day_groups(days[m]), iters=300)
        by_regime[clf.REGIME_NAMES[r]] = entry

    # ------------------------------------------------------------------ write
    result = {
        "provenance": "SYNTHETIC_DATASET",
        "warning": ("All numbers are computed on the repository's seeded synthetic "
                    "dataset. They measure internal consistency of the pipeline, "
                    "not skill against real NWP or real IMD observations."),
        "test_years": TEST_YEARS,
        "district_days": n,
        "calendar_days": int(test["date"].nunique()),
        "heavy_events_total": int((y_max >= HEAVY).sum()),
        "heavy_event_rate": round(float((y_max >= HEAVY).mean()), 4),
        "regime_classifier": {"accuracy": round(accuracy, 4), "per_class": per_class},
        "overall_heavy_skill": overall,
        "classifier_accuracy_sweep": sweep,
        "paired_comparisons_block_bootstrap": comparisons,
        "heavy_skill_by_regime": by_regime,
    }
    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # ------------------------------------------------------------------ plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        accs = [s["regime_classifier_accuracy"] for s in sweep]
        moe_csi = [s["mo_e_csi"] for s in sweep]
        lgb_csi = sweep[0]["global_lgb_csi"]

        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
        ax.plot(accs, moe_csi, "o-", color="#2563eb", lw=2,
                label="Regime-aware MoE (heavy-rain CSI)")
        ax.axhline(lgb_csi, color="#dc2626", ls="--", lw=1.6,
                   label=f"Regime-agnostic LightGBM (CSI {lgb_csi:.3f})")
        ax.axhline(overall["raw_nwp"]["csi"], color="#64748b", ls=":", lw=1.6,
                   label=f"Raw NWP (CSI {overall['raw_nwp']['csi']:.3f})")
        ax.set_xlabel("Regime-classifier accuracy")
        ax.set_ylabel("CSI, rainfall ≥ 64.5 mm/day")
        ax.set_title("When does regime conditioning stop paying?\n"
                     "Synthetic held-out years 2022–2023", fontsize=11)
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        fig.savefig(PLOT_OUT)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"[warn] plot skipped: {exc}")

    # ------------------------------------------------------------------ print
    print(f"\nHeld-out test: {n} district-days over {test['date'].nunique()} days "
          f"({TEST_YEARS[0]}–{TEST_YEARS[1]})")
    print(f"Heavy (>= {HEAVY} mm) events: {int((y_max >= HEAVY).sum())} "
          f"({100 * (y_max >= HEAVY).mean():.2f}% of district-days)")

    print(f"\n1. Regime classifier accuracy on held-out years: {100 * accuracy:.1f}%")
    print("   (synthetic regimes are drawn from the same latent parameters that")
    print("    generate the predictors, so this accuracy is an artefact of the")
    print("    generator, not evidence about real regime classification)")

    print("\n2. Where the skill comes from")
    print(f"   {'system':<28} {'RMSE':>7} {'CSI':>7} {'ETS':>7} {'POD':>7} {'FAR':>7}")
    for name in ["raw_nwp", "global_qm", "global_lgb", "mo_e_uniform_weights",
                 "mo_e_classifier", "mo_e_oracle_regime"]:
        s = overall[name]
        print(f"   {name:<28} {s['rmse']:7.3f} {s['csi']:7.4f} {s['ets']:7.4f} "
              f"{s['pod']:7.4f} {s['far']:7.4f}")

    print("\n3. Regime-classifier accuracy sweep (MoE vs global LightGBM)")
    print(f"   {'classifier acc':>15} {'MoE CSI':>9} {'beats global LGB':>18}")
    for s in sweep:
        print(f"   {s['regime_classifier_accuracy']:>15.2f} {s['mo_e_csi']:>9.4f} "
              f"{str(s['beats_global_lgb']):>18}")

    print("\n4. Heavy-rain skill by regime (held-out years)")
    print(f"   {'regime':<24} {'events':>7} {'raw CSI':>8} {'LGB CSI':>8} {'MoE CSI':>8} "
          f"{'MoE-raw CI95':>22}")
    for name, e in by_regime.items():
        ci = e["mo_e_minus_raw_nwp_csi"]
        flag = "" if e["scoreable"] else "  <- too few events to score"
        print(f"   {name:<24} {e['heavy_events']:>7} {e['raw_nwp']['csi']:>8.3f} "
              f"{e['global_lgb']['csi']:>8.3f} {e['mo_e']['csi']:>8.3f} "
              f"{'[' + format(ci['ci95_low'], '+.3f') + ', ' + format(ci['ci95_high'], '+.3f') + ']':>22}{flag}")

    print("\n5. Paired comparisons, day-block bootstrap (respects day-to-day correlation)")
    for name, c in comparisons.items():
        print(f"   {name}")
        print(f"      delta={c['delta']:+.4f}  95% CI [{c['ci95_low']:+.4f}, "
              f"{c['ci95_high']:+.4f}]  P(delta>0)={c['p_positive']}")

    print(f"\nWrote {METRICS_OUT} and {PLOT_OUT}")


if __name__ == "__main__":
    main()
