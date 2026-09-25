"""
Printable verification report.

Every statement in this PDF is derived from the metrics JSON - including the
significance language. If an interval does not support a claim, the report says
so in the same place it would have said "significant". Thin strata are printed
with their event count and marked "not reportable" instead of being scored.

Also produced: the three figures used by the console's verification view.
"""

import os
import json
import logging
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, PageBreak,
)

logger = logging.getLogger(__name__)

INK = colors.HexColor("#12181f")
MUTED = colors.HexColor("#5b6b7c")
RULE = colors.HexColor("#d5dbe1")
ACCENT = colors.HexColor("#1b4f8a")
WARN_BG = colors.HexColor("#fdf3d8")
BAD = colors.HexColor("#b3261e")
GOOD = colors.HexColor("#1a6b3c")


class VerificationReportGenerator:
    def __init__(self, metrics_dir: str = "artifacts/metrics",
                 plots_dir: str = "artifacts/plots",
                 reports_dir: str = "artifacts/reports"):
        self.metrics_dir = metrics_dir
        self.plots_dir = plots_dir
        self.reports_dir = reports_dir
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

    # ------------------------------------------------------------------ figures
    def generate_figures(self, summary: dict) -> dict:
        paths = {}
        prob = summary.get("probabilistic_verification", {}).get("heavy", {})
        rel = prob.get("reliability_diagram", {})
        roc = prob.get("roc", {})

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0), dpi=200)
        if rel:
            ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.7, label="perfect")
            ax1.plot(rel["mean_forecast_probs"], rel["observed_frequencies"], "o-",
                     color="#1b4f8a", lw=1.8, ms=4, label="forecast")
            ax1.set_title(f"Reliability, ≥{prob.get('threshold_mm', 64.5)} mm\n"
                          f"Brier {prob.get('brier_score', float('nan')):.4f}", fontsize=10)
            ax1.set_xlabel("forecast probability", fontsize=9)
            ax1.set_ylabel("observed frequency", fontsize=9)
            ax1.legend(fontsize=8)
            ax1.grid(alpha=0.3)
        if roc:
            ax2.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.7, label="no skill")
            ax2.plot(roc["fpr"], roc["tpr"], color="#1a6b3c", lw=1.8,
                     label=f"forecast (AUC {roc.get('auc', float('nan')):.3f})")
            ax2.set_title("ROC, heavy rainfall", fontsize=10)
            ax2.set_xlabel("false positive rate", fontsize=9)
            ax2.set_ylabel("hit rate", fontsize=9)
            ax2.legend(fontsize=8)
            ax2.grid(alpha=0.3)
        fig.tight_layout()
        p1 = os.path.join(self.plots_dir, "report_prob_curves.png")
        fig.savefig(p1)
        plt.close(fig)
        paths["prob_curves"] = p1

        # regime CSI with event counts; thin strata are greyed and labelled
        reg = summary.get("regime_stratified", {})
        if reg:
            names = list(reg.keys())
            raw = [reg[n]["systems"]["raw_nwp"]["csi"] for n in names]
            base = [reg[n]["systems"]["global_lgb"]["csi"] for n in names]
            corr = [reg[n]["systems"]["monsooniq"]["csi"] for n in names]
            events = [reg[n]["events"] for n in names]
            x = np.arange(len(names))
            w = 0.27
            fig, ax = plt.subplots(figsize=(10, 4.0), dpi=200)
            ax.bar(x - w, raw, w, label="raw NWP", color="#9aa7b4")
            ax.bar(x, base, w, label="regime-agnostic baseline", color="#c98a2b")
            ax.bar(x + w, corr, w, label="regime-aware", color="#1b4f8a")
            for xi, (ev, n) in enumerate(zip(events, names)):
                if ev < 10:
                    ax.annotate("n<10\nnot scored", (xi, 0.02), ha="center", fontsize=7,
                                color="#b3261e")
                else:
                    ax.annotate(f"{ev} events", (xi, max(corr[xi], raw[xi]) + 0.02),
                                ha="center", fontsize=7, color="#12181f")
            ax.set_xticks(x)
            ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8)
            ax.set_ylabel("CSI, ≥64.5 mm", fontsize=9)
            ax.set_title("Heavy-rainfall skill by regime (held-out period)", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3, axis="y")
            fig.tight_layout()
            p2 = os.path.join(self.plots_dir, "report_regime_csi.png")
            fig.savefig(p2)
            plt.close(fig)
            paths["regime_csi"] = p2

        # lead-time degradation + FSS vs scale
        lead = summary.get("lead_time_breakdown", {})
        fss = summary.get("grid_fss", {})
        if lead.get("available") or fss.get("available"):
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=200)
            ax1, ax2 = axes
            if lead.get("available"):
                days = sorted(lead["leads"].keys())
                for sys_name, label, colour in (("raw_nwp", "raw NWP", "#9aa7b4"),
                                                ("global_lgb", "regime-agnostic", "#c98a2b"),
                                                ("monsooniq", "regime-aware", "#1b4f8a")):
                    ys = [lead["leads"][d]["systems"][sys_name]["rmse"] for d in days]
                    ax1.plot(range(1, len(days) + 1), ys, "o-", color=colour, label=label, lw=1.6)
                ax1.set_xticks(range(1, len(days) + 1))
                ax1.set_xticklabels([d.replace("day_", "D") for d in days])
                ax1.set_ylabel("RMSE (mm/day)", fontsize=9)
                ax1.set_title("Skill vs forecast lead time", fontsize=10)
                ax1.legend(fontsize=8)
                ax1.grid(alpha=0.3)
            else:
                ax1.axis("off")
            if fss.get("available"):
                # Solid lines: the systems that beat raw. Dashed: the district-transfer
                # negative control, kept in the figure on purpose.
                wins = []
                for thr, colour in (("15.6", "#1a6b3c"), ("64.5", "#b3261e")):
                    block = fss["scores"].get(thr, {})
                    wins = sorted(int(k[1:]) for k in block.keys() if k.startswith("w"))
                    if not wins:
                        continue
                    for sys_name, style, label in (
                            ("raw_nwp", "--", f"raw NWP, ≥{thr} mm"),
                            ("regime_aware_grid_native", "-", f"regime-aware grid-native, ≥{thr} mm"),
                            ("regime_aware_district_transfer", ":",
                             f"district→grid transfer (negative control), ≥{thr} mm")):
                        if sys_name not in block.get(f"w{wins[0]}", {}):
                            continue
                        ys = [block[f"w{w}"].get(sys_name, {}).get("fss") for w in wins]
                        if any(y is None for y in ys):
                            continue
                        ax2.plot(range(1, len(wins) + 1), ys, marker="o", color=colour,
                                 linestyle=style, lw=1.5, markersize=4, label=label)
                ax2.set_xticks(range(1, len(wins) + 1) if wins else [])
                ax2.set_xticklabels([f"{w}×{w}" for w in wins])
                ax2.set_ylim(0, 1)
                ax2.set_ylabel("Fractions Skill Score", fontsize=9)
                ax2.set_title("Spatial skill vs neighbourhood size", fontsize=10)
                ax2.legend(fontsize=6.5)
                ax2.grid(alpha=0.3)
            else:
                ax2.axis("off")
            fig.tight_layout()
            p3 = os.path.join(self.plots_dir, "report_lead_fss.png")
            fig.savefig(p3)
            plt.close(fig)
            paths["lead_fss"] = p3
        return paths

    # ---------------------------------------------------------------------- pdf
    def generate_pdf(self, pdf_filename: str = "MonsoonIQ_Verification_Report.pdf") -> str:
        summary_file = os.path.join(self.metrics_dir, "verification_summary.json")
        if not os.path.exists(summary_file):
            logger.error("verification_summary.json missing; run the evaluator first.")
            return ""
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)
        figs = self.generate_figures(summary)

        pdf_path = os.path.join(self.reports_dir, pdf_filename)
        doc = SimpleDocTemplate(pdf_path, pagesize=A4, leftMargin=16 * mm,
                                rightMargin=16 * mm, topMargin=14 * mm, bottomMargin=14 * mm,
                                title="MonsoonIQ verification report")

        ss = getSampleStyleSheet()
        h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=16, leading=19, textColor=INK,
                            spaceAfter=2)
        sub = ParagraphStyle("sub", parent=ss["Normal"], fontSize=8.5, leading=11, textColor=MUTED)
        h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11, leading=13, textColor=INK,
                            spaceBefore=10, spaceAfter=4)
        body = ParagraphStyle("body", parent=ss["Normal"], fontSize=8.5, leading=11.5,
                              textColor=colors.HexColor("#28323c"))
        small = ParagraphStyle("small", parent=ss["Normal"], fontSize=7.5, leading=10,
                               textColor=MUTED)

        period = summary.get("verification_period", {})
        prov = summary.get("provenance_detail", {})
        elements = []

        elements.append(Paragraph("MonsoonIQ — Verification Report", h1))
        elements.append(Paragraph(
            f"Regime-aware post-processing of NWP rainfall · held-out period "
            f"{period.get('test_years')} · {period.get('calendar_days')} days · "
            f"{period.get('district_days')} district-days · generated "
            f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC", sub))
        elements.append(Spacer(1, 5))

        badge = (f"<b>DATA PROVENANCE: {summary.get('data_provenance')}</b> — "
                 f"{prov.get('note', '')} All skill figures below are measured against this "
                 f"archive and are <b>not</b> evidence of operational skill.")
        t_badge = Table([[Paragraph(badge, small)]], colWidths=[178 * mm])
        t_badge.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), WARN_BG),
                                     ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e6d08a")),
                                     ("LEFTPADDING", (0, 0), (-1, -1), 6),
                                     ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                                     ("TOPPADDING", (0, 0), (-1, -1), 5),
                                     ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
        elements.append(t_badge)

        # ---- 1 scorecard
        elements.append(Paragraph("1 · Scorecard (held-out period)", h2))
        cont = summary["continuous_metrics"]
        th = summary["threshold_metrics"]["64.5"]
        pairs = summary.get("paired_comparisons", {})
        vs_raw = pairs.get("heavy_csi_vs_raw_nwp", {})
        vs_lgb = pairs.get("heavy_csi_vs_global_lgb", {})

        def sig(flag):
            return "significant" if flag else "not significant"

        rows = [["Quantity", "Raw NWP", "Regime-agnostic", "Regime-aware", "Change vs raw", "Change vs baseline"]]
        rows.append(["RMSE (mm/day)", f"{cont['raw_nwp']['rmse']:.3f}", f"{cont['global_lgb']['rmse']:.3f}",
                     f"{cont['monsooniq']['rmse']:.3f}",
                     f"{(cont['monsooniq']['rmse'] - cont['raw_nwp']['rmse']):+.3f}",
                     f"{(cont['monsooniq']['rmse'] - cont['global_lgb']['rmse']):+.3f}"])
        rows.append(["Heavy-rain CSI (≥64.5 mm)", f"{th['raw_nwp']['csi']:.3f}",
                     f"{th['global_lgb']['csi']:.3f}", f"{th['monsooniq']['csi']:.3f}",
                     f"{vs_raw.get('delta', float('nan')):+.4f} ({sig(vs_raw.get('significant_95'))})",
                     f"{vs_lgb.get('delta', float('nan')):+.4f} ({sig(vs_lgb.get('significant_95'))})"])
        rows.append(["Heavy-rain ETS", f"{th['raw_nwp']['ets']:.3f}", f"{th['global_lgb']['ets']:.3f}",
                     f"{th['monsooniq']['ets']:.3f}",
                     f"{th['monsooniq']['ets'] - th['raw_nwp']['ets']:+.3f}",
                     f"{th['monsooniq']['ets'] - th['global_lgb']['ets']:+.3f}"])
        rows.append(["Very heavy CSI (≥115.6 mm)",
                     f"{summary['threshold_metrics']['115.6']['raw_nwp']['csi']:.3f}",
                     f"{summary['threshold_metrics']['115.6']['global_lgb']['csi']:.3f}",
                     f"{summary['threshold_metrics']['115.6']['monsooniq']['csi']:.3f}",
                     f"{summary['threshold_metrics']['115.6']['monsooniq']['csi'] - summary['threshold_metrics']['115.6']['raw_nwp']['csi']:+.3f}",
                     f"{summary['threshold_metrics']['115.6']['monsooniq']['csi'] - summary['threshold_metrics']['115.6']['global_lgb']['csi']:+.3f}"])

        t = Table(rows, colWidths=[46 * mm, 22 * mm, 30 * mm, 26 * mm, 32 * mm, 32 * mm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), INK), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5), ("GRID", (0, 0), (-1, -1), 0.4, RULE),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fa")]),
        ]))
        elements.append(t)
        elements.append(Paragraph(
            "The regime-agnostic column is a gradient-boosted model fitted on the same "
            "predictors without regime information. It is the baseline that decides whether "
            "regime conditioning earns its complexity; the raw model alone does not.", small))

        # ---- 2 heavy events + regime stratification
        elements.append(Paragraph("2 · Heavy and very heavy rainfall, by regime", h2))
        reg = summary.get("regime_stratified", {})
        if reg:
            rows = [["Regime", "Days", "Events\n(≥64.5)", "≥115.6", "Reliability",
                     "Raw CSI", "Regime-agnostic CSI", "Regime-aware CSI",
                     "Δ vs raw [95% CI]"]]
            for name, e in reg.items():
                imp = e.get("improvement") or {}
                if imp.get("suppressed"):
                    delta = "not reportable"
                else:
                    delta = (f"{imp.get('delta', float('nan')):+.3f} "
                             f"[{imp['ci95'][0]:+.3f}, {imp['ci95'][1]:+.3f}]")
                rows.append([
                    name, f"{e['sample_count']}", f"{e['events']}",
                    f"{e['event_counts']['very_heavy']}", e["reliability"],
                    f"{e['systems']['raw_nwp']['csi']:.3f}",
                    f"{e['systems']['global_lgb']['csi']:.3f}",
                    f"{e['systems']['monsooniq']['csi']:.3f}", delta,
                ])
            t2 = Table(rows, colWidths=[34 * mm, 12 * mm, 14 * mm, 12 * mm, 18 * mm,
                                        15 * mm, 24 * mm, 22 * mm, 27 * mm])
            t2.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#28323c")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 7), ("GRID", (0, 0), (-1, -1), 0.4, RULE),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fa")]),
                ("TEXTCOLOR", (4, 1), (4, -1), MUTED),
            ]))
            elements.append(t2)
            elements.append(Paragraph(
                "Strata with fewer than 10 events are marked <b>insufficient</b>: their CSI is "
                "printed for completeness only and is not a scored result. Intervals are "
                "day-block bootstraps — whole calendar days are resampled, because 53 districts "
                "on one day are a single weather system.", small))

        if figs.get("prob_curves"):
            elements.append(PageBreak())
            elements.append(Paragraph("3 · Reliability, ROC and regime breakdown", h2))
            elements.append(Image(figs["prob_curves"], width=178 * mm, height=71 * mm))
            if figs.get("regime_csi"):
                elements.append(Image(figs["regime_csi"], width=178 * mm, height=71 * mm))

        # ---- 4 regime value
        rv = summary.get("regime_value", {})
        if rv.get("available"):
            elements.append(Paragraph("4 · What regime awareness is worth", h2))
            head = rv["headline"]
            cmp_lgb = rv["comparisons"]["moe_classifier_vs_global_lgb"]
            rows = [
                ["Experiment", "Result"],
                ["Correction alone, uniform regime weights (CSI)", f"{rv['systems']['moe_uniform_weights']['csi']:.4f}"],
                ["Correction + classifier regime weights (CSI)", f"{rv['systems']['moe_classifier']['csi']:.4f}"],
                ["Regime-agnostic model, same predictors (CSI)", f"{rv['systems']['global_lgb']['csi']:.4f}"],
                ["Gain attributed to regime awareness",
                 f"{head['regime_conditioning_gain_over_regime_agnostic_csi']:+.4f} "
                 f"[{cmp_lgb['ci95'][0]:+.4f}, {cmp_lgb['ci95'][1]:+.4f}] — "
                 f"{'significant' if head['gain_significant_95'] else 'NOT significant'}"],
                ["Classifier accuracy at which the gain vanishes",
                 f"{head['crossover_accuracy']}"
                 if head.get("crossover_accuracy") is not None else "gain already vanishes"],
            ]
            t4 = Table(rows, colWidths=[95 * mm, 83 * mm])
            t4.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#28323c")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5), ("GRID", (0, 0), (-1, -1), 0.4, RULE),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fa")]),
            ]))
            elements.append(t4)
            elements.append(Paragraph(rv.get("interpretation", ""), small))
            curve = os.path.join(self.plots_dir, "regime_value_curve.png")
            if os.path.exists(curve):
                elements.append(Spacer(1, 4))
                elements.append(Image(curve, width=120 * mm, height=72 * mm))

        if figs.get("lead_fss"):
            elements.append(PageBreak())
            elements.append(Paragraph("5 · Lead time and spatial scale", h2))
            elements.append(Image(figs["lead_fss"], width=178 * mm, height=68 * mm))
            fss = summary.get("grid_fss", {})
            if fss.get("available"):
                elements.append(Paragraph(fss.get("method", ""), small))
                rows = [["Threshold", "Neighbourhood", "Raw NWP", "Grid-native",
                         "District→grid (control)"]]
                for thr, block in fss["scores"].items():
                    if not isinstance(block, dict):
                        continue
                    for key, val in block.items():
                        if not key.startswith("w"):
                            continue
                        raw = val.get("raw_nwp", {}).get("fss")
                        native = val.get("regime_aware_grid_native", {}).get("fss")
                        transfer = val.get("regime_aware_district_transfer", {}).get("fss")
                        fmt = lambda v: "—" if v is None else f"{v:.3f}"
                        rows.append([f"≥{thr} mm", f"{key[1:]}×{key[1:]} cells",
                                     fmt(raw), fmt(native), fmt(transfer)])
                t5 = Table(rows, colWidths=[24 * mm, 32 * mm, 24 * mm, 28 * mm, 34 * mm])
                t5.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#28323c")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTSIZE", (0, 0), (-1, -1), 7.5), ("GRID", (0, 0), (-1, -1), 0.4, RULE),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fa")]),
                ]))
                elements.append(t5)

        # ---- 6 claims + reproducibility
        elements.append(PageBreak())
        elements.append(Paragraph("6 · Claim check", h2))
        rows = [["Claim", "Verdict", "Evidence"]]
        for claim in summary.get("significance_statement", {}).get("claims", []):
            rows.append([claim["claim"], "SUPPORTED" if claim["supported"] else "NOT SUPPORTED",
                         claim["evidence"]])
        t6 = Table(rows, colWidths=[68 * mm, 26 * mm, 84 * mm])
        t6.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#28323c")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5), ("GRID", (0, 0), (-1, -1), 0.4, RULE),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        elements.append(t6)
        elements.append(Spacer(1, 6))
        elements.append(Paragraph(summary.get("significance_statement", {}).get("caveat", ""), small))

        elements.append(Paragraph("7 · Reproducibility", h2))
        elements.append(Paragraph(
            "python src/train.py &nbsp;→&nbsp; python src/evaluate.py &nbsp;→&nbsp; "
            "pytest tests/ &nbsp;→&nbsp; uvicorn src.api.main:app<br/>"
            f"Model manifest: artifacts/models/training_manifest.json "
            f"({', '.join(sorted(summary.get('systems', {}).values()))})<br/>"
            "Every figure and number in this report is regenerated from the raw archive by the "
            "commands above; no value is hand-entered.", small))

        doc.build(elements)
        logger.info("Verification report written to %s", pdf_path)
        return pdf_path


if __name__ == "__main__":
    print(VerificationReportGenerator().generate_pdf())
