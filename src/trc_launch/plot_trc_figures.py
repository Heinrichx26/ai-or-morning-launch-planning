from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "results" / "trc_smoke" / "batch20_weather_full360_fullquality"
RISK_RESULT_DIR = ROOT / "results" / "trc_smoke" / "batch21_risk_hardening"
SCALING_RESULT_DIR = ROOT / "results" / "trc_smoke" / "batch22_scenario_scaling"
FIG_DIR = ROOT / "article" / "trc_elsarticle" / "figures"

METHOD_LABELS = {
    "rccc_audited": "RCCC",
    "adaptive_certificate_regret_dense": "RCCC",
    "regret_portfolio_full": "Full regret",
    "saa_extensive": "SAA",
    "cvar_extensive": "CVaR",
    "minimax_extensive": "Minimax",
    "mean_cvar95_extensive": "Mean-CVaR95",
    "active_scenario_saa": "Active-scenario SAA",
}

METHOD_ORDER = [
    "rccc_audited",
    "regret_portfolio_full",
    "saa_extensive",
    "cvar_extensive",
    "minimax_extensive",
    "mean_cvar95_extensive",
    "active_scenario_saa",
]

DISPLAY_LABELS = {
    "rccc_audited": "RCCC",
    "adaptive_certificate_regret_dense": "RCCC",
    "regret_portfolio_full": "Full\nregret",
    "saa_extensive": "SAA",
    "cvar_extensive": "CVaR",
    "minimax_extensive": "Minimax",
    "mean_cvar95_extensive": "Mean\nCVaR95",
    "active_scenario_saa": "Active\nSAA",
}

COLORS = {
    "rccc_audited": "#0B6E69",
    "adaptive_certificate_regret_dense": "#0B6E69",
    "regret_portfolio_full": "#4A4A4A",
    "saa_extensive": "#9B5DE5",
    "cvar_extensive": "#F15BB5",
    "minimax_extensive": "#F77F00",
    "mean_cvar95_extensive": "#577590",
    "active_scenario_saa": "#D62828",
}

RISK_VARIANT_LABELS = {
    "stress_s60": "60 scenarios",
    "weather_mild": "Mild weather",
    "weather_severe": "Severe weather",
    "weight_high": "High weights",
    "weight_low": "Low weights",
}

RISK_VARIANT_ORDER = ["stress_s60", "weather_mild", "weather_severe", "weight_high", "weight_low"]
RISK_COMPARATOR_ORDER = ["saa_extensive", "cvar_extensive", "minimax_extensive", "active_scenario_saa"]


def _setup() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULT_DIR / "batch20_metrics_with_rccc_audited.csv"
    summary_path = RESULT_DIR / "batch20_summary_with_rccc_audited.csv"
    region_path = RESULT_DIR / "batch20_region_regret_with_rccc_audited.csv"
    metrics = pd.read_csv(metrics_path)
    summary = pd.read_csv(summary_path)
    region = pd.read_csv(region_path)
    return metrics, summary, region


def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _contrast_text_color(image, value: float, threshold: float = 0.50) -> str:
    rgba = image.cmap(image.norm(value))
    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "white" if luminance < threshold else "black"


def _box(ax, x, y, w, h, text, fc="#F7F7F7", ec="#2B2B2B", size=9):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.0,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, linespacing=1.2)


def _arrow(ax, start, end, color="#333333"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.0, color=color))


def problem_schematic() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _box(ax, 0.04, 0.62, 0.20, 0.22, "Previous-day\naircraft tails", "#EAF4F4")
    _box(ax, 0.04, 0.18, 0.20, 0.22, "First-wave\nflights", "#EAF4F4")
    _box(ax, 0.33, 0.40, 0.22, 0.24, "Tail-to-flight\nassignment", "#FFF7E6")
    _box(ax, 0.64, 0.62, 0.26, 0.22, "Weather-indexed\nairport-hour capacity", "#E8F1FB")
    _box(ax, 0.64, 0.18, 0.26, 0.22, "Scenario-wise\noracle assignment", "#FDEDEC")
    _box(ax, 0.38, 0.05, 0.25, 0.18, "Regret:\nrealized cost - oracle cost", "#EEF6EC", size=8.5)
    _arrow(ax, (0.24, 0.73), (0.33, 0.54))
    _arrow(ax, (0.24, 0.29), (0.33, 0.49))
    _arrow(ax, (0.55, 0.53), (0.64, 0.70))
    _arrow(ax, (0.77, 0.62), (0.77, 0.40))
    _arrow(ax, (0.64, 0.29), (0.63, 0.14))
    _arrow(ax, (0.50, 0.40), (0.50, 0.23))
    ax.text(0.5, 0.95, "Overnight fleet assignment before weather-disaster launch capacity is known", ha="center", fontsize=10.5)
    _save(fig, "fig1_problem_schematic")


def rccc_flow() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    steps = [
        ("Full candidate graph\nE", 0.04, 0.52, "#F7F7F7"),
        ("Dominance\ncertificate", 0.22, 0.68, "#EAF4F4"),
        ("Exposure\ncertificate", 0.22, 0.36, "#EAF4F4"),
        ("Tail coverage\ncertificate", 0.42, 0.52, "#EAF4F4"),
        ("Certificate closure\nE^C", 0.62, 0.52, "#FFF7E6"),
        ("Reduced regret\noptimization", 0.80, 0.52, "#EEF6EC"),
    ]
    for text, x, y, fc in steps:
        _box(ax, x, y, 0.15, 0.18, text, fc, size=8.2)
    _arrow(ax, (0.19, 0.61), (0.22, 0.76))
    _arrow(ax, (0.19, 0.61), (0.22, 0.45))
    _arrow(ax, (0.37, 0.77), (0.42, 0.63))
    _arrow(ax, (0.37, 0.45), (0.42, 0.58))
    _arrow(ax, (0.57, 0.61), (0.62, 0.61))
    _arrow(ax, (0.77, 0.61), (0.80, 0.61))
    _box(ax, 0.53, 0.12, 0.28, 0.18, "Regret-closure audit:\nexpand binding block-tail neighborhoods", "#FDEDEC", size=8.2)
    _arrow(ax, (0.88, 0.52), (0.77, 0.30), "#8B1E3F")
    _arrow(ax, (0.53, 0.21), (0.20, 0.52), "#8B1E3F")
    ax.text(0.5, 0.95, "Regret-critical certificate closure", ha="center", fontsize=12)
    _save(fig, "fig2_rccc_flow")


def regret_distribution(metrics: pd.DataFrame) -> None:
    data = [metrics[metrics["method"].eq(m)]["max_regret"].dropna().to_numpy() for m in METHOD_ORDER]
    labels = [
        {
            "mean_cvar95_extensive": "Mean-\nCVaR95",
            "active_scenario_saa": "Active-\nscenario SAA",
        }.get(m, METHOD_LABELS[m])
        for m in METHOD_ORDER
    ]
    fig, ax = plt.subplots(figsize=(6.6, 3.48))
    bp = ax.boxplot(
        data,
        patch_artist=True,
        showfliers=False,
        widths=0.55,
        medianprops={"color": "#111111", "linewidth": 1.1},
    )
    for patch, method in zip(bp["boxes"], METHOD_ORDER):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_ylabel("Maximum scenario regret")
    ax.set_title("Maximum-regret distribution")
    ax.grid(axis="y", alpha=0.25)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.12)
    label_y = ymin + (ax.get_ylim()[1] - ymin) * 0.96
    for i, values in enumerate(data, start=1):
        ax.text(
            i,
            label_y,
            f"{np.median(values):.2f}/{np.max(values):.2f}",
            ha="center",
            va="top",
            fontsize=7.8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.5},
        )
    fig.tight_layout(pad=0.25)
    _save(fig, "fig3_regret_distribution")


def retention_runtime(summary: pd.DataFrame) -> None:
    frame = summary[summary["method"].isin(METHOD_ORDER)].copy()
    fig, ax = plt.subplots(figsize=(5.3, 3.25))
    offsets = {
        "adaptive_certificate_regret_dense": (0.008, 0.015, "left"),
        "rccc_audited": (0.008, 0.015, "left"),
        "regret_portfolio_full": (0.008, 0.015, "left"),
        "saa_extensive": (0.012, 0.155, "left"),
        "cvar_extensive": (0.012, -0.040, "left"),
        "mean_cvar95_extensive": (0.008, -0.045, "left"),
        "minimax_extensive": (0.008, 0.020, "left"),
        "active_scenario_saa": (0.012, -0.115, "left"),
    }
    for _, row in frame.iterrows():
        method = row["method"]
        ax.scatter(
            row["mean_candidate_retention"],
            row["mean_solve_seconds"],
            s=80,
            color=COLORS.get(method, "#444444"),
        )
        dx, dy, ha = offsets.get(method, (0.01, 0.02, "left"))
        ax.text(
            row["mean_candidate_retention"] + dx,
            row["mean_solve_seconds"] + dy,
            f"{METHOD_LABELS.get(method, method)} ({row['mean_candidate_retention']:.3f}, {row['mean_solve_seconds']:.2f}s)",
            fontsize=7,
            ha=ha,
            va="center",
        )
    ax.set_xlabel("Mean candidate retention")
    ax.set_ylabel("Mean solve time (s)")
    ax.set_title("Candidate scale and solve time")
    ax.grid(alpha=0.25)
    ax.set_xlim(max(0.0, frame["mean_candidate_retention"].min() - 0.02), 1.30)
    ax.set_ylim(0.04, max(1.90, frame["mean_solve_seconds"].max() + 0.08))
    ax.set_xticks(np.arange(0.65, 1.01, 0.05))
    fig.tight_layout(pad=0.25)
    _save(fig, "fig4_retention_runtime")


def region_heatmap(region: pd.DataFrame) -> None:
    cols = [m for m in METHOD_ORDER if m in region.columns]
    mat = region.set_index("region")[cols]
    region_labels = {
        "bay_area": "Bay Area",
        "chicago": "Chicago",
        "dallas": "Dallas",
        "dc_baltimore": "DC--Baltimore",
        "los_angeles": "Los Angeles",
        "nyc": "New York",
        "south_florida": "South Florida",
    }
    fig, ax = plt.subplots(figsize=(7.3, 3.7))
    im = ax.imshow(mat.to_numpy(float), aspect="auto", cmap="YlGnBu_r")
    ax.set_xticks(range(len(cols)), [DISPLAY_LABELS.get(m, METHOD_LABELS[m]) for m in cols], rotation=0, ha="center")
    ax.set_yticks(range(len(mat.index)), [region_labels.get(v, v) for v in mat.index])
    ax.set_title("Mean maximum regret by multi-airport region")
    ax.set_xlim(-0.55, len(cols) - 0.35)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = float(mat.iloc[i, j])
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color=_contrast_text_color(im, value),
            )
    fig.colorbar(im, ax=ax, label="Mean maximum regret")
    fig.tight_layout(pad=0.25)
    _save(fig, "fig5_region_regret")


def risk_hardening_figure() -> None:
    stats_path = RISK_RESULT_DIR / "risk_hardening_statistical_audit.csv"
    summary_path = RISK_RESULT_DIR / "risk_hardening_summary.csv"
    if not stats_path.exists() or not summary_path.exists():
        return
    stats = pd.read_csv(stats_path)
    summary = pd.read_csv(summary_path)
    stats["variant"] = pd.Categorical(stats["variant"], RISK_VARIANT_ORDER, ordered=True)
    stats["comparison_method"] = pd.Categorical(stats["comparison_method"], RISK_COMPARATOR_ORDER, ordered=True)
    stats = stats.sort_values(["variant", "comparison_method"])

    mat = (
        stats.pivot(index="variant", columns="comparison_method", values="mean_regret_gap_vs_rccc")
        .reindex(RISK_VARIANT_ORDER)
        .reindex(columns=RISK_COMPARATOR_ORDER)
    )
    rccc = (
        summary[summary["method"].eq("rccc_audited")]
        .set_index("variant")
        .reindex(RISK_VARIANT_ORDER)
    )

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.2, 3.55), gridspec_kw={"width_ratios": [1.42, 0.78]})
    im = ax0.imshow(mat.to_numpy(float), aspect="auto", cmap="YlOrRd")
    ax0.set_xticks(range(len(RISK_COMPARATOR_ORDER)), [DISPLAY_LABELS.get(m, METHOD_LABELS[m]) for m in RISK_COMPARATOR_ORDER], rotation=0, ha="center")
    ax0.set_yticks(range(len(RISK_VARIANT_ORDER)), [RISK_VARIANT_LABELS[v] for v in RISK_VARIANT_ORDER])
    ax0.set_title("Regret gap versus RCCC")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = float(mat.iloc[i, j])
            ax0.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color=_contrast_text_color(im, value, threshold=0.58),
            )
    fig.colorbar(im, ax=ax0, fraction=0.038, pad=0.025)

    y = np.arange(len(RISK_VARIANT_ORDER))
    ax1.barh(y, rccc["mean_candidate_retention"], color="#0B6E69", alpha=0.78)
    ax1.set_xlim(0, 1.05)
    ax1.set_yticks(y, [""] * len(y))
    ax1.invert_yaxis()
    ax1.set_xlabel("Candidate retention")
    ax1.set_title("RCCC graph")
    for idx, variant in enumerate(RISK_VARIANT_ORDER):
        matches = int(rccc.loc[variant, "full_regret_matches"])
        cases = int(rccc.loc[variant, "cases"])
        value = float(rccc.loc[variant, "mean_candidate_retention"])
        ax1.text(value / 2, idx, f"{value:.3f}", va="center", ha="center", fontsize=7, color="white")
        ax1.text(min(value + 0.03, 0.98), idx, f"{matches}/{cases}", va="center", fontsize=7)
    ax1.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    _save(fig, "fig6_risk_hardening")


def scenario_scaling_figure() -> None:
    summary_path = SCALING_RESULT_DIR / "scenario_scaling_summary.csv"
    if not summary_path.exists():
        return
    summary = pd.read_csv(summary_path).sort_values("scenario_count")
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1.12, 0.88]})
    ax0.plot(summary["scenario_count"], summary["mean_full_seconds"], marker="o", color="#4A4A4A", label="Full regret")
    ax0.plot(summary["scenario_count"], summary["mean_rccc_seconds"], marker="o", color="#0B6E69", label="RCCC")
    for _, row in summary.iterrows():
        ax0.text(row["scenario_count"], row["mean_full_seconds"] + 0.55, f"{row['mean_full_seconds']:.1f}", ha="center", fontsize=7, color="#4A4A4A")
        ax0.text(row["scenario_count"], row["mean_rccc_seconds"] - 0.90, f"{row['mean_rccc_seconds']:.1f}", ha="center", fontsize=7, color="#0B6E69")
    ax0.set_xlabel("Scenario count")
    ax0.set_ylabel("Mean solve time (s)")
    ax0.set_title("Scenario scaling")
    ax0.grid(alpha=0.25)
    ax0.legend(frameon=False, fontsize=8)
    ax0.set_ylim(2.9, max(summary["mean_full_seconds"].max(), summary["mean_rccc_seconds"].max()) + 1.4)

    x = np.arange(len(summary))
    ax1.bar(x, summary["mean_rccc_retention"], color="#0B6E69", alpha=0.78)
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(x, summary["scenario_count"].astype(str))
    ax1.set_xlabel("Scenario count")
    ax1.set_ylabel("Candidate retention")
    ax1.set_title("RCCC graph")
    for idx, row in summary.reset_index(drop=True).iterrows():
        ax1.text(
            idx,
            float(row["mean_rccc_retention"]) / 2,
            f"{float(row['mean_rccc_retention']):.3f}",
            ha="center",
            va="center",
            fontsize=7,
            color="white",
        )
        ax1.text(
            idx,
            float(row["mean_rccc_retention"]) + 0.03,
            f"{int(row['rccc_regret_matches'])}/{int(row['cases'])}",
            ha="center",
            fontsize=7,
        )
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, "fig7_scenario_scaling")


def write_tables(summary: pd.DataFrame, metrics: pd.DataFrame) -> None:
    out_dir = ROOT / "article" / "trc_elsarticle" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    main_summary = summary[summary["method"].isin(METHOD_ORDER)].copy()
    main_summary.to_csv(out_dir / "weather360_summary_for_manuscript.csv", index=False)
    main_summary.to_csv(out_dir / "weather48_summary_for_manuscript.csv", index=False)
    audit_path = RESULT_DIR / "batch20_rccc_audit_paths.csv"
    if audit_path.exists():
        pd.read_csv(audit_path).to_csv(out_dir / "rccc_audit_paths.csv", index=False)
    rccc = metrics[metrics["method"].eq("adaptive_certificate_regret_dense")].copy()
    width = (
        rccc.groupby("width_label")
        .agg(
            cases=("instance_key", "count"),
            mean_candidate_retention=("candidate_retention", "mean"),
            mean_max_regret=("max_regret", "mean"),
        )
        .reset_index()
        .sort_values("width_label")
    )
    width.to_csv(out_dir / "rccc_width_summary.csv", index=False)
    risk_summary_path = RISK_RESULT_DIR / "risk_hardening_summary.csv"
    risk_stats_path = RISK_RESULT_DIR / "risk_hardening_statistical_audit.csv"
    risk_audit_path = RISK_RESULT_DIR / "risk_hardening_rccc_audit_paths.csv"
    if risk_summary_path.exists():
        pd.read_csv(risk_summary_path).to_csv(out_dir / "risk_hardening_summary.csv", index=False)
    if risk_stats_path.exists():
        pd.read_csv(risk_stats_path).to_csv(out_dir / "risk_hardening_statistical_audit.csv", index=False)
    if risk_audit_path.exists():
        pd.read_csv(risk_audit_path).to_csv(out_dir / "risk_hardening_rccc_audit_paths.csv", index=False)
    scaling_summary_path = SCALING_RESULT_DIR / "scenario_scaling_summary.csv"
    scaling_audit_path = SCALING_RESULT_DIR / "scenario_scaling_rccc_audit_paths.csv"
    if scaling_summary_path.exists():
        pd.read_csv(scaling_summary_path).to_csv(out_dir / "scenario_scaling_summary.csv", index=False)
    if scaling_audit_path.exists():
        pd.read_csv(scaling_audit_path).to_csv(out_dir / "scenario_scaling_rccc_audit_paths.csv", index=False)


def main() -> None:
    metrics, summary, region = _setup()
    regret_distribution(metrics)
    retention_runtime(summary)
    region_heatmap(region)
    risk_hardening_figure()
    scenario_scaling_figure()
    write_tables(summary, metrics)
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
