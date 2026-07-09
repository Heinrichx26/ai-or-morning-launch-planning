from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
AI_OR_SUMMARY = ROOT / "article" / "trc_elsarticle" / "generated" / "ai_or_main_summary.csv"
AI_OR_METRICS = ROOT / "results" / "trc_smoke" / "batch34_decision_intelligence_weather_full360" / "ai_or_metrics.csv"
AI_OR_STRESS_DIR = ROOT / "results" / "trc_smoke" / "batch34_decision_intelligence_weather_stress"
FIG_DIR = ROOT / "article" / "trc_elsarticle" / "figures"

COLORS = {
    "LG-RCCC audited": "#0B6E69",
    "LG-RCCC pre-audit": "#63A69F",
    "AI proposal only": "#8AB8B2",
    "RCCC audited": "#276FBF",
    "Full regret reference": "#4A4A4A",
    "SAA": "#9B5DE5",
    "CVaR": "#F15BB5",
    "Minimax": "#F77F00",
    "Active-scenario SAA": "#D62828",
}

ORDER = [
    "AI proposal only",
    "LG-RCCC audited",
    "RCCC audited",
    "Full regret reference",
    "SAA",
    "CVaR",
    "Minimax",
    "Active-scenario SAA",
]

METRIC_METHODS = {
    "ml_proposal_only": "AI proposal only",
    "lg_rccc_audited": "LG-RCCC audited",
    "rccc_audited": "RCCC audited",
    "regret_portfolio_full": "Full regret reference",
    "saa_extensive": "SAA",
    "cvar_extensive": "CVaR",
    "minimax_extensive": "Minimax",
    "active_scenario_saa": "Active-scenario SAA",
}

PLOT_ORDER = [
    "LG-RCCC audited",
    "AI proposal only",
    "RCCC audited",
    "Full regret reference",
    "SAA",
    "CVaR",
    "Minimax",
    "Active-scenario SAA",
]

LABELS = {
    "AI proposal only": "AI proposal",
    "LG-RCCC audited": "LG-RCCC",
    "RCCC audited": "RCCC",
    "Full regret reference": "Full regret",
    "SAA": "SAA",
    "CVaR": "CVaR",
    "Minimax": "Minimax",
    "Active-scenario SAA": "Active SAA",
}

DISPLAY_LABELS = {
    "LG-RCCC audited": "LG-RCCC",
    "AI proposal only": "AI\nproposal",
    "RCCC audited": "RCCC",
    "Full regret reference": "Full\nregret",
    "SAA": "SAA",
    "CVaR": "CVaR",
    "Minimax": "Minimax",
    "Active-scenario SAA": "Active\nSAA",
}

REGION_LABELS = {
    "bay_area": "Bay Area",
    "chicago": "Chicago",
    "dallas": "Dallas",
    "dc_baltimore": "DC--Baltimore",
    "los_angeles": "Los Angeles",
    "nyc": "New York",
    "south_florida": "South Florida",
}

STRESS_VARIANT_ORDER = ["stress_s60", "weather_mild", "weather_severe", "weight_high", "weight_low"]
STRESS_VARIANT_LABELS = {
    "stress_s60": "60 scenarios",
    "weather_mild": "Mild weather",
    "weather_severe": "Severe weather",
    "weight_high": "High weights",
    "weight_low": "Low weights",
}
STRESS_COMPARATOR_ORDER = ["saa_extensive", "cvar_extensive", "minimax_extensive", "active_scenario_saa"]
STRESS_COMPARATOR_LABELS = {
    "saa_extensive": "SAA",
    "cvar_extensive": "CVaR",
    "minimax_extensive": "Minimax",
    "active_scenario_saa": "Active\nSAA",
}


def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _ai_or_metrics() -> pd.DataFrame:
    if not AI_OR_METRICS.exists():
        raise FileNotFoundError(AI_OR_METRICS)
    metrics = pd.read_csv(AI_OR_METRICS)
    metrics = metrics[metrics["method"].isin(METRIC_METHODS)].copy()
    metrics["method_label"] = metrics["method"].map(METRIC_METHODS)
    metrics["method_label"] = pd.Categorical(metrics["method_label"], PLOT_ORDER, ordered=True)
    return metrics.sort_values("method_label")


def _contrast_text_color(cmap, norm, value: float) -> str:
    r, g, b, _ = cmap(norm(value))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < 0.50 else "black"


def plot_regret_distribution() -> None:
    metrics = _ai_or_metrics()
    data = [
        metrics[metrics["method_label"].eq(method)]["max_regret"].dropna().astype(float).to_numpy()
        for method in PLOT_ORDER
    ]
    labels = [DISPLAY_LABELS[method] for method in PLOT_ORDER]

    fig, ax = plt.subplots(figsize=(6.35, 3.35))
    bp = ax.boxplot(
        data,
        patch_artist=True,
        showfliers=False,
        widths=0.55,
        medianprops={"color": "#111111", "linewidth": 1.1},
    )
    for patch, method in zip(bp["boxes"], PLOT_ORDER):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.76)
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
            fontsize=6.6,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.45},
        )
    fig.tight_layout(pad=0.25)
    _save(fig, "fig3_regret_distribution")


def plot_region_regret() -> None:
    metrics = _ai_or_metrics()
    region = (
        metrics.groupby(["region", "method_label"], observed=True)["max_regret"]
        .mean()
        .reset_index()
    )
    mat = region.pivot(index="region", columns="method_label", values="max_regret").reindex(columns=PLOT_ORDER)
    mat = mat.reindex([key for key in REGION_LABELS if key in mat.index])

    fig, ax = plt.subplots(figsize=(7.35, 3.65))
    values = mat.to_numpy(float)
    cmap = plt.get_cmap("YlGnBu_r")
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(mat.columns)), [DISPLAY_LABELS[str(col)] for col in mat.columns], rotation=0, ha="center")
    ax.set_yticks(range(len(mat.index)), [REGION_LABELS.get(str(v), str(v)) for v in mat.index])
    ax.set_title("Mean maximum regret by multi-airport region")
    ax.set_xlim(-0.55, len(mat.columns) - 0.25)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = float(mat.iloc[i, j])
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6.6,
                color=_contrast_text_color(cmap, norm, value),
            )
    fig.colorbar(im, ax=ax, label="Mean maximum regret")
    fig.tight_layout(pad=0.25)
    _save(fig, "fig5_region_regret")


def plot_retention_runtime() -> None:
    summary = pd.read_csv(AI_OR_SUMMARY)
    frame = summary[summary["method"].isin(ORDER)].copy()
    frame["method"] = pd.Categorical(frame["method"], categories=ORDER, ordered=True)
    frame = frame.sort_values("method")

    fig, ax = plt.subplots(figsize=(6.05, 3.45))
    offsets = {
        "AI proposal only": (-0.010, -0.105, "right"),
        "LG-RCCC audited": (0.008, 0.075, "left"),
        "RCCC audited": (-0.010, 0.060, "right"),
        "Full regret reference": (-0.010, 0.080, "right"),
        "SAA": (0.008, -0.060, "left"),
        "CVaR": (0.008, 0.040, "left"),
        "Minimax": (0.008, 0.040, "left"),
        "Active-scenario SAA": (0.008, -0.035, "left"),
    }
    for _, row in frame.iterrows():
        method = str(row["method"])
        x = float(row["candidate_retention"])
        y = float(row["solve_seconds"])
        ax.scatter(x, y, s=82, color=COLORS.get(method, "#444444"), zorder=3)
        dx, dy, ha = offsets.get(method, (0.006, 0.02, "left"))
        ax.text(
            x + dx,
            y + dy,
            f"{LABELS.get(method, method)}\n{x:.3f}, {y:.2f}s",
            fontsize=6.6,
            ha=ha,
            va="center",
            linespacing=0.95,
        )
    ax.set_xlabel("Mean candidate retention")
    ax.set_ylabel("Mean solve time (s)")
    ax.set_title("Candidate scale and solve time")
    ax.grid(alpha=0.25)
    ax.set_xlim(0.58, 1.08)
    ax.set_ylim(0.05, 1.72)
    ax.set_xticks([0.60, 0.70, 0.80, 0.90, 1.00])
    fig.tight_layout(pad=0.25)
    _save(fig, "fig4_retention_runtime")


def plot_ai_or_stress() -> None:
    summary_path = AI_OR_STRESS_DIR / "ai_or_stress_summary.csv"
    stats_path = AI_OR_STRESS_DIR / "ai_or_stress_statistical_audit.csv"
    if not summary_path.exists() or not stats_path.exists():
        return
    summary = pd.read_csv(summary_path)
    stats = pd.read_csv(stats_path)
    stats = stats[
        stats["variant"].isin(STRESS_VARIANT_ORDER)
        & stats["comparison_method"].isin(STRESS_COMPARATOR_ORDER)
    ].copy()
    if stats.empty:
        return
    stats["variant"] = pd.Categorical(stats["variant"], STRESS_VARIANT_ORDER, ordered=True)
    stats["comparison_method"] = pd.Categorical(
        stats["comparison_method"], STRESS_COMPARATOR_ORDER, ordered=True
    )
    stats = stats.sort_values(["variant", "comparison_method"])
    mat = (
        stats.pivot(index="variant", columns="comparison_method", values="mean_regret_gap_vs_lg_rccc")
        .reindex(STRESS_VARIANT_ORDER)
        .reindex(columns=STRESS_COMPARATOR_ORDER)
    )
    lg = (
        summary[summary["method"].eq("lg_rccc_audited")]
        .set_index("variant")
        .reindex(STRESS_VARIANT_ORDER)
    )

    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(6.95, 3.45),
        gridspec_kw={"width_ratios": [1.55, 1.0]},
    )
    cmap = plt.get_cmap("YlOrRd")
    values = mat.to_numpy(float)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    im = ax0.imshow(values, aspect="auto", cmap=cmap, norm=norm)
    ax0.set_xticks(range(len(STRESS_COMPARATOR_ORDER)), [STRESS_COMPARATOR_LABELS[m] for m in STRESS_COMPARATOR_ORDER])
    ax0.set_yticks(range(len(STRESS_VARIANT_ORDER)), [STRESS_VARIANT_LABELS[v] for v in STRESS_VARIANT_ORDER])
    ax0.set_title("Regret gap versus LG-RCCC")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = float(mat.iloc[i, j])
            ax0.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7.0,
                color=_contrast_text_color(cmap, norm, value),
            )
    fig.colorbar(im, ax=ax0, fraction=0.045, pad=0.03)

    y = np.arange(len(STRESS_VARIANT_ORDER))
    retention = lg["mean_candidate_retention"].astype(float).to_numpy()
    matches = lg["full_regret_matches"].astype(int).to_numpy()
    cases = lg["cases"].astype(int).to_numpy()
    bars = ax1.barh(y, retention, color=COLORS["LG-RCCC audited"], alpha=0.82)
    ax1.set_yticks(y, [""] * len(y))
    ax1.invert_yaxis()
    ax1.set_xlim(0, 1.05)
    ax1.set_xlabel("Candidate retention")
    ax1.set_title("LG-RCCC audited graph")
    ax1.grid(axis="x", alpha=0.25)
    for idx, (bar, value, match, case) in enumerate(zip(bars, retention, matches, cases)):
        ax1.text(
            value / 2,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="center",
            fontsize=7.0,
            color="white",
        )
        ax1.text(
            min(1.02, value + 0.035),
            bar.get_y() + bar.get_height() / 2,
            f"{match}/{case}",
            va="center",
            ha="left",
            fontsize=7.0,
            color="#111111",
        )
    fig.tight_layout(pad=0.25)
    _save(fig, "fig6_risk_hardening")


def main() -> None:
    plot_regret_distribution()
    plot_retention_runtime()
    plot_region_regret()
    plot_ai_or_stress()


if __name__ == "__main__":
    main()
