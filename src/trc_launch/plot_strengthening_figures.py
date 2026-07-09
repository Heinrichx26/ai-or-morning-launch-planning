from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "results" / "trc_smoke" / "batch29_strengthening"
FIG_DIR = ROOT / "article" / "trc_elsarticle" / "figures"

METHOD_LABELS = {
    "regret_portfolio_full": "Full",
    "dominance_cost_floor": "Cost floor",
    "exposure_only": "Exposure",
    "tail_coverage_only": "Tail coverage",
    "adaptive_without_audit": "Adaptive",
    "rccc_audited": "RCCC",
    "saa_extensive": "SAA",
    "cvar_extensive": "CVaR",
}

METHOD_COLORS = {
    "regret_portfolio_full": "#555555",
    "dominance_cost_floor": "#D08C60",
    "exposure_only": "#577590",
    "tail_coverage_only": "#8AB17D",
    "adaptive_without_audit": "#E76F51",
    "rccc_audited": "#0B6E69",
    "saa_extensive": "#8E6C8A",
    "cvar_extensive": "#C44E52",
}


def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def mechanism_case_figure() -> None:
    summary_path = RESULT_DIR / "mechanism_ablation_summary.csv"
    blocks_path = RESULT_DIR / "operational_case_blocks.csv"
    scenarios_path = RESULT_DIR / "operational_case_scenarios.csv"
    if not summary_path.exists() or not blocks_path.exists() or not scenarios_path.exists():
        return

    summary = pd.read_csv(summary_path)
    blocks = pd.read_csv(blocks_path)
    scenarios = pd.read_csv(scenarios_path)
    method_order = [
        "regret_portfolio_full",
        "dominance_cost_floor",
        "exposure_only",
        "tail_coverage_only",
        "adaptive_without_audit",
        "rccc_audited",
    ]
    summary = summary.set_index("method").reindex(method_order).dropna(how="all").reset_index()

    with plt.rc_context(
        {
            "font.size": 7.2,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.2,
        }
    ):
        fig = plt.figure(figsize=(7.2, 4.65))
        gs = fig.add_gridspec(2, 2, height_ratios=[0.95, 1.05], hspace=0.56, wspace=0.36)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])

        x = np.arange(len(summary))
        colors = [METHOD_COLORS.get(m, "#777777") for m in summary["method"]]
        ax0.bar(x, summary["mean_max_regret"], color=colors, alpha=0.86)
        ax0.set_xticks(x, [METHOD_LABELS.get(m, m) for m in summary["method"]], rotation=32, ha="right")
        ax0.set_ylabel("Mean max regret")
        ax0.set_title("A. Mechanism contribution", loc="left", pad=4)
        ax0.grid(axis="y", alpha=0.25)

        ax1.bar(x, summary["mean_candidate_retention"], color=colors, alpha=0.86)
        ax1.set_ylim(0, 1.15)
        ax1.set_xticks(x, [METHOD_LABELS.get(m, m) for m in summary["method"]], rotation=32, ha="right")
        ax1.set_ylabel("Candidate retention")
        ax1.set_title("B. Graph size and matches", loc="left", pad=4)
        for idx, row in summary.iterrows():
            ax1.text(
                idx,
                min(1.08, float(row["mean_candidate_retention"]) + 0.035),
                f"{int(row['full_regret_matches'])}/{int(row['cases'])}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        ax1.grid(axis="y", alpha=0.25)

        heat = blocks.pivot_table(index="airport", columns="hour", values="p10_shortage", aggfunc="mean").fillna(0.0)
        im = ax2.imshow(heat.to_numpy(float), aspect="auto", cmap="YlOrRd")
        ax2.set_xticks(range(len(heat.columns)), heat.columns.astype(str))
        ax2.set_yticks(range(len(heat.index)), heat.index)
        ax2.set_xlabel("Departure hour")
        ax2.set_title("C. Case airport-hour exposure", loc="left", pad=4)
        for i in range(heat.shape[0]):
            for j in range(heat.shape[1]):
                ax2.text(j, i, f"{heat.iloc[i, j]:.1f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.025, label="P10 shortage")

        methods = ["regret_portfolio_full", "rccc_audited", "saa_extensive", "cvar_extensive"]
        for method in methods:
            frame = scenarios[scenarios["method"].eq(method)].sort_values("scenario_regret").reset_index(drop=True)
            if frame.empty:
                continue
            ax3.plot(
                np.arange(1, len(frame) + 1),
                frame["scenario_regret"],
                marker="o",
                markersize=2.4,
                linewidth=1.15,
                color=METHOD_COLORS.get(method, "#777777"),
                label=METHOD_LABELS.get(method, method),
            )
        ax3.set_xlabel("Scenario rank")
        ax3.set_ylabel("Scenario regret")
        ax3.set_title("D. Case scenario regret", loc="left", pad=4)
        ax3.grid(alpha=0.25)
        ax3.legend(frameon=False, loc="upper left")

        _save(fig, "fig8_mechanism_case")


def main() -> None:
    mechanism_case_figure()
    print(f"Wrote strengthening figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
