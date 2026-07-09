from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METHOD_LABELS = {
    "lg_rccc_audited": "LG-RCCC audited",
    "lg_rccc_pre_audit": "LG-RCCC pre-audit",
    "ml_proposal_only": "AI proposal only",
    "rccc_audited": "RCCC audited",
    "regret_portfolio_full": "Full regret reference",
    "saa_extensive": "SAA",
    "cvar_extensive": "CVaR",
    "minimax_extensive": "Minimax",
    "active_scenario_saa": "Active-scenario SAA",
}

MAIN_METHOD_ORDER = [
    "lg_rccc_audited",
    "lg_rccc_pre_audit",
    "ml_proposal_only",
    "rccc_audited",
    "regret_portfolio_full",
    "saa_extensive",
    "cvar_extensive",
    "minimax_extensive",
    "active_scenario_saa",
]


def _format_match(row: pd.Series) -> str:
    return f"{int(row['full_regret_matches'])}/{int(row['cases'])}"


def build_main_summary(summary: pd.DataFrame) -> pd.DataFrame:
    frame = summary[summary["method"].isin(MAIN_METHOD_ORDER)].copy()
    frame["method_order"] = frame["method"].map({m: i for i, m in enumerate(MAIN_METHOD_ORDER)})
    frame = frame.sort_values("method_order")
    out = pd.DataFrame(
        {
            "method": frame["method"].map(METHOD_LABELS).fillna(frame["method"]),
            "cases": frame["cases"].astype(int),
            "candidate_retention": frame["mean_candidate_retention"].astype(float),
            "mean_max_regret": frame["mean_max_regret"].astype(float),
            "worst_max_regret": frame["worst_case_regret"].astype(float),
            "expected_cost": frame["mean_expected_cost"].astype(float),
            "solve_seconds": frame["mean_solve_seconds"].astype(float),
            "full_regret_matches": frame.apply(_format_match, axis=1),
        }
    )
    return out


def build_trace_case(action_log: pd.DataFrame, preferred_key: str) -> pd.DataFrame:
    if action_log.empty:
        return action_log
    if preferred_key in set(action_log["instance_key"].astype(str)):
        trace = action_log[action_log["instance_key"].astype(str).eq(preferred_key)].copy()
    else:
        expanded = action_log[pd.to_numeric(action_log["audit_expansions"], errors="coerce").fillna(0) > 0]
        trace = expanded.head(1).copy() if not expanded.empty else action_log.head(1).copy()
    keep = [
        "instance_key",
        "observed_state",
        "ai_proposed_edges",
        "ai_proposed_share",
        "ai_scenario_attention",
        "ai_binding_blocks",
        "pre_audit_regret",
        "post_audit_regret",
        "audit_path",
        "final_retained_edges",
        "final_retention",
        "full_regret_match",
    ]
    return trace[[col for col in keep if col in trace.columns]]


def write_assets(input_dir: Path, generated_dir: Path, preferred_trace: str) -> None:
    generated_dir.mkdir(parents=True, exist_ok=True)
    summary_path = input_dir / "ai_or_summary.csv"
    action_path = input_dir / "ai_or_action_log.csv"
    stats_path = input_dir / "ai_or_statistical_audit.csv"
    importance_path = input_dir / "ai_or_feature_importance.csv"

    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    summary = pd.read_csv(summary_path)
    main_summary = build_main_summary(summary)
    main_summary.to_csv(generated_dir / "ai_or_main_summary.csv", index=False)

    if action_path.exists():
        action_log = pd.read_csv(action_path)
        action_log.to_csv(generated_dir / "ai_or_action_log.csv", index=False)
        build_trace_case(action_log, preferred_trace).to_csv(generated_dir / "ai_or_trace_case.csv", index=False)

    if stats_path.exists():
        pd.read_csv(stats_path).to_csv(generated_dir / "ai_or_statistical_audit.csv", index=False)

    if importance_path.exists():
        importance = pd.read_csv(importance_path)
        if "importance" in importance.columns:
            importance["importance"] = pd.to_numeric(importance["importance"], errors="coerce").fillna(0.0)
            importance = (
                importance.groupby("feature", as_index=False)["importance"]
                .mean()
                .sort_values("importance", ascending=False)
            )
        importance.to_csv(generated_dir / "ai_or_feature_importance.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write manuscript-facing AI-OR decision-intelligence assets.")
    parser.add_argument("--input-dir", default="results/trc_smoke/batch34_decision_intelligence_weather_full360")
    parser.add_argument("--generated-dir", default="article/trc_elsarticle/generated")
    parser.add_argument("--preferred-trace", default="nyc_DL_2025-07-15")
    args = parser.parse_args()
    write_assets(Path(args.input_dir), Path(args.generated_dir), args.preferred_trace)


if __name__ == "__main__":
    main()
