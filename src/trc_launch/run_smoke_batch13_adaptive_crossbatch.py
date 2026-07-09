from __future__ import annotations

from pathlib import Path

import pandas as pd

from .optimizers import LaunchOptimizer, compute_oracle_costs, evaluate_solution
from .run_smoke_batch1 import DEFAULT_BTS_CSV
from .run_smoke_batch7_filtered_regret import _filtered_instance
from .run_smoke_batch10_extended_sample import _instance_pressure
from .run_smoke_batch12_adaptive_filter import _adaptive_width
from .smoke_data import build_tail_launch_instance, load_bts_month


def _add_batch10_adaptive(rows: list[dict[str, object]]) -> None:
    month = load_bts_month(DEFAULT_BTS_CSV)
    selected = pd.read_csv("data/trc_open_smoke/batch10_extended_sample/selected_instances.csv")
    full = pd.read_csv("results/trc_smoke/batch10_extended_sample/batch10_extended_metrics.csv")
    full = full[full["method"].eq("regret_portfolio_full")][
        ["instance_key", "expected_cost", "max_regret"]
    ].rename(
        columns={
            "expected_cost": "full_expected_cost",
            "max_regret": "full_max_regret",
        }
    )
    full_lookup = full.set_index("instance_key").to_dict("index")

    for item in selected.itertuples(index=False):
        instance = build_tail_launch_instance(
            month,
            date=str(item.date),
            carrier=str(item.carrier),
            region_airports=("JFK", "LGA", "EWR"),
            scenario_count=30,
            scenario_mode="stress",
            seed=43,
        )
        pressure = _instance_pressure(instance)
        per_flight, per_aircraft, width_label = _adaptive_width(instance, pressure)
        filtered_instance, _ = _filtered_instance(instance, per_flight=per_flight, per_aircraft=per_aircraft)
        oracle_costs = compute_oracle_costs(instance, time_limit=7.0)
        result = LaunchOptimizer(filtered_instance, time_limit=45.0).solve(
            "regret_portfolio",
            oracle_costs=oracle_costs,
        )
        key = f"{item.carrier}_{item.date}"
        metrics = evaluate_solution(instance, result, oracle_costs=oracle_costs)
        metrics["method"] = "adaptive_certificate_regret"
        metrics["instance_key"] = key
        metrics["region"] = "nyc"
        metrics["sample_group"] = "batch10_nyc"
        metrics["candidate_count"] = len(filtered_instance.candidates)
        metrics["candidate_retention"] = len(filtered_instance.candidates) / max(1, len(instance.candidates))
        metrics["width_label"] = width_label
        metrics["per_flight"] = per_flight
        metrics["full_expected_cost"] = full_lookup[key]["full_expected_cost"]
        metrics["full_max_regret"] = full_lookup[key]["full_max_regret"]
        rows.append(metrics)


def _batch10_fixed() -> pd.DataFrame:
    metrics = pd.read_csv("results/trc_smoke/batch10_extended_sample/batch10_extended_metrics.csv")
    keep = metrics[metrics["method"].isin(["certificate_filtered_regret_k16", "certificate_filtered_regret_k20"])].copy()
    keep["method"] = keep["method"].map(
        {
            "certificate_filtered_regret_k16": "fixed_k16",
            "certificate_filtered_regret_k20": "fixed_k20",
        }
    )
    full = metrics[metrics["method"].eq("regret_portfolio_full")][
        ["instance_key", "expected_cost", "max_regret"]
    ].rename(
        columns={
            "expected_cost": "full_expected_cost",
            "max_regret": "full_max_regret",
        }
    )
    keep = keep.merge(full, on="instance_key", how="left")
    keep["region"] = "nyc"
    keep["sample_group"] = "batch10_nyc"
    keep["per_flight"] = keep["method"].map({"fixed_k16": 16, "fixed_k20": 20})
    keep["width_label"] = keep["method"]
    return keep


def _batch11_fixed_and_adaptive() -> pd.DataFrame:
    metrics = pd.read_csv("results/trc_smoke/batch12_adaptive_filter/batch12_adaptive_metrics.csv")
    keep = metrics[metrics["method"].isin(["fixed_k16", "fixed_k20", "adaptive_certificate_regret"])].copy()
    keep["sample_group"] = "batch11_multiregion"
    return keep


def _summary(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics["matches_full_cost"] = metrics["expected_cost"].round(8).eq(metrics["full_expected_cost"].round(8))
    metrics["matches_full_regret"] = metrics["max_regret"].round(8).eq(metrics["full_max_regret"].round(8))
    return (
        metrics.groupby("method")
        .agg(
            cases=("instance_key", "count"),
            regions=("region", "nunique"),
            mean_expected_cost=("expected_cost", "mean"),
            mean_cvar95_cost=("cvar95_cost", "mean"),
            mean_max_regret=("max_regret", "mean"),
            mean_solve_seconds=("solve_seconds", "mean"),
            mean_candidate_count=("candidate_count", "mean"),
            mean_candidate_retention=("candidate_retention", "mean"),
            cost_matches=("matches_full_cost", "sum"),
            regret_matches=("matches_full_regret", "sum"),
            mean_per_flight=("per_flight", "mean"),
        )
        .reset_index()
        .sort_values(["regret_matches", "mean_candidate_retention"], ascending=[False, True])
    )


def main() -> None:
    output_dir = Path("results/trc_smoke/batch13_adaptive_crossbatch")
    output_dir.mkdir(parents=True, exist_ok=True)

    adaptive_rows: list[dict[str, object]] = []
    _add_batch10_adaptive(adaptive_rows)
    batch10_adaptive = pd.DataFrame(adaptive_rows)
    batch10_adaptive["matches_full_cost"] = batch10_adaptive["expected_cost"].round(8).eq(
        batch10_adaptive["full_expected_cost"].round(8)
    )
    batch10_adaptive["matches_full_regret"] = batch10_adaptive["max_regret"].round(8).eq(
        batch10_adaptive["full_max_regret"].round(8)
    )
    batch10_adaptive.to_csv(output_dir / "batch10_adaptive_metrics.csv", index=False)

    combined = pd.concat([_batch10_fixed(), _batch11_fixed_and_adaptive(), batch10_adaptive], ignore_index=True)
    combined["matches_full_cost"] = combined["expected_cost"].round(8).eq(combined["full_expected_cost"].round(8))
    combined["matches_full_regret"] = combined["max_regret"].round(8).eq(combined["full_max_regret"].round(8))
    combined.to_csv(output_dir / "batch13_crossbatch_metrics.csv", index=False)

    summary = _summary(combined)
    summary.to_csv(output_dir / "batch13_crossbatch_summary.csv", index=False)
    width_counts = (
        combined[combined["method"].eq("adaptive_certificate_regret")]
        .groupby(["sample_group", "width_label", "per_flight"])
        .size()
        .rename("cases")
        .reset_index()
    )
    width_counts.to_csv(output_dir / "batch13_adaptive_width_counts.csv", index=False)
    failures = combined[(~combined["matches_full_cost"]) | (~combined["matches_full_regret"])]
    failures.to_csv(output_dir / "batch13_crossbatch_failures.csv", index=False)
    print(summary.to_string(index=False))
    print("\nFailures")
    if failures.empty:
        print("none")
    else:
        print(failures[["method", "sample_group", "instance_key", "expected_cost", "full_expected_cost", "max_regret", "full_max_regret", "per_flight"]].to_string(index=False))
    print("\nAdaptive width counts")
    print(width_counts.to_string(index=False))


if __name__ == "__main__":
    main()
