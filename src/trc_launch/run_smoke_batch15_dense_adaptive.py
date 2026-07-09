from __future__ import annotations

from pathlib import Path

import pandas as pd

from .optimizers import LaunchOptimizer, compute_oracle_costs, evaluate_solution
from .run_smoke_batch7_filtered_regret import _filtered_instance
from .run_smoke_batch11_multiregion import _bts_csv, _key
from .run_smoke_batch12_adaptive_filter import _adaptive_width
from .smoke_data import build_tail_launch_instance, load_bts_month


def _add_metrics(
    rows: list[dict[str, object]],
    instance_key: str,
    region_name: str,
    month: int,
    pressure: dict[str, object],
    instance,
    result,
    oracle_costs: dict[int, float],
    candidate_count: int,
    original_candidate_count: int,
    width_label: str,
    per_flight: int,
    per_aircraft: int,
) -> None:
    metrics = evaluate_solution(instance, result, oracle_costs=oracle_costs)
    metrics["method"] = "adaptive_certificate_regret_dense"
    metrics["instance_key"] = instance_key
    metrics["region"] = region_name
    metrics["month"] = month
    metrics["candidate_count"] = candidate_count
    metrics["candidate_retention"] = candidate_count / max(1, original_candidate_count)
    metrics["scenario_count"] = int(instance.scenarios["scenario_id"].nunique())
    metrics["binary_var_count"] = int(candidate_count + metrics["scenario_count"] * len(instance.flights))
    metrics["total_deficit"] = pressure["total_deficit"]
    metrics["max_airport_deficit"] = pressure["max_airport_deficit"]
    metrics["width_label"] = width_label
    metrics["per_flight"] = per_flight
    metrics["per_aircraft"] = per_aircraft
    rows.append(metrics)


def _summary(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby("method")
        .agg(
            cases=("instance_key", "count"),
            regions=("region", "nunique"),
            months=("month", "nunique"),
            mean_expected_cost=("expected_cost", "mean"),
            mean_cvar95_cost=("cvar95_cost", "mean"),
            mean_worst_cost=("worst_cost", "mean"),
            mean_launch_weight_share=("mean_launch_weight_share", "mean"),
            mean_max_regret=("max_regret", "mean"),
            mean_solve_seconds=("solve_seconds", "mean"),
            mean_candidate_count=("candidate_count", "mean"),
            mean_candidate_retention=("candidate_retention", "mean"),
            mean_binary_var_count=("binary_var_count", "mean"),
        )
        .reset_index()
        .sort_values(["mean_max_regret", "mean_expected_cost"])
    )


def main() -> None:
    output_dir = Path("results/trc_smoke/batch15_dense_adaptive")
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv("data/trc_open_smoke/batch14_year_round/selected_instances.csv")
    rows: list[dict[str, object]] = []
    month_cache: dict[int, pd.DataFrame] = {}

    for item in selected.itertuples(index=False):
        month = int(item.month)
        if month not in month_cache:
            month_cache[month] = load_bts_month(_bts_csv(month))
        airports = str(item.region_airports).split("|")
        instance = build_tail_launch_instance(
            month_cache[month],
            date=str(item.date),
            carrier=str(item.carrier),
            region_airports=airports,
            scenario_count=30,
            scenario_mode="stress",
            seed=83,
        )
        pressure = {
            "total_deficit": int(item.total_deficit),
            "max_airport_deficit": int(item.max_airport_deficit),
        }
        key = _key(str(item.region), str(item.carrier), str(item.date))
        original_count = len(instance.candidates)
        oracle_costs = compute_oracle_costs(instance, time_limit=8.0)
        per_flight, per_aircraft, width_label = _adaptive_width(instance, pressure)
        filtered_instance, _ = _filtered_instance(instance, per_flight=per_flight, per_aircraft=per_aircraft)
        result = LaunchOptimizer(filtered_instance, time_limit=50.0).solve(
            "regret_portfolio",
            oracle_costs=oracle_costs,
        )
        print(
            f"adaptive {key}: width={width_label} k={per_flight}/{per_aircraft} "
            f"candidates={len(filtered_instance.candidates)}/{original_count}"
        )
        _add_metrics(
            rows,
            key,
            str(item.region),
            month,
            pressure,
            instance,
            result,
            oracle_costs,
            len(filtered_instance.candidates),
            original_count,
            width_label,
            per_flight,
            per_aircraft,
        )

    dense_metrics = pd.DataFrame(rows)
    full_batch14 = pd.read_csv("results/trc_smoke/batch14_year_round/batch14_year_round_metrics.csv")
    original_adaptive = full_batch14[full_batch14["method"].eq("adaptive_certificate_regret")].copy()
    original_adaptive["method"] = "adaptive_certificate_regret_v1"
    baselines = full_batch14[~full_batch14["method"].eq("adaptive_certificate_regret")].copy()
    combined = pd.concat([baselines, original_adaptive, dense_metrics], ignore_index=True)

    full = baselines[baselines["method"].eq("regret_portfolio_full")][
        ["instance_key", "expected_cost", "max_regret", "solve_seconds"]
    ].rename(
        columns={
            "expected_cost": "full_expected_cost",
            "max_regret": "full_max_regret",
            "solve_seconds": "full_solve_seconds",
        }
    )
    dense_check = dense_metrics.merge(full, on="instance_key", how="left")
    dense_check["matches_full_cost"] = dense_check["expected_cost"].round(8).eq(
        dense_check["full_expected_cost"].round(8)
    )
    dense_check["matches_full_regret"] = dense_check["max_regret"].round(8).eq(
        dense_check["full_max_regret"].round(8)
    )
    dense_check["speedup_vs_full"] = dense_check["full_solve_seconds"] / dense_check["solve_seconds"].replace(0, pd.NA)

    combined.to_csv(output_dir / "batch15_dense_combined_metrics.csv", index=False)
    dense_check.to_csv(output_dir / "batch15_dense_adaptive_vs_full.csv", index=False)
    summary = _summary(combined)
    summary.to_csv(output_dir / "batch15_dense_combined_summary.csv", index=False)
    width_counts = (
        dense_check.groupby(["width_label", "per_flight", "per_aircraft"])
        .size()
        .rename("cases")
        .reset_index()
    )
    width_counts.to_csv(output_dir / "batch15_dense_width_counts.csv", index=False)
    failures = dense_check[(~dense_check["matches_full_cost"]) | (~dense_check["matches_full_regret"])]
    failures.to_csv(output_dir / "batch15_dense_failures.csv", index=False)

    print(summary.to_string(index=False))
    print("\nDense adaptive check")
    print(
        dense_check.agg(
            cost_matches=("matches_full_cost", "sum"),
            regret_matches=("matches_full_regret", "sum"),
            mean_speedup=("speedup_vs_full", "mean"),
            mean_retention=("candidate_retention", "mean"),
        ).to_string()
    )
    print("\nDense width counts")
    print(width_counts.to_string(index=False))
    print("\nFailures")
    if failures.empty:
        print("none")
    else:
        print(failures[["instance_key", "expected_cost", "full_expected_cost", "max_regret", "full_max_regret"]].to_string(index=False))


if __name__ == "__main__":
    main()
