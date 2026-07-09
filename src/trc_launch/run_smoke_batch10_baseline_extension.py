from __future__ import annotations

from pathlib import Path

import pandas as pd

from .optimizers import LaunchOptimizer, compute_oracle_costs, evaluate_solution
from .run_smoke_batch1 import DEFAULT_BTS_CSV
from .run_smoke_batch7_filtered_regret import _filtered_instance
from .run_smoke_batch10_extended_sample import _instance_pressure, _key, select_instances
from .smoke_data import build_tail_launch_instance, load_bts_month


def _add_metrics(
    rows: list[dict[str, object]],
    instance_key: str,
    pressure: dict[str, object],
    instance,
    result,
    oracle_costs: dict[int, float],
    method_name: str,
    candidate_count: int,
    original_candidate_count: int,
) -> None:
    metrics = evaluate_solution(instance, result, oracle_costs=oracle_costs)
    metrics["method"] = method_name
    metrics["instance_key"] = instance_key
    metrics["candidate_count"] = candidate_count
    metrics["candidate_retention"] = candidate_count / max(1, original_candidate_count)
    metrics["scenario_count"] = int(instance.scenarios["scenario_id"].nunique())
    metrics["binary_var_count"] = int(candidate_count + metrics["scenario_count"] * len(instance.flights))
    metrics["total_deficit"] = pressure["total_deficit"]
    metrics["max_airport_deficit"] = pressure["max_airport_deficit"]
    rows.append(metrics)


def _summary(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby("method")
        .agg(
            cases=("instance_key", "count"),
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
    output_dir = Path("results/trc_smoke/batch10_extended_sample")
    data_dir = Path("data/trc_open_smoke/batch10_extended_sample")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    month = load_bts_month(DEFAULT_BTS_CSV)
    selected_path = data_dir / "selected_instances.csv"
    if selected_path.exists():
        selected = pd.read_csv(selected_path)
    else:
        selected = select_instances(month, max_instances=20)
        selected.to_csv(selected_path, index=False)

    rows: list[dict[str, object]] = []
    for item in selected.itertuples(index=False):
        carrier = str(item.carrier)
        date = str(item.date)
        instance = build_tail_launch_instance(
            month,
            date=date,
            carrier=carrier,
            scenario_count=30,
            scenario_mode="stress",
            seed=43,
        )
        pressure = _instance_pressure(instance)
        key = _key(carrier, date)
        original_count = len(instance.candidates)
        oracle_costs = compute_oracle_costs(instance, time_limit=6.0)

        full_methods = [
            ("active_scenario_saa", "active_scenario_saa"),
            ("mean_cvar95_extensive", "mean_cvar95_extensive"),
            ("minimax_extensive", "minimax_extensive"),
        ]
        for method, method_name in full_methods:
            result = LaunchOptimizer(instance, time_limit=35.0).solve(method)
            _add_metrics(rows, key, pressure, instance, result, oracle_costs, method_name, original_count, original_count)

        filtered_instance, _ = _filtered_instance(instance, per_flight=20, per_aircraft=10)
        filtered_methods = [
            ("saa_extensive", "certificate_filtered_saa_k20"),
            ("mean_cvar95_extensive", "certificate_filtered_cvar95_k20"),
            ("minimax_extensive", "certificate_filtered_minimax_k20"),
        ]
        for method, method_name in filtered_methods:
            result = LaunchOptimizer(filtered_instance, time_limit=35.0).solve(method)
            _add_metrics(
                rows,
                key,
                pressure,
                instance,
                result,
                oracle_costs,
                method_name,
                len(filtered_instance.candidates),
                original_count,
            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "batch10_baseline_extension_metrics.csv", index=False)
    summary = _summary(metrics)
    summary.to_csv(output_dir / "batch10_baseline_extension_summary.csv", index=False)

    original_metrics = pd.read_csv(output_dir / "batch10_extended_metrics.csv")
    combined = pd.concat([original_metrics, metrics], ignore_index=True)
    combined_summary = _summary(combined)
    combined_summary.to_csv(output_dir / "batch10_combined_summary.csv", index=False)
    print(combined_summary.to_string(index=False))


if __name__ == "__main__":
    main()
