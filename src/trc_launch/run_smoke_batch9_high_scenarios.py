from __future__ import annotations

from pathlib import Path

import pandas as pd

from .optimizers import LaunchOptimizer, compute_oracle_costs, evaluate_solution
from .run_smoke_batch1 import DEFAULT_BTS_CSV
from .run_smoke_batch4_multicarrier import PAIRS
from .run_smoke_batch7_filtered_regret import _filtered_instance
from .smoke_data import build_tail_launch_instance, load_bts_month


def _key(carrier: str, date: str) -> str:
    return f"{carrier}_{date}"


def _add_metrics(
    rows: list[dict[str, object]],
    instance_key: str,
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
    metrics["flight_count"] = int(len(instance.flights))
    metrics["binary_var_count"] = int(candidate_count + metrics["scenario_count"] * len(instance.flights))
    rows.append(metrics)


def main() -> None:
    output_dir = Path("results/trc_smoke/batch9_high_scenarios")
    output_dir.mkdir(parents=True, exist_ok=True)

    month = load_bts_month(DEFAULT_BTS_CSV)
    rows: list[dict[str, object]] = []
    for carrier, date in PAIRS:
        instance = build_tail_launch_instance(
            month,
            date=date,
            carrier=carrier,
            scenario_count=30,
            scenario_mode="stress",
            seed=43,
        )
        if instance.flights.empty or instance.inventory.empty:
            continue
        key = _key(carrier, date)
        oracle_costs = compute_oracle_costs(instance, time_limit=6.0)
        original_count = len(instance.candidates)

        strong_methods = [
            ("saa_extensive", "saa_extensive"),
            ("cvar_extensive", "cvar_extensive"),
            ("minimax_extensive", "minimax_extensive"),
            ("regret_portfolio", "regret_portfolio_full"),
        ]
        for method, method_name in strong_methods:
            if method == "regret_portfolio":
                result = LaunchOptimizer(instance, time_limit=30.0).solve(method, oracle_costs=oracle_costs)
            else:
                result = LaunchOptimizer(instance, time_limit=30.0).solve(method)
            _add_metrics(rows, key, instance, result, oracle_costs, method_name, original_count, original_count)

        for per_flight, per_aircraft in [(12, 6), (16, 8), (20, 10), (24, 12)]:
            filtered_instance, _ = _filtered_instance(
                instance,
                per_flight=per_flight,
                per_aircraft=per_aircraft,
            )
            result = LaunchOptimizer(filtered_instance, time_limit=30.0).solve(
                "regret_portfolio",
                oracle_costs=oracle_costs,
            )
            _add_metrics(
                rows,
                key,
                instance,
                result,
                oracle_costs,
                f"certificate_filtered_regret_k{per_flight}",
                len(filtered_instance.candidates),
                original_count,
            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "batch9_high_scenario_metrics.csv", index=False)
    summary = (
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
    summary.to_csv(output_dir / "batch9_high_scenario_summary.csv", index=False)

    full = metrics[metrics["method"].eq("regret_portfolio_full")][
        ["instance_key", "expected_cost", "max_regret", "solve_seconds"]
    ].rename(
        columns={
            "expected_cost": "full_expected_cost",
            "max_regret": "full_max_regret",
            "solve_seconds": "full_solve_seconds",
        }
    )
    comparisons = []
    for method in [
        "certificate_filtered_regret_k12",
        "certificate_filtered_regret_k16",
        "certificate_filtered_regret_k20",
        "certificate_filtered_regret_k24",
    ]:
        comp = metrics[metrics["method"].eq(method)].merge(full, on="instance_key", how="left")
        comp["matches_full_cost"] = comp["expected_cost"].round(8).eq(comp["full_expected_cost"].round(8))
        comp["matches_full_regret"] = comp["max_regret"].round(8).eq(comp["full_max_regret"].round(8))
        comp["speedup_vs_full"] = comp["full_solve_seconds"] / comp["solve_seconds"].replace(0, pd.NA)
        comparisons.append(comp)
    compare = pd.concat(comparisons, ignore_index=True)
    compare.to_csv(output_dir / "batch9_filtered_vs_full.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
