from __future__ import annotations

from pathlib import Path

import pandas as pd

from .distilled_policy import fit_distilled_model, make_labeled_frame, solve_with_distilled_model
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
    candidate_retention: float,
) -> None:
    metrics = evaluate_solution(instance, result, oracle_costs=oracle_costs)
    metrics["method"] = method_name
    metrics["instance_key"] = instance_key
    metrics["candidate_count"] = candidate_count
    metrics["candidate_retention"] = candidate_retention
    rows.append(metrics)


def main() -> None:
    output_dir = Path("results/trc_smoke/batch8_strong_baselines")
    data_dir = Path("data/trc_open_smoke/batch8_strong_baselines")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    month = load_bts_month(DEFAULT_BTS_CSV)
    instances = {}
    for carrier, date in PAIRS:
        inst = build_tail_launch_instance(
            month,
            date=date,
            carrier=carrier,
            scenario_count=10,
            scenario_mode="stress",
            seed=31,
        )
        if not inst.flights.empty and not inst.inventory.empty:
            instances[_key(carrier, date)] = inst

    oracle_costs_by_key = {}
    teacher_results = {}
    for key, instance in instances.items():
        oracle_costs = compute_oracle_costs(instance, time_limit=6.0)
        oracle_costs_by_key[key] = oracle_costs
        teacher_results[key] = LaunchOptimizer(instance, time_limit=20.0).solve(
            "regret_portfolio",
            oracle_costs=oracle_costs,
        )
        make_labeled_frame(instance, teacher_results[key]).to_csv(
            data_dir / f"teacher_labels_{key}.csv",
            index=False,
        )

    rows: list[dict[str, object]] = []
    full_methods = [
        "saa_extensive",
        "active_scenario_saa",
        "cvar_extensive",
        "mean_cvar95_extensive",
        "minimax_extensive",
        "regret_portfolio",
        "certificate_value",
    ]
    filtered_methods = [
        ("saa_extensive", "certificate_filtered_saa_k16"),
        ("mean_cvar95_extensive", "certificate_filtered_cvar95_k16"),
        ("minimax_extensive", "certificate_filtered_minimax_k16"),
        ("regret_portfolio", "certificate_filtered_regret_k16"),
    ]

    for key, instance in instances.items():
        oracle_costs = oracle_costs_by_key[key]
        filtered_instance, original_count = _filtered_instance(instance, per_flight=16, per_aircraft=8)

        for method in full_methods:
            if method == "regret_portfolio":
                result = teacher_results[key]
                method_name = "regret_portfolio_full"
            else:
                result = LaunchOptimizer(instance, time_limit=20.0).solve(method)
                method_name = method
            _add_metrics(
                rows,
                key,
                instance,
                result,
                oracle_costs,
                method_name,
                len(instance.candidates),
                1.0,
            )

        train_frames = [
            make_labeled_frame(instances[train_key], teacher_results[train_key])
            for train_key in instances
            if train_key != key
        ]
        distilled = fit_distilled_model(train_frames)
        distilled_result = solve_with_distilled_model(instance, distilled, repair=True)
        _add_metrics(
            rows,
            key,
            instance,
            distilled_result,
            oracle_costs,
            "regret_distilled_repaired",
            len(instance.candidates),
            1.0,
        )

        for method, method_name in filtered_methods:
            if method == "regret_portfolio":
                result = LaunchOptimizer(filtered_instance, time_limit=20.0).solve(
                    "regret_portfolio",
                    oracle_costs=oracle_costs,
                )
            else:
                result = LaunchOptimizer(filtered_instance, time_limit=20.0).solve(method)
            _add_metrics(
                rows,
                key,
                instance,
                result,
                oracle_costs,
                method_name,
                len(filtered_instance.candidates),
                len(filtered_instance.candidates) / max(1, original_count),
            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "batch8_strong_metrics.csv", index=False)
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
            mean_ferry_moves=("ferry_moves", "mean"),
            mean_candidate_count=("candidate_count", "mean"),
            mean_candidate_retention=("candidate_retention", "mean"),
        )
        .reset_index()
        .sort_values(["mean_max_regret", "mean_expected_cost"])
    )
    summary.to_csv(output_dir / "batch8_strong_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

