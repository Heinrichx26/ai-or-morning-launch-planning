from __future__ import annotations

from pathlib import Path

import pandas as pd

from .distilled_policy import fit_distilled_model, make_labeled_frame, solve_with_distilled_model
from .optimizers import LaunchOptimizer, compute_oracle_costs, evaluate_solution
from .run_smoke_batch1 import DEFAULT_BTS_CSV
from .run_smoke_batch4_multicarrier import PAIRS
from .smoke_data import build_tail_launch_instance, load_bts_month, write_instance_audit


def _key(carrier: str, date: str) -> str:
    return f"{carrier}_{date}"


def main() -> None:
    output_dir = Path("results/trc_smoke/batch6_tail_multicase")
    data_dir = Path("data/trc_open_smoke/batch6_tail_multicase")
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
        if inst.flights.empty or inst.inventory.empty:
            continue
        key = _key(carrier, date)
        instances[key] = inst
        write_instance_audit(inst, data_dir)
        inst.flights.to_csv(data_dir / f"flights_tail_{key}.csv", index=False)
        inst.inventory.to_csv(data_dir / f"inventory_tail_{key}.csv", index=False)
        inst.scenarios.to_csv(data_dir / f"scenarios_tail_{key}.csv", index=False)

    teacher_results = {}
    oracle_costs_by_key = {}
    for key, instance in instances.items():
        oracle_costs = compute_oracle_costs(instance, time_limit=6.0)
        oracle_costs_by_key[key] = oracle_costs
        teacher_results[key] = LaunchOptimizer(instance, time_limit=20.0).solve(
            "regret_portfolio",
            oracle_costs=oracle_costs,
        )
        make_labeled_frame(instance, teacher_results[key]).to_csv(
            data_dir / f"teacher_labels_tail_{key}.csv",
            index=False,
        )

    methods = [
        "launch_value",
        "certificate_value",
        "saa_extensive",
        "cvar_extensive",
        "regret_portfolio",
        "predict_opt",
        "robust_quantile",
    ]
    rows: list[dict[str, object]] = []
    for test_key, instance in instances.items():
        train_frames = [
            make_labeled_frame(instances[key], teacher_results[key])
            for key in instances
            if key != test_key
        ]
        distilled = fit_distilled_model(train_frames)
        for repair in [False, True]:
            result = solve_with_distilled_model(instance, distilled, repair=repair)
            metrics = evaluate_solution(instance, result, oracle_costs=oracle_costs_by_key[test_key])
            metrics["instance_key"] = test_key
            rows.append(metrics)

        for method in methods:
            if method == "regret_portfolio":
                result = teacher_results[test_key]
            else:
                result = LaunchOptimizer(instance, time_limit=20.0).solve(method)
            metrics = evaluate_solution(instance, result, oracle_costs=oracle_costs_by_key[test_key])
            metrics["instance_key"] = test_key
            rows.append(metrics)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "batch6_tail_metrics.csv", index=False)
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
            mean_assigned_flights=("assigned_flights", "mean"),
        )
        .reset_index()
        .sort_values(["mean_max_regret", "mean_expected_cost"])
    )
    summary.to_csv(output_dir / "batch6_tail_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

