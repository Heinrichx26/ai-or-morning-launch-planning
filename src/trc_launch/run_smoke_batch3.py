from __future__ import annotations

from pathlib import Path

import pandas as pd

from .distilled_policy import fit_distilled_model, make_labeled_frame, solve_with_distilled_model
from .optimizers import LaunchOptimizer, compute_oracle_costs, evaluate_solution
from .run_smoke_batch1 import DEFAULT_BTS_CSV
from .smoke_data import build_launch_instance, load_bts_month


def main() -> None:
    output_dir = Path("results/trc_smoke/batch3_distilled")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path("data/trc_open_smoke/batch3_distilled")
    data_dir.mkdir(parents=True, exist_ok=True)

    dates = ["2025-01-06", "2025-01-07", "2025-01-11", "2025-01-20", "2025-01-30"]
    month = load_bts_month(DEFAULT_BTS_CSV)
    instances = {
        date: build_launch_instance(
            month,
            date=date,
            carrier="DL",
            scenario_count=24,
            scenario_mode="stress",
        )
        for date in dates
    }

    teacher_results = {}
    oracle_costs_by_date = {}
    for date, instance in instances.items():
        oracle_costs = compute_oracle_costs(instance, time_limit=6.0)
        oracle_costs_by_date[date] = oracle_costs
        teacher_results[date] = LaunchOptimizer(instance, time_limit=15.0).solve(
            "regret_portfolio",
            oracle_costs=oracle_costs,
        )
        make_labeled_frame(instance, teacher_results[date]).to_csv(
            data_dir / f"teacher_labels_DL_{date}.csv",
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
    for test_date in dates:
        train_frames = [
            make_labeled_frame(instances[date], teacher_results[date])
            for date in dates
            if date != test_date
        ]
        distilled = fit_distilled_model(train_frames)
        instance = instances[test_date]
        oracle_costs = oracle_costs_by_date[test_date]
        distilled_result = solve_with_distilled_model(instance, distilled)
        rows.append(evaluate_solution(instance, distilled_result, oracle_costs=oracle_costs))
        distilled_result.selected.to_csv(
            output_dir / f"selected_regret_distilled_DL_{test_date}.csv",
            index=False,
        )
        repaired_result = solve_with_distilled_model(instance, distilled, repair=True)
        rows.append(evaluate_solution(instance, repaired_result, oracle_costs=oracle_costs))
        repaired_result.selected.to_csv(
            output_dir / f"selected_regret_distilled_repaired_DL_{test_date}.csv",
            index=False,
        )

        for method in methods:
            if method == "regret_portfolio":
                result = teacher_results[test_date]
            else:
                result = LaunchOptimizer(instance, time_limit=15.0).solve(method)
            rows.append(evaluate_solution(instance, result, oracle_costs=oracle_costs))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "batch3_smoke_metrics.csv", index=False)
    summary = (
        metrics.groupby("method")
        .agg(
            cases=("date", "count"),
            mean_expected_cost=("expected_cost", "mean"),
            mean_cvar95_cost=("cvar95_cost", "mean"),
            mean_worst_cost=("worst_cost", "mean"),
            mean_launch_weight_share=("mean_launch_weight_share", "mean"),
            mean_max_regret=("max_regret", "mean"),
            mean_solve_seconds=("solve_seconds", "mean"),
            mean_ferry_moves=("ferry_moves", "mean"),
        )
        .reset_index()
        .sort_values(["mean_max_regret", "mean_expected_cost"])
    )
    summary.to_csv(output_dir / "batch3_smoke_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
