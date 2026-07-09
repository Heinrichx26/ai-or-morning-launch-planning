from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .optimizers import LaunchOptimizer, compute_oracle_costs, evaluate_solution
from .smoke_data import build_launch_instance, load_bts_month, write_instance_audit


DEFAULT_BTS_CSV = Path(
    "data/raw_us_bts/extracted/202501/"
    "On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2025_1.csv"
)


def run_batch1(
    bts_csv: Path = DEFAULT_BTS_CSV,
    output_dir: Path = Path("results/trc_smoke/batch1"),
    carrier: str = "DL",
    dates: list[str] | None = None,
    scenarios: int = 12,
    time_limit: float = 12.0,
    scenario_mode: str = "basic",
    include_certificate: bool = False,
) -> pd.DataFrame:
    if dates is None:
        dates = ["2025-01-06", "2025-01-07", "2025-01-08"]

    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path("data/trc_open_smoke/batch1")
    data_dir.mkdir(parents=True, exist_ok=True)

    month = load_bts_month(bts_csv)
    methods = [
        "launch_value",
        "saa_extensive",
        "predict_opt",
        "cvar_extensive",
        "regret_portfolio",
        "robust_quantile",
    ]
    if include_certificate:
        methods.insert(1, "certificate_value")
    rows: list[dict[str, object]] = []

    for date in dates:
        instance = build_launch_instance(
            month,
            date=date,
            carrier=carrier,
            scenario_count=scenarios,
            scenario_mode=scenario_mode,
        )
        write_instance_audit(instance, data_dir)
        instance.flights.to_csv(data_dir / f"flights_{carrier}_{date}.csv", index=False)
        instance.inventory.to_csv(data_dir / f"inventory_{carrier}_{date}.csv", index=False)
        instance.candidates.to_csv(data_dir / f"candidates_{carrier}_{date}.csv", index=False)
        instance.scenarios.to_csv(data_dir / f"scenarios_{carrier}_{date}.csv", index=False)

        oracle_costs = compute_oracle_costs(instance, time_limit=max(4.0, time_limit / 2))
        for method in methods:
            optimizer = LaunchOptimizer(instance, time_limit=time_limit)
            if method == "regret_portfolio":
                result = optimizer.solve(method, oracle_costs=oracle_costs)
            else:
                result = optimizer.solve(method)
            metrics = evaluate_solution(instance, result, oracle_costs=oracle_costs)
            rows.append(metrics)
            if not result.selected.empty:
                result.selected.to_csv(output_dir / f"selected_{method}_{carrier}_{date}.csv", index=False)

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "batch1_smoke_metrics.csv", index=False)
    summary = (
        out.groupby("method")
        .agg(
            cases=("date", "count"),
            mean_expected_cost=("expected_cost", "mean"),
            mean_cvar95_cost=("cvar95_cost", "mean"),
            mean_worst_cost=("worst_cost", "mean"),
            mean_launch_weight_share=("mean_launch_weight_share", "mean"),
            min_launch_weight_share=("min_launch_weight_share", "mean"),
            mean_max_regret=("max_regret", "mean"),
            mean_solve_seconds=("solve_seconds", "mean"),
            mean_ferry_moves=("ferry_moves", "mean"),
        )
        .reset_index()
        .sort_values(["mean_expected_cost", "mean_cvar95_cost"])
    )
    summary.to_csv(output_dir / "batch1_smoke_summary.csv", index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier", default="DL")
    parser.add_argument("--dates", nargs="*", default=["2025-01-06", "2025-01-07", "2025-01-08"])
    parser.add_argument("--scenarios", type=int, default=12)
    parser.add_argument("--time-limit", type=float, default=12.0)
    parser.add_argument("--output-dir", default="results/trc_smoke/batch1")
    parser.add_argument("--scenario-mode", default="basic", choices=["basic", "stress"])
    parser.add_argument("--include-certificate", action="store_true")
    args = parser.parse_args()

    out = run_batch1(
        carrier=args.carrier,
        dates=args.dates,
        scenarios=args.scenarios,
        time_limit=args.time_limit,
        scenario_mode=args.scenario_mode,
        include_certificate=args.include_certificate,
        output_dir=Path(args.output_dir),
    )
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
