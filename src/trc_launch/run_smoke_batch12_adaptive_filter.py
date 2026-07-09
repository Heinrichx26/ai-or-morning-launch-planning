from __future__ import annotations

from pathlib import Path

import pandas as pd

from .optimizers import LaunchOptimizer, compute_oracle_costs, evaluate_solution
from .run_smoke_batch7_filtered_regret import _filtered_instance
from .run_smoke_batch10_extended_sample import _instance_pressure
from .run_smoke_batch11_multiregion import _bts_csv, _key
from .smoke_data import build_tail_launch_instance, load_bts_month


def _adaptive_width(instance, pressure: dict[str, object]) -> tuple[int, int, str]:
    flights = len(instance.flights)
    candidates = len(instance.candidates)
    inventory = int(instance.inventory["aircraft_count"].sum()) if not instance.inventory.empty else 0
    surplus = inventory - flights
    total_deficit = int(pressure["total_deficit"])
    max_deficit = int(pressure["max_airport_deficit"])
    if candidates >= 3500 or (flights >= 50 and surplus >= 20):
        return 36, 30, "dense"
    if max_deficit >= 6 or total_deficit >= 6 or flights >= 50 or candidates >= 3000:
        return 20, 10, "hard"
    if max_deficit >= 3 or total_deficit >= 4 or flights >= 35 or candidates >= 1400:
        return 16, 8, "medium"
    return 12, 6, "easy"


def _add_metrics(
    rows: list[dict[str, object]],
    instance_key: str,
    region_name: str,
    instance,
    result,
    oracle_costs: dict[int, float],
    method_name: str,
    candidate_count: int,
    original_candidate_count: int,
    width_label: str,
    per_flight: int,
) -> None:
    metrics = evaluate_solution(instance, result, oracle_costs=oracle_costs)
    metrics["method"] = method_name
    metrics["instance_key"] = instance_key
    metrics["region"] = region_name
    metrics["candidate_count"] = candidate_count
    metrics["candidate_retention"] = candidate_count / max(1, original_candidate_count)
    metrics["width_label"] = width_label
    metrics["per_flight"] = per_flight
    rows.append(metrics)


def _summary(metrics: pd.DataFrame) -> pd.DataFrame:
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
        )
        .reset_index()
        .sort_values(["mean_max_regret", "mean_candidate_retention"])
    )


def main() -> None:
    output_dir = Path("results/trc_smoke/batch12_adaptive_filter")
    data_dir = Path("data/trc_open_smoke/batch11_multiregion")
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(data_dir / "selected_instances.csv")
    full_metrics = pd.read_csv("results/trc_smoke/batch11_multiregion/batch11_multiregion_metrics.csv")
    full = full_metrics[full_metrics["method"].eq("regret_portfolio_full")][
        ["instance_key", "expected_cost", "max_regret"]
    ].rename(
        columns={
            "expected_cost": "full_expected_cost",
            "max_regret": "full_max_regret",
        }
    )

    month_cache: dict[int, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []
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
            seed=67,
        )
        pressure = _instance_pressure(instance)
        key = _key(str(item.region), str(item.carrier), str(item.date))
        original_count = len(instance.candidates)
        oracle_costs = compute_oracle_costs(instance, time_limit=7.0)

        for per_flight, per_aircraft in [(12, 6), (16, 8), (20, 10)]:
            filtered_instance, _ = _filtered_instance(instance, per_flight=per_flight, per_aircraft=per_aircraft)
            result = LaunchOptimizer(filtered_instance, time_limit=45.0).solve(
                "regret_portfolio",
                oracle_costs=oracle_costs,
            )
            _add_metrics(
                rows,
                key,
                str(item.region),
                instance,
                result,
                oracle_costs,
                f"fixed_k{per_flight}",
                len(filtered_instance.candidates),
                original_count,
                f"fixed_k{per_flight}",
                per_flight,
            )

        per_flight, per_aircraft, width_label = _adaptive_width(instance, pressure)
        filtered_instance, _ = _filtered_instance(instance, per_flight=per_flight, per_aircraft=per_aircraft)
        result = LaunchOptimizer(filtered_instance, time_limit=45.0).solve(
            "regret_portfolio",
            oracle_costs=oracle_costs,
        )
        _add_metrics(
            rows,
            key,
            str(item.region),
            instance,
            result,
            oracle_costs,
            "adaptive_certificate_regret",
            len(filtered_instance.candidates),
            original_count,
            width_label,
            per_flight,
        )

    metrics = pd.DataFrame(rows)
    metrics = metrics.merge(full, on="instance_key", how="left")
    metrics["matches_full_cost"] = metrics["expected_cost"].round(8).eq(metrics["full_expected_cost"].round(8))
    metrics["matches_full_regret"] = metrics["max_regret"].round(8).eq(metrics["full_max_regret"].round(8))
    metrics.to_csv(output_dir / "batch12_adaptive_metrics.csv", index=False)

    summary = _summary(metrics)
    match_summary = (
        metrics.groupby("method")
        .agg(
            cost_matches=("matches_full_cost", "sum"),
            regret_matches=("matches_full_regret", "sum"),
            mean_per_flight=("per_flight", "mean"),
        )
        .reset_index()
    )
    summary = summary.merge(match_summary, on="method", how="left")
    summary.to_csv(output_dir / "batch12_adaptive_summary.csv", index=False)

    width_counts = (
        metrics[metrics["method"].eq("adaptive_certificate_regret")]
        .groupby(["width_label", "per_flight"])
        .size()
        .rename("cases")
        .reset_index()
    )
    width_counts.to_csv(output_dir / "batch12_adaptive_width_counts.csv", index=False)
    print(summary.to_string(index=False))
    print("\nAdaptive width counts")
    print(width_counts.to_string(index=False))


if __name__ == "__main__":
    main()
