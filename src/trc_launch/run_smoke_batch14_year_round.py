from __future__ import annotations

from pathlib import Path

import pandas as pd

from .optimizers import LaunchOptimizer, compute_oracle_costs, evaluate_solution
from .run_smoke_batch7_filtered_regret import _filtered_instance
from .run_smoke_batch11_multiregion import CARRIERS, REGIONS, _bts_csv, _key
from .run_smoke_batch12_adaptive_filter import _adaptive_width
from .smoke_data import _build_flights, _build_tail_inventory, build_tail_launch_instance, load_bts_month


MONTHS = list(range(202501, 202513))
DAY_CHOICES = [6, 13, 20, 27]


def _candidate_dates(month: int) -> list[pd.Timestamp]:
    month_start = pd.Timestamp(f"{str(month)[:4]}-{str(month)[4:]}-01")
    month_end = month_start + pd.offsets.MonthEnd(0)
    return [
        month_start + pd.Timedelta(days=day - 1)
        for day in DAY_CHOICES
        if day <= int(month_end.day)
    ]


def _pressure_from_parts(flights: pd.DataFrame, tail_inventory: pd.DataFrame, airports: tuple[str, ...]) -> dict[str, int]:
    demand = flights.groupby("Origin").size().to_dict()
    inventory = tail_inventory.groupby("overnight_airport").size().to_dict()
    deficits = {
        airport: max(0, int(demand.get(airport, 0)) - int(inventory.get(airport, 0)))
        for airport in airports
    }
    surpluses = {
        airport: max(0, int(inventory.get(airport, 0)) - int(demand.get(airport, 0)))
        for airport in airports
    }
    return {
        "flights": int(len(flights)),
        "inventory": int(len(tail_inventory)),
        "total_deficit": int(sum(deficits.values())),
        "max_airport_deficit": int(max(deficits.values()) if deficits else 0),
        "total_surplus": int(sum(surpluses.values())),
        "demand_airports": int(flights["Origin"].nunique()) if not flights.empty else 0,
        "inventory_airports": int(tail_inventory["overnight_airport"].nunique()) if not tail_inventory.empty else 0,
    }


def _score(row: dict[str, object]) -> float:
    return (
        float(row["total_deficit"]) * 130.0
        + float(row["max_airport_deficit"]) * 35.0
        + float(row["demand_airports"]) * 12.0
        + float(row["inventory_airports"]) * 8.0
        + float(row["flights"]) * 0.35
        + float(row["total_surplus"]) * 0.20
        - max(0.0, float(row["candidate_count_preview"]) - 4200.0) * 0.015
    )


def select_instances(max_instances: int = 48) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for month in MONTHS:
        month_df = load_bts_month(_bts_csv(month))
        for region_name, airports in REGIONS.items():
            for carrier in CARRIERS:
                for date in _candidate_dates(month):
                    flights = _build_flights(month_df, date, carrier, airports)
                    if flights.empty:
                        continue
                    tail_inventory = _build_tail_inventory(month_df, date, carrier, airports)
                    if tail_inventory.empty:
                        continue
                    pressure = _pressure_from_parts(flights, tail_inventory, airports)
                    candidate_count = int(len(flights) * len(tail_inventory))
                    if pressure["flights"] < 8 or pressure["flights"] > 75:
                        continue
                    if pressure["inventory"] < pressure["flights"]:
                        continue
                    if pressure["total_deficit"] <= 0:
                        continue
                    if pressure["demand_airports"] < 2 or pressure["inventory_airports"] < 2:
                        continue
                    if candidate_count > 5200:
                        continue
                    row: dict[str, object] = {
                        "month": month,
                        "region": region_name,
                        "region_airports": "|".join(airports),
                        "carrier": carrier,
                        "date": str(date.date()),
                        "candidate_count_preview": candidate_count,
                        **pressure,
                    }
                    row["score"] = _score(row)
                    rows.append(row)

    selected = pd.DataFrame(rows)
    if selected.empty:
        return selected
    selected = selected.sort_values("score", ascending=False)
    by_region = selected.groupby("region").head(5)
    by_month = selected.groupby("month").head(2)
    rest = selected[~selected.index.isin(pd.concat([by_region, by_month]).index)]
    out = pd.concat([by_region, by_month, rest.sort_values("score", ascending=False)], ignore_index=True)
    out = out.drop_duplicates(["region", "carrier", "date"]).head(max_instances)
    return out.reset_index(drop=True)


def _add_metrics(
    rows: list[dict[str, object]],
    instance_key: str,
    region_name: str,
    month: int,
    pressure: dict[str, object],
    instance,
    result,
    oracle_costs: dict[int, float],
    method_name: str,
    candidate_count: int,
    original_candidate_count: int,
    width_label: str = "",
    per_flight: int = 0,
) -> None:
    metrics = evaluate_solution(instance, result, oracle_costs=oracle_costs)
    metrics["method"] = method_name
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
    output_dir = Path("results/trc_smoke/batch14_year_round")
    data_dir = Path("data/trc_open_smoke/batch14_year_round")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    selected_path = data_dir / "selected_instances.csv"
    if selected_path.exists():
        selected = pd.read_csv(selected_path)
    else:
        selected = select_instances(max_instances=48)
        selected.to_csv(selected_path, index=False)

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
        print(f"running {key}: month={month} flights={len(instance.flights)} candidates={original_count}")
        oracle_costs = compute_oracle_costs(instance, time_limit=8.0)

        full_methods = [
            ("active_scenario_saa", "active_scenario_saa"),
            ("saa_extensive", "saa_extensive"),
            ("cvar_extensive", "cvar_extensive"),
            ("mean_cvar95_extensive", "mean_cvar95_extensive"),
            ("minimax_extensive", "minimax_extensive"),
            ("regret_portfolio", "regret_portfolio_full"),
        ]
        for method, method_name in full_methods:
            if method == "regret_portfolio":
                result = LaunchOptimizer(instance, time_limit=50.0).solve(method, oracle_costs=oracle_costs)
            else:
                result = LaunchOptimizer(instance, time_limit=50.0).solve(method)
            _add_metrics(
                rows,
                key,
                str(item.region),
                month,
                pressure,
                instance,
                result,
                oracle_costs,
                method_name,
                original_count,
                original_count,
            )

        adaptive_pressure = {
            "total_deficit": int(item.total_deficit),
            "max_airport_deficit": int(item.max_airport_deficit),
        }
        per_flight, per_aircraft, width_label = _adaptive_width(instance, adaptive_pressure)
        filtered_instance, _ = _filtered_instance(instance, per_flight=per_flight, per_aircraft=per_aircraft)
        result = LaunchOptimizer(filtered_instance, time_limit=50.0).solve(
            "regret_portfolio",
            oracle_costs=oracle_costs,
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
            "adaptive_certificate_regret",
            len(filtered_instance.candidates),
            original_count,
            width_label,
            per_flight,
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "batch14_year_round_metrics.csv", index=False)
    summary = _summary(metrics)
    summary.to_csv(output_dir / "batch14_year_round_summary.csv", index=False)

    full = metrics[metrics["method"].eq("regret_portfolio_full")][
        ["instance_key", "expected_cost", "max_regret", "solve_seconds"]
    ].rename(
        columns={
            "expected_cost": "full_expected_cost",
            "max_regret": "full_max_regret",
            "solve_seconds": "full_solve_seconds",
        }
    )
    adaptive = metrics[metrics["method"].eq("adaptive_certificate_regret")].merge(
        full,
        on="instance_key",
        how="left",
    )
    adaptive["matches_full_cost"] = adaptive["expected_cost"].round(8).eq(adaptive["full_expected_cost"].round(8))
    adaptive["matches_full_regret"] = adaptive["max_regret"].round(8).eq(adaptive["full_max_regret"].round(8))
    adaptive["speedup_vs_full"] = adaptive["full_solve_seconds"] / adaptive["solve_seconds"].replace(0, pd.NA)
    adaptive.to_csv(output_dir / "batch14_adaptive_vs_full.csv", index=False)

    region_summary = (
        metrics.pivot_table(index="region", columns="method", values="max_regret", aggfunc="mean")
        .reset_index()
    )
    region_summary.to_csv(output_dir / "batch14_region_regret_summary.csv", index=False)

    width_counts = (
        adaptive.groupby(["width_label", "per_flight"])
        .size()
        .rename("cases")
        .reset_index()
    )
    width_counts.to_csv(output_dir / "batch14_adaptive_width_counts.csv", index=False)
    print(summary.to_string(index=False))
    print("\nAdaptive check")
    print(
        adaptive.agg(
            cost_matches=("matches_full_cost", "sum"),
            regret_matches=("matches_full_regret", "sum"),
            mean_speedup=("speedup_vs_full", "mean"),
            mean_retention=("candidate_retention", "mean"),
        ).to_string()
    )
    print("\nAdaptive width counts")
    print(width_counts.to_string(index=False))


if __name__ == "__main__":
    main()
