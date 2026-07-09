from __future__ import annotations

from pathlib import Path

import pandas as pd

from .optimizers import LaunchOptimizer, compute_oracle_costs, evaluate_solution
from .run_smoke_batch1 import DEFAULT_BTS_CSV
from .run_smoke_batch7_filtered_regret import _filtered_instance
from .smoke_data import build_tail_launch_instance, load_bts_month


CARRIERS = ["DL", "YX", "AA", "B6", "UA", "NK", "WN"]


def _key(carrier: str, date: str) -> str:
    return f"{carrier}_{date}"


def _instance_pressure(instance) -> dict[str, object]:
    demand = instance.flights.groupby("Origin").size().to_dict()
    inventory = dict(zip(instance.inventory["overnight_airport"], instance.inventory["aircraft_count"]))
    deficits = {
        airport: max(0, int(demand.get(airport, 0)) - int(inventory.get(airport, 0)))
        for airport in instance.region_airports
    }
    surpluses = {
        airport: max(0, int(inventory.get(airport, 0)) - int(demand.get(airport, 0)))
        for airport in instance.region_airports
    }
    return {
        "flights": int(len(instance.flights)),
        "inventory": int(instance.inventory["aircraft_count"].sum()) if not instance.inventory.empty else 0,
        "total_deficit": int(sum(deficits.values())),
        "max_airport_deficit": int(max(deficits.values()) if deficits else 0),
        "total_surplus": int(sum(surpluses.values())),
        "demand_by_airport": demand,
        "inventory_by_airport": inventory,
        "deficits": deficits,
    }


def select_instances(month: pd.DataFrame, max_instances: int = 20) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for carrier in CARRIERS:
        for date in pd.date_range("2025-01-03", "2025-01-31"):
            inst = build_tail_launch_instance(
                month,
                date=str(date.date()),
                carrier=carrier,
                scenario_count=4,
                scenario_mode="stress",
                seed=101,
            )
            if inst.flights.empty or inst.inventory.empty:
                continue
            pressure = _instance_pressure(inst)
            if pressure["flights"] < 8:
                continue
            if pressure["inventory"] < pressure["flights"]:
                continue
            if pressure["total_deficit"] <= 0:
                continue
            rows.append({"carrier": carrier, "date": str(date.date()), **pressure})

    selected = pd.DataFrame(rows)
    if selected.empty:
        return selected
    selected["score"] = (
        selected["total_deficit"] * 100
        + selected["max_airport_deficit"] * 20
        + selected["flights"] * 0.2
        + selected["total_surplus"] * 0.1
    )
    # Keep diversity: take strongest cases per carrier first, then fill globally.
    primary = selected.sort_values("score", ascending=False).groupby("carrier").head(4)
    rest = selected[~selected.index.isin(primary.index)]
    out = pd.concat([primary, rest.sort_values("score", ascending=False)], ignore_index=True)
    out = out.drop_duplicates(["carrier", "date"]).head(max_instances)
    return out.reset_index(drop=True)


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


def main() -> None:
    output_dir = Path("results/trc_smoke/batch10_extended_sample")
    data_dir = Path("data/trc_open_smoke/batch10_extended_sample")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    month = load_bts_month(DEFAULT_BTS_CSV)
    selected = select_instances(month, max_instances=20)
    selected.to_csv(data_dir / "selected_instances.csv", index=False)

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

        strong_methods = [
            ("saa_extensive", "saa_extensive"),
            ("cvar_extensive", "cvar_extensive"),
            ("regret_portfolio", "regret_portfolio_full"),
        ]
        for method, method_name in strong_methods:
            if method == "regret_portfolio":
                result = LaunchOptimizer(instance, time_limit=35.0).solve(method, oracle_costs=oracle_costs)
            else:
                result = LaunchOptimizer(instance, time_limit=35.0).solve(method)
            _add_metrics(rows, key, pressure, instance, result, oracle_costs, method_name, original_count, original_count)

        for per_flight, per_aircraft in [(16, 8), (20, 10), (24, 12)]:
            filtered_instance, _ = _filtered_instance(instance, per_flight=per_flight, per_aircraft=per_aircraft)
            result = LaunchOptimizer(filtered_instance, time_limit=35.0).solve(
                "regret_portfolio",
                oracle_costs=oracle_costs,
            )
            _add_metrics(
                rows,
                key,
                pressure,
                instance,
                result,
                oracle_costs,
                f"certificate_filtered_regret_k{per_flight}",
                len(filtered_instance.candidates),
                original_count,
            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "batch10_extended_metrics.csv", index=False)
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
    summary.to_csv(output_dir / "batch10_extended_summary.csv", index=False)

    full = metrics[metrics["method"].eq("regret_portfolio_full")][
        ["instance_key", "expected_cost", "max_regret", "solve_seconds"]
    ].rename(
        columns={
            "expected_cost": "full_expected_cost",
            "max_regret": "full_max_regret",
            "solve_seconds": "full_solve_seconds",
        }
    )
    filtered = metrics[metrics["method"].str.startswith("certificate_filtered_regret")].merge(
        full,
        on="instance_key",
        how="left",
    )
    filtered["matches_full_cost"] = filtered["expected_cost"].round(8).eq(filtered["full_expected_cost"].round(8))
    filtered["matches_full_regret"] = filtered["max_regret"].round(8).eq(filtered["full_max_regret"].round(8))
    filtered["speedup_vs_full"] = filtered["full_solve_seconds"] / filtered["solve_seconds"].replace(0, pd.NA)
    filtered.to_csv(output_dir / "batch10_filtered_vs_full.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
