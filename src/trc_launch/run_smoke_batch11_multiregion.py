from __future__ import annotations

from pathlib import Path

import pandas as pd

from .optimizers import LaunchOptimizer, compute_oracle_costs, evaluate_solution
from .run_smoke_batch7_filtered_regret import _filtered_instance
from .run_smoke_batch10_extended_sample import _instance_pressure
from .smoke_data import build_tail_launch_instance, load_bts_month


REGIONS: dict[str, tuple[str, ...]] = {
    "nyc": ("JFK", "LGA", "EWR"),
    "los_angeles": ("LAX", "BUR", "LGB", "ONT", "SNA"),
    "bay_area": ("SFO", "OAK", "SJC"),
    "dc_baltimore": ("DCA", "IAD", "BWI"),
    "chicago": ("ORD", "MDW"),
    "dallas": ("DFW", "DAL"),
    "houston": ("IAH", "HOU"),
    "south_florida": ("MIA", "FLL", "PBI"),
}

MONTHS = [202501, 202504, 202507, 202510]
CARRIERS = ["AA", "B6", "DL", "F9", "NK", "UA", "WN", "YX"]
DAY_CHOICES = [6, 13, 20, 27]


def _bts_csv(month: int) -> Path:
    paths = sorted(Path("data/raw_us_bts/extracted").glob(f"{month}/On_Time*.csv"))
    if not paths:
        raise FileNotFoundError(f"No BTS CSV found for {month}")
    return paths[0]


def _key(region_name: str, carrier: str, date: str) -> str:
    return f"{region_name}_{carrier}_{date}"


def _score_instance(instance, pressure: dict[str, object]) -> float:
    demand_airports = instance.flights["Origin"].nunique() if not instance.flights.empty else 0
    inventory_airports = instance.inventory["overnight_airport"].nunique() if not instance.inventory.empty else 0
    candidate_count = len(instance.candidates)
    return (
        float(pressure["total_deficit"]) * 120.0
        + float(pressure["max_airport_deficit"]) * 30.0
        + float(demand_airports + inventory_airports) * 8.0
        + float(pressure["flights"]) * 0.4
        + float(pressure["total_surplus"]) * 0.2
        - max(0.0, candidate_count - 4500.0) * 0.01
    )


def select_instances(max_instances: int = 18) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for month in MONTHS:
        month_df = load_bts_month(_bts_csv(month))
        month_start = pd.Timestamp(f"{str(month)[:4]}-{str(month)[4:]}-01")
        month_end = month_start + pd.offsets.MonthEnd(0)
        dates = [
            month_start + pd.Timedelta(days=day - 1)
            for day in DAY_CHOICES
            if day <= int(month_end.day)
        ]
        for region_name, airports in REGIONS.items():
            for carrier in CARRIERS:
                for date in dates:
                    instance = build_tail_launch_instance(
                        month_df,
                        date=str(date.date()),
                        carrier=carrier,
                        region_airports=airports,
                        scenario_count=4,
                        scenario_mode="stress",
                        seed=151,
                    )
                    if instance.flights.empty or instance.inventory.empty:
                        continue
                    pressure = _instance_pressure(instance)
                    demand_airports = int(instance.flights["Origin"].nunique())
                    inventory_airports = int(instance.inventory["overnight_airport"].nunique())
                    candidate_count = int(len(instance.candidates))
                    if pressure["flights"] < 8 or pressure["flights"] > 70:
                        continue
                    if pressure["inventory"] < pressure["flights"]:
                        continue
                    if pressure["total_deficit"] <= 0:
                        continue
                    if demand_airports < 2 or inventory_airports < 2:
                        continue
                    if candidate_count > 6000:
                        continue
                    rows.append(
                        {
                            "month": month,
                            "region": region_name,
                            "region_airports": "|".join(airports),
                            "carrier": carrier,
                            "date": str(date.date()),
                            "demand_airports": demand_airports,
                            "inventory_airports": inventory_airports,
                            "candidate_count_preview": candidate_count,
                            **pressure,
                            "score": _score_instance(instance, pressure),
                        }
                    )

    selected = pd.DataFrame(rows)
    if selected.empty:
        return selected
    selected = selected.sort_values("score", ascending=False)
    primary = selected.groupby("region").head(2)
    rest = selected[~selected.index.isin(primary.index)]
    out = pd.concat([primary, rest], ignore_index=True)
    out = out.drop_duplicates(["region", "carrier", "date"]).head(max_instances)
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
    region_name: str,
) -> None:
    metrics = evaluate_solution(instance, result, oracle_costs=oracle_costs)
    metrics["method"] = method_name
    metrics["instance_key"] = instance_key
    metrics["region"] = region_name
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
            regions=("region", "nunique"),
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
    output_dir = Path("results/trc_smoke/batch11_multiregion")
    data_dir = Path("data/trc_open_smoke/batch11_multiregion")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    selected_path = data_dir / "selected_instances.csv"
    if selected_path.exists():
        selected = pd.read_csv(selected_path)
    else:
        selected = select_instances(max_instances=18)
        selected.to_csv(selected_path, index=False)

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
        print(f"running {key}: flights={len(instance.flights)} candidates={original_count}")

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
                result = LaunchOptimizer(instance, time_limit=45.0).solve(method, oracle_costs=oracle_costs)
            else:
                result = LaunchOptimizer(instance, time_limit=45.0).solve(method)
            _add_metrics(
                rows,
                key,
                pressure,
                instance,
                result,
                oracle_costs,
                method_name,
                original_count,
                original_count,
                str(item.region),
            )

        filtered_instance, _ = _filtered_instance(instance, per_flight=20, per_aircraft=10)
        result = LaunchOptimizer(filtered_instance, time_limit=45.0).solve(
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
            "certificate_filtered_regret_k20",
            len(filtered_instance.candidates),
            original_count,
            str(item.region),
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "batch11_multiregion_metrics.csv", index=False)
    summary = _summary(metrics)
    summary.to_csv(output_dir / "batch11_multiregion_summary.csv", index=False)

    full = metrics[metrics["method"].eq("regret_portfolio_full")][
        ["instance_key", "expected_cost", "max_regret", "solve_seconds"]
    ].rename(
        columns={
            "expected_cost": "full_expected_cost",
            "max_regret": "full_max_regret",
            "solve_seconds": "full_solve_seconds",
        }
    )
    filtered = metrics[metrics["method"].eq("certificate_filtered_regret_k20")].merge(
        full,
        on="instance_key",
        how="left",
    )
    filtered["matches_full_cost"] = filtered["expected_cost"].round(8).eq(filtered["full_expected_cost"].round(8))
    filtered["matches_full_regret"] = filtered["max_regret"].round(8).eq(filtered["full_max_regret"].round(8))
    filtered["speedup_vs_full"] = filtered["full_solve_seconds"] / filtered["solve_seconds"].replace(0, pd.NA)
    filtered.to_csv(output_dir / "batch11_filtered_vs_full.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
