from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .optimizers import LaunchOptimizer, SolveResult, compute_oracle_costs, evaluate_solution
from .run_fullscale_batch17_open_benchmark import _bts_csv, _key, _load_instance
from .smoke_data import LaunchInstance


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS = (
    ROOT
    / "results"
    / "trc_smoke"
    / "batch20_weather_full360_fullquality"
    / "batch20_metrics_with_rccc_audited.csv"
)
DEFAULT_SELECTED = ROOT / "data" / "trc_open_smoke" / "batch20_weather_full360_fullquality" / "selected_instances.csv"
DEFAULT_WEATHER = ROOT / "data" / "trc_open_smoke" / "batch20_weather_full360_fullquality" / "weather_features"
DEFAULT_OUTPUT = ROOT / "results" / "trc_smoke" / "batch35_environment"
DEFAULT_GENERATED = ROOT / "article" / "trc_elsarticle" / "generated"

# ICAO Doc 9889 / EEA 1.A.3.a default taxi-out time for jets.
TAXI_OUT_MINUTES = 19.0
# Extra surface taxi minutes for a cross-airport substitution.
FERRY_TAXI_MINUTES = 8.0
# Representative narrow-body idle/taxi fuel flow (kg/s), CFM56-class ICAO EEDB idle.
TAXI_FUEL_KG_PER_S = 0.11
# ICAO Doc 9889 CO2 emission index for Jet A-1.
CO2_KG_PER_KG_FUEL = 3.16

TAXI_FUEL_KG_PER_MIN = TAXI_FUEL_KG_PER_S * 60.0

CORE_METHODS = [
    "rccc_audited",
    "regret_portfolio_full",
    "saa_extensive",
    "cvar_extensive",
    "minimax_extensive",
    "active_scenario_saa",
]

METHOD_LABELS = {
    "rccc_audited": "LG-RCCC / RCCC audited",
    "adaptive_certificate_regret_dense": "Certificate graph",
    "regret_portfolio_full": "Full regret reference",
    "saa_extensive": "SAA",
    "cvar_extensive": "CVaR",
    "minimax_extensive": "Minimax",
    "active_scenario_saa": "Active-scenario SAA",
}

SMOKE_KEYS = [
    "los_angeles_AA_2025-01-08",
    "nyc_B6_2025-12-27",
    "los_angeles_UA_2025-05-27",
]


def fuel_from_minutes(minutes: float) -> float:
    return float(minutes) * TAXI_FUEL_KG_PER_MIN


def co2_from_fuel(fuel_kg: float) -> float:
    return float(fuel_kg) * CO2_KG_PER_KG_FUEL


def convert_metrics_row(row: pd.Series) -> dict[str, float]:
    expected_cost = float(row["expected_cost"])
    ferry_cost = float(row.get("ferry_cost", 0.0) or 0.0)
    launch_share = float(row["mean_launch_weight_share"])
    assigned_share = float(row.get("assigned_weight_share", np.nan))
    flight_count = float(row["flight_count"])
    assigned_flights = float(row.get("assigned_flights", np.nan))
    ferry_moves = float(row.get("ferry_moves", 0.0) or 0.0)
    mean_regret = float(row.get("mean_regret", np.nan))
    max_regret = float(row.get("max_regret", np.nan))
    min_launch_share = float(row.get("min_launch_weight_share", np.nan))

    residual = max(1.0e-9, 1.0 - launch_share)
    total_weight = (expected_cost - ferry_cost) / residual
    mean_weight = total_weight / max(1.0, flight_count)
    launched_weight = launch_share * total_weight
    assigned_weight = assigned_share * total_weight if np.isfinite(assigned_share) else launched_weight
    hold_weight = max(0.0, assigned_weight - launched_weight)
    unlaunched_weight = max(0.0, total_weight - launched_weight)

    mean_launched_flights = launched_weight / max(1.0e-9, mean_weight)
    mean_hold_flights = hold_weight / max(1.0e-9, mean_weight)
    mean_unlaunched_flights = unlaunched_weight / max(1.0e-9, mean_weight)
    worst_unlaunched_flights = (1.0 - min_launch_share) * flight_count if np.isfinite(min_launch_share) else np.nan
    avoidable_flights_mean = mean_regret / max(1.0e-9, mean_weight) if np.isfinite(mean_regret) else np.nan
    avoidable_flights_worst = max_regret / max(1.0e-9, mean_weight) if np.isfinite(max_regret) else np.nan

    hold_minutes = mean_hold_flights * TAXI_OUT_MINUTES
    unlaunched_minutes = mean_unlaunched_flights * TAXI_OUT_MINUTES
    ferry_minutes = ferry_moves * FERRY_TAXI_MINUTES
    avoidable_minutes = avoidable_flights_mean * TAXI_OUT_MINUTES if np.isfinite(avoidable_flights_mean) else np.nan
    avoidable_worst_minutes = (
        avoidable_flights_worst * TAXI_OUT_MINUTES if np.isfinite(avoidable_flights_worst) else np.nan
    )

    hold_fuel = fuel_from_minutes(hold_minutes)
    unlaunched_fuel = fuel_from_minutes(unlaunched_minutes)
    ferry_fuel = fuel_from_minutes(ferry_minutes)
    ground_fuel = hold_fuel + ferry_fuel
    avoidable_fuel = fuel_from_minutes(avoidable_minutes) if np.isfinite(avoidable_minutes) else np.nan
    avoidable_worst_fuel = fuel_from_minutes(avoidable_worst_minutes) if np.isfinite(avoidable_worst_minutes) else np.nan

    return {
        "total_weight": total_weight,
        "mean_flight_weight": mean_weight,
        "mean_launched_flights": mean_launched_flights,
        "mean_hold_flights": mean_hold_flights,
        "mean_unlaunched_flights": mean_unlaunched_flights,
        "worst_unlaunched_flights": worst_unlaunched_flights,
        "avoidable_unlaunched_flights_mean": avoidable_flights_mean,
        "avoidable_unlaunched_flights_worst": avoidable_flights_worst,
        "mean_hold_minutes": hold_minutes,
        "mean_unlaunched_taxi_minutes": unlaunched_minutes,
        "mean_ferry_taxi_minutes": ferry_minutes,
        "avoidable_taxi_minutes_mean": avoidable_minutes,
        "avoidable_taxi_minutes_worst": avoidable_worst_minutes,
        "hold_fuel_kg": hold_fuel,
        "unlaunched_taxi_fuel_kg": unlaunched_fuel,
        "ferry_fuel_kg": ferry_fuel,
        "ground_fuel_kg": ground_fuel,
        "avoidable_fuel_kg_mean": avoidable_fuel,
        "avoidable_fuel_kg_worst": avoidable_worst_fuel,
        "ground_co2_kg": co2_from_fuel(ground_fuel),
        "unlaunched_taxi_co2_kg": co2_from_fuel(unlaunched_fuel),
        "avoidable_co2_kg_mean": co2_from_fuel(avoidable_fuel) if np.isfinite(avoidable_fuel) else np.nan,
        "avoidable_co2_kg_worst": co2_from_fuel(avoidable_worst_fuel) if np.isfinite(avoidable_worst_fuel) else np.nan,
        "assigned_flights": assigned_flights,
        "ferry_moves": ferry_moves,
        "flight_count": flight_count,
    }


def convert_metrics_frame(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in metrics.iterrows():
        converted = convert_metrics_row(row)
        converted.update(
            {
                "instance_key": row.get("instance_key", ""),
                "region": row.get("region", ""),
                "carrier": row.get("carrier", ""),
                "date": row.get("date", ""),
                "method": row.get("method", ""),
                "max_regret": row.get("max_regret", np.nan),
                "mean_regret": row.get("mean_regret", np.nan),
                "mean_launch_weight_share": row.get("mean_launch_weight_share", np.nan),
                "candidate_retention": row.get("candidate_retention", np.nan),
                "full_regret_matches": row.get("matches_full_regret", row.get("full_regret_matches", np.nan)),
            }
        )
        rows.append(converted)
    return pd.DataFrame(rows)


def summarize_environment(converted: pd.DataFrame) -> pd.DataFrame:
    methods = list(CORE_METHODS)
    if "rccc_audited" not in set(converted["method"]) and "adaptive_certificate_regret_dense" in set(converted["method"]):
        methods = ["adaptive_certificate_regret_dense" if m == "rccc_audited" else m for m in methods]
    keep = converted[converted["method"].isin(methods)].copy()
    grouped = (
        keep.groupby("method", as_index=False)
        .agg(
            cases=("instance_key", "nunique"),
            mean_unlaunched_flights=("mean_unlaunched_flights", "mean"),
            mean_hold_flights=("mean_hold_flights", "mean"),
            mean_hold_minutes=("mean_hold_minutes", "mean"),
            mean_unlaunched_taxi_minutes=("mean_unlaunched_taxi_minutes", "mean"),
            mean_ferry_taxi_minutes=("mean_ferry_taxi_minutes", "mean"),
            mean_ground_fuel_kg=("ground_fuel_kg", "mean"),
            mean_ground_co2_kg=("ground_co2_kg", "mean"),
            mean_unlaunched_taxi_co2_kg=("unlaunched_taxi_co2_kg", "mean"),
            mean_avoidable_flights=("avoidable_unlaunched_flights_mean", "mean"),
            mean_worst_avoidable_flights=("avoidable_unlaunched_flights_worst", "mean"),
            worst_avoidable_flights=("avoidable_unlaunched_flights_worst", "max"),
            mean_avoidable_co2_kg=("avoidable_co2_kg_mean", "mean"),
            mean_worst_avoidable_co2_kg=("avoidable_co2_kg_worst", "mean"),
            worst_avoidable_co2_kg=("avoidable_co2_kg_worst", "max"),
            mean_max_regret=("max_regret", "mean"),
            total_unlaunched_taxi_co2_kg=("unlaunched_taxi_co2_kg", "sum"),
            total_worst_avoidable_co2_kg=("avoidable_co2_kg_worst", "sum"),
        )
        .copy()
    )
    order = {method: i for i, method in enumerate(CORE_METHODS)}
    grouped["order"] = grouped["method"].map(order)
    return grouped.sort_values("order").drop(columns=["order"])


def _launch_counts(instance: LaunchInstance, selected: pd.DataFrame) -> dict[str, float]:
    flights = instance.flights.set_index("flight_id")
    selected_ids = set(selected["flight_id"].tolist()) if not selected.empty else set()
    assigned = flights.loc[list(selected_ids)] if selected_ids else flights.iloc[0:0]
    launched_flights: list[float] = []
    hold_flights: list[float] = []
    unlaunched_flights: list[float] = []
    launched_weight: list[float] = []
    for _, caps in instance.scenarios.groupby("scenario_id"):
        launched = 0
        hold = 0
        weight = 0.0
        for block, cap_row in caps.groupby("block_id"):
            cap = int(cap_row["capacity"].iloc[0])
            block_flights = assigned[assigned["block_id"].eq(block)].sort_values("weight", ascending=False)
            take = block_flights.head(max(0, cap))
            launched += int(len(take))
            hold += int(max(0, len(block_flights) - len(take)))
            weight += float(take["weight"].sum()) if not take.empty else 0.0
        launched_flights.append(launched)
        hold_flights.append(hold)
        unlaunched_flights.append(int(len(flights) - launched))
        launched_weight.append(weight)
    ferry_moves = int((selected["ferry_cost"] > 0).sum()) if not selected.empty else 0
    hold_minutes = float(np.mean(hold_flights)) * TAXI_OUT_MINUTES
    unlaunched_minutes = float(np.mean(unlaunched_flights)) * TAXI_OUT_MINUTES
    ferry_minutes = ferry_moves * FERRY_TAXI_MINUTES
    hold_fuel = fuel_from_minutes(hold_minutes)
    unlaunched_fuel = fuel_from_minutes(unlaunched_minutes)
    ferry_fuel = fuel_from_minutes(ferry_minutes)
    return {
        "mean_launched_flights": float(np.mean(launched_flights)) if launched_flights else 0.0,
        "mean_hold_flights": float(np.mean(hold_flights)) if hold_flights else 0.0,
        "mean_unlaunched_flights": float(np.mean(unlaunched_flights)) if unlaunched_flights else 0.0,
        "worst_unlaunched_flights": float(np.max(unlaunched_flights)) if unlaunched_flights else 0.0,
        "mean_hold_minutes": hold_minutes,
        "mean_unlaunched_taxi_minutes": unlaunched_minutes,
        "mean_ferry_taxi_minutes": ferry_minutes,
        "hold_fuel_kg": hold_fuel,
        "unlaunched_taxi_fuel_kg": unlaunched_fuel,
        "ferry_fuel_kg": ferry_fuel,
        "ground_fuel_kg": hold_fuel + ferry_fuel,
        "ground_co2_kg": co2_from_fuel(hold_fuel + ferry_fuel),
        "unlaunched_taxi_co2_kg": co2_from_fuel(unlaunched_fuel),
        "mean_launched_weight": float(np.mean(launched_weight)) if launched_weight else 0.0,
        "ferry_moves": float(ferry_moves),
        "flight_count": float(len(flights)),
    }


def evaluate_environment(
    instance: LaunchInstance,
    result: SolveResult,
    oracle_costs: dict[int, float] | None = None,
    oracle_launch_weight: dict[int, float] | None = None,
) -> dict[str, float | str]:
    base = evaluate_solution(instance, result, oracle_costs=oracle_costs)
    env = _launch_counts(instance, result.selected)
    mean_weight = float(instance.total_weight) / max(1, len(instance.flights))
    mean_regret = float(base.get("mean_regret", np.nan))
    max_regret = float(base.get("max_regret", np.nan))
    avoidable_mean = mean_regret / max(1.0e-9, mean_weight) if np.isfinite(mean_regret) else np.nan
    avoidable_worst = max_regret / max(1.0e-9, mean_weight) if np.isfinite(max_regret) else np.nan
    if oracle_launch_weight:
        assigned_ids = set(result.selected["flight_id"].tolist()) if not result.selected.empty else set()
        flights = instance.flights.set_index("flight_id")
        assigned = flights.loc[list(assigned_ids)] if assigned_ids else flights.iloc[0:0]
        gaps: list[float] = []
        for sid, caps in instance.scenarios.groupby("scenario_id"):
            launched_weight = 0.0
            for block, cap_row in caps.groupby("block_id"):
                cap = int(cap_row["capacity"].iloc[0])
                block_flights = assigned[assigned["block_id"].eq(block)].sort_values("weight", ascending=False)
                if cap > 0 and not block_flights.empty:
                    launched_weight += float(block_flights.head(cap)["weight"].sum())
            oracle_w = float(oracle_launch_weight.get(int(sid), launched_weight))
            gaps.append(max(0.0, oracle_w - launched_weight) / max(1.0e-9, mean_weight))
        avoidable_mean = float(np.mean(gaps)) if gaps else avoidable_mean
        avoidable_worst = float(np.max(gaps)) if gaps else avoidable_worst
    env.update(
        {
            "avoidable_unlaunched_flights_mean": avoidable_mean,
            "avoidable_unlaunched_flights_worst": avoidable_worst,
            "avoidable_taxi_minutes_mean": avoidable_mean * TAXI_OUT_MINUTES if np.isfinite(avoidable_mean) else np.nan,
            "avoidable_taxi_minutes_worst": avoidable_worst * TAXI_OUT_MINUTES if np.isfinite(avoidable_worst) else np.nan,
            "avoidable_fuel_kg_mean": fuel_from_minutes(avoidable_mean * TAXI_OUT_MINUTES)
            if np.isfinite(avoidable_mean)
            else np.nan,
            "avoidable_co2_kg_mean": co2_from_fuel(fuel_from_minutes(avoidable_mean * TAXI_OUT_MINUTES))
            if np.isfinite(avoidable_mean)
            else np.nan,
            "avoidable_co2_kg_worst": co2_from_fuel(fuel_from_minutes(avoidable_worst * TAXI_OUT_MINUTES))
            if np.isfinite(avoidable_worst)
            else np.nan,
            "method": result.method,
            "date": instance.date,
            "carrier": instance.carrier,
        }
    )
    env.update({k: base[k] for k in ("max_regret", "mean_regret", "expected_cost", "status", "solve_seconds")})
    return env


def _oracle_launch_weight(instance: LaunchInstance, time_limit: float = 10.0) -> dict[int, float]:
    weights: dict[int, float] = {}
    for sid, scenario in instance.scenarios.groupby("scenario_id"):
        single = LaunchInstance(
            date=instance.date,
            carrier=instance.carrier,
            region_airports=instance.region_airports,
            flights=instance.flights,
            inventory=instance.inventory,
            candidates=instance.candidates,
            scenarios=scenario.copy(),
            total_weight=instance.total_weight,
        )
        res = LaunchOptimizer(single, time_limit=time_limit).solve("saa_extensive")
        counts = _launch_counts(single, res.selected)
        weights[int(sid)] = float(counts["mean_launched_weight"])
    return weights


def convert_existing_benchmark(
    metrics_path: Path = DEFAULT_METRICS,
    output_dir: Path = DEFAULT_OUTPUT,
    generated_dir: Path = DEFAULT_GENERATED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(metrics_path)
    converted = convert_metrics_frame(metrics)
    summary = summarize_environment(converted)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    converted.to_csv(output_dir / "environment_instance_metrics.csv", index=False)
    summary.to_csv(output_dir / "environment_method_summary.csv", index=False)
    converted.to_csv(generated_dir / "environment_instance_metrics.csv", index=False)
    summary.to_csv(generated_dir / "environment_method_summary.csv", index=False)
    return converted, summary


def run_smoke(
    selected_path: Path = DEFAULT_SELECTED,
    weather_dir: Path = DEFAULT_WEATHER,
    output_dir: Path = DEFAULT_OUTPUT,
    keys: list[str] | None = None,
    scenario_count: int = 30,
    seed: int = 2025,
    time_limit: float = 20.0,
) -> pd.DataFrame:
    selected = pd.read_csv(selected_path)
    selected["instance_key"] = selected.apply(
        lambda row: _key(str(row["region"]), str(row["carrier"]), str(row["date"])),
        axis=1,
    )
    keys = keys or SMOKE_KEYS
    subset = selected[selected["instance_key"].isin(keys)].copy()
    if subset.empty:
        raise ValueError(f"none of the smoke keys were found: {keys}")

    month_cache: dict[int, pd.DataFrame] = {}
    rows: list[dict[str, float | str]] = []
    methods = ["regret_portfolio_full", "saa_extensive", "cvar_extensive", "minimax_extensive"]
    args = argparse.Namespace()

    for item in subset.itertuples(index=False):
        key = str(item.instance_key)
        weather_path = weather_dir / f"weather_{key}.csv"
        weather = pd.read_csv(weather_path) if weather_path.exists() else None
        instance = _load_instance(
            item,
            month_cache,
            scenario_count=scenario_count,
            seed=seed,
            weather_features=weather,
        )
        oracle = compute_oracle_costs(instance, time_limit=time_limit)
        oracle_launch = _oracle_launch_weight(instance, time_limit=time_limit)
        for method in methods:
            opt = LaunchOptimizer(instance, time_limit=time_limit)
            result = opt.solve(method if method != "regret_portfolio_full" else "regret_portfolio", oracle_costs=oracle)
            if method == "regret_portfolio_full":
                result.method = "regret_portfolio_full"
            env = evaluate_environment(instance, result, oracle_costs=oracle, oracle_launch_weight=oracle_launch)
            env["instance_key"] = key
            env["region"] = str(item.region)
            rows.append(env)
            print(
                f"{key} {method}: unlaunched={env['mean_unlaunched_flights']:.2f} "
                f"avoidable={env['avoidable_unlaunched_flights_mean']:.2f} "
                f"co2={env['avoidable_co2_kg_mean']:.1f}",
                flush=True,
            )
        _ = args

    out = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_dir / "environment_smoke_exact.csv", index=False)
    return out


def plot_environment_summary(summary: pd.DataFrame, fig_dir: Path | None = None) -> None:
    import matplotlib.pyplot as plt

    fig_dir = fig_dir or (ROOT / "article" / "trc_elsarticle" / "figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    order = [m for m in CORE_METHODS if m in set(summary["method"])]
    frame = summary.set_index("method").loc[order].reset_index()
    labels = [METHOD_LABELS.get(m, m) for m in frame["method"]]
    colors = ["#0B6E69", "#4A4A4A", "#9B5DE5", "#F15BB5", "#F77F00", "#D62828"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    axes[0].bar(labels, frame["mean_worst_avoidable_flights"], color=colors[: len(frame)], alpha=0.9)
    axes[0].set_ylabel("Mean worst-scenario avoidable unlaunched flights")
    axes[0].set_title("Disaster-scenario avoidable unlaunched service")
    axes[0].tick_params(axis="x", labelrotation=25, labelsize=7)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, frame["mean_worst_avoidable_co2_kg"], color=colors[: len(frame)], alpha=0.9)
    axes[1].set_ylabel("Mean worst-scenario avoidable CO$_2$ (kg)")
    axes[1].set_title("Disaster-scenario avoidable taxi CO$_2$")
    axes[1].tick_params(axis="x", labelrotation=25, labelsize=7)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig9_environment.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "fig9_environment.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert launch assignments to fuel and CO2 using public LTO factors.")
    parser.add_argument("--mode", choices=["convert", "smoke", "both"], default="convert")
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--generated-dir", default=str(DEFAULT_GENERATED))
    parser.add_argument("--max-smoke", type=int, default=3)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    generated_dir = Path(args.generated_dir)
    if args.mode in {"convert", "both"}:
        converted, summary = convert_existing_benchmark(
            metrics_path=Path(args.metrics),
            output_dir=output_dir,
            generated_dir=generated_dir,
        )
        plot_environment_summary(summary)
        print(summary.to_string(index=False))
        print(f"converted {len(converted)} method-instance rows")
    if args.mode in {"smoke", "both"}:
        smoke = run_smoke(output_dir=output_dir, keys=SMOKE_KEYS[: args.max_smoke])
        print(smoke.to_string(index=False))


if __name__ == "__main__":
    main()
