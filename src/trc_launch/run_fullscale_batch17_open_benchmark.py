from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .distilled_policy import candidate_features, fit_distilled_model, make_labeled_frame
from .optimizers import LaunchOptimizer, compute_oracle_costs, evaluate_solution
from .run_smoke_batch7_filtered_regret import _filtered_instance
from .run_smoke_batch11_multiregion import _bts_csv, _key
from .run_smoke_batch12_adaptive_filter import _adaptive_width
from .run_smoke_batch14_year_round import select_instances
from .run_smoke_batch16_frontier_baselines import WIDTH_GRID
from .smoke_data import LaunchInstance, build_tail_launch_instance, load_bts_month, to_unit_service_instance
from .weather_asos import fetch_airport_date_weather


CORE_METHODS = [
    "regret_portfolio_full",
    "adaptive_certificate_regret_dense",
    "saa_extensive",
    "cvar_extensive",
    "mean_cvar95_extensive",
    "minimax_extensive",
    "active_scenario_saa",
]
FRONTIER_METHODS = [
    "ml_solution_reduction",
    "iterative_candidate_generation",
]
FIXED_METHODS = [
    "fixed_k12",
    "fixed_k16",
    "fixed_k20",
    "fixed_k36",
]
FIXED_WIDTHS = {
    "fixed_k12": (12, 6),
    "fixed_k16": (16, 8),
    "fixed_k20": (20, 10),
    "fixed_k36": (36, 30),
}


def _copy_with_candidates(instance: LaunchInstance, candidates: pd.DataFrame) -> LaunchInstance:
    return replace(instance, candidates=candidates.reset_index(drop=True))


def _load_selected(path: Path, max_instances: int, refresh: bool) -> pd.DataFrame:
    if path.exists() and not refresh:
        selected = pd.read_csv(path)
        if max_instances > 0 and len(selected) < max_instances:
            selected = select_instances(max_instances=max_instances)
            path.parent.mkdir(parents=True, exist_ok=True)
            selected.to_csv(path, index=False)
    else:
        selected = select_instances(max_instances=max_instances)
        path.parent.mkdir(parents=True, exist_ok=True)
        selected.to_csv(path, index=False)
    if max_instances > 0:
        selected = selected.head(max_instances).reset_index(drop=True)
    return selected


def _load_weather_features(item, args: argparse.Namespace) -> pd.DataFrame | None:
    if not args.use_weather:
        return None
    airports = str(item.region_airports).split("|")
    weather_dir = Path(args.weather_dir)
    features = fetch_airport_date_weather(
        airports=airports,
        date=str(item.date),
        output_dir=weather_dir,
        local_start_hour=args.weather_local_start_hour,
        local_end_hour=args.weather_local_end_hour,
        utc_offset_hours=args.weather_utc_offset,
        force=args.refresh_weather,
        pause_seconds=args.weather_pause_seconds,
        max_attempts=args.weather_max_attempts,
        request_timeout_seconds=args.weather_request_timeout_seconds,
    )
    if not features.empty:
        key = _key(str(item.region), str(item.carrier), str(item.date))
        feature_dir = Path(args.data_dir) / "weather_features"
        feature_dir.mkdir(parents=True, exist_ok=True)
        features.to_csv(feature_dir / f"weather_{key}.csv", index=False)
    return features


def _load_instance(
    item,
    month_cache: dict[int, pd.DataFrame],
    scenario_count: int,
    seed: int,
    weather_features: pd.DataFrame | None = None,
    value_mode: str = "distance_cost",
) -> LaunchInstance:
    month = int(item.month)
    if month not in month_cache:
        month_cache[month] = load_bts_month(_bts_csv(month))
    instance = build_tail_launch_instance(
        month_cache[month],
        date=str(item.date),
        carrier=str(item.carrier),
        region_airports=str(item.region_airports).split("|"),
        scenario_count=scenario_count,
        scenario_mode="stress",
        seed=seed,
        weather_features=weather_features,
    )
    if value_mode == "unit_service":
        return to_unit_service_instance(instance)
    if value_mode != "distance_cost":
        raise ValueError(f"unsupported value_mode: {value_mode}")
    return instance


def _meta_from_item(item) -> dict[str, object]:
    return {
        "region": str(item.region),
        "month": int(item.month),
        "carrier": str(item.carrier),
        "date": str(item.date),
        "total_deficit": int(item.total_deficit),
        "max_airport_deficit": int(item.max_airport_deficit),
    }


def _add_metrics(
    rows: list[dict[str, object]],
    method: str,
    key: str,
    meta: dict[str, object],
    instance: LaunchInstance,
    result,
    oracle_costs: dict[int, float],
    candidate_count: int,
    original_count: int,
    full_expected: float,
    full_regret: float,
    extra: dict[str, object] | None = None,
) -> None:
    metrics = evaluate_solution(instance, result, oracle_costs=oracle_costs)
    metrics["method"] = method
    metrics["instance_key"] = key
    metrics.update(meta)
    metrics["candidate_count"] = candidate_count
    metrics["candidate_retention"] = candidate_count / max(1, original_count)
    metrics["scenario_count"] = int(instance.scenarios["scenario_id"].nunique())
    metrics["flight_count"] = int(len(instance.flights))
    metrics["binary_var_count"] = int(candidate_count + metrics["scenario_count"] * len(instance.flights))
    metrics["full_expected_cost"] = full_expected
    metrics["full_max_regret"] = full_regret
    metrics["matches_full_cost"] = round(float(metrics["expected_cost"]), 8) == round(full_expected, 8)
    metrics["matches_full_regret"] = round(float(metrics["max_regret"]), 8) == round(full_regret, 8)
    if extra:
        metrics.update(extra)
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
            mean_max_regret=("max_regret", "mean"),
            worst_case_regret=("max_regret", "max"),
            mean_solve_seconds=("solve_seconds", "mean"),
            mean_candidate_count=("candidate_count", "mean"),
            mean_candidate_retention=("candidate_retention", "mean"),
            full_cost_matches=("matches_full_cost", "sum"),
            full_regret_matches=("matches_full_regret", "sum"),
        )
        .reset_index()
        .sort_values(["mean_max_regret", "mean_candidate_retention", "mean_solve_seconds"])
    )


def _ml_reduced_instance(instance: LaunchInstance, model, per_flight: int = 20, per_aircraft: int = 10) -> LaunchInstance:
    features = candidate_features(instance)
    if features.empty:
        return _copy_with_candidates(instance, instance.candidates.iloc[0:0].copy())
    probs = model.model.predict_proba(features[model.feature_columns].fillna(0.0))[:, 1]
    features = features.copy()
    features["ml_score"] = probs
    candidates = instance.candidates.reset_index(drop=True).copy()
    scored = candidates.merge(features[["candidate_id", "ml_score"]], on="candidate_id", how="left")
    scored["ml_score"] = scored["ml_score"].fillna(0.0)

    keep_ids: set[str] = set(scored[scored["ferry_cost"].eq(0)]["candidate_id"].astype(str).tolist())
    for _, group in scored.groupby("flight_id"):
        keep_ids.update(group.nlargest(per_flight, "ml_score")["candidate_id"].astype(str).tolist())
    if "aircraft_id" in scored.columns:
        for _, group in scored.groupby("aircraft_id"):
            keep_ids.update(group.nlargest(per_aircraft, "ml_score")["candidate_id"].astype(str).tolist())
    reduced = candidates[candidates["candidate_id"].astype(str).isin(keep_ids)].copy()
    return _copy_with_candidates(instance, reduced)


def _iterative_candidate_generation(
    instance: LaunchInstance,
    oracle_costs: dict[int, float],
    time_limit: float,
) -> tuple[object, LaunchInstance, str]:
    previous_regret = np.inf
    best_result = None
    best_instance = None
    history: list[str] = []
    stable_steps = 0
    for per_flight, per_aircraft in WIDTH_GRID:
        reduced, _ = _filtered_instance(instance, per_flight=per_flight, per_aircraft=per_aircraft)
        result = LaunchOptimizer(reduced, time_limit=time_limit).solve("regret_portfolio", oracle_costs=oracle_costs)
        metrics = evaluate_solution(instance, result, oracle_costs=oracle_costs)
        regret = float(metrics["max_regret"])
        history.append(f"{per_flight}/{per_aircraft}:{regret:.6f}")
        if best_result is None or regret < previous_regret - 1e-8:
            best_result = result
            best_instance = reduced
            previous_regret = regret
            stable_steps = 0
        else:
            stable_steps += 1
        if stable_steps >= 1 and per_flight >= 16:
            break
    assert best_result is not None and best_instance is not None
    best_result.method = "iterative_candidate_generation"
    return best_result, best_instance, ";".join(history)


def _method_list(args: argparse.Namespace) -> list[str]:
    if args.methods:
        return args.methods
    methods = list(CORE_METHODS)
    if args.include_frontier:
        methods.extend(FRONTIER_METHODS)
    if args.include_fixed:
        methods.extend(FIXED_METHODS)
    return methods


def _write_readme(path: Path, selected: pd.DataFrame, summary: pd.DataFrame, args: argparse.Namespace) -> None:
    regions = ", ".join(sorted(selected["region"].astype(str).unique()))
    months = ", ".join(str(x) for x in sorted(selected["month"].astype(int).unique()))
    best = summary.iloc[0].to_dict() if not summary.empty else {}
    text = f"""# Batch 17 Open Benchmark

## Purpose

Batch 17 is the parameterized benchmark entry for the TR-C revision. It uses open
Bureau of Transportation Statistics On-Time Performance data and supports both
smoke tests and the planned 300+ instance run.

## Run Settings

- Instances requested: {args.max_instances}
- Scenarios per instance: {args.scenario_count}
- Oracle time limit: {args.oracle_time_limit} seconds
- Method time limit: {args.time_limit} seconds
- Weather layer: {"IEM ASOS/METAR" if args.use_weather else "BTS-derived scenarios only"}
- Value mode: {args.value_mode}
- Regions represented: {regions}
- Months represented: {months}

## Result Assessment

The lowest mean maximum regret in this run is `{best.get("method", "n/a")}` with
mean maximum regret `{best.get("mean_max_regret", "n/a")}`. Use
`batch17_open_benchmark_summary.csv` as the smoke decision table before any
larger run.
"""
    path.write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    selected_path = data_dir / "selected_instances.csv"
    selected = _load_selected(selected_path, args.max_instances, args.refresh_selection)
    if selected.empty:
        raise RuntimeError("No eligible MLAP instances were selected.")

    methods = _method_list(args)
    rows: list[dict[str, object]] = []
    month_cache: dict[int, pd.DataFrame] = {}
    weather_cache: dict[str, pd.DataFrame | None] = {}
    instance_cache: dict[str, tuple[LaunchInstance, dict[str, object]]] = {}
    oracle_by_key: dict[str, dict[int, float]] = {}
    full_result_by_key = {}
    full_metrics_by_key: dict[str, dict[str, float]] = {}
    label_frames: dict[str, pd.DataFrame] = {}

    for item in selected.itertuples(index=False):
        meta = _meta_from_item(item)
        key = _key(str(meta["region"]), str(meta["carrier"]), str(meta["date"]))
        weather_features = _load_weather_features(item, args)
        weather_cache[key] = weather_features
        instance = _load_instance(
            item,
            month_cache,
            args.scenario_count,
            args.seed,
            weather_features=weather_features,
            value_mode=args.value_mode,
        )
        if instance.flights.empty or instance.inventory.empty or instance.candidates.empty:
            continue
        instance_cache[key] = (instance, meta)
        print(f"preparing {key}: flights={len(instance.flights)} candidates={len(instance.candidates)}", flush=True)
        oracle = compute_oracle_costs(instance, time_limit=args.oracle_time_limit)
        full = LaunchOptimizer(instance, time_limit=args.time_limit).solve("regret_portfolio", oracle_costs=oracle)
        full_metrics = evaluate_solution(instance, full, oracle_costs=oracle)
        oracle_by_key[key] = oracle
        full_result_by_key[key] = full
        full_metrics_by_key[key] = {
            "expected_cost": float(full_metrics["expected_cost"]),
            "max_regret": float(full_metrics["max_regret"]),
        }
        if "ml_solution_reduction" in methods:
            label_frame = make_labeled_frame(instance, full)
            label_frame["instance_key"] = key
            label_frames[key] = label_frame
            label_frame.to_csv(data_dir / f"teacher_labels_{key}.csv", index=False)

    for key, (instance, meta) in instance_cache.items():
        oracle = oracle_by_key[key]
        original_count = len(instance.candidates)
        full_expected = full_metrics_by_key[key]["expected_cost"]
        full_regret = full_metrics_by_key[key]["max_regret"]
        print(f"running methods for {key}", flush=True)

        if "regret_portfolio_full" in methods:
            _add_metrics(
                rows,
                "regret_portfolio_full",
                key,
                meta,
                instance,
                full_result_by_key[key],
                oracle,
                original_count,
                original_count,
                full_expected,
                full_regret,
            )

        if "adaptive_certificate_regret_dense" in methods:
            pressure = {
                "total_deficit": int(meta["total_deficit"]),
                "max_airport_deficit": int(meta["max_airport_deficit"]),
            }
            per_flight, per_aircraft, width_label = _adaptive_width(instance, pressure)
            adaptive_instance, _ = _filtered_instance(instance, per_flight=per_flight, per_aircraft=per_aircraft)
            adaptive = LaunchOptimizer(adaptive_instance, time_limit=args.time_limit).solve(
                "regret_portfolio",
                oracle_costs=oracle,
            )
            _add_metrics(
                rows,
                "adaptive_certificate_regret_dense",
                key,
                meta,
                instance,
                adaptive,
                oracle,
                len(adaptive_instance.candidates),
                original_count,
                full_expected,
                full_regret,
                {"width_label": width_label, "per_flight": per_flight, "per_aircraft": per_aircraft},
            )

        for method in ["saa_extensive", "cvar_extensive", "mean_cvar95_extensive", "minimax_extensive", "active_scenario_saa"]:
            if method not in methods:
                continue
            result = LaunchOptimizer(instance, time_limit=args.time_limit).solve(method)
            _add_metrics(
                rows,
                method,
                key,
                meta,
                instance,
                result,
                oracle,
                original_count,
                original_count,
                full_expected,
                full_regret,
            )

        for fixed_name, (per_flight, per_aircraft) in FIXED_WIDTHS.items():
            if fixed_name not in methods:
                continue
            fixed_instance, _ = _filtered_instance(instance, per_flight=per_flight, per_aircraft=per_aircraft)
            fixed = LaunchOptimizer(fixed_instance, time_limit=args.time_limit).solve(
                "regret_portfolio",
                oracle_costs=oracle,
            )
            _add_metrics(
                rows,
                fixed_name,
                key,
                meta,
                instance,
                fixed,
                oracle,
                len(fixed_instance.candidates),
                original_count,
                full_expected,
                full_regret,
                {"per_flight": per_flight, "per_aircraft": per_aircraft},
            )

        if "ml_solution_reduction" in methods:
            test_month = int(meta["month"])
            train_frames = [
                frame
                for train_key, frame in label_frames.items()
                if train_key != key and int(instance_cache[train_key][1]["month"]) != test_month
            ]
            if not train_frames:
                train_frames = [frame for train_key, frame in label_frames.items() if train_key != key]
            ml_model = fit_distilled_model(train_frames)
            ml_instance = _ml_reduced_instance(instance, ml_model, per_flight=20, per_aircraft=10)
            ml_result = LaunchOptimizer(ml_instance, time_limit=args.time_limit).solve("regret_portfolio", oracle_costs=oracle)
            _add_metrics(
                rows,
                "ml_solution_reduction",
                key,
                meta,
                instance,
                ml_result,
                oracle,
                len(ml_instance.candidates),
                original_count,
                full_expected,
                full_regret,
                {"train_split": "other_months"},
            )

        if "iterative_candidate_generation" in methods:
            iterative_result, iterative_instance, history = _iterative_candidate_generation(
                instance,
                oracle,
                time_limit=args.time_limit,
            )
            _add_metrics(
                rows,
                "iterative_candidate_generation",
                key,
                meta,
                instance,
                iterative_result,
                oracle,
                len(iterative_instance.candidates),
                original_count,
                full_expected,
                full_regret,
                {"iteration_history": history},
            )

        pd.DataFrame(rows).to_csv(output_dir / "batch17_open_benchmark_metrics_checkpoint.csv", index=False)

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise RuntimeError("No benchmark metrics were generated.")
    metrics.to_csv(output_dir / "batch17_open_benchmark_metrics.csv", index=False)
    summary = _summary(metrics)
    summary.to_csv(output_dir / "batch17_open_benchmark_summary.csv", index=False)
    region_summary = (
        metrics.pivot_table(index="region", columns="method", values="max_regret", aggfunc="mean")
        .reset_index()
    )
    region_summary.to_csv(output_dir / "batch17_region_regret_summary.csv", index=False)
    _write_readme(output_dir / "README.md", selected, summary, args)
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TR-C open-data MLAP benchmark.")
    parser.add_argument("--max-instances", type=int, default=20)
    parser.add_argument("--scenario-count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--time-limit", type=float, default=50.0)
    parser.add_argument("--oracle-time-limit", type=float, default=8.0)
    parser.add_argument("--include-frontier", action="store_true")
    parser.add_argument("--include-fixed", action="store_true")
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--refresh-selection", action="store_true")
    parser.add_argument("--use-weather", action="store_true")
    parser.add_argument("--refresh-weather", action="store_true")
    parser.add_argument("--weather-dir", default="data/weather_asos")
    parser.add_argument("--weather-utc-offset", type=int, default=None)
    parser.add_argument("--weather-local-start-hour", type=int, default=4)
    parser.add_argument("--weather-local-end-hour", type=int, default=12)
    parser.add_argument("--weather-pause-seconds", type=float, default=2.0)
    parser.add_argument("--weather-max-attempts", type=int, default=5)
    parser.add_argument("--weather-request-timeout-seconds", type=float, default=45.0)
    parser.add_argument(
        "--value-mode",
        choices=["distance_cost", "unit_service"],
        default="distance_cost",
        help="distance_cost uses BTS distance weights and regional repositioning penalties; "
        "unit_service uses one unit per flight and zero edge cost for a no-service-value audit.",
    )
    parser.add_argument("--data-dir", default="data/trc_open_smoke/batch17_open_benchmark")
    parser.add_argument("--output-dir", default="results/trc_smoke/batch17_open_benchmark")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
