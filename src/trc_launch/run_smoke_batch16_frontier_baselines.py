from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .distilled_policy import fit_distilled_model, make_labeled_frame, candidate_features
from .optimizers import LaunchOptimizer, compute_oracle_costs, evaluate_solution
from .run_smoke_batch7_filtered_regret import _filtered_instance
from .run_smoke_batch11_multiregion import _bts_csv, _key
from .run_smoke_batch12_adaptive_filter import _adaptive_width
from .smoke_data import LaunchInstance, build_tail_launch_instance, load_bts_month


WIDTH_GRID = [(12, 6), (16, 8), (20, 10), (36, 30)]


def _copy_with_candidates(instance: LaunchInstance, candidates: pd.DataFrame) -> LaunchInstance:
    return replace(instance, candidates=candidates.reset_index(drop=True))


def _select_smoke_cases(selected: pd.DataFrame, max_cases: int = 12) -> pd.DataFrame:
    selected = selected.copy()
    selected["score"] = (
        selected["total_deficit"] * 130
        + selected["max_airport_deficit"] * 35
        + selected["flights"] * 0.35
        + selected["candidate_count_preview"] * 0.002
    )
    regional = selected.sort_values("score", ascending=False).groupby("region").head(1)
    rest = selected[~selected.index.isin(regional.index)].sort_values("score", ascending=False)
    return pd.concat([regional, rest], ignore_index=True).head(max_cases).reset_index(drop=True)


def _load_instances(selected: pd.DataFrame) -> dict[str, tuple[LaunchInstance, dict[str, object]]]:
    month_cache: dict[int, pd.DataFrame] = {}
    instances: dict[str, tuple[LaunchInstance, dict[str, object]]] = {}
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
        key = _key(str(item.region), str(item.carrier), str(item.date))
        meta = {
            "region": str(item.region),
            "month": month,
            "total_deficit": int(item.total_deficit),
            "max_airport_deficit": int(item.max_airport_deficit),
        }
        instances[key] = (instance, meta)
    return instances


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
    metrics["region"] = meta["region"]
    metrics["month"] = meta["month"]
    metrics["candidate_count"] = candidate_count
    metrics["candidate_retention"] = candidate_count / max(1, original_count)
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
            mean_expected_cost=("expected_cost", "mean"),
            mean_cvar95_cost=("cvar95_cost", "mean"),
            mean_max_regret=("max_regret", "mean"),
            mean_solve_seconds=("solve_seconds", "mean"),
            mean_candidate_count=("candidate_count", "mean"),
            mean_candidate_retention=("candidate_retention", "mean"),
            full_cost_matches=("matches_full_cost", "sum"),
            full_regret_matches=("matches_full_regret", "sum"),
        )
        .reset_index()
        .sort_values(["mean_max_regret", "mean_candidate_retention"])
    )


def main() -> None:
    output_dir = Path("results/trc_smoke/batch16_frontier_baselines")
    data_dir = Path("data/trc_open_smoke/batch16_frontier_baselines")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv("data/trc_open_smoke/batch14_year_round/selected_instances.csv")
    selected = _select_smoke_cases(selected, max_cases=12)
    selected.to_csv(data_dir / "selected_instances.csv", index=False)
    instances = _load_instances(selected)

    oracle_by_key: dict[str, dict[int, float]] = {}
    teacher_by_key = {}
    full_metrics_by_key: dict[str, dict[str, float]] = {}
    label_frames: dict[str, pd.DataFrame] = {}

    for key, (instance, _) in instances.items():
        oracle = compute_oracle_costs(instance, time_limit=8.0)
        full = LaunchOptimizer(instance, time_limit=50.0).solve("regret_portfolio", oracle_costs=oracle)
        full_metrics = evaluate_solution(instance, full, oracle_costs=oracle)
        oracle_by_key[key] = oracle
        teacher_by_key[key] = full
        full_metrics_by_key[key] = {
            "expected_cost": float(full_metrics["expected_cost"]),
            "max_regret": float(full_metrics["max_regret"]),
        }
        label_frame = make_labeled_frame(instance, full)
        label_frame["instance_key"] = key
        label_frames[key] = label_frame
        label_frame.to_csv(data_dir / f"teacher_labels_{key}.csv", index=False)

    rows: list[dict[str, object]] = []
    for key, (instance, meta) in instances.items():
        oracle = oracle_by_key[key]
        original_count = len(instance.candidates)
        full_expected = full_metrics_by_key[key]["expected_cost"]
        full_regret = full_metrics_by_key[key]["max_regret"]

        full_result = teacher_by_key[key]
        _add_metrics(
            rows,
            "regret_portfolio_full",
            key,
            meta,
            instance,
            full_result,
            oracle,
            original_count,
            original_count,
            full_expected,
            full_regret,
        )

        pressure = {
            "total_deficit": int(meta["total_deficit"]),
            "max_airport_deficit": int(meta["max_airport_deficit"]),
        }
        per_flight, per_aircraft, width_label = _adaptive_width(instance, pressure)
        adaptive_instance, _ = _filtered_instance(instance, per_flight=per_flight, per_aircraft=per_aircraft)
        adaptive = LaunchOptimizer(adaptive_instance, time_limit=50.0).solve(
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

        train_frames = [frame for train_key, frame in label_frames.items() if train_key != key]
        ml_model = fit_distilled_model(train_frames)
        ml_instance = _ml_reduced_instance(instance, ml_model, per_flight=20, per_aircraft=10)
        ml_result = LaunchOptimizer(ml_instance, time_limit=50.0).solve("regret_portfolio", oracle_costs=oracle)
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
        )

        iterative_result, iterative_instance, history = _iterative_candidate_generation(
            instance,
            oracle,
            time_limit=50.0,
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

        for method in ["saa_extensive", "cvar_extensive", "minimax_extensive", "active_scenario_saa"]:
            result = LaunchOptimizer(instance, time_limit=50.0).solve(method)
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

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "batch16_frontier_metrics.csv", index=False)
    summary = _summary(metrics)
    summary.to_csv(output_dir / "batch16_frontier_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
