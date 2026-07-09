from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .optimizers import LaunchOptimizer, SolveResult, compute_oracle_costs, evaluate_solution, launch_value_envelope_terms
from .risk_hardening import _add_row, _build_instance, choose_stress_instances, frame_to_text_table
from .smoke_data import LaunchInstance


POLYMATROID_METHOD = "polymatroid_regret_cutting"


def _scenario_blocks(instance: LaunchInstance) -> list[tuple[int, str, int]]:
    rows: list[tuple[int, str, int]] = []
    for row in instance.scenarios[["scenario_id", "block_id", "capacity"]].drop_duplicates().itertuples(index=False):
        rows.append((int(row.scenario_id), str(row.block_id), int(row.capacity)))
    return rows


def _block_flights(instance: LaunchInstance, block: str) -> pd.DataFrame:
    return instance.flights[instance.flights["block_id"].eq(block)].copy().reset_index(drop=True)


def initial_threshold_pool(instance: LaunchInstance) -> dict[tuple[int, str], set[float]]:
    pool: dict[tuple[int, str], set[float]] = {}
    for sid, block, capacity in _scenario_blocks(instance):
        flights = _block_flights(instance, block)
        weights = [float(value) for value in flights["weight"].tolist()] if not flights.empty else []
        thresholds = {0.0}
        if weights and capacity < len(weights):
            thresholds.add(float(max(weights)))
        pool[(sid, block)] = thresholds
    return pool


def full_threshold_count(instance: LaunchInstance) -> int:
    total = 0
    for sid, block, capacity in _scenario_blocks(instance):
        weights = _block_flights(instance, block)["weight"].to_numpy(float)
        total += len(launch_value_envelope_terms(weights.tolist(), capacity=capacity))
    return int(total)


def cut_pool_size(pool: dict[tuple[int, str], set[float]]) -> int:
    return int(sum(len(values) for values in pool.values()))


def _assigned_vector(flights: pd.DataFrame, assigned_by_flight: dict[str, float]) -> np.ndarray:
    return np.array([float(assigned_by_flight.get(str(row.flight_id), 0.0)) for row in flights.itertuples(index=False)])


def separate_polymatroid_cuts(
    instance: LaunchInstance,
    pool: dict[tuple[int, str], set[float]],
    q_values: dict[tuple[int, str], float],
    assigned_by_flight: dict[str, float],
    tolerance: float = 1e-8,
) -> int:
    added = 0
    for sid, block, capacity in _scenario_blocks(instance):
        flights = _block_flights(instance, block)
        if flights.empty:
            continue
        weights = flights["weight"].to_numpy(float)
        assigned = _assigned_vector(flights, assigned_by_flight)
        q_value = float(q_values.get((sid, block), 0.0))
        best_violation = 0.0
        best_theta: float | None = None
        for theta, rhs, coefficients in launch_value_envelope_terms(weights.tolist(), capacity=capacity):
            if theta in pool.setdefault((sid, block), set()):
                continue
            bound = float(rhs + np.dot(coefficients, assigned))
            violation = q_value - bound
            if violation > best_violation + tolerance:
                best_violation = violation
                best_theta = float(theta)
        if best_theta is not None:
            pool[(sid, block)].add(best_theta)
            added += 1
    return added


def _clean_extra(extra: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in extra.items() if key not in {"q_values", "flight_assignment_values"}}


def solve_polymatroid_regret_cutting(
    instance: LaunchInstance,
    oracle_costs: dict[int, float],
    *,
    time_limit: float,
    max_iterations: int = 20,
    tolerance: float = 1e-8,
    full_expected: float | None = None,
    full_regret: float | None = None,
) -> tuple[SolveResult, dict[str, object]]:
    pool = initial_threshold_pool(instance)
    full_cut_count = full_threshold_count(instance)
    total_seconds = 0.0
    audit_tokens: list[str] = [f"init:{cut_pool_size(pool)}"]
    last_result: SolveResult | None = None
    stop_reason = "iteration_limit"
    last_added = 0

    for iteration in range(1, max_iterations + 1):
        result = LaunchOptimizer(instance, time_limit=time_limit).solve_envelope_regret_with_thresholds(
            oracle_costs=oracle_costs,
            threshold_pool=pool,
            method_name=POLYMATROID_METHOD,
        )
        total_seconds += float(result.solve_seconds)
        last_result = result
        q_values = result.extra.get("q_values", {})
        assigned = result.extra.get("flight_assignment_values", {})
        if not isinstance(q_values, dict) or not isinstance(assigned, dict):
            stop_reason = "missing_separation_state"
            break
        added = separate_polymatroid_cuts(instance, pool, q_values, assigned, tolerance=tolerance)
        last_added = int(added)
        audit_tokens.append(f"iter:{iteration}:add:{added}:pool:{cut_pool_size(pool)}")
        if added == 0:
            metrics = evaluate_solution(instance, result, oracle_costs=oracle_costs)
            if full_expected is None or full_regret is None:
                stop_reason = "exact_cut_closure"
                break
            if abs(float(metrics["expected_cost"]) - float(full_expected)) <= tolerance and abs(float(metrics["max_regret"]) - float(full_regret)) <= tolerance:
                stop_reason = "exact_cut_closure_match"
                break
            stop_reason = "exact_cut_closure_mismatch"
            break

    if last_result is None:
        last_result = SolveResult(POLYMATROID_METHOD, "empty", np.nan, 0.0, pd.DataFrame(), {})
    last_result.method = POLYMATROID_METHOD
    last_result.solve_seconds = total_seconds
    extra = _clean_extra(last_result.extra)
    extra.update(
        {
            "active_polymatroid_cut_count": int(cut_pool_size(pool)),
            "full_polymatroid_cut_count": int(full_cut_count),
            "polymatroid_cut_share": float(cut_pool_size(pool) / max(1, full_cut_count)),
            "polymatroid_iterations": int(len([token for token in audit_tokens if token.startswith("iter:")])),
            "polymatroid_last_added_cuts": int(last_added),
            "polymatroid_stop_reason": stop_reason,
            "polymatroid_audit_path": "->".join(audit_tokens),
        }
    )
    last_result.extra = extra
    return last_result, extra


def summarize_polymatroid(metrics: pd.DataFrame) -> pd.DataFrame:
    full = metrics[metrics["method"].eq("regret_portfolio_full")]
    poly = metrics[metrics["method"].eq(POLYMATROID_METHOD)]
    full_seconds = float(full["solve_seconds"].mean()) if not full.empty else np.nan
    poly_seconds = float(poly["solve_seconds"].mean()) if not poly.empty else np.nan
    full_binary = float(full["binary_var_count"].mean()) if "binary_var_count" in full and not full.empty else np.nan
    poly_binary = float(poly["binary_var_count"].mean()) if "binary_var_count" in poly and not poly.empty else np.nan
    return pd.DataFrame(
        [
            {
                "method": POLYMATROID_METHOD,
                "cases": int(poly["instance_key"].nunique()) if "instance_key" in poly else int(len(poly)),
                "mean_expected_cost": float(poly["expected_cost"].mean()) if not poly.empty else np.nan,
                "mean_max_regret": float(poly["max_regret"].mean()) if not poly.empty else np.nan,
                "worst_case_regret": float(poly["max_regret"].max()) if not poly.empty else np.nan,
                "mean_solve_seconds": poly_seconds,
                "mean_binary_var_count": poly_binary,
                "binary_reduction_share": float(1.0 - poly_binary / full_binary) if full_binary and not np.isnan(full_binary) else np.nan,
                "mean_active_polymatroid_cut_count": float(poly["active_polymatroid_cut_count"].mean()) if "active_polymatroid_cut_count" in poly else np.nan,
                "mean_full_polymatroid_cut_count": float(poly["full_polymatroid_cut_count"].mean()) if "full_polymatroid_cut_count" in poly else np.nan,
                "mean_polymatroid_cut_share": float(poly["polymatroid_cut_share"].mean()) if "polymatroid_cut_share" in poly else np.nan,
                "mean_polymatroid_iterations": float(poly["polymatroid_iterations"].mean()) if "polymatroid_iterations" in poly else np.nan,
                "regret_matches": int(poly["matches_full_regret"].sum()) if "matches_full_regret" in poly else 0,
                "cost_matches": int(poly["matches_full_cost"].sum()) if "matches_full_cost" in poly else 0,
                "full_to_polymatroid_time_ratio": float(full_seconds / max(poly_seconds, 1e-9)) if not np.isnan(full_seconds) and not np.isnan(poly_seconds) else np.nan,
            }
        ]
    )


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(args.selected_instances)
    main_metrics = pd.read_csv(args.main_metrics)
    weather_audit = pd.read_csv(args.weather_audit) if args.weather_audit and Path(args.weather_audit).exists() else None
    max_instances = args.smoke_instances if args.smoke else args.max_instances
    stress_selected = choose_stress_instances(
        selected,
        main_metrics,
        weather_audit,
        max_instances=max_instances,
        min_per_region=0 if args.smoke else args.min_per_region,
    )
    stress_selected.to_csv(data_dir / "selected_polymatroid_instances.csv", index=False)

    variant = {
        "variant": "polymatroid_smoke" if args.smoke else "polymatroid_core",
        "scenario_count": int(args.scenario_count),
        "weight_scale": 1.0,
        "weather_scale": 1.0,
    }
    rows: list[dict[str, object]] = []
    month_cache: dict[int, pd.DataFrame] = {}

    for item in stress_selected.itertuples(index=False):
        instance = _build_instance(item, month_cache, variant, args)
        if instance.flights.empty or instance.inventory.empty or instance.candidates.empty:
            continue
        print(
            f"running polymatroid regret {item.instance_key}: flights={len(instance.flights)} candidates={len(instance.candidates)} scenarios={instance.scenarios['scenario_id'].nunique()}",
            flush=True,
        )
        oracle = compute_oracle_costs(instance, time_limit=args.oracle_time_limit)
        full = LaunchOptimizer(instance, time_limit=args.time_limit).solve("regret_portfolio", oracle_costs=oracle)
        full_metrics = _add_row(
            rows,
            method="regret_portfolio_full",
            variant=variant,
            item=item,
            instance=instance,
            result=full,
            oracle_costs=oracle,
            candidate_count=len(instance.candidates),
            original_count=len(instance.candidates),
            full_expected=0.0,
            full_regret=0.0,
            extra=full.extra,
        )
        full_expected = float(full_metrics["expected_cost"])
        full_regret = float(full_metrics["max_regret"])
        full_metrics["full_expected_cost"] = full_expected
        full_metrics["full_max_regret"] = full_regret
        full_metrics["matches_full_cost"] = True
        full_metrics["matches_full_regret"] = True

        poly, poly_extra = solve_polymatroid_regret_cutting(
            instance,
            oracle,
            time_limit=args.time_limit,
            max_iterations=args.max_iterations,
            tolerance=args.match_tolerance,
            full_expected=full_expected,
            full_regret=full_regret,
        )
        _add_row(
            rows,
            method=POLYMATROID_METHOD,
            variant=variant,
            item=item,
            instance=instance,
            result=poly,
            oracle_costs=oracle,
            candidate_count=len(instance.candidates),
            original_count=len(instance.candidates),
            full_expected=full_expected,
            full_regret=full_regret,
            extra=poly_extra,
        )
        pd.DataFrame(rows).to_csv(output_dir / "polymatroid_regret_metrics_checkpoint.csv", index=False)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "polymatroid_regret_metrics.csv", index=False)
    summary = summarize_polymatroid(metrics)
    summary.to_csv(output_dir / "polymatroid_regret_summary.csv", index=False)
    audit = metrics[metrics["method"].eq(POLYMATROID_METHOD)][
        [
            "instance_key",
            "region",
            "scenario_count",
            "active_polymatroid_cut_count",
            "full_polymatroid_cut_count",
            "polymatroid_cut_share",
            "polymatroid_iterations",
            "polymatroid_stop_reason",
            "matches_full_regret",
            "matches_full_cost",
            "polymatroid_audit_path",
        ]
    ].copy()
    audit.to_csv(output_dir / "polymatroid_regret_audit_paths.csv", index=False)
    status_counts = metrics["status"].value_counts(dropna=False).rename_axis("status").reset_index(name="runs")
    status_counts.to_csv(output_dir / "polymatroid_regret_status.csv", index=False)
    text = [
        "# Polymatroid regret cutting-plane evaluation",
        "",
        f"- Instances: {stress_selected['instance_key'].nunique()}",
        f"- Scenario count: {int(args.scenario_count)}",
        f"- Method time limit: {args.time_limit} seconds",
        f"- Oracle time limit: {args.oracle_time_limit} seconds",
        f"- Raw method-instance rows: {len(metrics)}",
        "",
        "## Summary",
        "",
        frame_to_text_table(summary),
        "",
        "## Solver status",
        "",
        frame_to_text_table(status_counts),
        "",
        "## Audit paths",
        "",
        frame_to_text_table(audit),
    ]
    (output_dir / "polymatroid_regret_evaluation.md").write_text("\n".join(text), encoding="utf-8")
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run polymatroid regret cutting-plane experiments for W-MLLA-R.")
    parser.add_argument("--selected-instances", default="data/trc_open_smoke/batch20_weather_full360_fullquality/selected_instances.csv")
    parser.add_argument("--main-metrics", default="results/trc_smoke/batch20_weather_full360_fullquality/batch20_metrics_with_rccc_audited.csv")
    parser.add_argument("--weather-audit", default="results/trc_smoke/batch20_weather_full360_fullquality/batch20_weather_audit.csv")
    parser.add_argument("--max-instances", type=int, default=40)
    parser.add_argument("--min-per-region", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-instances", type=int, default=3)
    parser.add_argument("--scenario-count", type=int, default=30)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--match-tolerance", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--oracle-time-limit", type=float, default=180.0)
    parser.add_argument("--use-weather", action="store_true")
    parser.add_argument("--weather-dir", default="data/weather_asos")
    parser.add_argument("--weather-local-start-hour", type=int, default=4)
    parser.add_argument("--weather-local-end-hour", type=int, default=12)
    parser.add_argument("--weather-pause-seconds", type=float, default=3.0)
    parser.add_argument("--weather-max-attempts", type=int, default=12)
    parser.add_argument("--weather-request-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--data-dir", default="data/trc_open_smoke/batch25_polymatroid_regret")
    parser.add_argument("--output-dir", default="results/trc_smoke/batch25_polymatroid_regret")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
