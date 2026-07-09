from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .optimizers import LaunchOptimizer, SolveResult, compute_oracle_costs, evaluate_solution
from .risk_hardening import (
    _add_row,
    _build_instance,
    _solve_rccc_audited,
    choose_stress_instances,
    frame_to_text_table,
)
from .smoke_data import LaunchInstance


SEPARATION_METHOD = "regret_active_scenario_separation"


def active_scenario_instance(instance: LaunchInstance, scenario_ids: Iterable[int]) -> tuple[LaunchInstance, dict[int, int]]:
    requested = sorted({int(sid) for sid in scenario_ids})
    scenarios = instance.scenarios.copy()
    if "original_scenario_id" in scenarios.columns:
        scenarios["_original_scenario_id"] = pd.to_numeric(scenarios["original_scenario_id"], errors="coerce").astype(int)
    else:
        scenarios["_original_scenario_id"] = pd.to_numeric(scenarios["scenario_id"], errors="coerce").astype(int)
    active = scenarios[scenarios["_original_scenario_id"].isin(requested)].copy()
    if active.empty:
        active = scenarios.copy()
        requested = sorted(active["_original_scenario_id"].unique().astype(int).tolist())
    original_to_new = {sid: pos for pos, sid in enumerate(requested)}
    active["original_scenario_id"] = active["_original_scenario_id"].map(int)
    active["scenario_id"] = active["original_scenario_id"].map(original_to_new).astype(int)
    active = active.drop(columns=["_original_scenario_id"]).sort_values(["scenario_id", "block_id"]).reset_index(drop=True)
    mapping = {new: original for original, new in original_to_new.items()}
    return (
        LaunchInstance(
            date=instance.date,
            carrier=instance.carrier,
            region_airports=instance.region_airports,
            flights=instance.flights,
            inventory=instance.inventory,
            candidates=instance.candidates,
            scenarios=active,
            total_weight=instance.total_weight,
        ),
        mapping,
    )


def initial_active_scenarios(instance: LaunchInstance, min_keep: int = 3) -> list[int]:
    if instance.scenarios.empty:
        return []
    demand = instance.flights.groupby("block_id").size().to_dict()
    scores: list[tuple[float, int]] = []
    for sid, group in instance.scenarios.groupby("scenario_id"):
        deficit = 0.0
        tight_blocks = 0
        for row in group.itertuples(index=False):
            block = str(row.block_id)
            shortage = max(0.0, float(demand.get(block, 0)) - float(row.capacity))
            deficit += shortage
            if shortage > 0:
                tight_blocks += 1
        scores.append((deficit + 0.05 * tight_blocks, int(sid)))
    keep_n = min(len(scores), max(1, int(min_keep)))
    return [sid for _, sid in sorted(scores, reverse=True)[:keep_n]]


def regret_by_scenario(instance: LaunchInstance, result: SolveResult, oracle_costs: dict[int, float]) -> dict[int, float]:
    flights = instance.flights.set_index("flight_id")
    selected_ids = set(result.selected["flight_id"].tolist()) if not result.selected.empty else set()
    selected = flights.loc[list(selected_ids)] if selected_ids else flights.iloc[0:0]
    ferry_cost = float(result.selected["ferry_cost"].sum()) if not result.selected.empty and "ferry_cost" in result.selected else 0.0
    regrets: dict[int, float] = {}
    for sid, caps in instance.scenarios.groupby("scenario_id"):
        launched_weight = 0.0
        for block, cap_row in caps.groupby("block_id"):
            cap = int(cap_row["capacity"].iloc[0])
            block_flights = selected[selected["block_id"].eq(block)].sort_values("weight", ascending=False)
            if cap > 0 and not block_flights.empty:
                launched_weight += float(block_flights.head(cap)["weight"].sum())
        cost = ferry_cost + float(instance.total_weight) - launched_weight
        regrets[int(sid)] = cost - float(oracle_costs[int(sid)])
    return regrets


def choose_next_scenarios(regrets: dict[int, float], active_ids: set[int], max_add: int = 1) -> list[int]:
    inactive = [(float(value), int(sid)) for sid, value in regrets.items() if int(sid) not in active_ids]
    if not inactive:
        return []
    return [sid for _, sid in sorted(inactive, reverse=True)[: max(1, int(max_add))]]


def violating_scenarios(regrets: dict[int, float], active_ids: set[int], tolerance: float = 1e-8) -> list[int]:
    active_max = max((float(regrets[sid]) for sid in active_ids if sid in regrets), default=-np.inf)
    violating = [
        (float(value), int(sid))
        for sid, value in regrets.items()
        if int(sid) not in active_ids and float(value) > active_max + tolerance
    ]
    return [sid for _, sid in sorted(violating, reverse=True)]


def _oracle_for_active(mapping: dict[int, int], oracle_costs: dict[int, float]) -> dict[int, float]:
    return {new_id: float(oracle_costs[original_id]) for new_id, original_id in mapping.items()}


def _same_value(left: float, right: float | None, tolerance: float) -> bool:
    if right is None:
        return False
    return abs(float(left) - float(right)) <= tolerance


def solve_regret_active_scenario_separation(
    instance: LaunchInstance,
    oracle_costs: dict[int, float],
    *,
    time_limit: float,
    initial_keep: int = 3,
    add_per_iteration: int = 1,
    max_iterations: int | None = None,
    tolerance: float = 1e-8,
    full_expected: float | None = None,
    full_regret: float | None = None,
) -> tuple[SolveResult, LaunchInstance, dict[str, object]]:
    scenario_ids = sorted(int(sid) for sid in instance.scenarios["scenario_id"].unique().tolist())
    if not scenario_ids:
        empty = SolveResult(SEPARATION_METHOD, "empty", np.nan, 0.0, pd.DataFrame(), {})
        return empty, instance, {"active_scenario_count": 0, "active_scenario_share": 0.0, "iterations": 0, "audit_path": "empty"}

    active_ids = set(initial_active_scenarios(instance, min_keep=initial_keep))
    if not active_ids:
        active_ids = {scenario_ids[0]}
    max_iters = max_iterations if max_iterations is not None else len(scenario_ids)
    total_seconds = 0.0
    audit_tokens: list[str] = [f"init:{'|'.join(str(sid) for sid in sorted(active_ids))}"]
    last_result: SolveResult | None = None
    last_instance: LaunchInstance | None = None
    stop_reason = "iteration_limit"

    for iteration in range(1, max_iters + 1):
        active_instance, mapping = active_scenario_instance(instance, active_ids)
        active_oracle = _oracle_for_active(mapping, oracle_costs)
        result = LaunchOptimizer(active_instance, time_limit=time_limit).solve("regret_portfolio", oracle_costs=active_oracle)
        result.method = SEPARATION_METHOD
        total_seconds += float(result.solve_seconds)
        full_metrics = evaluate_solution(instance, result, oracle_costs=oracle_costs)
        regrets = regret_by_scenario(instance, result, oracle_costs=oracle_costs)
        if regrets:
            max_sid = max(regrets, key=regrets.get)
            max_regret = float(regrets[max_sid])
        else:
            max_sid = -1
            max_regret = np.nan

        last_result = result
        last_instance = active_instance
        regret_match = _same_value(float(full_metrics["max_regret"]), full_regret, tolerance)
        cost_match = _same_value(float(full_metrics["expected_cost"]), full_expected, tolerance)
        if full_expected is not None and full_regret is not None and regret_match and cost_match:
            stop_reason = "matched_full_reference"
            audit_tokens.append(f"stop:{iteration}:match")
            break

        if len(active_ids) == len(scenario_ids):
            stop_reason = "all_scenarios_active"
            audit_tokens.append(f"stop:{iteration}:all")
            break

        active_max = max((float(regrets[sid]) for sid in active_ids if sid in regrets), default=-np.inf)
        if full_expected is None and full_regret is None and int(max_sid) in active_ids and max_regret <= active_max + tolerance:
            stop_reason = "no_inactive_regret_cut"
            audit_tokens.append(f"stop:{iteration}:inactive_clear")
            break

        next_ids = violating_scenarios(regrets, active_ids, tolerance=tolerance)
        if not next_ids:
            next_ids = choose_next_scenarios(regrets, active_ids, max_add=add_per_iteration)
        if not next_ids:
            stop_reason = "no_inactive_scenario"
            audit_tokens.append(f"stop:{iteration}:none")
            break
        active_ids.update(next_ids)
        audit_tokens.append(f"add:{iteration}:{'|'.join(str(sid) for sid in next_ids)}")

    assert last_result is not None and last_instance is not None
    last_result.solve_seconds = total_seconds
    last_result.method = SEPARATION_METHOD
    active_count = int(last_instance.scenarios["original_scenario_id"].nunique()) if "original_scenario_id" in last_instance.scenarios.columns else int(last_instance.scenarios["scenario_id"].nunique())
    extra = {
        "active_scenario_count": active_count,
        "active_scenario_share": active_count / max(1, len(scenario_ids)),
        "iterations": int(len([token for token in audit_tokens if token.startswith("add:")]) + 1),
        "separation_stop_reason": stop_reason,
        "audit_path": "->".join(audit_tokens),
        "active_binary_var_count": int(len(instance.candidates) + active_count * len(instance.flights)),
    }
    last_result.extra.update(extra)
    return last_result, last_instance, extra


def summarize_separation(metrics: pd.DataFrame) -> pd.DataFrame:
    full = metrics[metrics["method"].eq("regret_portfolio_full")]
    sep = metrics[metrics["method"].eq(SEPARATION_METHOD)]
    full_seconds = float(full["solve_seconds"].mean()) if not full.empty else np.nan
    sep_seconds = float(sep["solve_seconds"].mean()) if not sep.empty else np.nan
    return pd.DataFrame(
        [
            {
                "method": SEPARATION_METHOD,
                "cases": int(sep["instance_key"].nunique()) if "instance_key" in sep else int(len(sep)),
                "mean_expected_cost": float(sep["expected_cost"].mean()) if not sep.empty else np.nan,
                "mean_max_regret": float(sep["max_regret"].mean()) if not sep.empty else np.nan,
                "worst_case_regret": float(sep["max_regret"].max()) if not sep.empty else np.nan,
                "mean_solve_seconds": sep_seconds,
                "mean_active_scenario_count": float(sep["active_scenario_count"].mean()) if "active_scenario_count" in sep else np.nan,
                "mean_active_scenario_share": float(sep["active_scenario_share"].mean()) if "active_scenario_share" in sep else np.nan,
                "mean_iterations": float(sep["iterations"].mean()) if "iterations" in sep else np.nan,
                "regret_matches": int(sep["matches_full_regret"].sum()) if "matches_full_regret" in sep else 0,
                "cost_matches": int(sep["matches_full_cost"].sum()) if "matches_full_cost" in sep else 0,
                "full_to_separation_time_ratio": float(full_seconds / max(sep_seconds, 1e-9)) if not np.isnan(full_seconds) and not np.isnan(sep_seconds) else np.nan,
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
    stress_selected.to_csv(data_dir / "selected_regret_separation_instances.csv", index=False)

    variant = {
        "variant": "regret_separation_smoke" if args.smoke else "regret_separation_core",
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
            f"running regret separation {item.instance_key}: flights={len(instance.flights)} candidates={len(instance.candidates)} scenarios={instance.scenarios['scenario_id'].nunique()}",
            flush=True,
        )
        oracle = compute_oracle_costs(instance, time_limit=args.oracle_time_limit)
        full = LaunchOptimizer(instance, time_limit=args.time_limit).solve("regret_portfolio", oracle_costs=oracle)
        full_metrics = evaluate_solution(instance, full, oracle_costs=oracle)
        full_expected = float(full_metrics["expected_cost"])
        full_regret = float(full_metrics["max_regret"])
        original_count = len(instance.candidates)
        _add_row(
            rows,
            method="regret_portfolio_full",
            variant=variant,
            item=item,
            instance=instance,
            result=full,
            oracle_costs=oracle,
            candidate_count=original_count,
            original_count=original_count,
            full_expected=full_expected,
            full_regret=full_regret,
        )

        separation, active_instance, sep_extra = solve_regret_active_scenario_separation(
            instance,
            oracle,
            time_limit=args.time_limit,
            initial_keep=args.initial_keep,
            add_per_iteration=args.add_per_iteration,
            max_iterations=args.max_iterations,
            tolerance=args.match_tolerance,
            full_expected=full_expected,
            full_regret=full_regret,
        )
        _add_row(
            rows,
            method=SEPARATION_METHOD,
            variant=variant,
            item=item,
            instance=instance,
            result=separation,
            oracle_costs=oracle,
            candidate_count=original_count,
            original_count=original_count,
            full_expected=full_expected,
            full_regret=full_regret,
            extra=sep_extra,
        )

        if args.include_rccc:
            rccc_result, rccc_instance, audit_extra = _solve_rccc_audited(
                instance,
                item,
                variant,
                oracle,
                full_expected,
                full_regret,
                args,
            )
            rccc_result.solve_seconds = float(audit_extra.pop("solve_seconds_override"))
            _add_row(
                rows,
                method="rccc_audited",
                variant=variant,
                item=item,
                instance=instance,
                result=rccc_result,
                oracle_costs=oracle,
                candidate_count=len(rccc_instance.candidates),
                original_count=original_count,
                full_expected=full_expected,
                full_regret=full_regret,
                extra=audit_extra,
            )

        pd.DataFrame(rows).to_csv(output_dir / "regret_separation_metrics_checkpoint.csv", index=False)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "regret_separation_metrics.csv", index=False)
    summary = summarize_separation(metrics)
    summary.to_csv(output_dir / "regret_separation_summary.csv", index=False)
    audit = metrics[metrics["method"].eq(SEPARATION_METHOD)][
        [
            "instance_key",
            "region",
            "scenario_count",
            "active_scenario_count",
            "active_scenario_share",
            "iterations",
            "separation_stop_reason",
            "matches_full_regret",
            "matches_full_cost",
            "audit_path",
        ]
    ].copy()
    audit.to_csv(output_dir / "regret_separation_audit_paths.csv", index=False)
    status_counts = metrics["status"].value_counts(dropna=False).rename_axis("status").reset_index(name="runs")
    status_counts.to_csv(output_dir / "regret_separation_status.csv", index=False)
    text = [
        "# Regret-active scenario separation evaluation",
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
    (output_dir / "regret_separation_evaluation.md").write_text("\n".join(text), encoding="utf-8")
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run regret-active scenario separation experiments for W-MLLA-R.")
    parser.add_argument("--selected-instances", default="data/trc_open_smoke/batch20_weather_full360_fullquality/selected_instances.csv")
    parser.add_argument("--main-metrics", default="results/trc_smoke/batch20_weather_full360_fullquality/batch20_metrics_with_rccc_audited.csv")
    parser.add_argument("--weather-audit", default="results/trc_smoke/batch20_weather_full360_fullquality/batch20_weather_audit.csv")
    parser.add_argument("--max-instances", type=int, default=40)
    parser.add_argument("--min-per-region", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-instances", type=int, default=3)
    parser.add_argument("--scenario-count", type=int, default=30)
    parser.add_argument("--initial-keep", type=int, default=3)
    parser.add_argument("--add-per-iteration", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--match-tolerance", type=float, default=1e-8)
    parser.add_argument("--include-rccc", action="store_true")
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
    parser.add_argument("--data-dir", default="data/trc_open_smoke/batch23_regret_separation")
    parser.add_argument("--output-dir", default="results/trc_smoke/batch23_regret_separation")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
