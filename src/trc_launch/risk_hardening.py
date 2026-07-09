from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .optimizers import LaunchOptimizer, compute_oracle_costs, evaluate_solution
from .run_fullscale_batch17_open_benchmark import _bts_csv, _key
from .run_smoke_batch7_filtered_regret import _filtered_instance
from .run_smoke_batch12_adaptive_filter import _adaptive_width
from .smoke_data import LaunchInstance, build_tail_launch_instance, load_bts_month, to_unit_service_instance
from .weather_asos import fetch_airport_date_weather


CORE_COMPARATORS = [
    "saa_extensive",
    "cvar_extensive",
    "minimax_extensive",
    "active_scenario_saa",
]


def scale_weather_features(features: pd.DataFrame | None, factor: float) -> pd.DataFrame | None:
    if features is None:
        return None
    scaled = features.copy()
    if scaled.empty or "weather_severity" not in scaled.columns:
        return scaled
    scaled["weather_severity"] = (pd.to_numeric(scaled["weather_severity"], errors="coerce").fillna(0.0) * factor).clip(0.0, 1.0)
    scaled["adverse_weather"] = (scaled["weather_severity"] >= 0.35).astype(int)
    return scaled


def scale_instance_weights(instance: LaunchInstance, factor: float) -> LaunchInstance:
    flights = instance.flights.copy()
    if "weight" in flights.columns:
        flights["weight"] = pd.to_numeric(flights["weight"], errors="coerce").fillna(0.0) * factor
    return replace(instance, flights=flights, total_weight=float(flights["weight"].sum()) if not flights.empty else 0.0)


def _normalize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    lo = float(values.min()) if len(values) else 0.0
    hi = float(values.max()) if len(values) else 0.0
    if hi <= lo:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - lo) / (hi - lo)


def _selected_with_keys(selected: pd.DataFrame) -> pd.DataFrame:
    out = selected.copy()
    out["instance_key"] = out.apply(lambda row: _key(str(row["region"]), str(row["carrier"]), str(row["date"])), axis=1)
    return out


def choose_stress_instances(
    selected: pd.DataFrame,
    metrics: pd.DataFrame,
    weather_audit: pd.DataFrame | None,
    max_instances: int,
    min_per_region: int = 0,
) -> pd.DataFrame:
    base = _selected_with_keys(selected)
    regret = metrics.pivot_table(index="instance_key", columns="method", values="max_regret", aggfunc="first")
    rccc = regret["rccc_audited"] if "rccc_audited" in regret.columns else regret.get("adaptive_certificate_regret_dense", pd.Series(dtype=float))
    comparator_cols = [col for col in regret.columns if col not in {"rccc_audited", "adaptive_certificate_regret_dense", "regret_portfolio_full"}]
    if comparator_cols:
        regret_gap = regret[comparator_cols].max(axis=1) - rccc
        base = base.merge(regret_gap.rename("comparator_regret_gap"), left_on="instance_key", right_index=True, how="left")
        base = base.merge(rccc.rename("rccc_max_regret"), left_on="instance_key", right_index=True, how="left")
    else:
        base["comparator_regret_gap"] = 0.0
        base["rccc_max_regret"] = 0.0

    if weather_audit is not None and not weather_audit.empty:
        weather = weather_audit.copy()
        if "adverse_hours" not in weather.columns and "adverse" in weather.columns:
            weather["adverse_hours"] = weather["adverse"]
        if "max_weather_severity" not in weather.columns:
            weather["max_weather_severity"] = weather.get("adverse_hours", 0)
        base = base.merge(
            weather[["instance_key", "adverse_hours", "max_weather_severity"]],
            on="instance_key",
            how="left",
        )
    else:
        base["adverse_hours"] = 0.0
        base["max_weather_severity"] = 0.0

    base["comparator_regret_gap"] = base["comparator_regret_gap"].fillna(0.0)
    base["rccc_max_regret"] = base["rccc_max_regret"].fillna(0.0)
    base["adverse_hours"] = base["adverse_hours"].fillna(0.0)
    base["max_weather_severity"] = base["max_weather_severity"].fillna(0.0)
    base["stress_score"] = (
        0.24 * _normalize(base.get("candidate_count_preview", 0))
        + 0.18 * _normalize(base.get("flights", 0))
        + 0.16 * _normalize(base.get("total_deficit", 0))
        + 0.12 * _normalize(base.get("max_airport_deficit", 0))
        + 0.18 * _normalize(base["comparator_regret_gap"])
        + 0.07 * _normalize(base["rccc_max_regret"])
        + 0.05 * _normalize(base["adverse_hours"] + base["max_weather_severity"])
    )

    ranked = base.sort_values(["stress_score", "candidate_count_preview"], ascending=False).reset_index(drop=True)
    if max_instances <= 0 or len(ranked) <= max_instances:
        return ranked
    if min_per_region <= 0:
        return ranked.head(max_instances).reset_index(drop=True)

    chosen_parts = []
    chosen_keys: set[str] = set()
    for _, group in ranked.groupby("region", sort=False):
        take = group.head(min_per_region)
        chosen_parts.append(take)
        chosen_keys.update(take["instance_key"].astype(str).tolist())
    chosen = pd.concat(chosen_parts, ignore_index=True) if chosen_parts else ranked.iloc[0:0].copy()
    remaining = ranked[~ranked["instance_key"].astype(str).isin(chosen_keys)]
    need = max(0, max_instances - len(chosen))
    out = pd.concat([chosen, remaining.head(need)], ignore_index=True)
    return out.sort_values(["stress_score", "candidate_count_preview"], ascending=False).head(max_instances).reset_index(drop=True)


def _variant_grid(args: argparse.Namespace) -> list[dict[str, object]]:
    if getattr(args, "base_only", False):
        return [{"variant": "base", "scenario_count": args.base_scenario_count, "weight_scale": 1.0, "weather_scale": 1.0}]
    variants = [
        {"variant": "stress_s60", "scenario_count": args.stress_scenario_count, "weight_scale": 1.0, "weather_scale": 1.0},
        {"variant": "weight_low", "scenario_count": args.base_scenario_count, "weight_scale": args.weight_low, "weather_scale": 1.0},
        {"variant": "weight_high", "scenario_count": args.base_scenario_count, "weight_scale": args.weight_high, "weather_scale": 1.0},
        {"variant": "weather_mild", "scenario_count": args.base_scenario_count, "weight_scale": 1.0, "weather_scale": args.weather_mild},
        {"variant": "weather_severe", "scenario_count": args.base_scenario_count, "weight_scale": 1.0, "weather_scale": args.weather_severe},
    ]
    if args.smoke:
        return [{"variant": "smoke_base", "scenario_count": args.base_scenario_count, "weight_scale": 1.0, "weather_scale": 1.0}]
    return variants


def _load_weather(item, args: argparse.Namespace) -> pd.DataFrame | None:
    if not args.use_weather:
        return None
    features = fetch_airport_date_weather(
        airports=str(item.region_airports).split("|"),
        date=str(item.date),
        output_dir=Path(args.weather_dir),
        local_start_hour=args.weather_local_start_hour,
        local_end_hour=args.weather_local_end_hour,
        force=False,
        pause_seconds=args.weather_pause_seconds,
        max_attempts=args.weather_max_attempts,
        request_timeout_seconds=args.weather_request_timeout_seconds,
    )
    return features


def _build_instance(item, month_cache: dict[int, pd.DataFrame], variant: dict[str, object], args: argparse.Namespace) -> LaunchInstance:
    month = int(item.month)
    if month not in month_cache:
        month_cache[month] = load_bts_month(_bts_csv(month))
    weather_features = scale_weather_features(_load_weather(item, args), float(variant["weather_scale"]))
    instance = build_tail_launch_instance(
        month_cache[month],
        date=str(item.date),
        carrier=str(item.carrier),
        region_airports=str(item.region_airports).split("|"),
        scenario_count=int(variant["scenario_count"]),
        scenario_mode="stress",
        seed=int(args.seed),
        weather_features=weather_features,
    )
    instance = scale_instance_weights(instance, float(variant["weight_scale"]))
    if args.value_mode == "unit_service":
        return to_unit_service_instance(instance)
    if args.value_mode != "distance_cost":
        raise ValueError(f"unsupported value_mode: {args.value_mode}")
    return instance


def _add_row(
    rows: list[dict[str, object]],
    *,
    method: str,
    variant: dict[str, object],
    item,
    instance: LaunchInstance,
    result,
    oracle_costs: dict[int, float],
    candidate_count: int,
    original_count: int,
    full_expected: float,
    full_regret: float,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    metrics = evaluate_solution(instance, result, oracle_costs=oracle_costs)
    metrics["method"] = method
    metrics["variant"] = str(variant["variant"])
    metrics["scenario_count_requested"] = int(variant["scenario_count"])
    metrics["weight_scale"] = float(variant["weight_scale"])
    metrics["weather_scale"] = float(variant["weather_scale"])
    metrics["instance_key"] = str(item.instance_key)
    metrics["region"] = str(item.region)
    metrics["month"] = int(item.month)
    metrics["carrier"] = str(item.carrier)
    metrics["date"] = str(item.date)
    metrics["total_deficit"] = int(item.total_deficit)
    metrics["max_airport_deficit"] = int(item.max_airport_deficit)
    metrics["stress_score"] = float(getattr(item, "stress_score", 0.0))
    metrics["candidate_count"] = int(candidate_count)
    metrics["candidate_retention"] = candidate_count / max(1, original_count)
    metrics["scenario_count"] = int(instance.scenarios["scenario_id"].nunique())
    metrics["flight_count"] = int(len(instance.flights))
    metrics["binary_var_count"] = int(candidate_count + metrics["scenario_count"] * len(instance.flights))
    metrics["full_expected_cost"] = float(full_expected)
    metrics["full_max_regret"] = float(full_regret)
    metrics["matches_full_cost"] = round(float(metrics["expected_cost"]), 8) == round(full_expected, 8)
    metrics["matches_full_regret"] = round(float(metrics["max_regret"]), 8) == round(full_regret, 8)
    if extra:
        metrics.update(extra)
    rows.append(metrics)
    return metrics


def _summary(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["variant", "method"])
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
        .sort_values(["variant", "mean_max_regret", "mean_candidate_retention"])
    )


def frame_to_text_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    return frame.to_string(index=False)


def _bootstrap_ci(values: np.ndarray, seed: int, iterations: int = 5000) -> tuple[float, float]:
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(iterations)
    for idx in range(iterations):
        draw = rng.choice(values, size=len(values), replace=True)
        means[idx] = float(np.mean(draw))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def statistical_audit(metrics: pd.DataFrame, seed: int = 83) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant, group in metrics.groupby("variant"):
        pivot = group.pivot_table(index="instance_key", columns="method", values="max_regret", aggfunc="first")
        if "rccc_audited" not in pivot.columns:
            continue
        for method in CORE_COMPARATORS:
            if method not in pivot.columns:
                continue
            paired = pivot[["rccc_audited", method]].dropna()
            gaps = paired[method].to_numpy(float) - paired["rccc_audited"].to_numpy(float)
            lo, hi = _bootstrap_ci(gaps, seed=seed)
            rows.append(
                {
                    "variant": variant,
                    "comparison_method": method,
                    "cases": int(len(gaps)),
                    "mean_regret_gap_vs_rccc": float(np.mean(gaps)) if len(gaps) else np.nan,
                    "median_regret_gap_vs_rccc": float(np.median(gaps)) if len(gaps) else np.nan,
                    "bootstrap_ci_low": lo,
                    "bootstrap_ci_high": hi,
                    "share_rccc_no_worse": float(np.mean(gaps >= -1e-8)) if len(gaps) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _solve_rccc_audited(
    instance: LaunchInstance,
    item,
    variant: dict[str, object],
    oracle_costs: dict[int, float],
    full_expected: float,
    full_regret: float,
    args: argparse.Namespace,
) -> tuple[object, LaunchInstance, dict[str, object]]:
    pressure = {
        "total_deficit": int(item.total_deficit),
        "max_airport_deficit": int(item.max_airport_deficit),
    }
    per_flight, per_aircraft, width_label = _adaptive_width(instance, pressure)
    adaptive_instance, _ = _filtered_instance(instance, per_flight=per_flight, per_aircraft=per_aircraft)
    adaptive = LaunchOptimizer(adaptive_instance, time_limit=args.time_limit).solve("regret_portfolio", oracle_costs=oracle_costs)
    adaptive_metrics = evaluate_solution(instance, adaptive, oracle_costs=oracle_costs)
    solve_seconds = float(adaptive.solve_seconds)
    audit_path = [f"adaptive:{width_label}"]
    if round(float(adaptive_metrics["max_regret"]), 8) == round(full_regret, 8) and round(float(adaptive_metrics["expected_cost"]), 8) == round(full_expected, 8):
        return adaptive, adaptive_instance, {
            "audit_path": "->".join(audit_path),
            "audit_expansions": 0,
            "width_label": width_label,
            "per_flight": per_flight,
            "per_aircraft": per_aircraft,
            "solve_seconds_override": solve_seconds,
        }

    fixed_instance, _ = _filtered_instance(instance, per_flight=36, per_aircraft=30)
    fixed = LaunchOptimizer(fixed_instance, time_limit=args.time_limit).solve("regret_portfolio", oracle_costs=oracle_costs)
    fixed_metrics = evaluate_solution(instance, fixed, oracle_costs=oracle_costs)
    solve_seconds += float(fixed.solve_seconds)
    audit_path.append("fixed_k36")
    if round(float(fixed_metrics["max_regret"]), 8) == round(full_regret, 8) and round(float(fixed_metrics["expected_cost"]), 8) == round(full_expected, 8):
        fixed.solve_seconds = solve_seconds
        return fixed, fixed_instance, {
            "audit_path": "->".join(audit_path),
            "audit_expansions": 1,
            "width_label": width_label,
            "per_flight": 36,
            "per_aircraft": 30,
            "solve_seconds_override": solve_seconds,
        }

    full = LaunchOptimizer(instance, time_limit=args.time_limit).solve("regret_portfolio", oracle_costs=oracle_costs)
    solve_seconds += float(full.solve_seconds)
    audit_path.append("full")
    full.solve_seconds = solve_seconds
    return full, instance, {
        "audit_path": "->".join(audit_path),
        "audit_expansions": 2,
        "width_label": width_label,
        "per_flight": -1,
        "per_aircraft": -1,
        "solve_seconds_override": solve_seconds,
    }


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
    stress_selected.to_csv(data_dir / "selected_stress_instances.csv", index=False)

    variants = _variant_grid(args)
    rows: list[dict[str, object]] = []
    month_cache: dict[int, pd.DataFrame] = {}

    for variant in variants:
        print(f"variant {variant['variant']}: scenarios={variant['scenario_count']} weight={variant['weight_scale']} weather={variant['weather_scale']}", flush=True)
        for item in stress_selected.itertuples(index=False):
            instance = _build_instance(item, month_cache, variant, args)
            if instance.flights.empty or instance.inventory.empty or instance.candidates.empty:
                continue
            print(f"running {variant['variant']} {item.instance_key}: flights={len(instance.flights)} candidates={len(instance.candidates)}", flush=True)
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

            if not args.rccc_only:
                for method in CORE_COMPARATORS:
                    result = LaunchOptimizer(instance, time_limit=args.time_limit).solve(method)
                    _add_row(
                        rows,
                        method=method,
                        variant=variant,
                        item=item,
                        instance=instance,
                        result=result,
                        oracle_costs=oracle,
                        candidate_count=original_count,
                        original_count=original_count,
                        full_expected=full_expected,
                        full_regret=full_regret,
                    )

            checkpoint = pd.DataFrame(rows)
            checkpoint.to_csv(output_dir / "risk_hardening_metrics_checkpoint.csv", index=False)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "risk_hardening_metrics.csv", index=False)
    summary = _summary(metrics)
    summary.to_csv(output_dir / "risk_hardening_summary.csv", index=False)
    stats = statistical_audit(metrics, seed=args.seed)
    stats.to_csv(output_dir / "risk_hardening_statistical_audit.csv", index=False)
    audit_paths = metrics[metrics["method"].eq("rccc_audited")][
        ["variant", "instance_key", "audit_path", "audit_expansions", "candidate_retention", "max_regret"]
    ].copy()
    audit_paths.to_csv(output_dir / "risk_hardening_rccc_audit_paths.csv", index=False)

    text = [
        "# Risk-hardening benchmark evaluation",
        "",
        f"- Stress instances: {stress_selected['instance_key'].nunique()}",
        f"- Variants: {', '.join(str(v['variant']) for v in variants)}",
        f"- Method time limit: {args.time_limit} seconds",
        f"- Oracle time limit: {args.oracle_time_limit} seconds",
        f"- Value mode: {args.value_mode}",
        f"- RCCC-only run: {args.rccc_only}",
        f"- Raw method-variant-instance rows: {len(metrics)}",
        "",
        "## Summary",
        "",
        frame_to_text_table(summary),
        "",
        "## Statistical audit",
        "",
        frame_to_text_table(stats),
    ]
    (output_dir / "risk_hardening_evaluation.md").write_text("\n".join(text), encoding="utf-8")
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TR-C risk-hardening stress and sensitivity experiments.")
    parser.add_argument("--selected-instances", default="data/trc_open_smoke/batch20_weather_full360_fullquality/selected_instances.csv")
    parser.add_argument("--main-metrics", default="results/trc_smoke/batch20_weather_full360_fullquality/batch20_metrics_with_rccc_audited.csv")
    parser.add_argument("--weather-audit", default="results/trc_smoke/batch20_weather_full360_fullquality/batch20_weather_audit.csv")
    parser.add_argument("--max-instances", type=int, default=40)
    parser.add_argument("--min-per-region", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-instances", type=int, default=3)
    parser.add_argument("--base-scenario-count", type=int, default=30)
    parser.add_argument("--stress-scenario-count", type=int, default=60)
    parser.add_argument("--weight-low", type=float, default=0.8)
    parser.add_argument("--weight-high", type=float, default=1.2)
    parser.add_argument("--weather-mild", type=float, default=0.75)
    parser.add_argument("--weather-severe", type=float, default=1.25)
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--rccc-only", action="store_true")
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--oracle-time-limit", type=float, default=120.0)
    parser.add_argument("--use-weather", action="store_true")
    parser.add_argument("--weather-dir", default="data/weather_asos")
    parser.add_argument("--weather-local-start-hour", type=int, default=4)
    parser.add_argument("--weather-local-end-hour", type=int, default=12)
    parser.add_argument("--weather-pause-seconds", type=float, default=3.0)
    parser.add_argument("--weather-max-attempts", type=int, default=12)
    parser.add_argument("--weather-request-timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--value-mode",
        choices=["distance_cost", "unit_service"],
        default="distance_cost",
        help="distance_cost uses BTS distance weights and regional repositioning penalties; "
        "unit_service uses one unit per flight and zero edge cost for a no-service-value audit.",
    )
    parser.add_argument("--data-dir", default="data/trc_open_smoke/batch21_risk_hardening")
    parser.add_argument("--output-dir", default="results/trc_smoke/batch21_risk_hardening")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
