from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .optimizers import LaunchOptimizer, compute_oracle_costs, evaluate_solution
from .risk_hardening import (
    _add_row,
    _build_instance,
    _solve_rccc_audited,
    choose_stress_instances,
    frame_to_text_table,
)


def scenario_grid(smoke: bool, custom_counts: list[int] | None = None) -> list[int]:
    if custom_counts:
        counts = sorted({int(value) for value in custom_counts if int(value) > 0})
        if not counts:
            raise ValueError("At least one positive scenario count is required.")
        return counts
    return [30, 60] if smoke else [30, 60, 90, 120]


def summarize_scaling(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario_count, group in metrics.groupby("scenario_count"):
        full = group[group["method"].eq("regret_portfolio_full")]
        rccc = group[group["method"].eq("rccc_audited")]
        rows.append(
            {
                "scenario_count": int(scenario_count),
                "cases": int(rccc["instance_key"].nunique()),
                "mean_binary_var_count": float(full["binary_var_count"].mean()) if "binary_var_count" in full.columns else 0.0,
                "mean_full_seconds": float(full["solve_seconds"].mean()),
                "mean_rccc_seconds": float(rccc["solve_seconds"].mean()),
                "full_to_rccc_time_ratio": float(full["solve_seconds"].mean() / max(float(rccc["solve_seconds"].mean()), 1e-9)),
                "mean_rccc_retention": float(rccc["candidate_retention"].mean()),
                "rccc_regret_matches": int(rccc["matches_full_regret"].sum()) if "matches_full_regret" in rccc.columns else 0,
                "rccc_cost_matches": int(rccc["matches_full_cost"].sum()) if "matches_full_cost" in rccc.columns else 0,
            }
        )
    return pd.DataFrame(rows).sort_values("scenario_count").reset_index(drop=True)


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
    stress_selected.to_csv(data_dir / "selected_scaling_instances.csv", index=False)

    rows: list[dict[str, object]] = []
    month_cache = {}
    scenario_counts = scenario_grid(args.smoke, args.scenario_counts)
    for scenario_count in scenario_counts:
        variant = {
            "variant": f"s{scenario_count}",
            "scenario_count": scenario_count,
            "weight_scale": 1.0,
            "weather_scale": 1.0,
        }
        print(f"scenario scaling: scenarios={scenario_count}", flush=True)
        for item in stress_selected.itertuples(index=False):
            instance = _build_instance(item, month_cache, variant, args)
            if instance.flights.empty or instance.inventory.empty or instance.candidates.empty:
                continue
            print(
                f"running s{scenario_count} {item.instance_key}: flights={len(instance.flights)} candidates={len(instance.candidates)}",
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
            pd.DataFrame(rows).to_csv(output_dir / "scenario_scaling_metrics_checkpoint.csv", index=False)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "scenario_scaling_metrics.csv", index=False)
    summary = summarize_scaling(metrics)
    summary.to_csv(output_dir / "scenario_scaling_summary.csv", index=False)
    audit_paths = metrics[metrics["method"].eq("rccc_audited")][
        ["scenario_count", "instance_key", "audit_path", "audit_expansions", "candidate_retention", "max_regret"]
    ].copy()
    audit_paths.to_csv(output_dir / "scenario_scaling_rccc_audit_paths.csv", index=False)

    status_counts = metrics["status"].value_counts(dropna=False).rename_axis("status").reset_index(name="runs")
    status_counts.to_csv(output_dir / "scenario_scaling_status.csv", index=False)
    text = [
        "# Scenario-scaling audit evaluation",
        "",
        f"- Scaling instances: {stress_selected['instance_key'].nunique()}",
        f"- Scenario counts: {', '.join(str(v) for v in scenario_counts)}",
        f"- Method time limit: {args.time_limit} seconds",
        f"- Oracle time limit: {args.oracle_time_limit} seconds",
        f"- Raw method-scenario-instance rows: {len(metrics)}",
        "",
        "## Summary",
        "",
        frame_to_text_table(summary),
        "",
        "## Solver status",
        "",
        frame_to_text_table(status_counts),
    ]
    (output_dir / "scenario_scaling_evaluation.md").write_text("\n".join(text), encoding="utf-8")
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scenario-scaling audit for W-MLLA-R.")
    parser.add_argument("--selected-instances", default="data/trc_open_smoke/batch20_weather_full360_fullquality/selected_instances.csv")
    parser.add_argument("--main-metrics", default="results/trc_smoke/batch20_weather_full360_fullquality/batch20_metrics_with_rccc_audited.csv")
    parser.add_argument("--weather-audit", default="results/trc_smoke/batch20_weather_full360_fullquality/batch20_weather_audit.csv")
    parser.add_argument("--max-instances", type=int, default=12)
    parser.add_argument("--min-per-region", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-instances", type=int, default=2)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--scenario-counts", type=int, nargs="*", default=None)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--oracle-time-limit", type=float, default=180.0)
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
        help="distance_cost uses BTS distance weights and regional repositioning penalties; unit_service counts one unit per scheduled service.",
    )
    parser.add_argument("--data-dir", default="data/trc_open_smoke/batch22_scenario_scaling")
    parser.add_argument("--output-dir", default="results/trc_smoke/batch22_scenario_scaling")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
