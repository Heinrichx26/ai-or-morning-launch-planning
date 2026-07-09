from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .optimizers import LaunchOptimizer, compute_oracle_costs
from .risk_hardening import _add_row, _build_instance, choose_stress_instances, frame_to_text_table


ENVELOPE_METHOD = "envelope_regret_portfolio"


def summarize_envelope(metrics: pd.DataFrame) -> pd.DataFrame:
    full = metrics[metrics["method"].eq("regret_portfolio_full")]
    envelope = metrics[metrics["method"].eq(ENVELOPE_METHOD)]
    full_seconds = float(full["solve_seconds"].mean()) if not full.empty else np.nan
    envelope_seconds = float(envelope["solve_seconds"].mean()) if not envelope.empty else np.nan
    full_binary = float(full["binary_var_count"].mean()) if "binary_var_count" in full and not full.empty else np.nan
    envelope_binary = float(envelope["binary_var_count"].mean()) if "binary_var_count" in envelope and not envelope.empty else np.nan
    return pd.DataFrame(
        [
            {
                "method": ENVELOPE_METHOD,
                "cases": int(envelope["instance_key"].nunique()) if "instance_key" in envelope else int(len(envelope)),
                "mean_expected_cost": float(envelope["expected_cost"].mean()) if not envelope.empty else np.nan,
                "mean_max_regret": float(envelope["max_regret"].mean()) if not envelope.empty else np.nan,
                "worst_case_regret": float(envelope["max_regret"].max()) if not envelope.empty else np.nan,
                "mean_solve_seconds": envelope_seconds,
                "mean_binary_var_count": envelope_binary,
                "mean_continuous_var_count": float(envelope["continuous_var_count"].mean()) if "continuous_var_count" in envelope else np.nan,
                "mean_envelope_cut_count": float(envelope["envelope_cut_count"].mean()) if "envelope_cut_count" in envelope else np.nan,
                "regret_matches": int(envelope["matches_full_regret"].sum()) if "matches_full_regret" in envelope else 0,
                "cost_matches": int(envelope["matches_full_cost"].sum()) if "matches_full_cost" in envelope else 0,
                "binary_reduction_share": float(1.0 - envelope_binary / full_binary) if full_binary and not np.isnan(full_binary) else np.nan,
                "full_to_envelope_time_ratio": float(full_seconds / max(envelope_seconds, 1e-9)) if not np.isnan(full_seconds) and not np.isnan(envelope_seconds) else np.nan,
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
    stress_selected.to_csv(data_dir / "selected_envelope_instances.csv", index=False)

    variant = {
        "variant": "envelope_smoke" if args.smoke else "envelope_core",
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
            f"running envelope regret {item.instance_key}: flights={len(instance.flights)} candidates={len(instance.candidates)} scenarios={instance.scenarios['scenario_id'].nunique()}",
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

        envelope = LaunchOptimizer(instance, time_limit=args.time_limit).solve(ENVELOPE_METHOD, oracle_costs=oracle)
        _add_row(
            rows,
            method=ENVELOPE_METHOD,
            variant=variant,
            item=item,
            instance=instance,
            result=envelope,
            oracle_costs=oracle,
            candidate_count=len(instance.candidates),
            original_count=len(instance.candidates),
            full_expected=full_expected,
            full_regret=full_regret,
            extra=envelope.extra,
        )
        pd.DataFrame(rows).to_csv(output_dir / "envelope_regret_metrics_checkpoint.csv", index=False)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "envelope_regret_metrics.csv", index=False)
    summary = summarize_envelope(metrics)
    summary.to_csv(output_dir / "envelope_regret_summary.csv", index=False)
    status_counts = metrics["status"].value_counts(dropna=False).rename_axis("status").reset_index(name="runs")
    status_counts.to_csv(output_dir / "envelope_regret_status.csv", index=False)
    text = [
        "# Envelope regret formulation evaluation",
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
    ]
    (output_dir / "envelope_regret_evaluation.md").write_text("\n".join(text), encoding="utf-8")
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exact launch-value envelope regret formulation experiments for W-MLLA-R.")
    parser.add_argument("--selected-instances", default="data/trc_open_smoke/batch20_weather_full360_fullquality/selected_instances.csv")
    parser.add_argument("--main-metrics", default="results/trc_smoke/batch20_weather_full360_fullquality/batch20_metrics_with_rccc_audited.csv")
    parser.add_argument("--weather-audit", default="results/trc_smoke/batch20_weather_full360_fullquality/batch20_weather_audit.csv")
    parser.add_argument("--max-instances", type=int, default=40)
    parser.add_argument("--min-per-region", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-instances", type=int, default=3)
    parser.add_argument("--scenario-count", type=int, default=30)
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
    parser.add_argument("--data-dir", default="data/trc_open_smoke/batch24_envelope_regret")
    parser.add_argument("--output-dir", default="results/trc_smoke/batch24_envelope_regret")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
