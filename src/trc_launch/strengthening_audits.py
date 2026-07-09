from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .optimizers import LaunchOptimizer, SolveResult, certificate_candidate_scores, compute_oracle_costs, evaluate_solution
from .risk_hardening import (
    _build_instance,
    _solve_rccc_audited,
    choose_stress_instances,
    frame_to_text_table,
)
from .run_smoke_batch7_filtered_regret import _filtered_instance
from .smoke_data import LaunchInstance


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SELECTED = ROOT / "data" / "trc_open_smoke" / "batch20_weather_full360_fullquality" / "selected_instances.csv"
DEFAULT_MAIN_METRICS = ROOT / "results" / "trc_smoke" / "batch20_weather_full360_fullquality" / "batch20_metrics_with_rccc_audited.csv"
DEFAULT_WEATHER_AUDIT = ROOT / "results" / "trc_smoke" / "batch20_weather_full360_fullquality" / "batch20_weather_audit.csv"
DEFAULT_DATA_DIR = ROOT / "data" / "trc_open_smoke" / "batch29_strengthening"
DEFAULT_RESULT_DIR = ROOT / "results" / "trc_smoke" / "batch29_strengthening"
DEFAULT_GENERATED_DIR = ROOT / "article" / "trc_elsarticle" / "generated"

ABLATION_METHODS = [
    "regret_portfolio_full",
    "dominance_cost_floor",
    "exposure_only",
    "tail_coverage_only",
    "adaptive_without_audit",
    "rccc_audited",
]


def _copy_with_candidates(instance: LaunchInstance, candidates: pd.DataFrame) -> LaunchInstance:
    return replace(instance, candidates=candidates.reset_index(drop=True).copy())


def _same_airport_mask(candidates: pd.DataFrame) -> pd.Series:
    if {"source_airport", "origin"}.issubset(candidates.columns):
        return candidates["source_airport"].astype(str).eq(candidates["origin"].astype(str))
    return candidates["ferry_cost"].eq(0)


def _scored_candidates(instance: LaunchInstance) -> pd.DataFrame:
    candidates = instance.candidates.reset_index(drop=True).copy()
    scores = certificate_candidate_scores(instance)
    scored = candidates.merge(scores[["candidate_id", "certificate_score"]], on="candidate_id", how="left")
    scored["certificate_score"] = pd.to_numeric(scored["certificate_score"], errors="coerce").fillna(-1e9)
    return scored


def _reduce_by_ids(instance: LaunchInstance, keep_ids: set[str]) -> LaunchInstance:
    candidates = instance.candidates.reset_index(drop=True).copy()
    reduced = candidates[candidates["candidate_id"].astype(str).isin(keep_ids)].copy()
    return _copy_with_candidates(instance, reduced)


def _dominance_cost_floor(instance: LaunchInstance, per_flight: int) -> LaunchInstance:
    scored = _scored_candidates(instance)
    keep_ids: set[str] = set(scored[_same_airport_mask(scored)]["candidate_id"].astype(str).tolist())
    for _, group in scored.groupby("flight_id"):
        ordered = group.sort_values(["ferry_cost", "certificate_score"], ascending=[True, False])
        keep_ids.update(ordered.head(per_flight)["candidate_id"].astype(str).tolist())
    return _reduce_by_ids(instance, keep_ids)


def _exposure_only(instance: LaunchInstance, per_flight: int) -> LaunchInstance:
    scored = _scored_candidates(instance)
    keep_ids: set[str] = set(scored[_same_airport_mask(scored)]["candidate_id"].astype(str).tolist())
    for _, group in scored.groupby("flight_id"):
        keep_ids.update(group.nlargest(per_flight, "certificate_score")["candidate_id"].astype(str).tolist())
    return _reduce_by_ids(instance, keep_ids)


def _tail_coverage_only(instance: LaunchInstance, per_aircraft: int) -> LaunchInstance:
    scored = _scored_candidates(instance)
    keep_ids: set[str] = set(scored[_same_airport_mask(scored)]["candidate_id"].astype(str).tolist())
    if "aircraft_id" in scored.columns:
        for _, group in scored.groupby("aircraft_id"):
            keep_ids.update(group.nlargest(per_aircraft, "certificate_score")["candidate_id"].astype(str).tolist())
    return _reduce_by_ids(instance, keep_ids)


def _add_metric_row(
    rows: list[dict[str, object]],
    *,
    method: str,
    item,
    instance: LaunchInstance,
    result: SolveResult,
    oracle_costs: dict[int, float],
    candidate_count: int,
    original_count: int,
    full_expected: float,
    full_regret: float,
    extra: dict[str, object] | None = None,
) -> None:
    metrics = evaluate_solution(instance, result, oracle_costs=oracle_costs)
    metrics["method"] = method
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
    metrics["full_expected_cost"] = float(full_expected)
    metrics["full_max_regret"] = float(full_regret)
    metrics["matches_full_cost"] = round(float(metrics["expected_cost"]), 8) == round(full_expected, 8)
    metrics["matches_full_regret"] = round(float(metrics["max_regret"]), 8) == round(full_regret, 8)
    if extra:
        metrics.update(extra)
    rows.append(metrics)


def _summarize_methods(metrics: pd.DataFrame) -> pd.DataFrame:
    out = (
        metrics.groupby("method")
        .agg(
            cases=("instance_key", "count"),
            regions=("region", "nunique"),
            months=("month", "nunique"),
            mean_max_regret=("max_regret", "mean"),
            worst_case_regret=("max_regret", "max"),
            mean_expected_cost=("expected_cost", "mean"),
            mean_candidate_retention=("candidate_retention", "mean"),
            mean_solve_seconds=("solve_seconds", "mean"),
            full_regret_matches=("matches_full_regret", "sum"),
            full_cost_matches=("matches_full_cost", "sum"),
        )
        .reset_index()
    )
    order = {method: idx for idx, method in enumerate(ABLATION_METHODS)}
    out["method_order"] = out["method"].map(order).fillna(99).astype(int)
    return out.sort_values(["method_order", "mean_max_regret"]).drop(columns=["method_order"]).reset_index(drop=True)


def _run_ablation(args: argparse.Namespace, selected: pd.DataFrame, metrics: pd.DataFrame, weather_audit: pd.DataFrame | None) -> pd.DataFrame:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    max_instances = args.smoke_instances if args.smoke else args.ablation_instances
    stress_selected = choose_stress_instances(
        selected,
        metrics,
        weather_audit,
        max_instances=max_instances,
        min_per_region=0 if args.smoke else args.min_per_region,
    )
    stress_selected.to_csv(data_dir / "mechanism_ablation_instances.csv", index=False)

    rows: list[dict[str, object]] = []
    month_cache: dict[int, pd.DataFrame] = {}
    variant = {"variant": "base", "scenario_count": args.scenario_count, "weight_scale": 1.0, "weather_scale": 1.0}
    build_args = SimpleNamespace(**vars(args))
    build_args.base_scenario_count = args.scenario_count
    build_args.value_mode = args.value_mode

    for item in stress_selected.itertuples(index=False):
        instance = _build_instance(item, month_cache, variant, build_args)
        if instance.flights.empty or instance.inventory.empty or instance.candidates.empty:
            continue
        print(f"mechanism {item.instance_key}: flights={len(instance.flights)} candidates={len(instance.candidates)}", flush=True)
        oracle = compute_oracle_costs(instance, time_limit=args.oracle_time_limit)
        full = LaunchOptimizer(instance, time_limit=args.time_limit).solve("regret_portfolio", oracle_costs=oracle)
        full_metrics = evaluate_solution(instance, full, oracle_costs=oracle)
        full_expected = float(full_metrics["expected_cost"])
        full_regret = float(full_metrics["max_regret"])
        original_count = len(instance.candidates)
        _add_metric_row(
            rows,
            method="regret_portfolio_full",
            item=item,
            instance=instance,
            result=full,
            oracle_costs=oracle,
            candidate_count=original_count,
            original_count=original_count,
            full_expected=full_expected,
            full_regret=full_regret,
        )

        reductions = {
            "dominance_cost_floor": _dominance_cost_floor(instance, args.ablation_per_flight),
            "exposure_only": _exposure_only(instance, args.ablation_per_flight),
            "tail_coverage_only": _tail_coverage_only(instance, args.ablation_per_aircraft),
        }
        adaptive_instance, _ = _filtered_instance(
            instance,
            per_flight=args.adaptive_per_flight,
            per_aircraft=args.adaptive_per_aircraft,
        )
        reductions["adaptive_without_audit"] = adaptive_instance

        for method, reduced in reductions.items():
            result = LaunchOptimizer(reduced, time_limit=args.time_limit).solve("regret_portfolio", oracle_costs=oracle)
            _add_metric_row(
                rows,
                method=method,
                item=item,
                instance=instance,
                result=result,
                oracle_costs=oracle,
                candidate_count=len(reduced.candidates),
                original_count=original_count,
                full_expected=full_expected,
                full_regret=full_regret,
                extra={"ablation_candidate_loss": original_count - len(reduced.candidates)},
            )

        rccc_result, rccc_instance, audit_extra = _solve_rccc_audited(
            instance,
            item,
            variant,
            oracle,
            full_expected,
            full_regret,
            build_args,
        )
        rccc_result.solve_seconds = float(audit_extra.pop("solve_seconds_override"))
        _add_metric_row(
            rows,
            method="rccc_audited",
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

        pd.DataFrame(rows).to_csv(output_dir / "mechanism_ablation_metrics_checkpoint.csv", index=False)

    ablation = pd.DataFrame(rows)
    ablation.to_csv(output_dir / "mechanism_ablation_metrics.csv", index=False)
    summary = _summarize_methods(ablation)
    summary.to_csv(output_dir / "mechanism_ablation_summary.csv", index=False)
    return summary


def _heldout_summary(metrics: pd.DataFrame, weather_audit: pd.DataFrame | None) -> pd.DataFrame:
    rccc = metrics[metrics["method"].eq("rccc_audited")].copy()
    if rccc.empty:
        rccc = metrics[metrics["method"].eq("adaptive_certificate_regret_dense")].copy()
    if weather_audit is not None and not weather_audit.empty:
        wx = weather_audit[["instance_key", "adverse"]].drop_duplicates("instance_key").copy()
        rccc = rccc.merge(wx, on="instance_key", how="left")
    if "adverse" not in rccc.columns:
        rccc["adverse"] = 0
    rccc["adverse"] = pd.to_numeric(rccc["adverse"], errors="coerce").fillna(0).astype(int)
    rccc["heldout_calendar"] = np.where(rccc["month"].astype(int) >= 202509, "Sep-Dec held-out", "Jan-Aug calibration")
    rccc["weather_subset"] = np.where(rccc["adverse"].gt(0), "Adverse-weather held-out", "No-adverse-weather held-out")

    rows: list[dict[str, object]] = []

    def add(split_type: str, subset: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        rows.append(
            {
                "split_type": split_type,
                "subset": subset,
                "cases": int(frame["instance_key"].nunique()),
                "regions": int(frame["region"].nunique()),
                "months": int(frame["month"].nunique()),
                "mean_max_regret": float(frame["max_regret"].mean()),
                "worst_case_regret": float(frame["max_regret"].max()),
                "mean_candidate_retention": float(frame["candidate_retention"].mean()),
                "full_regret_matches": int(frame["matches_full_regret"].sum()),
                "match_rate": float(frame["matches_full_regret"].mean()),
            }
        )

    for subset, frame in rccc.groupby("heldout_calendar"):
        add("calendar", str(subset), frame)
    for region, frame in rccc.groupby("region"):
        add("leave-region-group", str(region), frame)
    for subset, frame in rccc.groupby("weather_subset"):
        add("weather", str(subset), frame)

    region_rows = [row for row in rows if row["split_type"] == "leave-region-group"]
    if region_rows:
        region_frame = pd.DataFrame(region_rows)
        rows.append(
            {
                "split_type": "leave-region-group",
                "subset": "Worst region group",
                "cases": int(region_frame.loc[region_frame["match_rate"].idxmin(), "cases"]),
                "regions": 1,
                "months": int(region_frame.loc[region_frame["match_rate"].idxmin(), "months"]),
                "mean_max_regret": float(region_frame["mean_max_regret"].max()),
                "worst_case_regret": float(region_frame["worst_case_regret"].max()),
                "mean_candidate_retention": float(region_frame["mean_candidate_retention"].min()),
                "full_regret_matches": int(region_frame.loc[region_frame["match_rate"].idxmin(), "full_regret_matches"]),
                "match_rate": float(region_frame["match_rate"].min()),
            }
        )
    return pd.DataFrame(rows)


def _operational_kpi(metrics: pd.DataFrame, weather_audit: pd.DataFrame | None) -> pd.DataFrame:
    frame = metrics[metrics["method"].isin(["rccc_audited", "regret_portfolio_full"])].copy()
    if weather_audit is not None and not weather_audit.empty:
        wx = weather_audit[["instance_key", "adverse"]].drop_duplicates("instance_key").copy()
        frame = frame.merge(wx, on="instance_key", how="left")
    if "adverse" not in frame.columns:
        frame["adverse"] = 0
    frame["adverse"] = pd.to_numeric(frame["adverse"], errors="coerce").fillna(0).astype(int)
    frame["subset"] = "All instances"
    adverse = frame.copy()
    adverse["subset"] = np.where(adverse["adverse"].gt(0), "Adverse-weather instances", "No-adverse-weather instances")
    frame = pd.concat([frame, adverse], ignore_index=True)

    summary = (
        frame.groupby(["subset", "method"])
        .agg(
            cases=("instance_key", "nunique"),
            mean_assigned_flights=("assigned_flights", "mean"),
            mean_assigned_weight_share=("assigned_weight_share", "mean"),
            mean_ferry_moves=("ferry_moves", "mean"),
            mean_ferry_cost=("ferry_cost", "mean"),
            mean_launch_weight_share=("mean_launch_weight_share", "mean"),
            worst_min_launch_weight_share=("min_launch_weight_share", "min"),
            mean_candidate_retention=("candidate_retention", "mean"),
            mean_audit_expansions=("audit_expansions", "mean"),
            full_regret_matches=("matches_full_regret", "sum"),
        )
        .reset_index()
    )
    return summary.sort_values(["subset", "method"]).reset_index(drop=True)


def _scenario_details(
    instance: LaunchInstance,
    result: SolveResult,
    oracle_costs: dict[int, float],
    method: str,
    item,
) -> pd.DataFrame:
    flights = instance.flights.set_index("flight_id")
    selected_ids = set(result.selected["flight_id"].tolist()) if not result.selected.empty else set()
    selected = flights.loc[list(selected_ids)] if selected_ids else flights.iloc[0:0]
    ferry_cost = float(result.selected["ferry_cost"].sum()) if not result.selected.empty else 0.0
    rows: list[dict[str, object]] = []
    for sid, caps in instance.scenarios.groupby("scenario_id"):
        launched_weight = 0.0
        demand_total = 0
        capacity_total = 0
        for block, cap_row in caps.groupby("block_id"):
            cap = int(cap_row["capacity"].iloc[0])
            block_all = instance.flights[instance.flights["block_id"].eq(block)]
            block_selected = selected[selected["block_id"].eq(block)].sort_values("weight", ascending=False)
            demand_total += int(len(block_all))
            capacity_total += int(cap)
            if cap > 0 and not block_selected.empty:
                launched_weight += float(block_selected.head(cap)["weight"].sum())
        cost = ferry_cost + instance.total_weight - launched_weight
        oracle = float(oracle_costs.get(int(sid), np.nan))
        first = caps.iloc[0]
        rows.append(
            {
                "instance_key": str(item.instance_key),
                "region": str(item.region),
                "method": method,
                "scenario_id": int(sid),
                "scenario_type": str(first.get("scenario_type", "")),
                "demand_total": int(demand_total),
                "capacity_total": int(capacity_total),
                "launched_weight": float(launched_weight),
                "launch_weight_share": launched_weight / instance.total_weight if instance.total_weight > 0 else np.nan,
                "ferry_cost": ferry_cost,
                "scenario_cost": float(cost),
                "oracle_cost": oracle,
                "scenario_regret": float(cost - oracle) if np.isfinite(oracle) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _block_details(instance: LaunchInstance, item) -> pd.DataFrame:
    demand = instance.flights.groupby("block_id").size().rename("scheduled_flights").reset_index()
    caps = (
        instance.scenarios.groupby("block_id")
        .agg(
            mean_capacity=("capacity", "mean"),
            min_capacity=("capacity", "min"),
            p10_capacity=("capacity", lambda values: float(np.quantile(values, 0.10))),
        )
        .reset_index()
    )
    out = demand.merge(caps, on="block_id", how="left")
    out["mean_shortage"] = (out["scheduled_flights"] - out["mean_capacity"]).clip(lower=0.0)
    out["p10_shortage"] = (out["scheduled_flights"] - out["p10_capacity"]).clip(lower=0.0)
    out["airport"] = out["block_id"].astype(str).str.split("_").str[0]
    out["hour"] = out["block_id"].astype(str).str.split("_").str[1].astype(int)
    out["instance_key"] = str(item.instance_key)
    out["region"] = str(item.region)
    return out.sort_values(["airport", "hour"]).reset_index(drop=True)


def _edge_details(result: SolveResult, method: str, item) -> pd.DataFrame:
    if result.selected.empty:
        return pd.DataFrame()
    selected = result.selected.copy()
    selected["method"] = method
    selected["instance_key"] = str(item.instance_key)
    selected["region"] = str(item.region)
    selected["cross_airport"] = ~selected["source_airport"].astype(str).eq(selected["origin"].astype(str))
    selected["assignment_pair"] = selected["source_airport"].astype(str) + " -> " + selected["origin"].astype(str)
    return selected


def _run_case(args: argparse.Namespace, selected: pd.DataFrame, metrics: pd.DataFrame, weather_audit: pd.DataFrame | None) -> None:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    chosen = choose_stress_instances(selected, metrics, weather_audit, max_instances=1, min_per_region=0)
    chosen.to_csv(data_dir / "operational_case_instance.csv", index=False)
    item = next(chosen.itertuples(index=False))
    month_cache: dict[int, pd.DataFrame] = {}
    variant = {"variant": "base", "scenario_count": args.scenario_count, "weight_scale": 1.0, "weather_scale": 1.0}
    build_args = SimpleNamespace(**vars(args))
    instance = _build_instance(item, month_cache, variant, build_args)
    oracle = compute_oracle_costs(instance, time_limit=args.oracle_time_limit)
    full = LaunchOptimizer(instance, time_limit=args.time_limit).solve("regret_portfolio", oracle_costs=oracle)
    full_metrics = evaluate_solution(instance, full, oracle_costs=oracle)
    rccc_result, rccc_instance, audit_extra = _solve_rccc_audited(
        instance,
        item,
        variant,
        oracle,
        float(full_metrics["expected_cost"]),
        float(full_metrics["max_regret"]),
        build_args,
    )
    rccc_result.solve_seconds = float(audit_extra.pop("solve_seconds_override"))
    saa = LaunchOptimizer(instance, time_limit=args.time_limit).solve("saa_extensive")
    cvar = LaunchOptimizer(instance, time_limit=args.time_limit).solve("cvar_extensive")

    method_rows: list[dict[str, object]] = []
    for method, result, cand_count, extra in [
        ("regret_portfolio_full", full, len(instance.candidates), {}),
        ("rccc_audited", rccc_result, len(rccc_instance.candidates), audit_extra),
        ("saa_extensive", saa, len(instance.candidates), {}),
        ("cvar_extensive", cvar, len(instance.candidates), {}),
    ]:
        row = evaluate_solution(instance, result, oracle_costs=oracle)
        row["method"] = method
        row["instance_key"] = str(item.instance_key)
        row["region"] = str(item.region)
        row["candidate_count"] = cand_count
        row["candidate_retention"] = cand_count / max(1, len(instance.candidates))
        row.update(extra)
        method_rows.append(row)
    pd.DataFrame(method_rows).to_csv(output_dir / "operational_case_methods.csv", index=False)

    scenario_frames = []
    edge_frames = []
    for method, result in [
        ("regret_portfolio_full", full),
        ("rccc_audited", rccc_result),
        ("saa_extensive", saa),
        ("cvar_extensive", cvar),
    ]:
        scenario_frames.append(_scenario_details(instance, result, oracle, method, item))
        edge_frames.append(_edge_details(result, method, item))
    pd.concat(scenario_frames, ignore_index=True).to_csv(output_dir / "operational_case_scenarios.csv", index=False)
    pd.concat(edge_frames, ignore_index=True).to_csv(output_dir / "operational_case_edges.csv", index=False)
    _block_details(instance, item).to_csv(output_dir / "operational_case_blocks.csv", index=False)


def _write_evaluation(
    output_dir: Path,
    ablation_summary: pd.DataFrame,
    heldout: pd.DataFrame,
    operational: pd.DataFrame,
) -> None:
    rccc = ablation_summary[ablation_summary["method"].eq("rccc_audited")]
    ablation_ok = bool(not rccc.empty and int(rccc["full_regret_matches"].iloc[0]) == int(rccc["cases"].iloc[0]))
    heldout_ok = bool(not heldout.empty and heldout["match_rate"].min() >= 0.999)
    text = [
        "# TR-C strengthening audit evaluation",
        "",
        f"- Mechanism ablation usable for manuscript: {ablation_ok}",
        f"- Held-out robustness usable for manuscript: {heldout_ok}",
        "",
        "## Mechanism ablation summary",
        "",
        frame_to_text_table(ablation_summary),
        "",
        "## Held-out robustness summary",
        "",
        frame_to_text_table(heldout),
        "",
        "## Operational KPI summary",
        "",
        frame_to_text_table(operational),
    ]
    (output_dir / "strengthening_evaluation.md").write_text("\n".join(text), encoding="utf-8")


def copy_to_article_generated(output_dir: Path, generated_dir: Path) -> None:
    generated_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "mechanism_ablation_summary.csv",
        "mechanism_ablation_metrics.csv",
        "heldout_robustness_summary.csv",
        "operational_kpi_summary.csv",
        "operational_case_methods.csv",
        "operational_case_scenarios.csv",
        "operational_case_edges.csv",
        "operational_case_blocks.csv",
    ]:
        path = output_dir / name
        if path.exists():
            pd.read_csv(path).to_csv(generated_dir / name, index=False)


def run(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    generated_dir = Path(args.generated_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(args.selected_instances)
    metrics = pd.read_csv(args.main_metrics)
    weather_audit = pd.read_csv(args.weather_audit) if args.weather_audit and Path(args.weather_audit).exists() else None

    ablation_summary = _run_ablation(args, selected, metrics, weather_audit)
    heldout = _heldout_summary(metrics, weather_audit)
    heldout.to_csv(output_dir / "heldout_robustness_summary.csv", index=False)
    operational = _operational_kpi(metrics, weather_audit)
    operational.to_csv(output_dir / "operational_kpi_summary.csv", index=False)
    _run_case(args, selected, metrics, weather_audit)
    _write_evaluation(output_dir, ablation_summary, heldout, operational)
    copy_to_article_generated(output_dir, generated_dir)

    print(ablation_summary.to_string(index=False))
    print(heldout.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run supplementary strengthening audits for the TR-C manuscript.")
    parser.add_argument("--selected-instances", default=str(DEFAULT_SELECTED))
    parser.add_argument("--main-metrics", default=str(DEFAULT_MAIN_METRICS))
    parser.add_argument("--weather-audit", default=str(DEFAULT_WEATHER_AUDIT))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULT_DIR))
    parser.add_argument("--generated-dir", default=str(DEFAULT_GENERATED_DIR))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-instances", type=int, default=3)
    parser.add_argument("--ablation-instances", type=int, default=40)
    parser.add_argument("--min-per-region", type=int, default=2)
    parser.add_argument("--scenario-count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--oracle-time-limit", type=float, default=60.0)
    parser.add_argument("--ablation-per-flight", type=int, default=20)
    parser.add_argument("--ablation-per-aircraft", type=int, default=10)
    parser.add_argument("--adaptive-per-flight", type=int, default=20)
    parser.add_argument("--adaptive-per-aircraft", type=int, default=10)
    parser.add_argument("--use-weather", action="store_true")
    parser.add_argument("--weather-dir", default=str(ROOT / "data" / "weather_asos"))
    parser.add_argument("--weather-local-start-hour", type=int, default=4)
    parser.add_argument("--weather-local-end-hour", type=int, default=12)
    parser.add_argument("--weather-pause-seconds", type=float, default=3.0)
    parser.add_argument("--weather-max-attempts", type=int, default=12)
    parser.add_argument("--weather-request-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--value-mode", choices=["distance_cost", "unit_service"], default="distance_cost")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
