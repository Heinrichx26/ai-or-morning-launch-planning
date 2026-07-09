from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .optimizers import LaunchOptimizer, compute_oracle_costs, evaluate_solution
from .run_smoke_batch12_adaptive_filter import _adaptive_width
from .run_smoke_batch7_filtered_regret import _filtered_instance
from .run_smoke_batch11_multiregion import _bts_csv
from .smoke_data import LaunchInstance, build_tail_launch_instance, load_bts_month


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SELECTED = ROOT / "data" / "trc_open_smoke" / "batch20_weather_full360_fullquality" / "selected_instances.csv"
DEFAULT_METRICS = ROOT / "results" / "trc_smoke" / "batch20_weather_full360_fullquality" / "batch20_metrics_with_rccc_audited.csv"
DEFAULT_WEATHER_FEATURES = ROOT / "data" / "trc_open_smoke" / "batch20_weather_full360_fullquality" / "weather_features"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "trc_smoke" / "batch30_weather_event_audit"
DEFAULT_GENERATED_DIR = ROOT / "article" / "trc_elsarticle" / "generated"


def _instance_key(region: str, carrier: str, date: str) -> str:
    return f"{region}_{carrier}_{date}"


def _weather_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _event_score(row: dict[str, object]) -> float:
    disruption = (
        float(row["dep_delay15_share"])
        + 1.5 * float(row["cancel_share"])
        + 0.5 * float(row["dep_delay30_share"])
    )
    return (
        0.30 * float(row["adverse_station_hours"])
        + 2.00 * float(row["max_weather_severity"])
        + 3.00 * disruption
        + float(row["mean_dep_delay_min"]) / 120.0
        + 0.03 * float(row["total_deficit"])
    )


def build_event_audit(
    selected: pd.DataFrame,
    metrics: pd.DataFrame,
    weather_feature_dir: Path,
) -> pd.DataFrame:
    month_cache: dict[int, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []

    for item in selected.itertuples(index=False):
        month = int(item.month)
        if month not in month_cache:
            month_cache[month] = load_bts_month(_bts_csv(month))
        month_df = month_cache[month]
        date = pd.Timestamp(str(item.date))
        airports = str(item.region_airports).split("|")
        flights = month_df[
            month_df["FlightDate"].eq(date)
            & month_df["Reporting_Airline"].eq(str(item.carrier))
            & month_df["Origin"].isin(airports)
            & month_df["CRSDepTime_min"].between(6 * 60, 8 * 60 + 59)
        ].copy()
        if flights.empty:
            continue

        key = _instance_key(str(item.region), str(item.carrier), str(item.date))
        weather = _weather_features(weather_feature_dir / f"weather_{key}.csv")
        if weather.empty:
            adverse_station_hours = 0
            max_weather_severity = 0.0
            mean_weather_severity = 0.0
            adverse_airports = ""
        else:
            weather["station"] = weather["station"].astype(str).str.upper()
            weather["adverse_weather"] = pd.to_numeric(weather["adverse_weather"], errors="coerce").fillna(0).astype(int)
            weather["weather_severity"] = pd.to_numeric(weather["weather_severity"], errors="coerce").fillna(0.0)
            adverse = weather[weather["adverse_weather"].eq(1)]
            adverse_station_hours = int(adverse.shape[0])
            max_weather_severity = float(weather["weather_severity"].max())
            mean_weather_severity = float(weather["weather_severity"].mean())
            adverse_airports = "|".join(sorted(adverse["station"].dropna().unique().tolist()))

        row: dict[str, object] = {
            "instance_key": key,
            "region": str(item.region),
            "carrier": str(item.carrier),
            "date": str(item.date),
            "flights": int(flights.shape[0]),
            "adverse_station_hours": adverse_station_hours,
            "max_weather_severity": max_weather_severity,
            "mean_weather_severity": mean_weather_severity,
            "adverse_airports": adverse_airports,
            "dep_delay15_share": float((flights["DepDelayMinutes"] >= 15).mean()),
            "dep_delay30_share": float((flights["DepDelayMinutes"] >= 30).mean()),
            "mean_dep_delay_min": float(flights["DepDelayMinutes"].mean()),
            "cancel_share": float(flights["Cancelled"].mean()),
            "diverted_share": float(flights["Diverted"].mean()) if "Diverted" in flights else 0.0,
            "total_deficit": int(item.total_deficit),
            "max_airport_deficit": int(item.max_airport_deficit),
            "candidate_count_preview": int(item.candidate_count_preview),
        }
        row["event_score"] = _event_score(row)
        rows.append(row)

    audit = pd.DataFrame(rows)
    if audit.empty:
        return audit

    method_map = {
        "rccc_audited": "rccc",
        "regret_portfolio_full": "full",
        "saa_extensive": "saa",
        "cvar_extensive": "cvar",
        "minimax_extensive": "minimax",
    }
    for method, prefix in method_map.items():
        sub = metrics[metrics["method"].eq(method)].copy()
        if sub.empty:
            continue
        keep_cols = ["instance_key", "max_regret"]
        rename = {"max_regret": f"{prefix}_max_regret"}
        if method == "rccc_audited":
            keep_cols.extend(["candidate_retention", "matches_full_regret", "audit_expansions"])
            rename.update(
                {
                    "candidate_retention": "rccc_candidate_retention",
                    "matches_full_regret": "rccc_matches_full_regret",
                    "audit_expansions": "rccc_audit_expansions",
                }
            )
        audit = audit.merge(sub[keep_cols].rename(columns=rename), on="instance_key", how="left")

    for prefix in ["saa", "cvar", "minimax"]:
        col = f"{prefix}_max_regret"
        if col in audit.columns:
            audit[f"{prefix}_regret_gap"] = audit[col] - audit["rccc_max_regret"]

    return audit.sort_values("event_score", ascending=False).reset_index(drop=True)


def select_weather_events(audit: pd.DataFrame, case_count: int = 3) -> pd.DataFrame:
    if audit.empty:
        return audit
    candidates = audit[
        audit["adverse_station_hours"].gt(0)
        & (audit["dep_delay15_share"].ge(0.20) | audit["cancel_share"].ge(0.05))
        & audit["rccc_max_regret"].fillna(0).gt(1e-9)
        & (
            audit.get("saa_regret_gap", pd.Series(0.0, index=audit.index)).fillna(0).gt(1e-9)
            | audit.get("cvar_regret_gap", pd.Series(0.0, index=audit.index)).fillna(0).gt(1e-9)
        )
    ].copy()
    if candidates.empty:
        candidates = audit.copy()
    candidates = candidates.sort_values(
        ["event_score", "saa_regret_gap", "cvar_regret_gap"],
        ascending=[False, False, False],
    )

    selected_rows: list[pd.Series] = []
    used_regions: set[str] = set()
    for _, row in candidates.iterrows():
        region = str(row["region"])
        if region in used_regions:
            continue
        selected_rows.append(row)
        used_regions.add(region)
        if len(selected_rows) >= case_count:
            break
    if len(selected_rows) < case_count:
        used_keys = {str(row["instance_key"]) for row in selected_rows}
        for _, row in candidates.iterrows():
            if str(row["instance_key"]) in used_keys:
                continue
            selected_rows.append(row)
            used_keys.add(str(row["instance_key"]))
            if len(selected_rows) >= case_count:
                break
    if not selected_rows:
        return candidates.head(case_count).reset_index(drop=True)
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def _build_case_instance(row: pd.Series, month_df: pd.DataFrame, weather_feature_dir: Path, seed: int, scenario_count: int) -> LaunchInstance:
    key = str(row["instance_key"])
    weather = _weather_features(weather_feature_dir / f"weather_{key}.csv")
    return build_tail_launch_instance(
        month_df,
        date=str(row["date"]),
        carrier=str(row["carrier"]),
        region_airports=str(row["region_airports"]).split("|"),
        scenario_count=scenario_count,
        scenario_mode="stress",
        seed=seed,
        weather_features=weather,
    )


def _solve_audited_case(
    instance: LaunchInstance,
    row: pd.Series,
    time_limit: float,
    oracle_time_limit: float,
) -> tuple[object, LaunchInstance, dict[str, object], dict[int, float], dict[str, float | str]]:
    oracle = compute_oracle_costs(instance, time_limit=oracle_time_limit)
    full = LaunchOptimizer(instance, time_limit=time_limit).solve("regret_portfolio", oracle_costs=oracle)
    full_metrics = evaluate_solution(instance, full, oracle_costs=oracle)
    full_expected = float(full_metrics["expected_cost"])
    full_regret = float(full_metrics["max_regret"])
    pressure = {
        "total_deficit": int(row["total_deficit"]),
        "max_airport_deficit": int(row["max_airport_deficit"]),
    }
    per_flight, per_aircraft, width_label = _adaptive_width(instance, pressure)
    adaptive_instance, _ = _filtered_instance(instance, per_flight=per_flight, per_aircraft=per_aircraft)
    adaptive = LaunchOptimizer(adaptive_instance, time_limit=time_limit).solve("regret_portfolio", oracle_costs=oracle)
    adaptive_metrics = evaluate_solution(instance, adaptive, oracle_costs=oracle)
    audit_path = [f"adaptive:{width_label}"]
    solve_seconds = float(full.solve_seconds) + float(adaptive.solve_seconds)

    def matches(metrics: dict[str, float | str]) -> bool:
        return (
            round(float(metrics["max_regret"]), 8) == round(full_regret, 8)
            and round(float(metrics["expected_cost"]), 8) == round(full_expected, 8)
        )

    if matches(adaptive_metrics):
        adaptive.solve_seconds = solve_seconds
        return adaptive, adaptive_instance, {
            "audit_path": "->".join(audit_path),
            "audit_expansions": 0,
            "width_label": width_label,
            "per_flight": per_flight,
            "per_aircraft": per_aircraft,
        }, oracle, full_metrics

    fixed_instance, _ = _filtered_instance(instance, per_flight=36, per_aircraft=30)
    fixed = LaunchOptimizer(fixed_instance, time_limit=time_limit).solve("regret_portfolio", oracle_costs=oracle)
    fixed_metrics = evaluate_solution(instance, fixed, oracle_costs=oracle)
    audit_path.append("fixed_k36")
    solve_seconds += float(fixed.solve_seconds)
    if matches(fixed_metrics):
        fixed.solve_seconds = solve_seconds
        return fixed, fixed_instance, {
            "audit_path": "->".join(audit_path),
            "audit_expansions": 1,
            "width_label": width_label,
            "per_flight": 36,
            "per_aircraft": 30,
        }, oracle, full_metrics

    full.solve_seconds = solve_seconds
    audit_path.append("full")
    return full, instance, {
        "audit_path": "->".join(audit_path),
        "audit_expansions": 2,
        "width_label": width_label,
        "per_flight": -1,
        "per_aircraft": -1,
    }, oracle, full_metrics


def _block_exposure(row: pd.Series, month_df: pd.DataFrame, weather_feature_dir: Path) -> pd.DataFrame:
    date = pd.Timestamp(str(row["date"]))
    airports = str(row["region_airports"]).split("|")
    flights = month_df[
        month_df["FlightDate"].eq(date)
        & month_df["Reporting_Airline"].eq(str(row["carrier"]))
        & month_df["Origin"].isin(airports)
        & month_df["CRSDepTime_min"].between(6 * 60, 8 * 60 + 59)
    ].copy()
    if flights.empty:
        return pd.DataFrame()
    flights["dep_hour"] = (flights["CRSDepTime_min"] // 60).astype(int)
    flights["block_id"] = flights["Origin"].astype(str) + "_" + flights["dep_hour"].astype(str)
    flights["delay15"] = (flights["DepDelayMinutes"] >= 15).astype(float)
    flights["delay30"] = (flights["DepDelayMinutes"] >= 30).astype(float)
    block = (
        flights.groupby(["block_id", "Origin", "dep_hour"])
        .agg(
            scheduled_flights=("Flight_Number_Reporting_Airline", "size"),
            delay15_share=("delay15", "mean"),
            delay30_share=("delay30", "mean"),
            cancel_share=("Cancelled", "mean"),
            mean_dep_delay_min=("DepDelayMinutes", "mean"),
        )
        .reset_index()
        .rename(columns={"Origin": "airport", "dep_hour": "hour"})
    )
    weather = _weather_features(weather_feature_dir / f"weather_{row['instance_key']}.csv")
    if weather.empty:
        block["weather_severity"] = 0.0
        block["adverse_weather"] = 0
    else:
        weather = weather.copy()
        weather["station"] = weather["station"].astype(str).str.upper()
        weather["local_hour"] = pd.to_numeric(weather["local_hour"], errors="coerce").fillna(-1).astype(int)
        weather["weather_severity"] = pd.to_numeric(weather["weather_severity"], errors="coerce").fillna(0.0)
        weather["adverse_weather"] = pd.to_numeric(weather["adverse_weather"], errors="coerce").fillna(0).astype(int)
        wx = (
            weather.groupby(["station", "local_hour"])
            .agg(weather_severity=("weather_severity", "max"), adverse_weather=("adverse_weather", "max"))
            .reset_index()
            .rename(columns={"station": "airport", "local_hour": "hour"})
        )
        block = block.merge(wx, on=["airport", "hour"], how="left")
        block["weather_severity"] = block["weather_severity"].fillna(0.0)
        block["adverse_weather"] = block["adverse_weather"].fillna(0).astype(int)
    block["exposure_score"] = (
        2.0 * block["weather_severity"].astype(float)
        + 1.2 * block["delay15_share"].astype(float)
        + 1.8 * block["cancel_share"].astype(float)
        + block["mean_dep_delay_min"].astype(float) / 120.0
    )
    return block.sort_values("exposure_score", ascending=False).reset_index(drop=True)


def _selected_cross_airport(result) -> pd.DataFrame:
    if result.selected.empty:
        return pd.DataFrame()
    selected = result.selected.copy()
    return selected[selected["source_airport"].astype(str).ne(selected["origin"].astype(str))].copy()


def build_case_validation(
    selected_events: pd.DataFrame,
    selected_instances: pd.DataFrame,
    weather_feature_dir: Path,
    seed: int,
    scenario_count: int,
    time_limit: float,
    oracle_time_limit: float,
) -> pd.DataFrame:
    if selected_events.empty:
        return pd.DataFrame()
    merged = selected_events.merge(
        selected_instances[["month", "region", "region_airports", "carrier", "date", "total_deficit", "max_airport_deficit"]],
        on=["region", "carrier", "date", "total_deficit", "max_airport_deficit"],
        how="left",
    )
    rows: list[dict[str, object]] = []
    month_cache: dict[int, pd.DataFrame] = {}
    for row in merged.itertuples(index=False):
        item = pd.Series(row._asdict())
        month = int(item["month"])
        if month not in month_cache:
            month_cache[month] = load_bts_month(_bts_csv(month))
        month_df = month_cache[month]
        instance = _build_case_instance(item, month_df, weather_feature_dir, seed, scenario_count)
        if instance.flights.empty or instance.candidates.empty:
            continue
        result, audited_instance, audit_extra, oracle, full_metrics = _solve_audited_case(
            instance,
            item,
            time_limit=time_limit,
            oracle_time_limit=oracle_time_limit,
        )
        rccc_metrics = evaluate_solution(instance, result, oracle_costs=oracle)
        exposure = _block_exposure(item, month_df, weather_feature_dir)
        exposed = exposure[
            exposure["adverse_weather"].eq(1)
            | exposure["delay15_share"].ge(0.20)
            | exposure["cancel_share"].ge(0.05)
        ].copy()
        if exposed.empty:
            exposed = exposure.head(3).copy()
        exposed_blocks = set(exposed["block_id"].astype(str).tolist())
        retained_cross = audited_instance.candidates[
            audited_instance.candidates["source_airport"].astype(str).ne(audited_instance.candidates["origin"].astype(str))
            & audited_instance.candidates["block_id"].astype(str).isin(exposed_blocks)
        ].copy()
        selected_cross = _selected_cross_airport(result)
        selected_exposed = selected_cross[selected_cross["block_id"].astype(str).isin(exposed_blocks)].copy()
        top_blocks = exposed.head(3)
        top_block_text = "; ".join(
            f"{str(r.airport)}-{int(r.hour):02d}h"
            f" wx={float(r.weather_severity):.2f}"
            f" d15={100.0 * float(r.delay15_share):.0f}%"
            f" can={100.0 * float(r.cancel_share):.0f}%"
            for r in top_blocks.itertuples(index=False)
        )
        selected_pairs = "; ".join(
            sorted(
                {
                    f"{str(r.source_airport)}->{str(r.origin)}-{str(r.block_id).split('_')[-1]}h"
                    for r in selected_exposed.itertuples(index=False)
                }
            )
        )
        retained_pairs = "; ".join(
            sorted(
                {
                    f"{str(r.source_airport)}->{str(r.origin)}-{str(r.block_id).split('_')[-1]}h"
                    for r in retained_cross.itertuples(index=False)
                }
            )[:6]
        )
        rows.append(
            {
                "instance_key": str(item["instance_key"]),
                "region": str(item["region"]),
                "carrier": str(item["carrier"]),
                "date": str(item["date"]),
                "flights": int(len(instance.flights)),
                "exposed_blocks": top_block_text,
                "retained_cross_airport_to_exposed_blocks": int(len(retained_cross)),
                "selected_cross_airport_to_exposed_blocks": int(len(selected_exposed)),
                "retained_cross_airport_pairs": retained_pairs,
                "selected_cross_airport_pairs": selected_pairs,
                "candidate_retention": len(audited_instance.candidates) / max(1, len(instance.candidates)),
                "rccc_max_regret": float(rccc_metrics["max_regret"]),
                "full_max_regret": float(full_metrics["max_regret"]),
                "matches_full_regret": round(float(rccc_metrics["max_regret"]), 8) == round(float(full_metrics["max_regret"]), 8),
                **audit_extra,
            }
        )
    return pd.DataFrame(rows)


def choose_validated_events(
    selected_events: pd.DataFrame,
    case_validation: pd.DataFrame,
    case_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if selected_events.empty or case_validation.empty:
        return selected_events.head(case_count).reset_index(drop=True), case_validation
    validation = case_validation.copy()
    validation["_has_selected_cross"] = validation["selected_cross_airport_to_exposed_blocks"].fillna(0).astype(float).gt(0)
    validation = validation.sort_values(
        ["_has_selected_cross", "selected_cross_airport_to_exposed_blocks", "retained_cross_airport_to_exposed_blocks"],
        ascending=[False, False, False],
    )
    chosen_keys: list[str] = []
    used_regions: set[str] = set()
    for row in validation.itertuples(index=False):
        region = str(getattr(row, "region"))
        key = str(getattr(row, "instance_key"))
        if region in used_regions:
            continue
        chosen_keys.append(key)
        used_regions.add(region)
        if len(chosen_keys) >= case_count:
            break
    if len(chosen_keys) < case_count:
        for row in validation.itertuples(index=False):
            key = str(getattr(row, "instance_key"))
            if key in chosen_keys:
                continue
            chosen_keys.append(key)
            if len(chosen_keys) >= case_count:
                break
    selected_final = selected_events[selected_events["instance_key"].astype(str).isin(chosen_keys)].copy()
    selected_final["_order"] = selected_final["instance_key"].astype(str).map({key: idx for idx, key in enumerate(chosen_keys)})
    selected_final = selected_final.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    validation_final = case_validation[case_validation["instance_key"].astype(str).isin(chosen_keys)].copy()
    validation_final["_order"] = validation_final["instance_key"].astype(str).map({key: idx for idx, key in enumerate(chosen_keys)})
    validation_final = validation_final.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    return selected_final, validation_final


def _write_outputs(
    audit: pd.DataFrame,
    selected: pd.DataFrame,
    case_validation: pd.DataFrame,
    output_dir: Path,
    generated_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_dir / "weather_event_audit_full.csv", index=False)
    selected.to_csv(output_dir / "weather_event_audit_selected.csv", index=False)
    audit.to_csv(generated_dir / "weather_event_audit_full.csv", index=False)
    selected.to_csv(generated_dir / "weather_event_audit_selected.csv", index=False)
    if not case_validation.empty:
        case_validation.to_csv(output_dir / "weather_event_case_validation.csv", index=False)
        case_validation.to_csv(generated_dir / "weather_event_case_validation.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a historical weather-event audit from completed benchmark data.")
    parser.add_argument("--selected", default=str(DEFAULT_SELECTED))
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS))
    parser.add_argument("--weather-feature-dir", default=str(DEFAULT_WEATHER_FEATURES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--generated-dir", default=str(DEFAULT_GENERATED_DIR))
    parser.add_argument("--case-count", type=int, default=3)
    parser.add_argument("--validation-pool-size", type=int, default=6)
    parser.add_argument("--scenario-count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--oracle-time-limit", type=float, default=120.0)
    parser.add_argument("--skip-case-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = pd.read_csv(args.selected)
    metrics = pd.read_csv(args.metrics)
    audit = build_event_audit(selected, metrics, Path(args.weather_feature_dir))
    selected_events = select_weather_events(audit, case_count=max(args.case_count, args.validation_pool_size))
    if args.skip_case_validation:
        case_validation = pd.DataFrame()
        selected_events = selected_events.head(args.case_count).reset_index(drop=True)
    else:
        case_validation = build_case_validation(
            selected_events,
            selected,
            Path(args.weather_feature_dir),
            seed=args.seed,
            scenario_count=args.scenario_count,
            time_limit=args.time_limit,
            oracle_time_limit=args.oracle_time_limit,
        )
        selected_events, case_validation = choose_validated_events(selected_events, case_validation, args.case_count)
    _write_outputs(audit, selected_events, case_validation, Path(args.output_dir), Path(args.generated_dir))
    display_cols = [
        "instance_key",
        "flights",
        "adverse_station_hours",
        "max_weather_severity",
        "dep_delay15_share",
        "cancel_share",
        "rccc_candidate_retention",
        "rccc_max_regret",
        "saa_regret_gap",
        "cvar_regret_gap",
    ]
    print(selected_events[display_cols].to_string(index=False))
    if not case_validation.empty:
        print(case_validation[[
            "instance_key",
            "retained_cross_airport_to_exposed_blocks",
            "selected_cross_airport_to_exposed_blocks",
            "candidate_retention",
            "matches_full_regret",
            "audit_path",
        ]].to_string(index=False))


if __name__ == "__main__":
    main()
