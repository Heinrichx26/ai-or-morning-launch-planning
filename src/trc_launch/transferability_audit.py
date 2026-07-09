from __future__ import annotations

import argparse
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .optimizers import LaunchOptimizer, compute_oracle_costs, evaluate_solution
from .risk_hardening import frame_to_text_table
from .run_smoke_batch7_filtered_regret import _filtered_instance
from .smoke_data import LaunchInstance


@dataclass(frozen=True)
class FeedSpec:
    feed_id: str
    domain: str
    agency: str
    url: str
    service_window: tuple[int, int] = (6 * 60, 9 * 60)


TRANSFER_FEEDS = [
    FeedSpec(
        feed_id="bart_rail",
        domain="heavy_rail_gtfs",
        agency="Bay Area Rapid Transit",
        url="https://www.bart.gov/dev/schedules/google_transit.zip",
    ),
    FeedSpec(
        feed_id="mbta_multimodal",
        domain="multimodal_urban_transit_gtfs",
        agency="Massachusetts Bay Transportation Authority",
        url="https://cdn.mbta.com/MBTA_GTFS.zip",
    ),
    FeedSpec(
        feed_id="trimet_multimodal",
        domain="bus_light_rail_gtfs",
        agency="TriMet",
        url="https://developer.trimet.org/schedule/gtfs.zip",
    ),
]


def gtfs_time_to_minutes(value: object) -> int:
    if pd.isna(value):
        return -1
    parts = str(value).strip().split(":")
    if len(parts) != 3:
        return -1
    try:
        hour, minute, _ = [int(part) for part in parts]
    except ValueError:
        return -1
    if minute < 0 or minute >= 60:
        return -1
    return hour * 60 + minute


def hhmm_from_minutes(minutes: int) -> int:
    minutes = int(minutes)
    return (minutes // 60) * 100 + (minutes % 60)


def download_feed(spec: FeedSpec, cache_dir: Path, force: bool = False) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{spec.feed_id}.zip"
    if force or not path.exists() or path.stat().st_size == 0:
        request = Request(
            spec.url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; TRC-transferability-audit/1.0)",
                "Accept": "application/zip,application/octet-stream,*/*",
            },
        )
        with urlopen(request, timeout=120) as response, path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    return path


def _read_gtfs_table(archive: zipfile.ZipFile, name: str, columns: list[str] | None = None) -> pd.DataFrame:
    with archive.open(name) as handle:
        return pd.read_csv(handle, usecols=columns, low_memory=False)


def _station_key(stops: pd.DataFrame) -> pd.Series:
    if "parent_station" in stops.columns:
        parent = stops["parent_station"].fillna("").astype(str)
        return parent.where(parent.ne(""), stops["stop_id"].astype(str))
    return stops["stop_id"].astype(str)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    p1, p2 = np.radians([lat1, lat2])
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2.0) ** 2
    return float(2.0 * radius * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))


def build_gtfs_launch_instance(
    zip_path: Path,
    spec: FeedSpec,
    *,
    max_services: int = 24,
    scenario_count: int = 18,
    seed: int = 127,
) -> LaunchInstance:
    with zipfile.ZipFile(zip_path) as archive:
        stops = _read_gtfs_table(
            archive,
            "stops.txt",
            ["stop_id", "stop_name", "stop_lat", "stop_lon", "parent_station"],
        )
        stop_times = _read_gtfs_table(
            archive,
            "stop_times.txt",
            ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"],
        )
        trips = _read_gtfs_table(
            archive,
            "trips.txt",
            ["route_id", "service_id", "trip_id", "trip_headsign", "direction_id", "block_id"],
        )
        routes = _read_gtfs_table(archive, "routes.txt", ["route_id", "route_type"]) if "routes.txt" in archive.namelist() else pd.DataFrame()

    stops = stops.copy()
    stops["stop_id"] = stops["stop_id"].astype(str)
    stops["stop_group"] = _station_key(stops)
    stop_lookup = stops.set_index("stop_id")[["stop_group", "stop_lat", "stop_lon"]].to_dict("index")
    trips = trips.copy()
    trips["trip_id"] = trips["trip_id"].astype(str)
    trips["route_id"] = trips["route_id"].astype(str)
    if "block_id" not in trips.columns:
        trips["block_id"] = ""
    stop_times["trip_id"] = stop_times["trip_id"].astype(str)
    stop_times["stop_id"] = stop_times["stop_id"].astype(str)
    if routes.empty:
        trips["route_type"] = -1
    else:
        routes["route_id"] = routes["route_id"].astype(str)
        trips = trips.merge(routes, on="route_id", how="left")
        trips["route_type"] = pd.to_numeric(trips["route_type"], errors="coerce").fillna(-1).astype(int)

    service_id = str(trips["service_id"].value_counts().idxmax())
    trips = trips[trips["service_id"].astype(str).eq(service_id)].copy()
    stop_times = stop_times[stop_times["trip_id"].isin(trips["trip_id"])].copy()
    if trips.empty or stop_times.empty:
        return _empty_instance(spec)

    stop_times["stop_sequence"] = pd.to_numeric(stop_times["stop_sequence"], errors="coerce").fillna(0).astype(int)
    stop_times["dep_min"] = stop_times["departure_time"].map(gtfs_time_to_minutes)
    stop_times["arr_min"] = stop_times["arrival_time"].map(gtfs_time_to_minutes)
    first = stop_times.sort_values(["trip_id", "stop_sequence"]).groupby("trip_id").head(1).copy()
    last = stop_times.sort_values(["trip_id", "stop_sequence"]).groupby("trip_id").tail(1).copy()
    bounds = first[["trip_id", "stop_id", "dep_min"]].rename(columns={"stop_id": "origin_stop", "dep_min": "first_dep_min"})
    bounds = bounds.merge(
        last[["trip_id", "stop_id", "arr_min"]].rename(columns={"stop_id": "dest_stop", "arr_min": "last_arr_min"}),
        on="trip_id",
        how="left",
    )
    bounds = bounds.merge(trips[["trip_id", "route_id", "route_type", "block_id"]], on="trip_id", how="left")
    all_bounds = bounds.copy()
    start_min, end_min = spec.service_window
    bounds = bounds[bounds["first_dep_min"].between(start_min, end_min)].copy()
    if len(bounds) < 6:
        bounds = bounds[bounds["first_dep_min"].between(5 * 60, 10 * 60)].copy()
    if bounds.empty:
        return _empty_instance(spec)

    bounds["origin_group"] = bounds["origin_stop"].map(lambda sid: stop_lookup.get(str(sid), {}).get("stop_group", str(sid)))
    bounds["dest_group"] = bounds["dest_stop"].map(lambda sid: stop_lookup.get(str(sid), {}).get("stop_group", str(sid)))
    bounds["duration_min"] = (pd.to_numeric(bounds["last_arr_min"], errors="coerce") - pd.to_numeric(bounds["first_dep_min"], errors="coerce")).clip(lower=5)
    bounds["origin_count"] = bounds.groupby("origin_group")["trip_id"].transform("count")
    bounds = bounds.sort_values(["origin_count", "duration_min", "first_dep_min"], ascending=[False, False, True])
    selected = bounds.head(max_services).sort_values(["first_dep_min", "trip_id"]).reset_index(drop=True)

    if selected.empty:
        return _empty_instance(spec)

    max_duration = max(1.0, float(selected["duration_min"].max()))
    flights = selected.copy()
    flights["flight_id"] = spec.feed_id + "_" + flights["trip_id"].astype(str)
    flights["Origin"] = flights["origin_group"].astype(str)
    flights["Dest"] = flights["dest_group"].astype(str)
    flights["CRSDepTime_min"] = flights["first_dep_min"].astype(int)
    flights["CRSDepTime"] = flights["CRSDepTime_min"].map(hhmm_from_minutes)
    flights["dep_hour"] = (flights["CRSDepTime_min"] // 60).astype(int)
    flights["block_id"] = flights["Origin"] + "_" + flights["dep_hour"].astype(str)
    flights["Distance"] = flights["duration_min"].astype(float)
    flights["weight"] = 1.0 + (flights["duration_min"].astype(float) / max_duration).clip(0.0, 1.5)
    flights = flights[
        ["flight_id", "Origin", "Dest", "CRSDepTime", "CRSDepTime_min", "dep_hour", "block_id", "Distance", "weight"]
    ].reset_index(drop=True)

    preposition = all_bounds[all_bounds["first_dep_min"].between(max(0, start_min - 2 * 60), start_min - 1)].copy()
    if preposition.empty:
        preposition = all_bounds[all_bounds["first_dep_min"].lt(start_min)].copy()
    if preposition.empty:
        preposition = selected.copy()
    preposition["origin_group"] = preposition["origin_stop"].map(
        lambda sid: stop_lookup.get(str(sid), {}).get("stop_group", str(sid))
    )
    preposition["dest_group"] = preposition["dest_stop"].map(
        lambda sid: stop_lookup.get(str(sid), {}).get("stop_group", str(sid))
    )
    preposition["duration_min"] = (
        pd.to_numeric(preposition["last_arr_min"], errors="coerce")
        - pd.to_numeric(preposition["first_dep_min"], errors="coerce")
    ).clip(lower=5)
    resource_count = max(4, int(np.floor(len(selected) * 0.82)))
    resource_count = min(resource_count, len(selected), len(preposition))
    resource_pool = preposition.sort_values(["first_dep_min", "duration_min"], ascending=[False, False]).head(resource_count)

    inventory_rows = []
    for idx, row in resource_pool.reset_index(drop=True).iterrows():
        vehicle_id = str(row.get("block_id", ""))
        if not vehicle_id or vehicle_id.lower() == "nan":
            vehicle_id = str(row["trip_id"])
        vehicle_id = f"{spec.feed_id}_veh_{idx}_{vehicle_id}"
        inventory_rows.append(
            {
                "aircraft_id": vehicle_id,
                "overnight_airport": str(row["dest_group"]),
                "aircraft_count": 1,
            }
        )
    inventory = pd.DataFrame(inventory_rows)

    station_coords = (
        stops.dropna(subset=["stop_lat", "stop_lon"])
        .groupby("stop_group")[["stop_lat", "stop_lon"]]
        .mean()
        .to_dict("index")
    )
    candidates = []
    for aircraft in inventory.itertuples(index=False):
        source = str(aircraft.overnight_airport)
        source_coord = station_coords.get(source)
        for service in flights.itertuples(index=False):
            origin = str(service.Origin)
            origin_coord = station_coords.get(origin)
            if source == origin or source_coord is None or origin_coord is None:
                ferry_cost = 0.0 if source == origin else 0.35
            else:
                km = _haversine_km(
                    float(source_coord["stop_lat"]),
                    float(source_coord["stop_lon"]),
                    float(origin_coord["stop_lat"]),
                    float(origin_coord["stop_lon"]),
                )
                ferry_cost = float(min(0.85, 0.08 + km / 45.0))
            candidates.append(
                {
                    "candidate_id": f"{aircraft.aircraft_id}->{service.flight_id}",
                    "aircraft_id": str(aircraft.aircraft_id),
                    "source_airport": source,
                    "flight_id": str(service.flight_id),
                    "origin": origin,
                    "block_id": str(service.block_id),
                    "ferry_cost": ferry_cost,
                }
            )
    scenarios = _build_transfer_scenarios(flights, scenario_count=scenario_count, seed=seed)
    return LaunchInstance(
        date=service_id,
        carrier=spec.feed_id,
        region_airports=sorted(flights["Origin"].astype(str).unique().tolist()),
        flights=flights,
        inventory=inventory,
        candidates=pd.DataFrame(candidates),
        scenarios=scenarios,
        total_weight=float(flights["weight"].sum()),
    )


def _build_transfer_scenarios(flights: pd.DataFrame, scenario_count: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    block_counts = flights.groupby("block_id").size().rename("scheduled_count").reset_index()
    origins = sorted(flights["Origin"].astype(str).unique().tolist())
    hours = sorted(flights["dep_hour"].astype(int).unique().tolist())
    scenario_types = np.array(["normal", "regional", "terminal", "hour", "surge"])
    scenario_probs = np.array([0.24, 0.24, 0.22, 0.16, 0.14])
    rows: list[dict[str, object]] = []
    for sid in range(scenario_count):
        scenario_type = str(rng.choice(scenario_types, p=scenario_probs))
        shock_origin = str(rng.choice(origins)) if origins and scenario_type == "terminal" else ""
        shock_hour = int(rng.choice(hours)) if hours and scenario_type == "hour" else -1
        if scenario_type == "normal":
            base = rng.uniform(0.86, 1.03)
        elif scenario_type == "regional":
            base = rng.uniform(0.58, 0.76)
        elif scenario_type == "surge":
            base = rng.uniform(0.46, 0.68)
        else:
            base = rng.uniform(0.74, 0.94)
        for row in block_counts.itertuples(index=False):
            block = str(row.block_id)
            demand = int(row.scheduled_count)
            origin = str(flights.loc[flights["block_id"].eq(block), "Origin"].iloc[0])
            hour = int(flights.loc[flights["block_id"].eq(block), "dep_hour"].iloc[0])
            factor = base + rng.normal(0.0, 0.035)
            if scenario_type == "terminal" and origin == shock_origin:
                factor *= rng.uniform(0.36, 0.58)
            if scenario_type == "hour" and hour == shock_hour:
                factor *= rng.uniform(0.44, 0.65)
            capacity = max(0, min(demand, int(np.floor(demand * float(np.clip(factor, 0.0, 1.05))))))
            rows.append(
                {
                    "scenario_id": sid,
                    "block_id": block,
                    "capacity": capacity,
                    "scenario_type": scenario_type,
                    "shock_airport": shock_origin,
                    "shock_hour": shock_hour,
                }
            )
    return pd.DataFrame(rows)


def _empty_instance(spec: FeedSpec) -> LaunchInstance:
    return LaunchInstance(
        date="",
        carrier=spec.feed_id,
        region_airports=[],
        flights=pd.DataFrame(),
        inventory=pd.DataFrame(columns=["aircraft_id", "overnight_airport", "aircraft_count"]),
        candidates=pd.DataFrame(),
        scenarios=pd.DataFrame(),
        total_weight=0.0,
    )


def _solve_transfer_rccc(instance: LaunchInstance, oracle_costs: dict[int, float], full_expected: float, full_regret: float, time_limit: float):
    audit_path = []
    solve_seconds = 0.0
    for per_flight, per_aircraft, label in [(4, 3, "adaptive_k4"), (8, 5, "wide_k8"), (12, 8, "wide_k12")]:
        filtered, original_count = _filtered_instance(instance, per_flight=per_flight, per_aircraft=per_aircraft)
        result = LaunchOptimizer(filtered, time_limit=time_limit).solve("regret_portfolio", oracle_costs=oracle_costs)
        metrics = evaluate_solution(instance, result, oracle_costs=oracle_costs)
        solve_seconds += float(result.solve_seconds)
        audit_path.append(label)
        if round(float(metrics["expected_cost"]), 8) == round(full_expected, 8) and round(float(metrics["max_regret"]), 8) == round(full_regret, 8):
            result.solve_seconds = solve_seconds
            return result, filtered, {
                "audit_path": "->".join(audit_path),
                "audit_expansions": len(audit_path) - 1,
                "per_flight": per_flight,
                "per_aircraft": per_aircraft,
                "original_count": original_count,
            }
    full = LaunchOptimizer(instance, time_limit=time_limit).solve("regret_portfolio", oracle_costs=oracle_costs)
    solve_seconds += float(full.solve_seconds)
    full.solve_seconds = solve_seconds
    audit_path.append("full")
    return full, instance, {
        "audit_path": "->".join(audit_path),
        "audit_expansions": len(audit_path) - 1,
        "per_flight": -1,
        "per_aircraft": -1,
        "original_count": len(instance.candidates),
    }


def run_transferability_audit(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)
    cache_dir = data_dir / "gtfs"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    feeds = TRANSFER_FEEDS[:1] if args.smoke else TRANSFER_FEEDS
    rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    for spec in feeds:
        zip_path = download_feed(spec, cache_dir, force=args.force_download)
        max_services = args.smoke_max_services if args.smoke else args.max_services
        scenario_count = args.smoke_scenarios if args.smoke else args.scenario_count
        instance = build_gtfs_launch_instance(
            zip_path,
            spec,
            max_services=max_services,
            scenario_count=scenario_count,
            seed=args.seed,
        )
        source_rows.append(
            {
                "feed_id": spec.feed_id,
                "domain": spec.domain,
                "agency": spec.agency,
                "url": spec.url,
                "service_id": instance.date,
                "service_count": int(len(instance.flights)),
                "resource_count": int(instance.inventory["aircraft_count"].sum()) if not instance.inventory.empty else 0,
                "candidate_count": int(len(instance.candidates)),
                "scenario_count": int(instance.scenarios["scenario_id"].nunique()) if not instance.scenarios.empty else 0,
                "block_count": int(instance.flights["block_id"].nunique()) if not instance.flights.empty else 0,
            }
        )
        if instance.flights.empty or instance.inventory.empty or instance.candidates.empty:
            continue

        oracle = compute_oracle_costs(instance, time_limit=args.oracle_time_limit)
        full = LaunchOptimizer(instance, time_limit=args.time_limit).solve("regret_portfolio", oracle_costs=oracle)
        full_metrics = evaluate_solution(instance, full, oracle_costs=oracle)
        full_expected = float(full_metrics["expected_cost"])
        full_regret = float(full_metrics["max_regret"])
        _append_metrics(rows, spec, instance, full, oracle, "regret_portfolio_full", len(instance.candidates), len(instance.candidates), full_expected, full_regret)

        rccc, rccc_instance, audit_extra = _solve_transfer_rccc(instance, oracle, full_expected, full_regret, args.time_limit)
        _append_metrics(
            rows,
            spec,
            instance,
            rccc,
            oracle,
            "rccc_audited",
            len(rccc_instance.candidates),
            len(instance.candidates),
            full_expected,
            full_regret,
            audit_extra,
        )

        for method in ["envelope_regret_portfolio", "saa_extensive", "cvar_extensive", "minimax_extensive", "active_scenario_saa"]:
            if method == "envelope_regret_portfolio":
                result = LaunchOptimizer(instance, time_limit=args.time_limit).solve(method, oracle_costs=oracle)
            else:
                result = LaunchOptimizer(instance, time_limit=args.time_limit).solve(method)
            _append_metrics(rows, spec, instance, result, oracle, method, len(instance.candidates), len(instance.candidates), full_expected, full_regret)

        pd.DataFrame(rows).to_csv(output_dir / "transferability_metrics_checkpoint.csv", index=False)

    metrics = pd.DataFrame(rows)
    sources = pd.DataFrame(source_rows)
    summary = summarize_transferability(metrics)
    metrics.to_csv(output_dir / "transferability_metrics.csv", index=False)
    sources.to_csv(output_dir / "transferability_sources.csv", index=False)
    summary.to_csv(output_dir / "transferability_summary.csv", index=False)
    article_generated = Path(args.article_generated_dir)
    article_generated.mkdir(parents=True, exist_ok=True)
    sources.to_csv(article_generated / "transferability_sources.csv", index=False)
    summary.to_csv(article_generated / "transferability_summary.csv", index=False)
    text = [
        "# Transferability audit evaluation",
        "",
        f"- Public feeds: {len(sources)}",
        f"- Raw method-domain rows: {len(metrics)}",
        f"- Scenario count: {args.smoke_scenarios if args.smoke else args.scenario_count}",
        "",
        "## Source audit",
        "",
        frame_to_text_table(sources),
        "",
        "## Summary",
        "",
        frame_to_text_table(summary),
    ]
    (output_dir / "transferability_evaluation.md").write_text("\n".join(text), encoding="utf-8")
    return metrics, summary, sources


def _append_metrics(
    rows: list[dict[str, object]],
    spec: FeedSpec,
    instance: LaunchInstance,
    result,
    oracle_costs: dict[int, float],
    method: str,
    candidate_count: int,
    original_count: int,
    full_expected: float,
    full_regret: float,
    extra: dict[str, object] | None = None,
) -> None:
    metrics = evaluate_solution(instance, result, oracle_costs=oracle_costs)
    metrics["method"] = method
    metrics["feed_id"] = spec.feed_id
    metrics["domain"] = spec.domain
    metrics["agency"] = spec.agency
    metrics["instance_key"] = spec.feed_id
    metrics["candidate_count"] = int(candidate_count)
    metrics["candidate_retention"] = candidate_count / max(1, original_count)
    metrics["block_count"] = int(instance.flights["block_id"].nunique()) if not instance.flights.empty else 0
    metrics["scenario_count"] = int(instance.scenarios["scenario_id"].nunique()) if not instance.scenarios.empty else 0
    metrics["full_expected_cost"] = full_expected
    metrics["full_max_regret"] = full_regret
    metrics["matches_full_cost"] = round(float(metrics["expected_cost"]), 8) == round(full_expected, 8)
    metrics["matches_full_regret"] = round(float(metrics["max_regret"]), 8) == round(full_regret, 8)
    metrics["binary_var_count"] = int(result.extra.get("binary_var_count", candidate_count + metrics["scenario_count"] * len(instance.flights)))
    metrics["continuous_var_count"] = int(result.extra.get("continuous_var_count", 0))
    metrics["envelope_cut_count"] = int(result.extra.get("envelope_cut_count", 0))
    if extra:
        metrics.update(extra)
    rows.append(metrics)


def summarize_transferability(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    return (
        metrics.groupby("method")
        .agg(
            domains=("domain", "nunique"),
            cases=("instance_key", "nunique"),
            mean_services=("flight_count", "mean"),
            mean_candidates=("candidate_count", "mean"),
            mean_candidate_retention=("candidate_retention", "mean"),
            mean_max_regret=("max_regret", "mean"),
            worst_case_regret=("max_regret", "max"),
            mean_expected_cost=("expected_cost", "mean"),
            mean_solve_seconds=("solve_seconds", "mean"),
            full_regret_matches=("matches_full_regret", "sum"),
            full_cost_matches=("matches_full_cost", "sum"),
            mean_binary_vars=("binary_var_count", "mean"),
        )
        .reset_index()
        .sort_values(["mean_max_regret", "mean_candidate_retention", "mean_solve_seconds"])
    )


def smoke_args(**overrides) -> argparse.Namespace:
    defaults = {
        "data_dir": "data/trc_open_smoke/batch27_transferability",
        "output_dir": "results/trc_smoke/batch27_transferability",
        "article_generated_dir": "article/trc_elsarticle/generated",
        "smoke": True,
        "max_services": 24,
        "smoke_max_services": 10,
        "scenario_count": 18,
        "smoke_scenarios": 6,
        "seed": 127,
        "time_limit": 60.0,
        "oracle_time_limit": 30.0,
        "force_download": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cross-domain open-data transferability audit.")
    parser.add_argument("--data-dir", default="data/trc_open_smoke/batch27_transferability")
    parser.add_argument("--output-dir", default="results/trc_smoke/batch27_transferability")
    parser.add_argument("--article-generated-dir", default="article/trc_elsarticle/generated")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-services", type=int, default=24)
    parser.add_argument("--smoke-max-services", type=int, default=10)
    parser.add_argument("--scenario-count", type=int, default=18)
    parser.add_argument("--smoke-scenarios", type=int, default=6)
    parser.add_argument("--seed", type=int, default=127)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--oracle-time-limit", type=float, default=30.0)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    _, summary, sources = run_transferability_audit(args)
    print(frame_to_text_table(sources))
    print()
    print(frame_to_text_table(summary))


if __name__ == "__main__":
    main()
