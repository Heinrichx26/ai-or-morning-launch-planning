from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import run_smoke_batch14_year_round as year_round
from .run_smoke_batch11_multiregion import _key
from .smoke_data import load_bts_month
from .weather_asos import fetch_airport_date_weather


def _day_choices(stride: int) -> list[int]:
    if stride <= 1:
        return list(range(1, 32))
    return list(range(1, 32, stride))


def _candidate_dates(month: int, days: list[int]) -> list[pd.Timestamp]:
    month_start = pd.Timestamp(f"{str(month)[:4]}-{str(month)[4:]}-01")
    month_end = month_start + pd.offsets.MonthEnd(0)
    return [month_start + pd.Timedelta(days=day - 1) for day in days if day <= int(month_end.day)]


def _tail_locations_by_target_date(month_df: pd.DataFrame) -> pd.DataFrame:
    sub = month_df[
        (month_df["Tail_Number"].ne(""))
        & (month_df["Cancelled"].eq(0))
        & (month_df["Diverted"].eq(0))
    ].copy()
    if sub.empty:
        return pd.DataFrame(columns=["target_date", "Reporting_Airline", "Tail_Number", "Dest"])
    sub["arr_sort"] = sub["CRSArrTime_min"]
    sub.loc[sub["arr_sort"] < sub["CRSDepTime_min"], "arr_sort"] += 24 * 60
    final = (
        sub.sort_values(["FlightDate", "Reporting_Airline", "Tail_Number", "arr_sort", "CRSDepTime_min"])
        .groupby(["FlightDate", "Reporting_Airline", "Tail_Number"])
        .tail(1)
        .copy()
    )
    final["target_date"] = final["FlightDate"] + pd.Timedelta(days=1)
    return final[["target_date", "Reporting_Airline", "Tail_Number", "Dest"]]


def _pressure_from_counts(
    demand: dict[str, int],
    inventory: dict[str, int],
    airports: tuple[str, ...],
    flight_count: int,
    inventory_count: int,
) -> dict[str, int]:
    deficits = {airport: max(0, int(demand.get(airport, 0)) - int(inventory.get(airport, 0))) for airport in airports}
    surpluses = {airport: max(0, int(inventory.get(airport, 0)) - int(demand.get(airport, 0))) for airport in airports}
    return {
        "flights": int(flight_count),
        "inventory": int(inventory_count),
        "total_deficit": int(sum(deficits.values())),
        "max_airport_deficit": int(max(deficits.values()) if deficits else 0),
        "total_surplus": int(sum(surpluses.values())),
        "demand_airports": int(sum(1 for value in demand.values() if value > 0)),
        "inventory_airports": int(sum(1 for value in inventory.values() if value > 0)),
    }


def _fast_select_instances(max_instances: int, stride: int, region_quota: int, month_quota: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    days = _day_choices(stride)
    for month in year_round.MONTHS:
        print(f"selecting month {month}", flush=True)
        month_df = load_bts_month(year_round._bts_csv(month))
        early = month_df[month_df["CRSDepTime_min"].between(6 * 60, 8 * 60 + 59)].copy()
        tail_locations = _tail_locations_by_target_date(month_df)
        for date in _candidate_dates(month, days):
            for carrier in year_round.CARRIERS:
                flights_carrier = early[
                    early["FlightDate"].eq(date) & early["Reporting_Airline"].eq(carrier)
                ]
                if flights_carrier.empty:
                    continue
                inventory_carrier = tail_locations[
                    tail_locations["target_date"].eq(date) & tail_locations["Reporting_Airline"].eq(carrier)
                ]
                if inventory_carrier.empty:
                    continue
                for region_name, airports in year_round.REGIONS.items():
                    flights_region = flights_carrier[flights_carrier["Origin"].isin(airports)]
                    if flights_region.empty:
                        continue
                    inventory_region = inventory_carrier[inventory_carrier["Dest"].isin(airports)]
                    if inventory_region.empty:
                        continue
                    demand = flights_region.groupby("Origin").size().to_dict()
                    inventory = inventory_region.groupby("Dest").size().to_dict()
                    pressure = _pressure_from_counts(
                        demand,
                        inventory,
                        airports,
                        flight_count=len(flights_region),
                        inventory_count=len(inventory_region),
                    )
                    candidate_count = int(len(flights_region) * len(inventory_region))
                    if pressure["flights"] < 8 or pressure["flights"] > 75:
                        continue
                    if pressure["inventory"] < pressure["flights"]:
                        continue
                    if pressure["total_deficit"] <= 0:
                        continue
                    if pressure["demand_airports"] < 2 or pressure["inventory_airports"] < 2:
                        continue
                    if candidate_count > 5200:
                        continue
                    row: dict[str, object] = {
                        "month": month,
                        "region": region_name,
                        "region_airports": "|".join(airports),
                        "carrier": carrier,
                        "date": str(date.date()),
                        "candidate_count_preview": candidate_count,
                        **pressure,
                    }
                    row["score"] = year_round._score(row)
                    rows.append(row)

    selected = pd.DataFrame(rows)
    if selected.empty:
        return selected
    selected = selected.sort_values("score", ascending=False)
    by_region = selected.groupby("region").head(region_quota)
    by_month = selected.groupby("month").head(month_quota)
    rest = selected[~selected.index.isin(pd.concat([by_region, by_month]).index)]
    out = pd.concat([by_region, by_month, rest.sort_values("score", ascending=False)], ignore_index=True)
    out = out.drop_duplicates(["region", "carrier", "date"]).head(max_instances)
    return out.reset_index(drop=True)


def _select_expanded_instances(
    path: Path,
    max_instances: int,
    stride: int,
    region_quota: int,
    month_quota: int,
    refresh: bool,
) -> pd.DataFrame:
    if path.exists() and not refresh:
        selected = pd.read_csv(path)
        if len(selected) >= max_instances:
            return selected.head(max_instances).reset_index(drop=True)

    selected = _fast_select_instances(
        max_instances=max_instances,
        stride=stride,
        region_quota=region_quota,
        month_quota=month_quota,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(path, index=False)
    return selected.reset_index(drop=True)


def _prior_audit(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    audit = pd.read_csv(path)
    if "error" not in audit.columns:
        audit["error"] = ""
    audit["error"] = audit["error"].fillna("")
    return audit


def _completed_keys(audit: pd.DataFrame) -> set[str]:
    if audit.empty:
        return set()
    completed = audit[(audit["weather_rows"] > 0) & audit["error"].eq("")]
    return set(completed["instance_key"].astype(str))


def run(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    weather_dir = Path(args.weather_dir)
    feature_dir = data_dir / "weather_features"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)

    selected_path = data_dir / "selected_instances.csv"
    selected = _select_expanded_instances(
        selected_path,
        max_instances=args.max_instances,
        stride=args.day_stride,
        region_quota=args.region_quota,
        month_quota=args.month_quota,
        refresh=args.refresh_selection,
    )
    if selected.empty:
        raise RuntimeError("No eligible instances were selected.")

    download_limit = args.download_limit if args.download_limit > 0 else len(selected)
    selected = selected.head(download_limit).reset_index(drop=True)

    prior = _prior_audit(output_path)
    completed = _completed_keys(prior) if args.resume and not args.refresh_weather else set()
    rows = prior.to_dict("records") if not prior.empty else []
    seen = {str(row["instance_key"]) for row in rows if "instance_key" in row}

    for idx, item in enumerate(selected.itertuples(index=False), start=1):
        key = _key(str(item.region), str(item.carrier), str(item.date))
        if key in completed:
            print(f"skip {idx}/{len(selected)} {key}: cached audit row", flush=True)
            continue

        airports = str(item.region_airports).split("|")
        error = ""
        try:
            features = fetch_airport_date_weather(
                airports=airports,
                date=str(item.date),
                output_dir=weather_dir,
                local_start_hour=args.local_start_hour,
                local_end_hour=args.local_end_hour,
                force=args.refresh_weather,
                pause_seconds=args.pause_seconds,
                max_attempts=args.max_attempts,
                request_timeout_seconds=args.request_timeout_seconds,
            )
            if not features.empty:
                features.to_csv(feature_dir / f"weather_{key}.csv", index=False)
        except Exception as exc:
            features = pd.DataFrame()
            error = f"{type(exc).__name__}: {exc}"

        row = {
            "instance_key": key,
            "region": str(item.region),
            "date": str(item.date),
            "carrier": str(item.carrier),
            "airports": str(item.region_airports),
            "weather_rows": int(len(features)),
            "adverse_hours": int(features["adverse_weather"].sum()) if not features.empty else 0,
            "max_weather_severity": float(features["weather_severity"].max()) if not features.empty else 0.0,
            "mean_weather_severity": float(features["weather_severity"].mean()) if not features.empty else 0.0,
            "error": error,
        }

        if key in seen:
            rows = [existing for existing in rows if str(existing.get("instance_key")) != key]
        rows.append(row)
        seen.add(key)
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(
            f"{idx}/{len(selected)} {key}: rows={row['weather_rows']} "
            f"adverse_hours={row['adverse_hours']} max_severity={row['max_weather_severity']:.3f}",
            flush=True,
        )

    audit = pd.DataFrame(rows)
    audit.to_csv(output_path, index=False)
    current = audit[audit["instance_key"].isin(selected.apply(lambda r: _key(str(r.region), str(r.carrier), str(r.date)), axis=1))]
    errors = int(current["error"].fillna("").ne("").sum()) if not current.empty else 0
    print(
        f"completed={len(current)} weather_rows={int(current['weather_rows'].sum()) if not current.empty else 0} "
        f"adverse_instances={int(current['adverse_hours'].gt(0).sum()) if not current.empty else 0} "
        f"errors={errors}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download expanded IEM ASOS/METAR weather features for TR-C MLAP instances.")
    parser.add_argument("--max-instances", type=int, default=360)
    parser.add_argument("--download-limit", type=int, default=0)
    parser.add_argument("--day-stride", type=int, default=1)
    parser.add_argument("--region-quota", type=int, default=30)
    parser.add_argument("--month-quota", type=int, default=8)
    parser.add_argument("--local-start-hour", type=int, default=3)
    parser.add_argument("--local-end-hour", type=int, default=15)
    parser.add_argument("--pause-seconds", type=float, default=1.2)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--request-timeout-seconds", type=float, default=45.0)
    parser.add_argument("--weather-dir", default="data/weather_asos")
    parser.add_argument("--data-dir", default="data/trc_open_smoke/batch18_weather_fullscale360")
    parser.add_argument("--output", default="results/trc_smoke/batch18_weather_audit/window_0300_1500_fullscale360_feature_audit.csv")
    parser.add_argument("--refresh-selection", action="store_true")
    parser.add_argument("--refresh-weather", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
