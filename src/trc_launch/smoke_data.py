from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BTS_COLUMNS = [
    "FlightDate",
    "Reporting_Airline",
    "Tail_Number",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "CRSArrTime",
    "DepDelayMinutes",
    "ArrDelayMinutes",
    "Cancelled",
    "Diverted",
    "Distance",
]


@dataclass
class LaunchInstance:
    date: str
    carrier: str
    region_airports: list[str]
    flights: pd.DataFrame
    inventory: pd.DataFrame
    candidates: pd.DataFrame
    scenarios: pd.DataFrame
    total_weight: float


def hhmm_to_minutes(value: object) -> int:
    if pd.isna(value):
        return -1
    try:
        number = int(value)
    except (TypeError, ValueError):
        return -1
    hour = number // 100
    minute = number % 100
    if hour < 0 or hour > 29 or minute < 0 or minute >= 60:
        return -1
    return hour * 60 + minute


def load_bts_month(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=BTS_COLUMNS, low_memory=False)
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])
    for col in ["CRSDepTime", "CRSArrTime"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col + "_min"] = df[col].map(hhmm_to_minutes)
    for col in ["Cancelled", "Diverted", "Distance", "DepDelayMinutes", "ArrDelayMinutes"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["Tail_Number"] = df["Tail_Number"].fillna("").astype(str)
    return df


def _flight_id(row: pd.Series) -> str:
    return (
        f"{row.FlightDate:%Y%m%d}_{row.Reporting_Airline}_"
        f"{int(row.Flight_Number_Reporting_Airline)}_{row.Origin}_{row.Dest}_"
        f"{int(row.CRSDepTime)}"
    )


def _build_inventory(
    month_df: pd.DataFrame,
    date: pd.Timestamp,
    carrier: str,
    region_airports: Iterable[str],
) -> pd.DataFrame:
    previous_day = date - pd.Timedelta(days=1)
    region = set(region_airports)
    sub = month_df[
        (month_df["FlightDate"].eq(previous_day))
        & (month_df["Reporting_Airline"].eq(carrier))
        & (month_df["Tail_Number"].ne(""))
        & (month_df["Cancelled"].eq(0))
        & (month_df["Diverted"].eq(0))
    ].copy()
    if sub.empty:
        return pd.DataFrame(columns=["overnight_airport", "aircraft_count"])

    sub["arr_sort"] = sub["CRSArrTime_min"]
    sub.loc[sub["arr_sort"] < sub["CRSDepTime_min"], "arr_sort"] += 24 * 60
    final = sub.sort_values(["Tail_Number", "arr_sort", "CRSDepTime_min"]).groupby("Tail_Number").tail(1)
    final = final[final["Dest"].isin(region)]
    inv = (
        final.groupby("Dest")
        .size()
        .rename("aircraft_count")
        .reset_index()
        .rename(columns={"Dest": "overnight_airport"})
    )
    return inv.sort_values("overnight_airport").reset_index(drop=True)


def _build_tail_inventory(
    month_df: pd.DataFrame,
    date: pd.Timestamp,
    carrier: str,
    region_airports: Iterable[str],
) -> pd.DataFrame:
    previous_day = date - pd.Timedelta(days=1)
    region = set(region_airports)
    sub = month_df[
        (month_df["FlightDate"].eq(previous_day))
        & (month_df["Reporting_Airline"].eq(carrier))
        & (month_df["Tail_Number"].ne(""))
        & (month_df["Cancelled"].eq(0))
        & (month_df["Diverted"].eq(0))
    ].copy()
    if sub.empty:
        return pd.DataFrame(columns=["aircraft_id", "overnight_airport", "last_arrival_min"])

    sub["arr_sort"] = sub["CRSArrTime_min"]
    sub.loc[sub["arr_sort"] < sub["CRSDepTime_min"], "arr_sort"] += 24 * 60
    final = sub.sort_values(["Tail_Number", "arr_sort", "CRSDepTime_min"]).groupby("Tail_Number").tail(1)
    final = final[final["Dest"].isin(region)].copy()
    if final.empty:
        return pd.DataFrame(columns=["aircraft_id", "overnight_airport", "last_arrival_min"])
    out = final[["Tail_Number", "Dest", "arr_sort"]].rename(
        columns={
            "Tail_Number": "aircraft_id",
            "Dest": "overnight_airport",
            "arr_sort": "last_arrival_min",
        }
    )
    return out.sort_values(["overnight_airport", "aircraft_id"]).reset_index(drop=True)


def _build_flights(
    month_df: pd.DataFrame,
    date: pd.Timestamp,
    carrier: str,
    region_airports: Iterable[str],
) -> pd.DataFrame:
    region = set(region_airports)
    flights = month_df[
        (month_df["FlightDate"].eq(date))
        & (month_df["Reporting_Airline"].eq(carrier))
        & (month_df["Origin"].isin(region))
        & (month_df["CRSDepTime_min"].between(6 * 60, 8 * 60 + 59))
    ].copy()
    if flights.empty:
        return pd.DataFrame()
    flights = flights.reset_index(drop=True)
    flights["flight_id"] = flights.apply(_flight_id, axis=1)
    flights["dep_hour"] = (flights["CRSDepTime_min"] // 60).astype(int)
    flights["block_id"] = flights["Origin"] + "_" + flights["dep_hour"].astype(str)
    flights["weight"] = 1.0 + (flights["Distance"].fillna(0.0) / 1800.0).clip(0, 2.0)
    keep = [
        "flight_id",
        "FlightDate",
        "Reporting_Airline",
        "Flight_Number_Reporting_Airline",
        "Origin",
        "Dest",
        "CRSDepTime",
        "CRSDepTime_min",
        "dep_hour",
        "block_id",
        "Distance",
        "weight",
        "Cancelled",
    ]
    return flights[keep].sort_values(["Origin", "CRSDepTime_min", "flight_id"]).reset_index(drop=True)


def _ferry_cost(source: str, origin: str) -> float:
    if source == origin:
        return 0.0
    # The smoke model keeps the units normalized to one flight's service value.
    special = {
        ("JFK", "LGA"): 0.28,
        ("LGA", "JFK"): 0.28,
        ("JFK", "EWR"): 0.36,
        ("EWR", "JFK"): 0.36,
        ("LGA", "EWR"): 0.32,
        ("EWR", "LGA"): 0.32,
    }
    return special.get((source, origin), 0.35)


def _build_candidates(flights: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if flights.empty or inventory.empty:
        return pd.DataFrame(columns=["candidate_id", "source_airport", "flight_id", "ferry_cost"])
    for source in inventory["overnight_airport"]:
        for flight in flights.itertuples(index=False):
            rows.append(
                {
                    "candidate_id": f"{source}->{flight.flight_id}",
                    "source_airport": source,
                    "flight_id": flight.flight_id,
                    "origin": flight.Origin,
                    "block_id": flight.block_id,
                    "ferry_cost": _ferry_cost(source, flight.Origin),
                }
            )
    return pd.DataFrame(rows)


def _build_tail_candidates(flights: pd.DataFrame, tail_inventory: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if flights.empty or tail_inventory.empty:
        return pd.DataFrame(
            columns=["candidate_id", "aircraft_id", "source_airport", "flight_id", "origin", "block_id", "ferry_cost"]
        )
    for aircraft in tail_inventory.itertuples(index=False):
        for flight in flights.itertuples(index=False):
            source = str(aircraft.overnight_airport)
            rows.append(
                {
                    "candidate_id": f"{aircraft.aircraft_id}->{flight.flight_id}",
                    "aircraft_id": str(aircraft.aircraft_id),
                    "source_airport": source,
                    "flight_id": flight.flight_id,
                    "origin": flight.Origin,
                    "block_id": flight.block_id,
                    "ferry_cost": _ferry_cost(source, flight.Origin),
                    "last_arrival_min": int(aircraft.last_arrival_min),
                }
            )
    return pd.DataFrame(rows)


def _build_scenarios(
    flights: pd.DataFrame,
    month_df: pd.DataFrame,
    carrier: str,
    scenario_count: int,
    seed: int,
    scenario_mode: str = "basic",
    weather_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    scenario_columns = ["scenario_id", "block_id", "capacity", "scenario_type", "shock_airport", "shock_hour"]
    if flights.empty or "block_id" not in flights.columns:
        return pd.DataFrame(columns=scenario_columns)
    rng = np.random.default_rng(seed)
    block_counts = flights.groupby("block_id").size().rename("scheduled_count").reset_index()
    if block_counts.empty:
        return pd.DataFrame(columns=scenario_columns)

    delay_lookup: dict[str, float] = {}
    hist = month_df[
        (month_df["Reporting_Airline"].eq(carrier))
        & (month_df["Origin"].isin(flights["Origin"].unique()))
        & (month_df["CRSDepTime_min"].between(6 * 60, 8 * 60 + 59))
    ].copy()
    if not hist.empty:
        hist["dep_hour"] = (hist["CRSDepTime_min"] // 60).astype(int)
        hist["block_id"] = hist["Origin"] + "_" + hist["dep_hour"].astype(str)
        hist["delay15"] = (hist["DepDelayMinutes"] >= 15).astype(float)
        delay_lookup = hist.groupby("block_id")["delay15"].mean().to_dict()

    weather_lookup: dict[tuple[str, int], dict[str, float]] = {}
    if weather_features is not None and not weather_features.empty:
        weather = weather_features.copy()
        weather["station"] = weather["station"].astype(str).str.upper()
        if "local_hour" not in weather.columns:
            weather["local_hour"] = weather.get("utc_hour", pd.Series(dtype=int))
        for row in weather.itertuples(index=False):
            station = str(getattr(row, "station")).upper()
            local_hour = int(getattr(row, "local_hour"))
            weather_lookup[(station, local_hour)] = {
                "severity": float(getattr(row, "weather_severity", 0.0) or 0.0),
                "adverse": float(getattr(row, "adverse_weather", 0.0) or 0.0),
            }

    rows: list[dict[str, object]] = []
    if scenario_mode == "stress":
        airports = sorted(flights["Origin"].unique().tolist())
        hours = sorted(flights["dep_hour"].unique().tolist())
        if weather_lookup:
            scenario_types = np.array(["normal", "regional", "airport", "hour", "weather"])
            scenario_probs = np.array([0.28, 0.22, 0.20, 0.12, 0.18])
        else:
            scenario_types = np.array(["normal", "regional", "airport", "hour"])
            scenario_probs = np.array([0.35, 0.25, 0.25, 0.15])
        for sid in range(scenario_count):
            scenario_type = str(rng.choice(scenario_types, p=scenario_probs))
            shock_airport = str(rng.choice(airports)) if scenario_type == "airport" else ""
            shock_hour = int(rng.choice(hours)) if scenario_type == "hour" else -1
            if scenario_type == "normal":
                regional_factor = rng.uniform(0.88, 1.04)
            elif scenario_type == "regional":
                regional_factor = rng.uniform(0.58, 0.78)
            elif scenario_type == "weather":
                regional_factor = rng.uniform(0.78, 0.96)
            else:
                regional_factor = rng.uniform(0.80, 0.98)

            for row in block_counts.itertuples(index=False):
                airport, hour_text = str(row.block_id).split("_")
                hour = int(hour_text)
                delay_rate = float(delay_lookup.get(row.block_id, 0.15))
                multiplier = regional_factor - 0.10 * delay_rate + rng.normal(0.0, 0.04)
                weather_info = weather_lookup.get((airport.upper(), hour), {"severity": 0.0, "adverse": 0.0})
                weather_severity = float(weather_info["severity"])
                if weather_severity > 0:
                    multiplier -= 0.12 * weather_severity
                if scenario_type == "airport" and airport == shock_airport:
                    multiplier *= rng.uniform(0.38, 0.62)
                if scenario_type == "hour" and hour == shock_hour:
                    multiplier *= rng.uniform(0.45, 0.68)
                if scenario_type == "weather" and weather_severity > 0:
                    multiplier *= float(np.clip(1.0 - 0.42 * weather_severity, 0.42, 1.0))
                multiplier = float(np.clip(multiplier, 0.0, 1.08))
                capacity = int(np.floor(row.scheduled_count * multiplier))
                capacity = max(0, min(int(row.scheduled_count), capacity))
                rows.append(
                    {
                        "scenario_id": sid,
                        "block_id": row.block_id,
                        "capacity": capacity,
                        "scenario_type": scenario_type,
                        "shock_airport": shock_airport,
                        "shock_hour": shock_hour,
                    }
                )
        return pd.DataFrame(rows)

    factor_values = np.array([1.00, 0.90, 0.75, 0.60])
    factor_probs = np.array([0.42, 0.28, 0.20, 0.10])
    for sid in range(scenario_count):
        system_factor = rng.choice(factor_values, p=factor_probs)
        for row in block_counts.itertuples(index=False):
            delay_rate = float(delay_lookup.get(row.block_id, 0.15))
            local_noise = rng.normal(0.0, 0.05)
            multiplier = float(np.clip(system_factor - 0.12 * delay_rate + local_noise, 0.45, 1.05))
            airport, hour_text = str(row.block_id).split("_")
            weather_info = weather_lookup.get((airport.upper(), int(hour_text)), {"severity": 0.0})
            multiplier = float(np.clip(multiplier - 0.10 * float(weather_info["severity"]), 0.35, 1.05))
            capacity = int(np.floor(row.scheduled_count * multiplier))
            if row.scheduled_count > 0:
                capacity = max(1, min(int(row.scheduled_count), capacity))
            rows.append({"scenario_id": sid, "block_id": row.block_id, "capacity": capacity})
    out = pd.DataFrame(rows)
    out["scenario_type"] = "basic"
    out["shock_airport"] = ""
    out["shock_hour"] = -1
    return out


def build_launch_instance(
    month_df: pd.DataFrame,
    date: str,
    carrier: str = "DL",
    region_airports: Iterable[str] = ("JFK", "LGA", "EWR"),
    scenario_count: int = 12,
    seed: int = 7,
    scenario_mode: str = "basic",
    weather_features: pd.DataFrame | None = None,
) -> LaunchInstance:
    date_ts = pd.Timestamp(date)
    region = list(region_airports)
    flights = _build_flights(month_df, date_ts, carrier, region)
    inventory = _build_inventory(month_df, date_ts, carrier, region)
    candidates = _build_candidates(flights, inventory)
    scenarios = _build_scenarios(
        flights,
        month_df,
        carrier,
        scenario_count,
        seed + int(date_ts.strftime("%d")),
        scenario_mode=scenario_mode,
        weather_features=weather_features,
    )
    return LaunchInstance(
        date=str(date_ts.date()),
        carrier=carrier,
        region_airports=region,
        flights=flights,
        inventory=inventory,
        candidates=candidates,
        scenarios=scenarios,
        total_weight=float(flights["weight"].sum()) if not flights.empty else 0.0,
    )


def build_tail_launch_instance(
    month_df: pd.DataFrame,
    date: str,
    carrier: str = "DL",
    region_airports: Iterable[str] = ("JFK", "LGA", "EWR"),
    scenario_count: int = 12,
    seed: int = 7,
    scenario_mode: str = "stress",
    weather_features: pd.DataFrame | None = None,
) -> LaunchInstance:
    date_ts = pd.Timestamp(date)
    region = list(region_airports)
    flights = _build_flights(month_df, date_ts, carrier, region)
    tail_inventory = _build_tail_inventory(month_df, date_ts, carrier, region)
    inventory = (
        tail_inventory.groupby("overnight_airport")
        .size()
        .rename("aircraft_count")
        .reset_index()
        if not tail_inventory.empty
        else pd.DataFrame(columns=["overnight_airport", "aircraft_count"])
    )
    candidates = _build_tail_candidates(flights, tail_inventory)
    scenarios = _build_scenarios(
        flights,
        month_df,
        carrier,
        scenario_count,
        seed + int(date_ts.strftime("%d")),
        scenario_mode=scenario_mode,
        weather_features=weather_features,
    )
    return LaunchInstance(
        date=str(date_ts.date()),
        carrier=carrier,
        region_airports=region,
        flights=flights,
        inventory=inventory,
        candidates=candidates,
        scenarios=scenarios,
        total_weight=float(flights["weight"].sum()) if not flights.empty else 0.0,
    )


def to_unit_service_instance(instance: LaunchInstance) -> LaunchInstance:
    """Return a copy that removes service-value and cross-airport cost weights."""
    flights = instance.flights.copy()
    candidates = instance.candidates.copy()
    if not flights.empty and "weight" in flights.columns:
        flights["weight"] = 1.0
    if not candidates.empty and "ferry_cost" in candidates.columns:
        candidates["ferry_cost"] = 0.0
    return LaunchInstance(
        date=instance.date,
        carrier=instance.carrier,
        region_airports=instance.region_airports,
        flights=flights,
        inventory=instance.inventory.copy(),
        candidates=candidates,
        scenarios=instance.scenarios.copy(),
        total_weight=float(len(flights)) if not flights.empty else 0.0,
    )


def write_instance_audit(instance: LaunchInstance, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "date": instance.date,
        "carrier": instance.carrier,
        "airports": ",".join(instance.region_airports),
        "flight_count": int(len(instance.flights)),
        "inventory_count": int(instance.inventory["aircraft_count"].sum()) if not instance.inventory.empty else 0,
        "candidate_count": int(len(instance.candidates)),
        "scenario_count": int(instance.scenarios["scenario_id"].nunique()) if not instance.scenarios.empty else 0,
        "total_weight": instance.total_weight,
    }
    path = output_dir / f"instance_audit_{instance.carrier}_{instance.date}.csv"
    pd.DataFrame([summary]).to_csv(path, index=False)
    return path
