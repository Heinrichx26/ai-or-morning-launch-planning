from __future__ import annotations

import time
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
ASOS_FIELDS = ["tmpf", "dwpf", "relh", "sknt", "vsby", "p01i", "wxcodes", "metar"]
AIRPORT_TIME_ZONES = {
    "JFK": "America/New_York",
    "LGA": "America/New_York",
    "EWR": "America/New_York",
    "DCA": "America/New_York",
    "IAD": "America/New_York",
    "BWI": "America/New_York",
    "MIA": "America/New_York",
    "FLL": "America/New_York",
    "PBI": "America/New_York",
    "ORD": "America/Chicago",
    "MDW": "America/Chicago",
    "DFW": "America/Chicago",
    "DAL": "America/Chicago",
    "IAH": "America/Chicago",
    "HOU": "America/Chicago",
    "LAX": "America/Los_Angeles",
    "BUR": "America/Los_Angeles",
    "LGB": "America/Los_Angeles",
    "ONT": "America/Los_Angeles",
    "SNA": "America/Los_Angeles",
    "SFO": "America/Los_Angeles",
    "OAK": "America/Los_Angeles",
    "SJC": "America/Los_Angeles",
}


def _utc_timestamp(value: pd.Timestamp | str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _time_zone_for_airports(airports: list[str]) -> str:
    zones = {AIRPORT_TIME_ZONES.get(str(airport).upper(), "America/New_York") for airport in airports}
    if len(zones) == 1:
        return zones.pop()
    return AIRPORT_TIME_ZONES.get(str(airports[0]).upper(), "America/New_York")


def local_weather_window_utc(
    airports: list[str],
    date: str,
    local_start_hour: int = 6,
    local_end_hour: int = 9,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    local_date = pd.Timestamp(date)
    zone = ZoneInfo(_time_zone_for_airports(airports))
    local_start = pd.Timestamp(
        year=local_date.year,
        month=local_date.month,
        day=local_date.day,
        hour=local_start_hour,
        tz=zone,
    )
    local_end = pd.Timestamp(
        year=local_date.year,
        month=local_date.month,
        day=local_date.day,
        hour=local_end_hour,
        tz=zone,
    )
    return local_start.tz_convert("UTC"), local_end.tz_convert("UTC")


def build_iem_asos_url(station: str, start: pd.Timestamp | str, end: pd.Timestamp | str) -> str:
    start_ts = _utc_timestamp(start)
    end_ts = _utc_timestamp(end)
    params: list[tuple[str, object]] = [("station", station.upper())]
    params.extend(("data", field) for field in ASOS_FIELDS)
    params.extend(
        [
            ("tz", "Etc/UTC"),
            ("format", "onlycomma"),
            ("latlon", "no"),
            ("elev", "no"),
            ("missing", "empty"),
            ("trace", "0.0001"),
            ("direct", "no"),
            ("report_type", "1"),
            ("report_type", "3"),
            ("report_type", "4"),
            ("year1", start_ts.year),
            ("month1", start_ts.month),
            ("day1", start_ts.day),
            ("hour1", start_ts.hour),
            ("minute1", start_ts.minute),
            ("year2", end_ts.year),
            ("month2", end_ts.month),
            ("day2", end_ts.day),
            ("hour2", end_ts.hour),
            ("minute2", end_ts.minute),
        ]
    )
    return f"{IEM_ASOS_URL}?{urlencode(params)}"


def _cache_name(station: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"{station.upper()}_{start:%Y%m%d_%H%M}_{end:%H%M}.csv"


def fetch_iem_asos_range(
    stations: list[str],
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    output_dir: Path,
    force: bool = False,
    pause_seconds: float = 1.1,
    max_attempts: int = 3,
    request_timeout_seconds: float = 45.0,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    start_ts = _utc_timestamp(start)
    end_ts = _utc_timestamp(end)
    frames: list[pd.DataFrame] = []
    for idx, station in enumerate(sorted({str(s).upper() for s in stations})):
        cache_path = output_dir / _cache_name(station, start_ts, end_ts)
        if force or not cache_path.exists():
            time.sleep(pause_seconds)
            url = build_iem_asos_url(station, start_ts, end_ts)
            for attempt in range(1, max_attempts + 1):
                try:
                    with urlopen(url, timeout=request_timeout_seconds) as response:
                        cache_path.write_bytes(response.read())
                    break
                except (HTTPError, URLError, TimeoutError, OSError, HTTPException):
                    if attempt >= max_attempts:
                        raise
                    time.sleep(pause_seconds * attempt)
        frame = pd.read_csv(cache_path)
        if "station" in frame.columns:
            frame = frame[frame["station"].astype(str).str.upper().eq(station)].copy()
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["station", "valid", *ASOS_FIELDS])
    return pd.concat(frames, ignore_index=True)


def hourly_weather_features(
    raw: pd.DataFrame,
    utc_to_local_hours: int = -5,
    station_time_zones: dict[str, str] | None = None,
) -> pd.DataFrame:
    columns = [
        "station",
        "utc_hour",
        "local_hour",
        "mean_wind_kt",
        "max_wind_kt",
        "min_visibility_mi",
        "precip_in",
        "weather_observations",
        "weather_severity",
        "adverse_weather",
    ]
    if raw.empty:
        return pd.DataFrame(columns=columns)

    df = raw.copy()
    df["station"] = df["station"].astype(str).str.upper()
    df["valid"] = pd.to_datetime(df["valid"], errors="coerce", utc=True)
    df = df.dropna(subset=["valid"])
    if df.empty:
        return pd.DataFrame(columns=columns)

    for col in ["sknt", "vsby", "p01i"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df["sknt"] = df["sknt"].clip(lower=0)
    df["vsby"] = df["vsby"].clip(lower=0)
    df["p01i"] = df["p01i"].fillna(0.0).clip(lower=0)
    wx = df.get("wxcodes", pd.Series("", index=df.index)).fillna("").astype(str)
    metar = df.get("metar", pd.Series("", index=df.index)).fillna("").astype(str)
    wx_text = (wx + " " + metar).str.upper()
    df["wx_signal"] = wx_text.str.contains(
        r"\b(?:TS|SN|FZ|PL|SG|GR|GS|RA|DZ|BR|FG|HZ|SQ|VA|DU|SA)\b",
        regex=True,
    ).astype(float)
    df["utc_hour"] = df["valid"].dt.hour.astype(int)
    if station_time_zones:
        zones = {str(key).upper(): ZoneInfo(value) for key, value in station_time_zones.items()}

        def _local_hour(row: pd.Series) -> int:
            station = str(row["station"]).upper()
            zone = zones.get(station, ZoneInfo("America/New_York"))
            return int(row["valid"].tz_convert(zone).hour)

        df["local_hour"] = df.apply(_local_hour, axis=1)
    else:
        df["local_hour"] = ((df["utc_hour"] + int(utc_to_local_hours)) % 24).astype(int)

    grouped = (
        df.groupby(["station", "utc_hour", "local_hour"])
        .agg(
            mean_wind_kt=("sknt", "mean"),
            max_wind_kt=("sknt", "max"),
            min_visibility_mi=("vsby", "min"),
            precip_in=("p01i", "sum"),
            weather_observations=("station", "size"),
            wx_signal=("wx_signal", "max"),
        )
        .reset_index()
    )
    grouped["mean_wind_kt"] = grouped["mean_wind_kt"].fillna(0.0)
    grouped["max_wind_kt"] = grouped["max_wind_kt"].fillna(grouped["mean_wind_kt"]).fillna(0.0)
    grouped["min_visibility_mi"] = grouped["min_visibility_mi"].fillna(10.0).clip(lower=0.0, upper=10.0)
    grouped["precip_in"] = grouped["precip_in"].fillna(0.0)

    visibility_component = ((6.0 - grouped["min_visibility_mi"]) / 6.0).clip(lower=0.0, upper=1.0)
    wind_component = ((grouped["max_wind_kt"] - 18.0) / 22.0).clip(lower=0.0, upper=1.0)
    precip_component = (grouped["precip_in"] / 0.15).clip(lower=0.0, upper=1.0)
    wx_component = grouped["wx_signal"].clip(lower=0.0, upper=1.0)
    grouped["weather_severity"] = np.maximum.reduce(
        [
            visibility_component.to_numpy(),
            wind_component.to_numpy(),
            precip_component.to_numpy(),
            wx_component.to_numpy() * 0.65,
        ]
    )
    grouped["adverse_weather"] = (
        (grouped["weather_severity"] >= 0.35)
        | (grouped["min_visibility_mi"] < 5.0)
        | (grouped["max_wind_kt"] >= 25.0)
        | (grouped["precip_in"] >= 0.05)
    ).astype(int)
    return grouped[columns]


def fetch_airport_date_weather(
    airports: list[str],
    date: str,
    output_dir: Path = Path("data/weather_asos"),
    local_start_hour: int = 4,
    local_end_hour: int = 12,
    utc_offset_hours: int | None = None,
    force: bool = False,
    pause_seconds: float = 2.0,
    max_attempts: int = 5,
    request_timeout_seconds: float = 45.0,
) -> pd.DataFrame:
    local_date = pd.Timestamp(date)
    if utc_offset_hours is None:
        utc_start, utc_end = local_weather_window_utc(airports, date, local_start_hour, local_end_hour)
    else:
        utc_start = pd.Timestamp(
            year=local_date.year,
            month=local_date.month,
            day=local_date.day,
            hour=local_start_hour - utc_offset_hours,
            tz="UTC",
        )
        utc_end = pd.Timestamp(
            year=local_date.year,
            month=local_date.month,
            day=local_date.day,
            hour=local_end_hour - utc_offset_hours,
            tz="UTC",
        )
    date_dir = output_dir / f"{local_date:%Y%m%d}"
    date_dir.mkdir(parents=True, exist_ok=True)
    raw = fetch_iem_asos_range(
        stations=airports,
        start=utc_start,
        end=utc_end,
        output_dir=date_dir,
        force=force,
        pause_seconds=pause_seconds,
        max_attempts=max_attempts,
        request_timeout_seconds=request_timeout_seconds,
    )
    station_time_zones = {
        str(airport).upper(): AIRPORT_TIME_ZONES.get(str(airport).upper(), "America/New_York")
        for airport in airports
    }
    features = hourly_weather_features(
        raw,
        utc_to_local_hours=utc_offset_hours if utc_offset_hours is not None else -5,
        station_time_zones=station_time_zones,
    )
    if not features.empty:
        features.to_csv(date_dir / "hourly_weather_features.csv", index=False)
    return features
