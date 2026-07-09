from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .run_smoke_batch11_multiregion import _key
from .weather_asos import fetch_airport_date_weather


def run(args: argparse.Namespace) -> None:
    selected = pd.read_csv(args.selected_instances)
    if args.max_instances > 0:
        selected = selected.head(args.max_instances).reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for item in selected.itertuples(index=False):
        airports = str(item.region_airports).split("|")
        key = _key(str(item.region), str(item.carrier), str(item.date))
        error = ""
        try:
            features = fetch_airport_date_weather(
                airports=airports,
                date=str(item.date),
                output_dir=Path(args.weather_dir),
                local_start_hour=args.local_start_hour,
                local_end_hour=args.local_end_hour,
                force=args.refresh_weather,
                pause_seconds=args.pause_seconds,
                max_attempts=args.max_attempts,
                request_timeout_seconds=args.request_timeout_seconds,
            )
        except Exception as exc:
            features = pd.DataFrame()
            error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
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
        )
        print(f"{key}: adverse_hours={rows[-1]['adverse_hours']} max_severity={rows[-1]['max_weather_severity']:.3f}", flush=True)
    out = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(out.sort_values(["adverse_hours", "max_weather_severity"], ascending=False).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit IEM ASOS/METAR weather severity for selected MLAP instances.")
    parser.add_argument("--selected-instances", default="data/trc_open_smoke/batch17_open_benchmark/selected_instances.csv")
    parser.add_argument("--weather-dir", default="data/weather_asos")
    parser.add_argument("--output", default="results/trc_smoke/batch18_weather_audit/weather_feature_audit.csv")
    parser.add_argument("--max-instances", type=int, default=20)
    parser.add_argument("--local-start-hour", type=int, default=4)
    parser.add_argument("--local-end-hour", type=int, default=12)
    parser.add_argument("--pause-seconds", type=float, default=2.0)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--request-timeout-seconds", type=float, default=45.0)
    parser.add_argument("--refresh-weather", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
