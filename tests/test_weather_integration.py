from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from unittest.mock import patch

import pandas as pd

from src.trc_launch.smoke_data import _build_scenarios
from src.trc_launch.weather_asos import (
    build_iem_asos_url,
    fetch_airport_date_weather,
    fetch_iem_asos_range,
    hourly_weather_features,
    local_weather_window_utc,
)


SAMPLE_ASOS = """station,valid,tmpf,dwpf,relh,sknt,vsby,p01i,wxcodes,metar
JFK,2025-01-20 10:00,22.00,9.00,56.78,18.00,10.00,0.00,,KJFK 201000Z 32018KT 10SM CLR M06/M13 A3004
JFK,2025-01-20 11:00,21.00,8.00,56.62,24.00,2.00,0.08,-SN BR,KJFK 201100Z 32024KT 2SM -SN BR M06/M13 A3011
LGA,2025-01-20 11:00,20.00,8.00,58.00,12.00,8.00,0.00,,KLGA 201100Z 31012KT 8SM CLR M06/M13 A3010
"""


class FakeResponse:
    def __init__(self, text: str):
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return self.text.encode("utf-8")


class WeatherAsosTests(unittest.TestCase):
    def test_build_iem_url_contains_station_fields_and_utc_window(self) -> None:
        url = build_iem_asos_url(
            station="JFK",
            start=pd.Timestamp("2025-01-20 10:00", tz="UTC"),
            end=pd.Timestamp("2025-01-20 15:00", tz="UTC"),
        )

        self.assertIn("station=JFK", url)
        self.assertIn("data=sknt", url)
        self.assertIn("year1=2025", url)
        self.assertIn("hour1=10", url)
        self.assertIn("hour2=15", url)

    def test_hourly_weather_features_identify_adverse_station_hour(self) -> None:
        raw = pd.read_csv(pd.io.common.StringIO(SAMPLE_ASOS))
        features = hourly_weather_features(raw, station_time_zones={"JFK": "America/New_York", "LGA": "America/New_York"})

        jfk_11 = features[(features["station"].eq("JFK")) & (features["utc_hour"].eq(11))].iloc[0]
        lga_11 = features[(features["station"].eq("LGA")) & (features["utc_hour"].eq(11))].iloc[0]

        self.assertGreater(jfk_11["weather_severity"], lga_11["weather_severity"])
        self.assertEqual(int(jfk_11["adverse_weather"]), 1)
        self.assertEqual(int(lga_11["adverse_weather"]), 0)

    def test_local_weather_window_respects_daylight_saving_time(self) -> None:
        start, end = local_weather_window_utc(["JFK"], "2025-06-20", 6, 9)

        self.assertEqual(start.hour, 10)
        self.assertEqual(end.hour, 13)

    def test_fetch_iem_asos_range_uses_cache(self) -> None:
        with TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            cached = cache_dir / "JFK_20250120_1000_1500.csv"
            cached.write_text(SAMPLE_ASOS, encoding="utf-8")

            with patch("src.trc_launch.weather_asos.urlopen") as fake_download:
                out = fetch_iem_asos_range(
                    stations=["JFK"],
                    start=pd.Timestamp("2025-01-20 10:00", tz="UTC"),
                    end=pd.Timestamp("2025-01-20 15:00", tz="UTC"),
                    output_dir=cache_dir,
                )

            fake_download.assert_not_called()
            self.assertEqual(set(out["station"]), {"JFK"})
            self.assertEqual(len(out), 2)

    def test_fetch_iem_asos_range_retries_after_rate_limit(self) -> None:
        calls = {"count": 0}

        def fake_download(url, timeout):
            calls["count"] += 1
            if calls["count"] == 1:
                raise HTTPError(url, 429, "Too Many Requests", hdrs=None, fp=None)
            return FakeResponse(SAMPLE_ASOS)

        with TemporaryDirectory() as tmp:
            with patch("src.trc_launch.weather_asos.urlopen", side_effect=fake_download):
                with patch("src.trc_launch.weather_asos.time.sleep"):
                    out = fetch_iem_asos_range(
                        stations=["JFK"],
                        start=pd.Timestamp("2025-01-20 10:00", tz="UTC"),
                        end=pd.Timestamp("2025-01-20 15:00", tz="UTC"),
                        output_dir=Path(tmp),
                    )

        self.assertEqual(calls["count"], 2)
        self.assertEqual(set(out["station"]), {"JFK"})

    def test_fetch_airport_date_weather_uses_automatic_timezone_window(self) -> None:
        captured: dict[str, object] = {}

        def fake_fetch(stations, start, end, output_dir, force=False, **kwargs):
            captured["stations"] = stations
            captured["start"] = start
            captured["end"] = end
            return pd.read_csv(pd.io.common.StringIO(SAMPLE_ASOS))

        with TemporaryDirectory() as tmp:
            with patch("src.trc_launch.weather_asos.fetch_iem_asos_range", side_effect=fake_fetch):
                features = fetch_airport_date_weather(
                    ["JFK"],
                    "2025-06-20",
                    output_dir=Path(tmp),
                )

        self.assertEqual(captured["start"].hour, 8)
        self.assertEqual(captured["end"].hour, 16)
        self.assertFalse(features.empty)

    def test_weather_adjusted_scenarios_reduce_capacity_for_adverse_blocks(self) -> None:
        flights = pd.DataFrame(
            [
                {"flight_id": "f1", "Origin": "JFK", "block_id": "JFK_6", "dep_hour": 6},
                {"flight_id": "f2", "Origin": "JFK", "block_id": "JFK_6", "dep_hour": 6},
                {"flight_id": "f3", "Origin": "LGA", "block_id": "LGA_6", "dep_hour": 6},
                {"flight_id": "f4", "Origin": "LGA", "block_id": "LGA_6", "dep_hour": 6},
            ]
        )
        month_df = pd.DataFrame(
            {
                "Reporting_Airline": ["DL", "DL"],
                "Origin": ["JFK", "LGA"],
                "CRSDepTime_min": [360, 360],
                "DepDelayMinutes": [0, 0],
            }
        )
        weather_features = pd.DataFrame(
            [
                {"station": "JFK", "local_hour": 6, "weather_severity": 0.80, "adverse_weather": 1},
                {"station": "LGA", "local_hour": 6, "weather_severity": 0.00, "adverse_weather": 0},
            ]
        )

        scenarios = _build_scenarios(
            flights,
            month_df,
            carrier="DL",
            scenario_count=20,
            seed=11,
            scenario_mode="stress",
            weather_features=weather_features,
        )
        means = scenarios.groupby("block_id")["capacity"].mean()

        self.assertLess(means["JFK_6"], means["LGA_6"])
        self.assertIn("weather", set(scenarios["scenario_type"]))


if __name__ == "__main__":
    unittest.main()
