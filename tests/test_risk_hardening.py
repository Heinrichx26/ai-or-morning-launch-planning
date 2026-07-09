import unittest

import pandas as pd

from src.trc_launch.risk_hardening import (
    choose_stress_instances,
    frame_to_text_table,
    scale_instance_weights,
    scale_weather_features,
)
from src.trc_launch.run_smoke_batch7_filtered_regret import _filtered_instance
from src.trc_launch.smoke_data import LaunchInstance
from src.trc_launch.smoke_data import to_unit_service_instance


class RiskHardeningTests(unittest.TestCase):
    def test_scale_weather_features_updates_severity_and_adverse_flag(self) -> None:
        features = pd.DataFrame(
            [
                {"station": "JFK", "local_hour": 6, "weather_severity": 0.30, "adverse_weather": 0},
                {"station": "LGA", "local_hour": 7, "weather_severity": 0.90, "adverse_weather": 1},
            ]
        )

        scaled = scale_weather_features(features, 1.5)

        self.assertAlmostEqual(float(scaled.loc[0, "weather_severity"]), 0.45)
        self.assertAlmostEqual(float(scaled.loc[1, "weather_severity"]), 1.0)
        self.assertEqual(int(scaled.loc[0, "adverse_weather"]), 1)
        self.assertEqual(int(scaled.loc[1, "adverse_weather"]), 1)

    def test_scale_instance_weights_updates_flight_weights_and_total_weight(self) -> None:
        flights = pd.DataFrame(
            [
                {"flight_id": "f1", "weight": 1.0},
                {"flight_id": "f2", "weight": 2.0},
            ]
        )
        instance = LaunchInstance(
            date="2025-01-01",
            carrier="DL",
            region_airports=["JFK", "LGA"],
            flights=flights,
            inventory=pd.DataFrame({"overnight_airport": ["JFK"], "aircraft_count": [1]}),
            candidates=pd.DataFrame(),
            scenarios=pd.DataFrame(),
            total_weight=3.0,
        )

        scaled = scale_instance_weights(instance, 1.2)

        self.assertEqual(instance.total_weight, 3.0)
        self.assertAlmostEqual(float(scaled.total_weight), 3.6)
        self.assertAlmostEqual(float(scaled.flights["weight"].sum()), 3.6)

    def test_choose_stress_instances_prioritizes_pressure_and_regret_gap(self) -> None:
        selected = pd.DataFrame(
            [
                {"region": "nyc", "carrier": "DL", "date": "2025-01-01", "candidate_count_preview": 1000, "flights": 20, "total_deficit": 1, "max_airport_deficit": 1},
                {"region": "nyc", "carrier": "DL", "date": "2025-01-02", "candidate_count_preview": 4000, "flights": 50, "total_deficit": 8, "max_airport_deficit": 5},
            ]
        )
        metrics = pd.DataFrame(
            [
                {"instance_key": "nyc_DL_2025-01-01", "method": "rccc_audited", "max_regret": 0.4, "candidate_retention": 0.7},
                {"instance_key": "nyc_DL_2025-01-01", "method": "saa_extensive", "max_regret": 0.5, "candidate_retention": 1.0},
                {"instance_key": "nyc_DL_2025-01-02", "method": "rccc_audited", "max_regret": 1.0, "candidate_retention": 0.6},
                {"instance_key": "nyc_DL_2025-01-02", "method": "saa_extensive", "max_regret": 3.0, "candidate_retention": 1.0},
            ]
        )
        weather = pd.DataFrame(
            [
                {"instance_key": "nyc_DL_2025-01-01", "adverse_hours": 0, "max_weather_severity": 0.0},
                {"instance_key": "nyc_DL_2025-01-02", "adverse_hours": 4, "max_weather_severity": 0.8},
            ]
        )

        stress = choose_stress_instances(selected, metrics, weather, max_instances=1)

        self.assertEqual(stress.loc[0, "instance_key"], "nyc_DL_2025-01-02")
        self.assertGreater(float(stress.loc[0, "stress_score"]), 0.0)

    def test_frame_to_text_table_does_not_require_optional_markdown_dependency(self) -> None:
        frame = pd.DataFrame([{"method": "rccc_audited", "cases": 3, "mean_max_regret": 0.5}])

        text = frame_to_text_table(frame)

        self.assertIn("rccc_audited", text)
        self.assertIn("mean_max_regret", text)

    def test_to_unit_service_instance_removes_value_and_cost_weights(self) -> None:
        instance = LaunchInstance(
            date="2025-01-01",
            carrier="DL",
            region_airports=["JFK", "LGA"],
            flights=pd.DataFrame(
                [
                    {"flight_id": "f1", "weight": 1.7},
                    {"flight_id": "f2", "weight": 2.2},
                ]
            ),
            inventory=pd.DataFrame({"overnight_airport": ["JFK"], "aircraft_count": [1]}),
            candidates=pd.DataFrame(
                [
                    {"candidate_id": "c1", "source_airport": "JFK", "origin": "JFK", "flight_id": "f1", "ferry_cost": 0.35},
                    {"candidate_id": "c2", "source_airport": "JFK", "origin": "LGA", "flight_id": "f2", "ferry_cost": 0.28},
                ]
            ),
            scenarios=pd.DataFrame({"scenario_id": [0], "block_id": ["JFK_6"], "capacity": [1]}),
            total_weight=3.9,
        )

        unit = to_unit_service_instance(instance)

        self.assertAlmostEqual(float(unit.flights["weight"].sum()), 2.0)
        self.assertAlmostEqual(float(unit.total_weight), 2.0)
        self.assertTrue(unit.candidates["ferry_cost"].eq(0.0).all())
        self.assertAlmostEqual(float(instance.total_weight), 3.9)

    def test_filtered_instance_keeps_same_airport_edges_by_airport_not_cost(self) -> None:
        flights = pd.DataFrame(
            [
                {"flight_id": "f1", "Origin": "JFK", "block_id": "JFK_6", "weight": 1.0},
                {"flight_id": "f2", "Origin": "LGA", "block_id": "LGA_6", "weight": 1.0},
            ]
        )
        candidates = pd.DataFrame(
            [
                {"candidate_id": "same", "aircraft_id": "a1", "source_airport": "JFK", "origin": "JFK", "flight_id": "f1", "ferry_cost": 0.0},
                {"candidate_id": "cross", "aircraft_id": "a1", "source_airport": "JFK", "origin": "LGA", "flight_id": "f2", "ferry_cost": 0.0},
            ]
        )
        instance = LaunchInstance(
            date="2025-01-01",
            carrier="DL",
            region_airports=["JFK", "LGA"],
            flights=flights,
            inventory=pd.DataFrame({"overnight_airport": ["JFK"], "aircraft_count": [1]}),
            candidates=candidates,
            scenarios=pd.DataFrame({"scenario_id": [0, 0], "block_id": ["JFK_6", "LGA_6"], "capacity": [1, 1]}),
            total_weight=2.0,
        )

        filtered, _ = _filtered_instance(instance, per_flight=0, per_aircraft=0)

        self.assertEqual(filtered.candidates["candidate_id"].tolist(), ["same"])


if __name__ == "__main__":
    unittest.main()
