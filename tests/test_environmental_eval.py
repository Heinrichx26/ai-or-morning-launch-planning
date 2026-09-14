from __future__ import annotations

import unittest

import pandas as pd

from src.trc_launch.environmental_eval import (
    CO2_KG_PER_KG_FUEL,
    TAXI_FUEL_KG_PER_MIN,
    TAXI_OUT_MINUTES,
    convert_metrics_row,
    co2_from_fuel,
    fuel_from_minutes,
)


class EnvironmentalEvalTests(unittest.TestCase):
    def test_fuel_and_co2_factors(self) -> None:
        fuel = fuel_from_minutes(19.0)
        self.assertAlmostEqual(fuel, 19.0 * TAXI_FUEL_KG_PER_MIN)
        self.assertAlmostEqual(co2_from_fuel(fuel), fuel * CO2_KG_PER_KG_FUEL)

    def test_convert_metrics_row_recovers_weight_and_avoidable_flights(self) -> None:
        row = pd.Series(
            {
                "expected_cost": 10.4,
                "ferry_cost": 0.4,
                "mean_launch_weight_share": 0.5,
                "assigned_weight_share": 0.8,
                "flight_count": 10,
                "assigned_flights": 8,
                "ferry_moves": 1,
                "mean_regret": 2.0,
                "max_regret": 4.0,
                "min_launch_weight_share": 0.2,
            }
        )
        out = convert_metrics_row(row)
        self.assertAlmostEqual(out["total_weight"], 20.0)
        self.assertAlmostEqual(out["mean_flight_weight"], 2.0)
        self.assertAlmostEqual(out["mean_unlaunched_flights"], 5.0)
        self.assertAlmostEqual(out["mean_hold_flights"], 3.0)
        self.assertAlmostEqual(out["avoidable_unlaunched_flights_mean"], 1.0)
        self.assertAlmostEqual(out["mean_hold_minutes"], 3.0 * TAXI_OUT_MINUTES)
        self.assertGreater(out["avoidable_co2_kg_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
