import unittest

import pandas as pd

from src.trc_launch.regret_separation import (
    active_scenario_instance,
    choose_next_scenarios,
    regret_by_scenario,
    summarize_separation,
    violating_scenarios,
)
from src.trc_launch.optimizers import SolveResult
from src.trc_launch.smoke_data import LaunchInstance


def tiny_instance() -> LaunchInstance:
    flights = pd.DataFrame(
        [
            {
                "flight_id": "f1",
                "Origin": "AAA",
                "Dest": "BBB",
                "block_id": "AAA_6",
                "CRSDepTime": 600,
                "weight": 2.0,
            },
            {
                "flight_id": "f2",
                "Origin": "AAA",
                "Dest": "CCC",
                "block_id": "AAA_6",
                "CRSDepTime": 630,
                "weight": 1.0,
            },
        ]
    )
    inventory = pd.DataFrame(
        [
            {"aircraft_id": "n1", "overnight_airport": "AAA", "aircraft_count": 1},
            {"aircraft_id": "n2", "overnight_airport": "AAA", "aircraft_count": 1},
        ]
    )
    candidates = pd.DataFrame(
        [
            {"candidate_id": "n1->f1", "aircraft_id": "n1", "source_airport": "AAA", "flight_id": "f1", "ferry_cost": 0.0},
            {"candidate_id": "n2->f2", "aircraft_id": "n2", "source_airport": "AAA", "flight_id": "f2", "ferry_cost": 0.0},
        ]
    )
    scenarios = pd.DataFrame(
        [
            {"scenario_id": 5, "block_id": "AAA_6", "capacity": 2},
            {"scenario_id": 9, "block_id": "AAA_6", "capacity": 1},
        ]
    )
    return LaunchInstance(
        date="2025-01-01",
        carrier="XX",
        region_airports=["AAA"],
        flights=flights,
        inventory=inventory,
        candidates=candidates,
        scenarios=scenarios,
        total_weight=3.0,
    )


class RegretSeparationTests(unittest.TestCase):
    def test_active_scenario_instance_remaps_ids_and_preserves_original_ids(self) -> None:
        active, mapping = active_scenario_instance(tiny_instance(), [9, 5])

        self.assertEqual(mapping, {0: 5, 1: 9})
        self.assertEqual(active.scenarios["scenario_id"].tolist(), [0, 1])
        self.assertEqual(active.scenarios["original_scenario_id"].tolist(), [5, 9])

    def test_choose_next_scenarios_returns_highest_inactive_regret(self) -> None:
        regrets = {0: 0.2, 1: 2.0, 2: 1.4}

        self.assertEqual(choose_next_scenarios(regrets, active_ids={0}, max_add=1), [1])
        self.assertEqual(choose_next_scenarios(regrets, active_ids={0, 1}, max_add=2), [2])

    def test_violating_scenarios_returns_all_inactive_regret_cuts(self) -> None:
        regrets = {0: 1.0, 1: 1.8, 2: 1.4, 3: 0.7}

        self.assertEqual(violating_scenarios(regrets, active_ids={0}, tolerance=1e-8), [1, 2])

    def test_regret_by_scenario_uses_full_scenario_ids(self) -> None:
        instance = tiny_instance()
        selected = instance.candidates.merge(instance.flights[["flight_id", "block_id", "weight"]], on="flight_id", how="left")
        result = SolveResult("unit", "ok", 0.0, 0.0, selected, {})

        regrets = regret_by_scenario(instance, result, oracle_costs={5: 0.0, 9: 0.0})

        self.assertAlmostEqual(regrets[5], 0.0)
        self.assertAlmostEqual(regrets[9], 1.0)

    def test_summarize_separation_reports_active_share_match_and_speed_ratio(self) -> None:
        metrics = pd.DataFrame(
            [
                {
                    "method": "regret_portfolio_full",
                    "instance_key": "case-a",
                    "solve_seconds": 10.0,
                    "max_regret": 2.0,
                    "expected_cost": 7.0,
                    "active_scenario_share": 1.0,
                    "matches_full_regret": True,
                    "matches_full_cost": True,
                },
                {
                    "method": "regret_active_scenario_separation",
                    "instance_key": "case-a",
                    "solve_seconds": 4.0,
                    "max_regret": 2.0,
                    "expected_cost": 7.0,
                    "active_scenario_share": 0.25,
                    "matches_full_regret": True,
                    "matches_full_cost": True,
                },
            ]
        )

        summary = summarize_separation(metrics)

        row = summary.iloc[0]
        self.assertEqual(int(row["cases"]), 1)
        self.assertEqual(int(row["regret_matches"]), 1)
        self.assertEqual(int(row["cost_matches"]), 1)
        self.assertAlmostEqual(float(row["mean_active_scenario_share"]), 0.25)
        self.assertAlmostEqual(float(row["full_to_separation_time_ratio"]), 2.5)


if __name__ == "__main__":
    unittest.main()
