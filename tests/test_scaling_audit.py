import unittest

import pandas as pd

from src.trc_launch.scenario_scaling import scenario_grid, summarize_scaling


class ScenarioScalingTests(unittest.TestCase):
    def test_scenario_grid_uses_smoke_and_full_modes(self) -> None:
        self.assertEqual(scenario_grid(smoke=True), [30, 60])
        self.assertEqual(scenario_grid(smoke=False), [30, 60, 90, 120])

    def test_scenario_grid_accepts_custom_counts(self) -> None:
        self.assertEqual(scenario_grid(smoke=False, custom_counts=[240, 30, 180]), [30, 180, 240])

    def test_summarize_scaling_reports_match_retention_and_speed_ratio(self) -> None:
        metrics = pd.DataFrame(
            [
                {
                    "scenario_count": 30,
                    "method": "regret_portfolio_full",
                    "instance_key": "a",
                    "solve_seconds": 10.0,
                    "candidate_retention": 1.0,
                    "matches_full_regret": True,
                },
                {
                    "scenario_count": 30,
                    "method": "rccc_audited",
                    "instance_key": "a",
                    "solve_seconds": 4.0,
                    "candidate_retention": 0.5,
                    "matches_full_regret": True,
                },
                {
                    "scenario_count": 60,
                    "method": "regret_portfolio_full",
                    "instance_key": "a",
                    "solve_seconds": 20.0,
                    "candidate_retention": 1.0,
                    "matches_full_regret": True,
                },
                {
                    "scenario_count": 60,
                    "method": "rccc_audited",
                    "instance_key": "a",
                    "solve_seconds": 5.0,
                    "candidate_retention": 0.4,
                    "matches_full_regret": True,
                },
            ]
        )

        summary = summarize_scaling(metrics)

        row30 = summary[summary["scenario_count"].eq(30)].iloc[0]
        row60 = summary[summary["scenario_count"].eq(60)].iloc[0]
        self.assertEqual(int(row30["cases"]), 1)
        self.assertEqual(int(row60["rccc_regret_matches"]), 1)
        self.assertAlmostEqual(float(row30["full_to_rccc_time_ratio"]), 2.5)
        self.assertAlmostEqual(float(row60["full_to_rccc_time_ratio"]), 4.0)
        self.assertAlmostEqual(float(row60["mean_rccc_retention"]), 0.4)


if __name__ == "__main__":
    unittest.main()
