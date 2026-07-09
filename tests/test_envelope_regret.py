import unittest

import numpy as np
import pandas as pd

from src.trc_launch.optimizers import (
    LaunchOptimizer,
    compute_oracle_costs,
    evaluate_solution,
    launch_value_envelope_terms,
)
from src.trc_launch.envelope_regret import summarize_envelope
from src.trc_launch.integral_recourse import summarize_integral_recourse
from src.trc_launch.polymatroid_regret import (
    cut_pool_size,
    initial_threshold_pool,
    separate_polymatroid_cuts,
)
from src.trc_launch.smoke_data import LaunchInstance


def envelope_test_instance() -> LaunchInstance:
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
            {
                "flight_id": "f3",
                "Origin": "BBB",
                "Dest": "DDD",
                "block_id": "BBB_6",
                "CRSDepTime": 645,
                "weight": 1.5,
            },
        ]
    )
    inventory = pd.DataFrame(
        [
            {"aircraft_id": "n1", "overnight_airport": "AAA", "aircraft_count": 1},
            {"aircraft_id": "n2", "overnight_airport": "AAA", "aircraft_count": 1},
            {"aircraft_id": "n3", "overnight_airport": "BBB", "aircraft_count": 1},
        ]
    )
    candidates = pd.DataFrame(
        [
            {"candidate_id": "n1->f1", "aircraft_id": "n1", "source_airport": "AAA", "flight_id": "f1", "ferry_cost": 0.0},
            {"candidate_id": "n2->f2", "aircraft_id": "n2", "source_airport": "AAA", "flight_id": "f2", "ferry_cost": 0.0},
            {"candidate_id": "n3->f3", "aircraft_id": "n3", "source_airport": "BBB", "flight_id": "f3", "ferry_cost": 0.0},
            {"candidate_id": "n1->f3", "aircraft_id": "n1", "source_airport": "AAA", "flight_id": "f3", "ferry_cost": 0.3},
        ]
    )
    scenarios = pd.DataFrame(
        [
            {"scenario_id": 0, "block_id": "AAA_6", "capacity": 2},
            {"scenario_id": 0, "block_id": "BBB_6", "capacity": 1},
            {"scenario_id": 1, "block_id": "AAA_6", "capacity": 1},
            {"scenario_id": 1, "block_id": "BBB_6", "capacity": 0},
        ]
    )
    return LaunchInstance(
        date="2025-01-01",
        carrier="XX",
        region_airports=["AAA", "BBB"],
        flights=flights,
        inventory=inventory,
        candidates=candidates,
        scenarios=scenarios,
        total_weight=4.5,
    )


class EnvelopeRegretTests(unittest.TestCase):
    def test_launch_value_envelope_matches_top_k_launch_value(self) -> None:
        weights = [2.0, 1.0]
        assigned = np.array([1.0, 1.0])
        terms = launch_value_envelope_terms(weights, capacity=1)

        envelope = min(rhs + float(np.dot(coefficients, assigned)) for _, rhs, coefficients in terms)

        self.assertAlmostEqual(envelope, 2.0)

    def test_envelope_regret_portfolio_matches_extensive_regret_on_tiny_instance(self) -> None:
        instance = envelope_test_instance()
        oracle = compute_oracle_costs(instance, time_limit=30.0)
        extensive = LaunchOptimizer(instance, time_limit=30.0).solve("regret_portfolio", oracle_costs=oracle)
        envelope = LaunchOptimizer(instance, time_limit=30.0).solve("envelope_regret_portfolio", oracle_costs=oracle)

        extensive_metrics = evaluate_solution(instance, extensive, oracle_costs=oracle)
        envelope_metrics = evaluate_solution(instance, envelope, oracle_costs=oracle)

        self.assertAlmostEqual(float(envelope_metrics["max_regret"]), float(extensive_metrics["max_regret"]), places=8)
        self.assertAlmostEqual(float(envelope_metrics["expected_cost"]), float(extensive_metrics["expected_cost"]), places=8)
        self.assertLess(int(envelope.extra["binary_var_count"]), int(extensive.extra.get("binary_var_count", 9999)))

    def test_integral_recourse_regret_matches_extensive_regret_on_tiny_instance(self) -> None:
        instance = envelope_test_instance()
        oracle = compute_oracle_costs(instance, time_limit=30.0)
        extensive = LaunchOptimizer(instance, time_limit=30.0).solve("regret_portfolio", oracle_costs=oracle)
        integral_recourse = LaunchOptimizer(instance, time_limit=30.0).solve("integral_recourse_regret", oracle_costs=oracle)

        extensive_metrics = evaluate_solution(instance, extensive, oracle_costs=oracle)
        integral_metrics = evaluate_solution(instance, integral_recourse, oracle_costs=oracle)

        self.assertAlmostEqual(float(integral_metrics["max_regret"]), float(extensive_metrics["max_regret"]), places=8)
        self.assertAlmostEqual(float(integral_metrics["expected_cost"]), float(extensive_metrics["expected_cost"]), places=8)
        self.assertLess(int(integral_recourse.extra["binary_var_count"]), int(extensive.extra["binary_var_count"]))

    def test_summarize_envelope_reports_match_and_binary_reduction(self) -> None:
        metrics = pd.DataFrame(
            [
                {
                    "method": "regret_portfolio_full",
                    "instance_key": "a",
                    "solve_seconds": 10.0,
                    "max_regret": 1.0,
                    "expected_cost": 5.0,
                    "binary_var_count": 100,
                    "matches_full_regret": True,
                    "matches_full_cost": True,
                },
                {
                    "method": "envelope_regret_portfolio",
                    "instance_key": "a",
                    "solve_seconds": 4.0,
                    "max_regret": 1.0,
                    "expected_cost": 5.0,
                    "binary_var_count": 40,
                    "continuous_var_count": 20,
                    "envelope_cut_count": 90,
                    "matches_full_regret": True,
                    "matches_full_cost": True,
                },
            ]
        )

        summary = summarize_envelope(metrics)

        row = summary.iloc[0]
        self.assertEqual(int(row["regret_matches"]), 1)
        self.assertAlmostEqual(float(row["binary_reduction_share"]), 0.6)
        self.assertAlmostEqual(float(row["full_to_envelope_time_ratio"]), 2.5)

    def test_summarize_integral_recourse_reports_exact_match_and_binary_reduction(self) -> None:
        metrics = pd.DataFrame(
            [
                {
                    "method": "regret_portfolio_full",
                    "instance_key": "a",
                    "solve_seconds": 10.0,
                    "binary_var_count": 100,
                    "max_regret": 1.0,
                    "expected_cost": 5.0,
                    "matches_full_regret": True,
                    "matches_full_cost": True,
                },
                {
                    "method": "integral_recourse_regret",
                    "instance_key": "a",
                    "solve_seconds": 2.5,
                    "binary_var_count": 40,
                    "continuous_var_count": 61,
                    "max_regret": 1.0,
                    "expected_cost": 5.0,
                    "matches_full_regret": True,
                    "matches_full_cost": True,
                },
            ]
        )

        summary = summarize_integral_recourse(metrics)

        row = summary.iloc[0]
        self.assertEqual(int(row["regret_matches"]), 1)
        self.assertAlmostEqual(float(row["binary_reduction_share"]), 0.6)
        self.assertAlmostEqual(float(row["full_to_integral_recourse_time_ratio"]), 4.0)

    def test_separate_polymatroid_cuts_adds_only_violated_thresholds(self) -> None:
        instance = envelope_test_instance()
        pool = initial_threshold_pool(instance)
        q_values = {(1, "AAA_6"): 3.2, (1, "BBB_6"): 0.0}
        assigned_by_flight = {"f1": 1.0, "f2": 1.0, "f3": 0.0}

        added = separate_polymatroid_cuts(instance, pool, q_values, assigned_by_flight, tolerance=1e-8)

        self.assertGreaterEqual(added, 1)
        self.assertGreater(cut_pool_size(pool), 0)


if __name__ == "__main__":
    unittest.main()
