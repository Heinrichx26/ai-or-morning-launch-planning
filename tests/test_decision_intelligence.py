from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import pandas as pd

from src.trc_launch.decision_intelligence_audit import (
    _load_decision_weather_features,
    _learned_proposal_instance,
    _same_airport_ids,
    _solve_lg_rccc_audited,
)
from src.trc_launch.distilled_policy import DistilledModel, candidate_features, make_labeled_frame
from src.trc_launch.optimizers import SolveResult
from src.trc_launch.smoke_data import LaunchInstance


class ConstantModel:
    def predict_proba(self, x):
        return [[0.7, 0.3] for _ in range(len(x))]


def toy_instance() -> LaunchInstance:
    flights = pd.DataFrame(
        [
            {
                "flight_id": "f1",
                "Origin": "JFK",
                "Dest": "ORD",
                "CRSDepTime": 600,
                "CRSDepTime_min": 360,
                "dep_hour": 6,
                "block_id": "JFK_6",
                "Distance": 700,
                "weight": 1.5,
                "Cancelled": 0,
            },
            {
                "flight_id": "f2",
                "Origin": "LGA",
                "Dest": "BOS",
                "CRSDepTime": 610,
                "CRSDepTime_min": 370,
                "dep_hour": 6,
                "block_id": "LGA_6",
                "Distance": 200,
                "weight": 1.1,
                "Cancelled": 0,
            },
        ]
    )
    inventory = pd.DataFrame(
        [
            {"overnight_airport": "JFK", "aircraft_count": 1},
            {"overnight_airport": "LGA", "aircraft_count": 1},
        ]
    )
    candidates = pd.DataFrame(
        [
            {"candidate_id": "JFK_A->f1", "aircraft_id": "JFK_A", "source_airport": "JFK", "flight_id": "f1", "origin": "JFK", "block_id": "JFK_6", "ferry_cost": 0.0, "last_arrival_min": 1000},
            {"candidate_id": "LGA_A->f1", "aircraft_id": "LGA_A", "source_airport": "LGA", "flight_id": "f1", "origin": "JFK", "block_id": "JFK_6", "ferry_cost": 0.28, "last_arrival_min": 1000},
            {"candidate_id": "JFK_A->f2", "aircraft_id": "JFK_A", "source_airport": "JFK", "flight_id": "f2", "origin": "LGA", "block_id": "LGA_6", "ferry_cost": 0.28, "last_arrival_min": 1000},
            {"candidate_id": "LGA_A->f2", "aircraft_id": "LGA_A", "source_airport": "LGA", "flight_id": "f2", "origin": "LGA", "block_id": "LGA_6", "ferry_cost": 0.0, "last_arrival_min": 1000},
        ]
    )
    scenarios = pd.DataFrame(
        [
            {"scenario_id": 0, "block_id": "JFK_6", "capacity": 1, "scenario_type": "weather", "shock_airport": "JFK", "shock_hour": 6},
            {"scenario_id": 0, "block_id": "LGA_6", "capacity": 1, "scenario_type": "weather", "shock_airport": "JFK", "shock_hour": 6},
            {"scenario_id": 1, "block_id": "JFK_6", "capacity": 0, "scenario_type": "weather", "shock_airport": "JFK", "shock_hour": 6},
            {"scenario_id": 1, "block_id": "LGA_6", "capacity": 1, "scenario_type": "weather", "shock_airport": "JFK", "shock_hour": 6},
        ]
    )
    return LaunchInstance(
        date="2025-01-01",
        carrier="DL",
        region_airports=["JFK", "LGA"],
        flights=flights,
        inventory=inventory,
        candidates=candidates,
        scenarios=scenarios,
        total_weight=float(flights["weight"].sum()),
    )


class DecisionIntelligenceTests(unittest.TestCase):
    def test_candidate_features_are_deterministic(self) -> None:
        instance = toy_instance()
        first = candidate_features(instance).sort_values("candidate_id").reset_index(drop=True)
        second = candidate_features(instance).sort_values("candidate_id").reset_index(drop=True)

        pd.testing.assert_frame_equal(first, second)
        self.assertIn("certificate_score", first.columns)
        self.assertIn("weather_binding_share", first.columns)

    def test_labeled_frame_has_retained_and_selected_labels(self) -> None:
        instance = toy_instance()
        selected = instance.candidates[instance.candidates["candidate_id"].isin(["JFK_A->f1"])].copy()
        retained = instance.candidates[instance.candidates["candidate_id"].isin(["JFK_A->f1", "LGA_A->f1"])].copy()
        teacher = SolveResult("teacher", "ok", 0.0, 0.0, selected, {})

        labels = make_labeled_frame(instance, teacher, retained_candidates=retained)

        by_id = labels.set_index("candidate_id")
        self.assertEqual(int(by_id.loc["JFK_A->f1", "selected_assignment"]), 1)
        self.assertEqual(int(by_id.loc["LGA_A->f1", "selected_assignment"]), 0)
        self.assertEqual(int(by_id.loc["LGA_A->f1", "retained_after_audit"]), 1)
        self.assertEqual(int(by_id.loc["LGA_A->f1", "label"]), 1)

    def test_same_airport_edges_are_always_in_proposal(self) -> None:
        instance = toy_instance()
        model = DistilledModel(model=ConstantModel(), feature_columns=["same_airport"])

        proposal, _ = _learned_proposal_instance(instance, model, per_flight=1, per_aircraft=1)
        proposed_ids = set(proposal.candidates["candidate_id"].astype(str))

        self.assertTrue(_same_airport_ids(instance).issubset(proposed_ids))

    def test_saved_weather_features_are_loaded_without_fetch(self) -> None:
        with TemporaryDirectory() as tmp:
            feature_dir = Path(tmp) / "weather_features"
            feature_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "airport": "JFK",
                        "local_hour": 6,
                        "weather_severity": 0.7,
                    }
                ]
            ).to_csv(feature_dir / "weather_nyc_DL_2025-01-01.csv", index=False)
            item = SimpleNamespace(region="nyc", carrier="DL", date="2025-01-01")
            args = SimpleNamespace(
                weather_feature_dir=str(feature_dir),
                data_dir=tmp,
                use_weather=False,
            )

            features = _load_decision_weather_features(item, args)

        self.assertIsNotNone(features)
        self.assertEqual(float(features.loc[0, "weather_severity"]), 0.7)

    def test_lg_rccc_audit_can_fallback_to_full_graph(self) -> None:
        instance = toy_instance()
        proposal = LaunchInstance(
            date=instance.date,
            carrier=instance.carrier,
            region_airports=instance.region_airports,
            flights=instance.flights,
            inventory=instance.inventory,
            candidates=instance.candidates.head(1).copy(),
            scenarios=instance.scenarios,
            total_weight=instance.total_weight,
        )
        solved = []

        def fake_solve(self, method, oracle_costs=None):
            solved.append(len(self.instance.candidates))
            selected = self.instance.candidates.head(1).copy()
            return SolveResult(method, "ok", 0.0, 0.0, selected, {})

        def fake_eval(eval_instance, result, oracle_costs=None):
            if len(solved) < 3:
                return {"expected_cost": 0.0, "max_regret": 9.0}
            return {"expected_cost": 0.0, "max_regret": 1.0}

        with patch("src.trc_launch.decision_intelligence_audit.LaunchOptimizer.solve", new=fake_solve), patch(
            "src.trc_launch.decision_intelligence_audit.evaluate_solution", new=fake_eval
        ):
            _, final_instance, extra = _solve_lg_rccc_audited(
                instance,
                proposal,
                oracle_costs={0: 0.0, 1: 0.0},
                full_expected=0.0,
                full_regret=1.0,
                time_limit=1.0,
            )

        self.assertEqual(extra["fallback_stage"], "full")
        self.assertEqual(extra["audit_expansions"], 2)
        self.assertEqual(len(final_instance.candidates), len(instance.candidates))

    def test_ai_or_metrics_schema_numeric_fields_are_parseable(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "method": "lg_rccc_audited",
                    "status": "optimal_or_feasible",
                    "max_regret": "1.0",
                    "candidate_retention": "0.7",
                    "solve_seconds": "0.25",
                    "matches_full_regret": "True",
                    "instance_key": "toy",
                }
            ]
        )
        required = {
            "method",
            "status",
            "max_regret",
            "candidate_retention",
            "solve_seconds",
            "matches_full_regret",
            "instance_key",
        }

        self.assertTrue(required.issubset(frame.columns))
        for column in ["max_regret", "candidate_retention", "solve_seconds"]:
            parsed = pd.to_numeric(frame[column], errors="coerce")
            self.assertFalse(parsed.isna().any())


if __name__ == "__main__":
    unittest.main()
