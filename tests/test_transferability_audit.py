import tempfile
import unittest
import zipfile
from pathlib import Path

from src.trc_launch.optimizers import LaunchOptimizer, compute_oracle_costs, evaluate_solution
from src.trc_launch.transferability_audit import FeedSpec, build_gtfs_launch_instance, gtfs_time_to_minutes


class TransferabilityAuditTests(unittest.TestCase):
    def test_gtfs_time_to_minutes_handles_after_midnight_times(self) -> None:
        self.assertEqual(gtfs_time_to_minutes("06:15:00"), 375)
        self.assertEqual(gtfs_time_to_minutes("25:05:00"), 1505)
        self.assertEqual(gtfs_time_to_minutes("bad"), -1)

    def test_build_gtfs_launch_instance_runs_regret_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "tiny_gtfs.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "stops.txt",
                    "stop_id,stop_name,stop_lat,stop_lon,parent_station\n"
                    "A,A,42.0,-71.0,\nB,B,42.1,-71.1,\nC,C,42.2,-71.2,\n",
                )
                archive.writestr("routes.txt", "route_id,route_type\nR1,3\n")
                archive.writestr(
                    "trips.txt",
                    "route_id,service_id,trip_id,trip_headsign,direction_id,block_id\n"
                    "R1,WKD,t1,B,0,b1\nR1,WKD,t2,C,0,b2\nR1,WKD,t3,C,0,b3\n",
                )
                archive.writestr(
                    "stop_times.txt",
                    "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
                    "t1,06:00:00,06:00:00,A,1\n"
                    "t1,06:30:00,06:30:00,B,2\n"
                    "t2,06:10:00,06:10:00,B,1\n"
                    "t2,06:45:00,06:45:00,C,2\n"
                    "t3,06:20:00,06:20:00,C,1\n"
                    "t3,06:55:00,06:55:00,A,2\n",
                )

            spec = FeedSpec("tiny", "unit_gtfs", "Tiny Transit", "file://tiny")
            instance = build_gtfs_launch_instance(zip_path, spec, max_services=3, scenario_count=3, seed=5)
            self.assertEqual(len(instance.flights), 3)
            self.assertEqual(len(instance.inventory), 3)
            self.assertGreaterEqual(len(instance.candidates), 9)

            oracle = compute_oracle_costs(instance, time_limit=10.0)
            result = LaunchOptimizer(instance, time_limit=10.0).solve("regret_portfolio", oracle_costs=oracle)
            metrics = evaluate_solution(instance, result, oracle_costs=oracle)
            self.assertIn("max_regret", metrics)


if __name__ == "__main__":
    unittest.main()
