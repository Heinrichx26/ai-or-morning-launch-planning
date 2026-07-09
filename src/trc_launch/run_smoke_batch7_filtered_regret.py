from __future__ import annotations

from pathlib import Path

import pandas as pd

from .optimizers import (
    LaunchOptimizer,
    certificate_candidate_scores,
    compute_oracle_costs,
    evaluate_solution,
)
from .run_smoke_batch1 import DEFAULT_BTS_CSV
from .run_smoke_batch4_multicarrier import PAIRS
from .smoke_data import LaunchInstance, build_tail_launch_instance, load_bts_month


def _key(carrier: str, date: str) -> str:
    return f"{carrier}_{date}"


def _filtered_instance(instance: LaunchInstance, per_flight: int = 8, per_aircraft: int = 4) -> tuple[LaunchInstance, int]:
    scores = certificate_candidate_scores(instance)
    candidates = instance.candidates.reset_index(drop=True).copy()
    scored = candidates.merge(scores[["candidate_id", "certificate_score"]], on="candidate_id", how="left")
    scored["certificate_score"] = scored["certificate_score"].fillna(-1e9)

    keep_ids: set[str] = set()
    for _, group in scored.groupby("flight_id"):
        keep_ids.update(group.nlargest(per_flight, "certificate_score")["candidate_id"].astype(str).tolist())

    if "aircraft_id" in scored.columns:
        for _, group in scored.groupby("aircraft_id"):
            keep_ids.update(group.nlargest(per_aircraft, "certificate_score")["candidate_id"].astype(str).tolist())

    if {"source_airport", "origin"}.issubset(scored.columns):
        same_airport = scored["source_airport"].astype(str).eq(scored["origin"].astype(str))
    else:
        same_airport = scored["ferry_cost"].eq(0)
    keep_ids.update(scored[same_airport]["candidate_id"].astype(str).tolist())
    filtered_candidates = candidates[candidates["candidate_id"].astype(str).isin(keep_ids)].reset_index(drop=True)
    filtered = LaunchInstance(
        date=instance.date,
        carrier=instance.carrier,
        region_airports=instance.region_airports,
        flights=instance.flights,
        inventory=instance.inventory,
        candidates=filtered_candidates,
        scenarios=instance.scenarios,
        total_weight=instance.total_weight,
    )
    return filtered, len(candidates)


def main() -> None:
    output_dir = Path("results/trc_smoke/batch7_filtered_regret")
    output_dir.mkdir(parents=True, exist_ok=True)

    month = load_bts_month(DEFAULT_BTS_CSV)
    rows: list[dict[str, object]] = []
    for carrier, date in PAIRS:
        instance = build_tail_launch_instance(
            month,
            date=date,
            carrier=carrier,
            scenario_count=10,
            scenario_mode="stress",
            seed=31,
        )
        if instance.flights.empty or instance.inventory.empty:
            continue

        oracle_costs = compute_oracle_costs(instance, time_limit=6.0)
        full = LaunchOptimizer(instance, time_limit=20.0).solve("regret_portfolio", oracle_costs=oracle_costs)
        full_metrics = evaluate_solution(instance, full, oracle_costs=oracle_costs)
        full_metrics["method"] = "regret_portfolio_full"
        full_metrics["instance_key"] = _key(carrier, date)
        full_metrics["candidate_count"] = len(instance.candidates)
        full_metrics["candidate_retention"] = 1.0
        rows.append(full_metrics)

        for per_flight, per_aircraft in [(4, 2), (8, 4), (12, 6), (16, 8)]:
            filtered_instance, original_count = _filtered_instance(
                instance,
                per_flight=per_flight,
                per_aircraft=per_aircraft,
            )
            filtered = LaunchOptimizer(filtered_instance, time_limit=20.0).solve(
                "regret_portfolio",
                oracle_costs=oracle_costs,
            )
            filtered_metrics = evaluate_solution(instance, filtered, oracle_costs=oracle_costs)
            filtered_metrics["method"] = f"certificate_filtered_regret_k{per_flight}"
            filtered_metrics["instance_key"] = _key(carrier, date)
            filtered_metrics["candidate_count"] = len(filtered_instance.candidates)
            filtered_metrics["candidate_retention"] = len(filtered_instance.candidates) / max(1, original_count)
            rows.append(filtered_metrics)

        cert = LaunchOptimizer(instance, time_limit=20.0).solve("certificate_value")
        cert_metrics = evaluate_solution(instance, cert, oracle_costs=oracle_costs)
        cert_metrics["instance_key"] = _key(carrier, date)
        cert_metrics["candidate_count"] = len(instance.candidates)
        cert_metrics["candidate_retention"] = 1.0
        rows.append(cert_metrics)

        saa = LaunchOptimizer(instance, time_limit=20.0).solve("saa_extensive")
        saa_metrics = evaluate_solution(instance, saa, oracle_costs=oracle_costs)
        saa_metrics["instance_key"] = _key(carrier, date)
        saa_metrics["candidate_count"] = len(instance.candidates)
        saa_metrics["candidate_retention"] = 1.0
        rows.append(saa_metrics)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "batch7_filtered_metrics.csv", index=False)
    summary = (
        metrics.groupby("method")
        .agg(
            cases=("instance_key", "count"),
            mean_expected_cost=("expected_cost", "mean"),
            mean_cvar95_cost=("cvar95_cost", "mean"),
            mean_launch_weight_share=("mean_launch_weight_share", "mean"),
            mean_max_regret=("max_regret", "mean"),
            mean_solve_seconds=("solve_seconds", "mean"),
            mean_ferry_moves=("ferry_moves", "mean"),
            mean_candidate_count=("candidate_count", "mean"),
            mean_candidate_retention=("candidate_retention", "mean"),
        )
        .reset_index()
        .sort_values(["mean_max_regret", "mean_expected_cost"])
    )
    summary.to_csv(output_dir / "batch7_filtered_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
