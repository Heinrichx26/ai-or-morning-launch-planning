from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
except ModuleNotFoundError:
    HistGradientBoostingClassifier = None

from .optimizers import SolveResult, certificate_candidate_scores, evaluate_solution
from .smoke_data import LaunchInstance


@dataclass
class DistilledModel:
    model: object
    feature_columns: list[str]


class DistanceProbabilityClassifier:
    def fit(self, x: pd.DataFrame, y: pd.Series) -> "DistanceProbabilityClassifier":
        values = x.to_numpy(dtype=float)
        labels = y.to_numpy(dtype=int)
        self.mean_ = values.mean(axis=0)
        self.scale_ = values.std(axis=0) + 1e-6
        self.prior_ = float(np.clip(labels.mean() if len(labels) else 0.05, 0.01, 0.99))
        if len(np.unique(labels)) < 2:
            self.single_class_ = True
            return self
        self.single_class_ = False
        self.pos_center_ = values[labels == 1].mean(axis=0)
        self.neg_center_ = values[labels == 0].mean(axis=0)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        values = x.to_numpy(dtype=float)
        if getattr(self, "single_class_", False):
            p = np.full(len(values), self.prior_, dtype=float)
            return np.column_stack([1.0 - p, p])
        z = (values - self.mean_) / self.scale_
        pos = (self.pos_center_ - self.mean_) / self.scale_
        neg = (self.neg_center_ - self.mean_) / self.scale_
        d_pos = ((z - pos) ** 2).mean(axis=1)
        d_neg = ((z - neg) ** 2).mean(axis=1)
        score = d_neg - d_pos + np.log(self.prior_ / (1.0 - self.prior_))
        score = np.clip(score, -40.0, 40.0)
        p = 1.0 / (1.0 + np.exp(-score))
        return np.column_stack([1.0 - p, p])


def candidate_features(instance: LaunchInstance) -> pd.DataFrame:
    flights = instance.flights.copy()
    flights["block_rank"] = (
        flights.sort_values(["block_id", "weight"], ascending=[True, False])
        .groupby("block_id")
        .cumcount()
        + 1
    )
    demand_by_origin = flights.groupby("Origin").size().to_dict()
    demand_by_block = flights.groupby("block_id").size().to_dict()
    inventory_by_origin = dict(zip(instance.inventory["overnight_airport"], instance.inventory["aircraft_count"]))
    caps = instance.scenarios.pivot(index="scenario_id", columns="block_id", values="capacity").fillna(0)
    scenario_type_counts = instance.scenarios[["scenario_id", "scenario_type"]].drop_duplicates()
    weather_scenario_share = (
        float(scenario_type_counts["scenario_type"].astype(str).str.contains("weather", case=False, na=False).mean())
        if "scenario_type" in scenario_type_counts.columns and not scenario_type_counts.empty
        else 0.0
    )

    certificate_scores = certificate_candidate_scores(instance)
    certificate_lookup = (
        certificate_scores.set_index("candidate_id")["certificate_score"].to_dict()
        if not certificate_scores.empty
        else {}
    )

    block_stats: dict[str, dict[str, float]] = {}
    for block, demand in demand_by_block.items():
        values = caps[block].to_numpy(float) if block in caps else np.zeros(len(caps), dtype=float)
        shortages = np.maximum(0.0, demand - values) / max(1, demand)
        block_stats[block] = {
            "block_demand": float(demand),
            "cap_mean": float(values.mean()) if len(values) else 0.0,
            "cap_min": float(values.min()) if len(values) else 0.0,
            "cap_q10": float(np.quantile(values, 0.10)) if len(values) else 0.0,
            "cap_q25": float(np.quantile(values, 0.25)) if len(values) else 0.0,
            "shortage_mean": float(shortages.mean()) if len(values) else 1.0,
            "shortage_max": float(shortages.max()) if len(values) else 1.0,
        }

    rows: list[dict[str, float | int | str]] = []
    flight_lookup = flights.set_index("flight_id").to_dict("index")
    for cand in instance.candidates.itertuples(index=False):
        cand_dict = cand._asdict()
        flight = flight_lookup[cand.flight_id]
        origin = str(flight["Origin"])
        source = str(cand.source_airport)
        block = str(flight["block_id"])
        stats = block_stats[block]
        rank = int(flight["block_rank"])
        cap_values = caps[block].to_numpy(float) if block in caps else np.zeros(len(caps), dtype=float)
        survival = float(np.mean(cap_values >= rank)) if len(cap_values) else 0.0
        weather_binding = (
            float(np.mean((cap_values < stats["block_demand"]) & (cap_values <= rank)))
            if len(cap_values)
            else 0.0
        )
        origin_demand = demand_by_origin.get(origin, 0)
        source_demand = demand_by_origin.get(source, 0)
        origin_inventory = inventory_by_origin.get(origin, 0)
        source_inventory = inventory_by_origin.get(source, 0)
        last_arrival = float(cand_dict.get("last_arrival_min", 0.0) or 0.0)
        movement_burden = float(cand.ferry_cost) * (1.0 + max(0.0, origin_demand - origin_inventory))
        rows.append(
            {
                "candidate_id": cand.candidate_id,
                "aircraft_id": str(cand_dict.get("aircraft_id", "")),
                "flight_id": cand.flight_id,
                "source_airport": source,
                "origin": origin,
                "same_airport": int(source == origin),
                "ferry_cost": float(cand.ferry_cost),
                "weight": float(flight["weight"]),
                "dep_hour": int(flight["dep_hour"]),
                "block_rank": rank,
                "survival": survival,
                "origin_demand": float(origin_demand),
                "source_demand": float(source_demand),
                "origin_inventory": float(origin_inventory),
                "source_inventory": float(source_inventory),
                "origin_deficit": float(max(0, origin_demand - origin_inventory)),
                "source_surplus": float(max(0, source_inventory - source_demand)),
                "source_deficit": float(max(0, source_demand - source_inventory)),
                "movement_burden": movement_burden,
                "last_arrival_hour": last_arrival / 60.0,
                "certificate_score": float(certificate_lookup.get(str(cand.candidate_id), 0.0)),
                "weather_scenario_share": weather_scenario_share,
                "weather_binding_share": weather_binding,
                **stats,
            }
        )
    return pd.DataFrame(rows)


FEATURE_COLUMNS = [
    "same_airport",
    "ferry_cost",
    "weight",
    "dep_hour",
    "block_rank",
    "survival",
    "origin_demand",
    "source_demand",
    "origin_inventory",
    "source_inventory",
    "origin_deficit",
    "source_surplus",
    "source_deficit",
    "movement_burden",
    "last_arrival_hour",
    "certificate_score",
    "weather_scenario_share",
    "weather_binding_share",
    "block_demand",
    "cap_mean",
    "cap_min",
    "cap_q10",
    "cap_q25",
    "shortage_mean",
    "shortage_max",
]


def fit_distilled_model(training_frames: list[pd.DataFrame]) -> DistilledModel:
    if not training_frames:
        raise ValueError("training_frames must contain at least one labeled frame")
    train = pd.concat(training_frames, ignore_index=True)
    x = train[FEATURE_COLUMNS].fillna(0.0)
    y = train["label"].astype(int)
    if HistGradientBoostingClassifier is not None and y.nunique() > 1:
        model = HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.045,
            max_leaf_nodes=15,
            l2_regularization=0.02,
            random_state=11,
        )
    else:
        model = DistanceProbabilityClassifier()
    model.fit(x, y)
    return DistilledModel(model=model, feature_columns=FEATURE_COLUMNS)


def solve_with_distilled_model(
    instance: LaunchInstance,
    distilled: DistilledModel,
    repair: bool = False,
) -> SolveResult:
    features = candidate_features(instance)
    if features.empty:
        return SolveResult("regret_distilled", "empty", np.nan, 0.0, pd.DataFrame(), {})

    probs = distilled.model.predict_proba(features[distilled.feature_columns].fillna(0.0))[:, 1]
    features = features.copy()
    features["prob"] = probs
    features["score"] = features["prob"] * (features["weight"] + features["survival"]) - 0.25 * features["ferry_cost"]

    remaining = dict(zip(instance.inventory["overnight_airport"], instance.inventory["aircraft_count"]))
    selected_ids: list[str] = []
    selected_flights: set[str] = set()
    selected_aircraft: set[str] = set()
    for row in features.sort_values("score", ascending=False).itertuples(index=False):
        if row.score <= 0:
            continue
        if row.flight_id in selected_flights:
            continue
        if getattr(row, "aircraft_id", "") and row.aircraft_id in selected_aircraft:
            continue
        if remaining.get(row.source_airport, 0) <= 0:
            continue
        selected_ids.append(row.candidate_id)
        selected_flights.add(row.flight_id)
        if getattr(row, "aircraft_id", ""):
            selected_aircraft.add(row.aircraft_id)
        remaining[row.source_airport] = remaining.get(row.source_airport, 0) - 1

    if repair:
        selected_ids, selected_flights, remaining = _repair_origin_shortfalls(
            instance,
            features,
            selected_ids,
            selected_flights,
            remaining,
        )

    selected = instance.candidates[instance.candidates["candidate_id"].isin(selected_ids)].copy()
    selected = selected.merge(
        instance.flights[["flight_id", "Origin", "Dest", "block_id", "CRSDepTime", "weight"]],
        on="flight_id",
        how="left",
        suffixes=("", "_flight"),
    )
    result = SolveResult(
        method="regret_distilled_repaired" if repair else "regret_distilled",
        status="distilled_feasible",
        objective=float(-features.loc[features["candidate_id"].isin(selected_ids), "score"].sum()),
        solve_seconds=0.0,
        selected=selected.reset_index(drop=True),
        extra={"mean_teacher_probability": float(np.mean(probs))},
    )
    return result


def _repair_origin_shortfalls(
    instance: LaunchInstance,
    features: pd.DataFrame,
    selected_ids: list[str],
    selected_flights: set[str],
    remaining: dict[str, int],
) -> tuple[list[str], set[str], dict[str, int]]:
    flights = instance.flights.copy()
    demand_by_origin = flights.groupby("Origin").size().to_dict()
    caps = instance.scenarios.copy()
    caps["origin"] = caps["block_id"].astype(str).str.split("_").str[0]
    origin_capacity = caps.groupby(["scenario_id", "origin"])["capacity"].sum().reset_index()
    target_by_origin = (
        origin_capacity.groupby("origin")["capacity"]
        .quantile(0.85)
        .apply(np.ceil)
        .astype(int)
        .to_dict()
    )
    selected_feature = features[features["candidate_id"].isin(selected_ids)]
    selected_by_origin = selected_feature.groupby("origin").size().to_dict()
    selected_aircraft = set(selected_feature["aircraft_id"].dropna().astype(str).tolist())

    for origin, demand in demand_by_origin.items():
        target = min(int(demand), int(target_by_origin.get(origin, demand)))
        current = int(selected_by_origin.get(origin, 0))
        if current >= target:
            continue
        pool = features[
            (features["origin"].eq(origin))
            & (~features["flight_id"].isin(selected_flights))
            & (features["source_airport"].ne(origin))
        ].copy()
        if pool.empty:
            continue
        pool["repair_score"] = (
            pool["prob"] * (pool["weight"] + 0.5 * pool["survival"])
            + 0.15 * pool["origin_deficit"]
            + 0.05 * pool["source_surplus"]
            - 0.35 * pool["ferry_cost"]
        )
        for row in pool.sort_values("repair_score", ascending=False).itertuples(index=False):
            if current >= target:
                break
            if row.flight_id in selected_flights:
                continue
            if getattr(row, "aircraft_id", "") and row.aircraft_id in selected_aircraft:
                continue
            if remaining.get(row.source_airport, 0) <= 0:
                continue
            selected_ids.append(row.candidate_id)
            selected_flights.add(row.flight_id)
            if getattr(row, "aircraft_id", ""):
                selected_aircraft.add(row.aircraft_id)
            remaining[row.source_airport] = remaining.get(row.source_airport, 0) - 1
            current += 1
    return selected_ids, selected_flights, remaining


def make_labeled_frame(
    instance: LaunchInstance,
    teacher_result: SolveResult,
    retained_candidates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    features = candidate_features(instance)
    selected = set(teacher_result.selected["candidate_id"].tolist()) if not teacher_result.selected.empty else set()
    if retained_candidates is None:
        retained = selected
    else:
        retained = set(retained_candidates["candidate_id"].astype(str).tolist())
    features["selected_assignment"] = features["candidate_id"].astype(str).isin(selected).astype(int)
    features["retained_after_audit"] = features["candidate_id"].astype(str).isin(retained).astype(int)
    features["label"] = features["retained_after_audit"]
    return features
