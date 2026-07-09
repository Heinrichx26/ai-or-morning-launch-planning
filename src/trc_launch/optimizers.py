from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from time import perf_counter
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from .smoke_data import LaunchInstance


SolveMethod = Literal[
    "launch_value",
    "certificate_value",
    "saa_extensive",
    "active_scenario_saa",
    "predict_opt",
    "cvar_extensive",
    "mean_cvar95_extensive",
    "minimax_extensive",
    "regret_portfolio",
    "integral_recourse_regret",
    "envelope_regret_portfolio",
    "robust_quantile",
]


@dataclass
class SolveResult:
    method: str
    status: str
    objective: float
    solve_seconds: float
    selected: pd.DataFrame
    extra: dict[str, float]


def _capacity_table(instance: LaunchInstance) -> pd.DataFrame:
    table = instance.scenarios.pivot(index="scenario_id", columns="block_id", values="capacity")
    table = table.fillna(0).astype(int)
    return table


def launch_value_envelope_terms(weights: list[float] | np.ndarray, capacity: int) -> list[tuple[float, float, np.ndarray]]:
    clean = np.asarray(weights, dtype=float)
    if clean.size == 0:
        return [(0.0, 0.0, np.zeros(0))]
    thresholds = [0.0] + sorted({float(value) for value in clean if value > 0.0})
    terms: list[tuple[float, float, np.ndarray]] = []
    for theta in thresholds:
        coefficients = np.maximum(clean - theta, 0.0)
        terms.append((float(theta), float(max(0, int(capacity)) * theta), coefficients))
    return terms


def _collapse_scenarios(instance: LaunchInstance, mode: str) -> pd.DataFrame:
    caps = instance.scenarios.copy()
    if mode == "mean":
        one = caps.groupby("block_id")["capacity"].mean().round().astype(int).reset_index()
    elif mode == "q25":
        one = caps.groupby("block_id")["capacity"].quantile(0.25).apply(np.floor).astype(int).reset_index()
    else:
        raise ValueError(mode)
    one["scenario_id"] = 0
    one["capacity"] = one["capacity"].clip(lower=0)
    return one[["scenario_id", "block_id", "capacity"]]


def _copy_with_scenarios(instance: LaunchInstance, scenarios: pd.DataFrame) -> LaunchInstance:
    return LaunchInstance(
        date=instance.date,
        carrier=instance.carrier,
        region_airports=instance.region_airports,
        flights=instance.flights,
        inventory=instance.inventory,
        candidates=instance.candidates,
        scenarios=scenarios,
        total_weight=instance.total_weight,
    )


def _active_scenarios(instance: LaunchInstance, keep_share: float = 0.40, min_keep: int = 3) -> pd.DataFrame:
    flights = instance.flights.copy()
    demand = flights.groupby("block_id").size().to_dict()
    scenario_scores: list[tuple[float, int]] = []
    for sid, group in instance.scenarios.groupby("scenario_id"):
        deficit = 0.0
        for row in group.itertuples(index=False):
            block = str(row.block_id)
            deficit += max(0.0, float(demand.get(block, 0)) - float(row.capacity))
        scenario_scores.append((deficit, int(sid)))
    if not scenario_scores:
        return instance.scenarios.copy()
    keep_n = min(len(scenario_scores), max(min_keep, int(np.ceil(len(scenario_scores) * keep_share))))
    active_ids = {sid for _, sid in sorted(scenario_scores, reverse=True)[:keep_n]}
    active = instance.scenarios[instance.scenarios["scenario_id"].isin(active_ids)].copy()
    id_map = {sid: pos for pos, sid in enumerate(sorted(active["scenario_id"].unique()))}
    active["scenario_id"] = active["scenario_id"].map(id_map)
    return active.reset_index(drop=True)


class LaunchOptimizer:
    def __init__(self, instance: LaunchInstance, time_limit: float = 20.0):
        self.instance = instance
        self.time_limit = time_limit
        self.flights = instance.flights.reset_index(drop=True).copy()
        self.inventory = instance.inventory.reset_index(drop=True).copy()
        self.candidates = instance.candidates.reset_index(drop=True).copy()
        self.scenarios = instance.scenarios.reset_index(drop=True).copy()
        self.flight_index = {fid: i for i, fid in enumerate(self.flights["flight_id"])}
        self.candidates["flight_idx"] = self.candidates["flight_id"].map(self.flight_index)
        self.weight = self.flights["weight"].to_numpy(float)

    def solve(self, method: SolveMethod, oracle_costs: dict[int, float] | None = None) -> SolveResult:
        if method == "launch_value":
            return self._solve_launch_value()
        if method == "certificate_value":
            return self._solve_certificate_value()
        if method == "predict_opt":
            collapsed = _copy_with_scenarios(self.instance, _collapse_scenarios(self.instance, "mean"))
            result = LaunchOptimizer(collapsed, self.time_limit).solve("saa_extensive")
            result.method = "predict_opt"
            return result
        if method == "robust_quantile":
            collapsed = _copy_with_scenarios(self.instance, _collapse_scenarios(self.instance, "q25"))
            result = LaunchOptimizer(collapsed, self.time_limit).solve("saa_extensive")
            result.method = "robust_quantile"
            return result
        if method == "active_scenario_saa":
            active = _copy_with_scenarios(self.instance, _active_scenarios(self.instance))
            result = LaunchOptimizer(active, self.time_limit).solve("saa_extensive")
            result.method = "active_scenario_saa"
            return result
        if method == "saa_extensive":
            return self._solve_extensive("saa", oracle_costs=None)
        if method == "cvar_extensive":
            return self._solve_extensive("cvar", oracle_costs=None)
        if method == "mean_cvar95_extensive":
            return self._solve_extensive("cvar95", oracle_costs=None)
        if method == "minimax_extensive":
            return self._solve_extensive("minimax", oracle_costs=None)
        if method == "regret_portfolio":
            if oracle_costs is None:
                raise ValueError("regret_portfolio requires oracle_costs")
            return self._solve_extensive("regret", oracle_costs=oracle_costs)
        if method == "integral_recourse_regret":
            if oracle_costs is None:
                raise ValueError("integral_recourse_regret requires oracle_costs")
            return self._solve_extensive(
                "regret",
                oracle_costs=oracle_costs,
                method_name="integral_recourse_regret",
                relax_recourse=True,
            )
        if method == "envelope_regret_portfolio":
            if oracle_costs is None:
                raise ValueError("envelope_regret_portfolio requires oracle_costs")
            return self._solve_regret_envelope(oracle_costs=oracle_costs)
        raise ValueError(method)

    def solve_envelope_regret_with_thresholds(
        self,
        oracle_costs: dict[int, float],
        threshold_pool: dict[tuple[int, str], set[float]],
        method_name: str = "polymatroid_regret_cutting",
    ) -> SolveResult:
        return self._solve_regret_envelope(
            oracle_costs=oracle_costs,
            threshold_pool=threshold_pool,
            method_name=method_name,
            include_q_values=True,
        )

    def _base_assignment_constraints(self, n_vars: int, include_z: bool, z_offset: int, n_scenarios: int):
        rows = []
        lbs = []
        ubs = []

        def add(row_dict: dict[int, float], lb: float, ub: float) -> None:
            rows.append(row_dict)
            lbs.append(lb)
            ubs.append(ub)

        source_to_count = dict(zip(self.inventory["overnight_airport"], self.inventory["aircraft_count"]))
        for source, limit in source_to_count.items():
            idxs = self.candidates.index[self.candidates["source_airport"].eq(source)].to_list()
            add({int(i): 1.0 for i in idxs}, -np.inf, float(limit))

        if "aircraft_id" in self.candidates.columns:
            for aircraft_id in self.candidates["aircraft_id"].dropna().unique().tolist():
                idxs = self.candidates.index[self.candidates["aircraft_id"].eq(aircraft_id)].to_list()
                add({int(i): 1.0 for i in idxs}, -np.inf, 1.0)

        for fid, fidx in self.flight_index.items():
            idxs = self.candidates.index[self.candidates["flight_id"].eq(fid)].to_list()
            add({int(i): 1.0 for i in idxs}, -np.inf, 1.0)

        if include_z:
            n_flights = len(self.flights)
            for sid in range(n_scenarios):
                for fid, fidx in self.flight_index.items():
                    cidxs = self.candidates.index[self.candidates["flight_id"].eq(fid)].to_list()
                    zidx = z_offset + sid * n_flights + fidx
                    row = {int(zidx): 1.0}
                    for cidx in cidxs:
                        row[int(cidx)] = row.get(int(cidx), 0.0) - 1.0
                    add(row, -np.inf, 0.0)

            caps = _capacity_table(_copy_with_scenarios(self.instance, self.scenarios))
            block_lookup = self.flights["block_id"].to_dict()
            for sid in caps.index:
                pos = list(caps.index).index(sid)
                for block in caps.columns:
                    fidxs = [i for i, block_id in block_lookup.items() if block_id == block]
                    row = {z_offset + pos * n_flights + int(i): 1.0 for i in fidxs}
                    add(row, -np.inf, float(caps.loc[sid, block]))

        mat = lil_matrix((len(rows), n_vars), dtype=float)
        for r, row in enumerate(rows):
            for c, v in row.items():
                mat[r, c] = v
        return [LinearConstraint(mat.tocsr(), np.array(lbs), np.array(ubs))]

    def _solve_launch_value(self) -> SolveResult:
        start = perf_counter()
        n_c = len(self.candidates)
        if n_c == 0:
            return self._empty("launch_value", start)

        caps = _capacity_table(self.instance)
        block_counts = self.flights.groupby("block_id").size().to_dict()
        mean_caps = caps.mean(axis=0).to_dict()
        availability = {
            block: min(1.0, max(0.0, mean_caps.get(block, 0.0) / max(1, count)))
            for block, count in block_counts.items()
        }
        value = self.flights.set_index("flight_id").apply(
            lambda row: float(row["weight"]) * availability.get(row["block_id"], 0.0), axis=1
        )
        c = np.zeros(n_c)
        for i, row in self.candidates.iterrows():
            c[i] = float(row["ferry_cost"]) - float(value.loc[row["flight_id"]])

        constraints = self._base_assignment_constraints(n_c, include_z=False, z_offset=0, n_scenarios=0)
        res = milp(
            c=c,
            integrality=np.ones(n_c),
            bounds=Bounds(np.zeros(n_c), np.ones(n_c)),
            constraints=constraints,
            options={"time_limit": self.time_limit, "mip_rel_gap": 0.01},
        )
        selected = self._selected_from_x(res.x[:n_c] if res.x is not None else np.zeros(n_c))
        return SolveResult(
            method="launch_value",
            status=self._status_name(res.success, res.message),
            objective=float(res.fun) if res.fun is not None else np.nan,
            solve_seconds=perf_counter() - start,
            selected=selected,
            extra={"mean_block_availability": float(np.mean(list(availability.values()))) if availability else 0.0},
        )

    def _solve_certificate_value(self) -> SolveResult:
        start = perf_counter()
        n_c = len(self.candidates)
        if n_c == 0:
            return self._empty("certificate_value", start)

        score_frame = certificate_candidate_scores(self.instance)
        scored_rows = list(zip(score_frame["certificate_score"], score_frame["candidate_row"]))

        remaining = dict(zip(self.inventory["overnight_airport"], self.inventory["aircraft_count"]))
        selected_idx: list[int] = []
        used_flights: set[str] = set()
        used_aircraft: set[str] = set()
        for score, idx in sorted(scored_rows, reverse=True):
            if score <= 0:
                continue
            cand = self.candidates.loc[idx]
            source = str(cand["source_airport"])
            fid = str(cand["flight_id"])
            aircraft_id = str(cand["aircraft_id"]) if "aircraft_id" in self.candidates.columns else ""
            if remaining.get(source, 0) <= 0 or fid in used_flights:
                continue
            if aircraft_id and aircraft_id in used_aircraft:
                continue
            selected_idx.append(idx)
            used_flights.add(fid)
            if aircraft_id:
                used_aircraft.add(aircraft_id)
            remaining[source] = remaining.get(source, 0) - 1

        x = np.zeros(n_c)
        if selected_idx:
            x[selected_idx] = 1.0
        selected = self._selected_from_x(x)
        return SolveResult(
            method="certificate_value",
            status="heuristic_feasible",
            objective=float(-sum(score for score, idx in scored_rows if idx in selected_idx)),
            solve_seconds=perf_counter() - start,
            selected=selected,
            extra={"mean_certificate_score": float(score_frame["certificate_score"].mean()) if not score_frame.empty else 0.0},
        )

    def _solve_extensive(
        self,
        mode: Literal["saa", "cvar", "cvar95", "minimax", "regret"],
        oracle_costs: dict[int, float] | None,
        method_name: str | None = None,
        relax_recourse: bool = False,
    ):
        start = perf_counter()
        n_c = len(self.candidates)
        n_f = len(self.flights)
        scenario_ids = sorted(self.scenarios["scenario_id"].unique().tolist())
        n_s = len(scenario_ids)
        if n_c == 0 or n_f == 0 or n_s == 0:
            return self._empty(mode, start)

        z_offset = n_c
        n_z = n_s * n_f
        theta_idx = None
        w_offset = None
        r_idx = None
        n_vars = n_c + n_z
        if mode in {"cvar", "cvar95"}:
            theta_idx = n_vars
            w_offset = n_vars + 1
            n_vars += 1 + n_s
        elif mode in {"minimax", "regret"}:
            r_idx = n_vars
            n_vars += 1

        c = np.zeros(n_vars)
        for i, row in self.candidates.iterrows():
            c[i] = float(row["ferry_cost"])
        if mode == "saa":
            for sid_pos in range(n_s):
                for fidx, weight in enumerate(self.weight):
                    c[z_offset + sid_pos * n_f + fidx] = -float(weight) / n_s
        elif mode in {"cvar", "cvar95"}:
            beta = 0.6 if mode == "cvar" else 1.0
            alpha = 0.90 if mode == "cvar" else 0.95
            for sid_pos in range(n_s):
                for fidx, weight in enumerate(self.weight):
                    c[z_offset + sid_pos * n_f + fidx] = -(1.0 - beta) * float(weight) / n_s
            c[theta_idx] = beta
            for sid_pos in range(n_s):
                c[w_offset + sid_pos] = beta / ((1.0 - alpha) * n_s)
        elif mode == "minimax":
            for sid_pos in range(n_s):
                for fidx, weight in enumerate(self.weight):
                    c[z_offset + sid_pos * n_f + fidx] = -0.02 * float(weight) / n_s
            c[r_idx] = 1.0
        elif mode == "regret":
            for sid_pos in range(n_s):
                for fidx, weight in enumerate(self.weight):
                    c[z_offset + sid_pos * n_f + fidx] = -0.05 * float(weight) / n_s
            c[r_idx] = 1.0

        constraints = self._base_assignment_constraints(n_vars, include_z=True, z_offset=z_offset, n_scenarios=n_s)
        extra_rows = []
        lbs = []
        ubs = []

        def add_extra(row_dict: dict[int, float], lb: float, ub: float) -> None:
            extra_rows.append(row_dict)
            lbs.append(lb)
            ubs.append(ub)

        if mode in {"cvar", "cvar95"}:
            total_w = float(self.weight.sum())
            for sid_pos in range(n_s):
                row = {theta_idx: -1.0, w_offset + sid_pos: -1.0}
                for fidx, weight in enumerate(self.weight):
                    row[z_offset + sid_pos * n_f + fidx] = -float(weight)
                add_extra(row, -np.inf, -total_w)
        elif mode == "minimax":
            total_w = float(self.weight.sum())
            for sid_pos in range(n_s):
                row = {r_idx: -1.0}
                for i, cand in self.candidates.iterrows():
                    row[int(i)] = row.get(int(i), 0.0) + float(cand["ferry_cost"])
                for fidx, weight in enumerate(self.weight):
                    row[z_offset + sid_pos * n_f + fidx] = -float(weight)
                add_extra(row, -np.inf, -total_w)
        elif mode == "regret":
            total_w = float(self.weight.sum())
            assert oracle_costs is not None
            for sid_pos, sid in enumerate(scenario_ids):
                row = {r_idx: -1.0}
                for i, cand in self.candidates.iterrows():
                    row[int(i)] = row.get(int(i), 0.0) + float(cand["ferry_cost"])
                for fidx, weight in enumerate(self.weight):
                    row[z_offset + sid_pos * n_f + fidx] = -float(weight)
                add_extra(row, -np.inf, float(oracle_costs[int(sid)] - total_w))

        if extra_rows:
            mat = lil_matrix((len(extra_rows), n_vars), dtype=float)
            for r, row in enumerate(extra_rows):
                for col, val in row.items():
                    mat[r, col] = val
            constraints.append(LinearConstraint(mat.tocsr(), np.array(lbs), np.array(ubs)))

        lb = np.zeros(n_vars)
        ub = np.ones(n_vars)
        if mode in {"cvar", "cvar95"}:
            ub[theta_idx] = np.inf
            ub[w_offset : w_offset + n_s] = np.inf
        elif mode in {"minimax", "regret"}:
            ub[r_idx] = np.inf
        integrality = np.zeros(n_vars)
        integrality[:n_c] = 1
        if not relax_recourse:
            integrality[n_c : n_c + n_z] = 1

        res = milp(
            c=c,
            integrality=integrality,
            bounds=Bounds(lb, ub),
            constraints=constraints,
            options={"time_limit": self.time_limit, "mip_rel_gap": 0.01},
        )
        selected = self._selected_from_x(res.x[:n_c] if res.x is not None else np.zeros(n_c))
        return SolveResult(
            method={
                "saa": "saa_extensive",
                "cvar": "cvar_extensive",
                "cvar95": "mean_cvar95_extensive",
                "minimax": "minimax_extensive",
                "regret": "regret_portfolio",
            }[mode]
            if method_name is None
            else method_name,
            status=self._status_name(res.success, res.message),
            objective=float(res.fun) if res.fun is not None else np.nan,
            solve_seconds=perf_counter() - start,
            selected=selected,
            extra={
                "binary_var_count": int(n_c if relax_recourse else n_c + n_z),
                "continuous_var_count": int((n_z if relax_recourse else 0) + n_vars - n_c - n_z),
                "scenario_count": int(n_s),
                "recourse_relaxed": bool(relax_recourse),
            },
        )

    def _solve_regret_envelope(
        self,
        oracle_costs: dict[int, float],
        threshold_pool: dict[tuple[int, str], set[float]] | None = None,
        method_name: str = "envelope_regret_portfolio",
        include_q_values: bool = False,
    ) -> SolveResult:
        start = perf_counter()
        n_c = len(self.candidates)
        scenario_ids = sorted(self.scenarios["scenario_id"].unique().tolist())
        n_s = len(scenario_ids)
        if n_c == 0 or len(self.flights) == 0 or n_s == 0:
            return self._empty("envelope_regret_portfolio", start)

        caps = _capacity_table(_copy_with_scenarios(self.instance, self.scenarios))
        block_ids = sorted(caps.columns.tolist())
        block_pos = {block: pos for pos, block in enumerate(block_ids)}
        q_offset = n_c
        n_q = n_s * len(block_ids)
        r_idx = q_offset + n_q
        n_vars = r_idx + 1

        c = np.zeros(n_vars)
        for i, row in self.candidates.iterrows():
            c[i] = float(row["ferry_cost"])
        if n_s > 0:
            c[q_offset : q_offset + n_q] = -0.05 / n_s
        c[r_idx] = 1.0

        constraints = self._base_assignment_constraints(n_vars, include_z=False, z_offset=0, n_scenarios=0)
        extra_rows = []
        lbs = []
        ubs = []

        def add_extra(row_dict: dict[int, float], lb: float, ub: float) -> None:
            extra_rows.append(row_dict)
            lbs.append(lb)
            ubs.append(ub)

        flight_rows_by_block = {
            block: self.flights[self.flights["block_id"].eq(block)].copy().reset_index(drop=True)
            for block in block_ids
        }
        candidate_rows_by_flight = {
            fid: self.candidates.index[self.candidates["flight_id"].eq(fid)].to_list()
            for fid in self.flight_index
        }
        envelope_cut_count = 0
        for sid_pos, sid in enumerate(scenario_ids):
            for block in block_ids:
                flights = flight_rows_by_block.get(block, self.flights.iloc[0:0])
                if flights.empty:
                    continue
                q_idx = q_offset + sid_pos * len(block_ids) + block_pos[block]
                capacity = int(caps.loc[sid, block])
                weights = flights["weight"].to_numpy(float)
                terms = launch_value_envelope_terms(weights.tolist(), capacity=capacity)
                if threshold_pool is not None:
                    active_thresholds = threshold_pool.get((int(sid), str(block)), {0.0})
                    terms = [term for term in terms if float(term[0]) in active_thresholds]
                for _, rhs, coefficients in terms:
                    row = {q_idx: 1.0}
                    for local_idx, flight in flights.iterrows():
                        coeff = float(coefficients[int(local_idx)])
                        if coeff <= 0.0:
                            continue
                        for cidx in candidate_rows_by_flight.get(str(flight["flight_id"]), []):
                            row[int(cidx)] = row.get(int(cidx), 0.0) - coeff
                    add_extra(row, -np.inf, rhs)
                    envelope_cut_count += 1

        total_w = float(self.weight.sum())
        for sid_pos, sid in enumerate(scenario_ids):
            row = {r_idx: -1.0}
            for i, cand in self.candidates.iterrows():
                row[int(i)] = row.get(int(i), 0.0) + float(cand["ferry_cost"])
            for block in block_ids:
                q_idx = q_offset + sid_pos * len(block_ids) + block_pos[block]
                row[q_idx] = row.get(q_idx, 0.0) - 1.0
            add_extra(row, -np.inf, float(oracle_costs[int(sid)] - total_w))

        if extra_rows:
            mat = lil_matrix((len(extra_rows), n_vars), dtype=float)
            for r, row in enumerate(extra_rows):
                for col, val in row.items():
                    mat[r, col] = val
            constraints.append(LinearConstraint(mat.tocsr(), np.array(lbs), np.array(ubs)))

        lb = np.zeros(n_vars)
        ub = np.ones(n_vars)
        for sid_pos, _ in enumerate(scenario_ids):
            for block in block_ids:
                q_idx = q_offset + sid_pos * len(block_ids) + block_pos[block]
                block_weight = float(flight_rows_by_block[block]["weight"].sum()) if block in flight_rows_by_block else 0.0
                ub[q_idx] = max(0.0, block_weight)
        ub[r_idx] = np.inf
        integrality = np.zeros(n_vars)
        integrality[:n_c] = 1

        res = milp(
            c=c,
            integrality=integrality,
            bounds=Bounds(lb, ub),
            constraints=constraints,
            options={"time_limit": self.time_limit, "mip_rel_gap": 0.01},
        )
        solution = res.x if res.x is not None else np.zeros(n_vars)
        selected = self._selected_from_x(solution[:n_c])
        extra = {
            "binary_var_count": int(n_c),
            "continuous_var_count": int(n_q + 1),
            "scenario_count": int(n_s),
            "scenario_block_count": int(n_q),
            "envelope_cut_count": int(envelope_cut_count),
            "regret_cut_count": int(n_s),
        }
        if include_q_values:
            q_values: dict[tuple[int, str], float] = {}
            for sid_pos, sid in enumerate(scenario_ids):
                for block in block_ids:
                    q_idx = q_offset + sid_pos * len(block_ids) + block_pos[block]
                    q_values[(int(sid), str(block))] = float(solution[q_idx])
            flight_assignment_values = {}
            for fid in self.flight_index:
                cidxs = self.candidates.index[self.candidates["flight_id"].eq(fid)].to_list()
                flight_assignment_values[str(fid)] = float(sum(solution[int(cidx)] for cidx in cidxs))
            extra["q_values"] = q_values
            extra["flight_assignment_values"] = flight_assignment_values
        return SolveResult(
            method=method_name,
            status=self._status_name(res.success, res.message),
            objective=float(res.fun) if res.fun is not None else np.nan,
            solve_seconds=perf_counter() - start,
            selected=selected,
            extra=extra,
        )

    def _selected_from_x(self, x: np.ndarray) -> pd.DataFrame:
        if len(x) == 0:
            return self.candidates.iloc[0:0].copy()
        selected = self.candidates.loc[np.asarray(x) > 0.5].copy()
        selected = selected.merge(
            self.flights[["flight_id", "Origin", "Dest", "block_id", "CRSDepTime", "weight"]],
            on="flight_id",
            how="left",
            suffixes=("", "_flight"),
        )
        return selected.reset_index(drop=True)

    @staticmethod
    def _status_name(success: bool, message: str) -> str:
        return "optimal_or_feasible" if success else str(message)[:80]

    @staticmethod
    def _empty(method: str, start: float) -> SolveResult:
        return SolveResult(method, "empty", np.nan, perf_counter() - start, pd.DataFrame(), {})


def evaluate_solution(instance: LaunchInstance, result: SolveResult, oracle_costs: dict[int, float] | None = None) -> dict[str, float | str]:
    flights = instance.flights.set_index("flight_id")
    selected_ids = set(result.selected["flight_id"].tolist()) if not result.selected.empty else set()
    selected = flights.loc[list(selected_ids)] if selected_ids else flights.iloc[0:0]
    ferry_cost = float(result.selected["ferry_cost"].sum()) if not result.selected.empty else 0.0
    scenario_costs: list[float] = []
    scenario_coverages: list[float] = []
    scenario_regrets: list[float] = []

    for sid, caps in instance.scenarios.groupby("scenario_id"):
        launched_weight = 0.0
        for block, cap_row in caps.groupby("block_id"):
            cap = int(cap_row["capacity"].iloc[0])
            block_flights = selected[selected["block_id"].eq(block)].sort_values("weight", ascending=False)
            if cap > 0 and not block_flights.empty:
                launched_weight += float(block_flights.head(cap)["weight"].sum())
        cost = ferry_cost + instance.total_weight - launched_weight
        scenario_costs.append(cost)
        scenario_coverages.append(launched_weight / instance.total_weight if instance.total_weight > 0 else 0.0)
        if oracle_costs is not None and int(sid) in oracle_costs:
            scenario_regrets.append(cost - float(oracle_costs[int(sid)]))

    if scenario_costs:
        sorted_costs = np.sort(np.array(scenario_costs))
        tail_n = max(1, ceil(0.05 * len(sorted_costs)))
        cvar95 = float(sorted_costs[-tail_n:].mean())
        expected_cost = float(np.mean(scenario_costs))
        worst_cost = float(np.max(scenario_costs))
    else:
        cvar95 = expected_cost = worst_cost = np.nan

    assigned_weight = float(flights.loc[list(selected_ids), "weight"].sum()) if selected_ids else 0.0
    return {
        "date": instance.date,
        "carrier": instance.carrier,
        "method": result.method,
        "status": result.status,
        "objective": result.objective,
        "solve_seconds": result.solve_seconds,
        "flight_count": int(len(instance.flights)),
        "inventory_count": int(instance.inventory["aircraft_count"].sum()) if not instance.inventory.empty else 0,
        "assigned_flights": int(len(selected_ids)),
        "assigned_weight_share": assigned_weight / instance.total_weight if instance.total_weight > 0 else 0.0,
        "ferry_moves": int((result.selected["ferry_cost"] > 0).sum()) if not result.selected.empty else 0,
        "ferry_cost": ferry_cost,
        "expected_cost": expected_cost,
        "cvar95_cost": cvar95,
        "worst_cost": worst_cost,
        "mean_launch_weight_share": float(np.mean(scenario_coverages)) if scenario_coverages else np.nan,
        "min_launch_weight_share": float(np.min(scenario_coverages)) if scenario_coverages else np.nan,
        "max_regret": float(np.max(scenario_regrets)) if scenario_regrets else np.nan,
        "mean_regret": float(np.mean(scenario_regrets)) if scenario_regrets else np.nan,
    }


def compute_oracle_costs(instance: LaunchInstance, time_limit: float = 10.0) -> dict[int, float]:
    costs: dict[int, float] = {}
    for sid, scenario in instance.scenarios.groupby("scenario_id"):
        single = _copy_with_scenarios(instance, scenario.copy())
        res = LaunchOptimizer(single, time_limit=time_limit).solve("saa_extensive")
        metrics = evaluate_solution(single, res)
        costs[int(sid)] = float(metrics["expected_cost"])
    return costs


def certificate_candidate_scores(instance: LaunchInstance) -> pd.DataFrame:
    candidates = instance.candidates.reset_index(drop=True).copy()
    flights = instance.flights.reset_index(drop=True).copy()
    if candidates.empty or flights.empty:
        return pd.DataFrame(columns=["candidate_id", "candidate_row", "certificate_score"])

    caps = _capacity_table(instance)
    flights["block_rank"] = (
        flights.sort_values(["block_id", "weight"], ascending=[True, False])
        .groupby("block_id")
        .cumcount()
        + 1
    )
    block_demand = flights.groupby("block_id").size().to_dict()
    origin_demand = flights.groupby("Origin").size().to_dict()
    origin_inventory = dict(zip(instance.inventory["overnight_airport"], instance.inventory["aircraft_count"]))
    block_pressure: dict[str, float] = {}
    block_survival: dict[tuple[str, int], float] = {}

    for block, demand in block_demand.items():
        cap_values = caps[block].to_numpy(float) if block in caps else np.zeros(len(caps), dtype=float)
        shortage = np.maximum(0.0, demand - cap_values) / max(1, demand)
        q10 = float(np.quantile(cap_values, 0.10)) if len(cap_values) else 0.0
        tail_shortage = max(0.0, demand - q10) / max(1, demand)
        block_pressure[block] = 1.0 + float(shortage.mean()) + 0.75 * tail_shortage
        for rank in range(1, demand + 1):
            block_survival[(block, rank)] = float(np.mean(cap_values >= rank)) if len(cap_values) else 0.0

    flight_features = flights.set_index("flight_id")[["Origin", "block_id", "block_rank", "weight"]].to_dict("index")
    demand_total = max(1, sum(origin_demand.values()))
    origin_deficit = {
        airport: max(0.0, origin_demand.get(airport, 0) - origin_inventory.get(airport, 0))
        / max(1, origin_demand.get(airport, 0))
        for airport in set(origin_demand) | set(origin_inventory)
    }
    source_surplus = {
        airport: max(0.0, origin_inventory.get(airport, 0) - origin_demand.get(airport, 0)) / max(1, demand_total)
        for airport in set(origin_demand) | set(origin_inventory)
    }

    rows: list[dict[str, float | int | str]] = []
    for idx, cand in candidates.iterrows():
        feat = flight_features[cand["flight_id"]]
        block = str(feat["block_id"])
        rank = int(feat["block_rank"])
        origin = str(feat["Origin"])
        source = str(cand["source_airport"])
        weight = float(feat["weight"])
        pressure = block_pressure.get(block, 1.0)
        survival = block_survival.get((block, rank), 0.0)
        value = weight * survival * pressure
        if source != origin:
            value += 0.35 * weight * origin_deficit.get(origin, 0.0) * pressure
            value += 0.15 * weight * source_surplus.get(source, 0.0)
            value -= 0.25 * weight * origin_deficit.get(source, 0.0)
        score = value - float(cand["ferry_cost"]) * (1.0 + 0.25 * pressure)
        rows.append(
            {
                "candidate_id": str(cand["candidate_id"]),
                "candidate_row": int(idx),
                "flight_id": str(cand["flight_id"]),
                "source_airport": source,
                "certificate_score": float(score),
            }
        )
    return pd.DataFrame(rows)
