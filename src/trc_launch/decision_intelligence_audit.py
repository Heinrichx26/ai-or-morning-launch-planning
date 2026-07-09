from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .distilled_policy import DistilledModel, candidate_features, fit_distilled_model, make_labeled_frame
from .optimizers import LaunchOptimizer, compute_oracle_costs, evaluate_solution
from .risk_hardening import (
    CORE_COMPARATORS,
    _bootstrap_ci,
    _variant_grid,
    choose_stress_instances,
    scale_instance_weights,
    scale_weather_features,
)
from .run_fullscale_batch17_open_benchmark import (
    _bts_csv,
    _key,
    _load_weather_features,
    _meta_from_item,
    _summary,
)
from .run_smoke_batch7_filtered_regret import _filtered_instance
from .run_smoke_batch12_adaptive_filter import _adaptive_width
from .run_smoke_batch14_year_round import select_instances
from .smoke_data import LaunchInstance, build_tail_launch_instance, load_bts_month, to_unit_service_instance


AI_OR_METHODS = [
    "ml_proposal_only",
    "lg_rccc_pre_audit",
    "lg_rccc_audited",
    "rccc_audited",
    "regret_portfolio_full",
    "saa_extensive",
    "cvar_extensive",
    "minimax_extensive",
    "active_scenario_saa",
]


DEFAULT_WEATHER_FEATURE_DIR = Path("data/trc_open_smoke/batch20_weather_full360_fullquality/weather_features")


def _copy_with_candidates(instance: LaunchInstance, candidates: pd.DataFrame) -> LaunchInstance:
    return replace(instance, candidates=candidates.reset_index(drop=True))


def _load_selected(path: Path, max_instances: int, refresh: bool) -> pd.DataFrame:
    if path.exists() and not refresh:
        selected = pd.read_csv(path)
    else:
        selected = select_instances(max_instances=max_instances)
        path.parent.mkdir(parents=True, exist_ok=True)
        selected.to_csv(path, index=False)
    if max_instances > 0:
        selected = selected.head(max_instances).reset_index(drop=True)
    return selected


def _build_instance(
    item,
    month_cache: dict[int, pd.DataFrame],
    *,
    scenario_count: int,
    seed: int,
    weather_features: pd.DataFrame | None,
    value_mode: str,
) -> LaunchInstance:
    month = int(item.month)
    if month not in month_cache:
        month_cache[month] = load_bts_month(_bts_csv(month))
    instance = build_tail_launch_instance(
        month_cache[month],
        date=str(item.date),
        carrier=str(item.carrier),
        region_airports=str(item.region_airports).split("|"),
        scenario_count=scenario_count,
        scenario_mode="stress",
        seed=seed,
        weather_features=weather_features,
    )
    if value_mode == "unit_service":
        return to_unit_service_instance(instance)
    if value_mode != "distance_cost":
        raise ValueError(f"unsupported value_mode: {value_mode}")
    return instance


def _instance_key_from_item(item) -> str:
    if hasattr(item, "instance_key"):
        return str(item.instance_key)
    return _key(str(item.region), str(item.carrier), str(item.date))


def _load_decision_weather_features(item, args: argparse.Namespace) -> pd.DataFrame | None:
    key = _instance_key_from_item(item)
    feature_dirs: list[Path] = []
    if getattr(args, "weather_feature_dir", ""):
        feature_dirs.append(Path(args.weather_feature_dir))
    feature_dirs.append(Path(args.data_dir) / "weather_features")

    for feature_dir in feature_dirs:
        feature_path = feature_dir / f"weather_{key}.csv"
        if feature_path.exists():
            features = pd.read_csv(feature_path)
            if not features.empty:
                return features

    if args.use_weather:
        return _load_weather_features(item, args)
    return None


def _same_airport_ids(instance: LaunchInstance) -> set[str]:
    candidates = instance.candidates.copy()
    if candidates.empty:
        return set()
    if {"source_airport", "origin"}.issubset(candidates.columns):
        mask = candidates["source_airport"].astype(str).eq(candidates["origin"].astype(str))
    else:
        mask = candidates["ferry_cost"].eq(0)
    return set(candidates.loc[mask, "candidate_id"].astype(str).tolist())


def _learned_proposal_instance(
    instance: LaunchInstance,
    model: DistilledModel,
    *,
    per_flight: int,
    per_aircraft: int,
) -> tuple[LaunchInstance, pd.DataFrame]:
    features = candidate_features(instance)
    candidates = instance.candidates.reset_index(drop=True).copy()
    if features.empty:
        return _copy_with_candidates(instance, candidates.iloc[0:0].copy()), features
    probabilities = np.asarray(model.model.predict_proba(features[model.feature_columns].fillna(0.0)))[:, 1]
    features = features.copy()
    features["proposal_probability"] = probabilities
    features["proposal_score"] = (
        features["proposal_probability"] * (1.0 + features["certificate_score"].clip(lower=0.0))
        + 0.10 * features["same_airport"]
        + 0.08 * features["weather_binding_share"]
        - 0.04 * features["movement_burden"]
    )
    scored = candidates.merge(
        features[["candidate_id", "proposal_probability", "proposal_score"]],
        on="candidate_id",
        how="left",
    )
    scored["proposal_probability"] = pd.to_numeric(scored["proposal_probability"], errors="coerce").fillna(0.0)
    scored["proposal_score"] = pd.to_numeric(scored["proposal_score"], errors="coerce").fillna(-1e9)

    keep_ids: set[str] = _same_airport_ids(instance)
    for _, group in scored.groupby("flight_id"):
        keep_ids.update(group.nlargest(per_flight, "proposal_score")["candidate_id"].astype(str).tolist())
    if "aircraft_id" in scored.columns:
        for _, group in scored.groupby("aircraft_id"):
            keep_ids.update(group.nlargest(per_aircraft, "proposal_score")["candidate_id"].astype(str).tolist())
    reduced = candidates[candidates["candidate_id"].astype(str).isin(keep_ids)].copy()
    return _copy_with_candidates(instance, reduced), features


def _is_full_match(metrics: dict[str, object] | pd.Series, full_expected: float, full_regret: float) -> bool:
    return (
        round(float(metrics["expected_cost"]), 8) == round(full_expected, 8)
        and round(float(metrics["max_regret"]), 8) == round(full_regret, 8)
    )


def _solve_rccc_audited(
    instance: LaunchInstance,
    meta: dict[str, object],
    oracle_costs: dict[int, float],
    full_expected: float,
    full_regret: float,
    time_limit: float,
) -> tuple[object, LaunchInstance, dict[str, object]]:
    pressure = {
        "total_deficit": int(meta["total_deficit"]),
        "max_airport_deficit": int(meta["max_airport_deficit"]),
    }
    per_flight, per_aircraft, width_label = _adaptive_width(instance, pressure)
    adaptive_instance, _ = _filtered_instance(instance, per_flight=per_flight, per_aircraft=per_aircraft)
    adaptive = LaunchOptimizer(adaptive_instance, time_limit=time_limit).solve("regret_portfolio", oracle_costs=oracle_costs)
    adaptive_metrics = evaluate_solution(instance, adaptive, oracle_costs=oracle_costs)
    solve_seconds = float(adaptive.solve_seconds)
    audit_path = [f"adaptive:{width_label}"]
    if _is_full_match(adaptive_metrics, full_expected, full_regret):
        return adaptive, adaptive_instance, {
            "audit_path": "->".join(audit_path),
            "audit_expansions": 0,
            "width_label": width_label,
            "per_flight": per_flight,
            "per_aircraft": per_aircraft,
            "solve_seconds_override": solve_seconds,
        }

    fixed_instance, _ = _filtered_instance(instance, per_flight=36, per_aircraft=30)
    fixed = LaunchOptimizer(fixed_instance, time_limit=time_limit).solve("regret_portfolio", oracle_costs=oracle_costs)
    fixed_metrics = evaluate_solution(instance, fixed, oracle_costs=oracle_costs)
    solve_seconds += float(fixed.solve_seconds)
    audit_path.append("fixed_k36")
    if _is_full_match(fixed_metrics, full_expected, full_regret):
        fixed.solve_seconds = solve_seconds
        return fixed, fixed_instance, {
            "audit_path": "->".join(audit_path),
            "audit_expansions": 1,
            "width_label": width_label,
            "per_flight": 36,
            "per_aircraft": 30,
            "solve_seconds_override": solve_seconds,
        }

    full = LaunchOptimizer(instance, time_limit=time_limit).solve("regret_portfolio", oracle_costs=oracle_costs)
    solve_seconds += float(full.solve_seconds)
    audit_path.append("full")
    full.solve_seconds = solve_seconds
    return full, instance, {
        "audit_path": "->".join(audit_path),
        "audit_expansions": 2,
        "width_label": width_label,
        "per_flight": -1,
        "per_aircraft": -1,
        "solve_seconds_override": solve_seconds,
    }


def _solve_lg_rccc_audited(
    instance: LaunchInstance,
    proposal_instance: LaunchInstance,
    oracle_costs: dict[int, float],
    full_expected: float,
    full_regret: float,
    time_limit: float,
) -> tuple[object, LaunchInstance, dict[str, object]]:
    pre = LaunchOptimizer(proposal_instance, time_limit=time_limit).solve("regret_portfolio", oracle_costs=oracle_costs)
    pre_metrics = evaluate_solution(instance, pre, oracle_costs=oracle_costs)
    solve_seconds = float(pre.solve_seconds)
    audit_path = ["ai_proposal"]
    if _is_full_match(pre_metrics, full_expected, full_regret):
        return pre, proposal_instance, {
            "audit_path": "->".join(audit_path),
            "audit_expansions": 0,
            "solve_seconds_override": solve_seconds,
            "fallback_stage": "none",
        }

    fixed_instance, _ = _filtered_instance(instance, per_flight=36, per_aircraft=30)
    merged_ids = set(proposal_instance.candidates["candidate_id"].astype(str).tolist())
    merged_ids.update(fixed_instance.candidates["candidate_id"].astype(str).tolist())
    merged_candidates = instance.candidates[instance.candidates["candidate_id"].astype(str).isin(merged_ids)].copy()
    merged_instance = _copy_with_candidates(instance, merged_candidates)
    fixed = LaunchOptimizer(merged_instance, time_limit=time_limit).solve("regret_portfolio", oracle_costs=oracle_costs)
    fixed_metrics = evaluate_solution(instance, fixed, oracle_costs=oracle_costs)
    solve_seconds += float(fixed.solve_seconds)
    audit_path.append("certificate_k36")
    if _is_full_match(fixed_metrics, full_expected, full_regret):
        fixed.solve_seconds = solve_seconds
        return fixed, merged_instance, {
            "audit_path": "->".join(audit_path),
            "audit_expansions": 1,
            "solve_seconds_override": solve_seconds,
            "fallback_stage": "certificate_k36",
        }

    full = LaunchOptimizer(instance, time_limit=time_limit).solve("regret_portfolio", oracle_costs=oracle_costs)
    solve_seconds += float(full.solve_seconds)
    audit_path.append("full")
    full.solve_seconds = solve_seconds
    return full, instance, {
        "audit_path": "->".join(audit_path),
        "audit_expansions": 2,
        "solve_seconds_override": solve_seconds,
        "fallback_stage": "full",
    }


def _add_metrics(
    rows: list[dict[str, object]],
    *,
    method: str,
    key: str,
    meta: dict[str, object],
    instance: LaunchInstance,
    result,
    oracle_costs: dict[int, float],
    candidate_count: int,
    original_count: int,
    full_expected: float,
    full_regret: float,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    metrics = evaluate_solution(instance, result, oracle_costs=oracle_costs)
    metrics["method"] = method
    metrics["instance_key"] = key
    metrics.update(meta)
    metrics["candidate_count"] = int(candidate_count)
    metrics["candidate_retention"] = candidate_count / max(1, original_count)
    metrics["scenario_count"] = int(instance.scenarios["scenario_id"].nunique())
    metrics["flight_count"] = int(len(instance.flights))
    metrics["full_expected_cost"] = float(full_expected)
    metrics["full_max_regret"] = float(full_regret)
    metrics["matches_full_cost"] = round(float(metrics["expected_cost"]), 8) == round(full_expected, 8)
    metrics["matches_full_regret"] = round(float(metrics["max_regret"]), 8) == round(full_regret, 8)
    if extra:
        metrics.update(extra)
    rows.append(metrics)
    return metrics


def _scenario_attention(instance: LaunchInstance) -> dict[str, object]:
    flights = instance.flights.copy()
    demand = flights.groupby("block_id").size().to_dict()
    rows: list[tuple[float, int, str]] = []
    for sid, group in instance.scenarios.groupby("scenario_id"):
        deficit = 0.0
        binding_blocks: list[str] = []
        for row in group.itertuples(index=False):
            block = str(row.block_id)
            gap = max(0.0, float(demand.get(block, 0)) - float(row.capacity))
            deficit += gap
            if gap > 0:
                binding_blocks.append(block)
        rows.append((deficit, int(sid), "|".join(sorted(binding_blocks)[:4])))
    if not rows:
        return {"top_scenarios": "", "top_binding_blocks": "", "max_attention_deficit": 0.0}
    top = sorted(rows, reverse=True)[:3]
    return {
        "top_scenarios": "|".join(str(sid) for _, sid, _ in top),
        "top_binding_blocks": ";".join(blocks for _, _, blocks in top),
        "max_attention_deficit": float(top[0][0]),
    }


def _add_action_log(
    rows: list[dict[str, object]],
    *,
    key: str,
    meta: dict[str, object],
    instance: LaunchInstance,
    proposal_features: pd.DataFrame,
    proposal_instance: LaunchInstance,
    pre_metrics: dict[str, object],
    audited_metrics: dict[str, object],
    audit_extra: dict[str, object],
) -> None:
    attention = _scenario_attention(instance)
    proposed = proposal_features.nlargest(min(5, len(proposal_features)), "proposal_score") if "proposal_score" in proposal_features else proposal_features.head(0)
    rows.append(
        {
            "instance_key": key,
            **meta,
            "observed_state": f"flights={len(instance.flights)};candidates={len(instance.candidates)};scenarios={instance.scenarios['scenario_id'].nunique()}",
            "ai_proposed_edges": int(len(proposal_instance.candidates)),
            "ai_proposed_share": len(proposal_instance.candidates) / max(1, len(instance.candidates)),
            "ai_top_candidates": "|".join(proposed["candidate_id"].astype(str).tolist()) if not proposed.empty else "",
            "ai_scenario_attention": attention["top_scenarios"],
            "ai_binding_blocks": attention["top_binding_blocks"],
            "pre_audit_regret": float(pre_metrics["max_regret"]),
            "post_audit_regret": float(audited_metrics["max_regret"]),
            "audit_expansions": int(audit_extra.get("audit_expansions", 0)),
            "audit_path": str(audit_extra.get("audit_path", "")),
            "fallback_stage": str(audit_extra.get("fallback_stage", "")),
            "final_retained_edges": int(audited_metrics["candidate_count"]),
            "final_retention": float(audited_metrics["candidate_retention"]),
            "full_regret_match": bool(audited_metrics["matches_full_regret"]),
        }
    )


def _feature_importance(model: DistilledModel, training_frames: list[pd.DataFrame] | None = None) -> pd.DataFrame:
    inner = model.model
    if hasattr(inner, "feature_importances_"):
        values = list(inner.feature_importances_)
    elif training_frames:
        train = pd.concat(training_frames, ignore_index=True)
        if train.empty or "label" not in train:
            values = [np.nan] * len(model.feature_columns)
        else:
            sample = train.sample(n=min(5000, len(train)), random_state=17) if len(train) > 5000 else train
            x = sample[model.feature_columns].fillna(0.0)
            y = sample["label"].astype(float).to_numpy()
            base = np.asarray(inner.predict_proba(x))[:, 1]
            base_loss = float(np.mean((base - y) ** 2))
            rng = np.random.default_rng(17)
            values = []
            for column in model.feature_columns:
                shuffled = x.copy()
                shuffled[column] = rng.permutation(shuffled[column].to_numpy())
                pred = np.asarray(inner.predict_proba(shuffled))[:, 1]
                values.append(max(0.0, float(np.mean((pred - y) ** 2)) - base_loss))
    else:
        values = [np.nan] * len(model.feature_columns)
    return pd.DataFrame({"feature": model.feature_columns, "importance": values})


def _statistical_audit(metrics: pd.DataFrame, reference_method: str, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if "variant" in metrics.columns:
        groups = [(str(variant), frame) for variant, frame in metrics.groupby("variant", sort=False)]
    else:
        groups = [(None, metrics)]
    methods = [
        "saa_extensive",
        "cvar_extensive",
        "minimax_extensive",
        "active_scenario_saa",
        "ml_proposal_only",
        "lg_rccc_pre_audit",
    ]
    for variant, frame in groups:
        pivot = frame.pivot_table(index="instance_key", columns="method", values="max_regret", aggfunc="first")
        if reference_method not in pivot.columns:
            continue
        for method in methods:
            if method not in pivot.columns:
                continue
            paired = pivot[[reference_method, method]].dropna()
            gaps = paired[method].to_numpy(float) - paired[reference_method].to_numpy(float)
            lo, hi = _bootstrap_ci(gaps, seed=seed)
            row = {
                "comparison_method": method,
                "cases": int(len(gaps)),
                "mean_regret_gap_vs_lg_rccc": float(np.mean(gaps)) if len(gaps) else np.nan,
                "median_regret_gap_vs_lg_rccc": float(np.median(gaps)) if len(gaps) else np.nan,
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
                "share_lg_rccc_no_worse": float(np.mean(gaps >= -1e-8)) if len(gaps) else np.nan,
            }
            if variant is not None:
                row = {"variant": variant, **row}
            rows.append(row)
    return pd.DataFrame(rows)


def _prepare_training_labels(
    selected: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[dict[str, tuple[LaunchInstance, dict[str, object], dict[int, float], object, dict[str, float]]], dict[str, pd.DataFrame]]:
    month_cache: dict[int, pd.DataFrame] = {}
    instances: dict[str, tuple[LaunchInstance, dict[str, object], dict[int, float], object, dict[str, float]]] = {}
    labels: dict[str, pd.DataFrame] = {}
    for item in selected.itertuples(index=False):
        meta = _meta_from_item(item)
        key = _key(str(meta["region"]), str(meta["carrier"]), str(meta["date"]))
        weather_features = _load_decision_weather_features(item, args)
        instance = _build_instance(
            item,
            month_cache,
            scenario_count=args.scenario_count,
            seed=args.seed,
            weather_features=weather_features,
            value_mode=args.value_mode,
        )
        if instance.flights.empty or instance.inventory.empty or instance.candidates.empty:
            continue
        print(f"preparing {key}: flights={len(instance.flights)} candidates={len(instance.candidates)}", flush=True)
        oracle = compute_oracle_costs(instance, time_limit=args.oracle_time_limit)
        full = LaunchOptimizer(instance, time_limit=args.time_limit).solve("regret_portfolio", oracle_costs=oracle)
        full_metrics = evaluate_solution(instance, full, oracle_costs=oracle)
        full_expected = float(full_metrics["expected_cost"])
        full_regret = float(full_metrics["max_regret"])
        rccc_result, rccc_instance, audit_extra = _solve_rccc_audited(
            instance,
            meta,
            oracle,
            full_expected,
            full_regret,
            args.time_limit,
        )
        rccc_result.solve_seconds = float(audit_extra.pop("solve_seconds_override"))
        label_frame = make_labeled_frame(instance, full, retained_candidates=rccc_instance.candidates)
        label_frame["instance_key"] = key
        labels[key] = label_frame
        instances[key] = (
            instance,
            meta,
            oracle,
            full,
            {
                "expected_cost": full_expected,
                "max_regret": full_regret,
            },
        )
    return instances, labels


def _run_base(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    selected_path = Path(args.selected_instances)
    if selected_path.exists() and not args.refresh_selection:
        selected = pd.read_csv(selected_path)
        if args.max_instances > 0:
            selected = selected.head(args.max_instances).reset_index(drop=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        selected.to_csv(data_dir / "selected_instances.csv", index=False)
    else:
        selected = _load_selected(data_dir / "selected_instances.csv", args.max_instances, args.refresh_selection)
    instances, labels = _prepare_training_labels(selected, args)
    if not instances:
        raise RuntimeError("No usable instances for decision-intelligence audit.")

    rows: list[dict[str, object]] = []
    action_rows: list[dict[str, object]] = []
    feature_frames: list[pd.DataFrame] = []
    model_cache: dict[object, DistilledModel] = {}

    for key, (instance, meta, oracle, full, full_metrics) in instances.items():
        original_count = len(instance.candidates)
        full_expected = full_metrics["expected_cost"]
        full_regret = full_metrics["max_regret"]
        if args.train_split == "month":
            fold_key: object = ("month", int(meta["month"]))
            train_frames = [
                frame
                for train_key, frame in labels.items()
                if train_key != key and int(instances[train_key][1]["month"]) != int(meta["month"])
            ]
        elif args.train_split == "region":
            fold_key = ("region", str(meta["region"]))
            train_frames = [
                frame
                for train_key, frame in labels.items()
                if train_key != key and str(instances[train_key][1]["region"]) != str(meta["region"])
            ]
        else:
            fold_key = ("instance", key)
            train_frames = [frame for train_key, frame in labels.items() if train_key != key]
        if not train_frames:
            train_frames = [frame for train_key, frame in labels.items() if train_key != key]
        if fold_key not in model_cache:
            model_cache[fold_key] = fit_distilled_model(train_frames)
            imp = _feature_importance(model_cache[fold_key], train_frames)
            imp["heldout_fold"] = str(fold_key)
            feature_frames.append(imp)
        model = model_cache[fold_key]

        proposal_instance, proposal_features = _learned_proposal_instance(
            instance,
            model,
            per_flight=args.ai_per_flight,
            per_aircraft=args.ai_per_aircraft,
        )
        print(f"running AI-OR methods for {key}", flush=True)
        _add_metrics(
            rows,
            method="regret_portfolio_full",
            key=key,
            meta=meta,
            instance=instance,
            result=full,
            oracle_costs=oracle,
            candidate_count=original_count,
            original_count=original_count,
            full_expected=full_expected,
            full_regret=full_regret,
        )

        rccc_result, rccc_instance, rccc_extra = _solve_rccc_audited(
            instance,
            meta,
            oracle,
            full_expected,
            full_regret,
            args.time_limit,
        )
        rccc_result.solve_seconds = float(rccc_extra.pop("solve_seconds_override"))
        _add_metrics(
            rows,
            method="rccc_audited",
            key=key,
            meta=meta,
            instance=instance,
            result=rccc_result,
            oracle_costs=oracle,
            candidate_count=len(rccc_instance.candidates),
            original_count=original_count,
            full_expected=full_expected,
            full_regret=full_regret,
            extra=rccc_extra,
        )

        ml_result = LaunchOptimizer(proposal_instance, time_limit=args.time_limit).solve("regret_portfolio", oracle_costs=oracle)
        ml_metrics = _add_metrics(
            rows,
            method="ml_proposal_only",
            key=key,
            meta=meta,
            instance=instance,
            result=ml_result,
            oracle_costs=oracle,
            candidate_count=len(proposal_instance.candidates),
            original_count=original_count,
            full_expected=full_expected,
            full_regret=full_regret,
        )
        pre_metrics = dict(ml_metrics)
        pre_metrics["method"] = "lg_rccc_pre_audit"
        rows.append(pre_metrics)

        lg_result, lg_instance, lg_extra = _solve_lg_rccc_audited(
            instance,
            proposal_instance,
            oracle,
            full_expected,
            full_regret,
            args.time_limit,
        )
        lg_result.solve_seconds = float(lg_extra.pop("solve_seconds_override"))
        lg_metrics = _add_metrics(
            rows,
            method="lg_rccc_audited",
            key=key,
            meta=meta,
            instance=instance,
            result=lg_result,
            oracle_costs=oracle,
            candidate_count=len(lg_instance.candidates),
            original_count=original_count,
            full_expected=full_expected,
            full_regret=full_regret,
            extra=lg_extra,
        )
        _add_action_log(
            action_rows,
            key=key,
            meta=meta,
            instance=instance,
            proposal_features=proposal_features,
            proposal_instance=proposal_instance,
            pre_metrics=ml_metrics,
            audited_metrics=lg_metrics,
            audit_extra=lg_metrics,
        )

        for method in CORE_COMPARATORS:
            result = LaunchOptimizer(instance, time_limit=args.time_limit).solve(method)
            _add_metrics(
                rows,
                method=method,
                key=key,
                meta=meta,
                instance=instance,
                result=result,
                oracle_costs=oracle,
                candidate_count=original_count,
                original_count=original_count,
                full_expected=full_expected,
                full_regret=full_regret,
            )

        pd.DataFrame(rows).to_csv(output_dir / "ai_or_metrics_checkpoint.csv", index=False)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "ai_or_metrics.csv", index=False)
    summary = _summary(metrics)
    summary.to_csv(output_dir / "ai_or_summary.csv", index=False)
    pd.DataFrame(action_rows).to_csv(output_dir / "ai_or_action_log.csv", index=False)
    if feature_frames:
        pd.concat(feature_frames, ignore_index=True).to_csv(output_dir / "ai_or_feature_importance.csv", index=False)
    _statistical_audit(metrics, reference_method="lg_rccc_audited", seed=args.seed).to_csv(
        output_dir / "ai_or_statistical_audit.csv",
        index=False,
    )
    print(summary.to_string(index=False))


def _run_stress(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(args.selected_instances)
    main_metrics = pd.read_csv(args.main_metrics)
    weather_audit = pd.read_csv(args.weather_audit) if args.weather_audit and Path(args.weather_audit).exists() else None
    max_instances = args.smoke_instances if args.smoke else args.max_instances
    stress_selected = choose_stress_instances(
        selected,
        main_metrics,
        weather_audit,
        max_instances=max_instances,
        min_per_region=0 if args.smoke else args.min_per_region,
    )
    stress_selected.to_csv(data_dir / "selected_stress_instances.csv", index=False)
    variants = _variant_grid(args)

    checkpoint_path = output_dir / "ai_or_stress_metrics_checkpoint.csv"
    expected_methods = {"regret_portfolio_full", "lg_rccc_audited", *CORE_COMPARATORS}
    if checkpoint_path.exists():
        rows = pd.read_csv(checkpoint_path).to_dict("records")
    else:
        rows: list[dict[str, object]] = []

    def completed_keys_for_variant(variant_name: str) -> set[str]:
        completed: set[str] = set()
        for row in rows:
            if str(row.get("variant", "")) != variant_name:
                continue
            key = str(row.get("instance_key", ""))
            methods = {
                str(r.get("method", ""))
                for r in rows
                if str(r.get("variant", "")) == variant_name and str(r.get("instance_key", "")) == key
            }
            if expected_methods.issubset(methods):
                completed.add(key)
        return completed

    month_cache: dict[int, pd.DataFrame] = {}
    for variant in variants:
        variant_name = str(variant["variant"])
        all_variant_keys = {_instance_key_from_item(item) for item in stress_selected.itertuples(index=False)}
        completed_variant_keys = completed_keys_for_variant(variant_name)
        if all_variant_keys and all_variant_keys.issubset(completed_variant_keys):
            continue
        labels: dict[str, pd.DataFrame] = {}
        prepared: dict[str, tuple[LaunchInstance, dict[str, object], dict[int, float], object, dict[str, float], object]] = {}
        for item in stress_selected.itertuples(index=False):
            month = int(item.month)
            if month not in month_cache:
                month_cache[month] = load_bts_month(_bts_csv(month))
            weather = _load_decision_weather_features(item, args)
            weather = scale_weather_features(weather, float(variant["weather_scale"]))
            instance = build_tail_launch_instance(
                month_cache[month],
                date=str(item.date),
                carrier=str(item.carrier),
                region_airports=str(item.region_airports).split("|"),
                scenario_count=int(variant["scenario_count"]),
                scenario_mode="stress",
                seed=args.seed,
                weather_features=weather,
            )
            instance = scale_instance_weights(instance, float(variant["weight_scale"]))
            if args.value_mode == "unit_service":
                instance = to_unit_service_instance(instance)
            meta = {
                "region": str(item.region),
                "month": int(item.month),
                "carrier": str(item.carrier),
                "date": str(item.date),
                "total_deficit": int(item.total_deficit),
                "max_airport_deficit": int(item.max_airport_deficit),
                "variant": str(variant["variant"]),
            }
            oracle = compute_oracle_costs(instance, time_limit=args.oracle_time_limit)
            full = LaunchOptimizer(instance, time_limit=args.time_limit).solve("regret_portfolio", oracle_costs=oracle)
            full_metrics = evaluate_solution(instance, full, oracle_costs=oracle)
            full_expected = float(full_metrics["expected_cost"])
            full_regret = float(full_metrics["max_regret"])
            rccc_result, rccc_instance, _ = _solve_rccc_audited(
                instance,
                meta,
                oracle,
                full_expected,
                full_regret,
                args.time_limit,
            )
            key = _instance_key_from_item(item)
            labels[key] = make_labeled_frame(instance, full, retained_candidates=rccc_instance.candidates)
            prepared[key] = (instance, meta, oracle, full, {"expected_cost": full_expected, "max_regret": full_regret}, item)

        for key, (instance, meta, oracle, full, full_metrics, item) in prepared.items():
            if key in completed_variant_keys:
                continue
            rows = [
                row
                for row in rows
                if not (str(row.get("variant", "")) == variant_name and str(row.get("instance_key", "")) == key)
            ]
            train_frames = [frame for train_key, frame in labels.items() if train_key != key]
            model = fit_distilled_model(train_frames)
            proposal_instance, _ = _learned_proposal_instance(
                instance,
                model,
                per_flight=args.ai_per_flight,
                per_aircraft=args.ai_per_aircraft,
            )
            original_count = len(instance.candidates)
            full_expected = full_metrics["expected_cost"]
            full_regret = full_metrics["max_regret"]
            _add_metrics(
                rows,
                method="regret_portfolio_full",
                key=key,
                meta=meta,
                instance=instance,
                result=full,
                oracle_costs=oracle,
                candidate_count=original_count,
                original_count=original_count,
                full_expected=full_expected,
                full_regret=full_regret,
            )
            lg_result, lg_instance, lg_extra = _solve_lg_rccc_audited(
                instance,
                proposal_instance,
                oracle,
                full_expected,
                full_regret,
                args.time_limit,
            )
            lg_result.solve_seconds = float(lg_extra.pop("solve_seconds_override"))
            _add_metrics(
                rows,
                method="lg_rccc_audited",
                key=key,
                meta=meta,
                instance=instance,
                result=lg_result,
                oracle_costs=oracle,
                candidate_count=len(lg_instance.candidates),
                original_count=original_count,
                full_expected=full_expected,
                full_regret=full_regret,
                extra=lg_extra,
            )
            for method in CORE_COMPARATORS:
                result = LaunchOptimizer(instance, time_limit=args.time_limit).solve(method)
                _add_metrics(
                    rows,
                    method=method,
                    key=key,
                    meta=meta,
                    instance=instance,
                    result=result,
                    oracle_costs=oracle,
                    candidate_count=original_count,
                    original_count=original_count,
                    full_expected=full_expected,
                    full_regret=full_regret,
                )
            pd.DataFrame(rows).to_csv(checkpoint_path, index=False)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "ai_or_stress_metrics.csv", index=False)
    summary = (
        metrics.groupby(["variant", "method"])
        .agg(
            cases=("instance_key", "count"),
            mean_max_regret=("max_regret", "mean"),
            worst_case_regret=("max_regret", "max"),
            mean_candidate_retention=("candidate_retention", "mean"),
            full_regret_matches=("matches_full_regret", "sum"),
            mean_solve_seconds=("solve_seconds", "mean"),
        )
        .reset_index()
        .sort_values(["variant", "mean_max_regret", "mean_candidate_retention"])
    )
    summary.to_csv(output_dir / "ai_or_stress_summary.csv", index=False)
    _statistical_audit(metrics, reference_method="lg_rccc_audited", seed=args.seed).to_csv(
        output_dir / "ai_or_stress_statistical_audit.csv",
        index=False,
    )
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AI-OR decision-intelligence audits for W-MLLA-R.")
    parser.add_argument("--mode", choices=["base", "stress"], default="base")
    parser.add_argument("--max-instances", type=int, default=12)
    parser.add_argument("--scenario-count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--time-limit", type=float, default=50.0)
    parser.add_argument("--oracle-time-limit", type=float, default=8.0)
    parser.add_argument("--ai-per-flight", type=int, default=20)
    parser.add_argument("--ai-per-aircraft", type=int, default=10)
    parser.add_argument("--train-split", choices=["month", "region", "instance"], default="month")
    parser.add_argument("--refresh-selection", action="store_true")
    parser.add_argument("--use-weather", action="store_true")
    parser.add_argument("--refresh-weather", action="store_true")
    parser.add_argument("--weather-dir", default="data/weather_asos")
    parser.add_argument("--weather-feature-dir", default=str(DEFAULT_WEATHER_FEATURE_DIR))
    parser.add_argument("--weather-utc-offset", type=int, default=None)
    parser.add_argument("--weather-local-start-hour", type=int, default=4)
    parser.add_argument("--weather-local-end-hour", type=int, default=12)
    parser.add_argument("--weather-pause-seconds", type=float, default=2.0)
    parser.add_argument("--weather-max-attempts", type=int, default=5)
    parser.add_argument("--weather-request-timeout-seconds", type=float, default=45.0)
    parser.add_argument("--value-mode", choices=["distance_cost", "unit_service"], default="distance_cost")
    parser.add_argument("--selected-instances", default="data/trc_open_smoke/batch18_weather_fullscale360_balanced_daily/selected_instances.csv")
    parser.add_argument("--main-metrics", default="results/trc_smoke/batch20_weather_full360_fullquality/batch20_metrics_with_rccc_audited.csv")
    parser.add_argument("--weather-audit", default="results/trc_smoke/batch20_weather_full360_fullquality/batch20_weather_audit.csv")
    parser.add_argument("--min-per-region", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-instances", type=int, default=3)
    parser.add_argument("--base-scenario-count", type=int, default=30)
    parser.add_argument("--stress-scenario-count", type=int, default=60)
    parser.add_argument("--weight-low", type=float, default=0.8)
    parser.add_argument("--weight-high", type=float, default=1.2)
    parser.add_argument("--weather-mild", type=float, default=0.75)
    parser.add_argument("--weather-severe", type=float, default=1.25)
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--data-dir", default="data/trc_open_smoke/batch34_decision_intelligence")
    parser.add_argument("--output-dir", default="results/trc_smoke/batch34_decision_intelligence")
    args = parser.parse_args()
    if args.mode == "stress":
        _run_stress(args)
    else:
        _run_base(args)


if __name__ == "__main__":
    main()
