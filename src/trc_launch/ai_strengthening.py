from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

DEFAULT_WEATHER_FEATURE_DIR = "data/trc_open_smoke/batch20_weather_full360_fullquality/weather_features"


SPLIT_LABELS = {
    "month": "Leave-month-out",
    "region": "Leave-region-out",
    "carrier": "Leave-carrier-out",
}

METHOD_LABELS = {
    "ml_proposal_only": "AI proposal only",
    "lg_rccc_audited": "LG-RCCC audited",
    "rccc_audited": "OR-only RCCC audited",
    "regret_portfolio_full": "Full regret reference",
    "saa_extensive": "SAA",
    "cvar_extensive": "CVaR",
    "minimax_extensive": "Minimax",
    "active_scenario_saa": "Active-scenario SAA",
}

METHOD_ORDER = [
    "ml_proposal_only",
    "lg_rccc_audited",
    "rccc_audited",
    "regret_portfolio_full",
    "saa_extensive",
    "cvar_extensive",
    "minimax_extensive",
    "active_scenario_saa",
]

ERROR_LABELS = {
    "missing_exposed_block_substitution": "Missing exposed-block substitution",
    "missing_surplus_source_tail": "Missing surplus-source tail",
    "over_retained_low_value_ferry": "Over-retained low-value ferry edge",
    "scenario_attention_miss": "Scenario-attention miss",
}


def _copy_with_candidates(instance, candidates: pd.DataFrame):
    return replace(instance, candidates=candidates.reset_index(drop=True))


def _import_experiment_runtime():
    from .decision_intelligence_audit import (
        _add_metrics,
        _feature_importance,
        _learned_proposal_instance,
        _load_selected,
        _prepare_training_labels,
        _solve_lg_rccc_audited,
        _solve_rccc_audited,
    )
    from .distilled_policy import fit_distilled_model
    from .optimizers import LaunchOptimizer
    from .risk_hardening import CORE_COMPARATORS

    return {
        "_add_metrics": _add_metrics,
        "_feature_importance": _feature_importance,
        "_learned_proposal_instance": _learned_proposal_instance,
        "_load_selected": _load_selected,
        "_prepare_training_labels": _prepare_training_labels,
        "_solve_lg_rccc_audited": _solve_lg_rccc_audited,
        "_solve_rccc_audited": _solve_rccc_audited,
        "fit_distilled_model": fit_distilled_model,
        "LaunchOptimizer": LaunchOptimizer,
        "CORE_COMPARATORS": CORE_COMPARATORS,
    }


def _metadata_key(meta: dict[str, object], split: str) -> object:
    if split == "month":
        return int(meta["month"])
    if split == "region":
        return str(meta["region"])
    if split == "carrier":
        return str(meta["carrier"])
    raise ValueError(f"unsupported split: {split}")


def _train_frames_for_split(
    labels: dict[str, pd.DataFrame],
    instances: dict[str, tuple[object, dict[str, object], dict[int, float], object, dict[str, float]]],
    key: str,
    split: str,
) -> list[pd.DataFrame]:
    meta = instances[key][1]
    heldout_value = _metadata_key(meta, split)
    frames = [
        frame
        for train_key, frame in labels.items()
        if train_key != key and _metadata_key(instances[train_key][1], split) != heldout_value
    ]
    if frames:
        return frames
    return [frame for train_key, frame in labels.items() if train_key != key]


def _rank_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(int)
    positives = labels == 1
    negatives = labels == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, ranks)
    average_ranks = sums[inverse] / counts[inverse]
    rank_sum = float(average_ranks[positives].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / max(1.0, n_pos * n_neg)


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(int)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    cumulative = np.cumsum(y)
    precision = cumulative / (np.arange(len(y)) + 1)
    return float((precision * y).sum() / positives)


def _recall_at_share(labels: np.ndarray, scores: np.ndarray, share: float) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    k = max(1, int(np.ceil(len(labels) * share)))
    order = np.argsort(-scores, kind="mergesort")[:k]
    return float(labels[order].sum() / positives)


def _safe_divide(numer: float, denom: float) -> float:
    return float(numer / denom) if denom else float("nan")


def _classify_error(
    label_frame: pd.DataFrame,
    proposal_features: pd.DataFrame,
    proposal_instance,
    pre_metrics: dict[str, object],
    lg_metrics: dict[str, object],
) -> tuple[str, str]:
    proposal_ids = set(proposal_instance.candidates["candidate_id"].astype(str))
    labels = label_frame.copy()
    labels["candidate_id"] = labels["candidate_id"].astype(str)
    positives = labels[labels["retained_after_audit"].astype(int).eq(1)]
    excluded = positives[~positives["candidate_id"].isin(proposal_ids)].copy()
    if not excluded.empty:
        if (
            (pd.to_numeric(excluded.get("weather_binding_share", 0), errors="coerce").fillna(0) > 0).any()
            or (pd.to_numeric(excluded.get("shortage_max", 0), errors="coerce").fillna(0) > 0).any()
        ):
            return "missing_exposed_block_substitution", "|".join(excluded.head(3)["candidate_id"].tolist())
        if (
            (pd.to_numeric(excluded.get("source_surplus", 0), errors="coerce").fillna(0) > 0).any()
            or (pd.to_numeric(excluded.get("origin_deficit", 0), errors="coerce").fillna(0) > 0).any()
        ):
            return "missing_surplus_source_tail", "|".join(excluded.head(3)["candidate_id"].tolist())

    features = proposal_features.copy()
    if "proposal_score" in features.columns:
        features["candidate_id"] = features["candidate_id"].astype(str)
        kept = features[features["candidate_id"].isin(proposal_ids)].copy()
        kept_neg = kept[~kept["candidate_id"].isin(set(positives["candidate_id"]))]
        if not kept_neg.empty:
            high_ferry = pd.to_numeric(kept_neg.get("movement_burden", kept_neg.get("ferry_cost", 0)), errors="coerce")
            if (high_ferry.fillna(0) >= high_ferry.fillna(0).quantile(0.75)).any():
                return "over_retained_low_value_ferry", "|".join(kept_neg.head(3)["candidate_id"].tolist())

    if float(pre_metrics["max_regret"]) > float(lg_metrics["max_regret"]) + 1e-8:
        return "scenario_attention_miss", str(pre_metrics.get("audit_path", "ai_proposal"))
    return "scenario_attention_miss", ""


def _learning_row(
    *,
    split: str,
    key: str,
    meta: dict[str, object],
    label_frame: pd.DataFrame,
    proposal_features: pd.DataFrame,
    proposal_instance,
    pre_metrics: dict[str, object],
    lg_metrics: dict[str, object],
) -> dict[str, object]:
    merged = label_frame.merge(
        proposal_features[["candidate_id", "proposal_probability", "proposal_score"]],
        on="candidate_id",
        how="left",
    )
    merged["proposal_probability"] = pd.to_numeric(merged["proposal_probability"], errors="coerce").fillna(0.0)
    merged["proposal_score"] = pd.to_numeric(merged["proposal_score"], errors="coerce").fillna(-1e9)
    y_retained = merged["retained_after_audit"].astype(int).to_numpy()
    y_selected = merged["selected_assignment"].astype(int).to_numpy()
    scores = merged["proposal_score"].to_numpy(float)
    proposal_ids = set(proposal_instance.candidates["candidate_id"].astype(str))
    retained_ids = set(merged.loc[merged["retained_after_audit"].astype(int).eq(1), "candidate_id"].astype(str))
    selected_ids = set(merged.loc[merged["selected_assignment"].astype(int).eq(1), "candidate_id"].astype(str))
    proposed_negative = merged[
        merged["candidate_id"].astype(str).isin(proposal_ids)
        & merged["retained_after_audit"].astype(int).eq(0)
    ]
    excluded_positive = retained_ids.difference(proposal_ids)
    error_type, example_edges = _classify_error(
        label_frame,
        proposal_features,
        proposal_instance,
        pre_metrics,
        lg_metrics,
    )
    return {
        "split": split,
        "split_label": SPLIT_LABELS[split],
        "instance_key": key,
        **meta,
        "candidate_count": int(len(merged)),
        "proposal_edges": int(len(proposal_ids)),
        "candidate_retention": _safe_divide(len(proposal_ids), len(merged)),
        "retained_edge_count": int(len(retained_ids)),
        "selected_edge_count": int(len(selected_ids)),
        "edge_label_auc": _rank_auc(y_retained, scores),
        "edge_label_average_precision": _average_precision(y_retained, scores),
        "selected_label_auc": _rank_auc(y_selected, scores),
        "selected_label_average_precision": _average_precision(y_selected, scores),
        "top10_retained_recall": _recall_at_share(y_retained, scores, 0.10),
        "top20_retained_recall": _recall_at_share(y_retained, scores, 0.20),
        "retained_edge_recall": _safe_divide(len(retained_ids.intersection(proposal_ids)), len(retained_ids)),
        "selected_edge_recall": _safe_divide(len(selected_ids.intersection(proposal_ids)), len(selected_ids)),
        "false_exclusion_rate": _safe_divide(len(excluded_positive), len(retained_ids)),
        "false_inclusion_share": _safe_divide(len(proposed_negative), len(proposal_ids)),
        "proposal_only_full_regret_match": bool(pre_metrics["matches_full_regret"]),
        "audited_full_regret_match": bool(lg_metrics["matches_full_regret"]),
        "audit_recovered": bool((not pre_metrics["matches_full_regret"]) and lg_metrics["matches_full_regret"]),
        "audit_expansions": int(lg_metrics.get("audit_expansions", 0)),
        "fallback_stage": str(lg_metrics.get("fallback_stage", "")),
        "proposal_max_regret": float(pre_metrics["max_regret"]),
        "audited_max_regret": float(lg_metrics["max_regret"]),
        "proposal_solve_seconds": float(pre_metrics["solve_seconds"]),
        "audited_solve_seconds": float(lg_metrics["solve_seconds"]),
        "error_type": error_type,
        "example_edges": example_edges,
    }


def _summarize_learning(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    summary = (
        rows.groupby(["split", "split_label"], sort=False)
        .agg(
            cases=("instance_key", "nunique"),
            mean_edge_auc=("edge_label_auc", "mean"),
            mean_average_precision=("edge_label_average_precision", "mean"),
            mean_top10_retained_recall=("top10_retained_recall", "mean"),
            mean_top20_retained_recall=("top20_retained_recall", "mean"),
            mean_retained_edge_recall=("retained_edge_recall", "mean"),
            mean_selected_edge_recall=("selected_edge_recall", "mean"),
            mean_false_exclusion_rate=("false_exclusion_rate", "mean"),
            mean_false_inclusion_share=("false_inclusion_share", "mean"),
            mean_candidate_retention=("candidate_retention", "mean"),
            proposal_full_regret_matches=("proposal_only_full_regret_match", "sum"),
            audited_full_regret_matches=("audited_full_regret_match", "sum"),
            audit_recoveries=("audit_recovered", "sum"),
            full_fallbacks=("fallback_stage", lambda values: int((pd.Series(values).astype(str) == "full").sum())),
            mean_proposal_seconds=("proposal_solve_seconds", "mean"),
            mean_audited_seconds=("audited_solve_seconds", "mean"),
        )
        .reset_index()
    )
    return summary


def _summarize_methods(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    summary = (
        metrics.groupby(["split", "method"], sort=False)
        .agg(
            cases=("instance_key", "nunique"),
            candidate_retention=("candidate_retention", "mean"),
            mean_max_regret=("max_regret", "mean"),
            worst_max_regret=("max_regret", "max"),
            solve_seconds=("solve_seconds", "mean"),
            full_regret_matches=("matches_full_regret", "sum"),
        )
        .reset_index()
    )
    summary["method_label"] = summary["method"].map(METHOD_LABELS).fillna(summary["method"])
    summary["method_order"] = summary["method"].map({method: idx for idx, method in enumerate(METHOD_ORDER)}).fillna(99)
    return summary.sort_values(["split", "method_order"]).drop(columns=["method_order"])


def _summarize_errors(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    missed = rows[~rows["proposal_only_full_regret_match"]].copy()
    if missed.empty:
        return pd.DataFrame(
            {
                "error_type": list(ERROR_LABELS),
                "error_label": [ERROR_LABELS[key] for key in ERROR_LABELS],
                "cases": [0] * len(ERROR_LABELS),
                "recoveries": [0] * len(ERROR_LABELS),
            }
        )
    out = (
        missed.groupby("error_type")
        .agg(
            cases=("instance_key", "nunique"),
            recoveries=("audit_recovered", "sum"),
            mean_false_exclusion_rate=("false_exclusion_rate", "mean"),
            mean_proposal_regret=("proposal_max_regret", "mean"),
            mean_audited_regret=("audited_max_regret", "mean"),
        )
        .reset_index()
    )
    out["error_label"] = out["error_type"].map(ERROR_LABELS).fillna(out["error_type"])
    return _complete_error_rows(out)


def _complete_error_rows(errors: pd.DataFrame) -> pd.DataFrame:
    rows = errors.copy()
    if rows.empty:
        rows = pd.DataFrame({"error_type": list(ERROR_LABELS)})
    rows = rows.set_index("error_type", drop=False)
    for error_type, label in ERROR_LABELS.items():
        if error_type not in rows.index:
            rows.loc[error_type, "error_type"] = error_type
            rows.loc[error_type, "error_label"] = label
            rows.loc[error_type, "cases"] = 0
            rows.loc[error_type, "recoveries"] = 0
    rows["error_label"] = rows["error_type"].map(ERROR_LABELS).fillna(rows.get("error_label"))
    for col in ["cases", "recoveries"]:
        rows[col] = pd.to_numeric(rows.get(col, 0), errors="coerce").fillna(0).astype(int)
    for col in ["mean_false_exclusion_rate", "mean_proposal_regret", "mean_audited_regret"]:
        if col not in rows:
            rows[col] = np.nan
    return rows.loc[list(ERROR_LABELS)].reset_index(drop=True)


def _write_business_complexity(args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    selected = pd.read_csv(args.selected_instances)
    selected["candidate_count_preview"] = pd.to_numeric(selected["candidate_count_preview"], errors="coerce")
    selected["flights"] = pd.to_numeric(selected["flights"], errors="coerce")
    rows: list[dict[str, object]] = [
        {
            "stress_family": "multi_carrier_dense_region",
            "cases": int(len(selected)),
            "regions": int(selected["region"].nunique()),
            "carriers": int(selected["carrier"].nunique()),
            "mean_flights": float(selected["flights"].mean()),
            "mean_candidate_edges": float(selected["candidate_count_preview"].mean()),
            "max_candidate_edges": float(selected["candidate_count_preview"].max()),
            "evidence_source": "BTS-IEM launch benchmark",
        }
    ]
    weather_path = Path(args.weather_event_case_validation)
    if weather_path.exists():
        weather = pd.read_csv(weather_path)
        rows.append(
            {
                "stress_family": "weather_event_days",
                "cases": int(len(weather)),
                "regions": int(weather["region"].nunique()) if "region" in weather else 0,
                "carriers": int(weather["carrier"].nunique()) if "carrier" in weather else 0,
                "mean_flights": float(pd.to_numeric(weather.get("flights", pd.Series(dtype=float)), errors="coerce").mean()),
                "mean_candidate_edges": float("nan"),
                "max_candidate_edges": float("nan"),
                "evidence_source": "historical weather-event audit",
            }
        )
    risk_path = Path(args.risk_hardening_summary)
    if risk_path.exists():
        risk = pd.read_csv(risk_path)
        risk = risk[risk["variant"].isin(["weight_high", "weight_low"])]
        lg = risk[risk["method"].astype(str).str.contains("rccc_audited|lg_rccc_audited", case=False, regex=True)]
        rows.append(
            {
                "stress_family": "service_priority_perturbation",
                "cases": int(pd.to_numeric(lg.get("cases", pd.Series(dtype=float)), errors="coerce").sum()),
                "regions": int(pd.to_numeric(lg.get("regions", pd.Series(dtype=float)), errors="coerce").max()),
                "carriers": float("nan"),
                "mean_flights": float("nan"),
                "mean_candidate_edges": float("nan"),
                "max_candidate_edges": float("nan"),
                "evidence_source": "service-weight sensitivity audit",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "business_complexity_summary.csv", index=False)
    return out


def _plot_learning(summary: pd.DataFrame, output_dir: Path) -> None:
    if plt is None:
        return
    if summary.empty:
        return
    labels = summary["split_label"].astype(str).tolist()
    x = np.arange(len(labels))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.1, 3.35))
    width = 0.34
    ax0.bar(x - width / 2, summary["mean_edge_auc"].astype(float), width, label="AUC", color="#1F77B4")
    ax0.bar(x + width / 2, summary["mean_average_precision"].astype(float), width, label="Avg. precision", color="#2CA02C")
    ax0.set_xticks(x, labels, rotation=18, ha="right")
    ax0.set_ylim(0, 1.02)
    ax0.set_ylabel("Held-out edge-label score")
    ax0.set_title("Proposal-agent learning")
    ax0.grid(axis="y", alpha=0.25)
    ax0.legend(frameon=False, fontsize=7)
    ax1.bar(x - width / 2, summary["mean_retained_edge_recall"].astype(float), width, label="Retained-edge recall", color="#0B6E69")
    ax1.bar(x + width / 2, summary["mean_candidate_retention"].astype(float), width, label="Candidate retention", color="#9467BD")
    ax1.set_xticks(x, labels, rotation=18, ha="right")
    ax1.set_ylim(0, 1.02)
    ax1.set_title("Candidate graph quality")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(frameon=False, fontsize=7)
    fig.tight_layout(pad=0.35)
    fig.savefig(output_dir / "fig_ai_learning_generalization.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig_ai_learning_generalization.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_errors(errors: pd.DataFrame, output_dir: Path) -> None:
    if errors.empty:
        return
    errors = errors.copy()
    errors["error_label"] = errors["error_type"].map(ERROR_LABELS).fillna(errors["error_type"])
    if plt is None:
        _plot_errors_static(errors, output_dir)
        return
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    y = np.arange(len(errors))
    bars = ax.barh(y, errors["cases"].astype(float), color="#D55E00", alpha=0.82)
    ax.set_yticks(y, errors["error_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Proposal-only missed cases")
    ax.set_title("AI proposal error taxonomy")
    ax.grid(axis="x", alpha=0.25)
    for bar, recoveries in zip(bars, errors["recoveries"].astype(int)):
        value = bar.get_width()
        ax.text(value + 0.05, bar.get_y() + bar.get_height() / 2, f"recovered {recoveries}", va="center", fontsize=7)
    fig.tight_layout(pad=0.35)
    fig.savefig(output_dir / "fig_ai_error_taxonomy.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig_ai_error_taxonomy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_errors_static(errors: pd.DataFrame, output_dir: Path) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import landscape
        from reportlab.pdfgen import canvas
    except ModuleNotFoundError:
        return

    rows = errors.copy()
    rows["cases"] = pd.to_numeric(rows["cases"], errors="coerce").fillna(0).astype(int)
    rows["recoveries"] = pd.to_numeric(rows.get("recoveries", 0), errors="coerce").fillna(0).astype(int)
    rows = rows.sort_values("cases", ascending=True)
    max_cases = max(1, int(rows["cases"].max()))

    width, height = 1600, 760
    margin_left, margin_right, margin_top, margin_bottom = 470, 170, 130, 110
    bar_area = width - margin_left - margin_right
    row_gap = 118
    bar_h = 46
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    try:
        font_title = ImageFont.truetype("arial.ttf", 46)
        font_label = ImageFont.truetype("arial.ttf", 30)
        font_small = ImageFont.truetype("arial.ttf", 26)
    except OSError:
        font_title = ImageFont.load_default()
        font_label = ImageFont.load_default()
        font_small = ImageFont.load_default()

    draw.text((margin_left, 42), "AI proposal error taxonomy", fill=(20, 20, 20), font=font_title)
    draw.line((margin_left, height - margin_bottom + 10, width - margin_right, height - margin_bottom + 10), fill=(110, 110, 110), width=2)
    for tick in range(max_cases + 1):
        x = margin_left + bar_area * tick / max_cases
        draw.line((x, height - margin_bottom + 5, x, height - margin_bottom + 20), fill=(110, 110, 110), width=2)
        draw.text((x - 8, height - margin_bottom + 28), str(tick), fill=(50, 50, 50), font=font_small)
    draw.text((margin_left + bar_area / 2 - 155, height - 52), "Proposal-only missed cases", fill=(50, 50, 50), font=font_small)

    for idx, row in enumerate(rows.itertuples(index=False)):
        y = margin_top + idx * row_gap
        label = str(getattr(row, "error_label"))
        cases = int(getattr(row, "cases"))
        recoveries = int(getattr(row, "recoveries"))
        draw.text((35, y + 2), label, fill=(20, 20, 20), font=font_label)
        bar_w = int(bar_area * cases / max_cases)
        draw.rectangle((margin_left, y, margin_left + bar_w, y + bar_h), fill=(207, 94, 24))
        draw.text((margin_left + bar_w + 18, y + 5), f"{cases}; recovered {recoveries}", fill=(20, 20, 20), font=font_small)

    png_path = output_dir / "fig_ai_error_taxonomy.png"
    pdf_path = output_dir / "fig_ai_error_taxonomy.pdf"
    img.save(png_path, dpi=(300, 300))

    page_w, page_h = landscape((7.1 * 72, 3.4 * 72))
    c = canvas.Canvas(str(pdf_path), pagesize=(page_w, page_h))
    c.setFont("Helvetica-Bold", 13)
    c.drawString(180, page_h - 28, "AI proposal error taxonomy")
    x0, y0, chart_w = 170, 58, 270
    c.setStrokeColor(colors.HexColor("#777777"))
    c.line(x0, y0 - 8, x0 + chart_w, y0 - 8)
    for tick in range(max_cases + 1):
        x = x0 + chart_w * tick / max_cases
        c.line(x, y0 - 12, x, y0 - 6)
        c.setFont("Helvetica", 7)
        c.drawCentredString(x, y0 - 24, str(tick))
    c.drawCentredString(x0 + chart_w / 2, 16, "Proposal-only missed cases")
    for idx, row in enumerate(rows.itertuples(index=False)):
        y = page_h - 66 - idx * 44
        label = str(getattr(row, "error_label"))
        cases = int(getattr(row, "cases"))
        recoveries = int(getattr(row, "recoveries"))
        bar_w = chart_w * cases / max_cases
        c.setFont("Helvetica", 8.5)
        c.setFillColor(colors.black)
        c.drawRightString(x0 - 8, y + 5, label)
        c.setFillColor(colors.HexColor("#D55E00"))
        c.rect(x0, y, bar_w, 16, fill=1, stroke=0)
        c.setFillColor(colors.black)
        c.drawString(x0 + bar_w + 6, y + 4, f"{cases}; recovered {recoveries}")
    c.showPage()
    c.save()


def _bool_series(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().isin(["true", "1", "yes"])


def _method_summary_from_metrics(metrics: pd.DataFrame, split: str = "month") -> pd.DataFrame:
    frame = metrics[metrics["method"].isin(METHOD_ORDER)].copy()
    if "split" not in frame.columns:
        frame["split"] = split
    frame["matches_full_regret_bool"] = _bool_series(frame["matches_full_regret"])
    out = (
        frame.groupby(["split", "method"], sort=False)
        .agg(
            cases=("instance_key", "nunique"),
            candidate_retention=("candidate_retention", "mean"),
            mean_max_regret=("max_regret", "mean"),
            worst_max_regret=("max_regret", "max"),
            solve_seconds=("solve_seconds", "mean"),
            full_regret_matches=("matches_full_regret_bool", "sum"),
        )
        .reset_index()
    )
    out["method_label"] = out["method"].map(METHOD_LABELS).fillna(out["method"])
    return out


def _existing_learning_summary(metrics: pd.DataFrame, action_log: pd.DataFrame) -> pd.DataFrame:
    method_summary = _method_summary_from_metrics(metrics)
    ml = method_summary[method_summary["method"].eq("ml_proposal_only")].iloc[0]
    lg = method_summary[method_summary["method"].eq("lg_rccc_audited")].iloc[0]
    full = method_summary[method_summary["method"].eq("regret_portfolio_full")].iloc[0]
    rccc = method_summary[method_summary["method"].eq("rccc_audited")].iloc[0]
    audit_recoveries = int((~_bool_series(metrics[metrics["method"].eq("ml_proposal_only")]["matches_full_regret"])).sum())
    full_fallbacks = 0
    if not action_log.empty and "fallback_stage" in action_log:
        full_fallbacks = int(action_log["fallback_stage"].astype(str).eq("full").sum())
    return pd.DataFrame(
        [
            {
                "split": "month",
                "split_label": "Leave-month-out",
                "cases": int(ml["cases"]),
                "mean_edge_auc": np.nan,
                "mean_average_precision": np.nan,
                "mean_top10_retained_recall": np.nan,
                "mean_top20_retained_recall": np.nan,
                "mean_retained_edge_recall": float(1.0 - audit_recoveries / max(1, int(ml["cases"]))),
                "mean_selected_edge_recall": np.nan,
                "mean_false_exclusion_rate": float(audit_recoveries / max(1, int(ml["cases"]))),
                "mean_false_inclusion_share": np.nan,
                "mean_candidate_retention": float(ml["candidate_retention"]),
                "proposal_full_regret_matches": int(ml["full_regret_matches"]),
                "audited_full_regret_matches": int(lg["full_regret_matches"]),
                "audit_recoveries": audit_recoveries,
                "full_fallbacks": full_fallbacks,
                "mean_proposal_seconds": float(ml["solve_seconds"]),
                "mean_audited_seconds": float(lg["solve_seconds"]),
                "full_reference_seconds": float(full["solve_seconds"]),
                "or_only_rccc_seconds": float(rccc["solve_seconds"]),
            }
        ]
    )


def _existing_error_taxonomy(metrics: pd.DataFrame, action_log: pd.DataFrame) -> pd.DataFrame:
    ml = metrics[metrics["method"].eq("ml_proposal_only")].copy()
    ml["match"] = _bool_series(ml["matches_full_regret"])
    missed = ml[~ml["match"]].copy()
    if missed.empty:
        return pd.DataFrame(
            {
                "error_type": list(ERROR_LABELS),
                "error_label": [ERROR_LABELS[key] for key in ERROR_LABELS],
                "cases": [0] * len(ERROR_LABELS),
                "recoveries": [0] * len(ERROR_LABELS),
                "mean_proposal_regret": [np.nan] * len(ERROR_LABELS),
                "mean_audited_regret": [np.nan] * len(ERROR_LABELS),
            }
        )
    lg = metrics[metrics["method"].eq("lg_rccc_audited")][["instance_key", "max_regret"]].rename(
        columns={"max_regret": "audited_regret"}
    )
    missed = missed.merge(lg, on="instance_key", how="left")
    if not action_log.empty:
        keep = ["instance_key", "ai_binding_blocks", "audit_path", "fallback_stage"]
        missed = missed.merge(action_log[[col for col in keep if col in action_log.columns]], on="instance_key", how="left")
    labels = []
    for row in missed.itertuples(index=False):
        binding = str(getattr(row, "ai_binding_blocks", ""))
        fallback = str(getattr(row, "fallback_stage", ""))
        if binding and binding.lower() != "nan":
            labels.append("missing_exposed_block_substitution")
        elif fallback == "full":
            labels.append("missing_surplus_source_tail")
        else:
            labels.append("scenario_attention_miss")
    missed["error_type"] = labels
    out = (
        missed.groupby("error_type")
        .agg(
            cases=("instance_key", "nunique"),
            recoveries=("audited_regret", lambda values: int(values.notna().sum())),
            mean_proposal_regret=("max_regret", "mean"),
            mean_audited_regret=("audited_regret", "mean"),
        )
        .reset_index()
    )
    out["error_label"] = out["error_type"].map(ERROR_LABELS).fillna(out["error_type"])
    return _complete_error_rows(out)


def _existing_scenario_efficiency(args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    scaling_path = Path(args.scenario_scaling_summary)
    if not scaling_path.exists():
        return pd.DataFrame()
    scaling = pd.read_csv(scaling_path)
    out = pd.DataFrame(
        {
            "scenario_count": pd.to_numeric(scaling["scenario_count"], errors="coerce").astype(int),
            "cases": pd.to_numeric(scaling["cases"], errors="coerce").astype(int),
            "full_reference_seconds": pd.to_numeric(scaling["mean_full_seconds"], errors="coerce"),
            "audited_certificate_seconds": pd.to_numeric(scaling["mean_rccc_seconds"], errors="coerce"),
            "time_ratio_full_to_audited": pd.to_numeric(scaling["full_to_rccc_time_ratio"], errors="coerce"),
            "candidate_retention": pd.to_numeric(scaling["mean_rccc_retention"], errors="coerce"),
            "full_regret_matches": pd.to_numeric(scaling["rccc_regret_matches"], errors="coerce").astype(int),
        }
    )
    s180_path = Path(args.scenario_s180_summary)
    if s180_path.exists():
        s180 = pd.read_csv(s180_path)
        if {"scenario_count", "cases"}.issubset(s180.columns):
            s180_out = pd.DataFrame(
                {
                    "scenario_count": pd.to_numeric(s180["scenario_count"], errors="coerce").astype(int),
                    "cases": pd.to_numeric(s180["cases"], errors="coerce").astype(int),
                    "full_reference_seconds": pd.to_numeric(s180.get("mean_full_seconds"), errors="coerce"),
                    "audited_certificate_seconds": pd.to_numeric(s180.get("mean_rccc_seconds"), errors="coerce"),
                    "time_ratio_full_to_audited": pd.to_numeric(s180.get("full_to_rccc_time_ratio"), errors="coerce"),
                    "candidate_retention": pd.to_numeric(s180.get("mean_rccc_retention"), errors="coerce"),
                    "full_regret_matches": pd.to_numeric(s180.get("rccc_regret_matches"), errors="coerce").astype(int),
                }
            )
            out = pd.concat([out, s180_out], ignore_index=True, sort=False)
    out = out.drop_duplicates(subset=["scenario_count"], keep="first").sort_values("scenario_count")
    out.to_csv(output_dir / "ai_scenario_efficiency_summary.csv", index=False)
    return out


def write_from_existing(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(args.existing_metrics)
    action_log = pd.read_csv(args.existing_action_log) if Path(args.existing_action_log).exists() else pd.DataFrame()
    method_summary = _method_summary_from_metrics(metrics)
    learning_summary = _existing_learning_summary(metrics, action_log)
    errors = _existing_error_taxonomy(metrics, action_log)
    method_summary.to_csv(output_dir / "ai_audit_contribution_summary.csv", index=False)
    learning_summary.to_csv(output_dir / "ai_learning_summary.csv", index=False)
    errors.to_csv(output_dir / "ai_error_taxonomy.csv", index=False)
    if not action_log.empty:
        action_log.to_csv(output_dir / "ai_action_log_existing.csv", index=False)
    if Path(args.existing_feature_importance).exists():
        importance = pd.read_csv(args.existing_feature_importance)
        if "importance" in importance:
            importance["importance"] = pd.to_numeric(importance["importance"], errors="coerce").fillna(0.0)
            importance = importance.groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False)
        importance.to_csv(output_dir / "ai_feature_importance_summary.csv", index=False)
    _write_business_complexity(args, output_dir)
    _existing_scenario_efficiency(args, output_dir)
    _plot_learning(learning_summary, output_dir)
    _plot_errors(errors, output_dir)
    print(learning_summary.to_string(index=False))
    print(method_summary.to_string(index=False))


def run_diagnostics(args: argparse.Namespace) -> None:
    runtime = _import_experiment_runtime()
    _add_metrics = runtime["_add_metrics"]
    _feature_importance = runtime["_feature_importance"]
    _learned_proposal_instance = runtime["_learned_proposal_instance"]
    _load_selected = runtime["_load_selected"]
    _prepare_training_labels = runtime["_prepare_training_labels"]
    _solve_lg_rccc_audited = runtime["_solve_lg_rccc_audited"]
    _solve_rccc_audited = runtime["_solve_rccc_audited"]
    fit_distilled_model = runtime["fit_distilled_model"]
    LaunchOptimizer = runtime["LaunchOptimizer"]
    CORE_COMPARATORS = runtime["CORE_COMPARATORS"]

    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    selected_path = Path(args.selected_instances)
    if selected_path.exists() and not args.refresh_selection:
        selected = pd.read_csv(selected_path)
        if args.max_instances > 0:
            selected = selected.head(args.max_instances).reset_index(drop=True)
        selected.to_csv(data_dir / "selected_instances.csv", index=False)
    else:
        selected = _load_selected(data_dir / "selected_instances.csv", args.max_instances, args.refresh_selection)

    instances, labels = _prepare_training_labels(selected, args)
    if not instances:
        raise RuntimeError("No usable instances for AI strengthening diagnostics.")

    learning_rows: list[dict[str, object]] = []
    metrics_rows: list[dict[str, object]] = []
    feature_rows: list[pd.DataFrame] = []
    model_cache: dict[tuple[str, object], object] = {}
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]

    for split in splits:
        if split not in SPLIT_LABELS:
            raise ValueError(f"unsupported split: {split}")
        for key, (instance, meta, oracle, full, full_metrics) in instances.items():
            fold_key = (split, _metadata_key(meta, split))
            train_frames = _train_frames_for_split(labels, instances, key, split)
            if fold_key not in model_cache:
                model = fit_distilled_model(train_frames)
                model_cache[fold_key] = model
                importance = _feature_importance(model, train_frames)
                importance["split"] = split
                importance["heldout_fold"] = str(fold_key)
                feature_rows.append(importance)
            model = model_cache[fold_key]
            proposal_instance, proposal_features = _learned_proposal_instance(
                instance,
                model,
                per_flight=args.ai_per_flight,
                per_aircraft=args.ai_per_aircraft,
            )
            original_count = len(instance.candidates)
            full_expected = float(full_metrics["expected_cost"])
            full_regret = float(full_metrics["max_regret"])
            _add_metrics(
                metrics_rows,
                method="regret_portfolio_full",
                key=key,
                meta={**meta, "split": split},
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
                metrics_rows,
                method="rccc_audited",
                key=key,
                meta={**meta, "split": split},
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
            pre_metrics = _add_metrics(
                metrics_rows,
                method="ml_proposal_only",
                key=key,
                meta={**meta, "split": split},
                instance=instance,
                result=ml_result,
                oracle_costs=oracle,
                candidate_count=len(proposal_instance.candidates),
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
            lg_metrics = _add_metrics(
                metrics_rows,
                method="lg_rccc_audited",
                key=key,
                meta={**meta, "split": split},
                instance=instance,
                result=lg_result,
                oracle_costs=oracle,
                candidate_count=len(lg_instance.candidates),
                original_count=original_count,
                full_expected=full_expected,
                full_regret=full_regret,
                extra=lg_extra,
            )
            if args.include_comparators:
                for method in CORE_COMPARATORS:
                    result = LaunchOptimizer(instance, time_limit=args.time_limit).solve(method)
                    _add_metrics(
                        metrics_rows,
                        method=method,
                        key=key,
                        meta={**meta, "split": split},
                        instance=instance,
                        result=result,
                        oracle_costs=oracle,
                        candidate_count=original_count,
                        original_count=original_count,
                        full_expected=full_expected,
                        full_regret=full_regret,
                    )
            learning_rows.append(
                _learning_row(
                    split=split,
                    key=key,
                    meta=meta,
                    label_frame=labels[key],
                    proposal_features=proposal_features,
                    proposal_instance=proposal_instance,
                    pre_metrics=pre_metrics,
                    lg_metrics=lg_metrics,
                )
            )
            pd.DataFrame(learning_rows).to_csv(output_dir / "ai_learning_diagnostics_checkpoint.csv", index=False)
            pd.DataFrame(metrics_rows).to_csv(output_dir / "ai_strengthening_metrics_checkpoint.csv", index=False)

    learning = pd.DataFrame(learning_rows)
    metrics = pd.DataFrame(metrics_rows)
    learning.to_csv(output_dir / "ai_learning_diagnostics.csv", index=False)
    metrics.to_csv(output_dir / "ai_strengthening_metrics.csv", index=False)
    learning_summary = _summarize_learning(learning)
    method_summary = _summarize_methods(metrics)
    errors = _summarize_errors(learning)
    learning_summary.to_csv(output_dir / "ai_learning_summary.csv", index=False)
    method_summary.to_csv(output_dir / "ai_audit_contribution_summary.csv", index=False)
    errors.to_csv(output_dir / "ai_error_taxonomy.csv", index=False)
    if not learning.empty:
        missed = learning[~learning["proposal_only_full_regret_match"]].copy()
        missed.sort_values(["split", "proposal_max_regret"], ascending=[True, False]).head(12).to_csv(
            output_dir / "ai_error_trace_cases.csv",
            index=False,
        )
    if feature_rows:
        feature_summary = (
            pd.concat(feature_rows, ignore_index=True)
            .groupby(["split", "feature"], as_index=False)["importance"]
            .mean()
            .sort_values(["split", "importance"], ascending=[True, False])
        )
        feature_summary.to_csv(output_dir / "ai_feature_importance_summary.csv", index=False)
    _write_business_complexity(args, output_dir)
    _plot_learning(learning_summary, output_dir)
    _plot_errors(errors, output_dir)
    print(learning_summary.to_string(index=False))
    print(method_summary.to_string(index=False))


def write_manuscript_assets(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    generated_dir = Path(args.generated_dir)
    figure_dir = Path(args.figure_dir)
    generated_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    for name in [
        "ai_learning_summary.csv",
        "ai_audit_contribution_summary.csv",
        "ai_error_taxonomy.csv",
        "ai_error_trace_cases.csv",
        "ai_feature_importance_summary.csv",
        "ai_scenario_efficiency_summary.csv",
        "business_complexity_summary.csv",
    ]:
        source = output_dir / name
        if source.exists():
            pd.read_csv(source).to_csv(generated_dir / name, index=False)
    for stem in ["fig_ai_learning_generalization", "fig_ai_error_taxonomy"]:
        for suffix in ["pdf", "png"]:
            source = output_dir / f"{stem}.{suffix}"
            if source.exists():
                (figure_dir / f"{stem}.{suffix}").write_bytes(source.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AI proposal-agent strengthening diagnostics.")
    parser.add_argument("--mode", choices=["diagnostics", "from-existing", "assets"], default="diagnostics")
    parser.add_argument("--max-instances", type=int, default=12)
    parser.add_argument("--scenario-count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--time-limit", type=float, default=50.0)
    parser.add_argument("--oracle-time-limit", type=float, default=8.0)
    parser.add_argument("--ai-per-flight", type=int, default=20)
    parser.add_argument("--ai-per-aircraft", type=int, default=10)
    parser.add_argument("--splits", default="month,region,carrier")
    parser.add_argument("--include-comparators", action="store_true")
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
    parser.add_argument("--selected-instances", default="data/trc_open_smoke/batch20_weather_full360_fullquality/selected_instances.csv")
    parser.add_argument("--weather-event-case-validation", default="article/trc_elsarticle/generated/weather_event_case_validation.csv")
    parser.add_argument("--risk-hardening-summary", default="article/trc_elsarticle/generated/risk_hardening_summary.csv")
    parser.add_argument("--scenario-scaling-summary", default="article/trc_elsarticle/generated/scenario_scaling_summary.csv")
    parser.add_argument("--scenario-s180-summary", default="article/trc_elsarticle/generated/scenario_scaling_s180_subset_summary.csv")
    parser.add_argument("--existing-metrics", default="results/trc_smoke/batch34_decision_intelligence_weather_full360/ai_or_metrics.csv")
    parser.add_argument("--existing-action-log", default="results/trc_smoke/batch34_decision_intelligence_weather_full360/ai_or_action_log.csv")
    parser.add_argument("--existing-feature-importance", default="results/trc_smoke/batch34_decision_intelligence_weather_full360/ai_or_feature_importance.csv")
    parser.add_argument("--data-dir", default="data/trc_ai_strengthening")
    parser.add_argument("--output-dir", default="results/trc_ai_strengthening")
    parser.add_argument("--generated-dir", default="article/trc_elsarticle/generated")
    parser.add_argument("--figure-dir", default="article/trc_elsarticle/figures")
    args = parser.parse_args()
    if args.mode == "assets":
        write_manuscript_assets(args)
    elif args.mode == "from-existing":
        write_from_existing(args)
    else:
        run_diagnostics(args)


if __name__ == "__main__":
    main()
