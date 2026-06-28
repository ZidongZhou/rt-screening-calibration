#!/usr/bin/env python
"""Selective auditing and full-form-escalation analyses.

This module intentionally leaves ``engine.py`` as the primary simulation engine
and builds the new deployment-oriented audit experiments from observed-prefix
prediction frames.  All audit rankings use only information available after the
19-item reduced screen; complete questionnaire labels are used only as the
reference for evaluation or for the explicitly labeled oracle upper bound.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rt_screening.engine import (  # noqa:E402
    SCALE_NAMES,
    SCALE_ORDER,
    SCALES,
    _labels_for_scores,
    _scores_from_selected,
    ensure_dirs,
    fit_rt_residuals,
    get_calibration_sigmas,
    load_config,
    load_ids,
    make_targets_frame,
    predict_from_selected,
    root_path,
    simulate_item_selection,
)

TABLE_DIR = ROOT / "results" / "tables"
FIG_DIR = ROOT / "results" / "figures"

CONF_FEATURES = ["issued_confidence", "entropy", "margin"]
SCORE_FEATURES = ["score_distance", "n_scale_items", "scale_item_fraction", "scale_coverage"]
RT_FEATURES = ["reliability", "fast", "long_pause", "std", "mean_rt", "abs_mean_rt"]
ALL_FEATURES = CONF_FEATURES + SCORE_FEATURES + RT_FEATURES


def _audit_max_train_participants():
    """Return the configured or environment-overridden training participant cap.

    Set AUDIT_MAX_TRAIN_PARTICIPANTS=none to use the full training split for
    sensitivity checks without changing the compact-release default.
    """
    env = os.environ.get("AUDIT_MAX_TRAIN_PARTICIPANTS")
    if env is not None:
        env = env.strip()
        if env.lower() in {"", "none", "null", "full"}:
            return None
        return int(env)
    return load_config().get("selective_audit", {}).get("max_train_participants", None)


def _audit_cache_suffix(max_train) -> str:
    return "full" if max_train is None else f"mt{int(max_train)}"


def _softmax_entropy(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(np.asarray(prob, dtype=float), 1e-12, 1.0)
    return -np.sum(prob * np.log(prob), axis=1)


def _prob_margin(prob: np.ndarray) -> np.ndarray:
    prob = np.asarray(prob, dtype=float)
    if prob.shape[1] == 1:
        return np.ones(prob.shape[0])
    part = np.partition(prob, -2, axis=1)
    return part[:, -1] - part[:, -2]


def _score_distance_to_boundary(scale: str, scores: np.ndarray) -> np.ndarray:
    cutoffs = np.asarray(load_config()["labels"][scale]["cutoffs"][1:-1], dtype=float)
    if len(cutoffs) == 0:
        return np.zeros(len(scores), dtype=float)
    return np.min(np.abs(np.asarray(scores).reshape(-1, 1) - cutoffs.reshape(1, -1)), axis=1)


def _selected_scale_counts(selected: Dict[int, List[str]], ids: List[int]) -> pd.DataFrame:
    rows = []
    for pid in ids:
        items = selected[int(pid)]
        rec = {"participant_id": int(pid)}
        for sc in SCALE_ORDER:
            n = sum(1 for item in items if item.startswith(f"{sc}_"))
            rec[f"{sc}_n_scale_items"] = int(n)
            rec[f"{sc}_scale_item_fraction"] = float(n / SCALES[sc])
        rows.append(rec)
    return pd.DataFrame(rows)


def make_decision_audit_frame(
    long: pd.DataFrame,
    split: str,
    budget: int = 19,
    policy: str = "response_only_uncertainty",
    variant: str = "responses",
) -> pd.DataFrame:
    """Build a participant-by-scale decision-audit frame for a short-form policy.

    Output rows are short-form decisions, not raw item rows. Every feature is
    available from the observed prefix used by the reduced screen. The complete
    questionnaire-derived label is appended only as a reference outcome.

    The training split is processed in chunks. This avoids a pathological slow
    path in wide-pivot construction for very large participant batches while
    leaving the selected item policy, labels, and probabilities unchanged.
    """
    ensure_dirs()
    ids_all = list(map(int, load_ids(split)))
    max_train = _audit_max_train_participants()
    if split == "train" and max_train is not None and len(ids_all) > int(max_train):
        rng = np.random.default_rng(load_config()["seed"] + 4242)
        ids_all = sorted(map(int, rng.choice(np.asarray(ids_all, dtype=int), size=int(max_train), replace=False)))
    sigma_map = get_calibration_sigmas(policy, budget, variant, long)
    targets_all = make_targets_frame().set_index("participant_id")
    chunk_size = int(load_config().get("selective_audit", {}).get("audit_frame_chunk_size", 2500))
    chunks: List[pd.DataFrame] = []
    for start_i in range(0, len(ids_all), chunk_size):
        ids = ids_all[start_i:start_i + chunk_size]
        selected = simulate_item_selection(policy, budget, ids, long=long, seed=load_config()["seed"] + budget)
        pred, prob, state = predict_from_selected(ids, selected, variant=variant, long=long, sigma_map=sigma_map)
        scores, _ = _scores_from_selected(ids, selected, variant, long)
        targets = targets_all.loc[ids]
        counts = _selected_scale_counts(selected, ids).set_index("participant_id")
        st = state.set_index("participant_id").reindex(ids).fillna(0)

        rows: List[dict] = []
        for sc in SCALE_ORDER:
            y = targets[f"{sc}_severity_idx"].astype(int).values
            p = np.asarray(pred[sc], dtype=int)
            pr = np.asarray(prob[sc], dtype=float)
            conf = pr[np.arange(len(ids)), p]
            entropy = _softmax_entropy(pr)
            margin = _prob_margin(pr)
            high = int(load_config()["labels"][sc]["high_risk_index"])
            dist = _score_distance_to_boundary(sc, scores[sc])
            for i, pid in enumerate(ids):
                wrong = bool(p[i] != y[i])
                high_risk_fn = bool((y[i] >= high) and (p[i] < high))
                rows.append(
                    {
                        "participant_id": int(pid),
                        "split": split,
                        "scale": sc,
                        "scale_label": SCALE_NAMES[sc],
                        "true_label": int(y[i]),
                        "short_pred": int(p[i]),
                        "issued_confidence": float(conf[i]),
                        "entropy": float(entropy[i]),
                        "margin": float(margin[i]),
                        "score_estimate": float(scores[sc][i]),
                        "score_distance": float(dist[i]),
                        "wrong": int(wrong),
                        "high_risk_fn": int(high_risk_fn),
                        "true_high_risk": int(y[i] >= high),
                        "pred_high_risk": int(p[i] >= high),
                        "reliability": float(st.loc[pid].get("reliability", 1.0)),
                        "fast": float(st.loc[pid].get("fast", 0.0)),
                        "long_pause": float(st.loc[pid].get("long", 0.0)),
                        "std": float(st.loc[pid].get("std", 0.0)),
                        "mean_rt": float(st.loc[pid].get("mean_rt", 0.0)),
                        "abs_mean_rt": float(abs(st.loc[pid].get("mean_rt", 0.0))),
                        "posterior_entropy": float(st.loc[pid].get("posterior_entropy", 0.0)),
                        "scale_coverage": float(st.loc[pid].get("scale_coverage", 1.0)),
                        "n_scale_items": int(counts.loc[pid, f"{sc}_n_scale_items"]),
                        "scale_item_fraction": float(counts.loc[pid, f"{sc}_scale_item_fraction"]),
                    }
                )
        chunks.append(pd.DataFrame(rows))
    return pd.concat(chunks, ignore_index=True)

def _overconfidence_threshold(val: pd.DataFrame) -> float:
    q = float(load_config().get("selective_audit", {}).get("overconfidence_quantile", 0.75))
    wrong = val[val["wrong"].astype(int) == 1]
    # Validation-selected threshold for the subset of errors that were issued
    # with unusually high confidence.  Falling back to all decisions keeps the
    # function defined if a tiny validation split has no errors.
    base = wrong if len(wrong) else val
    return float(base["issued_confidence"].quantile(q))


def _add_overconfident_flag(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["overconfident_wrong"] = ((out["wrong"].astype(int) == 1) & (out["issued_confidence"] >= threshold)).astype(int)
    return out


def _feature_matrix(train: pd.DataFrame, test: pd.DataFrame, features: List[str], shuffle_rt: bool = False, seed: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr = train.copy()
    te = test.copy()
    if shuffle_rt:
        rng = np.random.default_rng(seed)
        for col in RT_FEATURES:
            if col in tr.columns:
                tr[col] = rng.permutation(tr[col].values)
            if col in te.columns:
                te[col] = rng.permutation(te[col].values)
    cols = [c for c in features if c in tr.columns]
    xtr = tr[cols + ["scale"]].copy()
    xte = te[cols + ["scale"]].copy()
    xtr = pd.get_dummies(xtr, columns=["scale"], drop_first=False)
    xte = pd.get_dummies(xte, columns=["scale"], drop_first=False)
    xte = xte.reindex(columns=xtr.columns, fill_value=0)
    return xtr.replace([np.inf, -np.inf], 0).fillna(0), xte.replace([np.inf, -np.inf], 0).fillna(0)


def _topk_metrics(y: np.ndarray, score: np.ndarray, high_risk_fn: np.ndarray, top_frac: float = 0.10) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    high_risk_fn = np.asarray(high_risk_fn).astype(int)
    k = max(1, int(math.ceil(len(y) * top_frac)))
    order = np.argsort(-score)[:k]
    return {
        "recall_at_top10pct": float(y[order].sum() / max(1, y.sum())),
        "precision_at_top10pct": float(y[order].mean()),
        "high_risk_fn_capture_at_top10pct": float(high_risk_fn[order].sum() / max(1, high_risk_fn.sum())),
    }


def _score_model(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    features: List[str],
    shuffle_rt: bool = False,
    seed: int = 0,
) -> Tuple[np.ndarray, float, object | None]:
    c_grid = load_config().get("selective_audit", {}).get("error_detector_C_grid", [0.1, 1.0, 10.0])
    y_train = train[target_col].astype(int).values
    y_val = val[target_col].astype(int).values
    if len(np.unique(y_train)) < 2:
        return np.repeat(float(y_train.mean()) if len(y_train) else 0.0, len(test)), float("nan"), None
    best_c, best_ap, best_model = float(c_grid[0]), -1.0, None
    for c in c_grid:
        xtr, xval = _feature_matrix(train, val, features, shuffle_rt=shuffle_rt, seed=seed + int(float(c) * 1000))
        model = make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(C=float(c), class_weight="balanced", max_iter=1000, solver="liblinear", random_state=seed),
        )
        model.fit(xtr, y_train)
        val_score = model.predict_proba(xval)[:, 1]
        ap = average_precision_score(y_val, val_score) if len(np.unique(y_val)) > 1 else 0.0
        if ap > best_ap:
            best_c, best_ap, best_model = float(c), float(ap), model
    xtrv, xte = _feature_matrix(pd.concat([train, val], ignore_index=True), test, features, shuffle_rt=shuffle_rt, seed=seed + 17)
    y_trv = pd.concat([train, val], ignore_index=True)[target_col].astype(int).values
    final = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(C=best_c, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=seed),
    )
    final.fit(xtrv, y_trv)
    score = final.predict_proba(xte)[:, 1]
    return score, best_c, final


def _evaluate_scores(y: np.ndarray, score: np.ndarray, high_risk_fn: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    out = {}
    if len(np.unique(y)) > 1:
        out["pr_auc"] = float(average_precision_score(y, score))
        out["roc_auc"] = float(roc_auc_score(y, score))
    else:
        out["pr_auc"] = float("nan")
        out["roc_auc"] = float("nan")
    out.update(_topk_metrics(y, score, high_risk_fn, top_frac=0.10))
    return out


def _naive_score(df: pd.DataFrame, kind: str) -> np.ndarray:
    if kind == "confidence_only":
        return (1.0 - df["issued_confidence"].values) + df["entropy"].values + (1.0 - df["margin"].values)
    if kind == "rt_only":
        return (1.0 - df["reliability"].values) + df["fast"].values + df["long_pause"].values + 0.25 * df["std"].values + 0.25 * df["abs_mean_rt"].values
    if kind == "score_coverage_only":
        return (1.0 / (1.0 + df["score_distance"].values)) + (1.0 - df["scale_item_fraction"].values)
    raise ValueError(kind)


def _prepare_audit_frames(long: pd.DataFrame, budget: int = 19) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    cache_dir = root_path("data/processed")
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = _audit_cache_suffix(_audit_max_train_participants())
    cache_paths = {sp: cache_dir / f"decision_audit_frame_{sp}_b{budget}_{suffix}.pkl" for sp in ["train", "val", "test"]}
    threshold_path = cache_dir / f"decision_audit_overconfidence_threshold_b{budget}_{suffix}.txt"
    if all(p.exists() for p in cache_paths.values()) and threshold_path.exists() and os.environ.get("NO_AUDIT_CACHE", "0") != "1":
        train = pd.read_pickle(cache_paths["train"])
        val = pd.read_pickle(cache_paths["val"])
        test = pd.read_pickle(cache_paths["test"])
        threshold = float(threshold_path.read_text().strip())
        return train, val, test, threshold
    train = make_decision_audit_frame(long, "train", budget=budget)
    val = make_decision_audit_frame(long, "val", budget=budget)
    test = make_decision_audit_frame(long, "test", budget=budget)
    threshold = _overconfidence_threshold(val)
    train = _add_overconfident_flag(train, threshold)
    val = _add_overconfident_flag(val, threshold)
    test = _add_overconfident_flag(test, threshold)
    train.to_pickle(cache_paths["train"])
    val.to_pickle(cache_paths["val"])
    test.to_pickle(cache_paths["test"])
    threshold_path.write_text(str(threshold))
    return train, val, test, threshold


def run_shortform_error_profile(long: pd.DataFrame) -> pd.DataFrame:
    budget = int(load_config().get("selective_audit", {}).get("budget", 19))
    train, val, test, threshold = _prepare_audit_frames(long, budget)
    test.to_csv(TABLE_DIR / "decision_audit_frame_test_b19.csv", index=False)
    rows = []
    for sc, g in test.groupby("scale", sort=False):
        rows.append(
            {
                "scale": SCALE_NAMES[sc],
                "n_decisions": int(len(g)),
                "shortform_error_rate": float(g["wrong"].mean()),
                "n_errors": int(g["wrong"].sum()),
                "high_risk_fn_rate": float(g["high_risk_fn"].mean()),
                "n_high_risk_fn": int(g["high_risk_fn"].sum()),
                "overconfident_threshold": threshold,
                "overconfident_wrong_rate": float(g["overconfident_wrong"].mean()),
                "n_overconfident_wrong": int(g["overconfident_wrong"].sum()),
                "mean_issued_confidence": float(g["issued_confidence"].mean()),
                "mean_entropy": float(g["entropy"].mean()),
                "mean_reliability": float(g["reliability"].mean()),
            }
        )
    rows.append(
        {
            "scale": "Mean / all scale-decisions",
            "n_decisions": int(len(test)),
            "shortform_error_rate": float(test["wrong"].mean()),
            "n_errors": int(test["wrong"].sum()),
            "high_risk_fn_rate": float(test["high_risk_fn"].mean()),
            "n_high_risk_fn": int(test["high_risk_fn"].sum()),
            "overconfident_threshold": threshold,
            "overconfident_wrong_rate": float(test["overconfident_wrong"].mean()),
            "n_overconfident_wrong": int(test["overconfident_wrong"].sum()),
            "mean_issued_confidence": float(test["issued_confidence"].mean()),
            "mean_entropy": float(test["entropy"].mean()),
            "mean_reliability": float(test["reliability"].mean()),
        }
    )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "table15_shortform_error_profile.csv", index=False)
    return out


def run_shortform_error_detection(long: pd.DataFrame) -> pd.DataFrame:
    budget = int(load_config().get("selective_audit", {}).get("budget", 19))
    train, val, test, threshold = _prepare_audit_frames(long, budget)
    model_specs = [
        ("confidence_only", CONF_FEATURES, False, None),
        ("score_coverage_only", SCORE_FEATURES, False, None),
        ("rt_only", RT_FEATURES, False, None),
        ("confidence_plus_rt", CONF_FEATURES + RT_FEATURES, False, None),
        ("all_observed_prefix", ALL_FEATURES, False, None),
        ("shuffled_rt_placebo", CONF_FEATURES + RT_FEATURES, True, None),
    ]
    rows = []
    pr_curves: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {"wrong": {}, "overconfident_wrong": {}}
    for target_col in ["wrong", "overconfident_wrong"]:
        y_test = test[target_col].astype(int).values
        hfn = test["high_risk_fn"].astype(int).values
        for name, features, shuffle, _ in model_specs:
            score, c, _model = _score_model(train, val, test, target_col, features, shuffle_rt=shuffle, seed=load_config()["seed"] + 250)
            metrics = _evaluate_scores(y_test, score, hfn)
            rows.append(
                {
                    "target": target_col,
                    "model": name,
                    "selected_C": c,
                    "overconfidence_threshold": threshold,
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "test_positive_n": int(y_test.sum()),
                    **metrics,
                }
            )
            if name in {"confidence_only", "rt_only", "confidence_plus_rt", "shuffled_rt_placebo"} and len(np.unique(y_test)) > 1:
                precision, recall, _ = precision_recall_curve(y_test, score)
                pr_curves[target_col][name] = (recall, precision)
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "table16_error_detection_models.csv", index=False)

    if pr_curves["wrong"] or pr_curves["overconfident_wrong"]:
        # Supplemental PR-curve diagnostic.  The manuscript tables carry the
        # primary result, while the code release keeps both ordinary-error and
        # overconfident-error curves for inspection.
        for target_col, title in [("wrong", "Short-form error detection"), ("overconfident_wrong", "Overconfident-error detection")]:
            if not pr_curves[target_col]:
                continue
            plt.figure(figsize=(6.2, 4.2))
            for name, (recall, precision) in pr_curves[target_col].items():
                plt.plot(recall, precision, label=name.replace("_", " "))
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"{title} at 19 items")
            plt.legend(fontsize=8)
            plt.tight_layout()
            suffix = "ordinary" if target_col == "wrong" else "ocw"
            plt.savefig(FIG_DIR / f"supplement_error_detection_pr_curves_{suffix}.png", dpi=300)
            plt.savefig(FIG_DIR / f"supplement_error_detection_pr_curves_{suffix}.pdf")
            plt.close()
    return out


def _participant_scores_from_rows(df: pd.DataFrame, row_score: np.ndarray, how: str = "max") -> pd.Series:
    tmp = df[["participant_id"]].copy()
    tmp["score"] = np.asarray(row_score, dtype=float)
    if how == "mean":
        return tmp.groupby("participant_id")["score"].mean()
    return tmp.groupby("participant_id")["score"].max()


def _post_escalation_metrics(test: pd.DataFrame, escalated: Iterable[int], strategy: str, rate: float) -> Dict[str, float]:
    escalated = set(map(int, escalated))
    df = test.copy()
    df["escalated"] = df["participant_id"].astype(int).isin(escalated).astype(int)
    df["post_pred"] = np.where(df["escalated"] == 1, df["true_label"], df["short_pred"])
    f1s, hrrs = [], []
    for sc, g in df.groupby("scale", sort=False):
        f1s.append(float(f1_score(g["true_label"].astype(int), g["post_pred"].astype(int), average="macro", zero_division=0)))
        high = int(load_config()["labels"][sc]["high_risk_index"])
        hrrs.append(float(recall_score(g["true_label"].astype(int) >= high, g["post_pred"].astype(int) >= high, zero_division=0)))
    n_participants = int(test["participant_id"].nunique())
    n_escalated = len(escalated)
    added_items = n_escalated * (37 - 19)
    errors_captured = int(((df["wrong"] == 1) & (df["escalated"] == 1)).sum())
    hfn_captured = int(((df["high_risk_fn"] == 1) & (df["escalated"] == 1)).sum())
    ocw_captured = int(((df["overconfident_wrong"] == 1) & (df["escalated"] == 1)).sum())
    total_errors = int(df["wrong"].sum())
    total_hfn = int(df["high_risk_fn"].sum())
    total_ocw = int(df["overconfident_wrong"].sum())
    return {
        "escalation_rate": float(rate),
        "strategy": strategy,
        "n_participants": n_participants,
        "n_escalated": n_escalated,
        "avg_items_used": float(19 + (n_escalated / n_participants) * (37 - 19)),
        "added_items_total": int(added_items),
        "errors_captured": errors_captured,
        "error_capture_rate": float(errors_captured / max(1, total_errors)),
        "high_risk_fn_captured": hfn_captured,
        "high_risk_fn_capture_rate": float(hfn_captured / max(1, total_hfn)),
        "overconfident_wrong_captured": ocw_captured,
        "overconfident_wrong_capture_rate": float(ocw_captured / max(1, total_ocw)),
        "post_macro_f1_mean": float(np.mean(f1s)),
        "post_high_risk_recall_mean": float(np.mean(hrrs)),
        "remaining_errors": int(total_errors - errors_captured),
        "remaining_high_risk_fn": int(total_hfn - hfn_captured),
        "remaining_overconfident_wrong": int(total_ocw - ocw_captured),
        "errors_corrected_per_100_added_items": float(100 * errors_captured / max(1, added_items)),
        "high_risk_fn_corrected_per_100_added_items": float(100 * hfn_captured / max(1, added_items)),
    }


def run_selective_fullform_escalation(long: pd.DataFrame) -> pd.DataFrame:
    cfg = load_config().get("selective_audit", {})
    budget = int(cfg.get("budget", 19))
    rates = [float(x) for x in cfg.get("escalation_rates", [0.05, 0.10, 0.20, 0.30])]
    random_repeats = int(cfg.get("random_repeats", 200))
    train, val, test, _threshold = _prepare_audit_frames(long, budget)

    score_err_detector, _, _ = _score_model(train, val, test, "wrong", CONF_FEATURES + RT_FEATURES, shuffle_rt=False, seed=load_config()["seed"] + 310)
    score_ocw_detector, _, _ = _score_model(train, val, test, "overconfident_wrong", CONF_FEATURES + RT_FEATURES, shuffle_rt=False, seed=load_config()["seed"] + 312)
    score_hfn_detector, _, _ = _score_model(train, val, test, "high_risk_fn", CONF_FEATURES + RT_FEATURES, shuffle_rt=False, seed=load_config()["seed"] + 313)
    score_shuffled, _, _ = _score_model(train, val, test, "wrong", CONF_FEATURES + RT_FEATURES, shuffle_rt=True, seed=load_config()["seed"] + 311)
    score_map = {
        "confidence_only": _participant_scores_from_rows(test, _naive_score(test, "confidence_only")),
        "rt_only": _participant_scores_from_rows(test, _naive_score(test, "rt_only")),
        "rt_plus_conf_error": _participant_scores_from_rows(test, score_err_detector),
        "rt_plus_conf_ocw": _participant_scores_from_rows(test, score_ocw_detector),
        "rt_plus_conf_hfn": _participant_scores_from_rows(test, score_hfn_detector),
        "shuffled_rt_placebo": _participant_scores_from_rows(test, score_shuffled),
        "oracle": test.groupby("participant_id")[["wrong", "high_risk_fn", "overconfident_wrong"]].apply(lambda g: float(g["wrong"].sum() + 2.0 * g["high_risk_fn"].sum() + 1.5 * g["overconfident_wrong"].sum())),
    }
    participants = np.asarray(sorted(test["participant_id"].unique()), dtype=int)
    rows = []
    rng = np.random.default_rng(load_config()["seed"] + 7777)
    for rate in rates:
        k = max(1, int(math.ceil(len(participants) * rate)))
        # Random escalation averaged over repeated draws.
        rand_rows = []
        for _ in range(random_repeats):
            chosen = rng.choice(participants, size=k, replace=False)
            rand_rows.append(_post_escalation_metrics(test, chosen, "random", rate))
        rand_df = pd.DataFrame(rand_rows)
        mean_row = rand_df.mean(numeric_only=True).to_dict()
        mean_row.update({"strategy": "random", "random_repeats": random_repeats})
        for col in ["errors_captured", "high_risk_fn_captured", "overconfident_wrong_captured", "post_macro_f1_mean", "remaining_overconfident_wrong"]:
            mean_row[f"{col}_sd"] = float(rand_df[col].std(ddof=0))
        rows.append(mean_row)
        # Deterministic strategies.
        for name, scores in score_map.items():
            scores = scores.reindex(participants).fillna(-np.inf)
            chosen = scores.sort_values(ascending=False).head(k).index.astype(int).tolist()
            rows.append(_post_escalation_metrics(test, chosen, name, rate))
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "table17_selective_fullform_escalation.csv", index=False)

    strategy_labels = {
        "random": "Random selection",
        "confidence_only": "Lowest confidence",
        "rt_only": "RT process score",
        "rt_plus_conf_error": "Confidence + RT (any error)",
        "rt_plus_conf_ocw": "Confidence + RT (overconfident error)",
        "rt_plus_conf_hfn": "Confidence + RT (elevated-score false negative)",
        "shuffled_rt_placebo": "Confidence + shuffled RT",
        "oracle": "Oracle upper bound",
    }
    plt.figure(figsize=(8.2, 5.0))
    for strategy, g in out.groupby("strategy", sort=False):
        g = g.sort_values("avg_items_used")
        plt.plot(
            g["avg_items_used"],
            100 * g["error_capture_rate"],
            marker="o",
            label=strategy_labels[strategy],
        )
    plt.xlabel("Average number of items completed")
    plt.ylabel("Short-form errors captured (%)")
    plt.title("Selective full-form escalation frontier")
    plt.legend(fontsize=8, ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure7_escalation_burden_frontier.png", dpi=300)
    plt.savefig(
        FIG_DIR / "figure7_escalation_burden_frontier.tiff",
        dpi=300,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.savefig(FIG_DIR / "figure7_escalation_burden_frontier.pdf")
    plt.close()
    return out


def _ece_fixed_from_rows(g: pd.DataFrame) -> float:
    conf = g["issued_confidence"].astype(float).values
    correct = 1 - g["wrong"].astype(int).values
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - conf[mask].mean()))
    return float(ece)


def _assign_process_strata(df: pd.DataFrame, thresholds: Dict[str, float], shuffle: bool = False, seed: int = 0) -> pd.Series:
    tmp = df[["participant_id", "reliability", "fast", "long_pause", "std", "mean_rt", "abs_mean_rt"]].drop_duplicates("participant_id").copy()
    if shuffle:
        rng = np.random.default_rng(seed)
        for col in ["reliability", "fast", "long_pause", "std", "mean_rt", "abs_mean_rt"]:
            tmp[col] = rng.permutation(tmp[col].values)
    def one(r):
        if r["fast"] >= thresholds["fast_q75"] and r["std"] >= thresholds["std_q75"]:
            return "fast_inconsistent"
        if r["long_pause"] >= thresholds["long_q75"] or r["abs_mean_rt"] >= thresholds["absmean_q90"]:
            return "slow_interrupted"
        if r["reliability"] >= thresholds["reliability_q75"] and r["std"] <= thresholds["std_q25"] and r["fast"] <= thresholds["fast_q50"] and r["long_pause"] <= thresholds["long_q50"]:
            return "stable_typical"
        return "mixed_unstable"
    return tmp.set_index("participant_id").apply(one, axis=1)


def run_process_quality_strata(long: pd.DataFrame) -> pd.DataFrame:
    budget = int(load_config().get("selective_audit", {}).get("budget", 19))
    train, val, test, _threshold = _prepare_audit_frames(long, budget)
    ptrain = train.drop_duplicates("participant_id")
    thresholds = {
        "fast_q75": float(ptrain["fast"].quantile(0.75)),
        "fast_q50": float(ptrain["fast"].quantile(0.50)),
        "long_q75": float(ptrain["long_pause"].quantile(0.75)),
        "long_q50": float(ptrain["long_pause"].quantile(0.50)),
        "std_q75": float(ptrain["std"].quantile(0.75)),
        "std_q25": float(ptrain["std"].quantile(0.25)),
        "reliability_q75": float(ptrain["reliability"].quantile(0.75)),
        "absmean_q90": float(ptrain["abs_mean_rt"].quantile(0.90)),
    }
    rows = []
    for source, shuffle in [("true_rt", False), ("shuffled_rt_placebo", True)]:
        strata = _assign_process_strata(test, thresholds, shuffle=shuffle, seed=load_config()["seed"] + 490)
        df = test.copy()
        df["process_quality_stratum"] = df["participant_id"].map(strata).fillna("mixed_unstable")
        for group, g in df.groupby("process_quality_stratum", sort=False):
            n_p = int(g["participant_id"].nunique())
            rows.append(
                {
                    "stratum_source": source,
                    "process_quality_stratum": group,
                    "n_participants": n_p,
                    "n_scale_decisions": int(len(g)),
                    "shortform_error_rate": float(g["wrong"].mean()),
                    "high_risk_fn_rate": float(g["high_risk_fn"].mean()),
                    "overconfident_error_rate": float(g["overconfident_wrong"].mean()),
                    "ece_l": _ece_fixed_from_rows(g),
                    "mean_confidence": float(g["issued_confidence"].mean()),
                    "mean_reliability": float(g["reliability"].mean()),
                    "errors_corrected_per_100_escalated_participants": float(100 * g["wrong"].sum() / max(1, n_p)),
                    "high_risk_fn_corrected_per_100_escalated_participants": float(100 * g["high_risk_fn"].sum() / max(1, n_p)),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "table18_process_quality_strata.csv", index=False)

    fig_df = out[out["stratum_source"] == "true_rt"].copy()
    if len(fig_df):
        label_map = {
            "fast_inconsistent": "Fast-inconsistent",
            "slow_interrupted": "Slow/interrupted",
            "stable_typical": "Stable-typical",
            "mixed_unstable": "Mixed pattern",
        }
        order = ["fast_inconsistent", "slow_interrupted", "stable_typical", "mixed_unstable"]
        fig_df = fig_df.set_index("process_quality_stratum").loc[order].reset_index()
        plt.figure(figsize=(7.2, 4.8))
        bars = plt.bar(
            [label_map[value] for value in fig_df["process_quality_stratum"]],
            100 * fig_df["shortform_error_rate"],
        )
        plt.ylabel("Short-form error rate (%)")
        plt.title("Short-form error rate by observed-prefix process quality")
        for bar, value in zip(bars, 100 * fig_df["shortform_error_rate"]):
            plt.text(bar.get_x() + bar.get_width() / 2, value + 0.4, f"{value:.1f}%", ha="center")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "figure8_process_quality_error_rates.png", dpi=300)
        plt.savefig(
            FIG_DIR / "figure8_process_quality_error_rates.tiff",
            dpi=300,
            pil_kwargs={"compression": "tiff_lzw"},
        )
        plt.savefig(FIG_DIR / "figure8_process_quality_error_rates.pdf")
        plt.close()
    return out


def run_cross_scale_transfer(long: pd.DataFrame) -> pd.DataFrame:
    budget = int(load_config().get("selective_audit", {}).get("budget", 19))
    train, val, test, _threshold = _prepare_audit_frames(long, budget)
    rows = []
    specs = [
        ("confidence_only", CONF_FEATURES, False),
        ("confidence_plus_rt", CONF_FEATURES + RT_FEATURES, False),
        ("shuffled_rt_placebo", CONF_FEATURES + RT_FEATURES, True),
    ]
    for holdout in SCALE_ORDER:
        tr = train[train["scale"] != holdout].copy()
        va = val[val["scale"] != holdout].copy()
        te = test[test["scale"] == holdout].copy()
        for name, feats, shuffle in specs:
            score, c, _ = _score_model(tr, va, te, "wrong", feats, shuffle_rt=shuffle, seed=load_config()["seed"] + 810 + SCALE_ORDER.index(holdout))
            metrics = _evaluate_scores(te["wrong"].astype(int).values, score, te["high_risk_fn"].astype(int).values)
            rows.append(
                {
                    "heldout_scale": SCALE_NAMES[holdout],
                    "train_scales": "+".join(SCALE_NAMES[s] for s in SCALE_ORDER if s != holdout),
                    "model": name,
                    "selected_C": c,
                    "n_test": int(len(te)),
                    "test_error_rate": float(te["wrong"].mean()),
                    **metrics,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "table19_cross_scale_transfer.csv", index=False)
    return out




def _bootstrap_capture_ci(test: pd.DataFrame, score_series: pd.Series, rate: float, seed: int, repeats: int = 300) -> Dict[str, float]:
    """Participant bootstrap CI for deterministic escalation capture rates."""
    rng = np.random.default_rng(seed)
    participants = np.asarray(sorted(test["participant_id"].unique()), dtype=int)
    participant_outcomes = (
        test.groupby("participant_id")[["wrong", "overconfident_wrong", "high_risk_fn"]]
        .sum()
        .reindex(participants)
        .fillna(0.0)
    )
    scores = score_series.reindex(participants).fillna(-np.inf).to_numpy(dtype=float)
    wrong = participant_outcomes["wrong"].to_numpy(dtype=float)
    ocw = participant_outcomes["overconfident_wrong"].to_numpy(dtype=float)
    hfn = participant_outcomes["high_risk_fn"].to_numpy(dtype=float)
    k = max(1, int(math.ceil(len(participants) * rate)))
    vals = {"error_capture_rate": [], "overconfident_wrong_capture_rate": [], "high_risk_fn_capture_rate": []}
    for _ in range(repeats):
        sampled_idx = rng.integers(0, len(participants), size=len(participants))
        sampled_scores = scores[sampled_idx]
        if k >= len(sampled_idx):
            chosen_pos = np.arange(len(sampled_idx))
        else:
            chosen_pos = np.argpartition(sampled_scores, -k)[-k:]
        swrong = wrong[sampled_idx]
        socw = ocw[sampled_idx]
        shfn = hfn[sampled_idx]
        vals["error_capture_rate"].append(float(swrong[chosen_pos].sum() / max(1.0, swrong.sum())))
        vals["overconfident_wrong_capture_rate"].append(float(socw[chosen_pos].sum() / max(1.0, socw.sum())))
        vals["high_risk_fn_capture_rate"].append(float(shfn[chosen_pos].sum() / max(1.0, shfn.sum())))
    out = {}
    for key, arr in vals.items():
        arr = np.asarray(arr, dtype=float)
        out[f"{key}_ci_low"] = float(np.nanquantile(arr, 0.025))
        out[f"{key}_ci_high"] = float(np.nanquantile(arr, 0.975))
    return out


def _bootstrap_random_capture_ci(test: pd.DataFrame, rate: float, seed: int, repeats: int = 300) -> Dict[str, float]:
    """Participant bootstrap CI with a fresh random escalation draw per replicate."""
    rng = np.random.default_rng(seed)
    participants = np.asarray(sorted(test["participant_id"].unique()), dtype=int)
    outcomes = (
        test.groupby("participant_id")[["wrong", "overconfident_wrong", "high_risk_fn"]]
        .sum().reindex(participants).fillna(0.0)
    )
    arrays = {
        "error_capture_rate": outcomes["wrong"].to_numpy(dtype=float),
        "overconfident_wrong_capture_rate": outcomes["overconfident_wrong"].to_numpy(dtype=float),
        "high_risk_fn_capture_rate": outcomes["high_risk_fn"].to_numpy(dtype=float),
    }
    k = max(1, int(math.ceil(len(participants) * rate)))
    vals = {key: [] for key in arrays}
    for _ in range(repeats):
        sampled_idx = rng.integers(0, len(participants), size=len(participants))
        chosen_pos = rng.choice(len(sampled_idx), size=k, replace=False)
        for key, source in arrays.items():
            sampled = source[sampled_idx]
            vals[key].append(float(sampled[chosen_pos].sum() / max(1.0, sampled.sum())))
    out = {}
    for key, arr in vals.items():
        out[f"{key}_ci_low"] = float(np.quantile(arr, 0.025))
        out[f"{key}_ci_high"] = float(np.quantile(arr, 0.975))
    return out


def run_escalation_bootstrap_ci(long: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap uncertainty for the principal deterministic escalation rankings."""
    cfg = load_config().get("selective_audit", {})
    budget = int(cfg.get("budget", 19))
    train, val, test, _threshold = _prepare_audit_frames(long, budget)
    rate = 0.20
    score_err, _, _ = _score_model(train, val, test, "wrong", CONF_FEATURES + RT_FEATURES, shuffle_rt=False, seed=load_config()["seed"] + 310)
    score_ocw, _, _ = _score_model(train, val, test, "overconfident_wrong", CONF_FEATURES + RT_FEATURES, shuffle_rt=False, seed=load_config()["seed"] + 312)
    scores = {
        "confidence_only": _participant_scores_from_rows(test, _naive_score(test, "confidence_only")),
        "rt_only": _participant_scores_from_rows(test, _naive_score(test, "rt_only")),
        "rt_plus_conf_error": _participant_scores_from_rows(test, score_err),
        "rt_plus_conf_ocw": _participant_scores_from_rows(test, score_ocw),
    }
    rows = []
    table17_path = TABLE_DIR / "table17_selective_fullform_escalation.csv"
    if table17_path.exists():
        table17 = pd.read_csv(table17_path)
        random_base = table17[
            (table17["escalation_rate"].astype(float).sub(rate).abs() < 1e-12)
            & (table17["strategy"] == "random")
        ]
        if len(random_base) == 1:
            base = random_base.iloc[0].to_dict()
            ci = _bootstrap_random_capture_ci(
                test, rate, seed=load_config()["seed"] + 919,
                repeats=int(cfg.get("bootstrap_repeats", 300)),
            )
            rows.append({**base, **ci})
    for i, (name, score_series) in enumerate(scores.items()):
        participants = np.asarray(sorted(test["participant_id"].unique()), dtype=int)
        k = max(1, int(math.ceil(len(participants) * rate)))
        chosen = score_series.reindex(participants).fillna(-np.inf).sort_values(ascending=False).head(k).index.astype(int).tolist()
        base = _post_escalation_metrics(test, chosen, name, rate)
        ci = _bootstrap_capture_ci(test, score_series, rate, seed=load_config()["seed"] + 920 + i, repeats=int(cfg.get("bootstrap_repeats", 300)))
        rows.append({**base, **ci})
    out = pd.DataFrame(rows)
    if table17_path.exists():
        reference = pd.read_csv(table17_path)
        shared = reference[
            (reference["escalation_rate"].astype(float).sub(rate).abs() < 1e-12)
            & reference["strategy"].isin(scores)
        ].set_index("strategy")
        metric_cols = [
            "errors_captured", "error_capture_rate",
            "high_risk_fn_captured", "high_risk_fn_capture_rate",
            "overconfident_wrong_captured", "overconfident_wrong_capture_rate",
        ]
        for _, row in out.iterrows():
            strategy = row["strategy"]
            if strategy not in shared.index:
                continue
            for metric in metric_cols:
                if not np.isclose(float(row[metric]), float(shared.loc[strategy, metric]), rtol=0, atol=1e-12):
                    raise RuntimeError(
                        f"table20 base metric mismatch with table17 for {strategy}: "
                        f"{metric}={row[metric]} versus {shared.loc[strategy, metric]}"
                    )
    out.to_csv(TABLE_DIR / "table20_escalation_bootstrap_ci.csv", index=False)
    return out


def run_ocw_threshold_sensitivity(long: pd.DataFrame) -> pd.DataFrame:
    """Sensitivity of the overconfident-error result to validation-selected confidence cutoffs."""
    budget = int(load_config().get("selective_audit", {}).get("budget", 19))
    train0, val0, test0, primary_threshold = _prepare_audit_frames(long, budget)
    primary_q = float(load_config().get("selective_audit", {}).get("overconfidence_quantile", 0.75))
    rows = []
    for q in [0.70, 0.75, 0.80, 0.90]:
        wrong_val = val0[val0["wrong"].astype(int) == 1]
        base = wrong_val if len(wrong_val) else val0
        if abs(float(q) - primary_q) < 1e-12:
            threshold = float(primary_threshold)
        else:
            threshold = float(base["issued_confidence"].quantile(q))
        train = _add_overconfident_flag(train0, threshold)
        val = _add_overconfident_flag(val0, threshold)
        test = _add_overconfident_flag(test0, threshold)
        y = test["overconfident_wrong"].astype(int).values
        for name, feats, shuffle in [
            ("confidence_only", CONF_FEATURES, False),
            ("rt_only", RT_FEATURES, False),
            ("confidence_plus_rt", CONF_FEATURES + RT_FEATURES, False),
            ("shuffled_rt_placebo", CONF_FEATURES + RT_FEATURES, True),
        ]:
            score, c, _ = _score_model(train, val, test, "overconfident_wrong", feats, shuffle_rt=shuffle, seed=load_config()["seed"] + 1040 + int(q * 100))
            metrics = _evaluate_scores(y, score, test["high_risk_fn"].astype(int).values)
            rows.append({
                "overconfidence_quantile": q,
                "threshold": threshold,
                "model": name,
                "selected_C": c,
                "test_positive_n": int(y.sum()),
                **metrics,
            })
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "table21_ocw_threshold_sensitivity.csv", index=False)
    return out


def run_per_scale_selective_escalation(long: pd.DataFrame) -> pd.DataFrame:
    """Per-scale 20% escalation yields to diagnose whether pooled gains are PSS-driven."""
    budget = int(load_config().get("selective_audit", {}).get("budget", 19))
    train, val, test, _threshold = _prepare_audit_frames(long, budget)
    rate = 0.20
    score_ocw, _, _ = _score_model(train, val, test, "overconfident_wrong", CONF_FEATURES + RT_FEATURES, shuffle_rt=False, seed=load_config()["seed"] + 312)
    base_scores = {
        "confidence_only": _naive_score(test, "confidence_only"),
        "rt_only": _naive_score(test, "rt_only"),
        "rt_plus_conf_ocw": score_ocw,
    }
    rows = []
    for sc, g in test.groupby("scale", sort=False):
        idx = g.index.values
        n = len(g)
        k = max(1, int(math.ceil(n * rate)))
        for name, raw_score in base_scores.items():
            sg = g.copy()
            sg["audit_score"] = np.asarray(raw_score)[idx]
            chosen_idx = set(sg.sort_values("audit_score", ascending=False).head(k).index)
            sg["escalated_scale_decision"] = sg.index.isin(chosen_idx).astype(int)
            rows.append({
                "scale": SCALE_NAMES[sc],
                "strategy": name,
                "escalation_rate": rate,
                "n_decisions": int(n),
                "n_errors": int(sg["wrong"].sum()),
                "n_ocw": int(sg["overconfident_wrong"].sum()),
                "n_high_risk_fn": int(sg["high_risk_fn"].sum()),
                "error_capture_rate": float(((sg["wrong"] == 1) & (sg["escalated_scale_decision"] == 1)).sum() / max(1, sg["wrong"].sum())),
                "overconfident_wrong_capture_rate": float(((sg["overconfident_wrong"] == 1) & (sg["escalated_scale_decision"] == 1)).sum() / max(1, sg["overconfident_wrong"].sum())),
                "high_risk_fn_capture_rate": float(((sg["high_risk_fn"] == 1) & (sg["escalated_scale_decision"] == 1)).sum() / max(1, sg["high_risk_fn"].sum())),
            })
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "table22_per_scale_selective_escalation.csv", index=False)
    return out


def run_training_sensitivity(long: pd.DataFrame) -> pd.DataFrame:
    """Compact-subsample versus full-train detector sensitivity for the main audit signal."""
    # This routine is optional for compact runs.  It honors AUDIT_MAX_TRAIN_PARTICIPANTS.
    budget = int(load_config().get("selective_audit", {}).get("budget", 19))
    rows = []
    original = os.environ.get("AUDIT_MAX_TRAIN_PARTICIPANTS")
    for label, cap in [("compact_5000_participants", "5000"), ("full_train", "none")]:
        try:
            os.environ["AUDIT_MAX_TRAIN_PARTICIPANTS"] = cap
            train, val, test, threshold = _prepare_audit_frames(long, budget)
            for target_col in ["wrong", "overconfident_wrong"]:
                score, c, _ = _score_model(train, val, test, target_col, CONF_FEATURES + RT_FEATURES, shuffle_rt=False, seed=load_config()["seed"] + 1500)
                metrics = _evaluate_scores(test[target_col].astype(int).values, score, test["high_risk_fn"].astype(int).values)
                rows.append({
                    "training_setting": label,
                    "target": target_col,
                    "n_train_rows": int(len(train)),
                    "selected_C": c,
                    "overconfidence_threshold": threshold,
                    **metrics,
                })
        finally:
            if original is None:
                os.environ.pop("AUDIT_MAX_TRAIN_PARTICIPANTS", None)
            else:
                os.environ["AUDIT_MAX_TRAIN_PARTICIPANTS"] = original
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "table23_training_sensitivity.csv", index=False)
    return out

def run_all_selective_auditing(long: pd.DataFrame | None = None) -> Dict[str, pd.DataFrame]:
    ensure_dirs()
    if long is None:
        long = fit_rt_residuals()
    outputs = {
        "error_profile": run_shortform_error_profile(long),
        "error_detection": run_shortform_error_detection(long),
        "selective_escalation": run_selective_fullform_escalation(long),
        "escalation_bootstrap_ci": run_escalation_bootstrap_ci(long),
        "ocw_threshold_sensitivity": run_ocw_threshold_sensitivity(long),
        "per_scale_selective_escalation": run_per_scale_selective_escalation(long),
        "process_quality_strata": run_process_quality_strata(long),
        "cross_scale_transfer": run_cross_scale_transfer(long),
    }
    if load_config().get("selective_audit", {}).get("run_full_train_sensitivity", False) or os.environ.get("RUN_FULL_TRAIN_SENSITIVITY", "0") == "1":
        outputs["training_sensitivity"] = run_training_sensitivity(long)
    return outputs


if __name__ == "__main__":
    long = fit_rt_residuals()
    run_all_selective_auditing(long)
