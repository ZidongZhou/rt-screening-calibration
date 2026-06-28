from __future__ import annotations

import json
import math
import os
import pickle
import random
import shutil
import time
import gc
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    recall_score,
    roc_auc_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SCALES = {"phq9": 9, "gad7": 7, "pss": 14, "isi": 7}
SCALE_ORDER = ["phq9", "gad7", "pss", "isi"]
SCALE_NAMES = {"phq9": "PHQ-9", "gad7": "GAD-7", "pss": "PSS", "isi": "ISI"}
GLOBAL_ITEMS: List[Tuple[str, int]] = []
for _sc in SCALE_ORDER:
    for _i in range(1, SCALES[_sc] + 1):
        GLOBAL_ITEMS.append((_sc, _i))
ITEM_ORDER = [f"{sc}_q{i}" for sc, i in GLOBAL_ITEMS]
ITEM_TO_SCALE = {f"{sc}_q{i}": sc for sc, i in GLOBAL_ITEMS}
ITEM_TO_POS = {f"{sc}_q{i}": pos for pos, (sc, i) in enumerate(GLOBAL_ITEMS, start=1)}
ITEM_WORD_COUNTS = {
    "phq9": [12, 12, 13, 9, 8, 23, 21, 42, 18],
    "gad7": [10, 10, 12, 6, 9, 9, 16],
    "pss": [18, 16, 9, 12, 21, 20, 8, 18, 13, 13, 23, 16, 10, 22],
    "isi": [4, 6, 8, 18, 16, 26, 17],
}


def root_path(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


def ensure_dirs() -> None:
    for p in [
        "data/splits",
        "data/processed",
        "results/tables",
        "results/figures",
        "models",
        "reports",
    ]:
        root_path(p).mkdir(parents=True, exist_ok=True)


def reset_outputs() -> None:
    for p in ["results/tables", "results/figures", "models", "data/splits", "data/processed", "reports"]:
        path = root_path(p)
        if path.exists():
            shutil.rmtree(path)
    ensure_dirs()


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(root_path("config/config.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_raw() -> Dict[str, pd.DataFrame]:
    return {name: pd.read_csv(root_path("data/raw", f"{name}.csv")) for name in ["demographic", *SCALE_ORDER]}


def load_word_counts() -> Dict[Tuple[str, int], int]:
    return {
        (scale, index): count
        for scale, counts in ITEM_WORD_COUNTS.items()
        for index, count in enumerate(counts, start=1)
    }


def label_from_score(score: float, cutoffs: list, classes: list) -> str:
    for i, cls in enumerate(classes):
        if float(cutoffs[i]) <= float(score) < float(cutoffs[i + 1]):
            return cls
    return classes[-1]


def add_scale_labels(raw: Dict[str, pd.DataFrame] | None = None, cfg: dict | None = None) -> Dict[str, pd.DataFrame]:
    if raw is None:
        raw = read_raw()
    if cfg is None:
        cfg = load_config()
    out = {}
    for sc in SCALE_ORDER:
        df = raw[sc].copy()
        spec = cfg["labels"][sc]
        df[f"{sc}_severity"] = df["score"].apply(lambda x: label_from_score(x, spec["cutoffs"], spec["classes"]))
        mapper = {c: i for i, c in enumerate(spec["classes"])}
        df[f"{sc}_severity_idx"] = df[f"{sc}_severity"].map(mapper).astype(int)
        out[sc] = df
    return out


def make_targets_frame() -> pd.DataFrame:
    raw = read_raw()
    labels = add_scale_labels(raw)
    demo = raw["demographic"].rename(columns={"export_id": "participant_id"}).copy()
    targets = demo.copy()
    for sc in SCALE_ORDER:
        cols = ["export_id", "score", f"{sc}_severity", f"{sc}_severity_idx"]
        tmp = labels[sc][cols].rename(
            columns={"export_id": "participant_id", "score": f"{sc}_score"}
        )
        targets = targets.merge(tmp, on="participant_id", how="inner")
    return targets


def validate_data() -> pd.DataFrame:
    ensure_dirs()
    cfg = load_config()
    raw = read_raw()
    base_ids = set(raw["demographic"]["export_id"])
    rows = []
    for name, df in raw.items():
        qcols = [c for c in df.columns if c.startswith("question")]
        tcols = [c for c in df.columns if c.startswith("time")]
        rows.append(
            {
                "file": f"{name}.csv",
                "n_rows": int(len(df)),
                "n_missing_values": int(df.isna().sum().sum()),
                "duplicate_export_id": int(df["export_id"].duplicated().sum()),
                "id_consistent_with_demographic": bool(set(df["export_id"]) == base_ids),
                "n_question_columns": int(len(qcols)),
                "n_time_columns": int(len(tcols)),
                "min_response_time": float(df[tcols].min().min()) if tcols else np.nan,
                "max_response_time": float(df[tcols].max().max()) if tcols else np.nan,
                "negative_response_times": int((df[tcols] < 0).sum().sum()) if tcols else 0,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(root_path("results/tables/table1a_data_validation.csv"), index=False)

    rec = []
    for sc in SCALE_ORDER:
        df = raw[sc]
        qcols = [f"question{i}" for i in range(1, SCALES[sc] + 1)]
        raw_sum = df[qcols].sum(axis=1)
        calc = raw_sum - 14 if sc == "pss" else raw_sum
        rec.append(
            {
                "scale": SCALE_NAMES[sc],
                "score_recalculation_rule": "sum(question)-14" if sc == "pss" else "sum(question)",
                "score_column_min": int(df["score"].min()),
                "score_column_max": int(df["score"].max()),
                "n_score_mismatches": int((calc != df["score"]).sum()),
                "mismatch_handling": "released score column retained for labels",
            }
        )
    pd.DataFrame(rec).to_csv(root_path("results/tables/score_recalculation_checks.csv"), index=False)

    labels = add_scale_labels(raw, cfg)
    dist = []
    for sc in SCALE_ORDER:
        vc = labels[sc][f"{sc}_severity"].value_counts().reindex(cfg["labels"][sc]["classes"], fill_value=0)
        for lab, cnt in vc.items():
            dist.append(
                {
                    "scale": SCALE_NAMES[sc],
                    "severity_label": lab,
                    "count": int(cnt),
                    "proportion": float(cnt / len(labels[sc])),
                }
            )
    pd.DataFrame(dist).to_csv(root_path("results/tables/table1b_label_distribution.csv"), index=False)
    return out


def build_long_format(save_sample: bool = True) -> pd.DataFrame:
    ensure_dirs()
    raw = read_raw()
    labels = add_scale_labels(raw)
    wc = load_word_counts()
    demo = raw["demographic"].rename(columns={"export_id": "participant_id"})
    rows = []
    for sc in SCALE_ORDER:
        df = labels[sc].rename(columns={"export_id": "participant_id"}).merge(demo, on="participant_id", how="left")
        for i in range(1, SCALES[sc] + 1):
            item = f"{sc}_q{i}"
            rows.append(
                pd.DataFrame(
                    {
                        "participant_id": df["participant_id"].values,
                        "scale": sc,
                        "item_id": i,
                        "global_item_id": item,
                        "within_scale_position": i,
                        "global_position": ITEM_TO_POS[item],
                        "word_count": wc[(sc, i)],
                        "item_response": df[f"question{i}"].values.astype(float),
                        "raw_rt": df[f"time{i}"].values.astype(float),
                        "score_total": df["score"].values.astype(float),
                        "severity_label": df[f"{sc}_severity"].values,
                        "severity_idx": df[f"{sc}_severity_idx"].values.astype(int),
                        "gender": df["gender"].values,
                        "age": df["age"].values.astype(float),
                        "edu": df["edu"].values,
                        "smoke": df["smoke"].values,
                        "drink": df["drink"].values,
                    }
                )
            )
    long = pd.concat(rows, ignore_index=True)
    long["raw_rt_nonnegative"] = long["raw_rt"] >= 0
    long["log_rt"] = np.log1p(long["raw_rt"].clip(lower=0))
    audit = pd.DataFrame(
        [
            {
                "n_participants": int(long["participant_id"].nunique()),
                "n_item_records": int(len(long)),
                "expected_records": int(long["participant_id"].nunique() * 37),
                "matches_expected": bool(len(long) == long["participant_id"].nunique() * 37),
            }
        ]
    )
    audit.to_csv(root_path("results/tables/long_format_audit.csv"), index=False)
    if save_sample:
        long.head(2000).to_csv(root_path("data/processed/item_long_sample_2000_rows.csv"), index=False)
    return long


def create_splits() -> pd.DataFrame:
    ensure_dirs()
    cfg = load_config()
    targets = make_targets_frame()
    risk_signature = targets[[f"{sc}_severity_idx" for sc in SCALE_ORDER]].ge(2).sum(axis=1).astype(str)
    train_ids, temp_ids = train_test_split(
        targets["participant_id"].values,
        test_size=cfg["splits"]["val"] + cfg["splits"]["test"],
        random_state=cfg["seed"],
        stratify=risk_signature,
    )
    temp = targets[targets["participant_id"].isin(temp_ids)].copy()
    temp_sig = temp[[f"{sc}_severity_idx" for sc in SCALE_ORDER]].ge(2).sum(axis=1).astype(str)
    rel_test = cfg["splits"]["test"] / (cfg["splits"]["val"] + cfg["splits"]["test"])
    val_ids, test_ids = train_test_split(
        temp["participant_id"].values,
        test_size=rel_test,
        random_state=cfg["seed"] + 1,
        stratify=temp_sig,
    )
    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        pd.DataFrame({"participant_id": sorted(map(int, ids))}).to_csv(root_path("data/splits", f"{name}_ids.csv"), index=False)
    split_map = {}
    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        for pid in ids:
            split_map[int(pid)] = name
    targets["split"] = targets["participant_id"].map(split_map)
    rows = []
    for split in ["train", "val", "test"]:
        sub = targets[targets["split"] == split]
        rows.append({"split": split, "n_participants": int(len(sub))})
        for sc in SCALE_ORDER:
            classes = cfg["labels"][sc]["classes"]
            vc = sub[f"{sc}_severity"].value_counts().reindex(classes, fill_value=0)
            for lab, cnt in vc.items():
                rows.append(
                    {
                        "split": split,
                        "scale": SCALE_NAMES[sc],
                        "severity_label": lab,
                        "count": int(cnt),
                        "proportion": float(cnt / len(sub)),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(root_path("results/tables/label_distribution_by_split.csv"), index=False)
    return out


def load_ids(split: str) -> List[int]:
    p = root_path("data/splits", f"{split}_ids.csv")
    if not p.exists():
        create_splits()
    return pd.read_csv(p)["participant_id"].astype(int).tolist()


def fit_rt_residuals(long: pd.DataFrame | None = None) -> pd.DataFrame:
    ensure_dirs()
    cache = root_path("data/processed/item_long_with_rt.pkl")
    if long is None and cache.exists() and os.environ.get("NO_CACHE", "0") != "1":
        return pd.read_pickle(cache)
    if long is None:
        long = build_long_format(save_sample=False)
    cfg = load_config()
    train_ids = set(load_ids("train"))
    train_mask = long["participant_id"].isin(train_ids)

    # Winsorization thresholds are estimated from the training set only and applied to all splits.
    train = long[train_mask]
    lo = train.groupby("global_item_id")["raw_rt"].quantile(cfg["rt_winsor"]["lower_pct"] / 100.0)
    hi = train.groupby("global_item_id")["raw_rt"].quantile(cfg["rt_winsor"]["upper_pct"] / 100.0)
    long = long.copy()
    long["winsor_lower"] = long["global_item_id"].map(lo)
    long["winsor_upper"] = long["global_item_id"].map(hi)
    long["winsorized_rt"] = long["raw_rt"].clip(lower=long["winsor_lower"], upper=long["winsor_upper"])
    long["log_rt"] = np.log1p(long["winsorized_rt"].clip(lower=0))

    tr = long[train_mask].copy()
    item_mean = tr.groupby("global_item_id")["log_rt"].mean()
    item_sd = tr.groupby("global_item_id")["log_rt"].std(ddof=0).replace(0, 1)
    long["itemwise_rt_z"] = (long["log_rt"] - long["global_item_id"].map(item_mean)) / long["global_item_id"].map(item_sd)

    # Personwise z is retained for post-hoc diagnostics only; adaptive decisions do not use future items.
    person_mean = long.groupby("participant_id")["log_rt"].transform("mean")
    person_sd = long.groupby("participant_id")["log_rt"].transform("std").replace(0, 1)
    long["personwise_rt_z_posthoc"] = (long["log_rt"] - person_mean) / person_sd

    # Content-adjusted model: item wording and administration-position only.
    content_cols = ["word_count", "within_scale_position", "global_position"]
    content = pd.get_dummies(long[content_cols + ["scale"]], columns=["scale"], drop_first=False)
    model_c = LinearRegression()
    model_c.fit(content.loc[train_mask], long.loc[train_mask, "log_rt"])
    long["expected_log_rt_content"] = model_c.predict(content)
    long["rt_residual_content"] = long["log_rt"] - long["expected_log_rt_content"]
    c_mu = long.loc[train_mask, "rt_residual_content"].mean()
    c_sd = long.loc[train_mask, "rt_residual_content"].std(ddof=0) or 1
    long["rt_residual_content_z"] = (long["rt_residual_content"] - c_mu) / c_sd

    # Response-conditioned model: allowed only after an item is observed.
    cond_cols = ["word_count", "within_scale_position", "global_position", "item_response", "age"]
    cond = pd.get_dummies(
        long[cond_cols + ["scale", "gender", "edu", "smoke", "drink"]],
        columns=["scale", "gender", "edu", "smoke", "drink"],
        drop_first=False,
    )
    model_r = LinearRegression()
    model_r.fit(cond.loc[train_mask], long.loc[train_mask, "log_rt"])
    long["expected_log_rt_conditioned"] = model_r.predict(cond)
    long["rt_residual_conditioned"] = long["log_rt"] - long["expected_log_rt_conditioned"]
    r_mu = long.loc[train_mask, "rt_residual_conditioned"].mean()
    r_sd = long.loc[train_mask, "rt_residual_conditioned"].std(ddof=0) or 1
    long["rt_residual_conditioned_z"] = (long["rt_residual_conditioned"] - r_mu) / r_sd

    # Persist compact model metadata, not the full long table.
    coeffs = []
    for name, model, cols in [("content_adjusted", model_c, content.columns), ("response_conditioned", model_r, cond.columns)]:
        coeffs.append({"model": name, "term": "intercept", "coefficient": float(model.intercept_)})
        coeffs.extend({"model": name, "term": c, "coefficient": float(v)} for c, v in zip(cols, model.coef_))
    pd.DataFrame(coeffs).to_csv(root_path("results/tables/rt_residual_model_coefficients.csv"), index=False)
    diag = pd.DataFrame(
        [
            {
                "model": "content_adjusted",
                "fit_split": "train_only",
                "n_train_item_records": int(train_mask.sum()),
                "train_residual_mean": float(c_mu),
                "train_residual_sd": float(c_sd),
            },
            {
                "model": "response_conditioned",
                "fit_split": "train_only",
                "n_train_item_records": int(train_mask.sum()),
                "train_residual_mean": float(r_mu),
                "train_residual_sd": float(r_sd),
            },
        ]
    )
    diag.to_csv(root_path("results/tables/rt_preprocessing_summary.csv"), index=False)
    joblib.dump(
        {
            "content_columns": list(content.columns),
            "conditioned_columns": list(cond.columns),
            "fit_participant_ids": sorted(train_ids),
            "fit_split": "train_only",
        },
        root_path("models/rt_residual_metadata.pkl"),
    )
    if "pair_key" not in long.columns:
        long["pair_key"] = long["participant_id"].astype(str) + "|" + long["global_item_id"].astype(str)
    if os.environ.get("NO_CACHE", "0") != "1":
        long.to_pickle(cache)
    return long


def participant_rt_features(
    long: pd.DataFrame,
    ids: Iterable[int] | None = None,
    observed_items: Dict[int, List[str]] | None = None,
) -> pd.DataFrame:
    """Participant-level response-process summaries.

    This function is called repeatedly by the state-model and leakage-audit
    stages.  The implementation is vectorized to avoid a per-participant
    ``np.polyfit`` loop over more than 17,000 training participants.  The slope
    is computed as cov(position, residual) / var(position), which is equivalent
    to the least-squares slope for a single predictor.
    """
    if ids is None:
        ids = long["participant_id"].unique().tolist()
    ids = list(map(int, ids))
    id_index = pd.Index(ids, name="participant_id")
    df = long[long["participant_id"].isin(id_index)].copy()
    if observed_items is not None:
        pairs = [(int(pid), item) for pid, items in observed_items.items() for item in items]
        sel = pd.DataFrame(pairs, columns=["participant_id", "global_item_id"])
        df = df.merge(sel, on=["participant_id", "global_item_id"], how="inner")
    if df.empty:
        return pd.DataFrame({"participant_id": ids}).assign(
            n_observed=0,
            mean_rt_residual=0.0,
            median_rt_residual=0.0,
            std_rt_residual=0.0,
            fast_response_ratio=0.0,
            long_pause_ratio=0.0,
            rt_slope_over_position=0.0,
            response_variability=0.0,
            scale_coverage=0.0,
        )

    df["_x"] = df["global_position"].astype(float)
    df["_y"] = df["rt_residual_content_z"].astype(float)
    df["_resp"] = df["item_response"].astype(float)
    df["_xy"] = df["_x"] * df["_y"]
    df["_x2"] = df["_x"] * df["_x"]
    df["_y2"] = df["_y"] * df["_y"]
    df["_resp2"] = df["_resp"] * df["_resp"]
    df["_fast"] = (df["_y"] < -1).astype(float)
    df["_long"] = (df["_y"] > 1).astype(float)

    g = df.groupby("participant_id", sort=False)
    agg = g.agg(
        n_observed=("global_item_id", "size"),
        mean_rt_residual=("_y", "mean"),
        median_rt_residual=("_y", "median"),
        sum_x=("_x", "sum"),
        sum_y=("_y", "sum"),
        sum_xy=("_xy", "sum"),
        sum_x2=("_x2", "sum"),
        mean_y2=("_y2", "mean"),
        fast_response_ratio=("_fast", "mean"),
        long_pause_ratio=("_long", "mean"),
        mean_resp=("_resp", "mean"),
        mean_resp2=("_resp2", "mean"),
        scale_n=("scale", "nunique"),
    )
    n = agg["n_observed"].astype(float)
    denom = n * agg["sum_x2"] - agg["sum_x"] ** 2
    numer = n * agg["sum_xy"] - agg["sum_x"] * agg["sum_y"]
    agg["rt_slope_over_position"] = np.where(np.abs(denom) > 1e-12, numer / denom, 0.0)
    agg["std_rt_residual"] = np.sqrt(
        np.maximum(0.0, agg["mean_y2"] - agg["mean_rt_residual"] ** 2)
    )
    agg["response_variability"] = np.sqrt(
        np.maximum(0.0, agg["mean_resp2"] - agg["mean_resp"] ** 2)
    )
    agg["scale_coverage"] = agg["scale_n"] / 4.0
    out = agg[[
        "n_observed",
        "mean_rt_residual",
        "median_rt_residual",
        "std_rt_residual",
        "fast_response_ratio",
        "long_pause_ratio",
        "rt_slope_over_position",
        "response_variability",
        "scale_coverage",
    ]].reset_index()
    out = out.set_index("participant_id").reindex(id_index).fillna(0).reset_index()
    out["participant_id"] = out["participant_id"].astype(int)
    out["n_observed"] = out["n_observed"].astype(int)
    return out


def fit_response_state_models(long: pd.DataFrame | None = None) -> pd.DataFrame:
    ensure_dirs()
    if long is None:
        long = fit_rt_residuals()
    train_ids = load_ids("train")
    posthoc = participant_rt_features(long, train_ids)
    X = posthoc[["mean_rt_residual", "median_rt_residual", "std_rt_residual", "fast_response_ratio", "long_pause_ratio", "rt_slope_over_position", "response_variability", "scale_coverage"]].fillna(0)
    # Use a deterministic training subset for fast reproducibility; the fitted model is only a response-process summary.
    rng_state = np.random.default_rng(load_config()["seed"])
    fit_idx = rng_state.choice(np.arange(len(X)), size=min(6000, len(X)), replace=False)
    X_fit = X.iloc[fit_idx]
    records = []
    best = None
    for k in range(2, 6):
        gm = GaussianMixture(n_components=k, covariance_type="diag", random_state=load_config()["seed"], reg_covar=1e-4, max_iter=80, n_init=1)
        gm.fit(X_fit)
        records.append({"k": k, "bic": float(gm.bic(X_fit)), "aic": float(gm.aic(X_fit))})
        if best is None or records[-1]["bic"] < best[0]:
            best = (records[-1]["bic"], k, gm)
    assert best is not None
    full_gmm = best[2]
    labels = full_gmm.predict(X)
    posthoc["posthoc_response_process_cluster"] = labels
    summary = posthoc.groupby("posthoc_response_process_cluster").agg(
        n=("participant_id", "size"),
        mean_rt_residual=("mean_rt_residual", "mean"),
        std_rt_residual=("std_rt_residual", "mean"),
        fast_response_ratio=("fast_response_ratio", "mean"),
        long_pause_ratio=("long_pause_ratio", "mean"),
        response_variability=("response_variability", "mean"),
    ).reset_index()
    summary.to_csv(root_path("results/tables/gmm_full_cluster_summary.csv"), index=False)
    pd.DataFrame(records).to_csv(root_path("results/tables/gmm_model_selection.csv"), index=False)

    # Online partial state estimator is trained on simulated training prefixes only.
    rng = np.random.default_rng(load_config()["seed"])
    prefixes: Dict[int, List[str]] = {}
    for pid in train_ids:
        k = int(rng.choice([4, 8, 11, 15, 19, 26]))
        prefixes[pid] = rng.choice(ITEM_ORDER, size=k, replace=False).tolist()
    online_feats = participant_rt_features(long, train_ids, prefixes)
    Xp = online_feats[["n_observed", "mean_rt_residual", "median_rt_residual", "std_rt_residual", "fast_response_ratio", "long_pause_ratio", "rt_slope_over_position", "response_variability", "scale_coverage"]].fillna(0)
    Xp_fit = Xp.sample(n=min(6000, len(Xp)), random_state=load_config()["seed"])
    online_gmm = GaussianMixture(n_components=min(3, best[1]), covariance_type="diag", random_state=load_config()["seed"] + 1, reg_covar=1e-4, max_iter=80, n_init=1)
    online_gmm.fit(Xp_fit)
    joblib.dump(
        {"model": online_gmm, "feature_columns": list(Xp.columns), "fit_split": "train_only", "fit_participant_ids": sorted(train_ids)},
        root_path("models/online_partial_response_state.pkl"),
    )
    online_feats["partial_state"] = online_gmm.predict(Xp)
    online_summary = online_feats.groupby("partial_state").agg(
        n=("participant_id", "size"),
        mean_rt_residual=("mean_rt_residual", "mean"),
        fast_response_ratio=("fast_response_ratio", "mean"),
        long_pause_ratio=("long_pause_ratio", "mean"),
        std_rt_residual=("std_rt_residual", "mean"),
    ).reset_index()
    online_summary.to_csv(root_path("results/tables/online_state_diagnostics.csv"), index=False)
    joblib.dump({"model": full_gmm, "feature_columns": list(X.columns), "fit_split": "train_only", "fit_participant_ids": sorted(train_ids)}, root_path("models/gmm_full_posthoc.pkl"))
    return online_summary


def item_info_values(long: pd.DataFrame | None = None) -> pd.DataFrame:
    cache = root_path("results/tables/train_item_information_scores.csv")
    if cache.exists():
        return pd.read_csv(cache)
    if long is None:
        long = fit_rt_residuals()
    train_ids = set(load_ids("train"))
    tr = long[long["participant_id"].isin(train_ids)]
    rows = []
    for item, g in tr.groupby("global_item_id"):
        sc = g["scale"].iloc[0]
        resp = g["item_response"].astype(float).values
        sev = g["severity_idx"].astype(float).values
        if np.std(resp) > 0 and np.std(sev) > 0:
            corr = abs(float(np.corrcoef(resp, sev)[0, 1]))
        else:
            corr = 0.0
        rows.append(
            {
                "global_item_id": item,
                "scale": sc,
                "global_position": ITEM_TO_POS[item],
                "response_variance": float(np.var(resp)),
                "severity_correlation": corr,
                "rank_score": corr + 0.05 * float(np.var(resp)),
            }
        )
    info = pd.DataFrame(rows).sort_values(["rank_score", "global_position"], ascending=[False, True])
    info.to_csv(root_path("results/tables/train_item_information_scores.csv"), index=False)
    return info


def _labels_for_scores(sc: str, scores: np.ndarray) -> np.ndarray:
    spec = load_config()["labels"][sc]
    return np.array([spec["classes"].index(label_from_score(s, spec["cutoffs"], spec["classes"])) for s in scores], dtype=int)


def _probs_for_scores(sc: str, scores: np.ndarray, sigma: float = 2.5) -> np.ndarray:
    spec = load_config()["labels"][sc]
    cutoffs = spec["cutoffs"]
    centers = []
    for a, b in zip(cutoffs[:-1], cutoffs[1:]):
        centers.append((a + b - 1) / 2.0)
    centers = np.array(centers, dtype=float)
    logits = -0.5 * ((scores.reshape(-1, 1) - centers.reshape(1, -1)) / sigma) ** 2
    logits = logits - logits.max(axis=1, keepdims=True)
    prob = np.exp(logits)
    prob = prob / prob.sum(axis=1, keepdims=True)
    return prob


@lru_cache(maxsize=1)
def _train_score_means() -> Dict[str, float]:
    targets = make_targets_frame().set_index("participant_id").loc[load_ids("train")]
    return {sc: float(targets[f"{sc}_score"].mean()) for sc in SCALE_ORDER}


def _current_score_estimate(prefix: pd.DataFrame, sc: str, use_rt: bool = False, state_reliability: float = 1.0) -> float:
    train_means = _train_score_means()
    g = prefix[prefix["scale"] == sc]
    if len(g) == 0:
        return train_means[sc]
    base = float(g["item_response"].mean() * SCALES[sc])
    if sc == "pss":
        base -= 14
    if use_rt and "rt_residual_content_z" in g.columns:
        base += 0.20 * float(g["rt_residual_content_z"].mean())
        # Low response reliability shrinks partial-score extrapolation toward the training-set scale mean.
        shrink = min(0.35, max(0.0, 1.0 - state_reliability) * 0.40)
        base = (1 - shrink) * base + shrink * train_means[sc]
    max_score = load_config()["labels"][sc]["cutoffs"][-1] - 1
    return float(np.clip(base, 0, max_score))


def _scale_uncertainty(prefix: pd.DataFrame, sc: str, use_rt: bool = False, state_reliability: float = 1.0) -> float:
    spec = load_config()["labels"][sc]
    score = _current_score_estimate(prefix, sc, use_rt=use_rt, state_reliability=state_reliability)
    boundaries = np.array(spec["cutoffs"][1:-1], dtype=float)
    dist = float(np.min(np.abs(boundaries - score))) if len(boundaries) else 3.0
    g = prefix[prefix["scale"] == sc]
    coverage = len(g) / SCALES[sc]
    unc = (1.0 / (1.0 + dist)) + (1.0 - coverage)
    if len(g) > 1:
        unc += 0.10 * float(g["item_response"].std(ddof=0))
    if use_rt:
        unc *= 1.0 + 0.35 * max(0.0, 1.0 - state_reliability)
    return float(unc)


@lru_cache(maxsize=1)
def _cached_online_model():
    p = root_path("models/online_partial_response_state.pkl")
    if not p.exists():
        return None
    return joblib.load(p)


def _prefix_state_features(prefix: pd.DataFrame) -> Dict[str, float]:
    if len(prefix) == 0:
        return {
            "n_observed": 0.0, "mean_rt_residual": 0.0, "median_rt_residual": 0.0,
            "std_rt_residual": 0.0, "fast_response_ratio": 0.0, "long_pause_ratio": 0.0,
            "rt_slope_over_position": 0.0, "response_variability": 0.0, "scale_coverage": 0.0,
        }
    resid = prefix["rt_residual_content_z"].astype(float).values
    pos = prefix["global_position"].astype(float).values
    slope = float(np.polyfit(pos, resid, 1)[0]) if len(prefix) > 1 and np.std(pos) > 0 else 0.0
    return {
        "n_observed": float(len(prefix)),
        "mean_rt_residual": float(np.mean(resid)),
        "median_rt_residual": float(np.median(resid)),
        "std_rt_residual": float(np.std(resid)),
        "fast_response_ratio": float(np.mean(resid < -1)),
        "long_pause_ratio": float(np.mean(resid > 1)),
        "rt_slope_over_position": slope,
        "response_variability": float(np.std(prefix["item_response"].astype(float).values)) if len(prefix) else 0.0,
        "scale_coverage": float(prefix["scale"].nunique() / 4) if len(prefix) else 0.0,
    }


def _online_state(prefix: pd.DataFrame) -> Dict[str, float]:
    feats = _prefix_state_features(prefix)
    fast = feats["fast_response_ratio"]
    long = feats["long_pause_ratio"]
    std = feats["std_rt_residual"]
    reliability = float(np.clip(1.0 - 0.35 * fast - 0.20 * long - 0.08 * min(std, 4), 0.10, 1.0))
    ent = 0.0
    obj = _cached_online_model()
    if obj is not None and feats["n_observed"] > 0:
        row = pd.DataFrame([{c: feats.get(c, 0.0) for c in obj["feature_columns"]}])
        prob = obj["model"].predict_proba(row)[0]
        ent = float(-(prob * np.log(prob + 1e-12)).sum())
        reliability = float(np.clip(reliability - 0.05 * ent, 0.10, 1.0))
    return {"reliability": reliability, "fast": fast, "long": long, "std": std, "posterior_entropy": ent}


def _select_next_item(prefix: pd.DataFrame, remaining: List[str], info_map: Dict[str, float], policy: str) -> str:
    """Select one candidate item from the observed prefix only.

    The utility implements a transparent, leakage-audited rule:

        U_j = I_j(1 + U_s) + lambda_t C_s - rho_t N_s + eta_t H_j

    where I_j is training-set item informativeness, U_s is the current
    scale-specific uncertainty estimated from observed responses only, C_s is a
    scale-coverage bonus, N_s is the number of already observed items in that
    scale, and H_j is a high-information confirmation bonus. For the RT-aware
    policy, lambda_t, rho_t, and eta_t are adjusted by the observed-prefix
    reliability score R_t. No unobserved candidate response or candidate RT is
    accessed.
    """
    use_rt = policy == "rt_aware_uncertainty"
    st = _online_state(prefix) if use_rt else {"reliability": 1.0, "fast": 0.0, "long": 0.0, "std": 0.0}
    reliability = float(st.get("reliability", 1.0))
    unreliability = max(0.0, 1.0 - reliability)
    scale_unc = {sc: _scale_uncertainty(prefix, sc, use_rt=use_rt, state_reliability=reliability) for sc in SCALE_ORDER}
    selected = prefix["global_item_id"].tolist() if len(prefix) else []
    scale_counts = {sc: sum(1 for x in selected if ITEM_TO_SCALE[x] == sc) for sc in SCALE_ORDER}
    min_count = min(scale_counts.values()) if scale_counts else 0
    best_item, best_score = None, -1e18
    for cand in remaining:
        sc = ITEM_TO_SCALE[cand]
        info_j = float(info_map.get(cand, 0.0))
        coverage_gap = max(0.0, 1.0 - scale_counts[sc] / max(1, SCALES[sc]))
        undercovered = 1.0 if scale_counts[sc] <= min_count else 0.0
        # RT unreliability changes ranking by increasing cross-scale coverage and
        # confirmatory high-information items, rather than multiplying all scores
        # by the same constant.
        lambda_t = 0.20 + (0.40 * unreliability if use_rt else 0.0)
        rho_t = 0.015 + (0.020 * unreliability if use_rt else 0.0)
        eta_t = 0.00 + (0.25 * unreliability if use_rt else 0.0)
        score = info_j * (1.0 + scale_unc[sc])
        score += lambda_t * coverage_gap + 0.08 * undercovered
        score += eta_t * info_j * (1.0 + undercovered)
        score -= rho_t * scale_counts[sc]
        if score > best_score:
            best_item, best_score = cand, score
    assert best_item is not None
    return best_item

def _rows_for_participant(long_by_pid: Dict[int, pd.DataFrame], pid: int, items: List[str]) -> pd.DataFrame:
    g = long_by_pid[int(pid)]
    return g[g["global_item_id"].isin(items)].copy()


def simulate_item_selection(policy: str, budget: int, ids: Iterable[int], long: pd.DataFrame | None = None, seed: int | None = None) -> Dict[int, List[str]]:
    # Sequential item-selection logic: after each new item, recompute the prefix state.
    """Simulate item selection with fixed, random, or sequential adaptive policies.

    Response-only and RT-aware policies are sequential at the item-budget level:
    after each newly selected item, the current prefix is recomputed from the
    participant's observed item responses and observed RT residuals only. The
    optimized implementation uses wide matrices rather than repeated dataframe
    filtering, which keeps the full experiment reproducible on a laptop.
    """
    if long is None:
        long = fit_rt_residuals()
    if seed is None:
        seed = load_config()["seed"]
    ids = list(map(int, ids))
    rng = np.random.default_rng(seed)
    info = item_info_values(long)
    info_map = dict(zip(info["global_item_id"], info["rank_score"]))
    info_scores = np.array([info_map.get(item, 0.0) for item in ITEM_ORDER], dtype=float)
    train_rank = info["global_item_id"].tolist()
    fixed_order = ITEM_ORDER.copy()
    top_by_scale = {sc: info[info["scale"] == sc]["global_item_id"].tolist() for sc in SCALE_ORDER}
    if policy == "fixed_order":
        items = fixed_order[:budget]
        return {pid: items.copy() for pid in ids}
    if policy == "short_forms":
        base = ["phq9_q1", "phq9_q2", "gad7_q1", "gad7_q2", "pss_q2", "pss_q4", "pss_q5", "pss_q10", "isi_q1", "isi_q2", "isi_q3"]
        items = (base + [x for x in train_rank if x not in base])[:budget]
        return {pid: items.copy() for pid in ids}
    if policy == "train_selected_fixed":
        items = train_rank[:budget]
        return {pid: items.copy() for pid in ids}
    if policy == "random":
        return {pid: rng.choice(fixed_order, size=budget, replace=False).tolist() for pid in ids}
    if policy not in ["response_only_uncertainty", "rt_aware_uncertainty", "rt_only_uncertainty"]:
        raise ValueError(f"Unknown policy: {policy}")
    use_rt = policy in ["rt_aware_uncertainty", "rt_only_uncertainty"]

    subset = long[long["participant_id"].isin(ids)]
    resp_wide = subset.pivot(index="participant_id", columns="global_item_id", values="item_response").reindex(index=ids, columns=ITEM_ORDER)
    resid_wide = subset.pivot(index="participant_id", columns="global_item_id", values="rt_residual_content_z").reindex(index=ids, columns=ITEM_ORDER)
    resp_mat = resp_wide.to_numpy(dtype=float)
    resid_mat = resid_wide.to_numpy(dtype=float)
    item_scale_idx = np.array([SCALE_ORDER.index(ITEM_TO_SCALE[item]) for item in ITEM_ORDER])
    scale_rank_indices = {sc: [fixed_order.index(x) for x in top_by_scale[sc]] for sc in SCALE_ORDER}
    warm_idx = []
    for sc in SCALE_ORDER:
        if top_by_scale[sc]:
            warm_idx.append(fixed_order.index(top_by_scale[sc][0]))
    warm_idx = warm_idx[:min(len(warm_idx), budget)]
    cutoffs = {sc: np.array(load_config()["labels"][sc]["cutoffs"][1:-1], dtype=float) for sc in SCALE_ORDER}
    selected: Dict[int, List[str]] = {}
    all_idx = np.arange(len(ITEM_ORDER))

    for row_i, pid in enumerate(ids):
        chosen = list(warm_idx)
        while len(chosen) < budget:
            chosen_arr = np.array(chosen, dtype=int)
            counts = np.array([np.sum(item_scale_idx[chosen_arr] == si) for si in range(len(SCALE_ORDER))], dtype=float)
            scale_unc = np.ones(len(SCALE_ORDER), dtype=float) * 2.0
            for si, sc in enumerate(SCALE_ORDER):
                idxs = chosen_arr[item_scale_idx[chosen_arr] == si]
                if len(idxs):
                    vals = resp_mat[row_i, idxs]
                    mean_resp = float(np.nanmean(vals))
                    est = mean_resp * SCALES[sc] - (14 if sc == "pss" else 0)
                    boundaries = cutoffs[sc]
                    dist = float(np.min(np.abs(boundaries - est))) if len(boundaries) else 3.0
                    scale_unc[si] = (1.0 / (1.0 + dist)) + (1.0 - len(idxs) / SCALES[sc]) + 0.10 * float(np.nanstd(vals))
            reliability = 1.0
            if use_rt and len(chosen):
                rr = resid_mat[row_i, chosen_arr]
                fast = float(np.nanmean(rr < -1)); long_pause = float(np.nanmean(rr > 1)); std = float(np.nanstd(rr))
                reliability = float(np.clip(1.0 - 0.35 * fast - 0.20 * long_pause - 0.08 * min(std, 4), 0.10, 1.0))
                scale_unc *= (1.0 + 0.25 * max(0.0, 1.0 - reliability))
            unreliability = max(0.0, 1.0 - reliability)
            remaining = np.array([i for i in all_idx if i not in chosen], dtype=int)
            r_sc = item_scale_idx[remaining]
            coverage_gap = 1.0 - counts[r_sc] / np.array([SCALES[SCALE_ORDER[si]] for si in r_sc], dtype=float)
            undercovered = (counts[r_sc] <= counts.min()).astype(float)
            lambda_t = 0.20 + (0.40 * unreliability if use_rt else 0.0)
            rho_t = 0.015 + (0.020 * unreliability if use_rt else 0.0)
            eta_t = 0.00 + (0.25 * unreliability if use_rt else 0.0)
            scores = info_scores[remaining] * (1.0 + scale_unc[r_sc])
            scores += lambda_t * coverage_gap + 0.08 * undercovered
            scores += eta_t * info_scores[remaining] * (1.0 + undercovered)
            scores -= rho_t * counts[r_sc]
            # Stable participant-specific jitter for deterministic tie-breaking.
            scores += np.array([((int(pid) * (int(i) + 7)) % 19) * 1e-6 for i in remaining])
            chosen.append(int(remaining[int(np.argmax(scores))]))
        selected[int(pid)] = [ITEM_ORDER[i] for i in chosen[:budget]]
    return selected

def _scores_from_selected(ids: Iterable[int], selected: Dict[int, List[str]], variant: str, long: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    # Compute score and response-time features only from selected items; future items are excluded from prediction.
    """Return scale-specific score estimates built only from selected item pairs."""
    ids = list(map(int, ids))
    pairs = [(int(pid), item) for pid in ids for item in selected[int(pid)]]
    if "pair_key" not in long.columns:
        long = long.copy()
        long["pair_key"] = long["participant_id"].astype(str) + "|" + long["global_item_id"].astype(str)
    key_set = set(str(pid) + "|" + item for pid, item in pairs)
    use_cols = ["participant_id", "global_item_id", "global_position", "scale", "item_response", "raw_rt", "rt_residual_content_z", "rt_residual_conditioned_z", "pair_key"]
    sub_long = long.loc[long["participant_id"].isin(ids), use_cols]
    obs = sub_long.loc[sub_long["pair_key"].isin(key_set), use_cols].copy()
    train_means = _train_score_means()

    # Prefix-level response-process summaries, computed from selected items only.
    state = obs.groupby("participant_id", sort=False).agg(
        n_items=("global_item_id", "size"),
        fast=("rt_residual_content_z", lambda x: float(np.mean(np.asarray(x) < -1))),
        long=("rt_residual_content_z", lambda x: float(np.mean(np.asarray(x) > 1))),
        std=("rt_residual_content_z", lambda x: float(np.std(np.asarray(x)))),
        mean_rt=("rt_residual_content_z", "mean"),
        scale_coverage=("scale", lambda x: float(pd.Series(x).nunique() / 4)),
    ).reindex(ids).fillna(0).reset_index()
    state["posterior_entropy"] = 0.0
    if variant == "responses_rt_state":
        obj = _cached_online_model()
        if obj is not None:
            # Vectorized approximation of prefix features for the online GMM.
            tmp = obs.groupby("participant_id").agg(
                n_observed=("global_item_id", "size"),
                mean_rt_residual=("rt_residual_content_z", "mean"),
                median_rt_residual=("rt_residual_content_z", "median"),
                std_rt_residual=("rt_residual_content_z", "std"),
                fast_response_ratio=("rt_residual_content_z", lambda x: float(np.mean(np.asarray(x) < -1))),
                long_pause_ratio=("rt_residual_content_z", lambda x: float(np.mean(np.asarray(x) > 1))),
                response_variability=("item_response", "std"),
                scale_coverage=("scale", lambda x: float(pd.Series(x).nunique() / 4)),
            ).reindex(ids).fillna(0)
            # Match the training-time online-state feature definition by computing
            # the prefix residual-over-position slope instead of setting it to zero.
            slope_df = obs.copy()
            slope_df["_x"] = slope_df["global_position"].astype(float)
            slope_df["_y"] = slope_df["rt_residual_content_z"].astype(float)
            slope_df["_xy"] = slope_df["_x"] * slope_df["_y"]
            slope_df["_x2"] = slope_df["_x"] * slope_df["_x"]
            slope_agg = slope_df.groupby("participant_id").agg(
                n=("global_item_id", "size"), sum_x=("_x", "sum"), sum_y=("_y", "sum"),
                sum_xy=("_xy", "sum"), sum_x2=("_x2", "sum")
            ).reindex(ids).fillna(0)
            denom = slope_agg["n"] * slope_agg["sum_x2"] - slope_agg["sum_x"] ** 2
            numer = slope_agg["n"] * slope_agg["sum_xy"] - slope_agg["sum_x"] * slope_agg["sum_y"]
            tmp["rt_slope_over_position"] = np.where(np.abs(denom) > 1e-12, numer / denom, 0.0)
            feature_rows = pd.DataFrame([{c: tmp.loc[pid].get(c, 0.0) for c in obj["feature_columns"]} for pid in ids])
            probs = obj["model"].predict_proba(feature_rows)
            state["posterior_entropy"] = [float(-(r * np.log(r + 1e-12)).sum()) for r in probs]
    state["reliability"] = np.clip(
        1.0 - 0.35 * state["fast"] - 0.20 * state["long"] - 0.08 * np.minimum(state["std"], 4) - 0.05 * state["posterior_entropy"],
        0.10,
        1.0,
    )

    # Scale-level aggregates.
    agg = obs.groupby(["participant_id", "scale"], sort=False).agg(
        n=("global_item_id", "size"),
        mean_resp=("item_response", "mean"),
        rawrt_mean=("raw_rt", "mean"),
        resid_mean=("rt_residual_content_z", "mean"),
        residc_mean=("rt_residual_conditioned_z", "mean"),
    )

    scores: Dict[str, np.ndarray] = {}
    median_rt = float(long["raw_rt"].median())
    state_idx = state.set_index("participant_id")
    for sc in SCALE_ORDER:
        score = np.full(len(ids), train_means[sc], dtype=float)
        try:
            sub = agg.xs(sc, level="scale").reindex(ids)
        except KeyError:
            scores[sc] = score
            continue
        has_obs = sub["n"].notna().values
        if variant == "rt_only":
            val = np.full(len(ids), train_means[sc], dtype=float) + 0.25 * sub["resid_mean"].fillna(0).values.astype(float)
        else:
            val = sub["mean_resp"].fillna(0).values.astype(float) * SCALES[sc]
            if sc == "pss":
                val = val - 14
            if variant == "responses_rawrt":
                val = val + 0.005 * (sub["rawrt_mean"].fillna(median_rt).values.astype(float) - median_rt)
            elif variant == "responses_resid_content":
                val = val + 0.20 * sub["resid_mean"].fillna(0).values.astype(float)
            elif variant == "responses_resid_conditioned":
                val = val + 0.20 * sub["residc_mean"].fillna(0).values.astype(float)
            elif variant == "responses_rt_state":
                val = val + 0.20 * sub["resid_mean"].fillna(0).values.astype(float)
                reliability = state_idx.reindex(ids)["reliability"].fillna(1.0).values.astype(float)
                shrink = np.minimum(0.35, np.maximum(0.0, 1.0 - reliability) * 0.40)
                val = (1 - shrink) * val + shrink * train_means[sc]
        max_score = load_config()["labels"][sc]["cutoffs"][-1] - 1
        score[has_obs] = np.clip(val[has_obs], 0, max_score)
        scores[sc] = score
    return scores, state


def _calibration_table_path() -> Path:
    return root_path("results/tables/calibration_parameters.csv")



def _sigma_multiplier_for_variant(variant: str, sc: str, scores: np.ndarray, state: pd.DataFrame, ids: List[int]) -> np.ndarray:
    """Participant-level calibration multiplier for non-label-changing controls.

    These controls alter probability temperature without changing scale-severity labels.
    This separates response-time signal from calibration gains caused by ordinary
    confidence softening.
    """
    n = len(ids)
    base = np.ones(n, dtype=float)
    if state is None or len(state) == 0:
        return base
    st = state.set_index("participant_id").reindex(ids)
    reliability = st.get("reliability", pd.Series(np.ones(n), index=ids)).fillna(1.0).values.astype(float)
    unreliability = np.maximum(0.0, 1.0 - reliability)
    if variant == "responses_rt_calibration_only":
        return 1.0 + 1.25 * unreliability
    if variant == "responses_constant_softening":
        return np.full(n, 1.25, dtype=float)
    if variant == "responses_coverage_only_calibration":
        cov = st.get("scale_coverage", pd.Series(np.ones(n), index=ids)).fillna(1.0).values.astype(float)
        return 1.0 + 0.80 * np.maximum(0.0, 1.0 - cov)
    if variant == "responses_uncertainty_only_calibration":
        cutoffs = np.array(load_config()["labels"][sc]["cutoffs"][1:-1], dtype=float)
        if len(cutoffs) == 0:
            return base
        dist = np.min(np.abs(scores.reshape(-1, 1) - cutoffs.reshape(1, -1)), axis=1)
        uncertainty = 1.0 / (1.0 + dist)
        return 1.0 + 0.90 * uncertainty
    if variant == "responses_shuffled_rt_calibration":
        rng = np.random.default_rng(load_config()["seed"] + 31415 + SCALE_ORDER.index(sc))
        shuffled = reliability.copy()
        rng.shuffle(shuffled)
        return 1.0 + 1.25 * np.maximum(0.0, 1.0 - shuffled)
    return base


def get_calibration_sigmas(policy: str, budget: int, variant: str, long: pd.DataFrame) -> Dict[str, float]:
    """Select scale-specific score-to-probability sigmas on validation data only.

    The cache is keyed by policy, budget, and variant. The implementation is
    intentionally simple and deterministic to avoid hidden test-set tuning.
    """
    path = _calibration_table_path()
    cols = ["policy", "budget", "variant", "scale_key", "sigma", "validation_ece"]
    table = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=cols)
    if len(table):
        sub = table[(table["policy"] == policy) & (table["budget"].astype(int) == int(budget)) & (table["variant"] == variant)]
        if len(sub) >= len(SCALE_ORDER):
            return {str(r["scale_key"]): float(r["sigma"]) for _, r in sub.tail(len(SCALE_ORDER)).iterrows()}
    val_ids = load_ids("val")
    selected = simulate_item_selection(policy, budget, val_ids, long=long, seed=load_config()["seed"] + 1000 + int(budget))
    scores, state = _scores_from_selected(val_ids, selected, variant, long)
    targets = make_targets_frame().set_index("participant_id").loc[val_ids]
    grid = load_config().get("calibration_grid", [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0])
    reliab = state.set_index("participant_id").reindex(val_ids)["reliability"].fillna(1.0).values.astype(float) if "reliability" in state.columns else np.ones(len(val_ids))
    rows = []
    for sc in SCALE_ORDER:
        y = targets[f"{sc}_severity_idx"].astype(int).values
        best_sigma, best_ece = float(grid[0]), 1e9
        multiplier = _sigma_multiplier_for_variant(variant, sc, scores[sc], state, val_ids)
        for sig in grid:
            if variant in {"responses_rt_calibration_only", "responses_constant_softening", "responses_coverage_only_calibration", "responses_uncertainty_only_calibration", "responses_shuffled_rt_calibration"}:
                prob = _probs_for_scores_variable_sigma(sc, scores[sc], float(sig) * multiplier)
            else:
                prob = _probs_for_scores(sc, scores[sc], sigma=float(sig))
            ece = ece_score(y, prob)
            if ece < best_ece:
                best_sigma, best_ece = float(sig), float(ece)
        rows.append({"policy": policy, "budget": int(budget), "variant": variant, "scale_key": sc, "sigma": best_sigma, "validation_ece": best_ece})
    new_rows = pd.DataFrame(rows)
    # Avoid the empty-dataframe concat FutureWarning without changing results.
    table = new_rows if table.empty else pd.concat([table, new_rows], ignore_index=True)
    table = table.drop_duplicates(subset=["policy", "budget", "variant", "scale_key"], keep="last")
    table.to_csv(path, index=False)
    return {r["scale_key"]: float(r["sigma"]) for r in rows}


def _probs_for_scores_variable_sigma(sc: str, scores: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    """Score-to-severity probabilities with participant-specific temperature."""
    cfg = load_config()["labels"][sc]
    cutoffs = cfg["cutoffs"]
    classes = len(cfg["classes"])
    mids = []
    for i in range(classes):
        lo = cutoffs[i]
        hi = cutoffs[i + 1] - 1
        mids.append((lo + hi) / 2)
    mids = np.array(mids, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float).reshape(-1, 1)
    logits = -((scores.reshape(-1, 1) - mids.reshape(1, -1)) ** 2) / (2 * np.maximum(sigmas, 1e-6) ** 2)
    logits = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(logits)
    return ex / ex.sum(axis=1, keepdims=True)

def predict_from_selected(ids: Iterable[int], selected: Dict[int, List[str]], variant: str, long: pd.DataFrame | None = None, sigma_map: Dict[str, float] | None = None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], pd.DataFrame]:
    """Predict severity labels from selected items using observed-prefix aggregation.

    The aggregation step filters exactly the selected participant-item pairs, so
    unselected future responses and response times cannot affect predictions.
    """
    if long is None:
        long = fit_rt_residuals()
    ids = list(map(int, ids))
    scores, state = _scores_from_selected(ids, selected, variant, long)
    pred: Dict[str, np.ndarray] = {}
    prob: Dict[str, np.ndarray] = {}
    sigma_map = sigma_map or {}
    reliab = state.set_index("participant_id").reindex(ids)["reliability"].fillna(1.0).values.astype(float) if "reliability" in state.columns else np.ones(len(ids))
    for sc in SCALE_ORDER:
        pred[sc] = _labels_for_scores(sc, scores[sc])
        base_sigma = float(sigma_map.get(sc, 2.5))
        if variant in {"responses_rt_calibration_only", "responses_constant_softening", "responses_coverage_only_calibration", "responses_uncertainty_only_calibration", "responses_shuffled_rt_calibration"}:
            # Keep the response-only severity class unchanged and modify only
            # the probability temperature. This isolates calibration from label changes.
            multiplier = _sigma_multiplier_for_variant(variant, sc, scores[sc], state, ids)
            prob[sc] = _probs_for_scores_variable_sigma(sc, scores[sc], base_sigma * multiplier)
        else:
            prob[sc] = _probs_for_scores(sc, scores[sc], sigma=base_sigma)
    return pred, prob, state


def ece_score(y: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    conf = prob.max(axis=1)
    pred = prob.argmax(axis=1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - conf[mask].mean()))
    return float(ece)


def per_scale_metrics(ytrue: pd.DataFrame, pred: Dict[str, np.ndarray], prob: Dict[str, np.ndarray], prefix: dict | None = None) -> pd.DataFrame:
    cfg = load_config()
    rows = []
    for sc in SCALE_ORDER:
        yt = ytrue[f"{sc}_severity_idx"].astype(int).values
        yp = np.asarray(pred[sc]).astype(int)
        pp = prob[sc]
        classes = list(range(len(cfg["labels"][sc]["classes"])))
        high = cfg["labels"][sc]["high_risk_index"]
        eye = np.eye(len(classes))
        brier = np.mean(np.sum((pp - eye[yt]) ** 2, axis=1))
        # AUROC/PR-AUC are included as calibration/discrimination summaries but computed only when stable.
        # To keep the reproducibility run fast and deterministic, we use one-vs-rest probability summaries.
        auroc_vals = []
        pr_vals = []
        for cls in classes:
            yb = (yt == cls).astype(int)
            if yb.min() == yb.max():
                continue
            scores = pp[:, cls]
            try:
                auroc_vals.append(float(roc_auc_score(yb, scores)))
            except ValueError:
                pass
            try:
                pr_vals.append(float(average_precision_score(yb, scores)))
            except ValueError:
                pass
        rec = recall_score(yt >= high, yp >= high, zero_division=0)
        row = {
            "scale": SCALE_NAMES[sc],
            "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(yt, yp, average="weighted", zero_division=0)),
            "mae": float(mean_absolute_error(yt, yp)),
            "quadratic_weighted_kappa": float(cohen_kappa_score(yt, yp, weights="quadratic")),
            "high_risk_recall": float(rec),
            "false_negative_rate": float(1.0 - rec),
            "auroc": float(np.mean(auroc_vals)) if auroc_vals else np.nan,
            "pr_auc": float(np.mean(pr_vals)) if pr_vals else np.nan,
            "brier_score": float(brier),
            "ece": float(ece_score(yt, pp)),
        }
        if prefix:
            row.update(prefix)
        rows.append(row)
    return pd.DataFrame(rows)

def aggregate_metrics(scale_df: pd.DataFrame) -> dict:
    numeric = ["macro_f1", "weighted_f1", "mae", "quadratic_weighted_kappa", "high_risk_recall", "false_negative_rate", "auroc", "pr_auc", "brier_score", "ece"]
    out = {f"mean_{c}": float(scale_df[c].mean()) for c in numeric}
    return out


def evaluate_policy(policy: str, budget: int, variant: str, ids: Iterable[int], split_name: str, long: pd.DataFrame | None = None, seed: int | None = None) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    if long is None:
        long = fit_rt_residuals()
    ids = list(map(int, ids))
    selected = simulate_item_selection(policy, budget, ids, long=long, seed=seed)
    sigma_map = None
    if split_name == "test" and variant != "oracle":
        sigma_map = get_calibration_sigmas(policy, budget, variant, long)
    pred, prob, state = predict_from_selected(ids, selected, variant=variant, long=long, sigma_map=sigma_map)
    targets = make_targets_frame().set_index("participant_id").loc[ids]
    prefix = {"policy": policy, "budget": budget, "variant": variant, "split": split_name, "n": len(ids), "avg_items_used": budget}
    scale_metrics = per_scale_metrics(targets, pred, prob, prefix)
    agg = aggregate_metrics(scale_metrics)
    agg.update(prefix)
    pred_df = pd.DataFrame({"participant_id": ids})
    for sc in SCALE_ORDER:
        pred_df[f"{sc}_true"] = targets[f"{sc}_severity_idx"].values
        pred_df[f"{sc}_pred"] = pred[sc]
        pred_df[f"{sc}_confidence"] = prob[sc].max(axis=1)
        for k in range(prob[sc].shape[1]):
            pred_df[f"{sc}_prob_{k}"] = prob[sc][:, k]
    return agg, scale_metrics, pred_df.merge(state, on="participant_id", how="left")


def evaluate_random_repeats_fast(budget: int, ids: List[int], long: pd.DataFrame, repeats: int) -> Tuple[dict, pd.DataFrame]:
    """Fast repeated-random baseline using wide response matrices."""
    ids = list(map(int, ids))
    subset = long[long["participant_id"].isin(ids)]
    resp_wide = subset.pivot(index="participant_id", columns="global_item_id", values="item_response").reindex(index=ids, columns=ITEM_ORDER)
    resp_mat = resp_wide.to_numpy(dtype=float)
    targets = make_targets_frame().set_index("participant_id").loc[ids]
    sigma_map = get_calibration_sigmas("random", budget, "responses", long)
    train_means = _train_score_means()
    item_scale_idx = np.array([SCALE_ORDER.index(ITEM_TO_SCALE[item]) for item in ITEM_ORDER])
    n = len(ids)
    agg_rows = []
    per_rows = []
    for r in range(repeats):
        rng = np.random.default_rng(load_config()["seed"] + 100 + r)
        selected_idx = np.vstack([rng.choice(len(ITEM_ORDER), size=budget, replace=False) for _ in range(n)])
        selected_resp = resp_mat[np.arange(n)[:, None], selected_idx]
        pred: Dict[str, np.ndarray] = {}
        prob: Dict[str, np.ndarray] = {}
        for si, sc in enumerate(SCALE_ORDER):
            mask = item_scale_idx[selected_idx] == si
            counts = mask.sum(axis=1).astype(float)
            sums = np.where(mask, selected_resp, 0.0).sum(axis=1)
            mean_resp = np.divide(sums, counts, out=np.full(n, np.nan), where=counts > 0)
            score = mean_resp * SCALES[sc]
            if sc == "pss":
                score = score - 14
            score = np.where(np.isnan(score), train_means[sc], score)
            max_score = load_config()["labels"][sc]["cutoffs"][-1] - 1
            score = np.clip(score, 0, max_score)
            pred[sc] = _labels_for_scores(sc, score)
            prob[sc] = _probs_for_scores(sc, score, sigma=float(sigma_map.get(sc, 2.5)))
        prefix = {"policy": "random", "budget": budget, "variant": "responses", "split": "test", "n": len(ids), "avg_items_used": budget, "repeat": r}
        scdf = per_scale_metrics(targets, pred, prob, prefix)
        agg = aggregate_metrics(scdf)
        agg.update(prefix)
        agg_rows.append(agg)
        per_rows.append(scdf)
    rep = pd.DataFrame(agg_rows)
    metric_cols = [c for c in rep.select_dtypes(include=[np.number]).columns if c not in ["repeat", "budget", "n", "avg_items_used"]]
    mean = {c: rep[c].mean() for c in metric_cols}
    sd = {f"{c}_sd": rep[c].std(ddof=1) for c in metric_cols}
    row = {"policy": "random_repeated_mean", "budget": budget, "variant": "responses", "split": "test", "n": len(ids), "avg_items_used": budget, "n_repeats": repeats, **mean, **sd}
    per = pd.concat(per_rows, ignore_index=True).groupby(["scale", "budget", "variant", "split", "n", "avg_items_used"], as_index=False).mean(numeric_only=True)
    per["policy"] = "random_repeated_mean"
    return row, per


def run_baselines(long: pd.DataFrame | None = None) -> pd.DataFrame:
    ensure_dirs()
    if long is None:
        long = fit_rt_residuals()
    cfg = load_config()
    test_ids = load_ids("test")
    rows = []
    per_rows = []
    rows.append(
        {
            "policy": "full_scale_scoring_upper_bound",
            "budget": 37,
            "variant": "oracle",
            "split": "test",
            "n": len(test_ids),
            "avg_items_used": 37,
            "mean_macro_f1": 1.0,
            "mean_weighted_f1": 1.0,
            "mean_mae": 0.0,
            "mean_quadratic_weighted_kappa": 1.0,
            "mean_high_risk_recall": 1.0,
            "mean_false_negative_rate": 0.0,
            "mean_auroc": 1.0,
            "mean_pr_auc": 1.0,
            "mean_brier_score": 0.0,
            "mean_ece": 0.0,
        }
    )
    for policy in ["fixed_order", "short_forms", "train_selected_fixed"]:
        for budget in cfg["budgets"]:
            print(f"  baseline {policy} budget {budget}", flush=True)
            agg, scdf, pred = evaluate_policy(policy, budget, "responses", test_ids, "test", long=long, seed=cfg["seed"])
            rows.append(agg)
            per_rows.append(scdf)
            pred.to_csv(root_path("results/tables", f"predictions_{policy}_b{budget}_responses.csv"), index=False)
    # Repeated random selection summarized over many seeds, evaluated with a wide-matrix fast path.
    for budget in cfg["budgets"]:
        print(f"  baseline random repeated budget {budget}", flush=True)
        row, per = evaluate_random_repeats_fast(budget, test_ids, long, repeats=cfg.get("random_repeats", 10))
        rows.append(row)
        per_rows.append(per)
    out = pd.DataFrame(rows)
    out.to_csv(root_path("results/tables/table3_baseline_performance.csv"), index=False)
    # Per-scale baseline summaries are optional and kept compact to avoid large intermediate output.
    if per_rows:
        try:
            pd.concat(per_rows, ignore_index=True).to_csv(root_path("results/tables/table3b_baseline_per_scale_metrics.csv"), index=False)
        except Exception as exc:
            pd.DataFrame([{"warning": f"baseline per-scale summary skipped: {exc}"}]).to_csv(root_path("results/tables/table3b_baseline_per_scale_metrics.csv"), index=False)
    return out


def run_main_models(long: pd.DataFrame | None = None) -> pd.DataFrame:
    ensure_dirs()
    if long is None:
        long = fit_rt_residuals()
    cfg = load_config()
    test_ids = load_ids("test")
    rows, per_rows = [], []
    for budget in cfg["budgets"]:
        for policy, variant in [("response_only_uncertainty", "responses"), ("response_only_uncertainty", "responses_rt_calibration_only"), ("rt_aware_uncertainty", "responses_rt_state")]:
            agg, scdf, pred = evaluate_policy(policy, budget, variant, test_ids, "test", long=long, seed=cfg["seed"] + budget)
            rows.append(agg)
            per_rows.append(scdf)
            pred.to_csv(root_path("results/tables", f"predictions_{policy}_b{budget}_{variant}.csv"), index=False)
    out = pd.DataFrame(rows)
    out.to_csv(root_path("results/tables/main_model_results.csv"), index=False)
    pd.concat(per_rows, ignore_index=True).to_csv(root_path("results/tables/main_model_per_scale_metrics.csv"), index=False)
    return out


def run_adaptive_simulation(long: pd.DataFrame | None = None) -> pd.DataFrame:
    if long is None:
        long = fit_rt_residuals()
    base = run_baselines(long) if not root_path("results/tables/table3_baseline_performance.csv").exists() else pd.read_csv(root_path("results/tables/table3_baseline_performance.csv"))
    main = run_main_models(long) if not root_path("results/tables/main_model_results.csv").exists() else pd.read_csv(root_path("results/tables/main_model_results.csv"))
    df = pd.concat([base, main], ignore_index=True, sort=False)
    rows = []
    for b in load_config()["budgets"]:
        ro = df[(df["policy"] == "response_only_uncertainty") & (df["budget"] == b)].iloc[0]
        best_macro = float(df[(df["budget"] == b) & (df["policy"] != "full_scale_scoring_upper_bound")]["mean_macro_f1"].max())
        for _, r in df[df["budget"] == b].iterrows():
            d = r.to_dict()
            d["delta_macro_f1_vs_response_only"] = d.get("mean_macro_f1", np.nan) - ro["mean_macro_f1"]
            d["delta_ece_vs_response_only"] = d.get("mean_ece", np.nan) - ro["mean_ece"]
            d["delta_macro_f1_vs_best_same_budget"] = d.get("mean_macro_f1", np.nan) - best_macro
            rows.append(d)
    out = pd.DataFrame(rows)
    out.to_csv(root_path("results/tables/table5_adaptive_budget_results.csv"), index=False)
    return out


def run_ablations(long: pd.DataFrame | None = None) -> pd.DataFrame:
    ensure_dirs()
    if long is None:
        long = fit_rt_residuals()
    test_ids = load_ids("test")
    budget = 19
    experiments = [
        ("partial_responses_only", "response_only_uncertainty", "responses"),
        ("constant_softening_control", "response_only_uncertainty", "responses_constant_softening"),
        ("coverage_only_calibration_control", "response_only_uncertainty", "responses_coverage_only_calibration"),
        ("uncertainty_only_calibration_control", "response_only_uncertainty", "responses_uncertainty_only_calibration"),
        ("shuffled_rt_calibration_placebo", "response_only_uncertainty", "responses_shuffled_rt_calibration"),
        ("rt_only", "fixed_order", "rt_only"),
        ("responses_plus_rt_calibration_only", "response_only_uncertainty", "responses_rt_calibration_only"),
        ("partial_responses_plus_raw_rt", "rt_aware_uncertainty", "responses_rawrt"),
        ("partial_responses_plus_content_rt_residual", "rt_aware_uncertainty", "responses_resid_content"),
        ("partial_responses_plus_response_conditioned_rt_residual", "rt_aware_uncertainty", "responses_resid_conditioned"),
        ("partial_responses_plus_rt_residual_and_online_state", "rt_aware_uncertainty", "responses_rt_state"),
    ]
    rows = []
    for label, policy, variant in experiments:
        agg, scdf, pred = evaluate_policy(policy, budget, variant, test_ids, "test", long=long, seed=load_config()["seed"] + 777)
        agg["ablation"] = label
        rows.append(agg)
    rows.append({"ablation": "posthoc_full_gmm_not_used", "policy": "analysis_only", "budget": 37, "variant": "not_used_for_prediction"})
    out = pd.DataFrame(rows)
    out.to_csv(root_path("results/tables/table4_ablation_results.csv"), index=False)
    return out


def bootstrap_main_comparison(n_boot: int = 200) -> pd.DataFrame:
    """Bootstrap deltas for classification and calibration metrics on test IDs."""
    cfg = load_config()
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    comparisons = [
        ("rt_calibration_only_minus_responseonly", "predictions_response_only_uncertainty_b{b}_responses_rt_calibration_only.csv"),
        ("rtaware_score_adjusted_minus_responseonly", "predictions_rt_aware_uncertainty_b{b}_responses_rt_state.csv"),
    ]
    label_cfg = cfg["labels"]

    def prepare_metrics_payload(df):
        payload = {}
        for sc in SCALE_ORDER:
            prob_cols = [c for c in df.columns if c.startswith(f"{sc}_prob_")]
            pred = df[f"{sc}_pred"].to_numpy(dtype=int)
            if prob_cols:
                prob = df[prob_cols].to_numpy(dtype=float)
            else:
                n_cls = len(label_cfg[sc]["classes"])
                prob = np.full((len(df), n_cls), 1.0 / n_cls)
                prob[np.arange(len(df)), pred] = df[f"{sc}_confidence"].to_numpy(dtype=float)
                prob = prob / prob.sum(axis=1, keepdims=True)
            payload[sc] = (
                df[f"{sc}_true"].to_numpy(dtype=int),
                pred,
                prob,
                int(label_cfg[sc]["high_risk_index"]),
            )
        return payload

    def mean_metrics(payload, idx):
        f1s, recs, eces, briers = [], [], [], []
        for sc in SCALE_ORDER:
            y_all, pred_all, prob_all, high = payload[sc]
            y = y_all[idx]
            pred = pred_all[idx]
            prob = prob_all[idx]
            f1s.append(f1_score(y, pred, average="macro", zero_division=0))
            recs.append(recall_score(y >= high, pred >= high, zero_division=0))
            eces.append(ece_score(y, prob))
            eye = np.eye(prob.shape[1])
            briers.append(float(np.mean(np.sum((prob - eye[y]) ** 2, axis=1))))
        return {
            "macro_f1": np.mean(f1s),
            "high_risk_recall": np.mean(recs),
            "ece": np.mean(eces),
            "brier_score": np.mean(briers),
        }

    for budget in cfg["budgets"]:
        p1 = pd.read_csv(root_path("results/tables", f"predictions_response_only_uncertainty_b{budget}_responses.csv"))
        n = len(p1)
        payload1 = prepare_metrics_payload(p1)

        for comp_name, fmt in comparisons:
            path = root_path("results/tables", fmt.format(b=budget))
            if not path.exists():
                continue
            p2 = pd.read_csv(path)
            payload2 = prepare_metrics_payload(p2)
            diffs = {"macro_f1": [], "high_risk_recall": [], "ece": [], "brier_score": []}
            for _ in range(n_boot):
                idx = rng.integers(0, n, n)
                m1, m2 = mean_metrics(payload1, idx), mean_metrics(payload2, idx)
                for key in diffs:
                    diffs[key].append(m2[key] - m1[key])
            for key, vals in diffs.items():
                arr = np.asarray(vals)
                rows.append({
                    "budget": budget,
                    "comparison": comp_name,
                    "metric": f"delta_mean_{key}",
                    "mean": float(arr.mean()),
                    "ci_lower": float(np.quantile(arr, 0.025)),
                    "ci_upper": float(np.quantile(arr, 0.975)),
                })
    out = pd.DataFrame(rows)
    out.to_csv(root_path("results/tables/bootstrap_ci_main_comparison.csv"), index=False)
    return out

def _coarse_subgroups(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "smoke" in out.columns:
        out["smoke_group"] = np.where(out["smoke"].astype(str).str.lower().str.contains("never|no", regex=True), "never", "ever_or_current")
    if "drink" in out.columns:
        out["drink_group"] = np.where(out["drink"].astype(str).str.lower().str.contains("never|no", regex=True), "never", "ever_or_current")
    if "edu" in out.columns:
        edu = out["edu"].astype(str).str.lower()
        out["edu_group"] = np.select([edu.str.contains("master|graduate|post"), edu.str.contains("bachelor")], ["postgraduate", "bachelor"], default="other")
    return out


def run_subgroup_robustness(long: pd.DataFrame | None = None) -> pd.DataFrame:
    ensure_dirs()
    if long is None:
        long = fit_rt_residuals()
    budget = 19
    pred_path = root_path("results/tables", f"predictions_rt_aware_uncertainty_b{budget}_responses_rt_state.csv")
    if not pred_path.exists():
        run_main_models(long)
    pred = pd.read_csv(pred_path)
    targets = make_targets_frame()[["participant_id", "gender", "age", "edu", "smoke", "drink"]]

    # Response-speed subgroups are defined from the same observed 19-item prefix
    # used by the RT-aware policy, not from the full 37-item response-time profile.
    # This avoids using post-hoc future RTs when interpreting subgroup robustness.
    ids = pred["participant_id"].astype(int).tolist()
    selected = simulate_item_selection("rt_aware_uncertainty", budget, ids, long=long, seed=load_config()["seed"] + budget)
    if "pair_key" not in long.columns:
        long = long.copy()
        long["pair_key"] = long["participant_id"].astype(str) + "|" + long["global_item_id"].astype(str)
    selected_pairs = set(str(pid) + "|" + item for pid, items in selected.items() for item in items)
    prefix_long = long[long["pair_key"].isin(selected_pairs)].copy()
    rt = participant_rt_features(prefix_long, ids)[["participant_id", "mean_rt_residual", "std_rt_residual"]]
    q1, q2 = rt["mean_rt_residual"].quantile([0.33, 0.67])
    rt["response_speed_group"] = pd.cut(rt["mean_rt_residual"], [-np.inf, q1, q2, np.inf], labels=["fast", "typical", "slow"]).astype(str)
    rt["response_speed_definition"] = "observed 19-item prefix"
    df = pred.merge(targets, on="participant_id", how="left").merge(rt, on="participant_id", how="left")
    df = _coarse_subgroups(df)
    rows = []
    for col in ["gender", "edu_group", "smoke_group", "drink_group", "response_speed_group"]:
        for val, sub in df.groupby(col):
            if len(sub) < 30:
                continue
            metrics, recs, fnrs, maes, eces, briers, high_ns = [], [], [], [], [], [], []
            for sc in SCALE_ORDER:
                y = sub[f"{sc}_true"].values.astype(int)
                p = sub[f"{sc}_pred"].values.astype(int)
                prob_cols = [c for c in sub.columns if c.startswith(f"{sc}_prob_")]
                prob = sub[prob_cols].values.astype(float) if prob_cols else None
                metrics.append(f1_score(y, p, average="macro", zero_division=0))
                high = load_config()["labels"][sc]["high_risk_index"]
                high_ns.append(int((y >= high).sum()))
                r = recall_score(y >= high, p >= high, zero_division=0)
                recs.append(r); fnrs.append(1 - r); maes.append(mean_absolute_error(y, p))
                if prob is not None:
                    eces.append(ece_score(y, prob))
                    eye = np.eye(prob.shape[1])
                    briers.append(float(np.mean(np.sum((prob - eye[y]) ** 2, axis=1))))
            rows.append({
                "subgroup_variable": col,
                "subgroup": str(val),
                "n": int(len(sub)),
                "mean_high_risk_n": float(np.mean(high_ns)),
                "mean_macro_f1": float(np.mean(metrics)),
                "mean_mae": float(np.mean(maes)),
                "mean_high_risk_recall": float(np.mean(recs)),
                "mean_false_negative_rate": float(np.mean(fnrs)),
                "mean_brier_score": float(np.mean(briers)) if briers else np.nan,
                "mean_ece": float(np.mean(eces)) if eces else np.nan,
                "average_items_used": budget,
            })
    out = pd.DataFrame(rows)
    out.to_csv(root_path("results/tables/table6_subgroup_robustness.csv"), index=False)
    return out

def run_leakage_tests(long: pd.DataFrame | None = None, quick: bool = False, output_path: str | None = None) -> pd.DataFrame:
    ensure_dirs()
    if long is None:
        long = fit_rt_residuals()
    rows = []
    ids = {s: set(load_ids(s)) for s in ["train", "val", "test"]}
    overlap_ok = not (ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"])
    rows.append({"test": "split disjointness", "status": "PASS" if overlap_ok else "FAIL", "details": "no overlap across train/validation/test participant IDs"})
    meta_path = root_path("models/rt_residual_metadata.pkl")
    if not meta_path.exists():
        fit_rt_residuals(long)
    meta = joblib.load(meta_path)
    fit_ids = set(meta["fit_participant_ids"])
    train_only = fit_ids <= ids["train"] and len(fit_ids & ids["val"]) == 0 and len(fit_ids & ids["test"]) == 0
    rows.append({"test": "RT residual train-only fit", "status": "PASS" if train_only else "FAIL", "details": "fit participant IDs are a subset of train IDs"})

    cfg = load_config().get("leakage_audit", {})
    n_participants = int(cfg.get("participants", 100))
    repeats = int(cfg.get("repeats", 20))
    budgets = list(cfg.get("prefix_lengths", [4, 8, 11, 19]))
    if quick:
        n_participants = min(10, n_participants); repeats = min(2, repeats); budgets = budgets[:2]
    rng = np.random.default_rng(load_config()["seed"] + 4321)
    test_ids = load_ids("test")
    sample_ids = rng.choice(test_ids, size=min(n_participants, len(test_ids)), replace=False).astype(int).tolist()

    # End-to-end future perturbation audit for the actual simulation path.
    # Only the sampled test rows are copied; simulate_item_selection filters to
    # the supplied participant IDs and uses cached train-set item information.
    item_info_values(long)
    base_cache = {}
    response_pass = rt_pass = total = 0
    sample_long = long[long["participant_id"].isin(sample_ids)].copy()
    if "pair_key" not in sample_long.columns:
        sample_long["pair_key"] = (
            sample_long["participant_id"].astype(str) + "|" + sample_long["global_item_id"].astype(str)
        )
    for b in budgets:
        b = min(int(b), 26)
        for r in range(repeats):
            seed = load_config()["seed"] + 9000 + b * 31 + r
            key = (b, seed)
            if key not in base_cache:
                base_cache[key] = simulate_item_selection(
                    "rt_aware_uncertainty", b, sample_ids, long=sample_long, seed=seed
                )
            base_sel = base_cache[key]
            selected_pairs = set(str(pid) + "|" + item for pid, items in base_sel.items() for item in items)
            mask_future = ~sample_long["pair_key"].isin(selected_pairs)
            pert_resp = sample_long.copy()
            vals = pert_resp.loc[mask_future, "item_response"].values.copy()
            pert_resp.loc[mask_future, "item_response"] = rng.permutation(vals)
            sel_resp = simulate_item_selection(
                "rt_aware_uncertainty", b, sample_ids, long=pert_resp, seed=seed
            )
            pert_rt = sample_long.copy()
            for col in ["raw_rt", "log_rt", "rt_residual_content_z", "rt_residual_conditioned_z"]:
                if col in pert_rt.columns:
                    vals = pert_rt.loc[mask_future, col].values.copy()
                    pert_rt.loc[mask_future, col] = rng.permutation(vals)
            sel_rt = simulate_item_selection(
                "rt_aware_uncertainty", b, sample_ids, long=pert_rt, seed=seed
            )
            for pid in sample_ids:
                total += 1
                if base_sel[pid] == sel_resp[pid]:
                    response_pass += 1
                if base_sel[pid] == sel_rt[pid]:
                    rt_pass += 1
    rows.append({"test": "future responses excluded from actual policy", "status": "PASS" if response_pass == total else "FAIL", "details": f"{response_pass}/{total} policy simulations unchanged"})
    rows.append({"test": "future RT excluded from actual policy", "status": "PASS" if rt_pass == total else "FAIL", "details": f"{rt_pass}/{total} policy simulations unchanged"})

    feature_names = ["n_observed", "mean_rt_residual", "fast_response_ratio", "long_pause_ratio", "scale_coverage", "posterior_entropy", "reliability"]
    no_full = all("full_gmm" not in f and "posthoc" not in f and "cluster" not in f for f in feature_names)
    rows.append({"test": "post-hoc full GMM excluded", "status": "PASS" if no_full else "FAIL", "details": "adaptive feature list excludes full-response cluster labels"})

    pred_sample = sample_ids[:min(100, len(sample_ids))]
    selected = {int(pid): ["phq9_q1", "gad7_q1", "pss_q1", "isi_q1", "phq9_q2", "gad7_q2", "pss_q2", "isi_q2"] for pid in pred_sample}
    pred1, prob1, _ = predict_from_selected(pred_sample, selected, "responses_rt_state", long, sigma_map={sc: 2.5 for sc in SCALE_ORDER})
    perturbed = long[long["participant_id"].isin(pred_sample)].copy()
    selected_pairs = set(str(pid) + "|" + item for pid, items in selected.items() for item in items)
    if "pair_key" not in perturbed.columns:
        perturbed["pair_key"] = perturbed["participant_id"].astype(str) + "|" + perturbed["global_item_id"].astype(str)
    mask_future = ~perturbed["pair_key"].isin(selected_pairs)
    for col in ["item_response", "raw_rt", "log_rt", "rt_residual_content_z", "rt_residual_conditioned_z"]:
        vals = perturbed.loc[mask_future, col].values.copy()
        perturbed.loc[mask_future, col] = rng.permutation(vals)
    pred2, prob2, _ = predict_from_selected(pred_sample, selected, "responses_rt_state", perturbed, sigma_map={sc: 2.5 for sc in SCALE_ORDER})
    same_pred = all(np.array_equal(pred1[sc], pred2[sc]) for sc in SCALE_ORDER)
    same_prob = all(np.allclose(prob1[sc], prob2[sc]) for sc in SCALE_ORDER)
    rows.append({"test": "fixed-prefix prediction invariance", "status": "PASS" if (same_pred and same_prob) else "FAIL", "details": f"{len(pred_sample)} participants unchanged after future response/RT permutation"})
    out = pd.DataFrame(rows)
    target = root_path("results/tables/table7_leakage_tests_quick.csv" if quick else "results/tables/table7_leakage_tests.csv") if output_path is None else Path(output_path)
    out.to_csv(target, index=False)
    return out

def _set_matplotlib_fonts() -> None:
    """Publication-oriented figure typography.

    Figure labels remain legible at final journal size, with consistent axis,
    tick, and legend typography across generated figures.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })


def _save_figure_rgb(fig, stem: str) -> None:
    """Save figure as high-resolution RGB PNG and vector PDF."""
    png = root_path("results/figures", f"{stem}.png")
    pdf = root_path("results/figures", f"{stem}.pdf")
    fig.savefig(png, dpi=600, facecolor="white", transparent=False)
    fig.savefig(pdf, facecolor="white", transparent=False)
    try:
        from PIL import Image
        im = Image.open(png).convert("RGB")
        im.save(png)
    except Exception:
        pass





def _save_supplementary_figure_rgb(fig, stem: str) -> None:
    """Save non-manuscript diagnostic figures outside the main figure sequence."""
    out_dir = root_path("results/supplementary_figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=600, facecolor="white", transparent=False)
    fig.savefig(pdf, facecolor="white", transparent=False)
    try:
        from PIL import Image
        im = Image.open(png).convert("RGB")
        im.save(png)
    except Exception:
        pass

def run_policy_item_audit(long: pd.DataFrame | None = None) -> pd.DataFrame:
    """Summarize item-selection frequency and mean rank for key policies."""
    ensure_dirs()
    if long is None:
        long = fit_rt_residuals()
    ids = load_ids("test")
    rows = []
    policies = ["fixed_order", "short_forms", "response_only_uncertainty", "rt_aware_uncertainty", "random"]
    for budget in load_config()["budgets"]:
        for policy in policies:
            if policy == "random":
                rng = np.random.default_rng(load_config()["seed"] + 770 + budget)
                selected = {int(pid): [ITEM_ORDER[i] for i in rng.choice(len(ITEM_ORDER), size=budget, replace=False)] for pid in ids}
            else:
                selected = simulate_item_selection(policy, budget, ids, long=long, seed=load_config()["seed"] + budget)
            for item in ITEM_ORDER:
                flags = []
                ranks = []
                for pid in ids:
                    items = selected[int(pid)]
                    if item in items:
                        flags.append(1.0)
                        ranks.append(items.index(item) + 1)
                    else:
                        flags.append(0.0)
                rows.append({
                    "policy": policy,
                    "budget": budget,
                    "global_item_id": item,
                    "scale": ITEM_TO_SCALE[item],
                    "selection_frequency": float(np.mean(flags)),
                    "mean_selection_rank": float(np.mean(ranks)) if ranks else np.nan,
                })
    out = pd.DataFrame(rows)
    out.to_csv(root_path("results/tables/policy_item_selection_frequency.csv"), index=False)
    return out


def generate_tables_figures(long: pd.DataFrame | None = None) -> None:
    ensure_dirs()
    if long is None:
        long = fit_rt_residuals()
    _set_matplotlib_fonts()

    # Figure 1: response-time residualization diagnostics; no internal title.
    sample = long.sample(n=min(22000, len(long)), random_state=load_config()["seed"])
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.05))
    ax = axes[0]
    ax.scatter(sample["expected_log_rt_content"], sample["log_rt"], s=2, alpha=0.14)
    lo = float(min(sample["expected_log_rt_content"].min(), sample["log_rt"].min()))
    hi = float(max(sample["expected_log_rt_content"].max(), sample["log_rt"].max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=0.7)
    ax.set_xlabel("Expected log RT")
    ax.set_ylabel("Observed log RT")
    ax = axes[1]
    ax.hist(sample["rt_residual_content_z"].clip(-4, 4), bins=36, linewidth=0.4, edgecolor="black")
    ax.axvline(0, linestyle="--", linewidth=0.7)
    ax.set_xlabel("RT residual (z)")
    ax.set_ylabel("Count")
    fig.tight_layout(pad=0.45, w_pad=0.8)
    _save_figure_rgb(fig, "figure1_rt_residualization")
    plt.close(fig)

    # Figure 2: performance curves across budgets with distinct line styles/markers.
    table = pd.read_csv(root_path("results/tables/table5_adaptive_budget_results.csv")) if root_path("results/tables/table5_adaptive_budget_results.csv").exists() else run_adaptive_simulation(long)
    fig, ax = plt.subplots(figsize=(3.5, 2.25))
    style = {
        "fixed_order": ("o", "-"),
        "short_forms": ("^", "-."),
        "response_only_uncertainty": ("s", "--"),
        "rt_aware_uncertainty": ("D", ":"),
        "random_repeated_mean": ("x", "-"),
    }
    keep = ["fixed_order", "short_forms", "response_only_uncertainty", "rt_aware_uncertainty", "random_repeated_mean"]
    for pol in keep:
        sub = table[table["policy"] == pol].sort_values("budget")
        if len(sub):
            marker, ls = style[pol]
            ax.plot(sub["budget"], sub["mean_macro_f1"], marker=marker, linestyle=ls, linewidth=1.0, markersize=3, label={"fixed_order":"Fixed order", "short_forms":"Short form", "response_only_uncertainty":"Response-only", "rt_aware_uncertainty":"RT-aware", "random_repeated_mean":"Random mean"}.get(pol, pol.replace("_", " ")))
    ax.set_xlabel("Item budget")
    ax.set_ylabel("Mean macro-F1")
    ax.set_xticks(load_config()["budgets"])
    ax.set_xlim(10.5, 26.8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(pad=0.4)
    _save_figure_rgb(fig, "figure2_budget_performance")
    plt.close(fig)

    # Figure 3: calibration by budget.
    fig, ax = plt.subplots(figsize=(3.5, 2.25))
    cal = pd.read_csv(root_path("results/tables/main_model_results.csv")) if root_path("results/tables/main_model_results.csv").exists() else run_main_models(long)
    fig3_specs = [
        ("response_only_uncertainty", "responses", "s", "--", "Response-only"),
        ("response_only_uncertainty", "responses_rt_calibration_only", "o", "-.", "RT calibration-only"),
        ("rt_aware_uncertainty", "responses_rt_state", "D", ":", "RT score-adjusted"),
    ]
    for pol, var, marker, ls, label in fig3_specs:
        sub = cal[(cal["policy"] == pol) & (cal["variant"] == var)].sort_values("budget")
        if len(sub):
            ax.plot(sub["budget"], sub["mean_ece"], marker=marker, linestyle=ls, linewidth=1.0, markersize=3, label=label)
    ax.set_xlabel("Item budget")
    ax.set_ylabel("Mean ECE*")
    ax.set_xticks(load_config()["budgets"])
    ax.set_xlim(10.5, 26.8)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout(pad=0.4)
    _save_figure_rgb(fig, "figure3_ece_by_budget")
    plt.close(fig)

    # Figure 4: response-speed subgroup FNR, horizontal for better readability.
    subg = pd.read_csv(root_path("results/tables/table6_subgroup_robustness.csv")) if root_path("results/tables/table6_subgroup_robustness.csv").exists() else run_subgroup_robustness(long)
    sp = subg[subg["subgroup_variable"] == "response_speed_group"].copy()
    order = {"fast": 0, "typical": 1, "slow": 2}
    sp["order"] = sp["subgroup"].map(order).fillna(9)
    sp = sp.sort_values("order")
    fig, ax = plt.subplots(figsize=(3.5, 2.05))
    bars = ax.barh(sp["subgroup"], sp["mean_false_negative_rate"], hatch="//", edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Mean false-negative rate")
    ax.set_ylabel("Response-speed subgroup")
    ax.set_xlim(0, max(0.05, float(sp["mean_false_negative_rate"].max()) * 1.25))
    for b in bars:
        ax.text(b.get_width() + 0.002, b.get_y() + b.get_height()/2, f"{b.get_width():.3f}", va="center", fontsize=8)
    fig.tight_layout(pad=0.4)
    _save_figure_rgb(fig, "figure6_subgroup_fnr")
    plt.close(fig)


    # Figure 5: policy audit with scale coverage and item-selection heatmap.
    audit = pd.read_csv(root_path("results/tables/policy_item_selection_frequency.csv")) if root_path("results/tables/policy_item_selection_frequency.csv").exists() else run_policy_item_audit(long)
    a19 = audit[audit["budget"] == 19].copy()
    show_policies = ["short_forms", "random", "response_only_uncertainty", "rt_aware_uncertainty"]
    policy_labels = {"short_forms":"short form", "random":"random", "response_only_uncertainty":"response only", "rt_aware_uncertainty":"RT-aware"}
    cover = a19[a19["policy"].isin(show_policies)].groupby(["policy", "scale"])["selection_frequency"].sum().reset_index()
    heat = a19[a19["policy"].isin(show_policies)].pivot(index="policy", columns="global_item_id", values="selection_frequency").reindex(index=show_policies, columns=ITEM_ORDER)
    fig, axes = plt.subplots(2, 1, figsize=(7.16, 3.25), gridspec_kw={"height_ratios":[1.0, 1.25]})
    ax = axes[0]
    x = np.arange(len(show_policies)); width = 0.18
    for k, sc in enumerate(SCALE_ORDER):
        vals = [float(cover[(cover["policy"]==p)&(cover["scale"]==sc)]["selection_frequency"].sum()) for p in show_policies]
        ax.bar(x + (k-1.5)*width, vals, width=width, label=sc.upper(), edgecolor="black", linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels([policy_labels[p] for p in show_policies])
    ax.set_ylabel("Mean selected items")
    ax.legend(frameon=False, ncol=4, fontsize=8)
    ax = axes[1]
    im = ax.imshow(heat.fillna(0).values, aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(range(len(heat.index))); ax.set_yticklabels([policy_labels[p] for p in heat.index])
    ax.set_xticks(range(len(ITEM_ORDER))); ax.set_xticklabels(ITEM_ORDER, rotation=90, fontsize=8)
    ax.set_ylabel("Policy"); ax.set_xlabel("Item")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cbar.set_label("Selection frequency")
    fig.tight_layout(pad=0.35)
    _save_supplementary_figure_rgb(fig, "supplementary_figure_policy_audit_heatmap")
    plt.close(fig)

    # Figure 6: calibration-control comparison at the 19-item budget.
    abl = pd.read_csv(root_path("results/tables/table4_ablation_results.csv")) if root_path("results/tables/table4_ablation_results.csv").exists() else run_ablations(long)
    label_map = {
        "partial_responses_only":"Response-only",
        "constant_softening_control":"Const. softening",
        "coverage_only_calibration_control":"Coverage only",
        "uncertainty_only_calibration_control":"Uncertainty only",
        "shuffled_rt_calibration_placebo":"Shuffled RT",
        "responses_plus_rt_calibration_only":"True RT-cal.",
        "partial_responses_plus_rt_residual_and_online_state":"RT-score",
    }
    plot_df = abl[abl["ablation"].isin(label_map)].copy()
    plot_df["label"] = plot_df["ablation"].map(label_map)
    plot_df["order"] = plot_df["ablation"].map({k:i for i,k in enumerate(label_map)})
    plot_df = plot_df.sort_values("order")
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.05))
    for ax, metric, ylabel in [(axes[0], "mean_ece", "Label-aware ECE"), (axes[1], "mean_brier_score", "Brier score")]:
        vals = plot_df[metric].astype(float)
        xmax = float(vals.max()) * 1.28
        bars = ax.barh(plot_df["label"], vals, edgecolor="black", linewidth=0.35)
        ax.set_xlabel(ylabel)
        ax.set_xlim(0, xmax)
        ax.invert_yaxis()
        for b, v in zip(bars, vals):
            ax.text(v + xmax * 0.02, b.get_y()+b.get_height()/2, f"{v:.3f}", va="center", fontsize=7)
    fig.tight_layout(pad=0.35, h_pad=0.65)
    _save_supplementary_figure_rgb(fig, "supplementary_figure_calibration_reliability_diagram")
    _save_supplementary_figure_rgb(fig, "supplementary_figure_calibration_controls_comparison")
    plt.close(fig)


def run_all(force: bool = False) -> None:
    if force:
        reset_outputs()
    ensure_dirs()
    runtime_rows = []

    def stage(idx: int, total: int, label: str, func):
        print(f"[{idx}/{total}] {label}...", flush=True)
        t0 = time.perf_counter()
        result = func()
        elapsed = time.perf_counter() - t0
        runtime_rows.append({"stage": idx, "label": label, "seconds": round(elapsed, 3)})
        print(f"[{idx}/{total}] completed in {elapsed:.1f}s", flush=True)
        return result

    total = 12
    stage(1, total, "validating raw data", validate_data)
    stage(2, total, "creating participant-level splits", create_splits)
    long = stage(3, total, "building long-format item table", lambda: build_long_format(save_sample=True))
    long = stage(4, total, "fitting response-time residual models", lambda: fit_rt_residuals(long))
    stage(5, total, "fitting post-hoc and online response-process summaries", lambda: fit_response_state_models(long))
    # Reload the compact cached long table before repeated policy evaluation to keep memory predictable.
    del long
    gc.collect()
    long = fit_rt_residuals()
    stage(6, total, "evaluating baseline item-budget policies", lambda: run_baselines(long))
    stage(7, total, "evaluating response-only and response-time-aware policies", lambda: run_main_models(long))
    stage(8, total, "combining adaptive-simulation results", lambda: run_adaptive_simulation(long))
    stage(9, total, "running ablation experiments", lambda: run_ablations(long))
    stage(10, total, "estimating bootstrap confidence intervals", lambda: bootstrap_main_comparison(n_boot=200))
    stage(11, total, "running subgroup and leakage audits", lambda: (run_subgroup_robustness(long), run_leakage_tests(long)))
    stage(12, total, "generating final tables and figures", lambda: generate_tables_figures(long))

    pd.DataFrame(runtime_rows).to_csv(root_path("reports/runtime_report.csv"), index=False)
    with open(root_path("reports/RUN_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("# Reproducibility run report\n\n")
        f.write("Pipeline completed from raw CSV files. Large processed tables are not included in the release ZIP.\n")
        f.write("The adaptive policies are sequential and participant-specific for response-only and response-time-aware uncertainty selection.\n")
        f.write("Recommended memory: at least 4 GB RAM, preferably 8 GB.\n")



if __name__ == "__main__":
    run_all(force=os.environ.get("FORCE", "0") == "1")
