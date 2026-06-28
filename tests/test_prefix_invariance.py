import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rt_screening.core import fit_rt_residuals, load_config, load_ids, predict_from_selected, simulate_item_selection, root_path


def _raw_data_available():
    expected = ["demographic.csv", "phq9.csv", "gad7.csv", "pss.csv", "isi.csv"]
    return all((root_path("data/raw") / name).exists() for name in expected)


def test_selected_prefix_prediction_does_not_use_future_items():
    if not _raw_data_available():
        pytest.skip("Raw CSV files are not stored in this repository; download them to data/raw to run prefix perturbation recomputation.")
    long = fit_rt_residuals()
    ids = list(map(int, load_ids("test")[:3]))
    selected = simulate_item_selection(
        "response_only_uncertainty", 19, ids, long=long, seed=load_config()["seed"] + 19
    )
    pred_a, prob_a, _ = predict_from_selected(ids, selected, "responses", long=long)

    perturbed = long.copy()
    rng = np.random.default_rng(load_config()["seed"] + 555)
    for pid in ids:
        future_mask = (perturbed["participant_id"].astype(int) == pid) & (
            ~perturbed["global_item_id"].isin(selected[int(pid)])
        )
        n_future = int(future_mask.sum())
        perturbed.loc[future_mask, "item_response"] = rng.integers(0, 4, size=n_future)
        perturbed.loc[future_mask, "rt_residual_content_z"] = rng.normal(99.0, 1.0, size=n_future)
        perturbed.loc[future_mask, "rt_residual_conditioned_z"] = rng.normal(-99.0, 1.0, size=n_future)

    pred_b, prob_b, _ = predict_from_selected(ids, selected, "responses", long=perturbed)
    for scale in pred_a:
        assert np.array_equal(pred_a[scale], pred_b[scale])
        assert np.allclose(prob_a[scale], prob_b[scale])
