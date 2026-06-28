import sys
from pathlib import Path
import pandas as pd
import joblib
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from rt_screening.core import root_path, run_leakage_tests, fit_rt_residuals, create_splits


def _raw_data_available():
    root = root_path()
    expected = ["demographic.csv", "phq9.csv", "gad7.csv", "pss.csv", "isi.csv"]
    return all((root / "data/raw" / name).exists() for name in expected)


def _skip_without_raw_data():
    if not _raw_data_available():
        pytest.skip("Raw CSV files are not stored in this repository; download them to data/raw to run full leakage recomputation tests.")


def _ensure_splits():
    if not _raw_data_available():
        pytest.skip("Raw CSV files are not stored in this repository; download them to data/raw to regenerate participant splits.")
    root = root_path()
    if not (root / 'data/splits/train_ids.csv').exists():
        create_splits()


def test_no_participant_overlap_between_splits():
    _ensure_splits()
    root = root_path()
    train = set(pd.read_csv(root / 'data/splits/train_ids.csv')['participant_id'])
    val = set(pd.read_csv(root / 'data/splits/val_ids.csv')['participant_id'])
    test = set(pd.read_csv(root / 'data/splits/test_ids.csv')['participant_id'])
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


def test_rt_residual_fit_train_only():
    _skip_without_raw_data()
    _ensure_splits()
    root = root_path()
    fit_rt_residuals()
    meta = joblib.load(root / 'models/rt_residual_metadata.pkl')
    train = set(pd.read_csv(root / 'data/splits/train_ids.csv')['participant_id'].astype(int))
    assert set(map(int, meta['fit_participant_ids'])).issubset(train)


def test_leakage_checks_are_empirical_and_pass():
    _skip_without_raw_data()
    table = run_leakage_tests(quick=True, output_path=str(root_path("results/tables/table7_leakage_tests_quick.csv")))
    assert set(table['status']) == {'PASS'}
    required = {
        'split disjointness',
        'RT residual train-only fit',
        'future responses excluded from actual policy',
        'future RT excluded from actual policy',
        'post-hoc full GMM excluded',
        'fixed-prefix prediction invariance',
    }
    assert required.issubset(set(table['test']))
    assert table['details'].astype(str).str.contains('/').any()
