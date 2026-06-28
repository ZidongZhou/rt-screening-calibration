import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rt_screening.core import SCALES, root_path


def test_long_format_record_count_from_release_outputs():
    # Raw CSV files are not stored in the repository, so this check uses
    # the released analysis audit table and fixed scale metadata.
    root = root_path()
    long_audit = pd.read_csv(root / "results/tables/long_format_audit.csv")
    row = long_audit.iloc[0]
    n_items = sum(SCALES.values())
    assert int(row["n_participants"]) == 24292
    assert n_items == 37
    assert int(row["n_item_records"]) == 898804
    assert int(row["expected_records"]) == 898804
    assert str(row["matches_expected"]).lower() == "true"


def test_frontiers_output_counts_and_key_values():
    out = root_path("results/frontiers_submission")
    assert len(list((out / "main_tables").glob("Table_*.csv"))) == 10
    assert len(list((out / "supplementary_tables").glob("Supplementary_Table_*.csv"))) == 11
    assert len(list((out / "main_figures").glob("Figure_*.png"))) == 4
    assert len(list((out / "supplementary_figures").glob("Supplementary_Figure_*.png"))) == 4

    table_10 = pd.read_csv(next((out / "main_tables").glob("Table_10_*.csv")))
    sf = table_10[table_10["Reduced-item policy"] == "Established short-form item set"].iloc[0]
    adaptive = table_10[table_10["Reduced-item policy"] == "RT-aware adaptive selection"].iloc[0]
    assert float(sf["Macro-F1"]) > float(adaptive["Macro-F1"])

    table_8 = pd.read_csv(next((out / "main_tables").glob("Table_8_*.csv")))
    rt_row = table_8[table_8["Escalation ranking"] == "RT process score"].iloc[0]
    assert abs(float(rt_row["OCW captured (%)"]) - 63.7) < 0.01
