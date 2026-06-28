"""Validate publication-ready Frontiers tables and figures."""

from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "frontiers_submission"

expected = (
    [OUT / "main_tables" / f"Table_{i}_" for i in range(1, 11)]
    + [OUT / "supplementary_tables" / f"Supplementary_Table_S{i}_" for i in range(1, 12)]
)

main_tables = sorted((OUT / "main_tables").glob("Table_*.csv"))
supp_tables = sorted((OUT / "supplementary_tables").glob("Supplementary_Table_*.csv"))
main_png = sorted((OUT / "main_figures").glob("Figure_*.png"))
main_tiff = sorted((OUT / "main_figures").glob("Figure_*.tiff"))
supp_png = sorted((OUT / "supplementary_figures").glob("Supplementary_Figure_*.png"))
supp_tiff = sorted((OUT / "supplementary_figures").glob("Supplementary_Figure_*.tiff"))

assert len(main_tables) == 10, f"Expected 10 main tables, found {len(main_tables)}"
assert len(supp_tables) == 11, f"Expected 11 supplementary tables, found {len(supp_tables)}"
assert len(main_png) == len(main_tiff) == 4
assert len(supp_png) == len(supp_tiff) == 4

table_7 = pd.read_csv(next((OUT / "main_tables").glob("Table_7_*.csv")))
table_8 = pd.read_csv(next((OUT / "main_tables").glob("Table_8_*.csv")))
table_9 = pd.read_csv(next((OUT / "main_tables").glob("Table_9_*.csv")))

assert list(table_7.columns[:2]) == ["Outcome", "Predictor set"]
assert table_7.iloc[2]["Outcome"].startswith("Short-form disagreement")
assert table_7.iloc[2]["Predictor set"] == "Confidence + response time"
assert abs(float(table_7.iloc[6]["PR-AUC"]) - 0.772) < 0.001
assert abs(
    float(
        table_8.loc[
            table_8["Escalation ranking"] == "RT process score",
            "OCW captured (%)",
        ].iloc[0]
    )
    - 63.7
) < 0.01
assert table_9.iloc[0, 0].startswith("Fast-inconsistent")
assert next((OUT / "supplementary_tables").glob("Supplementary_Table_S8_*.csv")).exists()
assert next((OUT / "supplementary_tables").glob("Supplementary_Table_S11_*.csv")).exists()

for figure in main_tiff + supp_tiff:
    with Image.open(figure) as image:
        assert image.mode == "RGB", f"{figure.name} is {image.mode}, not RGB"
        dpi = image.info.get("dpi", (0, 0))
        assert dpi[0] >= 299 and dpi[1] >= 299, f"{figure.name} dpi={dpi}"
        assert image.width >= 1500, f"{figure.name} width={image.width}"

print("Frontiers submission output validation passed.")
