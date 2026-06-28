import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rt_screening.core import build_long_format
build_long_format(save_sample=True)
