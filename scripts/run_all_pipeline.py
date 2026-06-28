import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import os
from rt_screening.core import run_all
run_all(force=os.environ.get('FORCE','0') == '1')
