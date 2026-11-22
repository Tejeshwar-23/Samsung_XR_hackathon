import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

sys.argv[0] = str(SRC / "main.py")
runpy.run_path(str(SRC / "main.py"), run_name="__main__")
