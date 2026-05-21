import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for path in [
    PROJECT_ROOT,
    os.path.join(PROJECT_ROOT, "ecotraffic", "src"),
    os.path.join(PROJECT_ROOT, "kedro_preprocessing", "src"),
]:
    if path not in sys.path:
        sys.path.insert(0, path)