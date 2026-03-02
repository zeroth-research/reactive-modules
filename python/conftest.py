import sys
from pathlib import Path

# Make python/examples/ importable so tests can use `from interpreter import ...`
sys.path.insert(0, str(Path(__file__).parent / "examples"))
