"""Run the full unit-test suite without requiring an editable install."""

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.discover(str(PROJECT_ROOT / "tests"),
                                                pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
