"""The exported tests use only this package, never the author's source checkout."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'library'))
