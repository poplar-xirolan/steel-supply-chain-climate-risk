import json
import os
from pathlib import Path
import subprocess
import sys


BUILDER = Path(__file__).resolve().parents[1] / "01_mfa/05_build_mfa.py"


EXERCISE_CALCULATE_MARKET_OUTPUT = r"""
import ast
import json
from pathlib import Path
import sys

import pandas as pd

builder = Path(sys.argv[1])
reverse_input = sys.argv[2] == "reverse"
tree = ast.parse(builder.read_text(encoding="utf-8"), filename=str(builder))
function = next(
    node
    for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "calculate_market_output"
)
namespace = {"pd": pd}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(builder), "exec"), namespace)
rows = [
    {
        "entity_from": entity,
        "entity_to": entity,
        "sector_from": "M1",
        "sector_to": "P2",
        "mass": mass,
    }
    for entity, mass in [("CCC", 3.0), ("AAA", 1.0), ("BBB", 2.0)]
]
if reverse_input:
    rows.reverse()
frame = pd.DataFrame(rows)
result = namespace["calculate_market_output"](frame, "M1")
print(json.dumps(list(result)))
"""


def run_market_output_order(hash_seed: int, input_order: str) -> list[str]:
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(hash_seed)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            EXERCISE_CALCULATE_MARKET_OUTPUT,
            str(BUILDER),
            input_order,
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
    )
    return json.loads(completed.stdout)


def test_market_output_entity_order_is_independent_of_hash_seed_and_input_order():
    observed = {
        tuple(run_market_output_order(seed, input_order))
        for seed in (1, 2, 123)
        for input_order in ("forward", "reverse")
    }

    assert observed == {("AAA", "BBB", "CCC")}
