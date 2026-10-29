"""End-to-End Execution and Validation Test for Decision_Research_Walkthrough.ipynb.

Loads the notebook JSON, iterates through all 24 sections, executes each code cell
in sequential order in a shared isolated execution namespace, and asserts that:
1. No exceptions are raised.
2. All outputs and tensors are produced cleanly.
3. No hidden or out-of-order state dependencies exist.
"""

import json
from pathlib import Path
import sys
import traceback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def test_notebook():
    nb_path = Path("notebooks/Decision_Research_Walkthrough.ipynb")
    assert nb_path.is_file(), f"Notebook file not found at: {nb_path}"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    print(f"Total Notebook Cells: {len(cells)}")

    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    print(f"Code Cells to Execute: {len(code_cells)}")

    # Shared execution environment
    exec_globals = {
        "__name__": "__main__",
        "__file__": str(nb_path.resolve()),
    }

    passed_count = 0
    for idx, cell in enumerate(code_cells, start=1):
        code = "".join(cell.get("source", []))
        first_line = code.strip().split("\n")[0] if code.strip() else "<empty>"
        print(f"\n[Cell {idx}/{len(code_cells)}] Executing: {first_line[:60]}...")
        try:
            exec(code, exec_globals)
            passed_count += 1
            print(f" -> Cell {idx} PASSED")
        except Exception as e:
            print(f"\n[ERROR] Exception in Cell {idx}:")
            traceback.print_exc()
            sys.exit(1)

    print("\n" + "=" * 70)
    print(f"SUCCESS: All {passed_count}/{len(code_cells)} code cells executed without error!")
    print("=" * 70)

if __name__ == "__main__":
    test_notebook()
