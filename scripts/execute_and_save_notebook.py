"""Executes notebooks/Decision_Research_Walkthrough.ipynb and saves outputs directly into the notebook file.

Executes all 22 code cells in sequential order in a shared Python environment,
captures stream outputs (stdout/stderr) and matplotlib figures (base64 PNG display_data),
assigns execution counts (1, 2, 3...), and writes the populated notebook back to disk.
"""

import base64
import io
import json
from pathlib import Path
import sys
import traceback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def execute_and_save_notebook():
    nb_path = PROJECT_ROOT / "notebooks/Decision_Research_Walkthrough.ipynb"
    assert nb_path.is_file(), f"Notebook file not found: {nb_path}"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    print(f"Loaded notebook with {len(cells)} total cells.")

    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    print(f"Found {len(code_cells)} code cells to execute.")

    # Shared execution environment
    exec_globals = {
        "__name__": "__main__",
        "__file__": str(nb_path.resolve()),
    }

    execution_counter = 0

    for idx, cell in enumerate(code_cells, start=1):
        execution_counter += 1
        cell["execution_count"] = execution_counter
        cell_outputs = []

        code = "".join(cell.get("source", []))
        first_line = code.strip().split("\n")[0] if code.strip() else "<empty>"
        print(f"\n[Executing Cell {execution_counter}/{len(code_cells)}] {first_line[:55]}...")

        # Setup stdout/stderr capturing
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        # Intercept plt.show()
        captured_figs = []

        def custom_show(*args, **kwargs):
            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=fig.dpi)
                buf.seek(0)
                img_b64 = base64.b64encode(buf.read()).decode("utf-8")
                captured_figs.append(img_b64)
                plt.close(fig)

        orig_plt_show = plt.show
        plt.show = custom_show

        exec_error = None
        try:
            exec(code, exec_globals)
        except Exception as e:
            exec_error = e
            traceback.print_exc(file=stderr_capture)
        finally:
            # Restore plt.show and std streams
            plt.show = orig_plt_show
            # Check for any remaining figures that didn't call plt.show()
            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=fig.dpi)
                buf.seek(0)
                img_b64 = base64.b64encode(buf.read()).decode("utf-8")
                captured_figs.append(img_b64)
                plt.close(fig)

            sys.stdout = orig_stdout
            sys.stderr = orig_stderr

        out_text = stdout_capture.getvalue()
        err_text = stderr_capture.getvalue()

        if out_text:
            lines = [l + "\n" for l in out_text.splitlines()]
            if lines and not out_text.endswith("\n"):
                lines[-1] = lines[-1].rstrip("\n")
            cell_outputs.append({
                "name": "stdout",
                "output_type": "stream",
                "text": lines
            })
            # Echo to console
            print(f" -> Output: {len(lines)} lines captured.")

        if err_text:
            lines = [l + "\n" for l in err_text.splitlines()]
            cell_outputs.append({
                "name": "stderr",
                "output_type": "stream",
                "text": lines
            })
            print(f" -> Stderr: {len(lines)} lines captured.")

        for fig_b64 in captured_figs:
            cell_outputs.append({
                "data": {
                    "image/png": fig_b64,
                    "text/plain": [
                        "<Figure size 1350x750 with 1 Axes>"
                    ]
                },
                "metadata": {},
                "output_type": "display_data"
            })
            print(" -> Display: Captured Matplotlib Figure (PNG base64).")

        cell["outputs"] = cell_outputs

        if exec_error:
            print(f"[FATAL] Cell {execution_counter} execution failed: {exec_error}")
            sys.exit(1)

    # Save populated notebook back to disk
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print("\n" + "=" * 70)
    print(f"SUCCESS: Executed all {execution_counter} code cells.")
    print(f"Saved fully populated notebook with outputs to: {nb_path}")
    print("=" * 70)

if __name__ == "__main__":
    execute_and_save_notebook()
