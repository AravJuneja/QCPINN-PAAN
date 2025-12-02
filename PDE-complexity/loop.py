# loop.py
#
# End-to-end pipeline:
#   1. Read PDEs from PDE.csv
#   2. Score PDEs (Dim + Nonlin + Boundary + Time + Coupling)
#   3. Run each PDE's QCPINN trainer module
#   4. Collect an error value (placeholder logic)
#   5. Write results.csv
#   6. Fit linear regression: Error vs Total_Score
#
# Run from repo root:
#   python PDE-complexity/loop.py

import csv
import math
import subprocess
import sys
from pathlib import Path

# ---------- CONFIG ----------

INPUT_CSV = "PDE.csv"
OUTPUT_CSV = "results.csv"
REGRESSION_TXT = "regression.txt"

# Which columns are used for the complexity score
SCORING_COLUMNS = [
    "Dimensionality",
    "Nonlinearity",
    "Boundary",
    "Time",
    "Coupling",
]


# ---------- HELPERS ----------

def parse_float(val, default=None):
    """Convert string to float; return default if missing / bad."""
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def compute_total_score(row):
    """Simple total complexity score = sum of scoring columns."""
    total = 0.0
    for col in SCORING_COLUMNS:
        v = parse_float(row.get(col), 0.0)
        total += v
    return total


def run_trainer_for_row(row, repo_root: Path):
    """
    Run the QCPINN trainer for this PDE, if TrainerModule is given.
    This just automates the same commands from the QCPINN README.
    """
    module = (row.get("TrainerModule") or "").strip()
    name = row.get("Name", "UNKNOWN")

    if not module:
        print(f"[SKIP] {name}: no TrainerModule specified.")
        return

    print(f"\n=== Running trainer for {name} ({module}) ===")
    cmd = [sys.executable, "-m", module]

    # Run from repo root so 'src' is importable
    subprocess.run(cmd, cwd=repo_root, check=True)
    print(f"[DONE] Trainer finished for {name}")


def get_error_for_row(row):
    """
    Placeholder: how we obtain an error metric for this PDE.

    Right now:
      - If L3_Error in the CSV is non-empty, we reuse it.
      - Otherwise, returns None.

    Later:
      - You can implement reading a file, log, or tensorboard output
        written by the trainer and return a fresh error value here.
    """
    existing = parse_float(row.get("L3_Error"), default=None)
    return existing


def linear_regression(xs, ys):
    """
    Simple least-squares linear regression: y = a * x + b.
    Returns (a, b).
    """
    n = len(xs)
    if n == 0:
        return None, None

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)

    if den == 0:
        return None, None

    a = num / den
    b = mean_y - a * mean_x
    return a, b


# ---------- MAIN PIPELINE ----------

def main():
    # Paths
    this_file = Path(__file__).resolve()
    complexity_dir = this_file.parent
    repo_root = complexity_dir.parent

    input_path = complexity_dir / INPUT_CSV
    output_path = complexity_dir / OUTPUT_CSV
    reg_path = complexity_dir / REGRESSION_TXT

    if not input_path.exists():
        print(f"ERROR: {input_path} not found. Make sure PDE.csv exists.")
        sys.exit(1)

    # 1) Read PDEs
    with input_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("No rows found in PDE.csv.")
        sys.exit(0)

    print(f"Loaded {len(rows)} PDEs from {INPUT_CSV}.")

    # 2) Score PDEs and 3–4) Run trainers + get errors
    for row in rows:
        # Compute total complexity score
        total_score = compute_total_score(row)
        row["Total_Score"] = total_score

        # Run the quantum/classical trainer for this PDE
        # Comment out the next line if you don't want to retrain every time.
        run_trainer_for_row(row, repo_root)

        # Get error value (from CSV or later from logs/files)
        error = get_error_for_row(row)
        if error is not None:
            row["L3_Error"] = error  # ensure it's stored as number/string

    # 5) Write results.csv
    # Ensure we include all original columns plus Total_Score
    fieldnames = list(rows[0].keys())
    if "Total_Score" not in fieldnames:
        fieldnames.append("Total_Score")

    with output_path.open("w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote scored results to {output_path.relative_to(repo_root)}")

    # 6) Regression: Error vs Total_Score
    xs = []
    ys = []
    for r in rows:
        score = parse_float(r.get("Total_Score"))
        err = parse_float(r.get("L3_Error"))
        if score is not None and err is not None and not math.isnan(err):
            xs.append(score)
            ys.append(err)

    a, b = linear_regression(xs, ys)
    if a is None:
        print("Not enough valid (score, error) pairs for regression.")
        return

    line = f"Linear regression (Error vs Total_Score):  y = {a:.4f} * x + {b:.4f}"
    print("\n" + line)

    with reg_path.open("w") as f_reg:
        f_reg.write(line + "\n")
        f_reg.write(f"Used {len(xs)} points.\n")

    print(f"Saved regression info to {reg_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
