# pde_scorer.py

import csv


COLUMNS_FOR_SCORING = [
    "Dimensionality",
    "Nonlinearity",
    "Boundary",
    "Time",
    "Coupling",
]


def parse_float(value):
    """
    Try to turn a string into a number.
    If it fails, treat it as 0.
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def compute_total_score(row):
    """
    Compute the complexity score for a PDE row.
    Right now: simply sum the numeric values in the
    columns listed in COLUMNS_FOR_SCORING.
    You can replace this logic with your rubric later.
    """
    total = 0.0
    for col in COLUMNS_FOR_SCORING:
        total += parse_float(row.get(col, 0))
    return total


def main():
    input_file = "PDE.csv"
    output_file = "results.csv"

    # Read all PDEs from PDE.csv
    with open(input_file, mode="r", newline="") as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)

        # Prepare output header: original columns + Total_Score
        fieldnames = reader.fieldnames if reader.fieldnames else []
        if "Total_Score" not in fieldnames:
            fieldnames = fieldnames + ["Total_Score"]

    # Compute scores and write results.csv
    with open(output_file, mode="w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            total_score = compute_total_score(row)
            row["Total_Score"] = total_score
            writer.writerow(row)

    print(f"Scored {len(rows)} PDEs. Results written to {output_file}.")


if __name__ == "__main__":
    main()
