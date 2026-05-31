import argparse
import os

import numpy as np
import pandas as pd


EXCLUDE_COLS = {
    "month",
    "item_id",
    "weight",
    "yy",
    "mm",
    "date",
    "permno",
    "ret",
}


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna()
    if not valid.any():
        return np.nan
    v = values[valid].to_numpy(dtype=float)
    w = weights[valid].to_numpy(dtype=float)
    w_sum = w.sum()
    if w_sum == 0:
        return np.nan
    return float(np.dot(v, w) / w_sum)


def compute_weighted_characteristics_by_month(df: pd.DataFrame) -> pd.DataFrame:
    if "month" not in df.columns:
        raise KeyError("Input file must contain a 'month' column.")
    if "weight" not in df.columns:
        raise KeyError("Input file must contain a 'weight' column.")

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    char_cols = [c for c in numeric_cols if c not in EXCLUDE_COLS]
    if not char_cols:
        raise ValueError("No characteristic columns found to aggregate.")

    rows = []
    grouped = df.groupby("month", sort=True)
    for month, g in grouped:
        row = {"month": month}
        for col in char_cols:
            row[col] = weighted_mean(g[col], g["weight"])
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)
    return out


def default_output_path(input_csv: str) -> str:
    folder = os.path.dirname(input_csv)
    name, _ = os.path.splitext(os.path.basename(input_csv))
    return os.path.join(folder, f"{name}_weighted_characteristics_by_month.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute weighted monthly means for all characteristic columns."
    )
    parser.add_argument("--input", required=True, help="Path to top_items CSV file")
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: <input>_weighted_characteristics_by_month.csv)",
    )
    args = parser.parse_args()

    input_csv = os.path.abspath(args.input)
    output_csv = os.path.abspath(args.output) if args.output else default_output_path(input_csv)

    df = pd.read_csv(input_csv)
    out = compute_weighted_characteristics_by_month(df)
    out.to_csv(output_csv, index=False)

    print(f"Saved weighted monthly characteristics to: {output_csv}")
    print("Columns:")
    print(", ".join(out.columns))


if __name__ == "__main__":
    main()
