import argparse
import os
import re

import matplotlib.pyplot as plt
import pandas as pd


def default_output_path(input_csv: str) -> str:
    folder = os.path.dirname(input_csv)
    name, _ = os.path.splitext(os.path.basename(input_csv))
    return os.path.join(folder, f"{name}_grid_5x2.png")


def default_anchor_source_path(input_csv: str) -> str:
    return input_csv.replace("_weighted_characteristics_by_month.csv", ".csv")


def parse_anchor_id_from_filename(path: str):
    m = re.search(r"anchor_id(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


def parse_k_from_filename(path: str):
    m = re.search(r"_k(\d+)_", os.path.basename(path))
    return int(m.group(1)) if m else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot characteristic time series in a 5x2 panel from weighted monthly CSV."
    )
    parser.add_argument("--input", required=True, help="Path to weighted monthly characteristics CSV")
    parser.add_argument("--output", default=None, help="Output plot path")
    parser.add_argument(
        "--anchor-input",
        default=None,
        help="Path to original top_items CSV with item_id/weight/characteristics",
    )
    parser.add_argument(
        "--anchor-id",
        type=int,
        default=None,
        help="Anchor item_id to mark (if omitted, parsed from filename like anchor_id6773)",
    )
    args = parser.parse_args()

    input_csv = os.path.abspath(args.input)
    output_png = os.path.abspath(args.output) if args.output else default_output_path(input_csv)
    anchor_input_csv = (
        os.path.abspath(args.anchor_input)
        if args.anchor_input
        else default_anchor_source_path(input_csv)
    )
    anchor_id = args.anchor_id if args.anchor_id is not None else parse_anchor_id_from_filename(input_csv)
    k_value = parse_k_from_filename(input_csv)

    df = pd.read_csv(input_csv)
    if "month" not in df.columns:
        raise KeyError("Input file must contain a 'month' column.")

    df["month_dt"] = pd.to_datetime(df["month"], format="%Y-%m", errors="coerce")
    if df["month_dt"].isna().any():
        raise ValueError("Some values in 'month' could not be parsed with format YYYY-MM.")
    df = df.sort_values("month_dt").reset_index(drop=True)

    characteristics = [
        c
        for c in df.columns
        if c not in {"month", "month_dt", "size"}
    ]
    if len(characteristics) != 10:
        raise ValueError(
            f"Expected 10 characteristics after excluding 'size', found {len(characteristics)}: {characteristics}"
        )

    if not os.path.exists(anchor_input_csv):
        raise FileNotFoundError(
            f"Anchor source file not found: {anchor_input_csv}. Use --anchor-input to provide it."
        )
    if anchor_id is None:
        raise ValueError("Could not infer anchor id from filename. Please provide --anchor-id.")

    source_df = pd.read_csv(anchor_input_csv)
    required_cols = {"month", "item_id"} | set(characteristics)
    missing_required = [c for c in required_cols if c not in source_df.columns]
    if missing_required:
        raise KeyError(f"Missing required columns in anchor source file: {missing_required}")

    first_month = df["month"].iloc[0]
    anchor_rows = source_df[
        (source_df["month"].astype(str) == str(first_month))
        & (source_df["item_id"].astype(int) == int(anchor_id))
    ]
    if anchor_rows.empty:
        raise ValueError(
            f"No row found for anchor_id={anchor_id} in first month={first_month} in {anchor_input_csv}"
        )
    anchor_row = anchor_rows.iloc[0]
    if "permno" not in anchor_row.index:
        raise KeyError("Anchor source file must include a 'permno' column to label the anchor by PERMNO.")
    anchor_permno = int(anchor_row["permno"])

    min_vals = {col: float(source_df[col].min()) for col in characteristics}
    max_vals = {col: float(source_df[col].max()) for col in characteristics}

    fig, axes = plt.subplots(5, 2, figsize=(16, 18), sharex=True)
    axes = axes.ravel()

    x = df["month_dt"]
    for i, col in enumerate(characteristics):
        ax = axes[i]
        local_min = float(df[col].min())
        local_max = float(df[col].max())
        ax.axhspan(
            local_min,
            local_max,
            color="lightgray",
            alpha=0.25,
            zorder=0,
            label="Time-series range of weighted mean",
        )
        ax.axhline(min_vals[col], color="gray", linestyle=":", linewidth=1.0, alpha=0.85, label="Global characteristic range")
        ax.axhline(max_vals[col], color="gray", linestyle=":", linewidth=1.0, alpha=0.85)
        ax.plot(x, df[col], linewidth=1.8, label="Weighted characteristic mean", zorder=2)
        ax.axhline(
            float(anchor_row[col]),
            color="crimson",
            linewidth=1.4,
            linestyle="--",
            zorder=3,
            label=f"Anchor (PERMNO {anchor_permno})",
        )
        ax.set_title(col)
        ax.set_ylabel("Weighted characteristic mean")
        ax.grid(alpha=0.25)

    tick_step = max(1, len(df) // 12)
    tick_pos = x.iloc[::tick_step]
    if x.iloc[-1] not in tick_pos.values:
        tick_pos = pd.concat([tick_pos, x.iloc[[-1]]])

    tick_labels = [d.strftime("%m-%Y") for d in tick_pos]
    for ax in axes[-2:]:
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_xlabel("Month")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.972))

    k_label = f" (k={k_value})" if k_value is not None else ""
    fig.suptitle(
        f"Weighted Average Characteristics of the Portfolio Around PERMNO {anchor_permno}{k_label} Over Time",
        fontsize=16,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(output_png, dpi=180)
    plt.close(fig)

    print(f"Saved: {output_png}")


if __name__ == "__main__":
    main()
