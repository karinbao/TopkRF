import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd

CHARACTERISTICS = [
    "LME",
    "BEME",
    "r12_2",
    "OP",
    "Investment",
    "ST_Rev",
    "LT_Rev",
    "AC",
    "IdioVol",
    "LTurnover",
]


def default_output_path(input_csv: str) -> str:
    folder = os.path.dirname(input_csv)
    name, _ = os.path.splitext(os.path.basename(input_csv))
    return os.path.join(folder, f"{name}_anchorstocks_heatmap.png")


def build_heatmap(input_csv: str, output_png: str) -> None:
    df = pd.read_csv(input_csv)

    required = ["rank", "anchor_id", *CHARACTERISTICS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    keep_cols = ["rank", "anchor_id", *CHARACTERISTICS]
    if "sr" in df.columns:
        keep_cols.append("sr")

    plot_df = df[keep_cols].copy()
    # Prefer SR-based ordering when available.
    if "sr" in plot_df.columns:
        plot_df = plot_df.sort_values("sr", ascending=False).reset_index(drop=True)
    else:
        plot_df = plot_df.sort_values("rank", ascending=False).reset_index(drop=True)

    values = plot_df[CHARACTERISTICS].to_numpy(dtype=float)
    y_labels = [str(int(rank)) for rank in plot_df["rank"]]
    fig_height = max(10, 0.28 * len(plot_df))
    fig, ax = plt.subplots(figsize=(14, fig_height))
    # YlGnBu maps low values to yellow and high values to dark blue.
    im = ax.imshow(values, aspect="auto", cmap="YlGnBu")

    ax.set_xticks(range(len(CHARACTERISTICS)))
    ax.set_xticklabels(CHARACTERISTICS, rotation=30, ha="right")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=8)

    ax.set_xlabel("Characteristics")
    ax.set_ylabel("Rank (1 = highest SR)")
    ax.set_title("Characteristic Profiles of the Top-50 Anchor Stocks")

    colorbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.02)
    colorbar.set_label("Rank-normalized characteristic value")
    colorbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    fig.tight_layout()
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an anchor-stock heatmap from top50 anchor rows."
    )
    parser.add_argument(
        "--input",
        default="results_ne100_md4_mfsqrt/top50_anchor_fullrows_k50.csv",
        help="Input CSV path",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path (default: <input>_anchorstocks_heatmap.png)",
    )
    args = parser.parse_args()

    input_csv = os.path.abspath(args.input)
    output_png = os.path.abspath(args.output) if args.output else default_output_path(input_csv)

    build_heatmap(input_csv=input_csv, output_png=output_png)
    print(f"Saved: {output_png}")


if __name__ == "__main__":
    main()
