"""Evaluate paired chord CSVs ONLY for pieces in the valid and/or test splits.

Usage:
    python chord-eval-unseen.py --automatic-dir chord_eval/predictions/ft1
"""

import argparse
import importlib.util
from pathlib import Path

import pandas as pd

from h5splits import get_split_pieces

# Reuse core helpers from chord-eval-script.py
_SPEC = importlib.util.spec_from_file_location(
    "chord_eval_script", Path(__file__).resolve().parent / "chord-eval-script.py"
)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)

prepare_progression_df = _MOD.prepare_progression_df
compare_progressions = _MOD.compare_progressions


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate chord predictions ONLY on valid and/or test split pieces."
    )
    parser.add_argument(
        "--minor-degrees",
        choices=("natural", "major"),
        default="major",
    )
    parser.add_argument(
        "--originals-dir",
        type=Path,
        default=Path("chord_eval/originals"),
    )
    parser.add_argument(
        "--automatic-dir",
        type=Path,
        required=True,
        help="Path to the model predictions dir, e.g., chord_eval/predictions/ft1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("valid", "test", "train"),
        default=["test"],
    )

    args = parser.parse_args()
    original_dir = args.originals_dir.resolve()
    automatic_dir = args.automatic_dir.resolve()
    output_dir = (
        args.output_dir
        or automatic_dir.parent.parent / "results" / f"dist-unseen-{automatic_dir.name}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    piece_to_split = get_split_pieces(splits=args.splits, seed=args.seed)
    metric_cols = ["sps", "vl", "tbt", "binary"]
    all_piece_means = []
    all_piece_statistics = []

    for original_csv in sorted(original_dir.glob("*.csv")):
        if original_csv.stem not in piece_to_split:
            continue
        automatic_csv = automatic_dir / f"clean_{original_csv.stem}_chords.csv"
        if not automatic_csv.is_file():
            continue

        split_name = piece_to_split[original_csv.stem]
        original = prepare_progression_df(original_csv, args.minor_degrees)
        automatic = prepare_progression_df(automatic_csv, args.minor_degrees, allow_unknown=True)
        progression = compare_progressions(original, automatic)

        means = progression[metric_cols].mean()
        all_piece_means.append(
            {"piece": original_csv.stem, "split": split_name, **means.to_dict()}
        )

        vals = progression[metric_cols]
        piece_stats = pd.DataFrame(
            {
                "n_comparisons": vals.count(),
                "n_total_comparisons": len(vals),
                "n_excluded": len(vals) - vals.count(),
                "mean": vals.mean(),
                "sd": vals.std(ddof=1),
                "min": vals.min(),
                "max": vals.max(),
                "range": vals.max() - vals.min(),
                "median": vals.median(),
            }
        ).rename_axis("metric").reset_index()
        piece_stats.insert(0, "split", split_name)
        piece_stats.insert(0, "piece", original_csv.stem)
        all_piece_statistics.append(piece_stats)

        progression.to_csv(
            output_dir / f"{original_csv.stem}_progression_distances.csv", index=False
        )

    pd.concat(all_piece_statistics, ignore_index=True).to_csv(
        output_dir / "all_piece_statistics.csv", index=False
    )

    piece_means_df = pd.DataFrame(all_piece_means)
    vals = piece_means_df[metric_cols]
    overall_statistics = pd.DataFrame(
        {
            "n_pieces": vals.count(),
            "mean": vals.mean(),
            "sd": vals.std(ddof=1),
            "min": vals.min(),
            "max": vals.max(),
            "range": vals.max() - vals.min(),
            "median": vals.median(),
        }
    ).rename_axis("metric").reset_index()
    overall_statistics.to_csv(output_dir / "overall_statistics.csv", index=False)

    summary_rows = []
    for split_name in args.splits:
        sub = piece_means_df[piece_means_df["split"] == split_name]
        if not sub.empty:
            summary_rows.append(
                {"piece": f"{split_name.upper()}_MEAN", "split": split_name, **sub[metric_cols].mean().to_dict()}
            )
    summary_rows.append(
        {"piece": "OVERALL", "split": "+".join(args.splits), **vals.mean().to_dict()}
    )

    piece_means_df = pd.concat([piece_means_df, pd.DataFrame(summary_rows)], ignore_index=True)
    piece_means_df.to_csv(output_dir / "all_piece_means.csv", index=False)

    print("\n" + "=" * 70)
    print(piece_means_df.to_string(index=False))
    print("\n" + overall_statistics.to_string(index=False))
    print(f"\nSaved tables to: {output_dir}")


if __name__ == "__main__":
    main()
