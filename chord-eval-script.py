"""Compare paired chord CSVs in choro_harmonies/choro_model_ft.

Run from the repository:
    python chord_eval_test.py
Or pass its location:
    python chord_eval_test.py --repo-dir /path/to/choro_harmonies

Inputs retain the original script's headerless seven-column format (the
seventh column is optional). Requires pandas and the chord_eval toolkit.
Results go to choro_model_ft/progression_distances. Existing results for
processed pieces and all_piece_means.csv are overwritten on rerun.
Only current matching pairs contribute to all_piece_means.csv; old per-piece
result files are not deleted. Actual CSV onset/offset times are preserved; means remain unweighted.
Roman degrees default to major-scale references in every key (VI in d = B).
Use --minor-degrees natural only for files using natural-minor degrees.


        ARGS

        "--minor-degrees", choices=("natural", "major"), default="major",
        Roman-degree convention in lowercase minor keys: natural or major-referenced (default)."

        "--originals-dir", type=Path, default=Path("chord_eval/originals"),
        Path to the dir containing ground-truth CSVs

        "--automatic-dir", type=Path, required=True,
        path to the model predictions dir, e.g., chord_eval/predictions/beat-best

        "--output-dir", type=Path, default=None,
        Path to write evaluation results, defaults to dist-<automatic-dir>

"""

import argparse
import re

from fractions import Fraction
from pathlib import Path

import pandas as pd

from chord_eval.scripts.data_types import ChordType
from chord_eval.scripts.metric import get_distance


# ============================================================
# FILE PATHS
# ============================================================

# def resolve_base_dir(repo_dir=None):
#     """Locate choro_model_ft from an explicit path, cwd, or script location."""
#     if repo_dir is not None:
#         candidates = [Path(repo_dir).expanduser().resolve()]
#     else:
#         candidates = []
#         for location in (Path.cwd(), Path(__file__).resolve().parent):
#             candidates.extend([location, *location.parents])

#     for candidate in candidates:
#         for base in (candidate / "choro_model_ft", candidate):
#             if ((base / "chords").is_dir()
#                     and (base / "chords_from_model_output").is_dir()):
#                 return base

#     raise FileNotFoundError(
#         "Could not locate choro_model_ft/chords and "
#         "choro_model_ft/chords_from_model_output. Run from the "
#         "choro_harmonies repository or pass --repo-dir /path/to/choro_harmonies."
#     )


def find_file_pairs(original_dir, automatic_dir):
    """Match example.csv to clean_example_chords.csv; ignore missing outputs."""
    pairs = []
    originals = sorted(original_dir.glob("*.csv"))
    for original_csv in originals:
        automatic_csv = automatic_dir / f"clean_{original_csv.stem}_chords.csv"
        if automatic_csv.is_file():
            pairs.append((original_csv, automatic_csv))

    paired_outputs = {automatic for _, automatic in pairs}
    unmatched_outputs = [
        path for path in sorted(automatic_dir.glob("*.csv"))
        if path not in paired_outputs
    ]
    print(f"Original CSVs: {len(originals)}")
    print(f"Matched pairs: {len(pairs)}")
    print(f"Originals without a model output (skipped): {len(originals) - len(pairs)}")
    for path in unmatched_outputs:
        print(f"WARNING: Model output has no matching original: {path.name}")
    if not pairs:
        raise ValueError("No matching CSV pairs found; no results were written.")
    return pairs


# ============================================================
# ROOT AND QUALITY CONVERSION
# ============================================================

ROOT_TO_MIDI = {
    "C": 0,
    "B#": 0,

    "C#": 1,
    "Db": 1,

    "D": 2,

    "D#": 3,
    "Eb": 3,

    "E": 4,
    "Fb": 4,

    "F": 5,
    "E#": 5,

    "F#": 6,
    "Gb": 6,

    "G": 7,

    "G#": 8,
    "Ab": 8,

    "A": 9,

    "A#": 10,
    "Bb": 10,

    "B": 11,
    "Cb": 11,
}


QUALITY_TO_CHORDTYPE = {
    "M": ChordType.MAJOR,
    "m": ChordType.MINOR,
    "d": ChordType.DIMINISHED,
    "a": ChordType.AUGMENTED,

    "D7": ChordType.MAJ_MIN7,
    "M7": ChordType.MAJ_MAJ7,
    "m7": ChordType.MIN_MIN7,
    "mM7": ChordType.MIN_MAJ7,

    "d7": ChordType.DIM7,
    "h7": ChordType.HALF_DIM7,
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_change(value):
    """
    Convert empty or missing change values to None.

    Examples of valid nonempty values:
        4:5
        11:0
        4:2
        :2
    """
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value == "":
        return None

    return value


def root_pitch(root, key, minor_degrees="major"):
    """Resolve absolute roots or Roman degrees; lowercase key means minor.

    Natural minor: III/VI/VII are 3/8/10 semitones above tonic.
    Major-referenced: III/VI/VII are 4/9/11, even in minor keys.
    Accidentals modify the selected degree. Applied numerals are rejected.
    """
    root = str(root).strip().replace("♭", "b").replace("♯", "#")
    if root in ROOT_TO_MIDI:
        return ROOT_TO_MIDI[root]
    match = re.fullmatch(r"([b#]*)(VII|VI|IV|V|III|II|I)", root.upper().replace("B", "b"))
    if not match:
        raise ValueError(f"Unsupported root: {root!r}")
    key = str(key).strip().replace("♭", "b").replace("♯", "#")
    if not re.fullmatch(r"[A-Ga-g][b#]?", key):
        raise ValueError(f"Unsupported key: {key!r}")
    tonic = ROOT_TO_MIDI[key[0].upper() + key[1:]]
    degrees = (0, 2, 3, 5, 7, 8, 10) if key[0].islower() and minor_degrees == "natural" else (0, 2, 4, 5, 7, 9, 11)
    accidental, numeral = match.groups()
    index = ("I", "II", "III", "IV", "V", "VI", "VII").index(numeral)
    return (tonic + degrees[index] + accidental.count("#") - accidental.count("b")) % 12


def prepare_progression_df(csv_file, minor_degrees="major", allow_unknown=False):
    """Read headerless onset, offset, key, root, quality, inversion[, changes]."""
    df = pd.read_csv(csv_file, header=None, dtype=str)
    if df.empty or len(df.columns) not in (6, 7):
        raise ValueError(f"{csv_file.name}: expected 6 or 7 columns and at least one row")
    rows = []
    for index, row in df.iterrows():
        try:
            onset, offset = Fraction(row[0].strip()), Fraction(row[1].strip())
            if offset <= onset:
                if allow_unknown:
                    continue
                raise ValueError("offset must be greater than onset")
            quality_label = str(row[4]).strip()
            unknown = quality_label.lower() == "unknown"
            if unknown and not allow_unknown:
                raise ValueError("Unknown reference chord: correct the original annotation first")
            quality = None if unknown else QUALITY_TO_CHORDTYPE[quality_label]
            inversion = int(row[5])
            if inversion != 0:
                raise ValueError("This evaluation assumes root-position chords; nonzero inversion found")
            rows.append(dict(
                onset=onset, offset=offset,
                chord_root_midi=None if unknown else root_pitch(row[3], row[2], minor_degrees),
                chord_type=quality, chord_inversion=inversion,
                chord_suspension_midi=normalize_change(row[6]) if 6 in df.columns else None,
                source_key=row[2], source_root=row[3], source_row=index + 1,
                source_quality=quality_label, unknown=unknown,
            ))
        except (ValueError, KeyError) as error:
            raise ValueError(f"{csv_file.name}, row {index + 1}: {error}") from error
    result = pd.DataFrame(rows).sort_values("onset").reset_index(drop=True)
    for i in range(1, len(result)):
        if result.iloc[i].onset < result.iloc[i - 1].offset:
            raise ValueError(f"{csv_file.name}: overlapping intervals within one annotation")
    return result


def compare_progressions(original, automatic):
    """Compare actual interval intersections; never shift either timeline.

    Regions with only one annotation are excluded and their durations reported.
    Time units are retained exactly as supplied by the CSV exporter.
    """
    rows = []
    i = j = 0
    covered = Fraction(0)
    while i < len(original) and j < len(automatic):
        left, right = original.iloc[i], automatic.iloc[j]
        start, end = max(left.onset, right.onset), min(left.offset, right.offset)
        if start < end:
            entry = dict(time=str(start), end=str(end), matched_duration=float(end-start))
            entry["comparison_status"] = "unknown_model_chord" if right.unknown else "evaluated"
            if right.unknown:
                entry.update(sps=float("nan"), vl=float("nan"), tbt=float("nan"), binary=1)
            else:
                kwargs = dict(
                    root1=int(left.chord_root_midi), root2=int(right.chord_root_midi),
                    chord_type1=left.chord_type, chord_type2=right.chord_type,
                    inversion1=int(left.chord_inversion), inversion2=int(right.chord_inversion),
                    changes1=left.chord_suspension_midi, changes2=right.chord_suspension_midi,
                    triad_reduction=False,
                )
                for name, distance in (("sps", "SPS"), ("vl", "mechanical"), ("tbt", "tone by tone")):
                    entry[name] = get_distance(distance=distance, **kwargs)
                entry["binary"] = int(any(left[field] != right[field] for field in (
                    "chord_root_midi", "chord_type", "chord_inversion", "chord_suspension_midi")))
            for number, chord in ((1, left), (2, right)):
                for suffix, field in (("root_midi", "chord_root_midi"), ("chord_type", "chord_type"),
                                      ("inversion", "chord_inversion"), ("changes", "chord_suspension_midi"),
                                      ("source_key", "source_key"), ("source_root", "source_root"),
                                      ("source_row", "source_row"), ("source_quality", "source_quality")):
                    value = chord[field]
                    entry[f"annotation{number}_{suffix}"] = (value.name if value is not None else "unknown") if suffix == "chord_type" else value
            rows.append(entry)
            covered += end - start
        if left.offset <= right.offset:
            i += 1
        if right.offset <= left.offset:
            j += 1
    if not rows:
        raise ValueError("No overlapping annotated time between these files")
    for name, annotation in (("Original", original), ("Model", automatic)):
        duration = sum((row.offset - row.onset for row in annotation.itertuples()), Fraction(0))
        print(f"{name} annotated duration excluded (no counterpart): {duration - covered}")
    result = pd.DataFrame(rows)
    n_unknown = int((result["comparison_status"] == "unknown_model_chord").sum())
    print(f"Unknown model comparisons: {n_unknown}/{len(result)}; binary=1, SPS/VL/TBT excluded")
    return result


# ============================================================
# PROCESS ALL FILE PAIRS
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare original chords with matching clean_*_chords.csv outputs."
    )

    parser.add_argument(
        "--minor-degrees", choices=("natural", "major"), default="major",
        help="Roman-degree convention in lowercase minor keys: natural or major-referenced (default).",
    )
    parser.add_argument(
        "--originals-dir", type=Path, default=Path("chord_eval/originals"),
        help="Path to the dir containing ground-truth CSVs",
    )
    parser.add_argument(
        "--automatic-dir", type=Path, required=True,
        help="path to the model predictions dir, e.g., chord_eval/predictions/beat-best"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Path to write evaluation results, defaults to <automatic-dir>_distances.",
    )

    args = parser.parse_args()
    print("Script version: 8 (mechanical metric compatibility)")
    print(f"Minor-key Roman degrees: {args.minor_degrees}")
    ORIGINAL_DIR = args.originals_dir.resolve()
    AUTOMATIC_DIR = args.automatic_dir.resolve()
    OUTPUT_DIR = (
        args.output_dir or AUTOMATIC_DIR.parent.parent / "results" / f"dist-{AUTOMATIC_DIR.name}"
    ).resolve()
    file_pairs = find_file_pairs(ORIGINAL_DIR, AUTOMATIC_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Results: {OUTPUT_DIR}")

    # IMPORTANT:
    # This is outside the loop so that it accumulates the means
    # from ALL pieces rather than being reset for every piece.
    all_piece_means = []
    all_piece_statistics = []


    for original_csv, automatic_csv in file_pairs:

        output_csv = OUTPUT_DIR / (
            f"{original_csv.stem}_progression_distances.csv"
        )

        print("\n" + "=" * 70)
        print(f"Processing: {original_csv.name}")
        print(f"Automatic:  {automatic_csv.name}")
        print("=" * 70)

        # --------------------------------------------------------
        # PREPARE INPUT DATA
        # --------------------------------------------------------

        original = prepare_progression_df(original_csv, args.minor_degrees)
        automatic = prepare_progression_df(automatic_csv, args.minor_degrees, allow_unknown=True)
        progression = compare_progressions(original, automatic)

        # --------------------------------------------------------
        # CALCULATE NON-WEIGHTED MEAN FOR THIS PIECE
        # --------------------------------------------------------

        means = progression[
            [
                "sps",
                "vl",
                "tbt",
                "binary",
            ]
        ].mean()

        # IMPORTANT:
        # This is INSIDE the loop, so every piece contributes one row.
        all_piece_means.append(
            {
                "piece": original_csv.stem,
                "sps": means["sps"],
                "vl": means["vl"],
                "tbt": means["tbt"],
                "binary": means["binary"],
            }
        )

        # Summarize comparison rows before adding any summary rows.
        # Each overlapping chord pair has equal weight (not duration weighted).
        metric_values = progression[["sps", "vl", "tbt", "binary"]]
        piece_statistics = pd.DataFrame({
            "n_comparisons": metric_values.count(),
            "n_total_comparisons": len(metric_values),
            "n_excluded": len(metric_values) - metric_values.count(),
            "mean": metric_values.mean(),
            "sd": metric_values.std(ddof=1),
            "min": metric_values.min(),
            "max": metric_values.max(),
            "range": metric_values.max() - metric_values.min(),
            "median": metric_values.median(),
        }).rename_axis("metric").reset_index()
        piece_statistics.insert(0, "piece", original_csv.stem)
        all_piece_statistics.append(piece_statistics)
        print("\nPER-PIECE STATISTICS (unweighted comparison rows)")
        print(piece_statistics.to_string(index=False))

        # Append the same statistics to the individual piece's output.
        # Sample SD is blank when fewer than two valid comparisons exist.
        summary_rows = []
        indexed_statistics = piece_statistics.set_index("metric")
        for statistic in ("mean", "sd", "min", "max", "range", "median"):
            summary_row = {column: None for column in progression.columns}
            summary_row["time"] = statistic.upper()
            for metric in ("sps", "vl", "tbt", "binary"):
                summary_row[metric] = indexed_statistics.loc[metric, statistic]
            summary_rows.append(summary_row)
        progression = pd.concat(
            [progression, pd.DataFrame(summary_rows)], ignore_index=True,
        )

        # --------------------------------------------------------
        # SAVE THIS PIECE
        # --------------------------------------------------------

        progression.to_csv(
            output_csv,
            index=False,
        )

        print(f"Saved {len(progression)} rows to:")
        print(output_csv.resolve())


    # ============================================================
    # FINISHED PROCESSING INDIVIDUAL PIECES
    # ============================================================

    print("\nFinished processing all file pairs.")


    # ============================================================
    # MEAN DISTANCES ACROSS ALL PIECES
    # ============================================================

    pd.concat(all_piece_statistics, ignore_index=True).to_csv(
        OUTPUT_DIR / "all_piece_statistics.csv", index=False,
    )
    print(f"Saved per-piece statistics to: {OUTPUT_DIR / 'all_piece_statistics.csv'}")

    piece_means_df = pd.DataFrame(all_piece_means)

    # Describe the piece-level means, giving each piece equal weight.
    # Calculate BEFORE adding the OVERALL row, so it is not an observation.
    metric_values = piece_means_df[["sps", "vl", "tbt", "binary"]]
    overall_statistics = pd.DataFrame({
        "n_pieces": metric_values.count(),
        "mean": metric_values.mean(),
        "sd": metric_values.std(ddof=1),
        "min": metric_values.min(),
        "max": metric_values.max(),
        "range": metric_values.max() - metric_values.min(),
        "median": metric_values.median(),
    }).rename_axis("metric").reset_index()
    # Sample SD is undefined (saved as blank) if fewer than two values exist.
    overall_statistics.to_csv(
        OUTPUT_DIR / "overall_statistics.csv", index=False,
    )

    # Calculate the mean of the piece-level means
    overall_means = piece_means_df[
        ["sps", "vl", "tbt", "binary"]
    ].mean()

    # Create final overall row
    overall_row = {
        "piece": "OVERALL",
        "sps": overall_means["sps"],
        "vl": overall_means["vl"],
        "tbt": overall_means["tbt"],
        "binary": overall_means["binary"],
    }

    # Append overall row
    piece_means_df = pd.concat(
        [
            piece_means_df,
            pd.DataFrame([overall_row]),
        ],
        ignore_index=True,
    )

    # Save everything in one CSV
    piece_means_df.to_csv(
        OUTPUT_DIR / "all_piece_means.csv",
        index=False,
    )

    print("\n" + "=" * 70)
    print("PIECE MEANS + OVERALL MEAN")
    print("=" * 70)
    print(piece_means_df)
    print("\nOVERALL STATISTICS ACROSS PIECE-LEVEL MEANS")
    print("SD = sample standard deviation; range = max - min.")
    print(overall_statistics.to_string(index=False))
    print(f"Saved overall statistics to: {OUTPUT_DIR / 'overall_statistics.csv'}")

if __name__ == "__main__":
    main()
