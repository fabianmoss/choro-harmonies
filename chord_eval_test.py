from fractions import Fraction
from pathlib import Path

import pandas as pd

from chord_eval.data_types import ChordType
from chord_eval.progression import (
    duration_overlap,
    get_progression,
    overlap,
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(
    "/Users/maurowindholz/Documents/Documents/Wuerzburg_Postdoc/"
    "Choro_transcriptions/absolute_chords_csvs"
)

ORIGINAL_DIR = BASE_DIR / "originals"
AUTOMATIC_DIR = BASE_DIR / "model_outputs_absolute"
OUTPUT_DIR = BASE_DIR / "progression_distances"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


def prepare_progression_df(
    csv_file: Path,
    add_label: bool = False,
) -> pd.DataFrame:
    """
    Read one unnamed-column CSV and convert it to the format expected
    by chord_eval.progression.get_progression().

    Expected source CSV columns:
        0 = quarter-note onset
        1 = quarter-note offset
        2 = key
        3 = absolute chord root
        4 = chord quality
        5 = inversion
        6 = chord changes/suspensions, when present
    """

    df = pd.read_csv(
        csv_file,
        header=None,
        dtype=str,
        keep_default_na=True,
    )

    # Some files have no seventh column.
    if 6 in df.columns:
        suspension = df[6].apply(normalize_change)
    else:
        suspension = pd.Series(
            [None] * len(df),
            index=df.index,
            dtype=object,
        )

    root_midi = df[3].str.strip().map(ROOT_TO_MIDI)
    chord_type = df[4].str.strip().map(QUALITY_TO_CHORDTYPE)

    # Stop early if a root was not recognized.
    if root_midi.isna().any():
        bad_rows = df.loc[root_midi.isna(), [3, 4]]
        raise ValueError(
            f"\nUnsupported chord roots in {csv_file.name}:\n"
            f"{bad_rows.to_string()}"
        )

    # Stop early if a quality was not recognized.
    if chord_type.isna().any():
        bad_rows = df.loc[chord_type.isna(), [3, 4]]
        raise ValueError(
            f"\nUnsupported chord qualities in {csv_file.name}:\n"
            f"{bad_rows.to_string()}"
        )

    durations = df.apply(
        lambda row: (
            Fraction(str(row[1]).strip())
            - Fraction(str(row[0]).strip())
        ),
        axis=1,
    )

    if (durations <= 0).any():
        bad_rows = df.loc[durations <= 0, [0, 1, 3, 4]]
        raise ValueError(
            f"\nNonpositive chord durations in {csv_file.name}:\n"
            f"{bad_rows.to_string()}"
        )

    out = pd.DataFrame(
        {
            "chord_root_midi": root_midi.astype(int),
            "chord_type": chord_type,

            # You said there are no inversions.
            "chord_inversion": 0,

            "duration": durations,
            "chord_suspension_midi": suspension,
        }
    )

    # progression.py accesses rdf2.label.
    # We add this only to annotation 2, avoiding its automatic swap.
    if add_label:
        out["label"] = None

    return out


def reconstruct_matched_metadata(
    annotation1: pd.DataFrame,
    annotation2: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reproduce progression.py's temporal-overlap matching so that the
    output contains the change/suspension information for both matched
    chords.

    annotation1 is the original annotation.
    annotation2 is the automatic annotation.
    """

    time1 = annotation1["duration"].cumsum().astype(float)

    intervals1 = [
        [start, end]
        for start, end in zip(
            [0.0] + list(time1.iloc[:-1]),
            list(time1),
        )
    ]

    time2 = annotation2["duration"].cumsum().astype(float)

    intervals2 = [
        [start, end]
        for start, end in zip(
            [0.0] + list(time2.iloc[:-1]),
            list(time2),
        )
    ]

    metadata_rows = []
    idx1 = 0

    for idx2, row2 in annotation2.iterrows():
        matched_indices = []

        # progression.py first checks the previous annotation1 chord.
        if (
            idx1 > 0
            and overlap(
                intervals1[idx1 - 1],
                intervals2[idx2],
            )
        ):
            matched_indices.append(idx1 - 1)

        # Then it checks all current/subsequent overlapping chords.
        while (
            idx1 < len(annotation1)
            and overlap(
                intervals1[idx1],
                intervals2[idx2],
            )
        ):
            matched_indices.append(idx1)
            idx1 += 1

        for matched_idx1 in matched_indices:
            row1 = annotation1.iloc[matched_idx1]

            matched_duration = duration_overlap(
                intervals1[matched_idx1],
                intervals2[idx2],
            )

            change1 = normalize_change(
                row1["chord_suspension_midi"]
            )

            change2 = normalize_change(
                row2["chord_suspension_midi"]
            )

            exact_match = (
                row1["chord_root_midi"]
                == row2["chord_root_midi"]

                and row1["chord_type"]
                == row2["chord_type"]

                and row1["chord_inversion"]
                == row2["chord_inversion"]

                and change1 == change2
            )

            metadata_rows.append(
                {
                    "annotation1_root_midi":
                        row1["chord_root_midi"],

                    "annotation1_chord_type":
                        row1["chord_type"].name,

                    "annotation1_inversion":
                        row1["chord_inversion"],

                    "annotation1_changes":
                        change1,

                    "annotation2_root_midi":
                        row2["chord_root_midi"],

                    "annotation2_chord_type":
                        row2["chord_type"].name,

                    "annotation2_inversion":
                        row2["chord_inversion"],

                    "annotation2_changes":
                        change2,

                    "verified_matched_duration":
                        matched_duration,

                    # 0 = exact match, 1 = mismatch
                    "binary_with_changes":
                        0 if exact_match else 1,
                }
            )

    return pd.DataFrame(metadata_rows)


# ============================================================
# PROCESS ALL FILE PAIRS
# ============================================================

# IMPORTANT:
# This is outside the loop so that it accumulates the means
# from ALL pieces rather than being reset for every piece.
all_piece_means = []


for original_csv in sorted(ORIGINAL_DIR.glob("*.csv")):

    # Example:
    # score_117-Bohemia_Terra-Irineu_de_Almeida.csv
    # ->
    # clean_score_117-Bohemia_Terra-Irineu_de_Almeida_chords.csv

    automatic_csv = AUTOMATIC_DIR / (
        f"clean_{original_csv.stem}_chords.csv"
    )

    # Skip originals for which no corresponding automatic file exists.
    if not automatic_csv.exists():
        print(
            f"\nWARNING: No automatic file found for "
            f"{original_csv.name}"
        )
        continue

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

    original = prepare_progression_df(
        original_csv,
        add_label=False,
    )

    automatic = prepare_progression_df(
        automatic_csv,
        add_label=True,
    )

    # --------------------------------------------------------
    # CALCULATE PROGRESSION DISTANCES
    # --------------------------------------------------------

    progression = get_progression(
        original,
        automatic,
    )

    # --------------------------------------------------------
    # ADD EXPLICIT CHANGE INFORMATION
    # --------------------------------------------------------

    matched_metadata = reconstruct_matched_metadata(
        original,
        automatic,
    )

    if len(matched_metadata) != len(progression):
        raise RuntimeError(
            "\nThe reconstructed overlap rows do not match "
            "get_progression()'s output.\n"
            f"File: {original_csv.name}\n"
            f"get_progression rows: {len(progression)}\n"
            f"metadata rows: {len(matched_metadata)}"
        )

    progression = pd.concat(
        [
            progression.reset_index(drop=True),
            matched_metadata.reset_index(drop=True),
        ],
        axis=1,
    )

    # Replace progression.py's original binary values with binary
    # values that explicitly include change/suspension strings.
    progression["binary_original_from_progression_py"] = (
        progression["binary"]
    )

    progression["binary"] = progression[
        "binary_with_changes"
    ]

    progression = progression.drop(
        columns=["binary_with_changes"]
    )

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

    # --------------------------------------------------------
    # APPEND MEAN ROW TO THIS PIECE'S OUTPUT
    # --------------------------------------------------------

    summary_row = {
        column: None
        for column in progression.columns
    }

    summary_row["time"] = "MEAN"
    summary_row["sps"] = means["sps"]
    summary_row["vl"] = means["vl"]
    summary_row["tbt"] = means["tbt"]
    summary_row["binary"] = means["binary"]

    progression = pd.concat(
        [
            progression,
            pd.DataFrame([summary_row]),
        ],
        ignore_index=True,
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

piece_means_df = pd.DataFrame(all_piece_means)

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