from fractions import Fraction
from pathlib import Path
from typing import List, Union
from harmonic_inference.data.corpus_constants import MEASURE_OFFSET
import music21
import pandas as pd
import numpy as np
from music21.converter import parse
from music21.harmony import ChordSymbol
from music21.stream import Measure, Stream
"""
a script which takes an annotated .xml/.mxl file as input and maps it into a .tsv file.
"""

    
def get_measures_df_from_music21_score(m21_score: music21.stream.Score) -> pd.DataFrame:
    """
    Compute and return a measures_df (that can be used to create a ScorePiece) from a
    parsed music21 Score.

    Parameters
    ----------
    m21_score : music21.stream.Score
        A music21 Score that has been parsed already.

    Returns
    -------
    measures_df : pd.DataFrame
        A measures_df with the following columns:
            'mc' (int): The measure index.
            'timesig' (str): The time signature of each measure.
            'start' (Fraction): The "offset" position at the start of each measure, in
                                whole notes since the beginning of the piece.
            'act_dur' (Fraction): The duration of the measure, in whole notes.
            'mc_offset' (Fraction): The starting position of this measure, in whole notes
                                 after the most recent downbeat.
            'next' (int): The measure index of the measure that follows each one.
    """
    # Lists to compute and add to measures_df
    time_signatures = []
    annotations = []
    starts = []
    lengths = []
    df_offsets = []
    mns = []

    # The start of the 2nd measure (the first full measure in the case of an anacrusis)
    ts_epoch = Fraction(list(m21_score.measureOffsetMap().keys())[1] / 4)

    # Default time signature
    time_signature = "4/4"
    ts_duration = Fraction(time_signature)

    # First and last measure number (inclusive) of all first endings
    first_endings = [
        (bracket.getFirst().number, bracket.getLast().number)
        for bracket in m21_score.flat.getElementsByClass(music21.spanner.RepeatBracket)
        if bracket.number.startswith("1")
    ]
    skipped_dur = 0

    # Add Annotations to a list

    chord_symbols = []
    for element in m21_score.recurse().getElementsByClass(ChordSymbol):
        measure = element.getContextByClass("Measure")
        # element.activeSite.remove(element)
        chord_symbols.append(element)
        # print(element, measure)
    # print(existing_chord_symbols)


    # Go through the measures and add them to the tracking lists
    for mc, (offset, measures_list) in enumerate(m21_score.measureOffsetMap().items()):
        offset = Fraction(offset) / 4 - skipped_dur
        measure = measures_list[0]

        skip = False
        for start, end in first_endings:
            if measure.measureNumber in range(start, end + 1):
                skip = True
                break
        if skip:
            skipped_dur += Fraction(measure.duration.quarterLength) / 4
            continue

        if measure.timeSignature is not None:
            
            if measure.timeSignature.ratioString != time_signature:
                # Time Signature change
                time_signature = measure.timeSignature.ratioString
                ts_duration = Fraction(time_signature)

                # Reset the ts_epoch to this location
                if mc != 0:
                    ts_epoch = offset

        if lengths:
            # Set the length of each bar to the difference between consecutive measure offsets
            lengths[-1] = offset - starts[-1]
        # Default (used only for the last measure)
        lengths.append(Fraction(measure.duration.quarterLength) / 4)

        starts.append(offset)
        time_signatures.append(time_signature)
        df_offsets.append((offset - ts_epoch) % ts_duration)
        mns.append(measure.measureNumber)
        annotations.append(chord_symbols)

    mcs = list(range(len(mns)))

    return pd.DataFrame(
        {
            "mc": mcs,
            "mn": mns,
            "annotations": annotations,
            "timesig": time_signatures,
            "start": starts,
            "act_dur": lengths,
            MEASURE_OFFSET: df_offsets,
            "next": mcs[1:] + [pd.NA],
        }
    )

def score_to_df(
    music_xml_path: Union[Path, str]
):
    m21_score: Stream = parse(music_xml_path)
    df = get_measures_df_from_music21_score(m21_score)
    print(df.head())

    # how to display the annotations in pandas df