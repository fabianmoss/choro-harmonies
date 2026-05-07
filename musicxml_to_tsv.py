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
import glob

def get_chords_and_measures_df_from_m21_score(m21_score: music21.stream.Score) -> pd.DataFrame:
    """
    a script which takes an annotated .xml/.mxl file as input and maps it into a .tsv file.
    """
    """
    Parameters
    ----------
    m21_score : music21.stream.Score
        A music21 Score that has been parsed already.

    Returns
    -------
    1. measures_df : pd.DataFrame
        A measures_df with the following columns:
            'mc' (int): The measure index.
            "annotations": The harmonic annotation of each measure.
            'timesig' (str): The time signature of each measure.
            'start' (Fraction): The "offset" position at the start of each measure, in
                                whole notes since the beginning of the piece.
            'act_dur' (Fraction): The duration of the measure, in whole notes.
            'mc_offset' (Fraction): The starting position of this measure, in whole notes
                                 after the most recent downbeat.
            'next' (int): The measure index of the measure that follows each one.

    2. chords_df : pd.DataFrame
        A chords_df with the following columns:
            'mn' (int): The measure number
            'annotations' (str): The annotations of each measure
    """
    # Lists to compute and add to output
    time_signatures = []
    # annotations = []
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
    
    # J: Add Annotations to a list

    chord_symbols = []
    chords_by_measure = []
    for element in m21_score.recurse().getElementsByClass(ChordSymbol):
        measure = element.getContextByClass("Measure")
        # element.activeSite.remove(element) # Do we need that?
        chord_symbols.append(element.figure)# OR str(element) OR element.pitchedCommonName
        chords_by_measure.append(element.measureNumber) 



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
        # annotations.append(chord_symbols)
        starts.append(offset)
        time_signatures.append(time_signature)
        df_offsets.append((offset - ts_epoch) % ts_duration)
        mns.append(measure.measureNumber)
        
    mcs = list(range(len(mns)))

    return pd.DataFrame(
        {
            "mc": mcs,
            "mn": mns,
            "timesig": time_signatures,
            "start": starts,
            "act_dur": lengths,
            MEASURE_OFFSET: df_offsets,
            "next": mcs[1:] + [pd.NA],
        }
    ), pd.DataFrame(
        {
        # "mc": mcs,
        "mn": chords_by_measure,
        "annotations": chord_symbols
        }
    )


"""
tsv conversion function
"""
def score_to_tsv(
    music_xml_path: Union[Path, str],
    output_dir: Union[Path, str] = None # make it optional 
):
    # Convert to Path immediately to use .is_dir() later on it
    music_xml_path = Path(music_xml_path)


    if output_dir is None:
        output_dir = music_xml_path.parent # save in parent folder of the musicxml-files
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True) # create new directory

    # Collect files
    if music_xml_path.is_dir():
        all_music_xml = [
            Path(x) for x in sorted(
                glob.glob(str(music_xml_path / "**" / "*.mxl"), recursive=True)
                + glob.glob(str(music_xml_path / "**" / "*.xml"), recursive=True)
            )
        ]
    else: # when passing in a single file
        all_music_xml = [Path(music_xml_path)]
    
    for music_xml_path in all_music_xml:
        m21_score = parse(music_xml_path)
        measures_df, chords_df = get_chords_and_measures_df_from_m21_score(m21_score)
        
        # Auto-generate filename from each input file
        output_path = output_dir / f"{music_xml_path.stem}.tsv"
        chords_df.to_csv(output_path, sep="\t", index=False)

    return chords_df # , measures_df

score_to_tsv(
    music_xml_path="mels_to_harmonize/",
    output_dir="tests_tsv"  # folder only
)