from pathlib import Path
from typing import Union
import music21
import pandas as pd
import numpy as np
from music21.converter import parse
from music21.harmony import ChordSymbol
import glob

def get_chords_df_from_m21_score(m21_score: music21.stream.Score) -> pd.DataFrame:
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
    1. chords_df : pd.DataFrame
        A chords_df with the following columns:
            'mn' (int): The measure number
            'annotations' (str): The annotations of each measure
    """

    # J: Add Annotations to a list

    chord_symbols = []
    chords_by_measure = []
    for element in m21_score.recurse().getElementsByClass(ChordSymbol):
        measure = element.getContextByClass("Measure")
        # element.activeSite.remove(element) # Do we need that?
        chord_symbols.append(element.figure)# OR str(element) OR element.pitchedCommonName
        chords_by_measure.append(element.measureNumber) 

    return pd.DataFrame(
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
        chords_df = get_chords_df_from_m21_score(m21_score)
        
        # Auto-generate filename from each input file
        output_path = output_dir / f"{music_xml_path.stem}.tsv"
        chords_df.to_csv(output_path, sep="\t", index=False)

    return chords_df


# CALL
score_to_tsv(
    music_xml_path="mels_to_harmonize/",
    output_dir="outputs_xml_to_tsv"  # folder only
)