import glob
import os
from pathlib import Path
import h5py

# Alphabetical list of scores matching create_h5_data.py
xml_files = sorted(glob.glob("choro_h5_creation/scores/*.xml"))


def get_split_pieces(splits=("valid", "test"), seed=0):
    """Return {piece_stem: split_name} for the requested splits."""
    piece_to_split = {}
    for split in splits:
        h5_file = f"h5_data/ChordTransitionDataset_{split}_seed_{seed}.h5"
        with h5py.File(h5_file, "r") as f:
            for fid in f["file_ids"][:]:
                stem = Path(xml_files[int(fid)]).stem
                piece_to_split[stem] = split
    return piece_to_split


if __name__ == "__main__":
    for split in ["train", "valid", "test"]:
        h5_file = f"h5_data/ChordTransitionDataset_{split}_seed_0.h5"
        with h5py.File(h5_file, "r") as f:
            file_ids = f["file_ids"][:]
            print(f"\n=== {split.upper()} ({len(file_ids)} pieces) ===")
            for fid in file_ids:
                print(f"  [{fid:2d}] {os.path.basename(xml_files[fid])}")
