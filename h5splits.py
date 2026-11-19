import glob, os, h5py

# Alphabetical list of scores matching create_h5_data.py
xml_files = sorted(glob.glob("choro_h5_creation/scores/*.xml"))

for split in ["train", "valid", "test"]:
    h5_file = f"h5_data/ChordTransitionDataset_{split}_seed_0.h5"
    with h5py.File(h5_file, "r") as f:
        file_ids = f["file_ids"][:]
        print(f"\n=== {split.upper()} ({len(file_ids)} pieces) ===")
        for fid in file_ids:
            print(f"  [{fid:2d}] {os.path.basename(xml_files[fid])}")