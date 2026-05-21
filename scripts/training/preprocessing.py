"""
Preprocessing functions for model training.

Frame-by-frame: Extract every frames and bounding box annotations
"""

from pathlib import Path
import sys
import zipfile
import shutil
from typing import List
import cv2
import yaml

# import argparse
from sklearn.model_selection import train_test_split
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.matching import parse_labelled_name, parse_unlabelled_name
from config import UNLABELLED_CLIPS_DIR, LABELLED_CLIPS_DIR

train_path = Path("data/processed/pipeline_testing/train.csv").absolute()

OUTPUT_DIR = Path("data/processed/pipeline_testing/yolo_format").absolute()

### PROBABLY BEST TO SAVE THESE TO ONEDRIVE!!! -- too many frames and txt files!


def extract_labels(
    input_paths: List[Path],
    output_dir,
    labels_root: Path,
    split: str,
    FORCE: bool = False,
):
    """Extract bounding box labels from .zip files (CVAT Output)"""

    # Add subdirectories to output directory
    output_dir = Path(output_dir) / "labels" / split

    # Do nothing if files already exist
    if Path(output_dir).exists() and not FORCE:
        print(
            f"Files already extracted at {Path(__file__) / Path(output_dir)}"
        )  # change to Path(__file__)
    else:
        # Delete path and files if they already exist
        if output_dir.exists() and output_dir.is_dir():
            # Recursively deletes the directory and all contents
            shutil.rmtree(output_dir)

        # Loop over zip file paths (CVAT Outputs)
        for input_path in input_paths:

            # Standardie Path to Posix Standard
            input_path = labels_root / input_path.replace("\\", "/")

            # Get numeric id and part id of labelled output
            numeric_id, part_id = parse_labelled_name(str(input_path.name))

            if part_id:
                # Format nicely
                part_id = f"part0{part_id}"

            # Target folder in zip file
            target_folder = "obj_train_data/"

            ## WHAT HAPPENS IF THIS THROWS AN ERROR!

            # Look in zip folder
            with zipfile.ZipFile(input_path, "r") as zip_ref:

                # List all files in zip folder
                all_files = zip_ref.namelist()

                # Isolate only the .txt files belonging to the target folder hierarchy
                files_to_extract = sorted(
                    [
                        f
                        for f in all_files
                        if f.startswith(target_folder) and ".txt" in f
                    ]
                )

                for file in files_to_extract:
                    # Extract individual files explicitly to target destination
                    zip_ref.extract(file, output_dir)

            # target_folder = "obj_train_data/"
            # path to target folder in output dir (extraction adds target folder in output hierarchy)
            target_folder = output_dir / target_folder

            if target_folder.exists() and target_folder.is_dir():
                # Iterate through all files inside the sub-folder
                for file_path in target_folder.iterdir():
                    if file_path.is_file():
                        # Define target path (e.g., extraction_output/train/0000_{part}_frame_000000.txt)
                        target_path = (
                            output_dir
                            / f"{int(numeric_id):04}_{part_id}_{str(file_path.name)}"
                        )

                        # Atomic filesystem move (Metadata update only, no disk write)
                        file_path.rename(target_path)

                # Delete the now-empty target folder from output dir
                target_folder.rmdir()
                print(f"Files saved to {output_dir}")
            else:
                print(f"{target_folder} structure not found or already processed.")


def extract_frames(
    videos: List[str], videos_root: Path, output_dir, split: str, FORCE: bool = False
):
    """Extract frames from all videos

    Input: list of realtive paths to unlabelled videos
    output: extract all frames from unlabelled videos into train or validation set

    should probably take in the root directory as well!
    requires parse functionality so needst to call in parsing functions
    """

    # Output dir
    output_dir = Path(output_dir) / "images" / split
    # Rewrite files on FORCE
    if Path(output_dir).exists() and not FORCE:
        print(f"Files already extracted at {Path(__file__) / Path(output_dir)}")
    else:
        # Delete path if it already exists
        if output_dir.exists() and output_dir.is_dir():
            # Recursively deletes the directory and all contents
            shutil.rmtree(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        for video_file in videos:

            # Standardize Path to Posix Standard
            video_file = videos_root / video_file.replace("\\", "/")

            # Print working video...
            print(f"Extracting frames from {video_file.name}...")

            # Get numeric id and part id of video clip
            numeric_id, part_id = parse_unlabelled_name(str(video_file.name))
            if part_id:
                part_id = f"part0{part_id}"

            # Video capture
            cap = cv2.VideoCapture(str(video_file))
            frame_idx = 0

            while cap.isOpened():
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break

                # Output path
                frame_path = (
                    output_dir
                    / f"{int(numeric_id):04}_{part_id}_frame_{frame_idx:06d}.jpg"
                )

                # Write frame to output dir
                cv2.imwrite(str(frame_path), frame)
                frame_idx += 1

            # release video
            cap.release()
            print(f"Frames saved to {output_dir}")


def create_yaml():
    # Quick YAML creation
    with open("data/processed/pipeline_testing/yolo_format/data.yaml", "w") as f:
        yaml.dump(
            {
                "path": "data/processed/pipeline_testing/yolo_format",
                "train": "images/train",
                "val": "images/val",
                "nc": 1,
                "names": {0: "cross-sucking"},
            },
            f,
        )

    print("Created data.yaml")


# UPDATE TO CLEAN AND TAKE IN ARGUMENTS
def main():
    train_df = pd.read_csv(train_path, index_col=0)

    train, val = train_test_split(
        train_df, test_size=0.4, train_size=0.6, random_state=1234
    )

    # Take the first data frame and ....
    # Extract frames to obj_img_data/train
    # Extract .txt files to obj_train_data/train

    extract_labels(
        input_paths=train["labelled_clip_relative_path"],
        labels_root=LABELLED_CLIPS_DIR,
        output_dir=OUTPUT_DIR,
        split="train",
        FORCE=True,
    )
    extract_frames(
        videos=train["clip_relative_path"],
        videos_root=UNLABELLED_CLIPS_DIR,
        output_dir=OUTPUT_DIR,
        split="train",
        FORCE=True,
    )

    # # tTake the second data frame and ...
    #     # Extract frames to obj_img_data/val
    #     # Extract .txt files to obj_train_data/val
    extract_labels(
        input_paths=val["labelled_clip_relative_path"],
        labels_root=LABELLED_CLIPS_DIR,
        output_dir=OUTPUT_DIR,
        split="val",
        FORCE=True,
    )
    extract_frames(
        videos=val["clip_relative_path"],
        videos_root=UNLABELLED_CLIPS_DIR,
        output_dir=OUTPUT_DIR,
        split="val",
        FORCE=True,
    )

    create_yaml()


if __name__ == "__main__":
    main()
