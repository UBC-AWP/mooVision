"""
Module to read all_clips_index.csv.

Filters for videos existing in file path, and appends corresponding labelled video paths.
"""

from pathlib import Path
import sys
import pandas as pd
import warnings
from matching import is_match
import argparse

sys.path.append(str(Path(__file__).parent.parent))

from config import (
    UNLABELLED_CLIPS_DIR,
    LABELLED_CLIPS_DIR,
    SOURCE_VIDEOS_DIR,
    INDEX_PATH,
)

PROCESSED_INDEX_OUTPUT = Path("data/processed/processed_clips_index.csv").absolute()
RAW_INDEX_OUTPUT = Path("data/raw/all_clips_index_raw.csv").absolute()


def read_data_from_index(
    index_path: Path,
    unlabelled_clips_dir: Path,
    labelled_clips_dir: Path,
    source_videos_dir: Path,  # filter out bad source video too!
    raw_index_output: Path,
    processed_index_output: Path,
    FORCE: bool = False,
):
    """
    Reads in data from index.csv file, adds labelled cross sucking clip
    (CVAT Output) paths to matching unlabelled cross sucking clips.
    Filters for rows which have a source video, unlabelled cross sucking clip,
    and labelled cross sucking output whose paths exists within the given
    directories.

    Saves a raw index.csv to the directory specified at out_index_dir, and
    an output index (filtered csv file) to directory specified at
    output_index_dir.

    Parameters
    ----------
    index_path : Path
        Path to index.csv file.
    unlabelled__clips_path : Path
        Path to folder with unlabelled cross sucking clips.
    labelled_clips_path : Path
        Path to folder with labelled cross sucking clips (CVAT Output).
    source_videos_path : Path
        Path to raw source videos.
    out_index_dir : Path
        Path to raw index.csv file
    output_index_dir : Path
        Path to output index.csv file
    FORCE : bool
        Force rewriting of indices if they already exist


    Returns
    -------
    None
        The function writes to disk and does not return a value.

    Raises
    ------
    FileNotFoundError
        If the path to index.csv does not exist does not exist.

    Notes
    -----
    This function ensures that the index file contains relative paths to the
    source files and cross sucking clips, as well as other specific column
    formats. See documentation for more details.

    Naming Conventions:
        Convention:
            CS_{clip_number)_{Weaning_period}_d{day_number}_p{pen_number}_cow{cow_identifier}_{ddmmyyyy}_{source_video_base_name}_{clip_start_time_s}_{clip_end_time_s}.mp4
        Example:
            CS_0001_POSTWEAN_d1_p2_cow6_02112025_ch02-20251102075200_684_702.mp4


    This is an internal funciton, input paths should be called via config.py.

    Examples
    --------
    >>> read_data_from_index(INDEX_PATH)
    """

    ### FILTER FOR EXISTING PATHS AND SOURCE PATHS
    index_path = Path(index_path).absolute()
    # --- Read in Index (from OneDrive) ---
    if index_path.exists():
        all_clips_index = pd.read_csv(index_path)
    else:
        raise FileNotFoundError(f"{index_path} does not exist.")

    # --- Save Raw index to disk ---

    # Do nothing if raw index already exists
    if raw_index_output.exists() and not FORCE:
        print(f"{raw_index_output} already exists.")

    # Save raw index to repo if it does not, or if forced
    else:
        # Create Path
        raw_index_output.parent.mkdir(parents=True, exist_ok=True)
        # Save Raw Index
        all_clips_index.to_csv(raw_index_output)
        # Print Confirmation
        print(f"Saved to {raw_index_output}")

    # --- Process Raw Index ---

    # Do nothing if processed file already exists
    if processed_index_output.exists() and not FORCE:
        print(f"{processed_index_output} already exists.")
    else:
        # Create directory
        processed_index_output.parent.mkdir(parents=True, exist_ok=True)

        # Create filter list for index
        exists = []
        for p in all_clips_index["clip_relative_path"]:

            # Create Consistent Path Structure in Posix Standard ("/")
            path = str(unlabelled_clips_dir / p)
            path = path.replace("\\", "/")

            # If clip exists in unlabelled_clips_path append True
            if Path(path).exists():
                exists.append(True)
            else:
                exists.append(False)

        # Keep only clips that exist at unlabelled_clips_path
        available_clips_index = all_clips_index[exists]

        # Get name and path for all labelled cross sucking files (.zip files)
        paths = []
        for path in labelled_clips_dir.rglob(
            "*.zip"
        ):  # assume all .zip files are labelled CS
            paths.append(
                (path.name, str(Path(*path.parts[-4:])))
            )  # Use relative path; assumes file structure.

        # --- Add labelled CS paths to index.csv ---

        # Get unlabelled clip names (for matching)
        clips = available_clips_index["clip_name"]

        # Create labelled Paths column
        labelled_paths = [None] * len(clips)

        # Loop over unlabelled names
        for i, name in enumerate(clips):
            matches = []

            # Search labelled names for matches (based on numeric id and part number)
            for path in paths:
                if is_match(name, path[0]):
                    matches.append(path[1])

            # Multiple matches raises error; all clips should have unqiue identifiers, except fixed videos
            if len(matches) > 2:
                raise ValueError(
                    f"Expected exactly 1 labelled cross sucking file for ID {name}, "
                    f"but found {len(matches)} matches:\n"
                    f"{matches}"
                )
            # Handle multiple matches with fixed video; default to base clip and warn user
            elif len(matches) == 2:  # assumes one base path and one fixed-video path
                if ("fixed_clips" in matches[0]) and ("fixed_clips" not in matches[1]):
                    labelled_paths[i] = matches[1]
                    warnings.warn(
                        f"\nAmbiguity Warning: {name} returned multiple matches: {matches}. \n"
                        f"Defaulting to use the base clip option: '{matches[1]}'.\n",
                        # f"If you want to use fixed clips, please run ",
                        category=UserWarning,
                        stacklevel=2,
                    )
                elif ("fixed_clips" in matches[1]) and (
                    "fixed_clips" not in matches[0]
                ):
                    labelled_paths[i] = matches[0]
                    warnings.warn(
                        f"\nAmbiguity Warning: {name} returned multiple matches: {matches}. \n"
                        f"Defaulting to use the base clip option: '{matches[0]}'.\n",
                        # f"If you want to use fixed clips, please run ",
                        category=UserWarning,
                        stacklevel=2,
                    )
                else:
                    raise ValueError(
                        f"Expected exactly 1 labelled cross sucking file for ID {name}, "
                        f"but found {len(matches)} matches:\n"
                        f"{matches}"
                    )
            # Add single match to labelled_paths, None if no match
            elif len(matches) == 1:
                labelled_paths[i] = matches[0]
            else:
                labelled_paths[i] = None

        # Add labelled Paths to index
        available_clips_index["labelled_clip_relative_path"] = labelled_paths

        #  Filter out rows with no labelled clips
        available_clips_index = available_clips_index[
            ~available_clips_index["labelled_clip_relative_path"].isna()
        ]

        available_clips_index.to_csv(processed_index_output)
        print(f"Saved to {processed_index_output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Data splitting.")
    parser.add_argument(
        "--index_path",
        default=INDEX_PATH,
        help="Path to data index.",
    )
    parser.add_argument(
        "--unlabelled_clips_dir",
        default=UNLABELLED_CLIPS_DIR,
        help="Path to cross-sucking clips directory.",
    )
    parser.add_argument(
        "--labelled_clips_dir",
        default=LABELLED_CLIPS_DIR,
        action="store_true",
        help="Path to annotations directory.",
    )
    parser.add_argument(
        "--source_videos_dir",
        default=SOURCE_VIDEOS_DIR,
        action="store_false",
        help="Path to source videos directory.",
    )
    parser.add_argument(
        "--raw_index_output",
        default=RAW_INDEX_OUTPUT,
        action="store_false",
        help="Output path for raw index file.",
    )
    parser.add_argument(
        "--processed_index_output",
        default=PROCESSED_INDEX_OUTPUT,
        action="store_false",
        help="Output path for processed index file.",
    )
    parser.add_argument(
        "--FORCE",
        default=False,
        action="store_true",
        help="Force overwrite of existing data.",
    )
    return parser.parse_args()


# run with

if __name__ == "__main__":
    args = parse_args()
    read_data_from_index(
        index_path=args.index_path,
        unlabelled_clips_dir=args.unlabelled_clips_dir,
        labelled_clips_dir=args.labelled_clips_dir,
        source_videos_dir=args.source_videos_dir,
        raw_index_output=args.raw_index_output,
        processed_index_output=args.processed_index_output,
        FORCE=args.FORCE,
    )
