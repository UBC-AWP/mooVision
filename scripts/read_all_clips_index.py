"""
Module to read all_clips_index.csv.

Filters for videos existing in file path, and appends corresponding labelled video paths.
"""

from pathlib import Path
import sys
import pandas as pd
import warnings
from matching import is_match

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
    unlabelled_clips_path: Path,
    labelled_clips_path: Path,
    source_videos_path: Path,  # filter out bad source video too!
    out_index_dir: Path,
    processed_index_dir: Path,
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

    Naming Convention:
        Convention:
            CS_{clip_number)_{Weaning_period}_d{day_number}_p{pen_number}_cow{cow_identifier}_{ddmmyyyy}_{source_video_base_name}_{clip_start_time_s}_{clip_end_time_s}.mp4
        Example:
            CS_0001_POSTWEAN_d1_p2_cow6_02112025_ch02-20251102075200_684_702.mp4


    This is an internal funciton, input paths should be called via config.py.

    Examples
    --------
    >>> read_data_from_index(ROOT / INDEX_PATH)
    """

    ### FILTER FOR EXISTING PATHS AND SOURCE PATHS

    # --- Read in Index (from OneDrive) ---
    if index_path.exists():
        all_clips_index = pd.read_csv(index_path)
    else:
        raise FileNotFoundError(f"{index_path} does not exist.")

    # --- Save Raw index to disk ---

    # Do nothing if raw index already exists
    if out_index_dir.exists() and not FORCE:
        print(f"{out_index_dir} already exists.")

    # Save raw index to repo if it does not, or if forced
    else:
        # Create Path
        out_index_dir.parent.mkdir(parents=True, exist_ok=True)
        # Save Raw Index
        all_clips_index.to_csv(out_index_dir)
        # Print Confirmation
        print(f"Saved to {out_index_dir}")

    # --- Process Raw Index ---

    # Do nothing if processed file already exists
    if processed_index_dir.exists() and not FORCE:
        print(f"{processed_index_dir} already exists.")
    else:
        # Create directory
        processed_index_dir.parent.mkdir(parents=True, exist_ok=True)

        # Create filter list for index
        exists = []
        for p in all_clips_index["clip_relative_path"]:

            # Create Consistent Path Structure in Posix Standard ("/")
            path = str(unlabelled_clips_path / p)
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
        for path in labelled_clips_path.rglob(
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

        available_clips_index.to_csv(processed_index_dir)
        print(f"Saved to {processed_index_dir}")


def main():
    read_data_from_index(
        index_path=INDEX_PATH,
        unlabelled_clips_path=UNLABELLED_CLIPS_DIR,
        labelled_clips_path=LABELLED_CLIPS_DIR,
        source_videos_path=SOURCE_VIDEOS_DIR,  # filter out bad source video too!
        out_index_dir=RAW_INDEX_OUTPUT,
        processed_index_dir=PROCESSED_INDEX_OUTPUT,
        FORCE=True,
    )


if __name__ == "__main__":
    main()
