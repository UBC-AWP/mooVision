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
import pandera as pa

sys.path.append(str(Path(__file__).parent.parent.parent))

from schema import schema, processed_schema
from config import (
    UNLABELLED_CLIPS_DIR,
    LABELLED_CLIPS_DIR,
    SOURCE_VIDEOS_DIR,
    INDEX_PATH,
)

PROCESSED_INDEX_OUTPUT = Path("data/processed/processed_clips_index.csv").absolute()
RAW_INDEX_OUTPUT = Path("data/raw/all_clips_index_raw.csv").absolute()


def read_data(
    index_path: str,
) -> pd.DataFrame:

    # Read in Index from any format and return df
    p = Path(index_path)

    if not p.exists():
        raise FileNotFoundError(f"{p} does not exist.")

    loaders = {
        ".csv": pd.read_csv,
        ".xlsx": pd.read_excel,
        ".parquet": pd.read_parquet,
        ".json": pd.read_json,
        ".tsv": lambda f: pd.read_csv(f, sep="\t"),
    }
    loader = loaders.get(p.suffix.lower())

    if not loader:
        raise ValueError(f"Unsupported format: {p.suffix}")
    else:
        return loader(index_path)


def validate_data(df: pd.DataFrame, df_schema: pa.DataFrameSchema) -> pd.DataFrame:
    # Validate df against scheme, return all errors
    try:
        return df_schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        raise ValueError(f"Data validation failed on raw index: {e}") from e


def save_data(df: pd.DataFrame, path: Path, force: bool = False) -> None:
    # Save df to Path
    if path.exists() and not force:
        print(f"{path} already exists.")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Saved raw file to {path}")


def filter_existing_clips(df: pd.DataFrame, processed_index_output: str, force: bool = False) -> pd.DataFrame:
    # Process Raw Index df
    # Assumes df has already been validated for correct schema.
    path = Path(processed_index_output)
    
    if processed_index_output.exists() and not force:
        print(f"{processed_index_output} already exists.")
    else:

        processed_index_output.parent.mkdir(parents=True, exist_ok=True)

        # Create filter
        exists = []
        for p in all_clips_index["clip_relative_path"]:

            path = str(unlabelled_clips_dir / p)
            path = path.replace("\\", "/")

            if Path(path).exists():
                exists.append(True)
            else:
                exists.append(False)

        # Filter for existing clips
        available_clips_index = all_clips_index[exists].copy()

        # Report Dropped Clips
        n_dropped = len(all_clips_index) - exists.count(True)
        if n_dropped:
            missing = all_clips_index[~pd.Series(exists)]["clip_name"].tolist()
            warnings.warn(f"Dropped {n_dropped} clips with no file on disk:\n{missing}")



def match_annotation_paths() -> pd.DataFrame:
    pass


def read_data_from_index_file(
    index_path: str,
    unlabelled_clips_dir: Path,
    labelled_clips_dir: Path,
    source_videos_dir: Path,  # filter out bad source video too!
    raw_index_output: Path,
    processed_index_output: Path,
    force: bool = False,
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
    index_path : str
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
    
        # Get name and path for all labelled cross sucking files (.zip files)
        paths = []
        for path in labelled_clips_dir.rglob(
            "*.zip"
        ):  # assumes all .zip files are annotations for CS events
            paths.append(
                (path.name, str(Path(*path.parts[-4:])))
            )  # Use relative path; assumes file structure.

        # --- Add labelled CS paths to index.csv ---

        # Get unlabelled clip names (for matching)
        clips = available_clips_index["clip_name"]
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
        available_clips_labels_index = available_clips_index[
            ~available_clips_index["labelled_clip_relative_path"].isna()
        ]

        # Report Dropped Clips
        n_dropped = len(available_clips_labels_index) - len(available_clips_index)
        if n_dropped:
            dropped = available_clips_index[
                available_clips_index["labelled_clip_relative_path"].isna()
            ]
            warnings.warn(
                f"Dropped {n_dropped} clips with no matching annotation files\n:{dropped}"
            )

        # --- Validate Processed Data Frame ---

        try:
            available_clips_labels_index = processed_schema.validate(
                available_clips_labels_index, lazy=True
            )
        except pa.errors.SchemaErrors as e:
            raise ValueError(f"Data validation failed for processed data: {e}") from e

        # --- Read to CSV
        available_clips_labels_index.to_csv(processed_index_output, index=False)
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
        help="Path to annotations directory.",
    )
    parser.add_argument(
        "--source_videos_dir",
        default=SOURCE_VIDEOS_DIR,
        help="Path to source videos directory.",
    )
    parser.add_argument(
        "--raw_index_output",
        default=RAW_INDEX_OUTPUT,
        help="Output path for raw index file.",
    )
    parser.add_argument(
        "--processed_index_output",
        default=PROCESSED_INDEX_OUTPUT,
        help="Output path for processed index file.",
    )
    parser.add_argument(
        "--FORCE",
        default=False,
        action="store_true",
        help="Force overwrite of existing data.",
    )
    return parser.parse_args()


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
