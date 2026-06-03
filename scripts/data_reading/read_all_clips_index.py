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
    path: Path,
) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")

    loaders = {
        ".csv": pd.read_csv,
        ".xlsx": pd.read_excel,
        ".parquet": pd.read_parquet,
        ".json": pd.read_json,
        ".tsv": lambda f: pd.read_csv(f, sep="\t"),
    }
    loader = loaders.get(path.suffix.lower())

    if not loader:
        raise ValueError(f"Unsupported format: {path.suffix}")
    else:
        return loader(path)


def validate_data(
    df: pd.DataFrame,
    df_schema: pa.DataFrameSchema,
) -> pd.DataFrame:
    # Check inputs
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df should be of type pd.DataFrame, got {type(df)}")
    if not isinstance(df_schema, pa.DataFrameSchema):
        raise TypeError(
            f"df should be of type pa.DataFrameSchema, got {type(df_schema)}"
        )
    # Check df not empty
    if df.empty:
        raise ValueError("df is empty.")
    # Validate df against schema, return all errors
    try:
        return df_schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        raise ValueError(f"Data validation failed: {e}") from e


def save_data(df: pd.DataFrame, path: Path) -> None:
    # Check inputs
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df should be of type pd.DataFrame, got {type(df)}")
    # Check df not empty
    if df.empty:
        raise ValueError("df is empty, nothing to save")
    # Check format is csv
    if path.suffix != ".csv":
        raise ValueError(f"Expected a .csv path, got {path.suffix}")
    # Write data to csv
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved raw file to {path}")


def filter_existing_clips(
    df: pd.DataFrame,
    clips_dir: Path,
) -> pd.DataFrame:

    # Check inputs
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be of type pd.Dataframe, got {type(df)}")
    # Check df is not empty
    if df.empty:
        raise ValueError("df is empty")

    # Process Raw Index df
    if not clips_dir.exists():
        raise FileNotFoundError(f"{clips_dir} does not exist.")

    # Create filter
    exists = []
    for p in df["clip_relative_path"]:  # Relies on Schema Validation

        # Consistent Path Structure
        path = str(clips_dir / p)
        path = path.replace("\\", "/")

        if Path(path).exists():
            exists.append(True)
        else:
            exists.append(False)

    # Filter clips
    filtered_df = df[exists].copy()

    # Raise warning if filtered_df is empty.
    if filtered_df.empty:
        warnings.warn("No clips remaining after filtering.")

    # Report Dropped Clips
    n_dropped = len(df) - exists.count(True)
    if n_dropped:
        dropped = df[~pd.Series(exists)]["clip_name"].tolist()
        warnings.warn(f"Dropped {n_dropped} clips with no file on disk:\n{dropped}")

    return filtered_df


def get_label_paths(labels_dir: Path) -> list[tuple[str, str]]:
    """
    Notes:
        Assumes all `.zip` files in the labels_dir directory are
        annotations for CS events. Ensure there are no errant zip files
        in the data as this function will read in all .zip files which
        may cause errors down the line.
    """
    # Get name and path for all labelled cross sucking files (.zip files)
    if not labels_dir.exists():
        raise FileNotFoundError(f"{labels_dir} does not exist.")
    paths = []
    for path in labels_dir.rglob(
        "*.zip"
    ):  # assumes all .zip files are annotations for CS events
        paths.append(
            (path.name, str(Path(*path.parts[-4:])))
        )  # Uses relative path; assumes file structure.

    if not paths:
        raise FileNotFoundError(f"No .zip files found in {labels_dir}")

    return paths


def match_label_paths(
    df: pd.DataFrame, label_paths: list[tuple[str, str]]
) -> list[str | None]:

    # Check inputs
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be of type pd.Dataframe, got {type(df)}")
    # Check for emty inputs
    if df.empty:
        raise ValueError("df is empty")
    if not label_paths:
        raise ValueError("label_paths is empty")

    # --- Add labelled CS paths to index.csv ---

    # Get unlabelled clip names (for matching)
    clips = df["clip_name"]
    labelled_paths = [None] * len(clips)

    # Loop over unlabelled names
    for i, name in enumerate(clips):
        matches = []

        # Search labelled names for matches (based on numeric id and part number)
        for path in label_paths:
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
            elif ("fixed_clips" in matches[1]) and ("fixed_clips" not in matches[0]):
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

    # Warn if no matches found for any clips
    if all(p is None for p in labelled_paths):
        warnings.warn("No matches found for any clips.")

    return labelled_paths


def add_label_paths(df: pd.DataFrame, labelled_paths: list[str]) -> pd.DataFrame:

    # Check inputs
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be of type pd.Dataframe, got {type(df)}")
    # Check for emty inputs
    if df.empty:
        raise ValueError("df is empty")
    if not labelled_paths:
        raise ValueError("labelled_paths is empty")

    # Add labelled Paths to df
    df["labelled_clip_relative_path"] = labelled_paths

    return df


def filter_label_paths(df: pd.DataFrame) -> pd.DataFrame:

    # Check inputs
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be of type pd.Dataframe, got {type(df)}")
    # Check for emty inputs
    if df.empty:
        raise ValueError("df is empty")

    #  Filter for rows with labels
    filtered_df = df[~df["labelled_clip_relative_path"].isna()]

    # Report Dropped Clips
    n_dropped = len(filtered_df) - len(df)
    if n_dropped:
        dropped = df[df["labelled_clip_relative_path"].isna()]["clip_name"].to_list()
        warnings.warn(
            f"Dropped {n_dropped} clips with no matching annotation files\n:{dropped}"
        )

    if filtered_df.empty:
        warnings.warn("No clips remaiing after filtering.")

    return filtered_df


def read_data_from_index_file(
    index_path: Path,
    clips_dir: Path,
    labels_dir: Path,
    source_dir: Path,  # filter out bad source video too!
    raw_output: Path,
    processed_output: Path,
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
    index_path : Path
        Path to index.csv file.
    clips_dir : Path
        Path to folder with unlabelled cross sucking clips.
    labels_dir : Path
        Path to folder with labelled cross sucking clips (CVAT Output).
    source_dir: Path
        Path to raw source videos.
    raw_output : Path
        Path to output raw index file
    processed_output : Path
        Path to output processed index file
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
    # Read in Raw Data
    df_raw = read_data(index_path)
    df_raw_validated = validate_data(df_raw, schema)
    # Save Raw index
    if raw_output.exists() and not force:

        print(f"File already exists at {raw_output}")

    else:

        save_data(df_raw_validated, raw_output, force=force)

        if processed_output.exists() and not force:

            print(f"File already exists at {processed_output}")

        else:

            df_filtered = filter_existing_clips(
                df=df_raw_validated,
                output_path=processed_output,
                clips_dir=clips_dir,
                force=force,
            )

            paths = get_label_paths(labels_dir=labels_dir)
            df_labels = add_label_paths(df=df_filtered, label_paths=paths)
            df_processed = filter_label_paths(df_labels)
            df_processed_validated = validate_data(df_processed, processed_schema)
            save_data(df=df_processed_validated, path=processed_output, force=force)


def parse_args():
    parser = argparse.ArgumentParser(description="Data splitting.")
    parser.add_argument(
        "--index_path",
        default=INDEX_PATH,
        type=Path,
        help="Path to data index.",
    )
    parser.add_argument(
        "--unlabelled_clips_dir",
        default=UNLABELLED_CLIPS_DIR,
        type=Path,
        help="Path to cross-sucking clips directory.",
    )
    parser.add_argument(
        "--labelled_clips_dir",
        default=LABELLED_CLIPS_DIR,
        type=Path,
        help="Path to annotations directory.",
    )
    parser.add_argument(
        "--source_videos_dir",
        default=SOURCE_VIDEOS_DIR,
        type=Path,
        help="Path to source videos directory.",
    )
    parser.add_argument(
        "--raw_index_output",
        default=RAW_INDEX_OUTPUT,
        type=Path,
        help="Output path for raw index file.",
    )
    parser.add_argument(
        "--processed_index_output",
        default=PROCESSED_INDEX_OUTPUT,
        type=Path,
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
    read_data_from_index_file(
        index_path=args.index_path,
        unlabelled_clips_dir=args.unlabelled_clips_dir,
        labelled_clips_dir=args.labelled_clips_dir,
        source_videos_dir=args.source_videos_dir,
        raw_index_output=args.raw_index_output,
        processed_index_output=args.processed_index_output,
        FORCE=args.FORCE,
    )
