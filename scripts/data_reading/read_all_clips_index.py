"""
Module to read all_clips_index.csv.

Filters for videos existing in file path, and appends corresponding labelled video paths.

Notes
-----
Naming Conventions:
    Convention:
        CS_{clip_number)_{Weaning_period}_d{day_number}_p{pen_number}_cow{cow_identifier}_{ddmmyyyy}_{source_video_base_name}_{clip_start_time_s}_{clip_end_time_s}.mp4
    Example:
        CS_0001_POSTWEAN_d1_p2_cow6_02112025_ch02-20251102075200_684_702.mp4
"""

from pathlib import Path
import sys
import pandas as pd
import warnings
import argparse
import pandera.pandas as pa
from .matching import is_match
from .schema import schema, processed_schema

sys.path.append(str(Path(__file__).parent.parent.parent))

from config import (
    UNLABELLED_CLIPS_DIR,
    LABELLED_CLIPS_DIR,
    SOURCE_VIDEOS_DIR,
    INDEX_PATH,
    ROOT_DIR,
)

PROCESSED_INDEX_OUTPUT = ROOT_DIR / "data/processed/processed_clips_index.csv"
RAW_INDEX_OUTPUT = ROOT_DIR / "data/raw/all_clips_index_raw.csv"


def read_data(
    path: Path,
) -> pd.DataFrame:
    """
    Read data from path and return it as a pandas dataframe. This is an
    internal function meant for use in `read_data_from_index_file()`

    Parameters
    ----------
    path : Path
        Path to data file to read in

    Returns
    -------
    pd.DataFrame
        A pandas dataframe of the existing file.

    Raises
    ------
    FileNotFoundError
        If the file path to the data frame does not exist.

    Notes
    -----
    Handles multiple file types: .csv, .xlsx, .parquet, .json, .tsv.

    Examples
    --------
    .. code-block:: python

        from config import ROOT_DIR
        INDEX_PATH = ROOT_DIR / "cross_sucking_clips" / "all_clips_index.csv"
        df = read_data(INDEX_PATH)
    """

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
    """
    This is an internal function meant for use in `read_data_from_index_file()`

    Validate a pandas dataframe against a data schema using pandera. Takes in a
    data frame and a dataframe schema and validates the data frame against the
    schema. This is deisgned to enforce data strucutres to ensure the pipeline
    runs as intended through downstream usage.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to validate.
    df_schema : pa.DataFrameSchema
        The pandera dataframe schema to validata against.

    Returns
    -------
    pd.DataFrame
        The validated dataframe.

    Raises
    ------
    TypeError
        If inputs are not a pandas DataFrame or a pandera DataFrameSchema
    ValueError
        If the dataframe is empty, or if there are any validation errors
        when calling df_schema.validate(...)

    Notes
    -----
    There are two data schemas for the MooVision project, held in schema.py.
    These are for validating the raw data file and the processed data frame.

    Examples
    --------
    >>> import pandas as pd
    >>> import pandera as pa
    >>> schema = pa.DataFrameSchema({
    ...     "age": pa.Column(int, pa.Check.ge(0)),
    ...     "name": pa.Column(str),
    ... })
    >>> df = pd.DataFrame({"age": [25, 30], "name": ["Alice", "Bob"]})
    >>> validate_data(df, schema)
    age   name
    0   25  Alice
    1   30    Bob

    >>> # Using the project schemas from schema.py
    >>> from schema import raw_schema
    >>> validated_df = validate_data(raw_df, raw_schema)

    """
    # Check inputs
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df should be of type pd.DataFrame, got {type(df)}")
    if not isinstance(df_schema, pa.DataFrameSchema):
        raise TypeError(
            f"df should be of type pa.DataFrameSchema, got {type(df_schema).__name__}"
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
    """
    This is an internal function meant for use in `read_data_from_index_file()`

    Takes in a dataframe and a path, and saves to dataframe to the location
    specified from the path. This is to be used to save the raw and processed
    data files to data/raw/ and data/processed.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to save
    path : Path
        The path to save the data frame to.

    Returns
    -------
    None
        This function reads to disk and does not return anything.

    Raises
    ------
    TypeError
        If df input is not a pandas DataFrame.
    ValueError
        If df is empty or if path does not end in `.csv`

    Notes
    -----
    This function assumes the output is a `.csv` file. To update this,
    you can update the format check in the function to replace ".csv" with
    whichever output extension you would like.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"age": [25, 30], "name": ["Alice", "Bob"]})
    >>> save_data(df, Path("data/processed/processed.csv))
    Saved file to ~/.../data/processed/processed.csv
    """
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
    print(f"Saved file to {path}")


def filter_existing_clips(
    df: pd.DataFrame,
    clips_dir: Path,
) -> pd.DataFrame:
    """
    This is an internal function meant to be used in read_data_from_index_file.

    Takes in a validated dataframe, the root directory for cross-sucking
    clips, and filters the data frame for rows which have a
    `"clip_relative_path"` that exists in the `clip_dir`.

    This function raises a warning if the filtered df is empty. It also
    returns a warning stating the number of rows (videos) dropped.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas dataframe meant to represent the validated dataframe
        output from validate_data()
    clips_dir : Path
        The path to the root directory holding all cross-sucking clips.

    Returns
    -------
    pd.DataFrame
        The validated dataframe filtered for rows which have cross-sucking
        clips in the file path.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame.
    ValueError
        If df is an empty dataframe.
    FileNotFoundError
        If clips_dir does not exist in the file path.

    Notes
    -----
    This function requires the "clip_relative_path" column in the data frame
    and relies on the data validation to ensure this column exists. This is an
    internal function that should only ever be called after validate_data().

    Examples
    --------
    >>> import pandas as pd
    >>> from pathlib import Path
    >>> df = pd.DataFrame({
    ...     "clip_relative_path": ["clips/a.mp4", "clips/b.mp4", "clips/c.mp4"],
    ...     "label": ["cat", "dog", "cat"],
    ... })
    >>> clips_dir = Path("/data/moovision/clips")  # doctest: +SKIP
    >>> filtered_df = filter_existing_clips(df, clips_dir)  # doctest: +SKIP
    >>> len(filtered_df)  # doctest: +SKIP
    """

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
        warnings.warn(
            f"Dropped {n_dropped} clips with no file on disk"
        )  #:\n{dropped}")

    return filtered_df


def get_label_paths(labels_dir: Path) -> list[tuple[str, str]]:
    """
    This is an internal function meant to be used in read_data_from_index_file.

    Looks for all folders containing bounding box annotations in the labelled
    clips directory, and creates a list of tuples with names and relative paths
    to the annotation folders.

    Assumes all `.zip` files in the labels_dir directory are annotations for CS
    events. This function will read in all zip files in the labels_dir which
    may cause errors down the line.

    Parameters
    ----------
    labels_dir : Path
        The path to the root directory containing bounding box annotations for
        the cross-sucking clips.

    Returns
    -------
    list[tuple[str, str]]
        A list of tuples containing the .zip file names and relative paths
        within the annotated labels directory.

    Raises
    ------
    FileNotFoundError
        If labels_dir does not exist, or if no .zip files are found in the
        specified directory.

    Examples
    --------
    >>> from pathlib import Path
    >>> labels_dir = Path("/data/moovision/labels")
    >>> label_paths = get_label_paths(labels_dir)  # doctest: +SKIP
    >>> label_paths  # doctest: +SKIP
    [("annotations_batch1.zip", "labels/annotations_batch1.zip"),
    ("annotations_batch2.zip", "labels/annotations_batch2.zip")]

    If no .zip files are found or the directory does not exist, an error is
    raised:

    >>> get_label_paths(Path("/data/moovision/empty_dir"))  # doctest: +SKIP
    FileNotFoundError: No .zip files found in /data/moovision/empty_dir
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
    """
    This is an internal function meant to be used in read_data_from_index_file.

    Takes in a df and a list of zip folder names and relative paths to
    annotated cross-sucking data folders output from get_label_paths(). For
    each cross-sucking clip in the df, match_label_paths searches over all
    names/paths in label_paths to find a matching annotaion folder. The
    function returns a list of relative paths to these matching annotation
    folders, or None if a matching folder cannot be found. This is meant
    to be passed on and later added to the index data frame as a column of
    relative paths to data annotation folders.

    These names use the is_match() function from matching.py to match
    cross-sucking clip names to annotation folder names usign regex
    captures to parse and compare numeric ID's and part ID's from each name.
    See the matching.py documentation for more detail. is_match() also
    assumes the correct naming conventions for cross-sucking clips and
    annotaion folders. This will raise an error if the function encounters
    an unkown naming format.

    If there are multiple/duplicate matches the function raises an error as it
    does not know which folder contains the correct annotations. If there are
    duplicate matches between normal video paths and video paths to fixed_clips
    folder, the normal video paths are used, and a warning is raised.

    A warning is also raised if no matches are found for any clip.

    Parameter
    ---------
    df : pd.DataFrame
        A pandas DataFrame
    label_paths : list[tuple[str, str]]
        a list of annotation folder names and relative paths inside the
        labelled data directory.

    Returns
    -------
    list[str | None]
        A list of relative paths to annotaion folders, or None values if
        a cross-sucking clip name does not match to any annotation folder
        names.

    Raises
    ------
    TypeError
        If df is not a Pandas DataFrame
    ValueError
        If df, or label_paths is empty.
        If there are more than 2 matches for a cross-sucking clip, or if
        there are exactly 2 matches, but neither are a fixed_path file.

    Notes
    -----
    This requires the dataset be previously validated and should only be called
    after validate_data().

    Examples
    --------

    """

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
    label_paths_list = [None] * len(clips)

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
                label_paths_list[i] = matches[1]
                warnings.warn(
                    f"\nAmbiguity Warning: {name} returned multiple matches: {matches}. \n"
                    f"Defaulting to use the base clip option: '{matches[1]}'.\n",
                    # f"If you want to use fixed clips, please run ",
                    category=UserWarning,
                    stacklevel=2,
                )
            elif ("fixed_clips" in matches[1]) and ("fixed_clips" not in matches[0]):
                label_paths_list[i] = matches[0]
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
            label_paths_list[i] = matches[0]
        else:
            label_paths_list[i] = None

    # Warn if no matches found for any clips
    if all(p is None for p in label_paths_list):
        warnings.warn("No matches found for any clips.")

    return label_paths_list


def add_label_paths(df: pd.DataFrame, label_paths: list[str | None]) -> pd.DataFrame:
    """
    This is an internal function meant to be used in read_data_from_index_file.

    Takes in a df and a list of relative paths to annotation data folders, and
    adds the list as a new column to the data frame. This adds corresponding
    annotation data paths for each cross-sucking clip to the data frame.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame
    label_paths : list[str | None]
        A list of relative paths to annotation data folder matched to
        cross-sucking clips, or None if a clip does not have a match.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame representing the filtered index file with an extra
        column added for relative paths to annotated data labels.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame.
    ValueError
        If either df, or labels_paths is empty.

    Notes
    -----
    This function requires data validation and should only be called after
    calling validate_data(). This output should also be validated using the
    processed_schema DataFrameSchema in schema.py.

    Examples
    --------


    """

    # Check inputs
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be of type pd.Dataframe, got {type(df)}")
    # Check for emty inputs
    if df.empty:
        raise ValueError("df is empty")
    if not label_paths:
        raise ValueError("labelled_paths is empty")

    # Add labelled Paths to df
    df["labelled_clip_relative_path"] = label_paths

    return df


def filter_label_paths(df: pd.DataFrame) -> pd.DataFrame:
    """
    This is an internal function meant to be used in read_data_from_index_file.

    Filter the data file for cross-sucking clips which have an associated
    annotation file present in the data. Takes in a data file which has had
    relative paths to annotation files added in a "labelled_clip_relative_path"
    and filters for rows where this columns is not None. The column
    "labelled_clip_relative_path" is the output of match_label_paths,
    and has None values where matches were not found in the data. Filtering
    out None values removes these rows from training sets as both cross-sucking
    clips and annotation labels are required for model training.

    This function requires the data is validated for the correct columns. It
    is meant to be called after add_label_paths, and after processed data has
    been validated.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas data frame representing the output of validate_data once
        the relative paths to annotation labels have been added via add_labels.

    Returns
    -------
    pd.DataFrame
        A pandas data frame containing only cross-sucking clips that have a
        matching annotation labels folder in the data.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame
    ValueError
        If df is empty

    Notes
    -----
    This function will raise warnings for the number of dropped cross-sucking clips,
    and if the returned df is empty.

    Examples
    --------


    """

    # Check inputs
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be of type pd.Dataframe, got {type(df)}")
    # Check for emty inputs
    if df.empty:
        raise ValueError("df is empty")

    #  Filter for rows with labels
    filtered_df = df[
        ~df["labelled_clip_relative_path"].isna()
    ]  # Existence of col is ensured by data validation.

    # Report Dropped Clips
    n_dropped = len(filtered_df) - len(df)
    if n_dropped:
        dropped = df[df["labelled_clip_relative_path"].isna()]["clip_name"].to_list()
        warnings.warn(
            f"Dropped {n_dropped} clips with no matching annotation files\n"
        )  #:{dropped}")

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

    This is an internal function, input paths should be called via config.py.

    Examples
    --------
    >>> from config import INDEX_PATH
    >>> read_data_from_index_file(INDEX_PATH)  # doctest: +SKIP
    """
    # Read in Raw Data
    df_raw = read_data(index_path)
    df_raw_validated = validate_data(df_raw, schema)
    # Save Raw index
    if raw_output.exists() and not force:

        print(f"File already exists at {raw_output}")

    else:

        save_data(df_raw_validated, raw_output)

        if processed_output.exists() and not force:

            print(f"File already exists at {processed_output}")

        else:

            df_filtered = filter_existing_clips(
                df=df_raw_validated,
                clips_dir=clips_dir,
            )
            paths = get_label_paths(labels_dir=labels_dir)
            label_paths = match_label_paths(df_filtered, label_paths=paths)
            df_labels = add_label_paths(df=df_filtered, label_paths=label_paths)
            df_processed = filter_label_paths(df_labels)
            df_labels_validated = validate_data(df_processed, processed_schema)

            save_data(df=df_labels_validated, path=processed_output)


def parse_args():
    parser = argparse.ArgumentParser(description="Data splitting.")
    parser.add_argument(
        "--index_path",
        default=INDEX_PATH,
        type=Path,
        help="Path to data index.",
    )
    parser.add_argument(
        "--clips_dir",
        default=UNLABELLED_CLIPS_DIR,
        type=Path,
        help="Path to cross-sucking clips directory.",
    )
    parser.add_argument(
        "--labels_dir",
        default=LABELLED_CLIPS_DIR,
        type=Path,
        help="Path to annotations directory.",
    )
    parser.add_argument(
        "--source_dir",
        default=SOURCE_VIDEOS_DIR,
        type=Path,
        help="Path to source videos directory.",
    )
    parser.add_argument(
        "--raw_output",
        default=RAW_INDEX_OUTPUT,
        type=Path,
        help="Output path for raw index file.",
    )
    parser.add_argument(
        "--processed_output",
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
        clips_dir=args.clips_dir,
        labels_dir=args.labels_dir,
        source_dir=args.source_dir,
        raw_output=args.raw_output,
        processed_output=args.processed_output,
        force=args.FORCE,
    )
