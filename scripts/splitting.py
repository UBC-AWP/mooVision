"""
Split data into train/test splits according to research questions defined in documentation.
"""

from pathlib import Path
import pandas as pd
from config import ROOT

OUTPUT_DIR = ...


def random_shuffle_split(
    unlbld_videos_path: Path,
    lbld_videos_path: Path,
    raw_video_path: Path,
    index: bool = False,
):
    """
    Random Shuffle clips into train and test.

    Reads raw data and clip names from path, randomly shuffles them into train and test, and outputs train.csv, test.csv.

    Parameters
    ----------
    path : Path
        Path to root directory of files (MooVision One Drive)
    index : bool
        If true, the path points to an index file containing paths and metadata for all videos.

    Returns
    -------
    None
        The function writes to disk and does not return a value.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    FileNotFoundError
        If looking for files where parents do not exist.

    Notes
    -----
    This function assumes that clipped videos adheres to the following structure:

        Pen
        |- Weaning Periods
            |- Days

    That unlabelled clips fall under cross_sucking_clips

    Examples
    --------
    >>> random_shuffle_split(ROOT)
    """
    data = pd.DataFrame({"place-holder": 0})
    data.to_csv(OUTPUT_DIR)
    pass


def day_based_split(path: Path):
    """
    Day-based split. Train/test split within the same pens and periods, grouped by day.
    """
    pass


def pen_based_split(path: Path):
    """
    Pan-based split. Withhold one pen for evaluation, train on remaining pens.

    Rotates the pen withheld for evaluation to create a total of 3 train/test splits.
    """
    pass


def period_based_split(path: Path):
    """
    Period-based split. Withhold one period for evaluation, train on remaining periods.

    Rotates the period withheld for evalution to create a total of 3 train/test splits.
    """
    pass


def main():
    # Add technical execuation logic
    print("Running data splitting...")
    print("Creating JSON files...")
    random_shuffle_split()
    day_based_split()
    pen_based_split()
    period_based_split()
    print("Checking files created...")
    print("All files created.")
    print("Data splitting done.")


if __name__ == "__main__":
    main()
