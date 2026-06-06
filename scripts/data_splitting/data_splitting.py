"""
Split data into train/test splits according to research questions defined in documentation.
"""

from pathlib import Path
import sys
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent.parent))

PROCESSED_INDEX = Path("data/processed/processed_clips_index.csv").absolute()
OUTPUT_DIR = Path("data/processed").absolute()


def train_test_to_csv(
    train: pd.DataFrame, test: pd.DataFrame, output_dir: Path, FORCE: bool = False
):
    """
    Save train and test df's as csv' with correct names.

    Parameters
    ----------
    train : pd.DataFrame
        df to save as train.csv
    test : pd.DataFrame
        df to save as test.csv
    output_dir : Path
        Path to output directory.
    FORCE : bool
        If true overwrite existing dataframs.

    Returns
    -------
    None
        This function writes generated train.csv and test.csv straight to disk.

    Raises
    ------
    ???
    # Enforces column structure Test functionality, test bad or good inputs, edge cases

    Notes
    -----
    Function is for internal use to reduce repetition in splitting.py

    Examples
    --------
    .. code-block:: python

        from pathlib import Path
        import pandas as pd
        from sklearn.model_selection import train_test_split

        # Create dummy data and define paths
        data = {"feature": [1, 2, 3, 4], "label": [0, 1, 0, 1]}
        df = pd.DataFrame(data)
        out_path = Path("data/processed/random")

        # Split into train and test
        train_df, test_df = train_test_split(
            df, test_size=0.50, random_state=42, shuffle=True
        )

        # Save to disk (overwriting any older runs)
        train_test_to_csv(train_df, test_df, output_dir=out_path, FORCE=True)
    """

    # Output paths
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"

    # Save train to csv
    if train_path.exists() and not FORCE:
        print(f"{train_path} already exists.")
    else:
        train_path.parent.mkdir(parents=True, exist_ok=True)
        train.to_csv(train_path)
        print(f"Saved to {train_path}")

    # Save test to csv
    if test_path.exists() and not FORCE:
        print(f"{test_path} already exists.")
    else:
        test_path.parent.mkdir(parents=True, exist_ok=True)
        test.to_csv(test_path)
        print(f"Saved to {test_path}")


def random_shuffle_split(input_path: Path, output_dir: Path, FORCE=False, run=True):
    """
    Random Shuffle clips into train and test.

    This function groups clipped data by their source videos (`source_video_basename`)
    before splitting. This guarantees a strict data separation: all sub-clips originating
    from the exact same source video are kept entirely within the same split (either
    train or test).

    Parameters
    ----------
    input_path : Path
        Path to processed_clips_index
    output_dir : Path
        Base output directory wehre 'random/' folder will be created
    FORCE : bool
        If True, forces overwrite of existing destination files.
    run : bool
        Controls Execution; map to command line argument flags.


    Returns
    -------
    None
        This function writes generated train.csv and test.csv straight to disk.

    Raises
    ------
    InputErrors
        for bad inputs
    TypeErrors
        for bad types
    ValueError
        for incorrect columns types.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path
        import pandas as pd
        from scripts.splitting import random_shuffle_split

        # Setup mock clips data index where multiple clips share a source video
        mock_data = {
            "clip_id": [1, 2, 3, 4, 5],
            "source_video_basename": ["vid_A", "vid_A", "vid_B", "vid_C", "vid_C"],
            "feature_score": [0.91, 0.42, 0.88, 0.12, 0.76]
        }
        df = pd.DataFrame(mock_data)

        # Save mock index to a temp file path
        csv_input = Path("data/processed_clips_index.csv")
        csv_input.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_input, index=False)

        # Output target directory
        base_output = Path("data/processed")

        # Perform Random split
        random_shuffle_split(
            input_path=csv_input,
            output_dir=base_output,
            FORCE=True,
            run=True
        )
    """
    # Do nothing if user passes false for random_split
    if not run:
        return
    else:
        # Output Path
        target_output_dir = output_dir / "random"

        if target_output_dir.exists() and not FORCE:
            print(f"{target_output_dir} already exists.")
        else:
            target_output_dir.mkdir(parents=True, exist_ok=True)

            # Read Data
            if input_path.exists():
                df = pd.read_csv(input_path)
            else:
                raise FileNotFoundError(f"{input_path} does not exist.")

            # Random Shuffle Source Videos
            temp_df = df["source_video_basename"].drop_duplicates()
            train_sources, test_sources = train_test_split(
                temp_df, test_size=0.33, train_size=0.67, random_state=300, shuffle=True
            )

            # Match clips to source video split
            condition = df["source_video_basename"].isin(train_sources)
            train_df = df[condition]
            test_df = df[~condition]

            # Save to csv
            train_test_to_csv(train_df, test_df, target_output_dir, FORCE)


def day_based_split(input_path: Path, output_dir: Path, FORCE=False, run=True):
    """
     Split data using a Day-based holdout strategy.

    Creates a train/test split by holding out cross-sucking clips from
    specific days as a testing set across the same pens and periods.
    This ensures the model is evaluated on data from entirely separate time
    periods/days to test temporal robustness.

    Parameters
    ----------
    input_path : Path
        Path to processed data file
    output_dir : Path
        Base output directory where the 'day_based/' folder structure will be
        created.
    FORCE : bool
        Base output directory where the 'pen_based/' folder structure will be
        created.
    run : bool
        Controls execution; map this directly to command line argument flags.

    Returns
    -------
    None
        This function writes generated train.csv and test.csv straight to disk.

    Raises
    ------
    InputErrors
        for bad inputs
    TypeErrors
        for bad types
    ValueError
        for incorrect columns types.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path
        import pandas as pd
        from scripts.splitting import day_based_split

        # Setup mock data spanning multiple recording days
        mock_data = {
            "clip_id": [101, 102, 103, 104, 105],
            "day": [1, 1, 2, 3, 4],
            "behavior": ["sucking", "normal", "sucking", "normal", "sucking"]
        }
        df = pd.DataFrame(mock_data)

        # Save mock data to a temp file path
        csv_input = Path("data/processed_clips_index.csv")
        csv_input.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_input, index=False)

        # Output target directory
        base_output = Path("data/processed")

        # Perform day-based holdout split
        day_based_split(
            input_path=csv_input,
            output_dir=base_output,
            FORCE=True,
            run=True
        )
    """
    # Do nothing if user passes false for day_based_split
    if not run:
        return
    else:
        # Output Path
        target_output_dir = output_dir / "day_based"

        if target_output_dir.exists() and not FORCE:
            print(f"{target_output_dir} already exists.")
        else:
            target_output_dir.mkdir(parents=True, exist_ok=True)
            # Read Data
            if input_path.exists():
                df = pd.read_csv(input_path)
            else:
                raise FileNotFoundError(f"{input_path} does not exist.")
            # Day level split
            condition = df["day"].isin([1, 2])
            train_df, test_df = df[condition], df[~condition]

            # Save to csv
            train_test_to_csv(train_df, test_df, target_output_dir, FORCE)


def pen_based_split(input_path: Path, output_dir: Path, FORCE=False, run=True):
    """
    Split data using a Pen-based holdout strategy.

    Iterates through all unique pens in the dataset. For each iteration,
    withold one pen as the test set, while the remaining pens form the
    training set. This creates separate train/test subfolders for each pen
    to evaluate how well the model generalizes to animals in completely
    unseen environments.

    Parameters
    ----------
    input_path : Path
        Path to processed data CSV file.
    output_dir : Path
        Base output directory where the 'pen_based/' folder structure will be
        created.
    FORCE : bool
        Base output directory where the 'pen_based/' folder structure will be
        created.
    run : bool
        Controls execution; map this directly to command line argument flags.

    Returns
    -------
    None
        This function writes generated train.csv and test.csv straight to disk.

    Raises
    ------
    InputErrors
        for bad inputs
    TypeErrors
        for bad types
    ValueError
        for incorrect columns types.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path
        import pandas as pd
        from scripts.splitting import day_based_split

        # Setup mock data spanning multiple recording days
        mock_data = {
            "clip_id": [101, 102, 103, 104, 105],
            "pen": [2, 3, 2, 3, 5],
            "behavior": ["sucking", "normal", "sucking", "normal", "sucking"]
        }
        df = pd.DataFrame(mock_data)

        # Save mock data to a temp file path
        csv_input = Path("data/processed_clips_index.csv")
        csv_input.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_input, index=False)

        # Output target directory
        base_output = Path("data/processed")

        # Perform day-based holdout split
        pen_based_split(
            input_path=csv_input,
            output_dir=base_output,
            FORCE=True,
            run=True
        )
    """
    # Do nothing if user passes false for random_split
    if not run:
        return
    else:
        # Output Path
        target_output_dir = output_dir / "pen_based"

        if target_output_dir.exists() and not FORCE:
            print(f"{target_output_dir} already exists.")
        else:
            target_output_dir.mkdir(parents=True, exist_ok=True)
            # Read Data
            if input_path.exists():
                df = pd.read_csv(input_path)
            else:
                raise FileNotFoundError(f"{input_path} does not exist.")

            # Pen-level Splits
            pens = df["pen"].unique()
            for pen in pens:
                filtered = [x for x in pens if x != pen]
                condition = df["pen"].isin(filtered)
                train_df, test_df = df[condition], df[~condition]

                # Save to csv
                train_test_to_csv(
                    train_df, test_df, target_output_dir / f"pen_{pen}", FORCE
                )


def period_based_split(input_path: Path, output_dir: Path, FORCE=False, run=True):
    """
    Split data using a Pen-based holdout strategy.

    Iterates through all unique periods in the dataset. For each iteration,
    withold one periods as the test set, while the remaining pens form the
    training set. This creates 3 different train/test splits, one for each
    of the PREWEANING, WEANING, and POSTWEANING phases.

    Parameters
    ----------
    input_path : Path
        Path to processed data CSV file.
    output_dir : Path
        Base output directory where the 'pen_based/' folder structure will be
        created.
    FORCE : bool
        Base output directory where the 'pen_based/' folder structure will be
        created.
    run : bool
        Controls execution; map this directly to command line argument flags.

    Returns
    -------
    None
        This function writes generated train.csv and test.csv straight to disk.

    Raises
    ------
    InputErrors
        for bad inputs
    TypeErrors
        for bad types
    ValueError
        for incorrect columns types.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path
        import pandas as pd
        from scripts.splitting import day_based_split

        # Setup mock data spanning multiple recording days
        mock_data = {
            "clip_id": [101, 102, 103, 104, 105],
            "phase": ["PREWEANING", "WEANING", "POSTWEANING", "WEANING", "PREWEANING"],
            "behavior": ["sucking", "normal", "sucking", "normal", "sucking"]
        }
        df = pd.DataFrame(mock_data)

        # Save mock data to a temp file path
        csv_input = Path("data/processed_clips_index.csv")
        csv_input.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_input, index=False)

        # Output target directory
        base_output = Path("data/processed")

        # Perform day-based holdout split
        period_based_split(
            input_path=csv_input,
            output_dir=base_output,
            FORCE=True,
            run=True
        )

    """
    # Do nothing if user passes false for random_split
    if not run:
        return
    else:
        # Output Path
        target_output_dir = output_dir / "period_based"

        if target_output_dir.exists() and not FORCE:
            print(f"{target_output_dir} already exists.")
        else:
            target_output_dir.mkdir(parents=True, exist_ok=True)
            # Read Data
            if input_path.exists():
                df = pd.read_csv(input_path)
            else:
                raise FileNotFoundError(f"{input_path} does not exist.")

            # Period-level Splits
            periods = df["phase"].unique()
            for period in periods:
                filtered = [x for x in periods if x != period]
                condition = df["phase"].isin(filtered)
                train, test = df[condition], df[~condition]

                # Save to csv
                train_test_to_csv(train, test, target_output_dir / f"{period}", FORCE)


def pipeline_testing(input_path: Path, output_dir: Path, FORCE=False, run=True):
    """
    Random Shuffle clips into train/test sets.

    Save small portion to use for pipeline testing.

    (DELETE THIS FUNCTION LATER - internal testing use only)
    """
    # Do nothing if user passes false for random_split
    if not run:
        pass
    else:
        # Output Path
        output_dir = output_dir / "pipeline_testing"

        if output_dir.exists() and not FORCE:
            print(f"{output_dir} already exists.")
        else:
            output_dir.parent.mkdir(parents=True, exist_ok=True)

            # Read Data
            if input_path.exists():
                df = pd.read_csv(input_path)
            else:
                raise FileNotFoundError(f"{input_path} does not exist.")

            # Random Shuffle Source Videos
            temp_df = df["source_video_basename"].drop_duplicates()
            train, test = train_test_split(
                temp_df, test_size=0.33, train_size=0.67, random_state=300, shuffle=True
            )

            # Match clips to source video split
            condition = df["source_video_basename"].isin(train)
            train = df[condition]
            test = df[~condition]

            train = train.iloc[5:13]
            test = test.iloc[14]

            # Save to csv
            train_test_to_csv(train, test, output_dir, FORCE)


def parse_args():
    parser = argparse.ArgumentParser(description="Data splitting.")
    parser.add_argument(
        "--data_path",
        default=PROCESSED_INDEX,
        help="Path to processed data file.",
    )
    parser.add_argument(
        "--output_dir",
        default=OUTPUT_DIR,
        help=f"Output directory for train/test splits (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--FORCE",
        default=False,
        action="store_true",
        help="Overwrite existing file folders (default: False)",
    )
    parser.add_argument(
        "--no_random_split",
        action="store_false",
        help="Disable random-based splitting",
    )
    parser.add_argument(
        "--no_day_split",
        action="store_false",
        help="Disable day-based splitting",
    )
    parser.add_argument(
        "--no_pen_split",
        action="store_false",
        help="Disable pen-based splitting",
    )
    parser.add_argument(
        "--no_period_split",
        action="store_false",
        help="Disable period-based splitting",
    )
    parser.add_argument(
        "--no_pipeline_testing",
        action="store_false",
        help="Disable day-based splitting",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print("Running data splitting...")
    args = parse_args()
    print("Creating CSV files...")
    random_shuffle_split(
        input_path=args.data_path,
        output_dir=args.output_dir,
        FORCE=args.FORCE,
        run=args.no_random_split,
    )
    day_based_split(
        input_path=args.data_path,
        output_dir=args.output_dir,
        FORCE=args.FORCE,
        run=args.no_day_split,
    )
    pen_based_split(
        input_path=args.data_path,
        output_dir=args.output_dir,
        FORCE=args.FORCE,
        run=args.no_pen_split,
    )
    period_based_split(
        input_path=args.data_path,
        output_dir=args.output_dir,
        FORCE=args.FORCE,
        run=args.no_period_split,
    )
    pipeline_testing(
        input_path=args.data_path,
        output_dir=args.output_dir,
        FORCE=args.FORCE,
        run=args.no_pipeline_testing,
    )
    print("Checking files created...")
    print("All files created.")
    print("Data splitting done.")
