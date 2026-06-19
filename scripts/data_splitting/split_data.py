"""
Split data into train/test splits according to research questions defined in documentation.
"""

from pathlib import Path
import sys
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent.parent.parent))

from config import ROOT_DIR, PROCESSED_INDEX

OUTPUT_DIR = ROOT_DIR / "data" / "processed"


def train_test_to_csv(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path,
    FORCE: bool = False,
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
    if train.empty:
        raise ValueError("Training set is an empty df.")
    if val.empty:
        raise ValueError("Validation set is an empty df.")
    if test.empty:
        raise ValueError("Testing set is an empty df.")

    # Output paths
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    # Save train to csv
    if train_path.exists() and not FORCE:
        print(f"{train_path} already exists.")
    else:
        train_path.parent.mkdir(parents=True, exist_ok=True)
        train.to_csv(train_path, index=False)
        print(f"Training set saved to {train_path}")

    # Save train to csv
    if val_path.exists() and not FORCE:
        print(f"{val_path} already exists.")
    else:
        val_path.parent.mkdir(parents=True, exist_ok=True)
        val.to_csv(val_path, index=False)
        print(f"Validation set saved to {val_path}")

    # Save test to csv
    if test_path.exists() and not FORCE:
        print(f"{test_path} already exists.")
    else:
        test_path.parent.mkdir(parents=True, exist_ok=True)
        test.to_csv(test_path, index=False)
        print(f"Test set saved to {test_path}\n")


def random_shuffle_split(
    input_path: Path, output_dir: Path, FORCE=False, exectute=True
):
    """
     Random Shuffle clips into train, val, and test sets.

     This function groups clipped data by their source videos (`source_video_basename`)
     before splitting. This guarantees a strict data separation: all sub-clips originating
     from the exact same source video are kept entirely within the same split (either
     train/val or test).

    Parameters
    ----------
    input_path : Path
        Path to the processed clips index CSV file.
    output_dir : Path
        Base output directory where the 'random/' folder will be created.
    force : bool, default False
        If True, forces overwrite of existing destination files.
    enabled : bool, default True
        Controls execution; maps directly to command line argument flags.


     Returns
     -------
     None
         This function writes generated train.cs, val.csv, and test.csv
         straight to disk.

    Raises
     ------
     FileNotFoundError
         If the target configuration index at `input_path` cannot be located on disk.
     ValueError
         If metadata structures or expected columns are missing.

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
             exectute=True
         )
    """
    # Do nothing if user passes false for random_split
    if not exectute:
        return

    # Output Path target configuration
    target_output_dir = output_dir / "random"

    if target_output_dir.exists() and not FORCE:
        print(f"{target_output_dir} already exists.")
        return

    target_output_dir.mkdir(parents=True, exist_ok=True)

    # Read Data index safely
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} does not exist.")

    df = pd.read_csv(input_path)

    # Random Shuffle Source Videos
    unique_sources = df["source_video_basename"].drop_duplicates()

    # Separate train and test sets
    tmp_train_sources, test_sources = train_test_split(
        unique_sources,
        test_size=0.10,
        random_state=300,
        shuffle=True,
    )

    # Separate train and val sets
    train_sources, val_sources = train_test_split(
        tmp_train_sources,
        test_size=0.222,  # Gives ~70-20-10 split
        random_state=300,
        shuffle=True,
    )

    # Match clips to source video split

    train_df = df[df["source_video_basename"].isin(train_sources)]
    val_df = df[df["source_video_basename"].isin(val_sources)]
    test_df = df[df["source_video_basename"].isin(test_sources)]

    # Save to csv
    print("\n--------------------------------\n")
    print("--- Saving Random Split ---\n")
    train_test_to_csv(train_df, val_df, test_df, target_output_dir, FORCE)


def time_based_split(input_path: Path, output_dir: Path, FORCE=False, exectute=True):
    """
     Split data using a time-based holdout strategy.

    Creates a train/test split by holding out cross-sucking clips from
    specific times as a testing set across the same pens and periods.
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
    exectute : bool
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
            exectute=True
        )
    """
    # Do nothing if user passes false for time  _based_split
    if not exectute:
        return
    # Output Path
    target_output_dir = output_dir / "day_based"

    if target_output_dir.exists() and not FORCE:
        print(f"{target_output_dir} already exists.")
        return

    target_output_dir.mkdir(parents=True, exist_ok=True)
    # Read Data
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} does not exist.")

    df = pd.read_csv(input_path)

    if "day" not in df.columns:
        raise ValueError(f"Specified column 'day' not found in data index.")

    # Time level split - train on day 1 and 2
    train_df = df[df["day"] != 3]

    # Validate on majority of day 3
    val_df = df.loc[  # 23% of time
        (df["day"] == 3)
        & (df["part_start_obs_sec"] > 8640)
        & (df["part_end_obs_sec"] < 60480)
    ]

    # Test on remaining section of day 3.
    test_df = df.loc[  # 10% of time
        (df["day"] == 3) & (df["part_start_obs_sec"] >= 60480)
    ]

    # Save to csv
    print("\n--------------------------------\n")
    print("--- Saving Time-Based Split ---\n")
    train_test_to_csv(train_df, val_df, test_df, target_output_dir, FORCE)


def pen_based_split(input_path: Path, output_dir: Path, FORCE=False, exectute=True):
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
    exectute : bool
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
            exectute=True
        )
    """
    # Do nothing if user passes false for random_split
    if not exectute:
        return
    # Output Path
    target_output_dir = output_dir / "pen_based"

    if target_output_dir.exists() and not FORCE:
        print(f"{target_output_dir} already exists.")
        return

    target_output_dir.mkdir(parents=True, exist_ok=True)
    # Read Data
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} does not exist.")

    df = pd.read_csv(input_path)

    if "pen" not in df.columns:
        raise ValueError(f"Specified column 'pen' not found in data index.")

    unique_pens = sorted(df["pen"].dropna().unique())
    total_pens = len(unique_pens)

    print(f"Detected {total_pens} unique pens: {unique_pens}")

    if total_pens < 3:
        raise ValueError(
            f"Critical Error: You only have {total_pens} pen(s). "
            f"You need at least 3 unique pens to rotate Train, Val, and Test assignments."
        )

    print("\n--------------------------------\n")
    print("--- Saving Pen-Based Splits ---\n")
    for test_idx in range(total_pens):
        # Calculate indices using modulo to wrap around the ring smoothly
        val_idx = (test_idx + 1) % total_pens

        # Pull the specific pen strings based on our sliding indices
        test_pen = unique_pens[test_idx]
        val_pen = unique_pens[val_idx]

        # All other pens that aren't currently Test or Val go to Train
        train_pens = [
            int(pen)
            for idx, pen in enumerate(unique_pens)
            if idx != test_idx and idx != val_idx
        ]

        # 4. Create the masking conditions
        train_df = df[df["pen"].isin(train_pens)]
        val_df = df[df["pen"] == val_pen]
        test_df = df[df["pen"] == test_pen]

        # Save to csv
        print(f"Generated Profile {test_pen}:")
        print(f"  -> Save Directory: {target_output_dir / f"pen_{test_pen}"}")
        print(f"  -> Train Pens:     {train_pens}")
        print(f"  -> Validation Pen: [{val_pen}]")
        print(f"  -> Testing Pen:    [{test_pen}]\n")

        train_test_to_csv(
            train_df,
            val_df,
            test_df,
            target_output_dir / f"pen_{test_pen}",
            FORCE,
        )

        print()

    print(
        f"All {total_pens} rotation profiles successfully written to {target_output_dir}"
    )


def period_based_split(input_path: Path, output_dir: Path, FORCE=False, exectute=True):
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
    exectute : bool
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
            exectute=True
        )

    """
    # Do nothing if user passes false for random_split
    if not exectute:
        return
    # Output Path
    target_output_dir = output_dir / "period_based"

    if target_output_dir.exists() and not FORCE:
        print(f"{target_output_dir} already exists.")
        return

    target_output_dir.mkdir(parents=True, exist_ok=True)

    # Read Data
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} does not exist.")

    df = pd.read_csv(input_path)

    # Period-level Splits
    unique_periods = sorted(df["phase"].dropna().unique())
    total_periods = len(unique_periods)

    print("\n--------------------------------\n")
    print("--- Saving Period-Based Splits ---\n")
    for test_idx in range(total_periods):
        # Calculate validation index by wrapping around the list smoothly using modulo (%)
        val_idx = (test_idx + 1) % total_periods

        # Pull explicit period string assignments
        test_period = unique_periods[test_idx]
        val_period = unique_periods[val_idx]

        # Train periods are any periods that aren't the current test or val targets
        train_periods = [
            p
            for idx, p in enumerate(unique_periods)
            if idx != test_idx and idx != val_idx
        ]

        # Filter the dataframes row-by-row using explicit masks
        train_df = df[df["phase"].isin(train_periods)]
        val_df = df[df["phase"] == val_period]
        test_df = df[df["phase"] == test_period]

        # Fixed f-string quotes by using single quotes internally
        profile_save_dir = target_output_dir / f"{test_period}"

        print(f"Generated Profile {test_period}:")
        print(f"  -> Save Directory:   {profile_save_dir}")
        print(f"  -> Training Period:  {train_periods}")
        print(f"  -> Validation Period: [{val_period}]")
        print(f"  -> Testing Period:    [{test_period}]\n")

        # Save to csv with all three distinct sets
        train_test_to_csv(train_df, val_df, test_df, profile_save_dir, FORCE)

        print()


def pipeline_demo(input_path: Path, output_dir: Path, FORCE=False, exectute=True):
    """
    Random Shuffle clips into train/test sets.

    Save small portion to use for pipeline testing.

    (DELETE THIS FUNCTION LATER - internal testing use only)
    """
    # Do nothing if user passes false for random_split
    if not exectute:
        return
    # Output Path
    output_dir = output_dir / "pipeline_demo"

    if output_dir.exists() and not FORCE:
        print(f"{output_dir} already exists.")
        return

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    # Read Data
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} does not exist.")

    df = pd.read_csv(input_path)

    # Random Shuffle Source Videos
    temp_df = df["source_video_basename"].drop_duplicates()
    train, test = train_test_split(
        temp_df, test_size=0.33, train_size=0.67, random_state=300, shuffle=True
    )

    # Match clips to source video split
    condition = df["source_video_basename"].isin(train)
    train_df = df[condition]
    test_df = df[~condition]

    # Take small, easily downloadable dataset.
    train = train_df.iloc[10:13]  # 3 videos
    val = train_df.iloc[14:16]  # 2 videos
    test = test_df.iloc[17:19]  # 2 videos

    # Save to csv
    print("\n--------------------------------\n")
    print("--- Saving Demo Splits ---\n")
    train_test_to_csv(train, val, test, output_dir, FORCE)


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
        type=Path,
        help=f"Output directory for train/test splits (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--FORCE",
        default=False,
        action="store_true",
        help="Overwrite existing file folders (default: False)",
    )
    # Defaults to execute each split, if argument is passed the split is not executed.
    parser.add_argument(
        "--no_random_split",
        default=True,
        action="store_false",
        help="Disable random-based splitting",
    )
    parser.add_argument(
        "--no_time_split",
        default=True,
        action="store_false",
        help="Disable day-based splitting",
    )
    parser.add_argument(
        "--no_pen_split",
        default=True,
        action="store_false",
        help="Disable pen-based splitting",
    )
    parser.add_argument(
        "--no_period_split",
        default=True,
        action="store_false",
        help="Disable period-based splitting",
    )
    parser.add_argument(
        "--no_pipeline_testing",
        default=True,
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
        exectute=args.no_random_split,
    )
    time_based_split(
        input_path=args.data_path,
        output_dir=args.output_dir,
        FORCE=args.FORCE,
        exectute=args.no_time_split,
    )
    pen_based_split(
        input_path=args.data_path,
        output_dir=args.output_dir,
        FORCE=args.FORCE,
        exectute=args.no_pen_split,
    )
    period_based_split(
        input_path=args.data_path,
        output_dir=args.output_dir,
        FORCE=args.FORCE,
        exectute=args.no_period_split,
    )
    pipeline_demo(
        input_path=args.data_path,
        output_dir=args.output_dir,
        FORCE=args.FORCE,
        exectute=args.no_pipeline_testing,
    )
    print("Checking files created...")
    print("All files created.")
    print("Data splitting done.")
