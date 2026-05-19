"""
Split data into train/test splits according to research questions defined in documentation.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

current_dir = Path.cwd()
PROCESSED_INDEX = current_dir / "data" / "raw" / "all_clips_index_raw.csv"
OUTPUT_DIR = current_dir / "data" / "processed"


def train_test_to_csv(
    train: pd.DataFrame, test: pd.DataFrame, output_dir: Path, force: bool = False
):
    """
    Save train and test df's to csv.
    """
    # Output paths
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"

    # Save train to csv
    if train_path.exists() and not force:
        print(f"{train_path} already exists.")
    else:
        train_path.parent.mkdir(parents=True, exist_ok=True)
        train.to_csv(train_path)
        print(f"Saved to {train_path}")

    # Save test to csv
    if test_path.exists() and not force:
        print(f"{test_path} already exists.")
    else:
        test_path.parent.mkdir(parents=True, exist_ok=True)
        test.to_csv(test_path)
        print(f"Saved to {test_path}")


def random_shuffle_split(input_path: Path, output_dir: Path, force=False):
    """
    Random Shuffle clips into train and test.

    Reads raw data and clip names from csv, randomly shuffles them into train and test, and outputs train.csv, test.csv.
    """
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

    # Save to csv
    train_test_to_csv(train, test, output_dir, force)


def day_based_split(input_path: Path, output_dir: Path, force=False):
    """
    Day-based split. Train/test split within the same pens and periods, grouped by day.
    Reads raw data and clip names from csv, outputs train.csv and test.csv.
    """
    # Read Data
    if input_path.exists():
        df = pd.read_csv(input_path)
    else:
        raise FileNotFoundError(f"{input_path} does not exist.")
    # Day level split
    condition = df["day"].isin([1, 2])
    train, test = df[condition], df[~condition]

    # Save to csv
    train_test_to_csv(train, test, output_dir, force)


def pen_based_split(input_path: Path, output_dir: Path, force=False):
    """
    Pan-based split. Withhold one pen for evaluation, train on remaining pens.

    Rotates the pen withheld for evaluation to create a total of 3 train/test splits.

    Reads raw data and clip names from csv, outputs train.csv and test.csv.
    """
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
        train, test = df[condition], df[~condition]

        # Save to csv
        train_test_to_csv(train, test, output_dir / f"pen_{pen}", force)


def period_based_split(input_path: Path, output_dir: Path, force=False):
    """
    Period-based split. Withhold one period for evaluation, train on remaining periods.

    Rotates the period withheld for evalution to create a total of 3 train/test splits.

    Reads raw data and clip names from csv, outputs train.csv and test.csv.
    """
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
        train_test_to_csv(train, test, output_dir / f"{period}", force)


def main():
    # --- Read in index file ---
    # if PROCESSED_INDEX.exists():
    #     processed_index = pd.read_csv(PROCESSED_INDEX)
    # else:
    #     raise FileNotFoundError(f"{PROCESSED_INDEX} does not exist.")
    # Add technical execuation logic
    print("Running data splitting...")
    print("Creating JSON files...")
    print(PROCESSED_INDEX)
    random_shuffle_split(
        input_path=PROCESSED_INDEX, output_dir=OUTPUT_DIR / "random", force=False
    )
    day_based_split(
        input_path=PROCESSED_INDEX, output_dir=OUTPUT_DIR / "day_based", force=False
    )
    pen_based_split(
        input_path=PROCESSED_INDEX, output_dir=OUTPUT_DIR / "pen_based", force=False
    )
    period_based_split(
        input_path=PROCESSED_INDEX, output_dir=OUTPUT_DIR / "period_based", force=False
    )
    print("Checking files created...")
    print("All files created.")
    print("Data splitting done.")


if __name__ == "__main__":
    main()
