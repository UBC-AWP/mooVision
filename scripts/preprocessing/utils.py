"""
Utility Functions For Preprocessing Data into Training Sets for YOLO models
"""

import sys
from typing import List, Tuple
from pathlib import Path
import yaml

sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.data_reading.matching import parse_labelled_name, parse_unlabelled_name


def validate_file_paths(
    paths: List[str],
    root: Path,
) -> List[Path]:
    """
    Clean file paths to standardize path structure
    and validate paths exist in the root directory `root`.

    Parameters
    ----------
    paths : List[str]
        List of relative paths within the labelled clips directory to
        zipped folders containing bounding box annotations for
        cross-sucking events, or relative paths to cross-sucking
        clips within the cross sucking clips directory.

    root : pathlib.Path
        Path to the root directory containing zipped folders
        with bounding box annotations for cross-sucking events,
        or cross-sucking videos.

    Returns
    -------
    validated_paths : List[pathlib.Path]
        List of Paths to zipped folders with bounding box labels,
        or cross-sucking videos.

    Raises
    ------
    TypeError
        If label_paths is not a list or root is not a Path object.
    FileNotFoundError
        If a constructed path does not exist.
    """
    # Type Validation
    if not isinstance(paths, list):
        raise TypeError(
            f"Expected 'label_paths' to be a list, got {type(paths).__name__}"
        )

    if not isinstance(root, Path):
        raise TypeError(
            f"Expected 'root' to be a Path object, got {type(root).__name__}"
        )
    # Input Pre-Validation (Fails fast before touching data)
    validated_paths = []
    for raw_path in paths:
        if not isinstance(raw_path, str):
            raise TypeError(f"ERROR: {raw_path} is not a string path.")

        # Standardize path string
        clean_path = root / raw_path.replace("\\", "/")
        if not clean_path.exists():
            raise FileNotFoundError(
                f"Attempting clean file paths, cleaned label file not found: {paths}"
            )

        validated_paths.append(clean_path)

    return validated_paths


def create_yaml(
    working_dir: str,
    class_names: list[str] = ["cross-sucking"],
    output_filename: str = "dataset.yaml",
) -> None:
    """
    Write a YOLO dataset YAML configuration file.

    Parameters
    ----------
    dataset_path : str
        Absolute path to the dataset root directory.
    class_names : list[str]
        Ordered list of class labels matching the IDs in label files.
    output_path : str, optional
        Destination path for the YAML file.

    Returns
    -------
    None
        This function reads straight to disk and does not return anything.
    """
    config = {
        "path": "./dataset",
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": class_names,
    }
    print("\n--- Creating YAML file ---")
    print("\n\n=========================")
    print("DATASET.YAML")
    out = working_dir + "/" + output_filename
    print(out)
    print(config)
    with open(out, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print("=========================\n\n")
