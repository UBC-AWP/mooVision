"""
Testing module for splitting.py
"""

import pytest

# Empty State: What happens if the directory is empty?

# Nested Structure: Can the function find files inside subdirectories?

# Mixed Contents: Can the function distinguish between files and directories?

# Permission Issues: How does the function handle directories it cannot read?

# Non-existent Path: Does it raise a clear error (e.g., FileNotFoundError) if the input path is invalid?


@pytest.fixture(scope="module")  # defined in uv environment
def temp_folder_structure(tmp_path_factory):
    """returns a fixed dictionary representing the file folder structure."""

    # Folder names
    cross_sucking_folders = [
        "cross_sucking_clips",
        "cross_sucking_labelled",
        "raw_cross_sucking_datalog/videos",
    ]
    pens = ["Pen 5 - Group 3", "Pen 2 - Group 2", "Pen 3 - Group 1"]
    periods = ["PREWEANING", "WEANING", "POSTWEANING"]
    days = ["Day 1", "Day 2", "Day 3"]

    # Root
    root_dir = tmp_path_factory.mktemp("root_path")

    # Folders
    for cs_folder in cross_sucking_folders:
        for pen in pens:
            for period in periods:
                for day in days:
                    (root_dir / cs_folder / pen / period / day).mkdir(
                        parents=True, exist_ok=True
                    )

    # # Create Raw Video Folders
    # for pen in pens:
    #     for period in periods:
    #         for day in days:
    #             (
    #                 root_dir
    #                 / "raw_cross_sucking_datalog"
    #                 / "videos"
    #                 / pen
    #                 / period
    #                 / day
    #             ).mkdir(parents=True, exist_ok=True)

    # Data Output Folder (DO NOT NEED HERE)
    (root_dir / "data").mkdir()

    # examples folder (DO NOT NEED HERE)
    (root_dir / "15_example_cross_sucking").mkdir()

    return root_dir


def test_check_temp_folder_structure(temp_folder_structure):
    """Check the tmp folder structure is created as expected"""
    cross_sucking_clips = temp_folder_structure / "cross_sucking_clips"
    cs_clips_pen5_group3 = (
        temp_folder_structure / "cross_sucking_clips" / "Pen 5 - Group 3"
    )
    raw_clips_pen5_group3_postweaning_day3 = (
        temp_folder_structure
        / "raw_cross_sucking_datalog"
        / "videos"
        / "Pen 5 - Group 3"
        / "POSTWEANING"
        / "Day 3"
    )
    assert cross_sucking_clips.exists()
    assert cs_clips_pen5_group3.exists()
    assert raw_clips_pen5_group3_postweaning_day3.exists()


def test_check_correct_return_value():
    pass


def test_check_empty_root_directory(tmp_path):
    """Raise FileNotFoundError if root directory is empty"""
    pass


def test_check_empty_root_directory(tmp_path):
    """Raise FileNotFoundError if root directory is empty"""
    pass


# --- csv Output Checks ---

# Test existence

# Test Strucutre

# Test Content (correct, values, len, )


# --- Index Tests ---
