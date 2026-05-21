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
        clip_num = 1
        for pen in pens:
            for period in periods:
                for day in days:
                    direc = root_dir / cs_folder / pen / period / day
                    (direc).mkdir(parents=True, exist_ok=True)
                    for i in range(5):
                        if i == 2:
                            part = "part_01"
                        elif i == 3:
                            part = "part_02"
                        else:
                            part = ""

                        file = f"CS_{clip_num:04d}_{period}_d{day[-1]}_p{pen[4]}_28082025_ch01-20250828084247_0345_0556{part}.mp4"
                        (direc / file).touch()
                        clip_num += 1

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
        / "Day 3" /

    )
    assert cross_sucking_clips.exists()
    assert cs_clips_pen5_group3.exists()
    assert raw_clips_pen5_group3_postweaning_day3.exists()


# --- Happy Tests ---


# --- Edge Cases ---

# --- Error Handling ---

# --- Data Intergrity ---


# --- csv Output Checks ---


# Test existence
def test_check_csv_exists(tmp_path):
    """Check the function returns the csv file at the correct path"""
    pass


# Test Structure
def test_check_col_names(tmp_path):
    """Check the function returns a csv file with the correct column names"""
    pass


# Test Content (correct, values, len, )
def test_check_length(tmp_path):
    """Check the csv has the correct length"""
    pass


# --- Index Tests ---

# --- csv Output Checks with index ---




###########################

import pytest
import pandas as pd
from pathlib import Path

@pytest.fixture
def video_setup(tmp_path):
    """Create test directory structure and CSV."""
    # Create some actual video files
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    
    (videos_dir / "video1.mp4").touch()
    (videos_dir / "subdir").mkdir()
    (videos_dir / "subdir" / "video2.avi").touch()
    
    # Create CSV with mix of existing and non-existing paths
    csv_path = tmp_path / "videos.csv"
    df = pd.DataFrame({
        'path': [
            str(videos_dir / "video1.mp4"),  # exists
            str(videos_dir / "nonexistent.mp4"),  # doesn't exist
            str(videos_dir / "subdir" / "video2.avi"),  # exists
            str(videos_dir / "missing" / "video3.mov"),  # doesn't exist
        ],
        'title': ['Video 1', 'Video 2', 'Video 3', 'Video 4'],
        'duration': [120, 90, 150, 200]
    })
    df.to_csv(csv_path, index=False)
    
    return {
        'csv_path': csv_path,
        'videos_dir': videos_dir,
        'existing_count': 2
    }


def test_filters_existing_paths(video_setup):
    """Should return only rows where video files exist."""
    result = filter_existing_videos(video_setup['csv_path'])
    assert len(result) == video_setup['existing_count']
    assert all(Path(p).exists() for p in result['path'])


def test_preserves_other_columns(video_setup):
    """Should keep all original columns intact."""
    result = filter_existing_videos(video_setup['csv_path'])
    assert list(result.columns) == ['path', 'title', 'duration']
    assert 'Video 1' in result['title'].values


def test_empty_result_when_no_paths_exist(tmp_path):
    """Should return empty DataFrame when no paths exist."""
    csv_path = tmp_path / "empty.csv"
    df = pd.DataFrame({
        'path': ['/nonexistent/video1.mp4', '/fake/video2.avi'],
        'title': ['A', 'B']
    })
    df.to_csv(csv_path, index=False)
    
    result = filter_existing_videos(csv_path)
    assert len(result) == 0
    assert isinstance(result, pd.DataFrame)


def test_handles_empty_csv(tmp_path):
    """Should handle CSV with no data rows."""
    csv_path = tmp_path / "empty.csv"
    df = pd.DataFrame(columns=['path', 'title'])
    df.to_csv(csv_path, index=False)
    
    result = filter_existing_videos(csv_path)
    assert len(result) == 0


def test_handles_null_paths(tmp_path):
    """Should handle None/NaN values in path column."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    (videos_dir / "real.mp4").touch()
    
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'path': [str(videos_dir / "real.mp4"), None, '', pd.NA],
        'title': ['A', 'B', 'C', 'D']
    })
    df.to_csv(csv_path, index=False)
    
    result = filter_existing_videos(csv_path)
    assert len(result) == 1


def test_csv_file_not_found():
    """Should raise appropriate error when CSV doesn't exist."""
    with pytest.raises(FileNotFoundError):
        filter_existing_videos('/nonexistent/file.csv')


def test_missing_path_column(tmp_path):
    """Should raise error when expected column is missing."""
    csv_path = tmp_path / "bad.csv"
    df = pd.DataFrame({'video_file': ['something'], 'title': ['A']})
    df.to_csv(csv_path, index=False)
    
    with pytest.raises((KeyError, ValueError)):
        filter_existing_videos(csv_path)


def test_relative_paths(tmp_path):
    """Should handle relative paths correctly."""
    # This tests whether your function resolves relative paths
    # from the CSV's directory or current working directory
    pass  # Implementation depends on your requirements


def test_does_not_mutate_original_csv(video_setup):
    """Should not modify the original CSV file."""
    original_df = pd.read_csv(video_setup['csv_path'])
    original_len = len(original_df)
    
    filter_existing_videos(video_setup['csv_path'])
    
    after_df = pd.read_csv(video_setup['csv_path'])
    assert len(after_df) == original_len