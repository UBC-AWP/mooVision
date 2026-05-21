"""
Module for testing read_all_clips_index.py
"""

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
    df = pd.DataFrame(
        {
            "path": [
                str(videos_dir / "video1.mp4"),  # exists
                str(videos_dir / "nonexistent.mp4"),  # doesn't exist
                str(videos_dir / "subdir" / "video2.avi"),  # exists
                str(videos_dir / "missing" / "video3.mov"),  # doesn't exist
            ],
            "title": ["Video 1", "Video 2", "Video 3", "Video 4"],
            "duration": [120, 90, 150, 200],
        }
    )
    df.to_csv(csv_path, index=False)

    return {"csv_path": csv_path, "videos_dir": videos_dir, "existing_count": 2}
