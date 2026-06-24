"""
Tests for scripts/preprocessing/utils.py
"""

import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.preprocessing.utils import (
    validate_file_paths,
)


class TestValidateFilePathsHappyPaths:
    """Grouped Unit tests validating happy path behaviours and clean results."""

    def test_validate_file_paths_success(self, tmp_path):
        """Ensure valid relative paths are resolved, standardized, and returned as Path objects."""
        # Arrange: Build an isolated mock environment via pytest's tmp_path fixture
        file1 = tmp_path / "video_001.mp4"
        file2 = tmp_path / "annotations" / "labels_002.zip"
        file3 = (
            tmp_path
            / "cross_sucking_clips"
            / "POSTWEANING"
            / "Pen 2"
            / "Day 2"
            / "CS_0123_POSTWEANING_p2_d2_01222026_ch_1029284728_1234_4223.mp4"
        )

        file1.touch()
        file2.parent.mkdir(parents=True, exist_ok=True)
        file2.touch()
        file3.parent.mkdir(parents=True, exist_ok=True)
        file3.touch()

        input_paths = [
            "video_001.mp4",
            "annotations/labels_002.zip",
            "cross_sucking_clips/POSTWEANING/Pen 2/Day 2/CS_0123_POSTWEANING_p2_d2_01222026_ch_1029284728_1234_4223.mp4",
        ]

        # Act
        result = validate_file_paths(input_paths, tmp_path)

        # Assert
        assert len(result) == 3
        assert result[0] == file1
        assert result[1] == file2
        assert result[2] == file3
        assert isinstance(result[0], Path)

    def test_validate_file_paths_handles_windows_slashes(self, tmp_path):
        """Verify that backslashes are normalized to POSIX standard paths successfully."""
        # Arrange
        nested_file = tmp_path / "nested" / "folder" / "clip.mp4"
        nested_file.parent.mkdir(parents=True, exist_ok=True)
        nested_file.touch()

        input_paths = ["nested\\folder\\clip.mp4"]

        # Act
        result = validate_file_paths(input_paths, tmp_path)

        # Assert
        assert result[0] == nested_file

    def test_validate_file_paths_empty_list(self, tmp_path):
        """An empty input list should return an empty list without failing."""
        result = validate_file_paths([], tmp_path)
        assert result == []


class TestValidateFilePathsErrorBounds:
    """Grouped unit test validation for boundary failures, error states, and type checks."""

    def test_validate_file_paths_raises_file_not_found(self, tmp_path):
        """Verify raises FileNotFoundError if clean_path cannot be found."""
        # Arrange
        input_paths = ["existing.mp4", "ghost_file.mp4"]
        (
            tmp_path / "existing.mp4"
        ).touch()  # Only simulate the first asset existing on disk

        # Calculate exactly what the absolute failing path boundary should look like
        expected_failing_clean_path = tmp_path / "ghost_file.mp4"

        # Act & Assert
        with pytest.raises(FileNotFoundError) as exc_info:
            validate_file_paths(input_paths, tmp_path)

        # Verify the error string specifically targets the single problematic file path
        error_msg = str(exc_info.value)
        assert "cleaned label file not found:" in error_msg
        assert str(expected_failing_clean_path) in error_msg

    def test_validate_file_paths_invalid_input_types(self):
        """Verify raises TypeError for invalid input types."""
        # Test invalid 'paths' argument type
        with pytest.raises(TypeError) as exc_info:
            validate_file_paths("not_a_list.mp4", Path("/dummy/root"))
        assert "Expected 'label_paths' to be a list" in str(exc_info.value)

        # Test invalid 'root' argument type
        with pytest.raises(TypeError) as exc_info:
            validate_file_paths(["video.mp4"], "/string/path/is/bad")
        assert "Expected 'root' to be a Path object" in str(exc_info.value)

    def test_validate_file_paths_invalid_inner_list_types(self, tmp_path):
        """Verify raises TypeError if paths list contains non-string elements."""
        input_paths = ["valid_string.mp4", 12345, "another_string.mp4"]
        (tmp_path / "valid_string.mp4").touch()

        with pytest.raises(TypeError) as exc_info:
            validate_file_paths(input_paths, tmp_path)

        assert "is not a string path" in str(exc_info.value)
