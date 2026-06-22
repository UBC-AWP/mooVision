"""
Tests for scripts/preprocessing/extract_labels.py
"""

import sys
from pathlib import Path
from typing import Tuple
import zipfile
import pytest
from unittest.mock import patch

sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.preprocessing.extract_labels import (
    generate_label_metadata,
    extract_single_zip,
    parse_zip_annotations,
    save_label_batch,
    extract_labels,
)


class TestGenerateLabelMetadata:
    """
    Tests metadata generation using live parse_labelled_name function.
    """

    @pytest.mark.parametrize(
        "zip_input, expected_key, expected_prefix",
        [
            ("0042_part1.zip", (42, 1), "0042_part01_"),
            ("105_part2.zip", (105, 2), "0105_part02_"),
            ("CS_0003_p1.zip", (3, 1), "0003_part01_"),
        ],
    )
    def test_generate_label_metadata__successfully_with_parts(
        self, zip_input: str, expected_key: Tuple[int, str], expected_prefix: str
    ):
        """Test valid filenames with parts names parse cleanly through the full pipeline."""
        video_key, file_prefix = generate_label_metadata(zip_input)

        assert video_key == expected_key
        assert file_prefix == expected_prefix

    @pytest.mark.parametrize(
        "zip_input, expected_key, expected_prefix",
        [
            ("0042.zip", (42, None), "0042_None_"),
            ("105.zip", (105, None), "0105_None_"),
            ("CS_0003.zip", (3, None), "0003_None_"),
        ],
    )
    def test_generate_label_metadata_successfully_without_parts(
        self, zip_input: str, expected_key: Tuple[int, None], expected_prefix: str
    ):
        """Test valid filenames without parts fall back to None across both functions."""
        video_key, file_prefix = generate_label_metadata(zip_input)

        assert video_key == expected_key
        assert file_prefix == expected_prefix

    @pytest.mark.parametrize(
        "malformed_zip",
        [
            "invalid_filename.zip",  # Completely missing numbers
            "cvat_export_abc_part1.zip",  # Non-numeric tracking ID
            "clip_0042_part2.zip",  # Additional `clip_` scetion at front of name
        ],
    )
    def test_generate_labelled_metadata_fails_on_invalid_filenames(
        self, malformed_zip: str
    ):
        """Verify that if parse_labelled_name fails, it bubbles up cleanly through metadata calls."""
        with pytest.raises((ValueError)):
            generate_label_metadata(malformed_zip)

    @pytest.mark.parametrize("invalid_input", [123, ["zip_name.zip"], dict(), None])
    def test_raises_type_error_for_non_string_inputs(self, invalid_input):
        """Verify TypeError is raised if the input is not a string."""
        with pytest.raises(TypeError) as exc_info:
            generate_label_metadata(invalid_input)

        assert "must be a string" in str(exc_info.value)

    @pytest.mark.parametrize("empty_input", ["", "   ", "\n", "\t"])
    def test_raises_value_error_for_empty_or_whitespace_string_inputs(
        self, empty_input
    ):
        """Verify ValueError is raised for empty or whitespace-only inputs."""
        with pytest.raises(ValueError) as exc_info:
            generate_label_metadata(empty_input)

        assert "cannot be an empty string or whitespace" in str(exc_info.value)


class TestExtractSingleZip:
    """Tests validating in-memory zip asset extraction, strides, and error checking."""

    # --- Mock Zipfile for Testing ---

    @pytest.fixture
    def create_mock_zip(self):
        """
        Helper factory fixture to generate mock zip archives in a temporary directory.
        Returns a callable that writes custom frame numbers into a zip file on disk.
        """

        def _generate_zip(tmp_path: Path, filename_list: list[str]) -> Path:
            zip_target = tmp_path / "mock_annotations.zip"
            with zipfile.ZipFile(zip_target, "w") as zf:
                for name in filename_list:
                    # Write simple bounding box content bytes inside the archive
                    zf.writestr(name, "0 0.5 0.5 0.2 0.2\n")
            return zip_target

        return _generate_zip

    # --- HAPPY PATH TESTS (Stride downsampling & Registry Filtering) ---

    def test_extract_single_zip_success_with_filtering(self, tmp_path, create_mock_zip):
        """Verify successful stride alignment and clean filtering against a valid frame registry."""

        archive_contents = [
            "obj_train_data/frame_000000.txt",  # Divisible by 5, inside registry -> Keep
            "obj_train_data/frame_000001.txt",  # Not divisible by 5 -> Drop by skip
            "obj_train_data/frame_000002.txt",  # Not divisible by 5 -> Drop by skip
            "obj_train_data/frame_000003.txt",  # Not divisible by 5 -> Drop by skip
            "obj_train_data/frame_000004.txt",  # Not divisible by 5 -> Drop by skip
            "obj_train_data/frame_000005.txt",  # Divisible by 5, missing from registry -> Drop by registry
            "obj_train_data/frame_000010.txt",  # Divisible by 5, inside registry -> Keep
            "ignored_folder/frame_000015.txt",  # Divisible by 5, but wrong folder -> Ignore
        ]
        mock_zip = create_mock_zip(tmp_path, archive_contents)

        v_key = (42, "part1")
        # Registry only contains image frames 0 and 10
        f_registry = {v_key: {0, 10}}
        prefix = "0042_part01_"

        label_batch, saved_frames = extract_single_zip(
            zip_path=mock_zip,
            frame_registry=f_registry,
            skip=5,
            file_prefix=prefix,
            video_key=v_key,
        )

        assert saved_frames == {0, 10}
        assert len(label_batch) == 2
        assert f"{prefix}frame_000000.txt" in label_batch
        assert f"{prefix}frame_000010.txt" in label_batch
        assert label_batch[f"{prefix}frame_000000.txt"] == b"0 0.5 0.5 0.2 0.2\n"

    def test_extract_single_zip_none_registry_extracts_all_matching_strides(
        self, tmp_path, create_mock_zip
    ):
        """If frame_registry is None, all elements meeting stride downsampling should be extracted."""

        archive_contents = [
            "obj_train_data/frame_000000.txt",  # Keep
            "obj_train_data/frame_000002.txt",  # Drop
            "obj_train_data/frame_000004.txt",  # Keep
        ]
        mock_zip = create_mock_zip(tmp_path, archive_contents)

        label_batch, saved_frames = extract_single_zip(
            zip_path=mock_zip,
            frame_registry=None,
            skip=4,
            file_prefix="demo_",
            video_key=(1, None),
        )

        assert saved_frames == {0, 4}
        assert len(label_batch) == 2

    # --- ERROR & EXCEPTION BOUNDARY TESTS ---

    def test_extract_single_zip_raises_file_not_found(self):
        """Ensure an explicit FileNotFoundError occurs if the target zip path doesn't exist."""
        ghost_path = Path("/dummy/directory/missing_archive.zip")
        with pytest.raises(FileNotFoundError) as exc_info:
            extract_single_zip(ghost_path, {}, 5, "prefix_", (1, None))
        assert "Target archive zip not found" in str(exc_info.value)

    def test_extract_single_zip_raises_value_error_for_malformed_internal_names(
        self, tmp_path, create_mock_zip
    ):
        """Ensure ValueError is thrown when internal filenames violate numerical slicing format."""
        # Arrange: Name is missing the structured numeric sequence format
        bad_contents = ["obj_train_data/frame_badname.txt"]
        mock_zip = create_mock_zip(tmp_path, bad_contents)

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            extract_single_zip(mock_zip, None, 1, "prefix_", (1, None))

        assert "CRITICAL: Structural naming mismatch" in str(exc_info.value)
        assert "Expected 'frame_XXXXXX.txt'" in str(exc_info.value)

    def test_extract_single_zip_raises_value_error_for_invalid_skip_value(
        self, tmp_path, create_mock_zip
    ):
        """Verify that skip values <= 0 throw an explicit ValueError."""
        mock_zip = create_mock_zip(tmp_path, ["obj_train_data/frame_000000.txt"])

        with pytest.raises(ValueError) as exc_info:
            extract_single_zip(mock_zip, None, 0, "prefix_", (1, None))
        assert "must be a positive integer greater than 0" in str(exc_info.value)

    @pytest.mark.parametrize(
        "arg_name, kwargs",
        [
            ("zip_path", {"zip_path": "not_a_path_obj.zip"}),
            (
                "frame_registry",
                {"frame_registry": [1, 2, 3]},
            ),  # List instead of Dict/None
            ("skip", {"skip": "five"}),
            ("file_prefix", {"file_prefix": 12345}),
            ("video_key", {"video_key": "not_a_tuple"}),
        ],
    )
    def test_extract_single_zip_raises_type_errors_on_invalid_inputs(
        self, arg_name, kwargs, tmp_path, create_mock_zip
    ):
        """Systematically verify TypeErrors trigger whenever inputs violate signature types."""
        # Setup base correct defaults
        default_zip = create_mock_zip(tmp_path, ["obj_train_data/frame_000000.txt"])
        base_args = {
            "zip_path": default_zip,
            "frame_registry": {},
            "skip": 5,
            "file_prefix": "prefix_",
            "video_key": (1, None),
        }
        # Overwrite the target validation parameter from parameterization configuration
        base_args.update(kwargs)

        with pytest.raises(TypeError) as exc_info:
            extract_single_zip(**base_args)
        assert f"Argument '{arg_name}' must be" in str(exc_info.value)


class TestParseZipAnnotations:
    """Tests validating the aggregation loop and orchestration for zip parsing."""

    @patch("scripts.preprocessing.extract_labels.extract_single_zip")
    @patch("scripts.preprocessing.extract_labels.generate_label_metadata")
    def test_parse_zip_annotations_aggregation_success(
        self, mock_metadata, mock_extract
    ):
        """Ensure multiple zip paths are iterated, metadata is resolved, and batches are merged."""
        # Arrange
        path1 = Path("cvat_export_01_part1.zip")
        path2 = Path("cvat_export_02.zip")
        validated_paths = [path1, path2]

        frame_registry = {(1, "1"): {0, 5}, (2, None): {0, 10}}
        skip = 5

        # Configure mock metadata returns for both loop cycles
        mock_metadata.side_effect = [
            ((1, "1"), "0001_part01_"),
            ((2, None), "0002_None_"),
        ]

        # Configure mock extractions simulating individual un-merged batches
        mock_extract.side_effect = [
            ({"0001_part01_frame_000000.txt": b"bytes1"}, {0}),
            ({"0002_None_frame_000000.txt": b"bytes2"}, {0}),
        ]

        # Act
        # Note: This test assumes you fixed the return bug to return global_label_batch!
        global_batch, labels_registry = parse_zip_annotations(
            validated_paths=validated_paths, frame_registry=frame_registry, skip=skip
        )

        # Assert: Verify the loop ran exactly twice with correct matching arguments
        assert mock_metadata.call_count == 2
        mock_metadata.assert_any_call(zip_name="cvat_export_01_part1.zip")
        mock_metadata.assert_any_call(zip_name="cvat_export_02.zip")

        assert mock_extract.call_count == 2
        mock_extract.assert_any_call(
            zip_path=path1,
            frame_registry=frame_registry,
            skip=skip,
            file_prefix="0001_part01_",
            video_key=(1, "1"),
        )

        # Assert: Verify global dictionary aggregation works flawlessly
        assert len(global_batch) == 2
        assert "0001_part01_frame_000000.txt" in global_batch
        assert "0002_None_frame_000000.txt" in global_batch

        # Assert: Verify tracking registry construction matches expected outputs
        assert labels_registry == {(1, "1"): {0}, (2, None): {0}}

    def test_parse_zip_annotations_handles_empty_input_list(self):
        """An empty list of validated paths should immediately return empty collections cleanly."""
        # Act
        global_batch, labels_registry = parse_zip_annotations(
            validated_paths=[], frame_registry={}, skip=5
        )

        # Assert
        assert global_batch == {}
        assert labels_registry == {}

    @patch("scripts.preprocessing.extract_labels.generate_label_metadata")
    def test_parse_zip_annotations_bubbles_up_exceptions(self, mock_metadata):
        """Ensure exceptions raised deeper in the pipeline components bubble up to stop execution."""
        # Arrange
        validated_paths = [Path("malformed_archive.zip")]

        # Simulate a regex breakdown or bad type passing inside metadata resolution
        mock_metadata.side_effect = ValueError("CRITICAL: Structural naming mismatch")

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            parse_zip_annotations(validated_paths, {}, 5)

        assert "Structural naming mismatch" in str(exc_info.value)

    @patch("scripts.preprocessing.extract_labels.extract_single_zip")
    @patch("scripts.preprocessing.extract_labels.generate_label_metadata")
    def test_parse_zip_annotations_bubbles_worker_value_error(
        self, mock_metadata, mock_extract
    ):
        """Ensure that if a single zip file is corrupted inside the loop, the error bubbles up instantly."""
        # Arrange
        validated_paths = [Path("good_file.zip"), Path("corrupted_file.zip")]
        mock_metadata.side_effect = [((1, None), "prefix_1_"), ((2, None), "prefix_2_")]

        # First zip succeeds, second zip raises a ValueError due to internal corruption
        mock_extract.side_effect = [
            ({"frame_1.txt": b"data"}, {1}),
            ValueError(
                "CRITICAL: Structural naming mismatch inside archive corrupted_file.zip"
            ),
        ]

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            parse_zip_annotations(validated_paths, {}, 5)

        assert "Structural naming mismatch" in str(exc_info.value)
        # Verify execution stopped immediately on the second file and didn't try to continue
        assert mock_extract.call_count == 2
