"""
Tests for scripts/preprocessing/extract_labels.py
"""

import sys
from pathlib import Path
from typing import Tuple
import pytest

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
    Tests metadata generation using live parse_labelled_name fn.
    """

    @pytest.mark.parametrize(
        "zip_input, expected_key, expected_prefix",
        [
            ("0042_part1.zip", (42, "1"), "0042_part01_"),
            ("105_part2.zip", (105, "2"), "0105_part02_"),
            ("CS_0003_p1.zip", (3, "1"), "0003_part01_"),
        ],
    )
    def test_successful_integration_with_partitions(
        self, zip_input: str, expected_key: Tuple[int, str], expected_prefix: str
    ):
        """Ensure real valid partition names parse cleanly through the full pipeline."""
        video_key, file_prefix = generate_label_metadata(zip_input)

        assert video_key == expected_key
        assert file_prefix == expected_prefix

    @pytest.mark.parametrize(
        "zip_input, expected_key, expected_prefix",
        [
            ("cvat_export_42.zip", (42, None), "0042_None_"),
            ("clip_105.zip", (105, None), "0105_None_"),
        ],
    )
    def test_successful_integration_without_partitions(
        self, zip_input: str, expected_key: Tuple[int, None], expected_prefix: str
    ):
        """Ensure real filenames without parts cleanly fall back to None across both functions."""
        video_key, file_prefix = generate_label_metadata(zip_input)

        assert video_key == expected_key
        assert file_prefix == expected_prefix

    @pytest.mark.parametrize(
        "malformed_zip",
        [
            "invalid_filename.zip",  # Completely missing numbers
            "cvat_export_abc_part1.zip",  # Non-numeric tracking ID
            "clip__part2.zip",  # Missing base asset identification
        ],
    )
    def test_integration_fails_on_malformed_regex_patterns(self, malformed_zip: str):
        """Verify that if parse_labelled_name fails, it bubbles up cleanly through metadata calls."""
        with pytest.raises((ValueError, AttributeError)):
            generate_label_metadata(malformed_zip)

    @pytest.mark.parametrize("invalid_input", [123, ["zip_name.zip"], dict(), None])
    def test_raises_type_error_for_non_strings(self, invalid_input):
        """Verify TypeError is explicitly raised if the input is not a string type."""
        with pytest.raises(TypeError) as exc_info:
            generate_label_metadata(invalid_input)

        assert "must be a string" in str(exc_info.value)

    @pytest.mark.parametrize("empty_input", ["", "   ", "\n", "\t"])
    def test_raises_value_error_for_empty_or_whitespace_strings(self, empty_input):
        """Verify ValueError is explicitly triggered for empty or whitespace-only inputs."""
        with pytest.raises(ValueError) as exc_info:
            generate_label_metadata(empty_input)

        assert "cannot be an empty string or whitespace" in str(exc_info.value)
