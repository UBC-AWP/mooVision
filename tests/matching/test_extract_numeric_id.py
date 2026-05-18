"""
Module for testing the extract_numeric_id funciton from matching.py
"""

import pytest
import sys
from pathlib import Path

# Load in root directory
current_file_path = Path(__file__).resolve()
current_dir = current_file_path.parent
parent_dir = current_file_path.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from scripts.matching import (
    extract_numeric_id,
)


class TestNumericIDExtraction:
    """Test numeric ID extraction logic."""

    def test_extract_numeric_id_cs_format(self):
        """Test extraction of ID from CS_XXXX format."""
        test_id = extract_numeric_id("CS_0317_WEAN_d2_p2_cow6.mp4.zip")
        assert test_id == "317"  # Leading zeros removed

    def test_extract_simple_numeric_id(self):
        """Test extraction of simple numeric ID."""
        test_id = extract_numeric_id("1015_part01.zip")
        assert test_id == "1015"

    def test_extract_id_with_leading_zeros(self):
        """Test that leading zeros are removed."""
        test_id = extract_numeric_id("CS_0001_test.zip")
        assert test_id == "1"

        test_id2 = extract_numeric_id("0123_test.zip")
        assert test_id2 == "123"

    def test_extract_id_with_short_input(self):
        """Test that leading zeros are removed."""
        test_id = extract_numeric_id("CS_445_test.zip")
        assert test_id == "445"

        test_id2 = extract_numeric_id("3_test.zip")
        assert test_id2 == "3"

    def test_no_numeric_id(self):
        """Test error is raised when no numeric ID can be extracted."""
        with pytest.raises(ValueError):
            extract_numeric_id("ABCD_test.zip")
