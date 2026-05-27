"""
Module for testing parse_labelled_names from matching.py
"""

import pytest
import sys
from pathlib import Path

# Load in root directory
current_file_path = Path(__file__).resolve()
current_dir = current_file_path.parent
root_dir = current_file_path.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from scripts.data_reading.matching import (
    parse_labelled_name,
)


class TestLabelledParsing:
    """Test filename parsing functions."""

    def test_parse_labelled_no_part(self):
        """Test parsing labelled name without part number."""
        numeric_id, part = parse_labelled_name("1234.zip")
        assert numeric_id == "1234"
        assert part is None

    def test_parse_labelled_double_dot_typo(self):
        """Test parsing labelled name without part number."""
        numeric_id, part = parse_labelled_name("1230..zip")
        assert numeric_id == "1230"
        assert part is None

    def test_parse_labelled_various_formats(self):
        """Test parsing various labelled formats."""
        # Format: _part01
        numeric_id, part = parse_labelled_name("0045_part01.zip")
        assert numeric_id == "45"
        assert part == 1

        # Format: CS_XXXX_..._.mp4.zip
        numeric_id, part = parse_labelled_name("CS_0317_WEAN_test.mp4.zip")
        assert numeric_id == "317"
        assert part is None

        # Format: CS_XXX_..._part01.mp4.zip
        numeric_id, part = parse_labelled_name("CS_0319_WEAN_test_part01.mp4.zip")
        assert numeric_id == "319"
        assert part == 1

        numeric_id, part = parse_labelled_name("0009_part1.zip")
        assert numeric_id == "9"
        assert part == 1

        # Format: _p01
        numeric_id, part = parse_labelled_name("4567_p01.zip")
        assert numeric_id == "4567"
        assert part == 1

        # Format: -p1
        numeric_id, part = parse_labelled_name("7891-p1.zip")
        assert numeric_id == "7891"
        assert part == 1

        # Format: - p1 (with spaces)
        numeric_id, part = parse_labelled_name("9021 - p1.zip")
        assert numeric_id == "9021"
        assert part == 1

        # Format: typo _part02zip
        numeric_id, part = parse_labelled_name("6789_part02zip.zip")
        assert numeric_id == "6789"
        assert part == 2

        # Format: 1234p1.zip
        numeric_id, part = parse_labelled_name("1234p1.zip")
        assert numeric_id == "1234"
        assert part == 1

    def test_parse_labelled_invalid_input(self):
        """Test that parse_labelled_name handles bad input."""
        with pytest.raises(TypeError):
            parse_labelled_name(123)

        with pytest.raises(ValueError):
            parse_labelled_name("")

    def test_parse_labelled_does_not_match_patterns(self):
        """Test that invalid labelled names raise ValueError."""
        with pytest.raises(ValueError):
            parse_labelled_name("invalid_name_no_numbers.zip")

        with pytest.raises(ValueError):
            parse_labelled_name("ABCD.zip")
