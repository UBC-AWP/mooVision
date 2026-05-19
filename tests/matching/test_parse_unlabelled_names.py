"""
Module for testing parse_unlabelled_names from matching.py
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

from scripts.matching import (
    parse_unlabelled_name,
)


class TestUnlabelledParsing:
    """Test filename parsing functions."""

    def test_parse_unlabelled_without_part(self):
        """Test parsing unlabelled name without part number."""
        numeric_id, part = parse_unlabelled_name("CS_0317_WEAN_test.mp4")
        assert numeric_id == "317"
        assert part is None

    def test_parse_unlabelled_with_part(self):
        """Test parsing unlabelled name with part number."""
        numeric_id, part = parse_unlabelled_name("CS_0319_WEAN_test_part01.mp4")
        assert numeric_id == "319"
        assert part == 1

        numeric_id, part = parse_unlabelled_name("CS_0319_WEAN_test_part02.mp4")
        assert numeric_id == "319"
        assert part == 2

    def test_parse_unlabelled_invalid_input(self):
        """Test parse_unlabelled_name handles bad inputs."""
        with pytest.raises(TypeError):
            parse_unlabelled_name(4123324)

        with pytest.raises(ValueError):
            parse_unlabelled_name("")
