"""
Testing module for matching.py
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
    find_match,
    extract_numeric_id,
    parse_labelled_name,
    parse_unlabelled_name,
)


class TestClipMatcherBasicMatching:
    """Test basic matching functionality."""

    # @pytest.fixture
    # def matcher(self):
    #     """Create a ClipMatcher instance for tests."""
    #     return ClipMatcher()

    @pytest.fixture
    def labelled_clips(self):
        """Standard set of labelled clips for testing."""
        return [
            "0001.zip",
            "0002_part02.zip",
            "0002_part01.zip",
            "0003_part1.zip",
            "0003_part2.zip",
            "0004_p01.zip",
            "0004_p02.zip",
            "CS_0005_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577.mp4.zip",
            "CS_0006_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part01.mp4.zip",
            "CS_0006_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part02.mp4.zip",
            "0007_p1.zip",
            "0007_p2.zip",
            "0008-p1.zip",
            "0008-p2.zip",
            "010.zip",
            "0011_part01zip.zip",
            "0011_part02zip.zip",
            "0012 - p1.zip",
            "0012 - p2.zip",
        ]

    # Basic Matching Functionality:
    def test_check_bad_input(self, labelled_clips):
        """Check matches normal result: 0001.zip"""
        unlabelled_name = 4
        expected = "0001.zip"
        result = find_match(unlabelled_name, labelled_clips)
        assert result == expected

    def test_check_basic_match(self, labelled_clips):
        """Check matches normal result: 0001.zip"""
        unlabelled_name = (
            "CS_0001_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577.mp4"
        )
        expected = "0001.zip"
        result = find_match(unlabelled_name, labelled_clips)
        assert result == expected

    def test_check_basic_match_not_found(self, labelled_clips):
        """Check matches normal result: 0001.zip"""
        unlabelled_name = (
            "CS_9999_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577.mp4"
        )
        expected = None
        result = find_match(unlabelled_name, labelled_clips)
        assert result == expected

    def test_check_basic_part_match(self, labelled_clips):
        """Check matches normal part result: 0002_part01.zip"""
        unlabelled_name = (
            "CS_0002_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part01.mp4"
        )
        expected = "0002_part01.zip"
        result = find_match(unlabelled_name, labelled_clips)
        assert result == expected

    def test_check_basic_part_match_2(self, labelled_clips):
        """Check matches normal part result: 0002_part02.zip"""
        unlabelled_name = (
            "CS_0002_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part02.mp4"
        )
        expected = "0002_part02.zip"
        result = find_match(unlabelled_name, labelled_clips)
        assert result == expected

    def test_check_basic_part_match_variation(self, labelled_clips):
        """Check matches variation of part result: 0003_part1.zip"""
        unlabelled_name = (
            "CS_0003_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part01.mp4"
        )
        expected = "0003_part1.zip"
        result = find_match(unlabelled_name, labelled_clips)
        assert result == expected

    def test_check_basic_part_match_variation_2(self, labelled_clips):
        """Check matches variation of part result: 0003_part4.zip"""
        unlabelled_name = (
            "CS_0003_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part02.mp4"
        )
        expected = "0003_part2.zip"
        result = find_match(unlabelled_name, labelled_clips)
        assert result == expected

    def test_check_basic_part_match_variation_3(self, labelled_clips):
        """Check matches variation of part result: 0004_p01.zip"""
        unlabelled_name = (
            "CS_0004_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part01.mp4"
        )
        expected = "0004_p01.zip"
        result = find_match(unlabelled_name, labelled_clips)
        assert result == expected

    def test_check_basic_part_match_variation_4(self, labelled_clips):
        """Check matches variation of part result: 0004_p02.zip"""
        unlabelled_name = (
            "CS_0004_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part02.mp4"
        )
        expected = "0004_p02.zip"
        result = find_match(unlabelled_name, labelled_clips)
        assert result == expected

    def test_check_basic_part_match_variation_5(self, labelled_clips):
        """Check matches variation of part result: 0007_p1.zip"""
        unlabelled_name = (
            "CS_0007_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part01.mp4"
        )
        expected = "0007_p1.zip"
        result = find_match(unlabelled_name, labelled_clips)
        assert result == expected

    def test_check_basic_part_match_variation_6(self, labelled_clips):
        """Check matches variation of part result: 0007_p2.zip"""
        unlabelled_name = (
            "CS_0007_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part02.mp4"
        )
        expected = "0007_p2.zip"
        result = find_match(unlabelled_name, labelled_clips)
        assert result == expected

    def test_check_basic_part_match_variation_7(self, labelled_clips):
        """Check matches variation of part result: 0008-p1.zip"""
        unlabelled_name = (
            "CS_0008_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part01.mp4"
        )
        expected = "0008-p1.zip"
        result = find_match(unlabelled_name, labelled_clips)
        assert result == expected

    def test_check_basic_part_match_variation_8(self, labelled_clips):
        """Check matches variation of part result: 0008-p2.zip"""
        unlabelled_name = (
            "CS_0008_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part02.mp4"
        )
        expected = "0008-p2.zip"
        result = find_match(unlabelled_name, labelled_clips)
        assert result == expected

    def test_check_basic_part_match_variation_9(self, labelled_clips):
        """Check matches variation of part result: 0012 - p1.zip"""
        unlabelled_name = (
            "CS_0012_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part01.mp4"
        )
        expected = "0012 - p1.zip"
        result = find_match(unlabelled_name, labelled_clips)
        assert result == expected

    def test_check_basic_part_match_variation_10(self, labelled_clips):
        """Check matches variation of part result: 0012 - p2.zip"""
        unlabelled_name = (
            "CS_0012_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part02.mp4"
        )
        expected = "0012 - p2.zip"
        result = find_match(unlabelled_name, labelled_clips)
        assert result == expected

    def test_check_basic_matches_typo(self, labelled_clips):
        """Check matches basic format typo: 010.zip"""
        unlabelled_name = (
            "CS_0010_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577.mp4"
        )
        expected = "010.zip"
        result = find_match(unlabelled_name, labelled_clips)
        assert result == expected

    def test_check_basic_matches_extra_zip(self, labelled_clips):
        """Check matches basic format typo: 0011_part01zip.zip"""
        unlabelled_name = (
            "CS_0011_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part01.mp4"
        )
        expected = "0011_part01zip.zip"
        result = find_match(unlabelled_name, labelled_clips)
        assert result == expected

    def test_check_basic_matches_extra_zip_2(self, labelled_clips):
        """Check matches basic format typo: 0011_part01zip.zip"""
        unlabelled_name = (
            "CS_0011_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part02.mp4"
        )
        expected = "0011_part02zip.zip"
        result = find_match(unlabelled_name, labelled_clips)
        assert result == expected

    def test_check_basic_match_full_name(self, labelled_clips):
        """Check matches basic format typo: CS_0005_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577.mp4.zip"""
        unlabelled_name = (
            "CS_0005_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577.mp4"
        )
        expected = (
            "CS_0005_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577.mp4.zip"
        )
        result = find_match(unlabelled_name, labelled_clips)
        assert result == expected

    def test_check_basic_match_full_name_with_part(self, labelled_clips):
        """Check matches basic format typo: CS_0006_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part01.mp4.zip"""
        unlabelled_name = (
            "CS_0006_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part01.mp4"
        )
        expected = "CS_0006_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part01.mp4.zip"
        result = find_match(unlabelled_name, labelled_clips)
        assert result == expected

    def test_check_basic_match_full_name_with_part_2(self, labelled_clips):
        """Check matches basic format typo: CS_0006_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part02.mp4.zip"""
        unlabelled_name = (
            "CS_0006_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part02.mp4"
        )
        expected = "CS_0006_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part02.mp4.zip"
        result = find_match(unlabelled_name, labelled_clips)
        assert result == expected


class TestEdgeCases:

    def test_bad_input():
        """Test Bad inputs to find match"""
        assert False


class TestNumericIDExtraction:
    """Test numeric ID extraction logic."""

    def test_extract_numeric_id_cs_format(self, matcher):
        """Test extraction of ID from CS_XXXX format."""
        test_id = extract_numeric_id("CS_0317_WEAN_d2_p2_cow6.mp4.zip")
        assert test_id == "317"  # Leading zeros removed

    def test_extract_simple_numeric_id(self, matcher):
        """Test extraction of simple numeric ID."""
        test_id = extract_numeric_id("1015_part01.zip")
        assert test_id == "1015"

    def test_extract_id_with_leading_zeros(self, matcher):
        """Test that leading zeros are removed."""
        test_id = extract_numeric_id("CS_0001_test.zip")
        assert test_id == "1"

        test_id2 = extract_numeric_id("0123_test.zip")
        assert test_id2 == "123"

    def test_extract_id_with_short_input(self, matcher):
        """Test that leading zeros are removed."""
        test_id = extract_numeric_id("CS_445_test.zip")
        assert test_id == "445"

        test_id2 = extract_numeric_id("3_test.zip")
        assert test_id2 == "3"

    def test_no_numeric_id(self, matcher):
        """Test when no numeric ID can be extracted."""
        test_id = extract_numeric_id("ABCD_test.zip")
        assert test_id is None

        # Raise ERROR (?)


class TestParsing:
    """Test filename parsing functions."""

    def test_parse_unlabelled_with_part(self, matcher):
        """Test parsing unlabelled name with part number."""
        base, part = parse_unlabelled_name("CS_0319_WEAN_test_part01.mp4")
        assert base == "CS_0319_WEAN_test"
        assert part == 1

    def test_parse_unlabelled_without_part(self, matcher):
        """Test parsing unlabelled name without part number."""
        numeric_id, part = parse_unlabelled_name("CS_0317_WEAN_test.mp4")
        assert numeric_id == "317"
        assert part is None

    def test_parse_labelled_various_formats(self, matcher):
        """Test parsing various labelled formats."""
        # Format: _part01
        numeric_id, part = parse_labelled_name("0045_part01.zip")
        assert numeric_id == "45"
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

    def test_parse_labelled_no_part(self, matcher):
        """Test parsing labelled name without part number."""
        numeric_id, part = parse_labelled_name("1234.zip")
        assert numeric_id == "1234"
        assert part is None
