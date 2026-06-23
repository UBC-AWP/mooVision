"""
Testing module for matching.py
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
    find_match,
)


class TestFindMatchBasicMatching:
    """Test basic matching functionality."""

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
            "0123..zip",
            "0124p1.zip",
        ]

    # Basic Matching Functionality:
    def test_check_basic_match_variations(self, labelled_clips):
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

    def test_check_basic_match_double_periods(self, labelled_clips):
        """Check matches double dot typo: 1234..zip"""
        unlabelled_name = (
            "CS_0123_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577.mp4"
        )
        expected = "0123..zip"
        result = find_match(unlabelled_name, labelled_clips)
        assert result == expected

    def test_check_basic_match_no_underscore(self, labelled_clips):
        """Check matches double dot typo: 1234..zip"""
        unlabelled_name = (
            "CS_0124_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part01.mp4"
        )
        expected = "0124p1.zip"
        result = find_match(unlabelled_name, labelled_clips)
        assert result == expected


class TestFindMatchEdgeCases:

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
            "0012_p2.zip",
            "1234p1.zip",
            "1230..zip",
        ]

    def test_bad_input_unlabelled_names(self, labelled_clips):
        """Check non-string inputs raises error"""
        with pytest.raises(TypeError):
            find_match(100, labelled_clips)

    def test_bad_input_labelled_names(self, labelled_clips):
        """Check non-string inputs raises error"""
        with pytest.raises(TypeError):
            find_match(
                "CS_0012_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577.mp4.zip",
                100,
            )

    def test_empty_unlabelled_name(self, labelled_clips):
        """Check empty unlabelled_name input raises error"""
        with pytest.raises(ValueError):
            find_match(" ", labelled_clips)
        with pytest.raises(ValueError):
            find_match("", labelled_clips)

    def test_empty_labelled_names(self):
        """An empty input list of labelled_names should raise an error"""
        with pytest.raises(ValueError):
            find_match(
                "CS_0012_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577.mp4.zip",
                [],
            )

    def test_labelled_names_list_incorrect_types(self):
        """Check a list of incorrect types raises an error"""
        with pytest.raises(TypeError):
            find_match(
                "CS_0012_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part02.mp4.zip",
                [1, 2, 3, 4, 5, 6, 7, 8, 9],
            )

    def test_multiple_matches_raises_error(self, labelled_clips):
        """Check multiple matches returns an error"""
        with pytest.raises(ValueError):
            find_match(
                "CS_0012_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part02.mp4",
                labelled_clips,
            )
