"""
Module for testing is_match from matching.py
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

from scripts.data_reading.matching import is_match


class TestIsMatch:

    # Basic Matching Functionality:
    def test_check_basic_match_variations(self):
        """Check matches normal result: 0001.zip"""
        unlabelled_name = (
            "CS_0001_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577.mp4"
        )
        labelled_name = "0001.zip"
        result = is_match(unlabelled_name, labelled_name)
        assert result

    def test_check_basic_match_not_found(self):
        """Check matches normal result: 0001.zip"""
        unlabelled_name = (
            "CS_9999_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577.mp4"
        )
        labelled_name = "0001.zip"
        result = is_match(unlabelled_name, labelled_name)
        assert not result

    def test_check_basic_part_match(self):
        """Check matches normal part result: 0002_part01.zip"""
        unlabelled_name = (
            "CS_0002_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part01.mp4"
        )
        labelled_name = "0002_part01.zip"
        result = is_match(unlabelled_name, labelled_name)
        assert result

    def test_check_basic_part_match_2(self):
        """Check matches normal part result: 0002_part02.zip"""
        unlabelled_name = (
            "CS_0002_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part02.mp4"
        )
        labelled_name = "0002_part02.zip"
        result = is_match(unlabelled_name, labelled_name)
        assert result

    def test_check_basic_part_match_variation(self):
        """Check matches variation of part result: 0003_part1.zip"""
        unlabelled_name = (
            "CS_0003_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part01.mp4"
        )
        labelled_name = "0003_part1.zip"
        result = is_match(unlabelled_name, labelled_name)
        assert result

    def test_check_basic_part_match_variation_2(self):
        """Check matches variation of part result: 0003_part4.zip"""
        unlabelled_name = (
            "CS_0003_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part02.mp4"
        )
        labelled_name = "0003_part2.zip"
        result = is_match(unlabelled_name, labelled_name)
        assert result

    def test_check_basic_part_match_variation_3(self):
        """Check matches variation of part result: 0004_p01.zip"""
        unlabelled_name = (
            "CS_0004_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part01.mp4"
        )
        labelled_name = "0004_p01.zip"
        result = is_match(unlabelled_name, labelled_name)
        assert result

    def test_check_basic_part_match_variation_4(self):
        """Check matches variation of part result: 0004_p02.zip"""
        unlabelled_name = (
            "CS_0004_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part02.mp4"
        )
        labelled_name = "0004_p02.zip"
        result = is_match(unlabelled_name, labelled_name)
        assert result

    def test_check_basic_part_match_variation_5(self):
        """Check matches variation of part result: 0007_p1.zip"""
        unlabelled_name = (
            "CS_0007_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part01.mp4"
        )
        labelled_name = "0007_p1.zip"
        result = is_match(unlabelled_name, labelled_name)
        assert result

    def test_check_basic_part_match_variation_6(self):
        """Check matches variation of part result: 0007_p2.zip"""
        unlabelled_name = (
            "CS_0007_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part02.mp4"
        )
        labelled_name = "0007_p2.zip"
        result = is_match(unlabelled_name, labelled_name)
        assert result

    def test_check_basic_part_match_variation_7(self):
        """Check matches variation of part result: 0008-p1.zip"""
        unlabelled_name = (
            "CS_0008_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part01.mp4"
        )
        labelled_name = "0008-p1.zip"
        result = is_match(unlabelled_name, labelled_name)
        assert result

    def test_check_basic_part_match_variation_8(self):
        """Check matches variation of part result: 0008-p2.zip"""
        unlabelled_name = (
            "CS_0008_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part02.mp4"
        )
        labelled_name = "0008-p2.zip"
        result = is_match(unlabelled_name, labelled_name)
        assert result

    def test_check_basic_part_match_variation_9(self):
        """Check matches variation of part result: 0012 - p1.zip"""
        unlabelled_name = (
            "CS_0012_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part01.mp4"
        )
        labelled_name = "0012 - p1.zip"
        result = is_match(unlabelled_name, labelled_name)
        assert result

    def test_check_basic_part_match_variation_10(self):
        """Check matches variation of part result: 0012 - p2.zip"""
        unlabelled_name = (
            "CS_0012_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part02.mp4"
        )
        labelled_name = "0012 - p2.zip"
        result = is_match(unlabelled_name, labelled_name)
        assert result

    def test_check_basic_matches_typo(self):
        """Check matches basic format typo: 010.zip"""
        unlabelled_name = (
            "CS_0010_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577.mp4"
        )
        labelled_name = "010.zip"
        result = is_match(unlabelled_name, labelled_name)
        assert result

    def test_check_basic_matches_extra_zip(self):
        """Check matches basic format typo: 0011_part01zip.zip"""
        unlabelled_name = (
            "CS_0011_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part01.mp4"
        )
        labelled_name = "0011_part01zip.zip"
        result = is_match(unlabelled_name, labelled_name)
        assert result

    def test_check_basic_matches_extra_zip_2(self):
        """Check matches basic format typo: 0011_part01zip.zip"""
        unlabelled_name = (
            "CS_0011_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part02.mp4"
        )
        labelled_name = "0011_part02zip.zip"
        result = is_match(unlabelled_name, labelled_name)
        assert result

    def test_check_basic_match_full_name(self):
        """Check matches basic format typo: CS_0005_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577.mp4.zip"""
        unlabelled_name = (
            "CS_0005_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577.mp4"
        )
        labelled_name = (
            "CS_0005_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577.mp4.zip"
        )
        result = is_match(unlabelled_name, labelled_name)
        assert result

    def test_check_basic_match_full_name_with_part(self):
        """Check matches basic format typo: CS_0006_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part01.mp4.zip"""
        unlabelled_name = (
            "CS_0006_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part01.mp4"
        )
        labelled_name = "CS_0006_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part01.mp4.zip"
        result = is_match(unlabelled_name, labelled_name)
        assert result

    def test_check_basic_match_full_name_with_part_2(self):
        """Check matches basic format typo: CS_0006_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part02.mp4.zip"""
        unlabelled_name = (
            "CS_0006_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part02.mp4"
        )
        labelled_name = "CS_0006_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part02.mp4.zip"
        result = is_match(unlabelled_name, labelled_name)
        assert result

    def test_check_basic_match_double_periods(self):
        """Check matches double dot typo: 1234..zip"""
        unlabelled_name = (
            "CS_0123_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577.mp4"
        )
        labelled_name = "0123..zip"
        result = is_match(unlabelled_name, labelled_name)
        assert result

    def test_check_basic_match_no_underscore(self):
        """Check matches double dot typo: 1234..zip"""
        unlabelled_name = (
            "CS_0124_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part01.mp4"
        )
        labelled_name = "0124p1.zip"
        result = is_match(unlabelled_name, labelled_name)
        assert result


class TestFindMatchEdgeCases:

    def test_bad_input_unlabelled_names(self):
        """Check non-string inputs raises error"""
        with pytest.raises(TypeError):
            is_match(100, "0012 - p2.zip")

    def test_bad_input_labelled_names(self):
        """Check non-string inputs raises error"""
        with pytest.raises(TypeError):
            is_match(
                "CS_0012_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577.mp4.zip",
                100,
            )

    def test_empty_unlabelled_name(self):
        """Check empty unlabelled_name input raises error"""
        with pytest.raises(ValueError):
            is_match(" ", "0012 - p2.zip")
        with pytest.raises(ValueError):
            is_match("", "0012 - p2.zip")
