"""
Module for testing read_all_clips_index.py
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from scripts.data_reading.read_all_clips_index import (
    read_data,
    validate_data,
    save_data,
    filter_existing_clips,
    get_label_paths,
    match_label_paths,
    add_label_paths,
    filter_label_paths,
    read_data_from_index_file,
)


class TestReadData:

    # --- Fixtures ---

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    @pytest.fixture
    def csv_file(self, tmp_path, sample_df):
        path = tmp_path / "data.csv"
        sample_df.to_csv(path, index=False)
        return path

    @pytest.fixture
    def xlsx_file(self, tmp_path, sample_df):
        path = tmp_path / "data.xlsx"
        sample_df.to_excel(path, index=False)
        return path

    @pytest.fixture
    def parquet_file(self, tmp_path, sample_df):
        path = tmp_path / "data.parquet"
        sample_df.to_parquet(path, index=False)
        return path

    @pytest.fixture
    def json_file(self, tmp_path, sample_df):
        path = tmp_path / "data.json"
        sample_df.to_json(path)
        return path

    @pytest.fixture
    def tsv_file(self, tmp_path, sample_df):
        path = tmp_path / "data.tsv"
        sample_df.to_csv(path, sep="\t", index=False)
        return path

    # --- Happy path ---

    @pytest.mark.parametrize(
        "fixture_name",
        ["csv_file", "xlsx_file", "parquet_file", "json_file", "tsv_file"],
    )
    def test_read_data_returns_dataframe(self, fixture_name, request, sample_df):
        path = request.getfixturevalue(fixture_name)
        result = read_data(path)
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, sample_df)

    def test_path_with_uppercase_extension(self, tmp_path, sample_df):
        path = tmp_path / "data.CSV"
        sample_df.to_csv(path, index=False)
        pd.testing.assert_frame_equal(read_data(path), sample_df)

    # --- Error cases ---

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            read_data(tmp_path / "ghost.csv")

    def test_unsupported_format_raises(self, tmp_path):
        path = tmp_path / "data.txt"
        path.touch()
        with pytest.raises(ValueError, match="Unsupported format"):
            read_data(path)

    def test_unsupported_format_case_insensitive(self, tmp_path):
        path = tmp_path / "data.TXT"
        path.touch()
        with pytest.raises(ValueError, match="Unsupported format"):
            read_data(path)

    def test_path_as_string_raises(self, tmp_path):
        with pytest.raises(AttributeError):
            read_data(str(tmp_path / "data.csv"))

    def test_file_is_a_directory(self, tmp_path):
        with pytest.raises(Exception):
            read_data(tmp_path)

    def test_corrupt_csv(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_bytes(b"\xff\xfe garbage not a csv")
        with pytest.raises(Exception):
            read_data(path)

    def test_wrong_content_for_extension(self, tmp_path):
        path = tmp_path / "data.parquet"
        path.write_text("a,b\n1,2\n")
        with pytest.raises(Exception):
            read_data(path)

    # --- Edge cases in valid files ---

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.csv"
        pd.DataFrame(columns=["a", "b"]).to_csv(path, index=False)
        result = read_data(path)
        assert result.empty
        assert list(result.columns) == ["a", "b"]

    def test_single_row(self, tmp_path):
        path = tmp_path / "data.csv"
        pd.DataFrame({"a": [1], "b": [2]}).to_csv(path, index=False)
        assert len(read_data(path)) == 1

    def test_special_characters_in_data(self, tmp_path):
        df = pd.DataFrame({"a": ["hello, world", 'say "hi"', "line\nbreak"]})
        path = tmp_path / "data.csv"
        df.to_csv(path, index=False)
        pd.testing.assert_frame_equal(read_data(path), df)

    # --- Data integrity ---

    def test_column_names_preserved(self, tmp_path):
        df = pd.DataFrame({"col with spaces": [1], "UPPER": [2], "123": [3]})
        path = tmp_path / "data.csv"
        df.to_csv(path, index=False)
        assert list(read_data(path).columns) == list(df.columns)

    def test_dtypes_preserved_parquet(self, tmp_path):
        df = pd.DataFrame(
            {"int_col": pd.array([1, 2], dtype="int32"), "str_col": ["a", "b"]}
        )
        path = tmp_path / "data.parquet"
        df.to_parquet(path, index=False)
        result = read_data(path)
        assert (
            result["int_col"].dtype == pd.Int32Dtype()
            or result["int_col"].dtype == "int32"
        )

    def test_tsv_not_split_on_commas(self, tmp_path):
        df = pd.DataFrame({"a": ["1,2,3"], "b": ["4,5,6"]})
        path = tmp_path / "data.tsv"
        df.to_csv(path, sep="\t", index=False)
        assert read_data(path).shape == (1, 2)
