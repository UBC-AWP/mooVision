
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
 
from scripts.data_reading.schema import schema, processed_schema
 
 
# ===========================================================================
# SHARED HELPER
# ===========================================================================
 
def make_valid_df(n=2, with_labels=False):
    df = pd.DataFrame({
        "clip_name":                    [f"CS_000{i}_clip.mp4" for i in range(n)],
        "clip_relative_path":           [f"Pen 2/clip_{i}.mp4" for i in range(n)],
        "clip_output_path":             [f"output/clip_{i}.mp4" for i in range(n)],
        "source_video_path":            [f"source/video_{i}.mp4" for i in range(n)],
        "source_video_basename":        [f"ch02_video_{i}.mp4" for i in range(n)],
        "export_status":                ["exported"] * n,
        "part_index":                   [0] * n,
        "part_count":                   [1] * n,
        "observation_id":               [f"obs_{i}" for i in range(n)],
        "group_name":                   ["Group 1"] * n,
        "phase":                        ["POSTWEAN"] * n,
        "day":                          [1] * n,
        "pen":                          [2] * n,
        "obs_date_raw":                 [20251102] * n,
        "subject":                      ["cow1"] * n,
        "modifiers":                    ["mod1"] * n,
        "interval_start_obs_sec":       [0.0] * n,
        "interval_end_obs_sec":         [10.0] * n,
        "part_start_obs_sec":           [0.0] * n,
        "part_end_obs_sec":             [10.0] * n,
        "source_segment_obs_start_sec": [0.0] * n,
        "source_segment_obs_end_sec":   [10.0] * n,
        "clip_start_in_source_sec":     [0.0] * n,
        "clip_end_in_source_sec":       [10.0] * n,
    })
    if with_labels:
        df["labelled_clip_relative_path"] = [f"Pen 2/POSTWEAN/Day 1/000{i}.zip" for i in range(n)]
    return df


# ===========================================================================
# TestReadData
# ===========================================================================
 
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


# ===========================================================================
# TestValidateData
# ===========================================================================
 
class TestValidateData:
 
    # --- Fixtures ---
 
    @pytest.fixture
    def valid_df(self):
        return make_valid_df()
 
    @pytest.fixture
    def valid_df_with_labels(self):
        return make_valid_df(with_labels=True)
 
    # --- Happy path ---
 
    def test_valid_df_passes_raw_schema(self, valid_df):
        result = validate_data(valid_df, schema)
        assert isinstance(result, pd.DataFrame)
 
    def test_valid_df_passes_processed_schema(self, valid_df_with_labels):
        result = validate_data(valid_df_with_labels, processed_schema)
        assert isinstance(result, pd.DataFrame)
 
    # --- Error cases ---
 
    def test_not_a_dataframe_raises_type_error(self):
        with pytest.raises(TypeError):
            validate_data("not a df", schema)
 
    def test_not_a_schema_raises_type_error(self, valid_df):
        with pytest.raises(TypeError):
            validate_data(valid_df, "not a schema")
 
    def test_empty_df_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            validate_data(pd.DataFrame(), schema)
 
    def test_missing_required_column_raises(self, valid_df):
        df = valid_df.drop(columns=["clip_name"])
        with pytest.raises(ValueError, match="Data validation failed"):
            validate_data(df, schema)
 
    # --- Edge cases ---
 
    def test_negative_day_raises(self, valid_df):
        valid_df["day"] = [-1, -2]
        with pytest.raises(ValueError, match="Data validation failed"):
            validate_data(valid_df, schema)
 
    def test_end_before_start_raises(self, valid_df):
        valid_df["clip_end_in_source_sec"] = [0.0, 0.0]
        with pytest.raises(ValueError, match="Data validation failed"):
            validate_data(valid_df, schema)
 
    def test_extra_column_raises_strict_schema(self, valid_df):
        valid_df["unexpected_column"] = ["x", "y"]
        with pytest.raises(ValueError, match="Data validation failed"):
            validate_data(valid_df, schema)
 