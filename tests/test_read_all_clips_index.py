
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
 
  
# ===========================================================================
# TestSaveData
# ===========================================================================
 
class TestSaveData:
 
    # --- Fixtures ---
 
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({"a": [1, 2], "b": [3, 4]})
 
    # --- Happy path ---
 
    def test_saves_csv_to_disk(self, tmp_path, sample_df):
        path = tmp_path / "output.csv"
        save_data(sample_df, path)
        assert path.exists()
        pd.testing.assert_frame_equal(pd.read_csv(path), sample_df)
 
    def test_creates_parent_directories(self, tmp_path, sample_df):
        path = tmp_path / "nested" / "deep" / "output.csv"
        save_data(sample_df, path)
        assert path.exists()
 
    # --- Error cases ---
 
    def test_not_a_dataframe_raises_type_error(self, tmp_path):
        with pytest.raises(TypeError):
            save_data("not a df", tmp_path / "output.csv")
 
    def test_empty_df_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            save_data(pd.DataFrame(), tmp_path / "output.csv")
 
    def test_non_csv_extension_raises(self, tmp_path, sample_df):
        with pytest.raises(ValueError, match=".csv"):
            save_data(sample_df, tmp_path / "output.parquet")
 
    # --- Edge cases ---
 
    def test_overwrites_existing_file(self, tmp_path, sample_df):
        path = tmp_path / "output.csv"
        save_data(sample_df, path)
        save_data(pd.DataFrame({"b": [2]}), path)
        result = pd.read_csv(path)
        assert "b" in result.columns

 
# ===========================================================================
# TestFilterExistingClips
# ===========================================================================
 
class TestFilterExistingClips:
 
    # --- Fixtures ---
 
    @pytest.fixture
    def clips_dir(self, tmp_path):
        clips = tmp_path / "clips"
        clips.mkdir()
        (clips / "clip_0.mp4").touch()
        (clips / "clip_1.mp4").touch()
        return clips
 
    @pytest.fixture
    def df_all_exist(self):
        df = make_valid_df(n=2)
        df["clip_relative_path"] = ["clip_0.mp4", "clip_1.mp4"]
        return df
 
    @pytest.fixture
    def df_some_missing(self):
        df = make_valid_df(n=2)
        df["clip_relative_path"] = ["clip_0.mp4", "missing_clip.mp4"]
        return df
 
    # --- Happy path ---
 
    def test_all_clips_exist_returns_full_df(self, df_all_exist, clips_dir):
        result = filter_existing_clips(df_all_exist, clips_dir)
        assert len(result) == 2
 
    def test_missing_clips_filtered_out(self, df_some_missing, clips_dir):
        result = filter_existing_clips(df_some_missing, clips_dir)
        assert len(result) == 1
        assert "clip_0.mp4" in result["clip_relative_path"].values
 
    # --- Error cases ---
 
    def test_not_a_dataframe_raises_type_error(self, clips_dir):
        with pytest.raises(TypeError):
            filter_existing_clips("not a df", clips_dir)
 
    def test_empty_df_raises_value_error(self, clips_dir):
        with pytest.raises(ValueError):
            filter_existing_clips(pd.DataFrame(), clips_dir)
 
    def test_clips_dir_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            filter_existing_clips(make_valid_df(), Path("/nonexistent/path"))
 
    # --- Edge cases ---
 
    def test_all_missing_returns_empty_with_warning(self, clips_dir):
        df = make_valid_df(n=2)
        df["clip_relative_path"] = ["missing_1.mp4", "missing_2.mp4"]
        with pytest.warns(UserWarning):
            result = filter_existing_clips(df, clips_dir)
        assert result.empty
 
    def test_missing_clips_raises_warning(self, df_some_missing, clips_dir):
        with pytest.warns(UserWarning, match="Dropped"):
            filter_existing_clips(df_some_missing, clips_dir)


 
# ===========================================================================
# TestGetLabelPaths
# ===========================================================================
 
class TestGetLabelPaths:
 
    # --- Fixtures ---
 
    @pytest.fixture
    def labels_dir(self, tmp_path):
        labels = tmp_path / "labels"
        (labels / "Pen 2" / "POSTWEAN" / "Day 1").mkdir(parents=True)
        (labels / "Pen 2" / "POSTWEAN" / "Day 1" / "0001.zip").touch()
        (labels / "Pen 2" / "POSTWEAN" / "Day 1" / "0002.zip").touch()
        return labels
 
    # --- Happy path ---
 
    def test_correct_number_of_zips_found(self, labels_dir):
        result = get_label_paths(labels_dir)
        assert len(result) == 2
 
    def test_returns_list_of_tuples_with_name_and_path(self, labels_dir):
        result = get_label_paths(labels_dir)
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) for item in result)
        names = [r[0] for r in result]
        assert "0001.zip" in names
 
    # --- Error cases ---
 
    def test_nonexistent_directory_raises(self):
        with pytest.raises(FileNotFoundError):
            get_label_paths(Path("/nonexistent/path"))
 
    def test_empty_directory_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="No .zip files"):
            get_label_paths(empty)
 
    # --- Edge cases ---
 
    def test_non_zip_files_ignored(self, tmp_path):
        labels = tmp_path / "labels"
        labels.mkdir()
        (labels / "0001.zip").touch()
        (labels / "readme.txt").touch()
        result = get_label_paths(labels)
        assert len(result) == 1
 
 
# ===========================================================================
# TestMatchLabelPaths
# ===========================================================================
 
class TestMatchLabelPaths:
 
    # --- Fixtures ---
 
    @pytest.fixture
    def simple_df(self):
        return make_valid_df(n=2)
 
    # --- Happy path ---
 
    def test_returns_list_of_correct_length(self, simple_df):
        result = match_label_paths(simple_df, [("9999.zip", "path/9999.zip")])
        assert isinstance(result, list)
        assert len(result) == len(simple_df)
 
    # --- Error cases ---
 
    def test_not_a_dataframe_raises_type_error(self):
        with pytest.raises(TypeError):
            match_label_paths("not a df", [("0001.zip", "path/0001.zip")])
 
    def test_empty_df_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            match_label_paths(pd.DataFrame(), [("0001.zip", "path/0001.zip")])
 
    def test_empty_label_paths_raises_value_error(self, simple_df):
        with pytest.raises(ValueError, match="empty"):
            match_label_paths(simple_df, [])
 
    # --- Edge cases ---
 
    def test_no_matches_returns_none_values_with_warning(self, simple_df):
        with pytest.warns(UserWarning):
            result = match_label_paths(simple_df, [("9999.zip", "path/9999.zip")])
        assert all(p is None for p in result)
 
 
# ===========================================================================
# TestAddLabelPaths
# ===========================================================================
 
class TestAddLabelPaths:
 
    # --- Fixtures ---
 
    @pytest.fixture
    def simple_df(self):
        return make_valid_df(n=2)
 
    @pytest.fixture
    def label_paths(self):
        return ["path/0001.zip", "path/0002.zip"]
 
    # --- Happy path ---
 
    def test_adds_column_to_dataframe(self, simple_df, label_paths):
        result = add_label_paths(simple_df, label_paths)
        assert "labelled_clip_relative_path" in result.columns
 
    def test_column_values_correct(self, simple_df, label_paths):
        result = add_label_paths(simple_df, label_paths)
        assert list(result["labelled_clip_relative_path"]) == label_paths
 
    # --- Error cases ---
 
    def test_not_a_dataframe_raises_type_error(self, label_paths):
        with pytest.raises(TypeError):
            add_label_paths("not a df", label_paths)
 
    def test_empty_df_raises_value_error(self, label_paths):
        with pytest.raises(ValueError, match="empty"):
            add_label_paths(pd.DataFrame(), label_paths)
 
    def test_empty_label_paths_raises_value_error(self, simple_df):
        with pytest.raises(ValueError):
            add_label_paths(simple_df, [])
 
    # --- Edge cases ---
 
    def test_none_values_added_correctly(self, simple_df):
        result = add_label_paths(simple_df, ["path/0001.zip", None])
        assert result["labelled_clip_relative_path"].iloc[1] is None
 
 
# ===========================================================================
# TestFilterLabelPaths
# ===========================================================================
 
class TestFilterLabelPaths:
 
    # --- Fixtures ---
 
    @pytest.fixture
    def df_all_labels(self):
        return make_valid_df(n=2, with_labels=True)
 
    @pytest.fixture
    def df_some_none(self):
        df = make_valid_df(n=2, with_labels=True)
        df["labelled_clip_relative_path"] = ["path/0001.zip", None]
        return df
 
    # --- Happy path ---
 
    def test_all_labels_present_returns_full_df(self, df_all_labels):
        result = filter_label_paths(df_all_labels)
        assert len(result) == 2
 
    def test_none_labels_filtered_out(self, df_some_none):
        result = filter_label_paths(df_some_none)
        assert len(result) == 1
        assert result["labelled_clip_relative_path"].iloc[0] == "path/0001.zip"
 
    # --- Error cases ---
 
    def test_not_a_dataframe_raises_type_error(self):
        with pytest.raises(TypeError):
            filter_label_paths("not a df")
 
    def test_empty_df_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            filter_label_paths(pd.DataFrame())
 
    # --- Edge cases ---
 
    def test_all_none_returns_empty_with_warning(self):
        df = make_valid_df(n=2, with_labels=True)
        df["labelled_clip_relative_path"] = [None, None]
        with pytest.warns(UserWarning):
            result = filter_label_paths(df)
        assert result.empty
 
 
 
 
