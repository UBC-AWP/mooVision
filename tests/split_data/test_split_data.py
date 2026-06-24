"""
Tests for scripts/split_data/split_data
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import patch
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.split_data.split_data import (
    train_test_to_csv,
    random_shuffle_split,
    time_based_split,
    pen_based_split,
    period_based_split,
)


class TestTrainTestToCsv:

    @pytest.fixture
    def valid_dfs(self):
        """Generates small, population-validated DataFrames for splitting tests."""
        return {
            "train": pd.DataFrame({"col": [1, 2]}),
            "val": pd.DataFrame({"col": [3, 4]}),
            "test": pd.DataFrame({"col": [5, 6]}),
        }

    # --- BOUNDARY VALIDATION TESTS (EMPTY DATAFRAMES) ---

    def test_raises_value_error_if_train_is_empty(self, valid_dfs, tmp_path):
        with pytest.raises(ValueError, match="Training set is an empty df."):
            train_test_to_csv(
                train=pd.DataFrame(),  # Empty
                val=valid_dfs["val"],
                test=valid_dfs["test"],
                output_dir=tmp_path,
            )

    def test_raises_value_error_if_val_is_empty(self, valid_dfs, tmp_path):
        with pytest.raises(ValueError, match="Validation set is an empty df."):
            train_test_to_csv(
                train=valid_dfs["train"],
                val=pd.DataFrame(),  # Empty
                test=valid_dfs["test"],
                output_dir=tmp_path,
            )

    def test_raises_value_error_if_test_is_empty(self, valid_dfs, tmp_path):
        with pytest.raises(ValueError, match="Testing set is an empty df."):
            train_test_to_csv(
                train=valid_dfs["train"],
                val=valid_dfs["val"],
                test=pd.DataFrame(),  # Empty
                output_dir=tmp_path,
            )

    # --- DISK OPERATIONS & OVERWRITE LOGIC TESTS ---

    def test_saves_all_csv_files_and_creates_directories_successfully(
        self, valid_dfs, tmp_path
    ):
        """Happy Path: Verifies that paths are created dynamically and CSVs match data."""
        # Define a deep nested directory path that does not exist yet
        nested_output_dir = tmp_path / "nested" / "output" / "dir"

        train_test_to_csv(
            train=valid_dfs["train"],
            val=valid_dfs["val"],
            test=valid_dfs["test"],
            output_dir=nested_output_dir,
            force=False,
        )

        # Check files physically materialized on disk
        train_file = nested_output_dir / "train.csv"
        val_file = nested_output_dir / "val.csv"
        test_file = nested_output_dir / "test.csv"

        assert train_file.exists()
        assert val_file.exists()
        assert test_file.exists()

        # Reload CSV entries via pandas to confirm integrity of data streams
        loaded_train = pd.read_csv(train_file)
        assert loaded_train.shape == (2, 1)
        assert list(loaded_train["col"]) == [1, 2]

    def test_skips_overwriting_existing_files_when_force_is_false(
        self, valid_dfs, tmp_path, capsys
    ):
        """Verifies that if pre-existing files exist, they remain intact if force=False."""
        train_file = tmp_path / "train.csv"
        val_file = tmp_path / "val.csv"
        test_file = tmp_path / "test.csv"

        # Touch/create physical pristine state files manually first
        train_file.touch()
        val_file.touch()
        test_file.touch()

        # Pass alternative data payloads with force disabled
        train_test_to_csv(
            train=valid_dfs["train"],
            val=valid_dfs["val"],
            test=valid_dfs["test"],
            output_dir=tmp_path,
            force=False,
        )

        # Check the files must be completely empty (not overridden by the valid_dfs content)
        assert train_file.stat().st_size == 0
        assert val_file.stat().st_size == 0
        assert test_file.stat().st_size == 0

        # Capture output streams to confirm standard console alerting behavior
        captured = capsys.readouterr()
        assert f"{train_file} already exists." in captured.out
        assert f"{val_file} already exists." in captured.out

    def test_overwrites_existing_files_when_force_is_true(
        self, valid_dfs, tmp_path, capsys
    ):
        """Verifies that old file configurations are completely wiped and rewritten if FORCE=True."""
        train_file = tmp_path / "train.csv"
        tmp_path.mkdir(parents=True, exist_ok=True)
        train_file.write_text("old junk header structural data string configurations")

        # Pass payload with FORCE enabled
        train_test_to_csv(
            train=valid_dfs["train"],
            val=valid_dfs["val"],
            test=valid_dfs["test"],
            output_dir=tmp_path,
            force=True,
        )
        # Check the old text content is gone, and can now be read correctly by pandas
        loaded_train = pd.read_csv(train_file)
        assert list(loaded_train["col"]) == [1, 2]

        captured = capsys.readouterr()
        assert f"Training set saved to {train_file}" in captured.out


class TestRandomShuffleSplit:

    @pytest.fixture
    def mock_csv_data(self, tmp_path):
        """Creates a realistic dummy dataset file where multiple clips share source videos."""
        input_csv = tmp_path / "processed_clips_index.csv"

        # 10 records spread across 4 distinct source videos
        data = {
            "clip_id": list(range(10)),
            "source_video_basename": [
                "vid_A",
                "vid_A",
                "vid_A",  # 3 clips
                "vid_B",
                "vid_B",  # 2 clips
                "vid_C",
                "vid_C",
                "vid_C",  # 3 clips
                "vid_D",
                "vid_D",  # 2 clips
            ],
            "feature": [0.1] * 10,
        }
        df = pd.DataFrame(data)
        df.to_csv(input_csv, index=False)
        return input_csv

    # --- BEHAVIORAL CONTROL FLAGS TESTS (EXECUTE & FORCE) ---

    def test_returns_immediately_if_execute_is_false(self, tmp_path):
        """Verifies function exits without looking at files or paths if exectute=False."""
        # This should run fine even though input_path does not exist
        random_shuffle_split(
            input_path=tmp_path / "non_existent.csv",
            output_dir=tmp_path,
            force=False,
            exectute=False,
        )
        assert not (tmp_path / "random").exists()

    @patch("scripts.split_data.split_data.train_test_to_csv")
    def test_exits_early_if_random_dir_exists_and_force_is_false(
        self, mock_save, mock_csv_data, tmp_path, capsys
    ):
        """Ensures logic stops and prints warning when destination directory exists without FORCE."""
        random_dir = tmp_path / "random"
        random_dir.mkdir(parents=True, exist_ok=True)

        random_shuffle_split(
            input_path=mock_csv_data, output_dir=tmp_path, force=False, exectute=True
        )

        # Check no parsing or downstream saving happened
        mock_save.assert_not_called()
        captured = capsys.readouterr()
        assert f"{random_dir} already exists." in captured.out

    # FILE SYSTEM EXCEPTION BOUNDARY TESTS

    def test_raises_file_not_found_on_missing_input_csv(self, tmp_path):
        """Ensures FileNotFoundError is triggered before computing split math."""
        missing_csv = tmp_path / "missing_index.csv"

        with pytest.raises(FileNotFoundError) as exc_info:
            random_shuffle_split(
                input_path=missing_csv, output_dir=tmp_path, force=True, exectute=True
            )
        assert "does not exist." in str(exc_info.value)

    # --- FUNCTIONAL SPLITTING & DATA SEPARATION LOGIC TESTS ---

    @patch("scripts.split_data.split_data.train_test_to_csv")
    def test_splits_data_correctly_without_mixing_source_videos(
        self, mock_save, mock_csv_data, tmp_path
    ):
        """Verifies core requirement: all sub-clips from a unique video stay in the same split."""

        random_shuffle_split(
            input_path=mock_csv_data, output_dir=tmp_path, force=True, exectute=True
        )

        # Check target target folder should successfully instantiate on disk
        assert (tmp_path / "random").exists()

        # Extract arguments sent to train_test_to_csv(train, val, test, target_dir, FORCE)
        mock_save.assert_called_once()
        args, kwargs = mock_save.call_args

        train_df, val_df, test_df, target_dir, force_flag = args

        # Ensure directory destination parameter maps out precisely
        assert target_dir == tmp_path / "random"
        assert force_flag is True

        # Check video integrity: Sets of unique videos per split must be mutually exclusive
        train_vids = set(train_df["source_video_basename"])
        val_vids = set(val_df["source_video_basename"])
        test_vids = set(test_df["source_video_basename"])

        assert train_vids.isdisjoint(val_vids)
        assert train_vids.isdisjoint(test_vids)
        assert val_vids.isdisjoint(test_vids)

        # Total row compilation checks (Sum of split rows must perfectly match input)
        total_rows_processed = len(train_df) + len(val_df) + len(test_df)
        assert total_rows_processed == 10


class TestTimeBasedSplit:

    @pytest.fixture
    def mock_time_csv(self, tmp_path):
        """Creates a realistic dummy dataset with day indices and observation window seconds."""
        input_csv = tmp_path / "time_clips_index.csv"

        data = {
            "clip_id": [1, 2, 3, 4, 5, 6],
            "day": [1, 2, 3, 3, 3, 4],
            # Time boundary markers targeting: 8640 and 60480
            "part_start_obs_sec": [100, 200, 9000, 5000, 65000, 500],
            "part_end_obs_sec": [500, 600, 15000, 62000, 70000, 900],
        }
        df = pd.DataFrame(data)
        df.to_csv(input_csv, index=False)
        return input_csv

    # --- BEHAVIORAL CONTROL FLAGS TESTS (EXECUTE & FORCE) ---

    def test_returns_immediately_if_execute_is_false(self, tmp_path):
        """Verifies function exits safely without evaluating filesystem if exectute=False."""
        time_based_split(
            input_path=tmp_path / "non_existent.csv",
            output_dir=tmp_path,
            force=False,
            exectute=False,
        )
        assert not (tmp_path / "day_based").exists()

    @patch("scripts.split_data.split_data.train_test_to_csv")
    def test_exits_early_if_day_based_dir_exists_and_force_is_false(
        self, mock_save, mock_time_csv, tmp_path, capsys
    ):
        """Ensures logic stops and prints warning when target directory exists without FORCE."""
        target_dir = tmp_path / "day_based"
        target_dir.mkdir(parents=True, exist_ok=True)

        time_based_split(
            input_path=mock_time_csv, output_dir=tmp_path, force=False, exectute=True
        )

        mock_save.assert_not_called()
        captured = capsys.readouterr()
        assert f"{target_dir} already exists." in captured.out

    # --- SCHEMA & VALUE VALIDATION BOUNDARY TESTS ---

    def test_raises_file_not_found_on_missing_input_csv(self, tmp_path):
        """Ensures FileNotFoundError is triggered before column validations."""
        missing_csv = tmp_path / "missing_index.csv"

        with pytest.raises(FileNotFoundError) as exc_info:
            time_based_split(
                input_path=missing_csv, output_dir=tmp_path, force=True, exectute=True
            )
        assert "does not exist." in str(exc_info.value)

    def test_raises_value_error_if_day_column_is_missing(self, tmp_path):
        """Verifies schema validation intercepts DataFrames missing the 'day' key."""
        bad_csv = tmp_path / "bad_schema.csv"
        pd.DataFrame({"clip_id": [1], "wrong_column": [2]}).to_csv(bad_csv, index=False)

        with pytest.raises(
            ValueError, match="Specified column 'day' not found in data index."
        ):
            time_based_split(
                input_path=bad_csv, output_dir=tmp_path, force=True, exectute=True
            )

    # --- TEMPORAL CHRONOLOGICAL SPLITTING TESTS ---

    @patch("scripts.split_data.split_data.train_test_to_csv")
    def test_allocates_subsets_correctly_based_on_time_windows(
        self, mock_save, mock_time_csv, tmp_path
    ):
        """Verifies mathematical temporal allocation conditions across Day 1, 2, 3, and 4."""

        time_based_split(
            input_path=mock_time_csv, output_dir=tmp_path, force=True, exectute=True
        )

        # Target directory should successfully instantiate on disk
        assert (tmp_path / "day_based").exists()

        # Extract DataFrames passed to train_test_to_csv
        mock_save.assert_called_once()
        args, _ = mock_save.call_args
        train_df, val_df, test_df, target_dir, force_flag = args

        assert target_dir == tmp_path / "day_based"
        assert force_flag is True

        # Check Train Allocation (Everything except Day 3)
        # IDs 1 (Day 1), 2 (Day 2), and 6 (Day 4) should stay in train
        assert set(train_df["clip_id"]) == {1, 2, 6}

        # Check Validation Allocation (Day 3 inside 8640 and 60480 window)
        # ID 3: Day 3, start=9000, end=15000 -> Valid (inside window)
        # ID 4: Day 3, start=5000, end=62000 -> Out (end is > 60480)
        assert set(val_df["clip_id"]) == {3}

        # Check Testing Allocation (Day 3 after or at 60480 window)
        # ID 5: Day 3, start=65000 -> Valid (>= 60480)
        assert set(test_df["clip_id"]) == {5}


class TestPenBasedSplit:

    @pytest.fixture
    def mock_pen_csv(self, tmp_path):
        """Creates a realistic dummy dataset with 4 distinct pens (10, 20, 30, 40)."""
        input_csv = tmp_path / "pen_clips_index.csv"
        data = {
            "clip_id": [1, 2, 3, 4, 5],
            "pen": [10, 20, 30, 40, 10],
            "value": [0.5] * 5,
        }
        df = pd.DataFrame(data)
        df.to_csv(input_csv, index=False)
        return input_csv

    # --- BEHAVIORAL CONTROL FLAGS TESTS (EXECUTE & FORCE) ---

    def test_returns_immediately_if_execute_is_false(self, tmp_path):
        """Verifies function exits safely without evaluating files if exectute=False."""
        pen_based_split(
            input_path=tmp_path / "non_existent.csv",
            output_dir=tmp_path,
            force=False,
            exectute=False,
        )
        assert not (tmp_path / "pen_based").exists()

    @patch("scripts.split_data.split_data.train_test_to_csv")
    def test_exits_early_if_pen_based_dir_exists_and_force_is_false(
        self, mock_save, mock_pen_csv, tmp_path, capsys
    ):
        """Ensures logic stops and prints warning when target directory exists without FORCE."""
        target_dir = tmp_path / "pen_based"
        target_dir.mkdir(parents=True, exist_ok=True)

        pen_based_split(
            input_path=mock_pen_csv, output_dir=tmp_path, force=False, exectute=True
        )

        mock_save.assert_not_called()
        captured = capsys.readouterr()
        assert f"{target_dir} already exists." in captured.out

    # --- SCHEMA & PEN POPULATION BOUNDARY TESTS ---

    def test_raises_file_not_found_on_missing_input_csv(self, tmp_path):
        """Ensures FileNotFoundError is triggered before validating dataframe parameters."""
        missing_csv = tmp_path / "missing_index.csv"

        with pytest.raises(FileNotFoundError) as exc_info:
            pen_based_split(
                input_path=missing_csv, output_dir=tmp_path, force=True, exectute=True
            )
        assert "does not exist." in str(exc_info.value)

    def test_raises_value_error_if_pen_column_is_missing(self, tmp_path):
        """Verifies validation intercepts dataframes missing the tracking 'pen' column key."""
        bad_csv = tmp_path / "bad_schema.csv"
        pd.DataFrame({"clip_id": [1], "wrong_column": [2]}).to_csv(bad_csv, index=False)

        with pytest.raises(
            ValueError, match="Specified column 'pen' not found in data index."
        ):
            pen_based_split(
                input_path=bad_csv, output_dir=tmp_path, force=True, exectute=True
            )

    def test_raises_value_error_if_unique_pens_count_under_three(self, tmp_path):
        """Ensures function crashes if there aren't enough pens to rotate splits safely."""
        insufficient_csv = tmp_path / "insufficient_pens.csv"
        # Only 2 unique pens: 10 and 20
        pd.DataFrame({"clip_id": [1, 2], "pen": [10, 20]}).to_csv(
            insufficient_csv, index=False
        )

        with pytest.raises(
            ValueError, match="Critical Error: You only have 2 pen\\(s\\)."
        ):
            pen_based_split(
                input_path=insufficient_csv,
                output_dir=tmp_path,
                force=True,
                exectute=True,
            )

    # --- ROTATIONAL ROUND-ROBIN LEAVE-ONE-OUT TESTS ---

    @patch("scripts.split_data.split_data.train_test_to_csv")
    def test_executes_sliding_cross_validation_rotations_correctly(
        self, mock_save, mock_pen_csv, tmp_path
    ):
        """Verifies the ring-modulo math isolates correct subfolders and assignments."""

        pen_based_split(
            input_path=mock_pen_csv, output_dir=tmp_path, force=True, exectute=True
        )

        assert (tmp_path / "pen_based").exists()

        # Unique pens are [10, 20, 30, 40] -> total_pens = 4.
        # mock_save should be called exactly 4 times (once per test pen rotation).
        assert mock_save.call_count == 4

        # Iterate through the calls chronologically to verify sliding assignments
        # Profile 0: Test=10, Val=20, Train=[30, 40]
        args_0, _ = mock_save.call_args_list[0]
        assert args_0[3] == tmp_path / "pen_based" / "pen_10"
        assert set(args_0[0]["pen"]) == {30, 40}  # Train
        assert set(args_0[1]["pen"]) == {20}  # Val
        assert set(args_0[2]["pen"]) == {10}  # Test

        # Profile 1: Test=20, Val=30, Train=[10, 40]
        args_1, _ = mock_save.call_args_list[1]
        assert args_1[3] == tmp_path / "pen_based" / "pen_20"
        assert set(args_1[0]["pen"]) == {10, 40}
        assert set(args_1[1]["pen"]) == {30}
        assert set(args_1[2]["pen"]) == {20}

        # Profile 2: Test=30, Val=40, Train=[10, 20]
        args_2, _ = mock_save.call_args_list[2]
        assert args_2[3] == tmp_path / "pen_based" / "pen_30"
        assert set(args_2[0]["pen"]) == {10, 20}
        assert set(args_2[1]["pen"]) == {40}
        assert set(args_2[2]["pen"]) == {30}

        # Profile 3: Test=40, Val=10 (Wraps via modulo!), Train=[20, 30]
        args_3, _ = mock_save.call_args_list[3]
        assert args_3[3] == tmp_path / "pen_based" / "pen_40"
        assert set(args_3[0]["pen"]) == {20, 30}
        assert set(args_3[1]["pen"]) == {10}
        assert set(args_3[2]["pen"]) == {40}


class TestPeriodBasedSplit:

    @pytest.fixture
    def mock_period_csv(self, tmp_path):
        """Creates a realistic dummy dataset with 3 production developmental phases."""
        input_csv = tmp_path / "period_clips_index.csv"
        data = {
            "clip_id": [1, 2, 3, 4, 5],
            "phase": ["PREWEANING", "WEANING", "POSTWEANING", "WEANING", "PREWEANING"],
            "score": [0.9] * 5,
        }
        df = pd.DataFrame(data)
        df.to_csv(input_csv, index=False)
        return input_csv

    # --- BEHAVIORAL CONTROL FLAGS TESTS (EXECUTE & FORCE) ---

    def test_returns_immediately_if_execute_is_false(self, tmp_path):
        """Verifies function exits safely without evaluating filesystem if exectute=False."""
        period_based_split(
            input_path=tmp_path / "non_existent.csv",
            output_dir=tmp_path,
            force=False,
            exectute=False,
        )
        assert not (tmp_path / "period_based").exists()

    @patch("scripts.split_data.split_data.train_test_to_csv")
    def test_exits_early_if_period_based_dir_exists_and_force_is_false(
        self, mock_save, mock_period_csv, tmp_path, capsys
    ):
        """Ensures logic stops and prints warning when target directory exists without FORCE."""
        target_dir = tmp_path / "period_based"
        target_dir.mkdir(parents=True, exist_ok=True)

        period_based_split(
            input_path=mock_period_csv, output_dir=tmp_path, force=False, exectute=True
        )

        mock_save.assert_not_called()
        captured = capsys.readouterr()
        assert f"{target_dir} already exists." in captured.out

    # --- SCHEMA & FILE BOUNDARY TESTS ---

    def test_raises_file_not_found_on_missing_input_csv(self, tmp_path):
        """Ensures FileNotFoundError is triggered before computing split math."""
        missing_csv = tmp_path / "missing_index.csv"

        with pytest.raises(FileNotFoundError) as exc_info:
            period_based_split(
                input_path=missing_csv, output_dir=tmp_path, force=True, exectute=True
            )
        assert "does not exist." in str(exc_info.value)

    def test_raises_key_error_or_value_error_if_phase_column_is_missing(self, tmp_path):
        """Verifies validation catches missing tracking 'phase' columns."""
        bad_csv = tmp_path / "bad_schema.csv"
        pd.DataFrame({"clip_id": [1], "wrong_column": ["WEANING"]}).to_csv(
            bad_csv, index=False
        )

        # The current implementation has pandas raise a KeyError when indexing df["phase"]
        with pytest.raises(KeyError):
            period_based_split(
                input_path=bad_csv, output_dir=tmp_path, force=True, exectute=True
            )

    # --- CHRONOLOGICAL ROUND-ROBIN LEAVE-ONE-OUT PHASE TESTS ---

    @patch("scripts.split_data.split_data.train_test_to_csv")
    def test_executes_sliding_cross_validation_rotations_correctly(
        self, mock_save, mock_period_csv, tmp_path
    ):
        """Verifies the ring-modulo math maps correct subsets to alphabetical phases."""

        period_based_split(
            input_path=mock_period_csv, output_dir=tmp_path, force=True, exectute=True
        )

        assert (tmp_path / "period_based").exists()

        # Unique sorted phases: ['POSTWEANING', 'PREWEANING', 'WEANING']
        # mock_save should be called exactly 3 times.
        assert mock_save.call_count == 3

        # Profile 0: Test='POSTWEANING', Val='PREWEANING', Train=['WEANING']
        args_0, _ = mock_save.call_args_list[0]
        assert args_0[3] == tmp_path / "period_based" / "POSTWEANING"
        assert set(args_0[0]["phase"]) == {"WEANING"}  # Train
        assert set(args_0[1]["phase"]) == {"PREWEANING"}  # Val
        assert set(args_0[2]["phase"]) == {"POSTWEANING"}  # Test

        # Profile 1: Test='PREWEANING', Val='WEANING', Train=['POSTWEANING']
        args_1, _ = mock_save.call_args_list[1]
        assert args_1[3] == tmp_path / "period_based" / "PREWEANING"
        assert set(args_1[0]["phase"]) == {"POSTWEANING"}
        assert set(args_1[1]["phase"]) == {"WEANING"}
        assert set(args_1[2]["phase"]) == {"PREWEANING"}

        # Profile 2: Test='WEANING', Val='POSTWEANING' (Wraps via modulo), Train=['PREWEANING']
        args_2, _ = mock_save.call_args_list[2]
        assert args_2[3] == tmp_path / "period_based" / "WEANING"
        assert set(args_2[0]["phase"]) == {"PREWEANING"}
        assert set(args_2[1]["phase"]) == {"POSTWEANING"}
        assert set(args_2[2]["phase"]) == {"WEANING"}
