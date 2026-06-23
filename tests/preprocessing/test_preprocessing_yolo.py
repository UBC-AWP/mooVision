"""
Tests for scripts/preprocessing/preprocessing_yolo.py
"""

import sys
import os
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.preprocessing.preprocessing_yolo import (
    resolve_working_directory,
    process_single_split,
    validate_dataset,
    package_tar_file,
    run_yolo_preprocessing,
)

from scripts.preprocessing import extract_frames as extract_frames_mod


class TestResolveWorkingDirectory:

    # --- TYPE VALIDATION SAFETY CHECKS ---

    def test_type_error_on_invalid_input(self):
        """Verifies that passing anything other than a pathlib.Path raises a TypeError."""
        with pytest.raises(TypeError) as exc_info:
            resolve_working_directory("/not/a/path/object")  # type: ignore

        assert "must be a Path object" in str(exc_info.value)

    # --- LOCAL PLAYGROUND ENVIRONMENT TESTS (Not on a Cluster) ---

    @patch.dict(os.environ, {}, clear=True)
    @patch.object(Path, "mkdir")
    def test_local_dev_fallback(self, mock_mkdir, tmp_path):
        """When running locally with an empty environment, it falls back to output_dir

        and appends 'dataset'.
        """
        fallback_dir = tmp_path / "fallback_output"

        working_dir, base_dir, on_cluster = resolve_working_directory(fallback_dir)

        assert on_cluster is False
        assert base_dir == fallback_dir
        assert working_dir == fallback_dir / "dataset"

        # Verify that directory creation was attempted safely
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch.dict(os.environ, {"SLURM_TMPDIR": "/tmp/custom_local_env"}, clear=True)
    @patch.object(Path, "mkdir")
    def test_local_dev_with_custom_tmpdir(self, mock_mkdir, tmp_path):
        """When not on a cluster, SLURM_TMPDIR is still respected as the base directory

        if present.
        """
        fallback_dir = tmp_path / "fallback_output"

        working_dir, base_dir, on_cluster = resolve_working_directory(fallback_dir)

        assert on_cluster is False
        assert base_dir == Path("/tmp/custom_local_env")
        assert working_dir == Path("/tmp/custom_local_env/dataset")
        mock_mkdir.assert_called_once()

    # --- HPC CLUSTER ENVIRONMENT TESTS (SLURM / PBS) ---

    @pytest.mark.parametrize("cluster_env_var", ["SLURM_JOB_ID", "PBS_JOBID"])
    @patch.object(Path, "mkdir")
    def test_cluster_true_node_local_nvme(
        self, mock_mkdir, cluster_env_var, tmp_path, capsys
    ):
        """Verifies true node-local storage path assembly when running on a cluster

        and checks for success log routing.
        """
        fallback_dir = tmp_path / "fallback"

        # Simulating true node-local scratch storage on an HPC compute partition node
        mock_env = {
            cluster_env_var: "123456",
            "SLURM_TMPDIR": "/localnvme",
            "SLURM_ARRAY_TASK_ID": "42",
        }

        with patch.dict(os.environ, mock_env, clear=True):
            working_dir, base_dir, on_cluster = resolve_working_directory(fallback_dir)

        # Test correct path structures
        assert on_cluster is True
        assert base_dir == Path("/localnvme/data/training/job_array_42_yolo_build")
        assert working_dir == base_dir / "dataset"
        mock_mkdir.assert_called_once()

        # Capture log prints to verify the correct branch executed
        captured = capsys.readouterr()
        assert "Success: True Node-Local NVMe Storage engaged" in captured.out

    @patch.dict(os.environ, {"SLURM_JOB_ID": "999"}, clear=True)
    @patch.object(Path, "mkdir")
    def test_cluster_shared_storage_warning_triggered(
        self, mock_mkdir, tmp_path, capsys
    ):
        """If node_local_storage defaults back to output_dir on a cluster, a warning

        must be issued about shared filesystem performance.
        """
        fallback_dir = tmp_path / "shared_nfs_volume"

        working_dir, base_dir, on_cluster = resolve_working_directory(fallback_dir)

        assert on_cluster is True
        assert "job_array_local_dev_yolo_build" in str(base_dir)

        # Verify warning log message
        captured = capsys.readouterr()
        assert "WARNING: Shared storage detected!" in captured.out

    @patch.dict(
        os.environ,
        {"PBS_JOBID": "888", "SLURM_TMPDIR": "/home/user/scratch_space"},
        clear=True,
    )
    @patch.dict(
        os.environ,
        {"PBS_JOBID": "888", "SLURM_TMPDIR": "/home/user/scratch_space"},
        clear=True,
    )
    @patch.object(Path, "mkdir")
    def test_cluster_scratch_substring_warning(
        self, mock_mkdir, tmp_path, capsys
    ):  # <-- Added tmp_path here
        working_dir, _, _ = resolve_working_directory(tmp_path)

        captured = capsys.readouterr()
        assert "WARNING: Shared storage detected!" in captured.out

    # --- MUTATION PROTECTION (E2E Integration Check) ---

    @patch.dict(os.environ, {}, clear=True)
    def test_physical_directory_generation(self, tmp_path):
        """Integration test verifying that the function physically makes directories

        on disk when not mocked.
        """
        target_base = tmp_path / "sandbox"
        # Ensure it does not exist yet
        assert not (target_base / "dataset").exists()

        working_dir, _, _ = resolve_working_directory(target_base)

        # Ensure directory is physically manifest and accessible on your hardware
        assert working_dir.exists()
        assert working_dir.is_dir()


class TestProcessSingleSplit:

    @pytest.fixture
    def mock_dataframe(self):
        """Generates a dummy pandas DataFrame containing the relative tracking columns."""
        return pd.DataFrame(
            {
                "clip_relative_path": ["vid1.mp4", "vid2.mp4"],
                "labelled_clip_relative_path": ["lbl1.json", "lbl2.json"],
            }
        )

    @pytest.fixture
    def standard_setup(self, mock_dataframe, tmp_path):
        """Provides clean structural input paths and configurations for the pipeline runner."""
        return {
            "df": mock_dataframe,
            "working_dir": tmp_path / "workspace",
            "split": "train",
            "skip": 2,
            "force": False,
            "clips_root_dir": tmp_path / "clips",
            "labels_root_dir": tmp_path / "labels",
        }

    @pytest.fixture
    def valid_args(self, mock_dataframe, tmp_path):
        """Generates valid standard argument dict matching function signatures."""
        return {
            "df": mock_dataframe,
            "working_dir": tmp_path / "workspace",
            "split": "train",
            "skip": 2,
            "force": False,
            "clips_root_dir": tmp_path / "clips",
            "labels_root_dir": tmp_path / "labels",
        }

    # --- ORCHESTRATION PIPELINE DELEGATION TESTS ---

    @patch("scripts.preprocessing.preprocessing_yolo.extract_labels")
    @patch("scripts.preprocessing.preprocessing_yolo.extract_frames")
    def test_pipeline_delegates_parameters_correctly(
        self, mock_frames, mock_labels, standard_setup
    ):
        """Verifies parameter delegation using explicit string path decorators."""
        # 1. Define exactly what the mock frame registry looks like
        fake_frame_registry = {(1, 1): {0, 1, 2}}

        # 2. Assign the return values clearly to the correct mocks
        mock_frames.return_value = fake_frame_registry
        mock_labels.return_value = {}  # Empty dict to satisfy the orphan loop safely

        # 3. Act
        process_single_split(**standard_setup)

        # 4. Assert: Check exactly what extract_frames received
        mock_frames.assert_called_once_with(
            video_paths=["vid1.mp4", "vid2.mp4"],
            videos_root=standard_setup["clips_root_dir"],
            working_dir=standard_setup["working_dir"],
            split=standard_setup["split"],
            skip=standard_setup["skip"],
            force=standard_setup["force"],
        )

        # 5. Assert: Check that extract_labels received the frame registry output from step 1
        mock_labels.assert_called_once_with(
            label_paths=["lbl1.json", "lbl2.json"],
            labels_root=standard_setup["labels_root_dir"],
            working_dir=standard_setup["working_dir"],
            frame_registry=fake_frame_registry,  # <-- This confirms the handoff works!
            split=standard_setup["split"],
            skip=standard_setup["skip"],
            force=standard_setup["force"],
        )

    # --- SANITIZATION AND ORPHAN PURGE LOGIC TESTS ---

    @patch("scripts.preprocessing.preprocessing_yolo.extract_labels")
    @patch("scripts.preprocessing.preprocessing_yolo.extract_frames")
    def test_orphan_purging_deletes_unannotated_frames(
        self, mock_extract_frames, mock_extract_labels, standard_setup, capsys
    ):
        """Tests the set difference logic.

        If a frame index exists in the frame registry but not in the label registry,
        the physical image asset must be deleted.
        """
        # Arrange paths
        working_dir = standard_setup["working_dir"]
        img_dir = working_dir / "images" / "train"
        img_dir.mkdir(parents=True, exist_ok=True)

        # Set up a target video key: (video_id, part_id) -> (1, 1)
        video_key = (1, 1)

        # Scenario: Frames 0, 2, and 4 were extracted. But only 0 and 2 got labeled.
        # Frame 4 is an orphan!
        mock_extract_frames.return_value = {video_key: {0, 2, 4}}
        mock_extract_labels.return_value = {video_key: {0, 2}}

        # Create physical mock images on our virtual test disk
        # Naming rule: f"{int(video_key[0]):04}_part0{video_key[1]}_frame_{frame:06d}.jpg"
        img_0 = img_dir / "0001_part01_frame_000000.jpg"
        img_2 = img_dir / "0001_part01_frame_000002.jpg"
        img_4 = img_dir / "0001_part01_frame_000004.jpg"

        img_0.touch()
        img_2.touch()
        img_4.touch()

        # Act
        process_single_split(**standard_setup)

        # Assert: Matching data must survive
        assert img_0.exists(), "Valid frame 0 was accidentally deleted!"
        assert img_2.exists(), "Valid frame 2 was accidentally deleted!"

        # Assert: Orphaned file must be cleanly purged
        assert not img_4.exists(), "Orphaned frame 4 was not purged from disk!"

        # Check console logs to confirm the tracking count report matched
        captured = capsys.readouterr()
        assert "Dropped 1 unannotated trailing train frames." in captured.out

    @patch("scripts.preprocessing.preprocessing_yolo.extract_labels")
    @patch("scripts.preprocessing.preprocessing_yolo.extract_frames")
    def test_no_action_taken_when_registries_perfectly_aligned(
        self, mock_extract_frames, mock_extract_labels, standard_setup, capsys
    ):
        """If frames and labels match perfectly, no files should be touched or unlinked."""
        img_dir = standard_setup["working_dir"] / "images" / "train"
        img_dir.mkdir(parents=True, exist_ok=True)

        video_key = (2, 3)  # video 2, part 3
        mock_extract_frames.return_value = {video_key: {0, 1}}
        mock_extract_labels.return_value = {video_key: {0, 1}}

        img_0 = img_dir / "0002_part03_frame_000000.jpg"
        img_1 = img_dir / "0002_part03_frame_000001.jpg"
        img_0.touch()
        img_1.touch()

        # Act
        process_single_split(**standard_setup)

        # Assert
        assert img_0.exists()
        assert img_1.exists()

        captured = capsys.readouterr()
        assert "Dropped 0 unannotated trailing train frames." in captured.out

    @patch("scripts.preprocessing.preprocessing_yolo.extract_labels")
    @patch("scripts.preprocessing.preprocessing_yolo.extract_frames")
    def test_orphan_purging_handles_none_part_value(
        self, mock_extract_frames, mock_extract_labels, standard_setup
    ):
        """Verifies naming convention fallback if video_key contains a None part string.

        When part is None, the naming logic changes string evaluation behavior.
        """
        img_dir = standard_setup["working_dir"] / "images" / "train"
        img_dir.mkdir(parents=True, exist_ok=True)

        # Video key with a None part variant: (5, None)
        video_key = (5, None)
        mock_extract_frames.return_value = {video_key: {10}}
        mock_extract_labels.return_value = {video_key: set()}  # Frame 10 is an orphan

        # If video_key[1] is None, part_str evaluates to None.
        # String prefix structure then maps literally to: f"{5:04}_{None}_frame_" -> "0005_None_frame_"
        orphan_img = img_dir / "0005_None_frame_000010.jpg"
        orphan_img.touch()

        # Act
        process_single_split(**standard_setup)

        # Assert
        assert (
            not orphan_img.exists()
        ), "Orphan frame with None part string was not purged!"

    def test_raises_type_error_on_invalid_df(self, valid_args):
        valid_args["df"] = ["not", "a", "dataframe"]
        with pytest.raises(TypeError) as exc:
            process_single_split(**valid_args)
        assert "Argument 'df' must be a pandas DataFrame" in str(exc.value)

    def test_raises_type_error_on_invalid_working_dir(self, valid_args):
        valid_args["working_dir"] = "/string/path/is/bad"
        with pytest.raises(TypeError) as exc:
            process_single_split(**valid_args)
        assert "Argument 'working_dir' must be a pathlib.Path instance" in str(
            exc.value
        )

    def test_raises_type_error_on_invalid_split(self, valid_args):
        valid_args["split"] = 123  # Should be string
        with pytest.raises(TypeError) as exc:
            process_single_split(**valid_args)
        assert "Argument 'split' must be a str" in str(exc.value)

    def test_raises_type_error_on_invalid_skip(self, valid_args):
        valid_args["skip"] = "two"  # Should be int
        with pytest.raises(TypeError) as exc:
            process_single_split(**valid_args)
        assert "Argument 'skip' must be an int" in str(exc.value)

    def test_raises_type_error_on_boolean_skip_edge_case(self, valid_args):
        valid_args["skip"] = (
            True  # bool checks as int in plain isinstance, explicitly blocked
        )
        with pytest.raises(TypeError) as exc:
            process_single_split(**valid_args)
        assert "Argument 'skip' must be an int" in str(exc.value)

    def test_raises_type_error_on_invalid_force(self, valid_args):
        valid_args["force"] = "False"  # String representation instead of actual bool
        with pytest.raises(TypeError) as exc:
            process_single_split(**valid_args)
        assert "Argument 'force' must be a bool" in str(exc.value)

    def test_raises_type_error_on_invalid_clips_root_dir(self, valid_args):
        valid_args["clips_root_dir"] = None
        with pytest.raises(TypeError) as exc:
            process_single_split(**valid_args)
        assert "Argument 'clips_root_dir' must be a pathlib.Path instance" in str(
            exc.value
        )

    def test_raises_type_error_on_invalid_labels_root_dir(self, valid_args):
        valid_args["labels_root_dir"] = {}
        with pytest.raises(TypeError) as exc:
            process_single_split(**valid_args)
        assert "Argument 'labels_root_dir' must be a pathlib.Path instance" in str(
            exc.value
        )

    def test_raises_key_error_on_missing_dataframe_columns(self, valid_args):
        # Dataframe is valid type but missing required structural keys
        valid_args["df"] = pd.DataFrame({"wrong_column_name": [1, 2]})
        with pytest.raises(KeyError) as exc:
            process_single_split(**valid_args)
        assert "missing the required column" in str(exc.value)


class TestValidateDataset:

    @pytest.fixture
    def setup_dirs(self, tmp_path):
        """Helper fixture to construct the base directory structure tree."""
        (tmp_path / "images" / "train").mkdir(parents=True, exist_ok=True)
        (tmp_path / "images" / "val").mkdir(parents=True, exist_ok=True)
        (tmp_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (tmp_path / "labels" / "val").mkdir(parents=True, exist_ok=True)
        return tmp_path

    def test_validate_dataset_success_when_counts_match(self, setup_dirs):
        """Happy Path: Ensuring function completes silently/passes when counts match perfectly."""
        working_dir = setup_dirs

        # Create 2 train pairs
        (working_dir / "images" / "train" / "img1.jpg").touch()
        (working_dir / "images" / "train" / "img2.png").touch()
        (working_dir / "labels" / "train" / "img1.txt").touch()
        (working_dir / "labels" / "train" / "img2.txt").touch()

        # Create 1 validation pair
        (working_dir / "images" / "val" / "img3.jpeg").touch()
        (working_dir / "labels" / "val" / "img3.txt").touch()

        # Act & Assert: This should run without throwing any AssertionError
        validate_dataset(working_dir)

    def test_validate_dataset_raises_on_train_mismatch(self, setup_dirs):
        """Verifies that an AssertionError is thrown if train images != train labels."""
        working_dir = setup_dirs

        # Mismatch: 2 images, 1 label
        (working_dir / "images" / "train" / "img1.jpg").touch()
        (working_dir / "images" / "train" / "img2.jpg").touch()
        (working_dir / "labels" / "train" / "img1.txt").touch()

        with pytest.raises(AssertionError) as exc_info:
            validate_dataset(working_dir)

        assert "Train mismatch detected!" in str(exc_info.value)

    def test_validate_dataset_raises_on_val_mismatch(self, setup_dirs):
        """Verifies that an AssertionError is thrown if val images != val labels."""
        working_dir = setup_dirs

        # Keep train balanced so it passes the first check
        (working_dir / "images" / "train" / "img1.jpg").touch()
        (working_dir / "labels" / "train" / "img1.txt").touch()

        # Mismatch: 1 image, 2 labels
        (working_dir / "images" / "val" / "img2.jpg").touch()
        (working_dir / "labels" / "val" / "img2.txt").touch()
        (working_dir / "labels" / "val" / "img3.txt").touch()

        with pytest.raises(AssertionError) as exc_info:
            validate_dataset(working_dir)

        assert "Validation mismatch detected!" in str(exc_info.value)

    def test_validate_dataset_handles_empty_folders(self, setup_dirs):
        """Verifies that completely empty datasets pass since 0 == 0."""
        working_dir = setup_dirs

        # Act & Assert: 0 files everywhere should not trigger assertions
        validate_dataset(working_dir)

    def test_validate_dataset_ignores_unsupported_extensions(self, setup_dirs):
        """Ensures that validation ignores files that aren't valid images or labels."""
        working_dir = setup_dirs

        # Valid balanced pair
        (working_dir / "images" / "train" / "frame_0.jpg").touch()
        (working_dir / "labels" / "train" / "frame_0.txt").touch()

        # Junk files that should be ignored by glob logic
        (working_dir / "images" / "train" / ".DS_Store").touch()
        (working_dir / "images" / "train" / "annotation_dump.json").touch()
        (working_dir / "labels" / "train" / "backup_labels.bak").touch()

        # Act & Assert: If junk files are ignored correctly, counts remain 1 == 1 and passes
        validate_dataset(working_dir)

    @pytest.mark.parametrize(
        "invalid_input",
        ["/absolute/path/as/string", None, {"path": "/some/dir"}, 12345],
    )
    def test_validate_dataset_raises_type_error_on_invalid_types(self, invalid_input):
        """Verifies a TypeError is thrown if working_dir is not a Path object."""
        with pytest.raises(TypeError) as exc_info:
            validate_dataset(invalid_input)

        assert "Argument 'working_dir' must be a pathlib.Path instance" in str(
            exc_info.value
        )
        assert type(invalid_input).__name__ in str(exc_info.value)


class TestPackageTarFile:

    @pytest.fixture
    def mock_env(self, tmp_path):
        """Builds valid temporary paths to mock local disk behavior."""
        base_local = tmp_path / "slurm_tmp"
        working = base_local / "dataset"
        output = tmp_path / "scratch_storage"

        working.mkdir(parents=True)
        (working / "dummy.txt").touch()

        return {
            "base_local_dir": base_local,
            "working_dir": working,
            "output_dir": output,
        }

    # --- TYPE CHECKING TESTS ---

    def test_package_tar_raises_type_error_on_invalid_base_local_dir(self, mock_env):
        mock_env["base_local_dir"] = "/string/paths/fail"
        with pytest.raises(TypeError) as exc:
            package_tar_file(**mock_env)
        assert "Argument 'base_local_dir' must be a pathlib.Path instance" in str(
            exc.value
        )

    def test_package_tar_raises_type_error_on_invalid_working_dir(self, mock_env):
        mock_env["working_dir"] = None
        with pytest.raises(TypeError) as exc:
            package_tar_file(**mock_env)
        assert "Argument 'working_dir' must be a pathlib.Path instance" in str(
            exc.value
        )

    def test_package_tar_raises_type_error_on_invalid_output_dir(self, mock_env):
        mock_env["output_dir"] = {"path": "invalid"}
        with pytest.raises(TypeError) as exc:
            package_tar_file(**mock_env)
        assert "Argument 'output_dir' must be a pathlib.Path instance" in str(exc.value)

    # --- DISK BOUNDARY EXISTENCE TESTS ---

    def test_package_tar_raises_file_not_found_on_missing_working_dir(self, mock_env):
        # Delete the path fixture created beforehand
        import shutil

        shutil.rmtree(mock_env["working_dir"])

        with pytest.raises(FileNotFoundError) as exc:
            package_tar_file(**mock_env)
        assert "Working dataset directory to archive does not exist" in str(exc.value)

    # --- FUNCTIONAL FUNCTION EXECUTION TEST ---

    def test_package_tar_file_compiles_transfers_and_cleans_up_successfully(
        self, mock_env
    ):
        base_dir = mock_env["base_local_dir"]
        output_dir = mock_env["output_dir"]

        # Run the real packing pipeline inside our sandbox
        package_tar_file(**mock_env)

        # 1. The target scratch directory should have received the compiled dataset archive
        assert (output_dir / "dataset.tar").exists()

        # 2. The temporary compute-node scratch folder must be purged entirely
        assert not base_dir.exists()


class TestRunYoloPreprocessing:

    @pytest.fixture
    def setup_mock_csvs(self, tmp_path):
        """Constructs synthetic temporary dataset files to satisfy existence boundaries."""
        train_csv = tmp_path / "mock_train.csv"
        val_csv = tmp_path / "mock_val.csv"

        # Build minimal index layouts
        df = pd.DataFrame(
            {
                "clip_relative_path": ["v1.mp4"],
                "labelled_clip_relative_path": ["l1.zip"],
            }
        )
        df.to_csv(train_csv)
        df.to_csv(val_csv)

        return {
            "train_path": "mock_train.csv",
            "val_path": "mock_val.csv",
            "output_path": "output_runs",
            "skip": 2,
            "force": False,
        }

    # ==============================================================================
    # RUNTIME TYPE INTERCEPT CHECK TESTS
    # ==============================================================================

    @pytest.mark.parametrize(
        "param_key, invalid_value, expected_msg",
        [
            ("train_path", 12345, "Argument 'train_path' must be a str"),
            ("val_path", Path("val.csv"), "Argument 'val_path' must be a str"),
            ("output_path", None, "Argument 'output_path' must be a str"),
            ("skip", "three", "Argument 'skip' must be an int"),
            (
                "skip",
                True,
                "Argument 'skip' must be an int",
            ),  # Booleans caught explicitly
            ("force", "True", "Argument 'force' must be a bool"),
        ],
    )
    def test_raises_type_errors_on_invalid_argument_signatures(
        self, param_key, invalid_value, expected_msg, setup_mock_csvs
    ):
        args = setup_mock_csvs
        args[param_key] = invalid_value

        # Patch ROOT_DIR string path targeting to isolate execution frame inside testing sandbox
        with patch("scripts.preprocessing.preprocessing_yolo.ROOT_DIR", "."):
            with pytest.raises(TypeError) as exc:
                run_yolo_preprocessing(**args)
            assert expected_msg in str(exc.value)

    # ==============================================================================
    # BOUNDARY EXISTENCE VERIFICATION TESTS
    # ==============================================================================

    def test_raises_file_not_found_when_train_csv_missing(self, setup_mock_csvs):
        args = setup_mock_csvs
        args["train_path"] = "missing_file_index_path.csv"

        with patch(
            "scripts.preprocessing.preprocessing_yolo.ROOT_DIR", str(Path("/tmp"))
        ):
            with pytest.raises(FileNotFoundError) as exc:
                run_yolo_preprocessing(**args)
            assert "Missing required training configuration metadata" in str(exc.value)

    def test_raises_file_not_found_when_val_csv_missing(
        self, setup_mock_csvs, tmp_path
    ):
        args = setup_mock_csvs
        args["val_path"] = "missing_val_index_path.csv"

        with patch("scripts.preprocessing.preprocessing_yolo.ROOT_DIR", str(tmp_path)):
            with pytest.raises(FileNotFoundError) as exc:
                run_yolo_preprocessing(**args)
            assert "Missing required validation configuration metadata" in str(
                exc.value
            )

    # ==============================================================================
    # ORCHESTRATION PIPELINE SEQUENCE FLOW TESTS (MOCKED)
    # ==============================================================================

    @patch("scripts.preprocessing.preprocessing_yolo.package_tar_file")
    @patch("scripts.preprocessing.preprocessing_yolo.validate_dataset")
    @patch("scripts.preprocessing.preprocessing_yolo.create_yaml")
    @patch("scripts.preprocessing.preprocessing_yolo.process_single_split")
    @patch("scripts.preprocessing.preprocessing_yolo.resolve_working_directory")
    def test_orchestrator_coordinates_pipeline_correctly_on_local(
        self,
        mock_resolve,
        mock_process,
        mock_yaml,
        mock_validate,
        mock_tar,
        setup_mock_csvs,
        tmp_path,
    ):
        """Verifies full execution mapping tree behaviors when operating locally."""
        args = setup_mock_csvs

        # Route mock configurations
        mock_resolve.return_value = (
            tmp_path / "dataset",
            tmp_path,
            False,
        )  # on_cluster = False

        with patch("scripts.preprocessing.preprocessing_yolo.ROOT_DIR", str(tmp_path)):
            run_yolo_preprocessing(**args)

        # Confirm working directory mapped accurately
        mock_resolve.assert_called_once_with(output_dir=tmp_path / args["output_path"])

        # Confirm process_single_split was called exactly twice (Train and Val)
        assert mock_process.call_count == 2

        # Confirm tracking cleanup loops and configurations fired in order
        mock_yaml.assert_called_once_with(str(tmp_path / "dataset"))
        mock_validate.assert_called_once_with(tmp_path / "dataset")

        # Local runner must skip compiling structural tar files
        mock_tar.assert_not_called()

    @patch("scripts.preprocessing.preprocessing_yolo.package_tar_file")
    @patch("scripts.preprocessing.preprocessing_yolo.validate_dataset")
    @patch("scripts.preprocessing.preprocessing_yolo.create_yaml")
    @patch("scripts.preprocessing.preprocessing_yolo.process_single_split")
    @patch("scripts.preprocessing.preprocessing_yolo.resolve_working_directory")
    def test_orchestrator_triggers_tarball_packaging_on_hpc_cluster(
        self,
        mock_resolve,
        mock_process,
        mock_yaml,
        mock_validate,
        mock_tar,
        setup_mock_csvs,
        tmp_path,
    ):
        """Verifies that archive extraction triggers whenever running inside cluster loops."""
        args = setup_mock_csvs
        working_dir = tmp_path / "dataset"
        base_local = tmp_path / "base"

        mock_resolve.return_value = (working_dir, base_local, True)  # on_cluster = True

        with patch("scripts.preprocessing.preprocessing_yolo.ROOT_DIR", str(tmp_path)):
            run_yolo_preprocessing(**args)

        # Archive packaging tracking verification check
        mock_tar.assert_called_once_with(
            base_local, working_dir, tmp_path / args["output_path"]
        )
