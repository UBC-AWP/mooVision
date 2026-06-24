"""
Module for testing training_yolo.py
"""

import sys
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import yaml
import tarfile
import shutil

sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.training import training_yolo as ty


class TestSetupNodeDataset:

    @pytest.fixture
    def mock_yolo_structure(self):
        """Constructs a valid internal helper layout for structured directories."""

        def _create_structure(base_path: Path):
            images_dir = base_path / "images"
            labels_dir = base_path / "labels"

            for split in ["train", "val"]:
                (images_dir / split).mkdir(parents=True, exist_ok=True)
                (labels_dir / split).mkdir(parents=True, exist_ok=True)

                # Drop valid matching sample data pairing files
                (images_dir / split / "frame001.jpg").touch()
                (labels_dir / split / "frame001.txt").touch()

            # Create standard template yaml file
            yaml_path = base_path / "dataset.yaml"
            with open(yaml_path, "w") as f:
                yaml.dump({"path": "./placeholder", "train": "images/train"}, f)

        return _create_structure

    #  ---LAPTOP FALLBACK TESTS ---

    def test_laptop_fallback_resolves_locally_without_cluster_vars(
        self, tmp_path, mock_yolo_structure, monkeypatch
    ):
        """Validates that running without Slurm vars correctly tracks ROOT_DIR paths."""
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("PBS_JOBID", raising=False)

        # Setup real local folder targets
        local_dataset_root = tmp_path / "my_project" / "local_dataset"
        local_dataset_root.mkdir(parents=True)
        mock_yolo_structure(local_dataset_root)

        with patch(
            "scripts.training.training_yolo.ROOT_DIR",
            str(tmp_path / "my_project"),
        ):
            yaml_out, on_cluster = ty.setup_node_dataset("local_dataset/dataset.yaml")

            assert on_cluster is False
            assert yaml_out == local_dataset_root / "dataset.yaml"

            # Verify internal mapping update inside configuration
            with open(yaml_out, "r") as f:
                updated_config = yaml.safe_load(f)
            assert updated_config["path"] == str(local_dataset_root.resolve())

    # --- CLUSTER ENVIRONMENT BOUNDARY VALIDATIONS ---

    def test_cluster_raises_value_error_if_slurm_tmpdir_missing(
        self, tmp_path, monkeypatch
    ):
        """Crashes systematically if cluster markers exist but no node storage target exists."""
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        monkeypatch.delenv("SLURM_TMPDIR", raising=False)

        mock_tar = tmp_path / "dataset.tar"
        mock_tar.touch()

        with pytest.raises(ValueError, match="Could not find SLURM_TMPDIR."):
            ty.setup_node_dataset(str(mock_tar))

    def test_cluster_raises_file_not_found_on_missing_input_tarball(
        self, tmp_path, monkeypatch
    ):
        """Crashes early if the input archive string points to an invalid path location."""
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        monkeypatch.setenv("SLURM_TMPDIR", str(tmp_path))

        with pytest.raises(FileNotFoundError, match="Dataset archive missing at"):
            ty.setup_node_dataset(str(tmp_path / "non_existent_archive.tar"))

    # --- EXTRACTION FLOW & INTEGRITY TESTS ---

    @patch("subprocess.run")
    def test_cluster_handles_tar_extraction_failures_gracefully(
        self, mock_run, tmp_path, monkeypatch
    ):
        """Confirms rmtree triggers structural wipes if subprocess calls return errors."""
        monkeypatch.setenv("SLURM_JOB_ID", "9999")
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")
        monkeypatch.setenv("SLURM_TMPDIR", str(tmp_path))

        tar_path = tmp_path / "source.tar"
        tar_path.touch()

        # Mock sub-process exception raise loop profiles
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=2, cmd="tar", stderr=b"Tarball compression corrupt error"
        )

        expected_target_dir = tmp_path / "job_9999_task_1_dataset"

        with pytest.raises(
            RuntimeError, match="Native tar extraction failed for task 1"
        ):
            ty.setup_node_dataset(str(tar_path))

        # Confirm the broken task directory path clean-sweep wiped out
        assert not expected_target_dir.exists()

    @patch("subprocess.run")
    def test_cluster_happy_path_extraction_and_validation(
        self, mock_run, tmp_path, mock_yolo_structure, monkeypatch
    ):
        """Validates standard successful end-to-end routing behavior inside HPC setups."""
        monkeypatch.setenv("SLURM_JOB_ID", "8888")
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "2")
        monkeypatch.setenv("SLURM_TMPDIR", str(tmp_path))

        tar_path = tmp_path / "source.tar"
        tar_path.touch()

        expected_target_dir = tmp_path / "job_8888_task_2_dataset"

        # Side-effect simulation: create the file-structure exactly when tar would unroll it
        def simulate_tar_unroll(*args, **kwargs):
            expected_target_dir.mkdir(parents=True, exist_ok=True)
            mock_yolo_structure(expected_target_dir)
            return MagicMock(returncode=0)

        mock_run.side_effect = simulate_tar_unroll

        yaml_out, on_cluster = ty.setup_node_dataset(str(tar_path))

        assert on_cluster is True
        assert yaml_out == expected_target_dir / "dataset.yaml"
        mock_run.assert_called_once()

        # Check path synchronization changes inside configuration
        with open(yaml_out, "r") as f:
            updated_config = yaml.safe_load(f)
        assert updated_config["path"] == str(expected_target_dir.resolve())

    @patch("subprocess.run")
    def test_cluster_raises_runtime_error_if_extracted_folders_invalid(
        self, mock_run, tmp_path, monkeypatch
    ):
        """Triggers RuntimeError if validation fails to catch valid split headers."""
        monkeypatch.setenv("SLURM_JOB_ID", "7777")
        monkeypatch.setenv("SLURM_TMPDIR", str(tmp_path))

        tar_path = tmp_path / "source.tar"
        tar_path.touch()

        expected_target_dir = tmp_path / "job_7777_task_0_dataset"

        # Simulate extracting garbage file data streams rather than images/labels folders
        def simulate_bad_unroll(*args, **kwargs):
            expected_target_dir.mkdir(parents=True, exist_ok=True)
            (expected_target_dir / "unknown_garbage_folder").mkdir()
            return MagicMock(returncode=0)

        mock_run.side_effect = simulate_bad_unroll

        with pytest.raises(
            RuntimeError, match="Validation Failed: No valid YOLO folders"
        ):
            ty.setup_node_dataset(str(tar_path))

        # Check clean up sweeping verified
        assert not expected_target_dir.exists()


class TestSetupNodeDatasetIntegration:

    @pytest.fixture
    def integration_cluster_sandbox(self, tmp_path):
        """Constructs an unmocked file system sandbox mirroring a real cluster loop.

        Generates a physical .tar archive populated with a valid YOLO split template
        hierarchy and an isolated staging ground directory.
        """
        # Setup absolute paths within our test boundary
        project_root = tmp_path / "project_space"
        slurm_scratch = tmp_path / "localscratch_node_nvme"
        project_root.mkdir()
        slurm_scratch.mkdir()

        source_dataset_dir = project_root / "dataset"
        images_dir = source_dataset_dir / "images" / "train"
        labels_dir = source_dataset_dir / "labels" / "train"

        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        # Add realistic text files to satisfy size validation benchmarks
        (images_dir / "sample_001.jpg").touch()
        (labels_dir / "sample_001.txt").touch()

        # 3. Inject standard template configuration data parameters
        yaml_content = {
            "path": "../placeholder",
            "train": "images/train",
            "val": "images/val",
            "nc": 1,
            "names": ["cross-sucking"],
        }
        with open(source_dataset_dir / "dataset.yaml", "w") as f:
            yaml.dump(yaml_content, f)

        # Pack the hierarchy into a real physical tar file
        tar_destination = project_root / "dataset.tar"
        with tarfile.open(tar_destination, "w") as tar:
            # We add with an arcname of "dataset" to replicate typical pipeline packing outputs
            tar.add(source_dataset_dir, arcname="dataset")

        # Clean the raw folder to ensure our verification code proves successful extraction
        shutil.rmtree(source_dataset_dir)

        return {
            "tar_path": tar_destination,
            "slurm_scratch": slurm_scratch,
            "job_id": "123456",
            "task_id": "3",
        }

    @pytest.mark.integration
    def test_setup_node_dataset_end_to_end_cluster_execution(
        self, integration_cluster_sandbox, monkeypatch
    ):
        """End-to-End Integration: Simulates unmocked cluster environmental runtime triggers,

        verifying physical tar archive decompression and structural dynamic schema locked modifications.
        """
        sandbox = integration_cluster_sandbox

        # Set environment parameters to fool the function into cluster routing tracks
        monkeypatch.setenv("SLURM_JOB_ID", sandbox["job_id"])
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", sandbox["task_id"])
        monkeypatch.setenv("SLURM_TMPDIR", str(sandbox["slurm_scratch"]))

        # Execute the complete unmocked setup logic
        yaml_path, on_cluster = ty.setup_node_dataset(
            dataset=str(sandbox["tar_path"]), base_name="dataset"
        )

        # Verify path environment identification flags evaluated as true
        assert on_cluster is True

        # Verify target isolation workspace directory names resolved properly
        expected_isolated_dirname = (
            f"job_{sandbox['job_id']}_task_{sandbox['task_id']}_dataset"
        )
        expected_node_dir = sandbox["slurm_scratch"] / expected_isolated_dirname

        assert expected_node_dir.exists()
        assert yaml_path == expected_node_dir / "dataset.yaml"

        # Verify the internal files were unrolled natively on disk successfully
        assert (expected_node_dir / "images" / "train" / "sample_001.jpg").exists()
        assert (expected_node_dir / "labels" / "train" / "sample_001.txt").exists()

        # Verify the dynamic absolute path re-routing key modification works cleanly
        with open(yaml_path, "r") as f:
            updated_config = yaml.safe_load(f)

        assert updated_config["path"] == str(expected_node_dir.resolve())
        assert updated_config["nc"] == 1
        assert updated_config["names"] == ["cross-sucking"]
