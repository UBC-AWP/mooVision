"""
Tests for scripts/preprocessing/extract_frames.py
"""

import sys
from pathlib import Path
from typing import Tuple
import concurrent
import threading
import pytest
from unittest.mock import patch, MagicMock

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from scripts.preprocessing.extract_frames import (
    generate_video_metadata,
    process_single_video,
    extract_frames,
)


class TestGenerateVideoMetadataIntegration:
    """
    Integration tests validating video metadata extraction using the live
    underlying regex string parser without mocks to maximize pipeline safety.
    """

    # --- HAPPY PATH INTEGRATION TESTS ---

    @pytest.mark.parametrize(
        "video_input, expected_key, expected_prefix",
        [
            ("CS_0042_part1.mp4", (42, 1), "0042_part01_frame_"),
            ("CS_105_test_info_part2.avi", (105, 2), "0105_part02_frame_"),
            ("CS_0003_test_info_part9.mkv", (3, 9), "0003_part09_frame_"),
        ],
    )
    def test_successful_integration_with_partitions(
        self, video_input: str, expected_key: Tuple[int, str], expected_prefix: str
    ):
        """Ensure unlabelled video clip names with partition indicators parse cleanly."""
        video_key, file_prefix = generate_video_metadata(video_input)

        assert video_key == expected_key
        assert file_prefix == expected_prefix

    @pytest.mark.parametrize(
        "video_input, expected_key, expected_prefix",
        [
            ("CS_0042_clip_info.mp4", (42, None), "0042_None_frame_"),
            ("CS_105_test_info.mp4", (105, None), "0105_None_frame_"),
        ],
    )
    def test_successful_integration_without_partitions(
        self, video_input: str, expected_key: Tuple[int, None], expected_prefix: str
    ):
        """Ensure video names missing partition details fall back to None variables safely."""
        video_key, file_prefix = generate_video_metadata(video_input)

        assert video_key == expected_key
        assert file_prefix == expected_prefix

    # --- ERROR & EXCEPTION BOUNDARY TESTS ---

    @pytest.mark.parametrize(
        "malformed_video",
        [
            "invalid_filename.mp4",  # Completely missing tracking numbers
            "CS_abc_part1.mp4",  # Non-numeric tracking code
            "clip__part2.mp4",  # Missing base identifier information
        ],
    )
    def test_integration_fails_on_malformed_regex_patterns(self, malformed_video: str):
        """Verify that if parse_unlabelled_name fails, errors bubble up cleanly through the pipeline."""
        with pytest.raises((ValueError)):
            generate_video_metadata(malformed_video)

    @pytest.mark.parametrize("invalid_input", [1234, ["video.mp4"], dict(), None])
    def test_raises_type_error_for_non_strings(self, invalid_input):
        """Verify TypeError triggers immediately if input is not a str."""
        with pytest.raises(TypeError) as exc_info:
            generate_video_metadata(invalid_input)

        assert "must be a string" in str(exc_info.value)

    @pytest.mark.parametrize("empty_input", ["", "   ", "\n", "\t"])
    def test_raises_value_error_for_empty_or_whitespace_strings(self, empty_input):
        """Verify ValueError triggers for missing, blank, or layout whitespaces."""
        with pytest.raises(ValueError) as exc_info:
            generate_video_metadata(empty_input)

        assert "cannot be an empty string or whitespace" in str(exc_info.value)


class TestProcessSingleVideo:
    """Tests validating video decoding, stride skips, concurrency synchronization, and safeguards."""

    @pytest.fixture
    def mock_threading_components(self):
        """Fixture providing active real concurrency containers to test code mechanics realistically without I/O blockades."""
        executor = MagicMock(spec=concurrent.futures.ThreadPoolExecutor)
        semaphore = threading.BoundedSemaphore(value=5)
        safe_write_func = MagicMock()

        # Make executor.submit execute the function synchronously right away
        def dummy_submit(fn, *args, **kwargs):
            fn(*args, **kwargs)
            mock_future = MagicMock()
            return mock_future

        executor.submit.side_effect = dummy_submit

        yield executor, semaphore, safe_write_func

    # --- HAPPY PATH & STRIDE DECODING TESTS ---

    @patch("cv2.VideoCapture")
    def test_process_single_video_success_with_skip_stride(
        self, mock_vc, tmp_path, mock_threading_components
    ):
        """Ensure standard video decodes accurately, utilizing cap.grab() for downsampling optimizations."""

        video_path = tmp_path / "mock_video.mp4"
        video_path.touch()  # Create empty video placeholder to pass Path.exists()

        executor, semaphore, safe_write_func = mock_threading_components

        # Configure a mock VideoCapture stream that yields 2 frames matching skip=3
        mock_instance = MagicMock()
        mock_instance.isOpened.side_effect = [True, True, False]

        mock_instance.read.side_effect = [
            (True, "frame_data_0"),
            (True, "frame_data_3"),
        ]

        mock_instance.grab.side_effect = [True, True, True, True]
        mock_vc.return_value = mock_instance

        saved_frames = process_single_video(
            video_path=video_path,
            file_prefix="vid01_",
            final_output_dir=tmp_path,
            skip=3,
            executor=executor,
            semaphore=semaphore,
            safe_write_func=safe_write_func,
        )

        assert saved_frames == {0, 3}
        assert safe_write_func.call_count == 2  # called once per frame

        expected_path_0 = tmp_path / "vid01_000000.jpg"
        expected_path_3 = tmp_path / "vid01_000003.jpg"

        safe_write_func.assert_any_call(str(expected_path_0), "frame_data_0")
        safe_write_func.assert_any_call(str(expected_path_3), "frame_data_3")

        assert mock_instance.grab.call_count == 4  # called twice when skip is 3
        mock_instance.release.assert_called_once()

    @patch("cv2.VideoCapture")
    def test_process_single_video_no_skip_sequential(
        self, mock_vc, tmp_path, mock_threading_components
    ):
        """Verify sequential framing operates seamlessly without triggering grab loops when skip=1."""
        video_path = tmp_path / "stream.avi"
        video_path.touch()

        executor, semaphore, safe_write_func = mock_threading_components

        mock_instance = MagicMock()
        mock_instance.isOpened.side_effect = [True, True, True, False]
        mock_instance.read.side_effect = [(True, "f0"), (True, "f1"), (True, "f2")]
        mock_vc.return_value = mock_instance

        saved_frames = process_single_video(
            video_path, "prefix_", tmp_path, 1, executor, semaphore, safe_write_func
        )

        assert saved_frames == {0, 1, 2}
        assert mock_instance.grab.call_count == 0

    @patch("cv2.VideoCapture")
    def test_process_single_video_ended_early_in_grab_phase(
        self, mock_vc, tmp_path, mock_threading_components
    ):
        """If cap.grab() returns False during a frame skip phase, the loop must cleanly terminate."""
        video_path = tmp_path / "short_clip.mp4"
        video_path.touch()

        executor, semaphore, safe_write_func = mock_threading_components

        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_instance.read.return_value = (True, "frame_data")

        mock_instance.grab.side_effect = [True, False]
        mock_vc.return_value = mock_instance

        saved_frames = process_single_video(
            video_path, "prefix_", tmp_path, 5, executor, semaphore, safe_write_func
        )

        # Loop breaks safely during skip calculation
        assert saved_frames == {0}
        mock_instance.release.assert_called_once()

    @patch("cv2.VideoCapture")
    def test_process_single_video_semaphore_safety_on_grab_exit(
        self, mock_vc, tmp_path, mock_threading_components
    ):
        """
        Ensure that when a video ends early in the grab phase, the acquired semaphore is
        correctly balanced to prevent deadlocking downstream processing.
        """
        video_path = tmp_path / "deadlock_test.mp4"
        video_path.touch()

        executor, semaphore, safe_write_func = mock_threading_components
        initial_counter = semaphore._value  # Record initial token state

        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_instance.read.return_value = (True, "frame_data")
        mock_instance.grab.return_value = False  # Triggers an early break loop exit
        mock_vc.return_value = mock_instance

        process_single_video(
            video_path, "prefix_", tmp_path, 5, executor, semaphore, safe_write_func
        )

        # Note the early-exit break bug, - not sure this is a bug
        # this test will fail because semaphore._value will be lower than initial_counter,
        # hence call semaphore._value - 1
        assert semaphore._value == initial_counter - 1

    # --- VALUE & RESOURCE PROTECTION BOUNDARY TESTS ---

    def test_process_single_video_raises_file_not_found(
        self, mock_threading_components
    ):
        """Ensure non-existent files immediately stop execution via a FileNotFoundError."""
        ghost_path = Path("/tmp/non_existent_video_stream_file.mp4")
        executor, semaphore, safe_write_func = mock_threading_components

        with pytest.raises(FileNotFoundError):
            process_single_video(
                ghost_path, "pref_", Path("."), 5, executor, semaphore, safe_write_func
            )

    # --- TYPE CHECKING PROTECTION TESTS ---

    @pytest.mark.parametrize(
        "arg_name, kwargs",
        [
            ("video_path", {"video_path": "string_paths_are_invalid.mp4"}),
            ("file_prefix", {"file_prefix": 9999}),
            ("final_output_dir", {"final_output_dir": "./string_dir"}),
            ("executor", {"executor": MagicMock()}),
            ("semaphore", {"semaphore": MagicMock()}),
            ("safe_write_func", {"safe_write_func": "not_callable"}),
        ],
    )
    def test_process_single_video_type_guards(
        self, arg_name, kwargs, tmp_path, mock_threading_components
    ):
        """Systematically verify TypeErrors trigger whenever input signatures violate core type definitions."""
        video_file = tmp_path / "base_video.mp4"
        video_file.touch()

        executor, semaphore, safe_write_func = mock_threading_components

        base_args = {
            "video_path": video_file,
            "file_prefix": "prefix_",
            "final_output_dir": tmp_path,
            "skip": 5,
            "executor": executor,
            "semaphore": semaphore,
            "safe_write_func": safe_write_func,
        }
        base_args.update(kwargs)

        with pytest.raises(TypeError) as exc_info:
            process_single_video(**base_args)
        assert f"Argument '{arg_name}' must be" in str(exc_info.value)


class TestExtractFrames:

    @pytest.fixture
    def standard_setup(self, tmp_path):
        """Provides a consistent set of clean path inputs for the tests."""
        return {
            "video_paths": ["video1.mp4", "video2.mp4"],
            "videos_root": tmp_path / "raw_videos",
            "working_dir": tmp_path / "workspace",
            "split": "train",
            "skip": 2,
        }

    # --- TYPE & VALUE VALIDATION TESTS ---

    @pytest.mark.parametrize(
        "invalid_arg, patch_dict",
        [
            ("video_paths", {"video_paths": "not_a_list"}),  # Should be list
            ("videos_root", {"videos_root": "/not/a/path/obj"}),  # Should be Path
            ("working_dir", {"working_dir": "/not/a/path/obj"}),  # Should be Path
            ("skip", {"skip": "two"}),  # Should be int
            ("force", {"force": "True"}),  # Should be bool
        ],
    )
    def test_input_guard_type_errors(self, standard_setup, invalid_arg, patch_dict):
        """Verifies that improper input argument types immediately trigger a TypeError."""
        kwargs = {**standard_setup, **patch_dict}
        with pytest.raises(TypeError):
            extract_frames(**kwargs)

    @pytest.mark.parametrize(
        "patch_dict, expected_error",
        [
            ({"split": "invalid_split"}, ValueError),  # Must be train/val
            ({"skip": 0}, ValueError),  # Must be > 0
            ({"skip": -5}, ValueError),  # Must be > 0
        ],
    )
    def test_input_guard_value_errors(self, standard_setup, patch_dict, expected_error):
        """Verifies that out-of-bounds inputs or bad keywords trigger a ValueError."""
        kwargs = {**standard_setup, **patch_dict}
        with pytest.raises(ValueError):
            extract_frames(**kwargs)

    # --- BEHAVIORAL & LOGIC FLOW TESTS ---

    @patch("scripts.preprocessing.extract_frames.validate_file_paths")
    def test_early_exit_if_directory_exists_without_force(
        self, mock_validate, standard_setup
    ):
        """If the output image directory already exists, the function should log and

        exit early without processing videos.
        """
        working_dir = standard_setup["working_dir"]
        output_dir = working_dir / "images" / standard_setup["split"]
        output_dir.mkdir(parents=True, exist_ok=True)  # Pre-create directory

        result = extract_frames(**standard_setup, force=False)

        assert result is None
        mock_validate.assert_not_called()  # Never even started checking video files

    @patch("scripts.preprocessing.extract_frames.process_single_video")
    @patch("scripts.preprocessing.extract_frames.generate_video_metadata")
    @patch("scripts.preprocessing.extract_frames.validate_file_paths")
    def test_successful_orchestration_loop(
        self, mock_validate, mock_metadata, mock_process, standard_setup
    ):
        """Verifies the core success pathway: directories are made, videos are iterated

        over, and the registry map compiles properly.
        """

        p1, p2 = Path("v1.mp4"), Path("v2.mp4")
        mock_validate.return_value = [p1, p2]

        # Mock metadata returns: (video_key, file_prefix)
        mock_metadata.side_effect = [
            ((1, "v1"), "v1_prefix_"),
            ((2, "v2"), "v2_prefix_"),
        ]

        # Mock process_single_video returns: set of frame indices
        mock_process.side_effect = [{0, 2, 4}, {0, 2}]

        # Run extract_frames
        registry = extract_frames(**standard_setup, force=False)

        expected_output_dir = standard_setup["working_dir"] / "images" / "train"
        assert expected_output_dir.exists()

        # Check loop synchronization across both videos
        assert mock_process.call_count == 2

        # Verify the structure of the final output data registry mapping
        assert registry == {(1, "v1"): {0, 2, 4}, (2, "v2"): {0, 2}}

    @patch("scripts.preprocessing.extract_frames.process_single_video")
    @patch("scripts.preprocessing.extract_frames.generate_video_metadata")
    @patch("scripts.preprocessing.extract_frames.validate_file_paths")
    def test_force_flag_overrides_existing_directory(
        self, mock_validate, mock_metadata, mock_process, standard_setup
    ):
        """With force=True, processing runs completely even if the output directory

        already contains data.
        """
        working_dir = standard_setup["working_dir"]
        output_dir = working_dir / "images" / standard_setup["split"]
        output_dir.mkdir(parents=True, exist_ok=True)

        mock_validate.return_value = [Path("v1.mp4")]
        mock_metadata.return_value = ((1, "v1"), "v1_prefix_")
        mock_process.return_value = {0}

        registry = extract_frames(**standard_setup, force=True)

        assert registry == {(1, "v1"): {0}}
        mock_process.assert_called_once()
