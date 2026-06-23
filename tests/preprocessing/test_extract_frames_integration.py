"""
Integration tests for extract_frames.py
"""

from pathlib import Path
import sys
import concurrent.futures
import threading
import cv2
import numpy as np
import pytest

sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.preprocessing.extract_frames import process_single_video, extract_frames


class TestProcessSingleVideoIntegration:
    """Integration tests for process_single_video using real treading and semaphore executors"""

    @pytest.fixture
    def synthetic_video_path(self, tmp_path):
        """Generates a real, valid 10-frame sample video file on disk for true integration testing."""
        video_file = tmp_path / "synthetic_sample.mp4"

        # Configure an OpenCV VideoWriter (using a widely supported codec)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        width, height = 640, 480
        video_writer = cv2.VideoWriter(str(video_file), fourcc, 30.0, (width, height))

        try:
            # Write 10 real binary frames of solid gray pixels
            for _ in range(10):
                frame = np.ones((height, width, 3), dtype=np.uint8) * 128
                video_writer.write(frame)
        finally:
            video_writer.release()

        return video_file

    @pytest.mark.integration
    def test_process_single_video_end_to_end(self, synthetic_video_path, tmp_path):
        """
        Integration Test: Verifies that real video streams decode, pass through active
        OS threads, and physically write verified JPEG assets onto the filesystem.
        """
        output_dir = tmp_path / "extracted_frames"
        output_dir.mkdir()

        # Spin up real, active hardware concurrency containers
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        semaphore = threading.BoundedSemaphore(value=2)

        def real_safe_write_func(file_path: str, frame: cv2.Mat) -> None:
            """A real production-grade file writer that physically flushes frames to disk."""
            try:
                cv2.imwrite(file_path, frame)
            finally:
                semaphore.release()

        try:
            # Run the live video through the processing loop with a stride skip of 4
            saved_frames = process_single_video(
                video_path=synthetic_video_path,
                file_prefix="integ_test_",
                final_output_dir=output_dir,
                skip=4,
                executor=executor,
                semaphore=semaphore,
                safe_write_func=real_safe_write_func,
            )

            # Shut down the pool and force all background disk I/O threads to flush completely
            executor.shutdown(wait=True)

            # Verify the function mathematically tracked frames 0, 4, and 8
            assert saved_frames == {0, 4, 8}

            # Physical Disk Verification
            expected_file_0 = output_dir / "integ_test_000000.jpg"
            expected_file_4 = output_dir / "integ_test_000004.jpg"
            expected_file_8 = output_dir / "integ_test_000008.jpg"

            assert (
                expected_file_0.exists()
            ), "Frame 0 was not physically written to disk!"
            assert (
                expected_file_4.exists()
            ), "Frame 4 was not physically written to disk!"
            assert (
                expected_file_8.exists()
            ), "Frame 8 was not physically written to disk!"

            # Image Integrity Check (Verify the written files aren't 0-byte corrupt blanks)
            img = cv2.imread(str(expected_file_0))
            assert img is not None
            assert img.shape == (480, 640, 3)

        finally:
            # Fallback safeguard to clean up threads if assertions fail mid-flight
            executor.shutdown(wait=False)


class TestExtractFramesIntegration:
    """Integration tests for end-to-end extract_frames running."""

    @pytest.fixture
    def multiple_synthetic_videos(self, tmp_path):
        """Generates two real, short sample video files on disk for multi-video integration testing."""
        videos_root = tmp_path / "raw_videos"
        videos_root.mkdir()

        video_filenames = ["CS_0001_test_info_part01.mp4", "CS_1342_test_info.mp4"]
        video_paths = []

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        width, height = 320, 240  # Keep resolutions small so tests run instantly

        for name in video_filenames:
            video_file = videos_root / name
            video_writer = cv2.VideoWriter(
                str(video_file), fourcc, 30.0, (width, height)
            )

            try:
                # Write 7 frames per video
                for _ in range(9):
                    frame = np.ones((height, width, 3), dtype=np.uint8) * 100
                    video_writer.write(frame)
            finally:
                video_writer.release()

            # We need the relative path string as input for the orchestrator
            video_paths.append(name)

        return {
            "video_paths": video_paths,
            "videos_root": videos_root,
        }

    @pytest.mark.integration
    def test_extract_frames_orchestrator_end_to_end(
        self, multiple_synthetic_videos, tmp_path
    ):
        """
        Integration Test: Verifies that the top-level orchestrator successfully
        manages multiple concurrent video extractions, writes image assets to the
        filesystem split directory, and handles thread pooling automatically.
        """
        working_dir = tmp_path / "workspace"
        split = "val"
        skip = 3  # Stride skip of 3 means frames 0, 3, and 6 should be written for each video

        # Run the complete end-to-end pipeline execution
        registry = extract_frames(
            video_paths=multiple_synthetic_videos["video_paths"],
            videos_root=multiple_synthetic_videos["videos_root"],
            working_dir=working_dir,
            split=split,
            skip=skip,
            force=False,
        )

        # --- Verify Registry Metadata Map Structure ---
        # The keys depend on what your project's generate_video_metadata() outputs,
        # but the dictionary values should map to the exact sets of frames processed.
        assert isinstance(registry, dict)
        assert len(registry) == 2
        for video_key, saved_frames in registry.items():
            assert saved_frames == {0, 3, 6}

        # --- Physical Directory and Disk File Verification ---
        expected_output_dir = working_dir / "images" / split
        assert expected_output_dir.exists()

        # Get a list of all files physically written to the output path
        written_files = list(expected_output_dir.glob("*.jpg"))

        # With 2 videos, each writing 3 frames (0, 3, and 6), we expect exactly 6 images total
        assert len(written_files) == 6

        # Physical Disk Verification
        expected_file_0001_0 = expected_output_dir / "0001_part01_frame_000000.jpg"
        expected_file_0001_3 = expected_output_dir / "0001_part01_frame_000003.jpg"
        expected_file_0001_6 = expected_output_dir / "0001_part01_frame_000006.jpg"

        expected_file_1342_0 = expected_output_dir / "1342_None_frame_000000.jpg"
        expected_file_1342_3 = expected_output_dir / "1342_None_frame_000003.jpg"
        expected_file_1342_6 = expected_output_dir / "1342_None_frame_000006.jpg"

        assert (
            expected_file_0001_0.exists()
        ), "Frame 0 was not physically written to disk!"
        assert (
            expected_file_0001_3.exists()
        ), "Frame 3 was not physically written to disk!"
        assert (
            expected_file_0001_6.exists()
        ), "Frame 6 was not physically written to disk!"

        assert (
            expected_file_1342_0.exists()
        ), "Frame 0 was not physically written to disk!"
        assert (
            expected_file_1342_3.exists()
        ), "Frame 3 was not physically written to disk!"
        assert (
            expected_file_1342_6.exists()
        ), "Frame 6 was not physically written to disk!"

        # Ensure images are readable binary files and not empty 0-byte corrupt assets
        for img_path in written_files:
            img = cv2.imread(str(img_path))
            assert img is not None
            assert img.shape == (240, 320, 3)

    @pytest.mark.integration
    def test_extract_frames_orchestrator_skips_existing_by_default(
        self, multiple_synthetic_videos, tmp_path
    ):
        """
        Integration Test: Assures that if the image target folder already exists,
        the pipeline avoids repeating heavy processing unless 'force=True' is passed.
        """
        working_dir = tmp_path / "workspace"
        expected_output_dir = working_dir / "images" / "val"
        expected_output_dir.mkdir(parents=True, exist_ok=True)

        # Place a placeholder dummy file inside to see if it gets wiped or skipped
        sentinel_file = expected_output_dir / "should_not_be_overwritten.txt"
        sentinel_file.write_text("touch me not")

        # Attempt execution without force=True
        registry = extract_frames(
            video_paths=multiple_synthetic_videos["video_paths"],
            videos_root=multiple_synthetic_videos["videos_root"],
            working_dir=working_dir,
            split="val",
            skip=3,
            force=False,
        )

        # Should return early (None) and leave our sentinel file completely intact
        assert registry is None
        assert sentinel_file.exists()
