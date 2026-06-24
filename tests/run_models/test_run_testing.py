"""
Tests for the run-testing-2.py orchestrator script.

Run with:
    uv run pytest tests/test_run_testing_2.py -v
"""

from pathlib import Path

import pandas as pd
import pytest
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import LOCAL_DIR, ROOT_DIR, UNLABELLED_CLIPS_DIR

PROCESSED_DIR = Path(LOCAL_DIR) / "data" / "processed"
MODELS_DIR    = Path(__file__).parent / "models"
RESULTS_DIR   = Path(ROOT_DIR) / "results" / "metadata"
RUN_SCRIPT    = Path(__file__).parent / "run-testing.py"

import scripts.run_models.run_testing as rt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_root(tmp_path, mocker):
    """Patch ROOT_DIR to a temp dir and return it."""
    mocker.patch.object(rt, "ROOT_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def fake_source_videos_dir(tmp_path, mocker):
    """Patch SOURCE_VIDEOS_DIR to a temp dir and return it."""
    src_dir = tmp_path / "source_videos"
    src_dir.mkdir(parents=True, exist_ok=True)
    mocker.patch.object(rt, "SOURCE_VIDEOS_DIR", src_dir)
    return src_dir


# ---------------------------------------------------------------------------
# load_test_csv
# ---------------------------------------------------------------------------

class TestLoadTestCsv:
    def test_loads_valid_csv(self, fake_root):
        data_path = "data/processed/random/test.csv"
        full_path = fake_root / data_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"source_video_path": ["a.mp4", "b.mp4"]}).to_csv(full_path)

        df = rt.load_test_csv(data_path)

        assert len(df) == 2
        assert "source_video_path" in df.columns

    def test_raises_filenotfound_when_missing(self, fake_root):
        with pytest.raises(FileNotFoundError):
            rt.load_test_csv("data/processed/missing/test.csv")

    def test_raises_valueerror_when_empty(self, fake_root):
        data_path = "data/processed/empty/test.csv"
        full_path = fake_root / data_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"source_video_path": []}).to_csv(full_path)

        with pytest.raises(ValueError):
            rt.load_test_csv(data_path)


# ---------------------------------------------------------------------------
# get_split_label
# ---------------------------------------------------------------------------

class TestGetSplitLabel:
    def test_simple_split(self, fake_root):
        # Anchor the input path relative to our fake_root
        data_path = "data/processed/random/test.csv"
        assert rt.get_split_label(data_path) == "random"

    def test_nested_split(self, fake_root):
        # Anchor a nested path relative to our fake_root
        data_path = "data/processed/pen_based/pen_2/test.csv"
        assert rt.get_split_label(data_path) == "pen_based/pen_2"

    def test_relative_to_mismatch_raises(self, fake_root):
        # An absolute or relative path outside 'data/processed' will now trigger a ValueError
        with pytest.raises(ValueError):
            rt.get_split_label("some/other/path/test.csv")


# ---------------------------------------------------------------------------
# clean_video_paths
# ---------------------------------------------------------------------------

class TestCleanVideoPaths:
    def test_keeps_existing_videos(self, fake_source_videos_dir):
        rel = Path("pen1") / "2024" / "01" / "video1.mp4"
        full = fake_source_videos_dir / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.touch()

        # source_video_path values just need >=4 trailing parts matching rel
        series = pd.Series([str(Path("X") / rel)])
        result = rt.clean_video_paths(series)

        assert result == [str(full)]

    def test_drops_nonexistent_videos(self, fake_source_videos_dir):
        rel = Path("pen1") / "2024" / "01" / "missing.mp4"
        series = pd.Series([str(Path("X") / rel)])

        result = rt.clean_video_paths(series)

        assert result == []

    def test_dedupes_and_sorts(self, fake_source_videos_dir):
        rel_a = Path("pen1") / "2024" / "01" / "a.mp4"
        rel_b = Path("pen1") / "2024" / "01" / "b.mp4"
        for rel in (rel_a, rel_b):
            full = fake_source_videos_dir / rel
            full.parent.mkdir(parents=True, exist_ok=True)
            full.touch()

        # duplicate entry for b.mp4
        series = pd.Series(
            [
                str(Path("X") / rel_b),
                str(Path("X") / rel_a),
                str(Path("X") / rel_b),
            ]
        )
        result = rt.clean_video_paths(series)

        assert result == sorted({str(fake_source_videos_dir / rel_a), str(fake_source_videos_dir / rel_b)})

    def test_handles_windows_style_backslashes(self, fake_source_videos_dir):
        rel = Path("pen1") / "2024" / "01" / "video1.mp4"
        full = fake_source_videos_dir / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.touch()

        windows_path = "X\\pen1\\2024\\01\\video1.mp4"
        series = pd.Series([windows_path])

        result = rt.clean_video_paths(series)

        assert result == [str(full)]

    def test_empty_series_returns_empty_list(self, fake_source_videos_dir):
        assert rt.clean_video_paths(pd.Series([], dtype=str)) == []


# ---------------------------------------------------------------------------
# select_chunk
# ---------------------------------------------------------------------------

class TestSelectChunk:
    @pytest.fixture
    def videos(self):
        return [f"video_{i}.mp4" for i in range(10)]

    def test_first_chunk_of_ten(self, videos):
        result = rt.select_chunk(videos, chunk=0, chunk_pct=0.1)
        assert result == ["video_0.mp4"]

    def test_last_chunk_of_ten(self, videos):
        result = rt.select_chunk(videos, chunk=9, chunk_pct=0.1)
        assert result == ["video_9.mp4"]

    def test_chunks_cover_all_videos_without_overlap(self, videos):
        seen = []
        for chunk in range(10):
            seen.extend(rt.select_chunk(videos, chunk=chunk, chunk_pct=0.1))
        assert sorted(seen) == sorted(videos)

    def test_half_split_two_chunks(self, videos):
        first_half = rt.select_chunk(videos, chunk=0, chunk_pct=0.5)
        second_half = rt.select_chunk(videos, chunk=1, chunk_pct=0.5)
        assert first_half == videos[:5]
        assert second_half == videos[5:]

    def test_chunk_out_of_range_raises(self, videos):
        with pytest.raises(ValueError, match="out of range"):
            rt.select_chunk(videos, chunk=10, chunk_pct=0.1)

    def test_negative_chunk_raises(self, videos):
        with pytest.raises(ValueError, match="out of range"):
            rt.select_chunk(videos, chunk=-1, chunk_pct=0.1)

    def test_chunk_with_too_few_videos_raises_empty(self):
        # With small video counts relative to n_chunks, `start` can land at
        # or beyond the last valid index, producing an empty slice rather
        # than literally exceeding n. select_chunk should still raise.
        small_list = ["only_one.mp4"]
        with pytest.raises(ValueError, match="empty"):
            rt.select_chunk(small_list, chunk=5, chunk_pct=0.1)

    def test_uneven_split_distributes_remainder(self):
        videos = [f"v{i}.mp4" for i in range(8)]
        # 8 videos / 8 chunks (chunk_pct=0.1 rounds to 10 chunks) -> some empty chunks possible
        # use a cleaner case: 8 videos, chunk_pct=0.25 -> 4 chunks of 2 each
        chunks = [rt.select_chunk(videos, chunk=c, chunk_pct=0.25) for c in range(4)]
        assert [len(c) for c in chunks] == [2, 2, 2, 2]
        assert sum(chunks, []) == videos


# ---------------------------------------------------------------------------
# load_yolo_model
# ---------------------------------------------------------------------------

class TestLoadYoloModel:
    def test_loads_relative_path(self, fake_root, mocker):
        weights_rel = "models/best.pt"
        weights_abs = fake_root / weights_rel
        weights_abs.parent.mkdir(parents=True, exist_ok=True)
        weights_abs.touch()

        mock_yolo_cls = mocker.patch.object(rt, "YOLO")
        mock_instance = mocker.Mock()
        mock_yolo_cls.return_value = mock_instance

        model, resolved = rt.load_yolo_model(weights_rel)

        mock_yolo_cls.assert_called_once_with(str(weights_abs))
        assert model is mock_instance
        assert resolved == str(weights_abs)

    def test_loads_absolute_path(self, fake_root, mocker):
        weights_abs = fake_root / "abs_models" / "best.pt"
        weights_abs.parent.mkdir(parents=True, exist_ok=True)
        weights_abs.touch()

        mock_yolo_cls = mocker.patch.object(rt, "YOLO")

        rt.load_yolo_model(str(weights_abs))

        mock_yolo_cls.assert_called_once_with(str(weights_abs))

    def test_missing_weights_raises_filenotfound(self, fake_root, mocker):
        mocker.patch.object(rt, "YOLO")
        with pytest.raises(FileNotFoundError):
            rt.load_yolo_model("models/does_not_exist.pt")


# ---------------------------------------------------------------------------
# check_processed
# ---------------------------------------------------------------------------

class TestCheckProcessed:
    def test_true_when_both_jsons_exist(self, tmp_path):
        output_dir = tmp_path / "out"
        (output_dir / "yolo").mkdir(parents=True)
        (output_dir / "seq-nms").mkdir(parents=True)
        (output_dir / "yolo" / "clip1_results.json").touch()
        (output_dir / "seq-nms" / "clip1_results.json").touch()

        assert rt.check_processed(Path("/videos/clip1.mp4"), output_dir) is True

    def test_false_when_only_yolo_exists(self, tmp_path):
        output_dir = tmp_path / "out"
        (output_dir / "yolo").mkdir(parents=True)
        (output_dir / "seq-nms").mkdir(parents=True)
        (output_dir / "yolo" / "clip1_results.json").touch()

        assert rt.check_processed(Path("/videos/clip1.mp4"), output_dir) is False

    def test_false_when_neither_exists(self, tmp_path):
        output_dir = tmp_path / "out"
        (output_dir / "yolo").mkdir(parents=True)
        (output_dir / "seq-nms").mkdir(parents=True)

        assert rt.check_processed(Path("/videos/clip1.mp4"), output_dir) is False


# ---------------------------------------------------------------------------
# run_single_video
# ---------------------------------------------------------------------------

class TestRunSingleVideo:
    def _common_kwargs(self, output_dir):
        return dict(
            model=object(),
            model_path="/fake/best.pt",
            output_dir=output_dir,
            conf_threshold=0.25,
            iou_threshold=0.5,
            min_duration=1.0,
            buffer=30,
            frame_skip=10,
            target_class="cross-sucking",
        )

    def test_skips_when_already_processed(self, tmp_path, mocker):
        output_dir = tmp_path / "out"
        (output_dir / "yolo").mkdir(parents=True)
        (output_dir / "seq-nms").mkdir(parents=True)
        (output_dir / "yolo" / "clip1_results.json").touch()
        (output_dir / "seq-nms" / "clip1_results.json").touch()

        mock_run_models = mocker.patch.object(rt, "run_models")

        status = rt.run_single_video(
            video_str="/videos/clip1.mp4",
            overwrite=False,
            **self._common_kwargs(output_dir),
        )

        assert status == "skipped"
        mock_run_models.assert_not_called()

    def test_reprocesses_when_overwrite_true(self, tmp_path, mocker):
        output_dir = tmp_path / "out"
        (output_dir / "yolo").mkdir(parents=True)
        (output_dir / "seq-nms").mkdir(parents=True)
        (output_dir / "yolo" / "clip1_results.json").touch()
        (output_dir / "seq-nms" / "clip1_results.json").touch()

        mock_run_models = mocker.patch.object(rt, "run_models")

        status = rt.run_single_video(
            video_str="/videos/clip1.mp4",
            overwrite=True,
            **self._common_kwargs(output_dir),
        )

        assert status == "processed"
        mock_run_models.assert_called_once()

    def test_processes_when_not_yet_done(self, tmp_path, mocker):
        output_dir = tmp_path / "out"
        mock_run_models = mocker.patch.object(rt, "run_models")

        status = rt.run_single_video(
            video_str="/videos/new_clip.mp4",
            overwrite=False,
            **self._common_kwargs(output_dir),
        )

        assert status == "processed"
        mock_run_models.assert_called_once()
        _, call_kwargs = mock_run_models.call_args
        assert call_kwargs["video_path"] == "/videos/new_clip.mp4"
        assert call_kwargs["show_video"] is False

    def test_returns_failed_on_exception(self, tmp_path, mocker):
        output_dir = tmp_path / "out"
        mocker.patch.object(rt, "run_models", side_effect=RuntimeError("boom"))

        status = rt.run_single_video(
            video_str="/videos/bad_clip.mp4",
            overwrite=False,
            **self._common_kwargs(output_dir),
        )

        assert status == "failed"


# ---------------------------------------------------------------------------
# run_testing (integration of the orchestration logic, with everything
# below it mocked out)
# ---------------------------------------------------------------------------

class TestRunTesting:
    @pytest.fixture
    def setup_csv(self, fake_root):
        data_path = "data/processed/random/test.csv"
        full_path = fake_root / data_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"source_video_path": [f"X/pen1/2024/01/video_{i}.mp4" for i in range(10)]}
        ).to_csv(full_path)
        return data_path

    def test_processes_all_selected_videos(self, fake_root, setup_csv, mocker):
        mocker.patch.object(
            rt, "clean_video_paths", return_value=[f"/abs/video_{i}.mp4" for i in range(10)]
        )
        mocker.patch.object(rt, "YOLO")
        mocker.patch.object(rt, "load_yolo_model", return_value=(mocker.Mock(), "/fake/best.pt"))
        mock_run_single = mocker.patch.object(rt, "run_single_video", return_value="processed")

        rt.run_testing(
            model_path="best.pt",
            data_path=setup_csv,
            conf_threshold=0.25,
            iou_threshold=0.5,
            min_duration=1.0,
            buffer=30,
            frame_skip=10,
            chunk=0,
            chunk_pct=0.1,
        )

        assert mock_run_single.call_count == 1  # chunk 0 of 10% -> 1 video

    def test_creates_output_dir_under_results_metadata(self, fake_root, setup_csv, mocker):
        mocker.patch.object(
            rt, "clean_video_paths", return_value=[f"/abs/video_{i}.mp4" for i in range(10)]
        )
        mocker.patch.object(rt, "YOLO")
        mocker.patch.object(rt, "load_yolo_model", return_value=(mocker.Mock(), "/fake/best.pt"))
        mocker.patch.object(rt, "run_single_video", return_value="processed")

        rt.run_testing(
            model_path="best.pt",
            data_path=setup_csv,
            conf_threshold=0.25,
            iou_threshold=0.5,
            min_duration=1.0,
            buffer=30,
            frame_skip=10,
            chunk=0,
            chunk_pct=0.1,
        )

        expected_dir = fake_root / "results" / "metadata" / "random"
        assert expected_dir.exists()

    def test_counts_processed_skipped_failed(self, fake_root, setup_csv, mocker):
        mocker.patch.object(
            rt, "clean_video_paths", return_value=[f"/abs/video_{i}.mp4" for i in range(3)]
        )
        mocker.patch.object(rt, "YOLO")
        mocker.patch.object(rt, "load_yolo_model", return_value=(mocker.Mock(), "/fake/best.pt"))
        mocker.patch.object(
            rt, "run_single_video", side_effect=["processed", "skipped", "failed"]
        )
        # force a single chunk containing all 3 videos
        mocker.patch.object(rt, "select_chunk", side_effect=lambda videos, chunk, chunk_pct: videos)

        # should not raise; counts are only printed, so we mainly verify no crash
        rt.run_testing(
            model_path="best.pt",
            data_path=setup_csv,
            conf_threshold=0.25,
            iou_threshold=0.5,
            min_duration=1.0,
            buffer=30,
            frame_skip=10,
            chunk=0,
            chunk_pct=1.0,
        )

    def test_propagates_filenotfound_for_missing_csv(self, fake_root, mocker):
        mocker.patch.object(rt, "YOLO")
        with pytest.raises(FileNotFoundError):
            rt.run_testing(
                model_path="best.pt",
                data_path="data/processed/missing/test.csv",
                conf_threshold=0.25,
                iou_threshold=0.5,
                min_duration=1.0,
                buffer=30,
                frame_skip=10,
            )

    def test_propagates_filenotfound_for_missing_model(self, fake_root, setup_csv, mocker):
        mocker.patch.object(
            rt, "clean_video_paths", return_value=[f"/abs/video_{i}.mp4" for i in range(10)]
        )
        mocker.patch.object(rt, "YOLO")
        # do not mock load_yolo_model -> real one runs and should raise
        with pytest.raises(FileNotFoundError):
            rt.run_testing(
                model_path="does_not_exist.pt",
                data_path=setup_csv,
                conf_threshold=0.25,
                iou_threshold=0.5,
                min_duration=1.0,
                buffer=30,
                frame_skip=10,
            )


# ---------------------------------------------------------------------------
# parse_args (CLI surface)
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_required_args_only(self, mocker):
        mocker.patch(
            "sys.argv",
            [
                "run-testing-2.py",
                "--data_path", "data/processed/random/test.csv",
                "--model_path", "best.pt",
            ],
        )
        args = rt.parse_args()

        assert args.data_path == "data/processed/random/test.csv"
        assert args.model_path == "best.pt"
        assert args.frame_skip == rt.DEFAULT_FRAME_SKIP
        assert args.chunk == 0
        assert args.chunk_pct == 0.10
        assert args.overwrite is False

    def test_missing_required_arg_exits(self, mocker):
        mocker.patch("sys.argv", ["run-testing-2.py", "--data_path", "x.csv"])
        with pytest.raises(SystemExit):
            rt.parse_args()

    def test_overwrite_flag_sets_true(self, mocker):
        mocker.patch(
            "sys.argv",
            [
                "run-testing-2.py",
                "--data_path", "x.csv",
                "--model_path", "m.pt",
                "--overwrite",
            ],
        )
        args = rt.parse_args()
        assert args.overwrite is True

    def test_chunk_and_chunk_pct_parsed(self, mocker):
        mocker.patch(
            "sys.argv",
            [
                "run-testing-2.py",
                "--data_path", "x.csv",
                "--model_path", "m.pt",
                "--chunk", "3",
                "--chunk_pct", "0.25",
            ],
        )
        args = rt.parse_args()
        assert args.chunk == 3
        assert args.chunk_pct == 0.25