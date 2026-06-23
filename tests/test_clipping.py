import json
import pytest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent)) 
from scripts.clipping import split_by_json_events


def test_split_by_json_events_no_events(tmp_path: Path):
    """
    If JSON has no events, the function should return 0 and produce no clips.
    The output directory is not created when there are no events to process.
    """
    video_path = tmp_path / "dummy.mp4"
    video_path.write_bytes(b"not a real mp4")

    data = {
        "video_path": str(video_path),
        "identifier": "dummy.mp4",
        "events": [],
        "fps": 30.0,
    }
    json_path = tmp_path / "events.json"
    json_path.write_text(json.dumps(data), encoding="utf-8")

    out_dir = tmp_path / "out"
    result = split_by_json_events(json_path, out_dir)

    assert result == 0
    assert list(out_dir.rglob("*.mp4")) == [] if out_dir.exists() else True

def test_split_by_json_events_missing_video_raises(tmp_path: Path):
    """Basic error handling: missing video_path should raise FileNotFoundError."""
    missing_video = tmp_path / "missing.mp4"

    data = {
        "video_path": str(missing_video),
        "events": [{"start_sec": 0, "end_sec": 1}],
        "fps": 30.0,
    }
    json_path = tmp_path / "events.json"
    json_path.write_text(json.dumps(data), encoding="utf-8")

    out_dir = tmp_path / "out"

    with pytest.raises(FileNotFoundError):
        split_by_json_events(json_path, out_dir)
        
def test_split_by_json_events_skips_event_when_reproduce_fails(tmp_path, mocker):
    """If reproduce_clip fails, that event should not count toward total_success."""
    mocker.patch("pathlib.Path.exists", return_value=True)
    mocker.patch("scripts.clipping.reproduce_clip", return_value=False)

    json_path = tmp_path / "test.json"
    json_path.write_text(json.dumps({
        "video_path": "/fake/videos/Pen1/Stage1/Day1/cam.mp4",
        "identifier": "cam.mp4",
        "events":     [{"start_sec": 0.0, "end_sec": 1.0, "intersection_box": []}],
        "fps":        30.0,
    }))

    result = split_by_json_events(json_path, tmp_path / "out")

    assert result == 0

def test_split_by_json_events_returns_zero_for_empty_dir(tmp_path):
    """A directory with no JSON files should return 0."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    result = split_by_json_events(empty_dir, tmp_path / "out")

    assert result == 0