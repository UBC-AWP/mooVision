import os
import sys
import pytest
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent)) 
from scripts.models.baseline.baseline import compute_iou, frame_has_overlap, extract_events, detect_video


# ---------------------------------------------------------------------------
# compute_iou
# ---------------------------------------------------------------------------

def test_compute_iou_perfect_overlap():
    """Identical boxes yield IoU of 1.0 and correct intersection."""
    box_a = [100, 100, 200, 200]
    box_b = [100, 100, 200, 200]
    iou, inter_box = compute_iou(box_a, box_b)

    assert iou == 1.0
    assert inter_box == [100, 100, 200, 200]


def test_compute_iou_partial_overlap():
    """Standard box math for partial overlap."""
    box_a = [0, 0, 10, 10]   # Area = 100
    box_b = [5, 0, 15, 10]   # Area = 100
    # Intersection x=5..10, y=0..10 → Area = 50; Union = 150
    iou, inter_box = compute_iou(box_a, box_b)
    assert pytest.approx(iou, rel=1e-4) == 50 / 150
    assert inter_box == [5, 0, 10, 10]

    # Minimal overlap (area = 1)
    box_a = [0, 0, 10, 10]   # Area = 100
    box_b = [9, 9, 11, 11]   # Area = 4
    # Intersection x=9..10, y=9..10 → Area = 1; Union = 103
    iou, inter_box = compute_iou(box_a, box_b)
    assert pytest.approx(iou, rel=1e-4) == 1 / 103
    assert inter_box == [9, 9, 10, 10]

    # Box fully inside another box
    box_a = [0, 0, 10, 10]   # Area = 100
    box_b = [1, 9,  9, 10]   # Area = 8
    # Intersection x=1..9, y=9..10 → Area = 8; Union = 100
    iou, inter_box = compute_iou(box_a, box_b)
    assert pytest.approx(iou, rel=1e-4) == 8 / 100
    assert inter_box == [1, 9, 9, 10]


def test_compute_iou_no_overlap():
    """Completely disconnected boxes yield IoU of 0.0."""
    box_a = [0,   0,  50,  50]
    box_b = [100, 100, 150, 150]
    iou, inter_box = compute_iou(box_a, box_b)
    assert iou == 0.0
    assert inter_box is None

    # Touching edges (no actual overlap)
    box_a = [0,  0, 50, 50]
    box_b = [50, 0, 100, 50]
    iou, inter_box = compute_iou(box_a, box_b)
    assert iou == 0.0
    assert inter_box is None


# ---------------------------------------------------------------------------
# frame_has_overlap
# ---------------------------------------------------------------------------

def test_frame_has_overlap_picks_highest_iou_pair():
    """frame_has_overlap identifies the highest-IoU pair correctly."""
    boxes = [
        [0,  0,  100, 100],   # Box 0
        [90, 90, 190, 190],   # Box 1 — tiny overlap with Box 0
        [10, 10,  90,  90],   # Box 2 — large overlap inside Box 0
    ]
    has_overlap, best_box = frame_has_overlap(boxes, iou_threshold=0.2)
    assert has_overlap is True
    assert best_box == [10, 10, 90, 90]


def test_frame_has_overlap_returns_false_for_single_box():
    """A single box cannot form a pair — should return (False, None)."""
    boxes = [[0, 0, 100, 100]]
    has_overlap, best_box = frame_has_overlap(boxes, iou_threshold=0.1)
    assert has_overlap is False
    assert best_box is None


def test_frame_has_overlap_returns_false_for_empty_boxes():
    """Empty box list should return (False, None) without error."""
    has_overlap, best_box = frame_has_overlap([], iou_threshold=0.1)
    assert has_overlap is False
    assert best_box is None


def test_frame_has_overlap_below_threshold():
    """Overlap below threshold should not be flagged."""
    box_a = [0, 0, 10, 10]
    box_b = [9, 9, 11, 11]   # IoU = 1/103 ≈ 0.0097
    has_overlap, _ = frame_has_overlap([box_a, box_b], iou_threshold=0.1)
    assert has_overlap is False


# ---------------------------------------------------------------------------
# extract_events
# ---------------------------------------------------------------------------

def test_extract_events_filters_short_noise():
    """Events shorter than min_duration are discarded."""
    fps          = 10.0
    min_duration = 1.0   # requires ≥ 10 consecutive frames
    frame_skip   = 1

    # 12 flagged (keep) + 5 gap + 3 flagged (discard)
    frame_flags  = [True] * 12 + [False] * 5 + [True] * 3
    confidences  = [0.9] * len(frame_flags)
    frame_boxes  = [[10, 20, 30, 40]] * len(frame_flags)
    frame_indices = list(range(len(frame_flags)))

    events = extract_events(
        frame_flags, fps, min_duration, frame_skip,
        confidences, frame_boxes, frame_indices,
    )

    assert len(events) == 1
    assert events[0]["start_sec"]   == 0.0
    assert events[0]["end_sec"]     == pytest.approx(12.0 / fps)
    assert events[0]["duration_sec"] == pytest.approx(1.2)


def test_extract_events_keeps_event_at_end_of_video():
    """An event running to the last frame is not dropped."""
    fps          = 10.0
    min_duration = 0.5
    frame_skip   = 1

    frame_flags   = [False] * 5 + [True] * 10
    confidences   = [0.8] * len(frame_flags)
    frame_boxes   = [[0, 0, 50, 50]] * len(frame_flags)
    frame_indices = list(range(len(frame_flags)))

    events = extract_events(
        frame_flags, fps, min_duration, frame_skip,
        confidences, frame_boxes, frame_indices,
    )

    assert len(events) == 1
    assert events[0]["start_sec"] == pytest.approx(5.0 / fps)


def test_extract_events_no_flags():
    """All-False flags produce no events."""
    fps          = 10.0
    n            = 20
    frame_flags  = [False] * n
    confidences  = [0.0] * n
    frame_boxes  = [None] * n
    frame_indices = list(range(n))

    events = extract_events(
        frame_flags, fps, 0.5, 1,
        confidences, frame_boxes, frame_indices,
    )
    assert events == []


def test_extract_events_respects_frame_skip():
    """frame_skip > 1 raises the min_frames bar proportionally."""
    fps          = 10.0
    min_duration = 1.0   # needs ≥ 10 source-video frames worth
    frame_skip   = 2     # each processed frame = 2 source frames
                         # → min_frames = round(1.0 * 10 / 2) = 5 processed frames

    # 4 processed frames flagged — should be discarded (4 < 5)
    frame_flags   = [True] * 4
    confidences   = [0.9] * 4
    frame_boxes   = [[0, 0, 10, 10]] * 4
    frame_indices = [0, 2, 4, 6]   # every 2nd source frame

    events = extract_events(
        frame_flags, fps, min_duration, frame_skip,
        confidences, frame_boxes, frame_indices,
    )
    assert events == []


# ---------------------------------------------------------------------------
# detect_video — validation
# ---------------------------------------------------------------------------

def test_detect_video_raises_on_missing_video(mocker):
    """detect_video raises FileNotFoundError if OpenCV cannot open the video."""
    mock_cap = mocker.patch("cv2.VideoCapture")
    mock_cap.return_value.isOpened.return_value = False

    with pytest.raises(FileNotFoundError, match="Cannot open video"):
        detect_video(
            video_path     = Path("missing.mp4"),
            model          = mocker.MagicMock(),
            target_ids     = {42},
            iou_threshold  = 0.1,
            conf_threshold = 0.5,
            min_duration   = 1.0,
            frame_skip     = 1,
        )


# ---------------------------------------------------------------------------
# detect_video — full pipeline
# ---------------------------------------------------------------------------

def test_detect_video_full_pipeline_success(mocker):
    """
    End-to-end integration test of detect_video.
    Mocks OpenCV and YOLO; asserts metadata structure and event detection.
    """
    mock_model = mocker.MagicMock()
    target_ids = {42}

    mock_cap = mocker.MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        5: 10.0,  # CAP_PROP_FPS
        3: 640,   # CAP_PROP_FRAME_WIDTH
        4: 480,   # CAP_PROP_FRAME_HEIGHT
        7: 2,     # CAP_PROP_FRAME_COUNT
    }.get(prop, 0.0)
    mock_cap.read.side_effect = [
        (True,  np.zeros((480, 640, 3), dtype=np.uint8)),
        (True,  np.zeros((480, 640, 3), dtype=np.uint8)),
        (False, None),
    ]
    mocker.patch("cv2.VideoCapture", return_value=mock_cap)

    # Two overlapping cow boxes
    def make_box(xyxy, conf):
        b = mocker.MagicMock()
        b.cls  = [mocker.MagicMock(item=lambda: 42)]
        b.xyxy = [np.array(xyxy)]
        b.conf = [mocker.MagicMock(item=lambda: conf)]
        return b

    mock_result = mocker.MagicMock()
    mock_result.boxes = [
        make_box([10, 10, 100, 100], 0.88),
        make_box([15, 15, 105, 105], 0.92),
    ]
    mock_model.return_value = [mock_result]

    metadata = detect_video(
        video_path     = Path("test_video.mp4"),
        model          = mock_model,
        target_ids     = target_ids,
        iou_threshold  = 0.1,
        conf_threshold = 0.5,
        min_duration   = 0.1,
        frame_skip     = 1,
    )

    assert metadata["identifier"]             == "test_video.mp4"
    assert metadata["fps"]                    == 10.0
    assert metadata["total_frames"]           == 2
    assert metadata["cross_sucking_detected"] is True
    assert metadata["num_events"]             == 1


def test_detect_video_no_events_when_below_threshold(mocker):
    """No events produced when IoU is always below threshold."""
    mock_model = mocker.MagicMock()
    target_ids = {42}

    mock_cap = mocker.MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {5: 10.0, 3: 640, 4: 480, 7: 2}.get(prop, 0.0)
    mock_cap.read.side_effect = [
        (True,  np.zeros((480, 640, 3), dtype=np.uint8)),
        (True,  np.zeros((480, 640, 3), dtype=np.uint8)),
        (False, None),
    ]
    mocker.patch("cv2.VideoCapture", return_value=mock_cap)

    # Two boxes far apart — IoU ≈ 0
    def make_box(xyxy, conf):
        b = mocker.MagicMock()
        b.cls  = [mocker.MagicMock(item=lambda: 42)]
        b.xyxy = [np.array(xyxy)]
        b.conf = [mocker.MagicMock(item=lambda: conf)]
        return b

    mock_result = mocker.MagicMock()
    mock_result.boxes = [
        make_box([0,   0,  50,  50], 0.9),
        make_box([200, 200, 250, 250], 0.9),
    ]
    mock_model.return_value = [mock_result]

    metadata = detect_video(
        video_path     = Path("test_video.mp4"),
        model          = mock_model,
        target_ids     = target_ids,
        iou_threshold  = 0.1,
        conf_threshold = 0.5,
        min_duration   = 0.1,
        frame_skip     = 1,
    )

    assert metadata["cross_sucking_detected"] is False
    assert metadata["num_events"]             == 0