import os
import sys
import pytest
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.baseline.baseline import compute_iou, frame_has_overlap, extract_events, run_detection

# Compute IoU tests
def test_compute_iou_perfect_overlap():
    """Verify that identical boxes yield an IoU of 1.0 and correct intersection."""
    box_a = [100, 100, 200, 200]
    box_b = [100, 100, 200, 200]
    iou, inter_box = compute_iou(box_a, box_b)
    
    assert iou == 1.0
    assert inter_box == [100, 100, 200, 200]


def test_compute_iou_partial_overlap():
    """Verify standard box math for partial overlap."""
    box_a = [0, 0, 10, 10]    # Area = 100
    box_b = [5, 0, 15, 10]    # Area = 100
    # Intersection is from x=5 to 10, y=0 to 10 -> Area = 50
    # Union = 100 + 100 - 50 = 150
    # Expected IoU = 50 / 150
    iou, inter_box = compute_iou(box_a, box_b)
    
    assert pytest.approx(iou, rel=1e-4) == 50/150
    assert inter_box == [5, 0, 10, 10]

    # test just overlap with area value of 1
    box_a = [0, 0, 10, 10]    # Area = 100
    box_b = [9, 9, 11, 11]    # Area = 4
    # Intersection is from x=9 to 10, y=9 to 10 -> Area = 1
    # Union = 100 + 4 - 1 = 103
    # Expected IoU = 1 / 103
    iou, inter_box = compute_iou(box_a, box_b)
    
    assert pytest.approx(iou, rel=1e-4) == 1/103
    assert inter_box == [9, 9, 10, 10]

    # box is inside another box
    box_a = [0, 0, 10, 10]    # Area = 100
    box_b = [1, 9, 9, 10]    # Area = 4
    # Intersection is from x=1 to 9, y=9 to 10 -> Area = 8
    # Union = 100 + 8 - 8 = 100
    # Expected IoU = 8/100
    iou, inter_box = compute_iou(box_a, box_b)
    
    assert pytest.approx(iou, rel=1e-4) == 8/100
    assert inter_box == [1, 9, 9, 10]


def test_compute_iou_no_overlap():
    """Verify that completely disconnected boxes yield 0.0 IoU."""
    box_a = [0, 0, 50, 50]
    box_b = [100, 100, 150, 150]
    iou, inter_box = compute_iou(box_a, box_b)
    
    assert iou == 0.0
    assert inter_box is None

    # edge cases where the boxes are next to each other
    box_a = [0, 0, 50, 50]
    box_b = [50, 0, 100, 50]
    iou, inter_box = compute_iou(box_a, box_b)
    
    assert iou == 0.0
    assert inter_box is None

# Frame. has overlap tests

def test_frame_has_overlap_picks_highest_iou_pair():
    """Ensure frame parsing identifies the highest overlapping pair correctly."""
    boxes = [
        [0, 0, 100, 100],     # Box 0
        [90, 90, 190, 190],   # Box 1 (Tiny overlap with Box 0)
        [10, 10, 90, 90]      # Box 2 (Massive overlap inside Box 0)
    ]
    # The pair (Box 0, Box 2) should easily beat the pair (Box 0, Box 1)
    has_overlap, best_box = frame_has_overlap(boxes, iou_threshold=0.2)
    
    assert has_overlap is True
    assert best_box == [10, 10, 90, 90]


# Extract events tests

def test_extract_events_filters_out_short_noise_durations():
    """Verify events shorter than min_duration are successfully discarded."""
    fps = 10.0
    min_duration = 1.0  # Requires at least 10 consecutive frames to qualify
    
    # Sequence: 12 true frames (Keep), 5 false frames, 3 true frames (Discard as noise)
    frame_flags = [True] * 12 + [False] * 5 + [True] * 3
    confidences = [0.9] * len(frame_flags)
    frame_boxes = [[10, 20, 30, 40]] * len(frame_flags)
    frame_indices = list(range(len(frame_flags)))

    events = extract_events(frame_flags, fps, min_duration, confidences, frame_boxes, frame_indices)

    # Only 1 valid event should have survived the filter
    assert len(events) == 1
    assert events[0]["start_sec"] == 0.0
    assert events[0]["end_sec"] == 12.0 / fps  # 11th index frame
    assert events[0]["duration_sec"] == 1.2


# Validation tests

def test_run_detection_raises_file_not_found_on_missing_video(mocker):
    """Ensure pipeline breaks gracefully if the video path doesn't point to a file."""
    # Mock os.path.exists to simulate that the model exists but the video doesn't
    mocker.patch("os.path.exists", side_effect=lambda path: path == "valid_model.pt")
    
    with pytest.raises(FileNotFoundError, match="Video file not found"):
        run_detection("missing_video.mp4", "valid_model.pt", 0.1, 0.5, 1.0, 1)

def test_run_detection_raises_file_not_found_on_missing_model(mocker):
    """Ensure pipeline breaks gracefully if the model path doesn't exist."""
    # First call (video check) returns True. Second call (model check) returns False.
    mocker.patch("os.path.exists", side_effect=[True, False])
    
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        run_detection("valid_video.mp4", "missing_model.pt", 0.1, 0.5, 1.0, 1)

def test_run_detection_raises_value_error_on_missing_model_classes(mocker):
    """Ensure pipeline crashes cleanly if the user attempts to find a cow using an ML model not trained on cows."""
    mocker.patch("os.path.exists", return_value=True)
    
    # Stub out YOLO completely
    mock_yolo = mocker.patch("scripts.baseline.baseline.YOLO")
    mock_instance = mock_yolo.return_value
    # Give it an arbitrary class map lacking "cow"
    mock_instance.names = {0: "person", 1: "dog"}

    with pytest.raises(ValueError, match="cow is not found in model classes"):
        run_detection("valid_video.mp4", "valid_model.pt", 0.1, 0.5, 1.0, 1)

# Test full pipeline

def test_run_detection_full_pipeline_success(mocker):
    """
    Executes an end-to-end integration loop of the full pipeline logic.
    Mocks away the heavy hardware/disk dependencies (OpenCV, YOLO, Disk Write).
    """
    # 1. Mock IO Safety checks
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("os.makedirs")
    
    mock_open = mocker.patch("builtins.open", mocker.mock_open())

    # 2. Mock YOLO setup
    mock_yolo_class = mocker.patch("scripts.baseline.baseline.YOLO")
    mock_model_instance = mocker.MagicMock()
    # Provide the necessary class names dictionary mapping containing our target
    mock_model_instance.names = {0: "person", 42: "cow"}
    mock_yolo_class.return_value = mock_model_instance

    # 3. Mock OpenCV Video Engine
    mock_cv2_cap_class = mocker.patch("cv2.VideoCapture")
    mock_cap_instance = mocker.MagicMock()
    mock_cap_instance.isOpened.return_value = True
    
    # Route Cap property queries (FPS, Width, Height, Frame Count) safely
    mock_cap_instance.get.side_effect = lambda prop: {
        5: 10.0,   # cv2.CAP_PROP_FPS
        3: 640,    # cv2.CAP_PROP_FRAME_WIDTH
        4: 480,    # cv2.CAP_PROP_FRAME_HEIGHT
        7: 2       # cv2.CAP_PROP_FRAME_COUNT
    }.get(prop, 0.0)
    
    # Configure video reader to yield 2 valid empty image frames, then signal EOF (False)
    mock_cap_instance.read.side_effect = [
        (True, np.zeros((480, 640, 3), dtype=np.uint8)),
        (True, np.zeros((480, 640, 3), dtype=np.uint8)),
        (False, None)
    ]
    mock_cv2_cap_class.return_value = mock_cap_instance

    # Intercept window renderings so UI dialogue boxes don't pop up on the monitor
    mocker.patch("cv2.imshow")
    mocker.patch("cv2.waitKey", return_value=1)

    # 4. Mock Artificial YOLO Inference Results 
    # Construct two dummy bounding boxes positioned right on top of each other
    mock_box_a = mocker.MagicMock()
    mock_box_a.cls = [mocker.MagicMock(item=lambda: 42)]  # Class 42 matches our cow target
    mock_box_a.xyxy = [np.array([10, 10, 100, 100])]
    mock_box_a.conf = [mocker.MagicMock(item=lambda: 0.88)]

    mock_box_b = mocker.MagicMock()
    mock_box_b.cls = [mocker.MagicMock(item=lambda: 42)]
    mock_box_b.xyxy = [np.array([15, 15, 105, 105])]
    mock_box_b.conf = [mocker.MagicMock(item=lambda: 0.92)]

    mock_result_frame = mocker.MagicMock()
    mock_result_frame.boxes = [mock_box_a, mock_box_b]
    mock_result_frame.plot.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

    # The pipeline reads element [0] of the object list returned by calling the model
    mock_model_instance.return_value = [mock_result_frame]

    # Run pipeline processing with min_duration set very low so 2 frames easily make an event
    metadata = run_detection(
        video_path="test_pasture_video.mp4",
        model_path="fake_yolo.pt",
        iou_threshold=0.1,
        conf_threshold=0.5,
        min_duration=0.1,
        frame_skip=1
    )

    # Validate metadata object structures
    assert metadata["identifier"] == "test_pasture_video.mp4"
    assert metadata["fps"] == 10.0
    assert metadata["total_frames"] == 2
    assert metadata["cross_sucking_detected"] is True
    assert metadata["num_events"] == 1
    
    # Assert JSON file save protocol was triggered correctly
    mock_open.assert_called_once()