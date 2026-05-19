# `scripts/clipping.py` 
### Clip reproduction & annotation

This module reproduces short clips from longer videos and (optionally) overlays bounding boxes on the resulting clips.

## Functions

### `reproduce_clip(raw_video_path: Path, start_sec: float, end_sec: float, output_path: Path) -> bool`

Extracts a clip from a source video between `start_sec` and `end_sec` and writes it to `output_path`.

**How it works**
- Opens the input video with `cv2.VideoCapture`.
- Reads `fps`, frame width, and frame height from the input.
- Converts seconds -> frame indices using `int(seconds * fps)`.
- Seeks to the start frame and writes frames until the end frame.
- Encodes output with `mp4v`.

**Parameters**
- `raw_video_path`: Path to the source `.mp4`.
- `start_sec`: Clip start time (seconds).
- `end_sec`: Clip end time (seconds).
- `output_path`: Output `.mp4` path.

**Returns**
- `True` if the video could be opened and the writer completed without early open failure.
- `False` if the input video cannot be opened.

**Notes / gotchas**
- If `cap.read()` fails before reaching `end_frame`, the function stops early and still returns `True` (it does not currently treat early EOF as failure).
- `output_path.parent` is **not created** in this function; ensure the output directory exists before calling (the JSON workflow does this).