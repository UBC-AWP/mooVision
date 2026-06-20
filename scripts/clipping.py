"""
scripts.clipping
================

Reproduce short video clips from longer source videos.

Overview
--------
This module provides a small pipeline for reproducing clip video segments from
source `.mp4` files using one of two input formats:

1. CSV index mode (`split_by_index`):
   Reads a CSV index describing clip start/end times in the source video.

2. JSON events mode (`split_by_json_events`):
   Reads one JSON file (or a directory of JSON files) describing events to clip,
   optionally producing an additional annotated version of each clip with
   bounding boxes.

Input/Output
------------
Inputs
  - Source videos (`*.mp4`) located under :root:`config.SOURCE_VIDEOS_DIR`.
  - A CSV index at :root:`config.INDEX_PATH` (index mode), or JSON metadata files
    under :root:`config.BASELINE_METADATA_DIR` (JSON events mode).

Outputs
  - Reproduced clips written to :root:`config.REPRODUCED_CLIPS_DIR`.

Notes
-----
The module relies on OpenCV for video I/O. Configuration is centralized in
`config.py` and typically driven by environment variables in a `.env` file.

"""

import sys
import cv2
import tempfile, os
import time
import json
import re
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent)) 
from config import SOURCE_VIDEOS_DIR,INDEX_PATH,REPRODUCED_CLIPS_DIR,BASELINE_MODEL_OUTPUT_DIR

def reproduce_clip(raw_video_path: Path, start_sec: float, end_sec: float, output_path: Path) -> bool:
    """
    Reproduce a clip from a source video between two timestamps.

    Overview
    --------
    Opens a source `.mp4` video with OpenCV, seeks to `start_sec`, and writes
    frames until `end_sec` into a new `.mp4` file at `output_path`.

    Input/Output
    ------------
    Input
      - `raw_video_path`: Source video file.
      - `start_sec`, `end_sec`: Start/end times in seconds (float).

    Output
      - `output_path`: Written `.mp4` clip.

    Parameters
    ----------
    raw_video_path : pathlib.Path
        Path to the source `.mp4` file.
    start_sec : float
        Start time (seconds) in the source video.
    end_sec : float
        End time (seconds) in the source video.
    output_path : pathlib.Path
        Destination path for the reproduced clip.

    Returns
    -------
    bool
        `True` if the clip was written successfully; `False` if the source video
        could not be opened or frames could not be read.

    Notes
    -----
    - Uses codec `"mp4v"` via OpenCV.
    - The output frame size and FPS are taken from the source video.
    - This function does not create `output_path.parent`; callers should ensure
      the directory exists if needed.

    Examples
    --------
    >>> from pathlib import Path
    >>> ok = reproduce_clip(Path("in.mp4"), 10.0, 12.5, Path("out.mp4"))
    >>> print(ok)
    True
    """
    cap = cv2.VideoCapture(str(raw_video_path))
    if not cap.isOpened():
        print(f"Could not open: {raw_video_path.name}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

    cap.release()
    writer.release()
    return True

def split_by_index(index_path: Path, output_path: Path) -> None:
    """
    Reproduce clips defined by a CSV index file.

    Overview
    --------
    Reads a CSV index into a pandas DataFrame, matches rows against `.mp4` source
    videos found under :root:`config.SOURCE_VIDEOS_DIR`, and writes the resulting
    clips to `output_path`.

    Input/Output
    ------------
    Input
      - `index_path`: CSV file describing clips.
      - Source videos discovered recursively under :root:`config.SOURCE_VIDEOS_DIR`.

    Output
      - `.mp4` clips written under `output_path` (file name taken from the CSV).

    Parameters
    ----------
    index_path : pathlib.Path
        Path to the index CSV.
    output_path : pathlib.Path
        Directory to write reproduced clips to.

    Returns
    -------
    None

    Notes
    -----
    - The CSV is expected to include at least the columns:
      `source_video_basename`, `source_video_path`, `clip_name`,
      `clip_start_in_source_sec`, `clip_end_in_source_sec`.
    - **Current behavior**: only the first matched row is processed due to
      `matched[:1]`. Remove that slice to reproduce all matching clips.

    Raises
    ------
    FileNotFoundError
        If `index_path` does not exist (raised by pandas).
    KeyError
        If required columns are missing from the CSV.
    """
    index_df = pd.read_csv(index_path)
    # get all raw videos in RAW_DIR
    raw_videos = {f.name: f for f in SOURCE_VIDEOS_DIR.rglob("*.mp4")}

    matched = index_df[index_df["source_video_basename"].isin(raw_videos.keys())]
    print(f"Found {len(matched)} clips to reproduce from {matched['source_video_basename'].nunique()} raw videos\n")

    success = 0
    for _, row in matched[:1].iterrows():
        source_video_path = row["source_video_path"]
        source_relative_path = re.search(r"Pen.*", source_video_path).group(0)
        # normalize Windows separators -> POSIX
        source_relative_path = source_relative_path.replace("\\", "/")
        raw_path = SOURCE_VIDEOS_DIR / "videos" / source_relative_path
        save_path = output_path / row["clip_name"]
        start_sec = float(row["clip_start_in_source_sec"])
        end_sec = float(row["clip_end_in_source_sec"])

        print(f"{row['clip_name']}  {start_sec:.1f}s -> {end_sec:.1f}s")
        print(f"from: {raw_path}")
        if reproduce_clip(raw_path, start_sec, end_sec, save_path):
                print(f"saved to {save_path.name}")
                success += 1

        print(f"\nDone — {success} reproduced")
        
def split_by_json_events(json_path: Path, output_dir: Path) -> int:
    """
    Reproduce clips defined by JSON event metadata, optionally with annotations.

    Overview
    --------
    Loads one JSON metadata file or processes all `*.json` files in a directory.
    For each event (`start_sec`, `end_sec`), a clip is reproduced. If
    `annotate=True`, an additional `*_boxed.mp4` is written with bounding boxes
    overlaid.

    Input/Output
    ------------
    Input
      - `json_path`: A JSON file or a directory of JSON files.
      - Each JSON must specify `video_path` and `events`.

    Output
      - Clips written under `output_dir`.
      - If `annotate=True`, additional annotated clips are written beside the
        unboxed clips.

    Parameters
    ----------
    json_path : pathlib.Path
        A single JSON file, or a directory containing JSON files (non-recursive).
    output_dir : pathlib.Path
        Output root directory where clips will be created.
    annotate : bool, default=True
        Whether to also produce annotated clips with bounding boxes.

    Returns
    -------
    int
        Total number of successfully reproduced (unboxed) clips across all
        processed JSON files.

    Raises
    ------
    FileNotFoundError
        If a JSON references a `video_path` that does not exist.

    Notes
    -----
    Expected JSON schema (minimum):
      - `video_path` : str
      - `events` : list[dict] with keys `start_sec`, `end_sec`

    Optional keys:
      - `identifier` : str
      - `fps` : float (used for annotation alignment)
      - `events[*].intersection_box` : list[dict] with keys `frame`, `x1`, `y1`,
        `x2`, `y2` (frames are in source-video coordinates)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if json_path.is_dir():
        json_files = sorted(json_path.glob("*.json"))  # non-recursive: only this folder
        if not json_files:
            print(f"No .json files found in {json_path}")
            return 0
    else:
        json_files = [json_path]

    total_success = 0

    for jf in json_files:
        data = json.loads(jf.read_text(encoding="utf-8"))
        video_path = Path(data["video_path"])
        identifier = data.get("identifier", video_path.name)
        events = data.get("events", [])
        fps = float(data.get("fps", 30.0))

        if not video_path.exists():
            raise FileNotFoundError(f"video_path does not exist: {video_path} (from {jf})")
        if not events:
            print(f"No events found in {jf.name}")
            continue
        
        # extract the relative path after "cross_sucking_clips/" to find the raw video in RAW_DIR
        m = re.search(r"(Pen \d+ - Group \d+[\\/]\w+[\\/]Day \d+)", str(video_path))
        rel_parent = Path(m.group(1)) if m else Path()

        video_stem = Path(identifier).stem          # ch02_20250913081207
        out_folder = output_dir / rel_parent / video_stem
        
        out_folder.mkdir(parents=True, exist_ok=True)
        
        success = 0
        for i, ev in enumerate(events, start=1):
            start_sec = float(ev["start_sec"])
            end_sec = float(ev["end_sec"])

            base_name = f"{Path(identifier).stem}_event{i:03d}_{start_sec:.1f}-{end_sec:.1f}"
            out_path = out_folder / f"{base_name}.mp4"

            print(f"{out_path.name} {start_sec:.1f}s -> {end_sec:.1f}s")
            print(f"from: {video_path.name}")
            
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            
            if not reproduce_clip(video_path, start_sec, end_sec, tmp_path):
                tmp_path.unlink(missing_ok=True)
                continue

            event_start_frame = int(start_sec * fps)
            boxes = ev.get("intersection_box", [])
            clip_boxes = []
            for b in boxes:
                f = int(b["frame"]) - event_start_frame
                if f < 0:
                    continue
                clip_boxes.append({**b, "frame": f})

            boxed_path = out_folder / f"{base_name}_boxed.mp4"
            if annotate_clip_with_boxes(tmp_path, boxed_path, clip_boxes):
                print(f"saved boxed to {boxed_path.name}")
                success += 1
            tmp_path.unlink(missing_ok=True)
            
            total_success += 1

        print(f"\nDone — {success} reproduced from {jf.name}")
    return total_success

def annotate_clip_with_boxes(input_clip: Path,output_clip: Path,boxes: list[dict],color=(0, 255, 0),thickness: int = 2,) -> bool:
    """
    Annotate a clip with per-frame bounding boxes and write a new video.

    Overview
    --------
    Reads `input_clip` frame-by-frame, draws any boxes associated with the current
    frame index, and writes the resulting frames to `output_clip`.

    Input/Output
    ------------
    Input
      - `input_clip`: Existing clip to annotate.
      - `boxes`: Clip-relative frame boxes (frame 0 is the first frame of the clip).

    Output
      - `output_clip`: Annotated clip written to disk.

    Parameters
    ----------
    input_clip : pathlib.Path
        Path to the input clip (`.mp4`).
    output_clip : pathlib.Path
        Path where the annotated clip will be written.
    boxes : list[dict]
        List of bounding boxes in clip-relative frame coordinates. Each dict must
        include: `frame`, `x1`, `y1`, `x2`, `y2`.
    color : tuple[int, int, int], default=(0, 255, 0)
        Rectangle color in BGR.
    thickness : int, default=2
        Rectangle thickness (pixels).

    Returns
    -------
    bool
        `True` if annotation succeeded, else `False`.

    Notes
    -----
    - Creates `output_clip.parent` if it does not exist.
    - Supports multiple boxes per frame.
    """
    cap = cv2.VideoCapture(str(input_clip))
    if not cap.isOpened():
        print(f"Could not open clip: {input_clip}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_clip.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_clip), fourcc, fps, (w, h))

    # multiple boxes could occur in same frame; group them
    boxes_by_frame = {}
    for b in boxes:
        f = int(b["frame"])
        boxes_by_frame.setdefault(f, []).append(b)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for b in boxes_by_frame.get(frame_idx, []):
            x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return True

def run_splitting(func) -> None:
    """
    Dispatch to a chosen splitting strategy.

    Overview
    --------
    Calls either :func:`split_by_index` or :func:`split_by_json_events` using
    project-configured input/output paths from `config.py`.

    Input/Output
    ------------
    Input
      - `func`: A function object indicating which strategy to run.

    Output
      - Reproduced clips written to :root:`config.REPRODUCED_CLIPS_DIR`.

    Parameters
    ----------
    func : callable
        Either :func:`split_by_index` or :func:`split_by_json_events`.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `func` is not a supported splitting function.
    """
    if func == split_by_index:
        split_by_index(INDEX_PATH, REPRODUCED_CLIPS_DIR)
    elif func == split_by_json_events:
        split_by_json_events(BASELINE_MODEL_OUTPUT_DIR, REPRODUCED_CLIPS_DIR)
    else:
        raise ValueError(f"Unknown splitting function: {func}")
    
def main():
    """
    Run the default clipping workflow.

    Overview
    --------
    By default, runs JSON-events mode via :func:`run_splitting`.

    Input/Output
    ------------
    Input
      - JSON metadata from :root:`config.BASELINE_METADATA_DIR`.

    Output
      - Clips written to :root:`config.REPRODUCED_CLIPS_DIR`.

    Returns
    -------
    None
    """
    run_splitting(split_by_json_events)

if __name__ == "__main__":
    main()