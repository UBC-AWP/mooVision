"""
scripts.clipping.clpipping
================

Reproduce short annotated video clips from longer source videos.

Overview
--------
This module provides a pipeline for reproducing clip video segments from
source `.mp4` files using JSON event metadata produced by any stage of the
detection pipeline (baseline, fine-tuned YOLO, or sequence-linked).

For each event in each JSON, a clip is reproduced from the source video and
an annotated ``*_boxed.mp4`` is written with per-frame bounding boxes overlaid.

Input/Output
------------
Inputs
  - Source videos (``*.mp4``) resolved via ``ROOT_DIR`` in ``config.py``.
  - A single JSON file, or a directory searched recursively for ``*.json``
    metadata files, passed via ``--input`` on the command line.

Outputs
  - Annotated clips written under ``results/result_clips/``, preserving
    everything after ``metadata/`` in the input JSON path, with one
    subfolder per video named after its ``identifier`` stem::

        results/result_clips/<split_name>/<model_name>/<identifier>/
            <identifier>_event001_<start>-<end>_boxed.mp4
            <identifier>_event002_<start>-<end>_boxed.mp4

    Examples:
        input  → results/metadata/pipeline_demo/yolo/ch02_....json
        output → results/result_clips/pipeline_demo/yolo/ch02_.../ch02_...mp4

        input  → results/metadata/pipeline_demo/baseline/ch02_....json
        output → results/result_clips/pipeline_demo/baseline/ch02_.../ch02_...mp4

Notes
-----
  - Relies on OpenCV for video I/O.
  - Configuration is centralised in ``config.py``, driven by environment
    variables in a ``.env`` file at the repo root.
  - If annotation fails for an event, the unboxed intermediate clip is kept
    as a fallback and a warning is printed.
"""

import sys
import cv2
import tempfile
import argparse
import json
import re
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent)) 
from config import RESULT_CLIPS_DIR,ROOT_DIR

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
        
def split_by_json_events(json_path: Path, output_dir: Path) -> int:
    """
    Reproduce annotated clips defined by JSON event metadata files.

    Overview
    --------
    Recursively searches `json_path` for `*.json` metadata files produced by the
    detection pipeline. For each event in each JSON, reproduces a clip from the
    source video and writes an annotated `*_boxed.mp4` with bounding boxes
    overlaid. The output directory structure is derived entirely from the JSON
    file's path relative to the `metadata/` folder, so it mirrors the input
    hierarchy regardless of how deeply nested the JSON files are.

    Input/Output
    ------------
    Input
      - `json_path`: A single JSON file, or a directory searched recursively
                     for `*.json` files produced by the detection pipeline.
      - Each JSON must specify `video_path` and `events`, and must live
        somewhere under a `metadata/` directory.

    Output
      - Annotated clips written under:
            <output_dir>/<path_after_metadata>/<identifier_stem>/<stem>_event###_<start>-<end>_boxed.mp4

        Examples:
            input  → results/metadata/pipeline_demo/yolo/ch02_....json
            output → results/result_clips/pipeline_demo/yolo/ch02_.../

            input  → results/metadata/baseline/pipeline_demo/Pen 2/PREWEANING/Day 1/ch02_....json
            output → results/result_clips/baseline/pipeline_demo/Pen 2/PREWEANING/Day 1/ch02_.../

    Parameters
    ----------
    json_path : pathlib.Path
        A single JSON file, or a root directory to search recursively for JSON files.
    output_dir : pathlib.Path
        Output root directory (e.g. RESULT_CLIPS_DIR).

    Returns
    -------
    int
        Total number of successfully reproduced clips across all JSON files.

    Raises
    ------
    FileNotFoundError
        If a JSON references a `video_path` that does not exist on disk.

    Notes
    -----
    Expected JSON schema (minimum):
      - `video_path` : str
      - `events`     : list[dict] with keys `start_sec`, `end_sec`

    Optional keys used if present:
      - `identifier`                 : str — used as the per-video output folder name
      - `fps`                        : float — used for annotation frame alignment
      - `events[*].intersection_box` : list[dict] with keys `frame`, `x1`, `y1`,
                                       `x2`, `y2` in source-video frame coordinates
    """
    if json_path.is_dir():
        json_files = sorted(json_path.rglob("*.json"))  # recursive search for JSON files
        if not json_files:
            print(f"No .json files found in {json_path}")
            return 0
    else:
        json_files = [json_path]

    total_success = 0

    for jf in json_files:
        data = json.loads(jf.read_text(encoding="utf-8"))
        video_path = Path(data["video_path"])
        p = re.search(r"raw_cross_sucking_datalog[\\/](.*)$", str(video_path))
        video_path = ROOT_DIR / "raw_cross_sucking_datalog" / p.group(1) if p else video_path
        identifier = data.get("identifier", video_path.name)
        events = data.get("events", [])
        fps = float(data.get("fps", 30.0))
        print(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"video_path does not exist: {video_path} (from {jf})")
        if not events:
            print(f"No events found in {jf.name}")
            continue
        
        # extract split name from JSON path 
        m_meta = re.search(r"metadata[\\/](.+)$", str(jf.parent))
        rel_from_metadata = Path(m_meta.group(1)) if m_meta else Path()
        video_stem = Path(identifier).stem
        out_folder = output_dir / rel_from_metadata / video_stem
                        
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

            # convert source-video frames -> clip frames
            clip_boxes = []
            for b in boxes:
                f = int(b["frame"]) - event_start_frame
                if f < 0:
                    continue
                clip_boxes.append({**b, "frame": f})

            boxed_path = out_folder / f"{base_name}_boxed.mp4"
            if annotate_clip_with_boxes(tmp_path, boxed_path, clip_boxes):
                print(f"saved boxed to {boxed_path.name}")
                if out_path.exists():
                    out_path.unlink()
            else:
                print(f"Warning: Failed to annotate {out_path.name}, keeping unboxed version.")
                tmp_path.rename(out_path)
                
            tmp_path.unlink(missing_ok=True)
            success += 1
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
    
def main():
    """
    Run the clipping pipeline from the command line.

    Overview
    --------
    Accepts a JSON file or directory of JSON files as input and reproduces
    annotated clips for all detected events, written under the output directory
    mirroring the source video hierarchy.

    Input/Output
    ------------
    Input
      - ``--input``  : A single JSON file or a directory searched recursively
                       for ``*.json`` files produced by the detection pipeline.

    Output
      - Annotated clips written under:
            <output>/<split_name>/<model_name>/<stem>/<stem>_event###_<start>-<end>_boxed.mp4
      - ``--output`` defaults to ``RESULT_CLIPS_DIR`` (``results/result_clips``).

    Parameters
    ----------
    None
        Arguments are parsed from the command line:
          --input   Path  Required. JSON file or directory of JSON files.
          --output  Path  Optional. Output root. Default: RESULT_CLIPS_DIR.

    Returns
    -------
    None

    Examples
    --------
    Clip all events from a directory of JSONs::

        uv run scripts/clipping/clipping.py --input results/metadata/<split_name>/<model_name>

    Clip events from a single JSON file::

        uv run scripts/clipping/clipping.py --input results/metadata/pen_based/pen_3/yolo

    Clip to a custom output directory::

        uv run scripts/clipping/clipping.py  --input results/metadata/<split_name>/<model_name>/ --output /tmp/clips/
    """
    parser = argparse.ArgumentParser(description="Clip events from prediction JSONs.")
    parser.add_argument("--input", type=Path, required=True,
                        help="JSON file or directory of JSON files.")
    parser.add_argument("--output", type=Path, default=RESULT_CLIPS_DIR,
                        help="Output root directory. Default: results/result_clips")
    args = parser.parse_args()
    split_by_json_events(ROOT_DIR / args.input, args.output)
    
if __name__ == "__main__":
    main()