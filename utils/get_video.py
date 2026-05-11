import cv2
import time
from pathlib import Path
import pandas as pd

def get_video_info(video_path: Path, max_retries: int = 3, wait_sec: int = 1) -> dict | None:
    """Return basic video stats (fps, frame_count, duration) plus path-derived fields.

    Retries a few times because OpenCV can intermittently fail to read some files.
    Expects path layout: .../<pen>/<weaning_stage>/<day>/<file>.mp4
    """
    for attempt in range(1, max_retries + 1):
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if fps > 0 and frame_count > 0:
            duration_sec = frame_count / fps
            return {
                "file_name": video_path.name,
                "relative_path": str(video_path),
                "pen": video_path.parts[-4],
                "weaning_stage": video_path.parts[-3],
                "day": video_path.parts[-2],
                "fps": fps,
                "frame_count": frame_count,
                "duration_sec": round(duration_sec, 2),
            }

        print(f"attempt {attempt}/{max_retries} failed: {video_path.name} — retrying in {wait_sec}s...")
        time.sleep(wait_sec)

    print(f"gave up after {max_retries} attempts: {video_path.name}")
    return None

def collect_all_metadata(root: Path) -> tuple[pd.DataFrame, list[Path]]:
    """Scan `root` for Pen-related .mp4 files and return (metadata_df, failed_paths)."""
    records = []
    failed = []
    # all files in the directory and subdirectories with .mp4 extension
    # all_files = sorted(root.rglob("*.mp4"))
    
    all_files = sorted(
    f for f in root.rglob("*.mp4")
    if any("Pen" in part for part in f.parts)
    )
    total = len(all_files)

    for i, f in enumerate(all_files, 1):
        print(f"[{i}/{total}] {f.name}")
        info = get_video_info(f)
        if info:
            records.append(info)
        else:
            failed.append(f)

    return pd.DataFrame(records), failed