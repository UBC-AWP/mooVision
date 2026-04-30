import cv2
from pathlib import Path
import pandas as pd

def get_video_info(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps if fps > 0 else 0
    cap.release()

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

def collect_all_metadata(root: Path) -> pd.DataFrame:
    records = []
    for f in sorted(root.rglob("*.mp4")):
        if "Pen" in f.parts:
            records.append(get_video_info(f))
    return pd.DataFrame(records)