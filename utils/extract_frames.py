import cv2
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent)) 

from config import LOCAL_DIR

FRAMES_DIR = LOCAL_DIR / "sample_videos" / "cross_sucking_clip_sample"

def extract_frames(video_path: Path) -> None:
    raw_videos = {f.name: f for f in video_path.rglob("*.mp4")}
    for video_name, video_path in raw_videos.items():
        print(f"\nName: {video_name}")

        clip_name = video_path.stem
        out_dir   = FRAMES_DIR / clip_name
        out_dir.mkdir(parents=True, exist_ok=True)

        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"skipping {clip_name} — already done")
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"could not open: {video_path}")
            continue

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(str(out_dir / f"frame_{frame_idx:06d}.jpg"), frame)
            frame_idx += 1
        cap.release()
        print(f"{clip_name} → {frame_idx} frames extracted")

if __name__ == "__main__":
    extract_frames(FRAMES_DIR)