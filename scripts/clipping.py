import sys
import cv2
import time
import re
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent)) 
from config import ROOT, RAW_DIR, INDEX_PATH

OUTPUT_DIR = ROOT.parent / "reproduced_clips"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def reproduce_clip(raw_video_path: Path, start_sec: float, end_sec: float, output_path: Path) -> bool:
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

def main():
    index_df = pd.read_csv(INDEX_PATH)

    # get all raw videos in RAW_DIR
    raw_videos = {f.name: f for f in RAW_DIR.rglob("*.mp4")}

    matched = index_df[index_df["source_video_basename"].isin(raw_videos.keys())]
    print(f"Found {len(matched)} clips to reproduce from {matched['source_video_basename'].nunique()} raw videos\n")

    success = 0
    for _, row in matched[:1].iterrows():
        # source_video_path = row["source_video_path"]
        # source_realtive_path = re.search(r"Pen.*", source_video_path).group(0)
        # raw_path = RAW_DIR / "videos"/ source_realtive_path
        output_path = OUTPUT_DIR / row["clip_name"]
        start_sec = float(row["clip_start_in_source_sec"])
        end_sec = float(row["clip_end_in_source_sec"])

        print(f"[{row['clip_name']}]  {start_sec:.1f}s → {end_sec:.1f}s")
        raw_path = Path("/Users/raymondwang/Library/CloudStorage/OneDrive-SharedLibraries-UBC/Animal Welfare-mooVision - Documents/raw_cross_sucking_datalog/videos/Pen 2 - Group 2/POSTWEANING/Day 1/ch02_20251102075200.mp4")
        print(f"from: {raw_path}")
        if reproduce_clip(raw_path, start_sec, end_sec, output_path):
                print(f"saved to {output_path.name}")
                success += 1

        print(f"\nDone — {success} reproduced")

if __name__ == "__main__":
    main()