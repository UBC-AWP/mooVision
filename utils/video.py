import cv2
import pandas as pd
from pathlib import Path


root = Path("/Users/raymondwang/Library/CloudStorage/OneDrive-SharedLibraries-UBC/Animal Welfare-mooVision - Documents/cross_sucking_clips")
test_folder = root / "Pen 2 - Group 2" / "POSTWEANING" / "test"

clips_dict = {}
def get_video_info(video_path: Path) -> tuple[int, float]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps if fps > 0 else 0
    cap.release()
    return frame_count, duration_sec

def print_clips_cal(folder, indent=0):
    prefix = "  " * indent
    print(f"{prefix}{folder.name}/")

    for f in sorted(folder.iterdir()):
        if f.is_file() and f.suffix == ".mp4":
            frame_count, duration_sec = get_video_info(f)
            clips_dict[f] = (frame_count, duration_sec)
            print(f"{prefix}  {f.name:<50} {frame_count:>6} frames   {duration_sec:.2f}s")

    for child in sorted(folder.iterdir()):
        if child.is_dir():
            print_clips_cal(child, indent + 1)

print(f"{'Video':<60} {'Frames':>6}   {'Duration':>10}")
print("-" * 80)

print_clips_cal(test_folder)

# for child in sorted(root.iterdir())[:1]:
#     if child.is_dir():
#         print_clips_cal(child)

print()
print(pd.DataFrame.from_dict(clips_dict, orient="index", columns=["frame_count", "duration_sec"]))
        

