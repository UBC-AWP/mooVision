import cv2
from pathlib import Path


root = Path("/Users/raymondwang/Library/CloudStorage/OneDrive-SharedLibraries-UBC/Animal Welfare-mooVision - Documents/cross_sucking_clips")
test_video = root / "Pen 2 - Group 2" / "POSTWEANING" / "Day 1" / "CS_0001_POSTWEAN_d1_p2_cow6_02112025_ch02-20251102075200_684_702.mp4"

def get_video_info(video_path:Path) -> tuple[int, float]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps if fps > 0 else 0
    cap.release()
    return frame_count, duration_sec

frame_count, duration_sec = get_video_info(test_video)
print(f"Frame Count: {frame_count}")
print(f"Duration (seconds): {duration_sec:.2f}")