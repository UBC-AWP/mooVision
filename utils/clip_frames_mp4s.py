import cv2
import os
from pathlib import Path

def extract_frames(video_path, num_frames=4):
    video = cv2.VideoCapture(video_path)
    
    frames = []
    frame_count = 0
    
    while frame_count < num_frames:
        success, frame = video.read()
        
        if not success:
            break
            
        frames.append(frame)
        frame_count += 1
    
    video.release()
    return frames

# Usage

path_unlabeled_example_clips = "/Users/devon/OneDrive_UBC/Animal Welfare-mooVision - Documents/15_example_cross_sucking/15_example_videos"
video = "CS_0007_PREWEAN_d3_p5_cowT_12112025_ch05-20251012134826_21696_21739.mp4"
video_path = os.path.join(path_unlabeled_example_clips, video)
frames = extract_frames(video_path, 6)

# Output Dir
output_dir = "../EDA/frames"
os.makedirs(output_dir, exist_ok=True)

# Save frames as PNG images
for i, frame in enumerate(frames):
    output_path = os.path.join(output_dir, f'frame_{i}.png')
    cv2.imwrite(output_path, frame)