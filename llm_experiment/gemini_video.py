import os
import sys
import json
import time
import cv2
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

try:
    from google import genai
    from google.genai import types
except ImportError:
    sys.exit("[error] google-genai not installed. Run: uv add google-genai")
    
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize the Gemini Client
client = genai.Client(api_key=GEMINI_API_KEY)

# CS prompt
CROSS_SUCKING_VIDEO_PROMPT = (
    "You are an expert in dairy cattle behaviour. Identify cross-sucking "
    "in this video.\n\n"
    "Cross-sucking is defined narrowly here: one calf sucking on the "
    "hind/rear back, udder, or lower stomach/belly area of another calf. "
    "It requires visible oral contact between one calf's mouth and that "
    "hindquarter region of another calf. Do NOT count sucking on any "
    "other body part (ears, muzzle, legs, etc.), and do NOT count mere "
    "proximity, sniffing, head-butting, or social licking.\n\n"
    "Analyse the video systematically. For every timestamp where cross-sucking "
    "occurs, return a 2D bounding box that encloses the event.\n\n"
    "Respond ONLY with a JSON array — no markdown, no explanation. Each element MUST include the timestamp:\n"
    "[\n"
    "  {\n"
    '    "timestamp_sec": <float, seconds into the video>,\n'
    '    "label": "cross-sucking",\n'
    '    "box_2d": [ymin, xmin, ymax, xmax]\n'
    "  }\n"
    "]\n\n"
    "If cross-sucking does not occur anywhere in the video, return []."
)

def draw_boxes_and_clip(raw_video_path: Path, api_output_json: list, output_path: Path) -> bool:
    """
    Parses the Gemini JSON tracking list, seeking to the clip bounds, 
    and draws bounding box markers onto frames before exporting to the current folder.
    """
    if not api_output_json:
        print(f"Skipping video production: No cross-sucking detections returned for {raw_video_path.name}")
        return False

    # Extract timeline endpoints from JSON data metrics
    timestamps = [item["timestamp_sec"] for item in api_output_json]
    start_sec = min(timestamps)
    # Add an extra 1-second pad to capture the tail end of the last flag segment
    end_sec = max(timestamps) + 1.0 

    # Create a quick key-lookup table for frame matching
    box_lookup = {int(item["timestamp_sec"]): item["box_2d"] for item in api_output_json}

    cap = cv2.VideoCapture(str(raw_video_path))
    if not cap.isOpened():
        print(f"Could not open video file: {raw_video_path.name}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    # Move capture playhead straight to the event start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print(f"Drawing frames from {start_sec}s to {end_sec}s...")
    for current_frame_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
            
        current_second = int(current_frame_idx // fps)
        
        # Overlay box if coordinates match current time slice
        if current_second in box_lookup:
            ymin_norm, xmin_norm, ymax_norm, xmax_norm = box_lookup[current_second]
            
            # Map normalized 0-1000 grid space to real clip resolutions
            xmin = int((xmin_norm / 1000) * width)
            ymin = int((ymin_norm / 1000) * height)
            xmax = int((xmax_norm / 1000) * width)
            ymax = int((ymax_norm / 1000) * height)
            
            # Draw a thick red warning border (BGR: 0, 0, 255)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
            cv2.putText(frame, "cross-sucking", (xmin, ymin - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Successfully generated labeled asset: {output_path.name}")
    return True

# Upload the video using the Files API
EXAMPLE_VIDEOS_DIR = Path("../sample_videos/cross_sucking_clip_sample")
raw_videos = {f.name: f for f in EXAMPLE_VIDEOS_DIR.rglob("*.mp4")}
for video_name, video_path in raw_videos.items():
    print(f"Name: {video_name}, Path: {video_path}")
    
    video_file = client.files.upload(file=video_path)

    # Wait for Google backend to process the uploaded video frames
    while video_file.state.name == "PROCESSING":
        print("Waiting for video processing...")
        time.sleep(5)
        video_file = client.files.get(name=video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError(f"Video processing failed: {video_file.error.message}")

    print("Video successfully processed and ready.")

    # Request the model to analyze and output raw JSON
    print("Analyzing video tracking coordinates...")
    response = client.models.generate_content(
        model="gemini-2.5-flash", 
        contents=[video_file, CROSS_SUCKING_VIDEO_PROMPT],
        config=types.GenerateContentConfig(
            # This restricts Gemini to only output valid JSON code strings
            response_mime_type="application/json",
            temperature=0.1, 
        ),
    )

    # View and load results
    print("\n--- RAW TEXT RESPONSE FROM GEMINI ---")
    print(response.text)
    
    # SAFE JSON PARSING & RENDERING LAYER
    try:
        parsed_json_data = json.loads(response.text)
        
        # Direct output path file name structure targets current execution folder
        output_clip_name = Path(f"labeled_{video_path.stem}.mp4")
        
        draw_boxes_and_clip(
            raw_video_path=video_path,
            api_output_json=parsed_json_data,
            output_path=output_clip_name
        )
        
    except json.JSONDecodeError:
        print(f"[Warning] Response for {video_name} was not valid JSON. Skipping draw step.")

