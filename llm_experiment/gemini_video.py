import os
import sys
import time
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

