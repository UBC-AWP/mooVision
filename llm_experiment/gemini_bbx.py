import os
import sys
import time
import json
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
CROSS_SUCKING_PROMPT = (
    "You are an expert in dairy cattle behaviour. Identify cross-sucking "
    "in the video.\n\n"
    "Cross-sucking is defined narrowly here: one calf sucking on the "
    "hind/rear back, udder, or lower stomach/belly area of another calf. "
    "It requires visible oral contact between one calf's mouth and that "
    "hindquarter region of another calf. \n\n"
    "Respond ONLY with a JSON array — no markdown, no explanation. Each element:\n"
    "{\n"
    '  "timestamp_sec": <float, seconds into the video>,\n'
    '  "label": "cross-sucking",\n'
    '  "box_2d": [ymin, xmin, ymax, xmax]  // integers 0-1000, origin top-left\n'
    "}\n\n"
    "If cross-sucking does not clearly occur anywhere in the video, return []. "
    "Do not guess — report only clearly visible oral contact."
)

# Upload the video using the Files API
video_path = "/sample_videos"
for video in video_path:
    print("Uploading video to Gemini API...")
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
        contents=[video_file, CROSS_SUCKING_PROMPT],
        config=types.GenerateContentConfig(
            # This restricts Gemini to only output valid JSON code strings
            response_mime_type="application/json",
            temperature=0.1, 
        ),
    )

    # View and load results
    print("\n--- RAW TEXT RESPONSE FROM GEMINI ---")
    print(response.text)

