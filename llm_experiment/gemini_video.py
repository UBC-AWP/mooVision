import os
import sys
import json
import time
import cv2
from pathlib import Path
from dotenv import load_dotenv
sys.path.append(str(Path(__file__).parent.parent)) 

from config import LOCAL_DIR

# Clear out any old environment artifacts in Python memory first
if "GEMINI_API_KEY" in os.environ: del os.environ["GEMINI_API_KEY"]

load_dotenv(override=True) # Forces Python to overwrite cached keys with .env updates

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
    Draw bounding boxes on detected cross-sucking events and export clipped video.

    Parses Gemini API JSON tracking data to identify cross-sucking event timestamps,
    extracts the relevant video segment with temporal padding, and renders bounding
    box annotations onto frames before writing the annotated clip to disk.

    Parameters
    ----------
    raw_video_path : Path
        Path to the source video file to be processed.
    api_output_json : list
        List of detection dictionaries from Gemini API, each containing:
        - "timestamp_sec" (float): Time in seconds when detection occurred
        - "box_2d" (tuple): Normalized bounding box coordinates (ymin, xmin, ymax, xmax)
          in 0-1000 grid space
    output_path : Path
        Path where the annotated video clip will be saved (MP4 format).

    Returns
    -------
    bool
        True if video was successfully processed and written to disk.
        False if no detections were found or video could not be opened.

    Raises
    ------
    cv2.error
        If the video file is corrupted or cannot be decoded.
    IOError
        If the output path is not writable.

    Notes
    -----
    - Bounding box coordinates are normalized to a 0-1000 grid and must be scaled
      to match the actual video resolution
    - A 1-second temporal pad is added after the last detection to capture trailing frames
    - The video playhead is seeked directly to the start frame for efficiency
    - Bounding boxes are drawn in red (BGR: 0, 0, 255) with 3-pixel thickness
    - A text label "cross-sucking" is overlaid above each bounding box
    - Output video uses MP4V codec and maintains original FPS and resolution

    Examples
    --------
    >>> from pathlib import Path
    >>> video_path = Path("sample_video.mp4")
    >>> detections = [
    ...     {"timestamp_sec": 5.2, "box_2d": (100, 150, 300, 400)},
    ...     {"timestamp_sec": 7.1, "box_2d": (120, 160, 320, 420)}
    ... ]
    >>> output = Path("output_annotated.mp4")
    >>> success = draw_boxes_and_clip(video_path, detections, output)
    >>> if success:
    ...     print(f"Annotated video saved to {output}")
    Drawing frames from 5.2s to 8.1s...
    Successfully generated labeled asset: output_annotated.mp4

    See Also
    --------
    cv2.VideoCapture : For reading video files
    cv2.VideoWriter : For writing video files
    cv2.rectangle : For drawing bounding boxes
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

def main():
    """
    Process cross-sucking detection in videos using Gemini API and generate annotated clips.

    Orchestrates the complete video analysis pipeline:
    1. Discovers all MP4 videos in the sample directory
    2. Uploads each video to Google's Files API for processing
    3. Sends processed videos to Gemini 2.5 Flash for cross-sucking detection analysis
    4. Parses JSON response containing bounding box coordinates and timestamps
    5. Renders annotated video clips with detection overlays

    Returns
    -------
    None
        This function executes the pipeline and writes annotated videos to disk.

    Raises
    ------
    ValueError
        If video processing fails on the Google backend.
    json.JSONDecodeError
        If Gemini response cannot be parsed as valid JSON (handled gracefully with warning).
    FileNotFoundError
        If EXAMPLE_VIDEOS_DIR does not exist or contains no MP4 files.
    google.api_core.exceptions.GoogleAPIError
        If API calls to Google Files API or Gemini fail.

    Notes
    -----
    - Videos are uploaded to Google's Files API and processed asynchronously
    - Processing status is polled every 5 seconds until completion
    - Gemini model is configured with temperature=0.1 for deterministic JSON output
    - Response MIME type is restricted to "application/json" to ensure valid JSON
    - Output videos are saved to the current execution directory as "labeled_{video_stem}.mp4"
    - Failed JSON parsing is logged as a warning but does not halt execution
    - Requires global variables: LOCAL_DIR, CROSS_SUCKING_VIDEO_PROMPT, client
    - Requires: google.generativeai, pathlib, json, time

    Examples
    --------
    >>> # Assuming credentials and prompts are configured
    >>> main()
    Name: sample_video.mp4, Path: /path/to/sample_video.mp4
    Waiting for video processing...
    Video successfully processed and ready.
    Analyzing video tracking coordinates...
    
    --- RAW TEXT RESPONSE FROM GEMINI ---
    [{"timestamp_sec": 5.2, "box_2d": [100, 150, 300, 400]}, ...]
    
    Drawing frames from 5.2s to 6.2s...
    Successfully generated labeled asset: labeled_sample_video.mp4

    See Also
    --------
    draw_boxes_and_clip : Renders bounding boxes and exports annotated video
    client.files.upload : Uploads files to Google Files API
    client.models.generate_content : Sends content to Gemini for analysis
    """
    # Upload the video using the Files API
    EXAMPLE_VIDEOS_DIR = LOCAL_DIR / "sample_videos" / "cross_sucking_clip_sample"
    
    json_output_dir = LOCAL_DIR / "llm_experiment" / "video_json_outputs"
    json_output_dir.mkdir(parents=True, exist_ok=True)
    
    raw_videos = {f.name: f for f in EXAMPLE_VIDEOS_DIR.glob("*.mp4")}
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
        
        # Save the raw JSON response to disk for record-keeping and debugging
        json_file_path = json_output_dir / f"response_labeled_{video_path.stem}.json"

        try:
            parsed_json_data = json.loads(response.text)
            with open(json_file_path, "w") as jf:
                json.dump(parsed_json_data, jf, indent=2)
            print(f"Saved tracking array timeline data to: {json_file_path.name}")
            
            output_clip_name = Path(f"labeled_{video_path.stem}.mp4")
            draw_boxes_and_clip(
                raw_video_path=video_path,
                api_output_json=parsed_json_data,
                output_path=output_clip_name
            )
            
        except json.JSONDecodeError:
            print(f"[Warning] Response for {video_name} was not valid JSON. Writing as pure log text.")
            with open(json_file_path.with_suffix(".txt"), "w") as tf:
                tf.write(response.text)

if __name__ == "__main__":
    main()