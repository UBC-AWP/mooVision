import os
import sys
import json
import time
import cv2
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
sys.path.append(str(Path(__file__).parent.parent)) 

from config import LOCAL_DIR
from utils.extract_frames import extract_frames

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

# Adjusted prompt specifically for static image analysis
IMAGE_CROSS_SUCKING_PROMPT = (
    "You are an expert in dairy cattle behaviour. Identify cross-sucking "
    "in this image.\n\n"
    "Cross-sucking is defined narrowly here: one calf sucking on the "
    "hind/rear back, udder, or lower stomach/belly area of another calf. "
    "It requires visible oral contact between one calf's mouth and that "
    "hindquarter region of another calf. Do NOT count sucking on any "
    "other body part (ears, muzzle, legs, etc.), and do NOT count mere "
    "proximity, sniffing, head-butting, or social licking.\n\n"
    "If cross-sucking clearly occurs, return one tight 2D bounding box that encloses both "
    "the calf doing the sucking AND the hind/udder/stomach area being sucked, "
    "with the label 'cross-sucking'.\n\n"
    "Respond ONLY with a JSON array — no markdown, no explanation. Each element:\n"
    "{\n"
    '  "label": "cross-sucking",\n'
    '  "box_2d": [ymin, xmin, ymax, xmax]  // integers 0-1000, origin top-left\n'
    "}\n\n"
    "If cross-sucking does not clearly occur in the image, return []."
)

def draw_boxes_and_labels(parsed_detections, image_path, video_path):
    """
    Translate Gemini relative coordinates and draw bounding boxes on a local frame.

    Takes a parsed JSON array containing 0-1000 normalized spatial anchors, 
    de-normalizes them back to the raw image pixel dimensions using OpenCV, 
    and exports a target visual copy labeled with the original source context.

    Parameters
    ----------
    parsed_detections : list of dict
        A list of detection objects returned by the Gemini API. Each dict 
        should contain a 'box_2d' key with a list of 4 normalized integers 
        [ymin, xmin, ymax, xmax] and an optional 'label' string.
    image_path : pathlib.Path
        The exact file path pointing to the original input source frame.
    video_path : pathlib.Path
        The original video source path used to establish structural prefixes 
        for exported visual files.

    Returns
    -------
    None
        Saves the annotated image copy directly to the workspace directory.
    """
    try:
        if not parsed_detections:
            print("No cross-sucking detected in this frame.")
            return 
            
        print(f"Detected {len(parsed_detections)} event(s). Drawing boxes...")
        
        cv_img = cv2.imread(str(image_path))
        height, width, _ = cv_img.shape
        
        for item in parsed_detections:
            if "box_2d" in item:
                ymin_norm, xmin_norm, ymax_norm, xmax_norm = item["box_2d"]
                
                xmin = int((xmin_norm / 1000) * width)
                ymin = int((ymin_norm / 1000) * height)
                xmax = int((xmax_norm / 1000) * width)
                ymax = int((ymax_norm / 1000) * height)
                
                # Draw Box
                cv2.rectangle(cv_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
                
                # Draw Label text
                label = item.get("label", "cross-sucking")
                cv2.putText(cv_img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Ensure target subdirectory exists before writing
        output_dir = LOCAL_DIR / "llm_experiment"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{video_path.stem}_{image_path.stem}_labeled.jpg"
        cv2.imwrite(str(output_path), cv_img)
        print(f"Saved annotated image to: {output_path.name}")
        
    except Exception as e:
        print(f"[Error] Failed to draw boxes on image {image_path.name}: {e}")

def main():      
    """
    Execute sequential frame extraction analysis pipelines across cattle footage.

    Loops through detected directory arrays, extracts targets chronologically, 
    and handles rate-limited payloads while managing error-handling boundaries.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Prints execution logs to stdout and delegates image writing tasks 
        to `draw_boxes_and_labels`.
    """ 
    EXAMPLE_VIDEOS_DIR = LOCAL_DIR / "sample_videos" / "cross_sucking_clip_sample"
    json_output_dir = LOCAL_DIR / "llm_experiment" / "image_json_outputs"
    json_output_dir.mkdir(parents=True, exist_ok=True)
    
    raw_videos = {f.name: f for f in EXAMPLE_VIDEOS_DIR.glob("*.mp4")}

    for video_name, video_path in raw_videos.items():
        folder_path = EXAMPLE_VIDEOS_DIR / video_path.stem
        
        if folder_path.is_dir():
            print(f"\n{video_name} → Found folder")
        else:
            print(f"{folder_path} → No matching folder")
            print(f"\nCreating folder: {folder_path}")
            extract_frames(EXAMPLE_VIDEOS_DIR)
            
        target_images = list(folder_path.glob("*.jpg"))[:5]
            
        for image_path in target_images:
            print(f"\nPacing delay for 5 RPM limit...")
            time.sleep(13)
            
            try:
                pil_image = Image.open(image_path)
            except Exception as e:
                print(f"Could not open image {image_path.name}: {e}")
                continue
                
            print(f"Processing image: {image_path.name}")
            print("Sending static image to Gemini API...")
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[pil_image, IMAGE_CROSS_SUCKING_PROMPT],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1, 
                ),
            )

            print("\n--- RAW TEXT RESPONSE FROM GEMINI ---")
            print(response.text)
            
            json_file_path = json_output_dir / f"{video_path.stem}_{image_path.name}.json"
            try:
                # Parse to validate it's real JSON, then pretty-print save it
                json_data = json.loads(response.text)
                with open(json_file_path, "w") as jf:
                    json.dump(json_data, jf, indent=2)
                print(f"Saved tracking coordinates to: {json_file_path.name}")
                
                # Draw boxes using the successfully parsed object
                draw_boxes_and_labels(json_data, image_path, video_path)
                
            except json.JSONDecodeError:
                print(f"[Warning] Response was not valid JSON. Saving raw text as fallback.")
                with open(json_file_path.with_suffix(".txt"), "w") as tf:
                    tf.write(response.text)
                
if __name__ == "__main__":   
    main()