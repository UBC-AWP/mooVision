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
    Handles coordinate translation and overlays bounding boxes onto the target frame.
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
    EXAMPLE_VIDEOS_DIR = LOCAL_DIR / "sample_videos" / "cross_sucking_clip_sample"
    raw_videos = {f.name: f for f in EXAMPLE_VIDEOS_DIR.rglob("*.mp4")}

    for video_name, video_path in raw_videos.items():
        folder_path = EXAMPLE_VIDEOS_DIR / video_path.stem
        
        if folder_path.is_dir():
            print(f"\n{video_name} → Found folder")
        else:
            print(f"{folder_path} → No matching folder")
            print(f"\nCreating folder: {folder_path}")
            extract_frames(EXAMPLE_VIDEOS_DIR)
            
        target_images = list(folder_path.glob("*.jpg"))[:3]
            
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
            
            # SAFE JSON PARSING LAYER IN MAIN LOOP
            try:
                parsed_json_data = json.loads(response.text)
                
                draw_boxes_and_labels(parsed_json_data, image_path, video_path)
                
            except json.JSONDecodeError:
                print(f"[Warning] Response for {video_name} was not valid JSON. Skipping draw step.")
                
if __name__ == "__main__":   
    main()