import os
import sys
import json
import time
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
sys.path.append(str(Path(__file__).parent.parent)) 

from config import LOCAL_DIR
from utils.extract_frames import extract_frames

load_dotenv()

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

# Open the image file locally using PIL
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
        
    for image_path in list(folder_path.glob("*.jpg"))[:5]:
        print(f"\nProcessing image: {image_path}")
        time.sleep(5)
        pil_image = Image.open(image_path)
        print(f"\nProcessing image: {image_path}")
        print("Sending static image to Gemini API...")
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            # Pass the PIL Image directly in the contents array alongside the prompt
            contents=[pil_image, IMAGE_CROSS_SUCKING_PROMPT],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,  # Forces deterministic choices for testing
            ),
        )

        print("\n--- RAW TEXT RESPONSE FROM GEMINI ---")
        print(response.text)

        # # 3. Parse and Draw Bounding Boxes using OpenCV
        # try:
        #     detections = json.loads(response.text)
        #     print(f"\nSuccessfully parsed JSON. Detected objects: {len(detections)}")
            
        #     # Read the image via OpenCV to handle geometric drawing additions
        #     cv_img = cv2.imread(image_path)
        #     height, width, _ = cv_img.shape
            
        #     for item in detections:
        #         if "box_2d" in item:
        #             ymin_norm, xmin_norm, ymax_norm, xmax_norm = item["box_2d"]
                    
        #             # Map normalized scale [0-1000] to raw pixel dimensions
        #             xmin = int((xmin_norm / 1000) * width)
        #             ymin = int((ymin_norm / 1000) * height)
        #             xmax = int((xmax_norm / 1000) * width)
        #             ymax = int((ymax_norm / 1000) * height)
                    
        #             # Draw standard red rectangle marker
        #             cv2.rectangle(cv_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
                    
        #             # Write identifying string flag
        #             label_text = item.get("label", "cross-sucking")
        #             cv2.putText(cv_img, label_text, (xmin, ymin - 10), 
        #                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
        #     # Save the output file locally
        #     output_path = "output_labeled_image.jpg"
        #     cv2.imwrite(output_path, cv_img)
        #     print(f"Success! Annotated image saved as '{output_path}'.")

        # except json.JSONDecodeError:
        #     print("\nFailed to parse text as JSON structure.")