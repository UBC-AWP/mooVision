import json
import numpy as np
from pathlib import Path

# Setup standard repository paths
try:
    from config import LOCAL_DIR
except ImportError:
    LOCAL_DIR = Path(".")

JSON_DIR = LOCAL_DIR / "llm_experiment" / "image_json_outputs"
FRAMES_BASE_DIR = LOCAL_DIR / "sample_videos" / "cross_sucking_clip_sample"

def parse_yolo_txt(txt_path: Path) -> list:
    """Parses a ground truth YOLO annotation file and returns a list of boxes."""
    boxes = []
    if not txt_path.exists():
        return boxes
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                boxes.append({
                    "x_center": float(parts[1]),
                    "y_center": float(parts[2]),
                    "w": float(parts[3]),
                    "h": float(parts[4])
                })
    return boxes

def evaluate_image_pipeline(gemini_detections: list, gt_txt_path: Path) -> dict:
    """Evaluates a single static frame using Containment and Centroid distance."""
    gt_boxes = parse_yolo_txt(gt_txt_path)
    has_gt = len(gt_boxes) > 0
    has_pred = len(gemini_detections) > 0 and "box_2d" in gemini_detections[0]

    # Macro Evaluation Matrix
    tp, fp, fn, tn = 0, 0, 0, 0
    if has_pred and has_gt: tp = 1
    elif has_pred and not has_gt: fp = 1
    elif not has_pred and has_gt: fn = 1
    else: tn = 1

    containment_scores = []
    centroid_errors = []

    if has_pred and has_gt:
        # Take Gemini's box and convert 0-1000 scale to standard 0.0-1.0 ratios
        pred = gemini_detections[0]["box_2d"]
        p_ymin, p_xmin, p_ymax, p_xmax = [coord / 1000.0 for coord in pred]
        p_xc = (p_xmin + p_xmax) / 2
        p_yc = (p_ymin + p_ymax) / 2

        for gt in gt_boxes:
            g_xmin = gt["x_center"] - (gt["w"] / 2)
            g_xmax = gt["x_center"] + (gt["w"] / 2)
            g_ymin = gt["y_center"] - (gt["h"] / 2)
            g_ymax = gt["y_center"] + (gt["h"] / 2)

            # Calculate Intersection Area
            int_w = max(0, min(p_xmax, g_xmax) - max(p_xmin, g_xmin))
            int_h = max(0, min(p_ymax, g_ymax) - max(p_ymin, g_ymin))
            intersection_area = int_w * int_h
            gt_area = gt["w"] * gt["h"]

            # Containment Score
            containment = intersection_area / gt_area if gt_area > 0 else 0.0
            containment_scores.append(containment)

            # Centroid Distance Error
            distance = np.sqrt((p_xc - gt["x_center"])**2 + (p_yc - gt["y_center"])**2)
            centroid_errors.append(distance)

    return {
        "confusion_matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "containment": containment_scores,
        "centroid_error": centroid_errors
    }

def main():
    json_files = list(JSON_DIR.glob("*.json"))
    if not json_files:
        print(f"No image JSON files found in {JSON_DIR}")
        return

    print(f"Evaluating {len(json_files)} static image trials...\n")
    
    global_tp, global_fp, global_fn, global_tn = 0, 0, 0, 0
    all_containments = []
    all_centroid_errors = []

    for json_path in json_files:
        # Deduce video folder and frame name from filename pattern:
        # "response_CS_0031_..._frame_000120.json"
        filename = json_path.stem.replace("response_", "") # strip "response_"
        
        # Split out the frame segment
        if "_frame_" in filename:
            video_stem, frame_part = filename.split("_frame_")
            frame_name = f"frame_{frame_part}" # "frame_000120"
            frame_name = frame_name.replace(".jpg", "") # just in case
        else:
            print(f"Could not parse pattern for filename: {json_path.name}")
            continue

        # Point directly to the ground truth .txt file inside the specific subfolder
        gt_txt_path = FRAMES_BASE_DIR / video_stem / f"{frame_name}.txt"

        if not gt_txt_path.exists():
            print(f"  [Warning] Missing matching ground truth file: {gt_txt_path.name}")
            continue

        with open(json_path, "r") as f:
            gemini_data = json.load(f)

        res = evaluate_image_pipeline(gemini_data, gt_txt_path)
        cm = res["confusion_matrix"]
        
        global_tp += cm["TP"]
        global_fp += cm["FP"]
        global_fn += cm["FN"]
        global_tn += cm["TN"]
        
        all_containments.extend(res["containment"])
        all_centroid_errors.extend(res["centroid_error"])

        print(f"File: {json_path.name}")
        print(f"  -> Match Result: {cm}")
        if res["containment"]:
            print(f"  -> Containment (Coverage) : {res['containment'][0]:.1%}")
            print(f"  -> Centroid Distance Error: {res['centroid_error'][0]:.4f}")

    # Final Overall Printout
    print("\n" + "="*50)
    print("         STATIC IMAGE PIPELINE FINAL RESULTS      ")
    print("="*50)
    precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
    
    print(f"Aggregated Matrix -> TP: {global_tp}, FP: {global_fp}, FN: {global_fn}")
    print(f"Image Precision   -> {precision:.3f}")
    print(f"Image Recall      -> {recall:.3f}")
    print(f"Mean Containment Score (Coverage)  -> {np.mean(all_containments):.1%}")
    print(f"Mean Centroid Distance Error       -> {np.mean(all_centroid_errors):.4f}")
    print("="*50)

if __name__ == "__main__":
    main()