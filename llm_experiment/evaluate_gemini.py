import json
import numpy as np
from pathlib import Path

# Placeholder configuration fallback to make code completely runnable standalone
try:
    from config import LOCAL_DIR
except ImportError:
    LOCAL_DIR = Path(".")

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

def calculate_metrics(tp: int, fp: int, fn: int, beta: float = 2.0) -> dict:
    """Calculates standard Precision, Recall, F1, and F_beta scores safely."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    beta_sq = beta ** 2
    f_beta = ((1 + beta_sq) * precision * recall) / ((beta_sq * precision) + recall) if ((beta_sq * precision) + recall) > 0 else 0.0
    
    return {"precision": precision, "recall": recall, "f1": f1, f"f{beta}": f_beta}

def evaluate_video_pipeline(gemini_video_json: list, frames_folder_path: Path, source_fps: float) -> dict:
    """
    Evaluates Gemini's video response by checking frame directories frame-by-frame.
    Maps Gemini timeline timestamps directly back onto the sorted YOLO ground-truth files.
    """
    # Safe lookup generation to handle empty arrays or missing tracking keys
    gemini_lookup = {}
    for item in gemini_video_json:
        if isinstance(item, dict) and "timestamp_sec" in item and "box_2d" in item:
            gemini_lookup[int(float(item["timestamp_sec"]))] = item["box_2d"]
    
    # Gather extracted chronological directory items
    sorted_gt_files = sorted(list(frames_folder_path.glob("frame_*.txt")))
    
    if not sorted_gt_files:
        print(f"[Warning] No frame text files discovered in path: {frames_folder_path}")
        return None

    v_tp, v_fp, v_fn, v_tn = 0, 0, 0, 0
    all_containments = []

    for frame_idx, gt_file in enumerate(sorted_gt_files):
        # Determine what whole second this exact frame index maps back onto
        current_second = int(frame_idx // source_fps)
        
        gt_boxes = parse_yolo_txt(gt_file)
        has_gt = len(gt_boxes) > 0
        has_pred = current_second in gemini_lookup

        # Classification Matrix Assignment
        if has_pred and has_gt: v_tp += 1
        elif has_pred and not has_gt: v_fp += 1
        elif not has_pred and has_gt: v_fn += 1
        else: v_tn += 1

        # Spatial check on overlapping segments
        if has_pred and has_gt:
            pred = gemini_lookup[current_second]
            p_ymin, p_xmin, p_ymax, p_xmax = [coord / 1000.0 for coord in pred]
            
            for gt in gt_boxes:
                g_xmin = gt["x_center"] - (gt["w"] / 2)
                g_xmax = gt["x_center"] + (gt["w"] / 2)
                g_ymin = gt["y_center"] - (gt["h"] / 2)
                g_ymax = gt["y_center"] + (gt["h"] / 2)

                int_w = max(0, min(p_xmax, g_xmax) - max(p_xmin, g_xmin))
                int_h = max(0, min(p_ymax, g_ymax) - max(p_ymin, g_ymin))
                
                containment = (int_w * int_h) / (gt["w"] * gt["h"]) if (gt["w"] * gt["h"]) > 0 else 0.0
                all_containments.append(containment)

    # Compile baseline metrics array calculations
    metrics = calculate_metrics(v_tp, v_fp, v_fn, beta=2.0)
    metrics["avg_containment"] = np.mean(all_containments) if all_containments else 0.0
    
    # CRITICAL ADDITION: Append confusion matrix data to satisfy your loop script requirements
    metrics["confusion_matrix"] = {"TP": v_tp, "FP": v_fp, "FN": v_fn, "TN": v_tn}
    
    return metrics


if __name__ == "__main__":
    # Define exact system asset paths securely using path objects
    json_path = LOCAL_DIR / "llm_experiment" / "video_json_outputs" / "response_labeled_CS_0031_POSTWEAN_d1_p2_cow3_02112025_ch02-20251103001956_60818_60835.json"
    frames_dir = LOCAL_DIR / "sample_videos" / "cross_sucking_clip_sample" / "CS_0031_POSTWEAN_d1_p2_cow3_02112025_ch02-20251103001956_60818_60835"

    # --- FIX 1: Safely load and parse file data into a list before sending to evaluation ---
    if not json_path.exists():
        print(f"Error: JSON file missing at location -> {json_path}")
    elif not frames_dir.exists():
        print(f"Error: Frame tracking path missing at location -> {frames_dir}")
    else:
        with open(json_path, "r") as f:
            raw_gemini_list = json.load(f)
            
        print(f"Initiating frame tracking evaluation for: {json_path.name}")
        results = evaluate_video_pipeline(raw_gemini_list, frames_dir, source_fps=30.0)
        
        if results:
            print("\n--- CLIP PERFORMANCE EVALUATION COMPLETED ---")
            print(f"Confusion Matrix Metrics: {results['confusion_matrix']}")
            print(f"Precision : {results['precision']:.3f}")
            print(f"Recall    : {results['recall']:.3f}")
            # print(f"F2 Score  : {results['f2']:.3f}")
            print(f"Coverage Containment Percentage: {results['avg_containment']:.2%}")