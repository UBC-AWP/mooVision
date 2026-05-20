import sys
import cv2
import time
import json
import re
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent)) 
from config import RAW_DIR,CLIPS_INDEX_DIR,REPRODUCED_CLIPS_DIR,BASELINE_METADATA_DIR

def reproduce_clip(raw_video_path: Path, start_sec: float, end_sec: float, output_path: Path) -> bool:
    """ Extract a clip from raw_video_path between start_sec and end_sec, and save it to output_path."""
    cap = cv2.VideoCapture(str(raw_video_path))
    if not cap.isOpened():
        print(f"Could not open: {raw_video_path.name}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

    cap.release()
    writer.release()
    return True

def split_by_index(index_path: Path, output_path: Path) -> None:
    """ Reproduce clips based on the index CSV and save them to output_path."""
    index_df = pd.read_csv(index_path)
    # get all raw videos in RAW_DIR
    raw_videos = {f.name: f for f in RAW_DIR.rglob("*.mp4")}

    matched = index_df[index_df["source_video_basename"].isin(raw_videos.keys())]
    print(f"Found {len(matched)} clips to reproduce from {matched['source_video_basename'].nunique()} raw videos\n")

    success = 0
    for _, row in matched[:1].iterrows():
        source_video_path = row["source_video_path"]
        source_relative_path = re.search(r"Pen.*", source_video_path).group(0)
        # normalize Windows separators -> POSIX
        source_relative_path = source_relative_path.replace("\\", "/")
        raw_path = RAW_DIR / "videos" / source_relative_path
        save_path = output_path / row["clip_name"]
        start_sec = float(row["clip_start_in_source_sec"])
        end_sec = float(row["clip_end_in_source_sec"])

        print(f"[{row['clip_name']}]  {start_sec:.1f}s → {end_sec:.1f}s")
        print(f"from: {raw_path}")
        if reproduce_clip(raw_path, start_sec, end_sec, save_path):
                print(f"saved to {save_path.name}")
                success += 1

        print(f"\nDone — {success} reproduced")
        
def split_by_json_events(json_path: Path, output_dir: Path, annotate: bool = True) -> int:
    """Reproduce clips based on events specified in a JSON file, and optionally annotate them."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if json_path.is_dir():
        json_files = sorted(json_path.glob("*.json"))  # non-recursive: only this folder
        if not json_files:
            print(f"No .json files found in {json_path}")
            return 0
    else:
        json_files = [json_path]

    total_success = 0

    for jf in json_files:
        data = json.loads(jf.read_text(encoding="utf-8"))
        video_path = Path(data["video_path"])
        identifier = data.get("identifier", video_path.name)
        events = data.get("events", [])
        fps = float(data.get("fps", 30.0))

        if not video_path.exists():
            raise FileNotFoundError(f"video_path does not exist: {video_path} (from {jf})")
        if not events:
            print(f"No events found in {jf.name}")
            continue

        success = 0
        for i, ev in enumerate(events, start=1):
            start_sec = float(ev["start_sec"])
            end_sec = float(ev["end_sec"])

            base_name = f"{Path(identifier).stem}__event{i:03d}_{start_sec:.1f}-{end_sec:.1f}"
            out_path = output_dir / f"{base_name}.mp4"

            print(f"[{out_path.name}] {start_sec:.1f}s → {end_sec:.1f}s")
            print(f"from: {video_path}")

            if not reproduce_clip(video_path, start_sec, end_sec, out_path):
                continue

            # annotate (optional)
            if annotate:
                event_start_frame = int(start_sec * fps)
                boxes = ev.get("intersection_box", [])

                # convert source-video frames -> clip frames
                clip_boxes = []
                for b in boxes:
                    f = int(b["frame"]) - event_start_frame
                    if f < 0:
                        continue
                    clip_boxes.append({**b, "frame": f})

                boxed_path = output_dir / f"{base_name}__boxed.mp4"
                if annotate_clip_with_boxes(out_path, boxed_path, clip_boxes):
                    print(f"saved boxed to {boxed_path.name}")

            print(f"saved to {out_path.name}")
            success += 1
            total_success += 1

        print(f"\nDone — {success} reproduced from {jf.name}")
    return total_success

def annotate_clip_with_boxes(input_clip: Path,output_clip: Path,boxes: list[dict],color=(0, 255, 0),thickness: int = 2,) -> bool:
    """ Annotate input_clip with bounding boxes and save as output_clip."""
    cap = cv2.VideoCapture(str(input_clip))
    if not cap.isOpened():
        print(f"Could not open clip: {input_clip}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_clip.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_clip), fourcc, fps, (w, h))

    # multiple boxes could occur in same frame; group them
    boxes_by_frame = {}
    for b in boxes:
        f = int(b["frame"])
        boxes_by_frame.setdefault(f, []).append(b)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for b in boxes_by_frame.get(frame_idx, []):
            x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return True

def run_splitting(func) -> None:
    if func == split_by_index:
        split_by_index(CLIPS_INDEX_DIR, REPRODUCED_CLIPS_DIR)
    elif func == split_by_json_events:
        split_by_json_events(BASELINE_METADATA_DIR, REPRODUCED_CLIPS_DIR)
    else:
        raise ValueError(f"Unknown splitting function: {func}")
    
def main():
    run_splitting(split_by_json_events)

if __name__ == "__main__":
    main()