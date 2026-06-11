import cv2
import sys
import shutil
import cv2
import zipfile
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent)) 

from config import LOCAL_DIR,LABELLED_CLIPS_DIR

FRAMES_DIR = LOCAL_DIR / "sample_videos" / "cross_sucking_clip_sample"

def extract_frames(input_path: Path=FRAMES_DIR) -> None:
    """
        Extract all frames from MP4 videos in a directory.

        Recursively searches for all MP4 files in the given directory and extracts
        every frame from each video as a JPEG image. Frames are saved to individual
        subdirectories named after each video clip. Skips videos that have already
        been processed.

        Parameters
        ----------
        video_path : Path
            Path to directory containing MP4 video files. Can contain subdirectories.

        Returns
        -------
        None
            This function writes frames to disk and does not return anything.

        Raises
        ------
        FileNotFoundError
            If `video_path` does not exist.
        cv2.error
            If a video file is corrupted or cannot be decoded.

        Notes
        -----
        - Frames are extracted to subdirectories under `FRAMES_DIR` (global variable)
        - Each subdirectory is named after the video stem (filename without extension)
        - If a video has already been processed (output directory exists and contains files),
        it will be skipped
        - Frame filenames follow the format: `frame_XXXXXX.jpg` (zero-padded 6-digit index)
        - Requires OpenCV (cv2) and pathlib

        Examples
        --------
        >>> from pathlib import Path
        >>> video_dir = Path("/videos")
        >>> extract_frames(video_dir)
        Name: sample_video.mp4
        sample_video → 150 frames extracted

        >>> # Output structure created:
        >>> # FRAMES_DIR/sample_video/frame_000000.jpg
        >>> # FRAMES_DIR/sample_video/frame_000001.jpg
        >>> # ...
        """
    raw_videos = {f.name: f for f in input_path.rglob("*.mp4")}
    for video_name, video_path in raw_videos.items():
        print(f"\nName: {video_name}")
        clip_name = video_path.stem
        out_dir   = input_path / clip_name
        out_dir.mkdir(parents=True, exist_ok=True)

        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"skipping {clip_name} — already done")
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"could not open: {video_path}")
            continue

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(str(out_dir / f"frame_{frame_idx:06d}.jpg"), frame)
            frame_idx += 1
        cap.release()
        print(f"{clip_name} -> {frame_idx} frames extracted")
        copy_zip_content(video_name,out_dir)
        
        # Compile the annotated video file
        output_video_path = out_dir / f"{clip_name}_with_bounding_boxes.mp4"
        reconstruct_clip_with_boxes(out_dir, output_video_path, fps=30)
        
def copy_zip_content(video_name: str, out_dir: Path) -> None:
    # If text files are already copied, don't repeat the work
    if any(out_dir.glob("frame_*.txt")):
        print("Annotation text files already exist. Skipping zip processing.")
        return

    clip_index_path = LOCAL_DIR / "data" / "processed" / "processed_clips_index.csv"
    if not clip_index_path.exists():
        print(f"Index file missing: {clip_index_path}")
        return

    index_df = pd.read_csv(clip_index_path)
    index_df = index_df[index_df["clip_name"] == video_name]
    
    if index_df.empty:
        print(f"No index record found for video: {video_name}")
        return

    for _, row in index_df.iterrows():
        zip_relative_path = row["labelled_clip_relative_path"]
        
        # Keep slash normalization active to protect against Windows-to-Linux path breaks
        clean_zip_path = str(zip_relative_path).replace("\\", "/")
        full_zip_path  = LABELLED_CLIPS_DIR / clean_zip_path

        if not full_zip_path.exists():
            print(f"Zip file path not found: {full_zip_path}")
            continue

        tmp_dir = out_dir / "_tmp_unzip"
        tmp_dir.mkdir(exist_ok=True)

        try:
            with zipfile.ZipFile(full_zip_path, "r") as zf:
                zf.extractall(tmp_dir)

            obj_train_dir = tmp_dir / "obj_train_data"
            if not obj_train_dir.exists():
                matches = list(tmp_dir.rglob("obj_train_data"))
                obj_train_dir = matches[0] if matches else None

            if obj_train_dir is None:
                print(f"obj_train_data/ not found in zip: {full_zip_path.name}")
                continue

            txt_files = list(obj_train_dir.glob("frame_*.txt"))
            for txt in txt_files:
                shutil.copy(txt, out_dir / txt.name)

            print(f"{len(txt_files)} annotation txts copied from zip")

        except Exception as e:
            print(f"Error processing zip file {full_zip_path.name}: {e}")
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
                
def reconstruct_clip_with_boxes(clip_folder_path: Path, output_video_path: Path, fps: float) -> None:
    """Reads frames and YOLO coordinates sequentially to compile an annotated MP4 video."""
    frame_paths = sorted(list(clip_folder_path.glob("frame_*.jpg")))
    
    if not frame_paths:
        print(f"No frames available to compile in {clip_folder_path.name}")
        return

    # Check if compiled video already exists to prevent re-rendering identical work
    if output_video_path.exists():
        print(f"Annotated video already exists for {clip_folder_path.name}. Skipping generation.")
        return

    first_frame = cv2.imread(str(frame_paths[0]))
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    print(f"Compiling {len(frame_paths)} frames into annotated clip at {fps:.2f} FPS...")

    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        
        txt_path = frame_path.with_suffix(".txt")
        if txt_path.exists():
            with open(txt_path, "r") as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue 
                
                # Parse YOLO structure
                x_center = float(parts[1])
                y_center = float(parts[2])
                box_w    = float(parts[3])
                box_h    = float(parts[4])

                # Transform normalized ratios back to bounding pixel coordinates
                xmin = int((x_center - box_w / 2) * width)
                ymin = int((y_center - box_h / 2) * height)
                xmax = int((x_center + box_w / 2) * width)
                ymax = int((y_center + box_h / 2) * height)

                # Clamp values securely within resolution margins
                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(width, xmax), min(height, ymax)

                # Draw solid bounding rectangle and custom tracking indicator tag
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
                cv2.putText(frame, "cross-sucking", (xmin, max(15, ymin - 8)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        video_writer.write(frame)

    video_writer.release()
    print(f"Annotated video successfully saved to: {output_video_path.name}")
    
if __name__ == "__main__":
    extract_frames()