import cv2
import sys
import shutil
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
        
def copy_zip_content(video_name:str,out_dir: Path) -> None:
    clip_index_path = LOCAL_DIR / "data" / "processed" / "processed_clips_index.csv"
    index_df = pd.read_csv(clip_index_path)

    index_df = index_df[index_df["clip_name"] == video_name]
    # clean_zip_path = Path(str(zip_relative_path).replace("\\", "/"))
    # full_zip_path  = LABELLED_CLIPS_DIR / clean_zip_path

    # tmp_dir = out_dir / "_tmp_unzip"
    # tmp_dir.mkdir(exist_ok=True)

    # with zipfile.ZipFile(full_zip_path, "r") as zf:
    #     zf.extractall(tmp_dir)

    # # find obj_train_data/ inside the unzipped content
    # obj_train_dir = tmp_dir / "obj_train_data"
    # if not obj_train_dir.exists():
    #     # search one level deeper in case zip has a subfolder
    #     matches = list(tmp_dir.rglob("obj_train_data"))
    #     obj_train_dir = matches[0] if matches else None

    # if obj_train_dir is None:
    #     print(f"obj_train_data/ not found in zip: {full_zip_path.name}")
    #     shutil.rmtree(tmp_dir)

    # # copy all frame_xxx.txt files into out_dir
    # txt_files = list(obj_train_dir.glob("frame_*.txt"))
    # for txt in txt_files:
    #     shutil.copy(txt, out_dir / txt.name)

    # shutil.rmtree(tmp_dir)  # clean up temp unzip folder
    # print(f"{len(txt_files)} annotation txts copied from zip")

if __name__ == "__main__":
    extract_frames()