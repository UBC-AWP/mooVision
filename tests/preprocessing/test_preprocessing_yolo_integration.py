"""
Integration tests
"""

import sys
from pathlib import Path
import pytest
import zipfile
import cv2
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.preprocessing.preprocessing_yolo import (
    process_single_split,
)


class TestProcessSingleSplitIntegration:

    @pytest.fixture
    def synthetic_dataset(self, tmp_path):
        """Generates a synthetic CVAT YOLO structured environment with specific

        video and zip naming conventions.
        """
        clips_dir = tmp_path / "raw_clips"
        labels_dir = tmp_path / "raw_labels"
        clips_dir.mkdir()
        labels_dir.mkdir()

        # 1. Create a real 5-frame video asset with the requested naming convention
        video_name = "CS_0001_test_info.mp4"
        video_file = clips_dir / video_name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_file), fourcc, 30.0, (320, 240))
        try:
            for _ in range(5):  # Frames 0, 1, 2, 3, 4
                writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
        finally:
            writer.release()

        # 2. Build the structural CVAT ZIP archive named 0001zip.zip
        zip_name = "0001.zip"
        zip_file_path = labels_dir / zip_name

        # Create a tiny dummy 1x1 PNG image array to satisfy validation checks
        _, png_bytes = cv2.imencode(".png", np.zeros((1, 1, 3), dtype=np.uint8))

        # We simulate that frames 0 and 2 have matching CVAT labels.
        with zipfile.ZipFile(zip_file_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # Frame 0 CVAT Exports (Raw structural names expected by your code validation)
            zipf.writestr(
                "obj_train_data/frame_000000.txt",
                "0 0.5 0.5 0.2 0.2",
            )
            zipf.writestr(
                "obj_train_data/frame_000000.png",
                png_bytes.tobytes(),
            )

            # Frame 2 CVAT Exports
            zipf.writestr(
                "obj_train_data/frame_000002.txt",
                "0 0.6 0.6 0.3 0.3",
            )
            zipf.writestr(
                "obj_train_data/frame_000002.png",
                png_bytes.tobytes(),
            )

        # 3. Assemble driving Pandas DataFrame tracking row
        df = pd.DataFrame(
            {
                "clip_relative_path": [video_name],
                "labelled_clip_relative_path": [zip_name],
            }
        )

        return {
            "df": df,
            "clips_root_dir": clips_dir,
            "labels_root_dir": labels_dir,
        }

    @pytest.mark.integration
    def test_process_single_split_end_to_end_execution(
        self, synthetic_dataset, tmp_path
    ):
        """Integration Test: Runs the split processor end-to-end unmocked to verify

        CVAT archive unpacking, tracking, and final frame synchronization.
        """
        working_dir = tmp_path / "workspace"

        # Act: Fire the pipeline unmocked
        process_single_split(
            df=synthetic_dataset["df"],
            working_dir=working_dir,
            split="train",
            skip=1,
            force=False,
            clips_root_dir=synthetic_dataset["clips_root_dir"],
            labels_root_dir=synthetic_dataset["labels_root_dir"],
        )

        # Assert: Check that the final workspace image outputs directory exists
        img_dir = working_dir / "images" / "train"
        assert img_dir.exists(), "Pipeline failed to generate final images workspace!"

        # Inspect the disk to confirm which physical frames survived the sync phase
        remaining_files = [f.name for f in img_dir.glob("*.jpg")]

        # Annotated frames (0 and 2) must remain intact on disk
        assert any(
            "000000" in name for name in remaining_files
        ), "Annotated CVAT frame 0 was missing from workspace!"
        assert any(
            "000002" in name for name in remaining_files
        ), "Annotated CVAT frame 2 was missing from workspace!"

        # Orphaned frames (1, 3, and 4) must be cleanly unlinked/deleted
        assert not any(
            "000001" in name for name in remaining_files
        ), "Orphan frame 1 was not dropped!"
        assert not any(
            "000003" in name for name in remaining_files
        ), "Orphan frame 3 was not dropped!"
        assert not any(
            "000004" in name for name in remaining_files
        ), "Orphan frame 4 was not dropped!"
