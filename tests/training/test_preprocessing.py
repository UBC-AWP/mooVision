"""
Module for testing preprocessing.py
"""

import pytest

### Tests to run ###


### Input Handling ###
# - Correct Types (raise errors)
# - labels_path is empty (raise error), list of strings (raise error)
# - labels_root does not exist (raise error)
# - split not one of train or val (raise error)
# - skip large than number of frames? -- some videos are pretty short.

### Normal Outputs (all assumptions are valid, creates a good output)###
# - Images should be under images/train/ or images/val/ (check exists)
# - Labels should be under labels/train or labels/val/ (check exists)
# - Outputs .jpg and .txt files. (check files are correct form)
# - Should be same number of Images and Labels, with matching names. (check number of files in each output folder)
# - Correct naming conventions correcponding to inputs (check names match across folders)


### Naming Conventions ###
# - Outputs frames with correct naming conventions under normal circumstances.
# - Outputs labels with correct naming conventions. under normal circumstances.
# - Images and Frames have matching naming conventions under normal circumstances.
# - Edge Cases:
#       naming conventions for video name do not match expected form (raise error)  -- should raise error from matching functions
#       naming conventions for label names do not match known form (raise error)  -- should raise error though mathcing functions
#       naming conventions for .txt files do not match expected form (raise error)  -- would throw off additional naming conventions, check when reading names.

### Edge Cases ###
# - labels_path paths no not exist in extract_labels (raises error)
# - video paths do not exist in extract_frames (raises error)
# - target_folder does not exist in extract_labels (inside the zip file) (raises error)
# - No .txt files in the target folder in the zip file (raises error)
# - List of returned files in target folder is empty (not due to no .txt files) (raises error)
# - Video is corrupted and cannot be read (raises error)
# - Reading in multiple videos should still return a good output (i.e. ensures good naming conventions)
# - Number of files in annotated data does not match number of frames from video, use running tally's to record this (raise error is not the same)
# -
