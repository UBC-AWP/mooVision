# Reading in Data

This module serves as the data entry point and engineering foundation for the MooVision project. It handles the parsing, structural validation, asset tracking, and label synchronization required to transform raw, manually maintained clip indices and unstructured CVAT annotation exports into clean datasets for model training.

---

## Summary

By matching `.zip` annotations to clip names we build a data index housing the neccessary information on videos and label folder required for all downstream pipeline functionality. Moreover, through strict data schemas and filtering methods, it catches formatting errors, missing videos, or duplicate labels early, allowing all neccessary data to reach downstream machine learning workflows while preventing corrupted data from moving beyond the intial boundary point.

| Property | Detail |
| --- | --- |
| **Primary Script** | `read_data_from_index_file.py` |
| **Supporting Modules** | `matching.py` (String Matcher), `schema.py` (Pandera Validation) |
| **Libraries Used** | Pandas + Pandera |
| **Input** | Index sheet (CSV/Excel/etc.) + Unlabelled clips folder + CVAT `.zip` annotations |
| **Output** | Validated raw index copy + Processed model-ready index CSV |
| **CLI Support** | Yes |

> **Note:** For explicit column definitions and data type constraints governing these dataframes, refer directly to `Data Schema` below.

---

## Input

* **Clips Index File:** A tabular spreadsheet (`.csv`, `.xlsx`, etc.) containing clip metadata, animal identifiers, pen numbers, and video attributes.
* **Unlabelled Clips Directory (`clips_dir`):** The folder containing sliced, short-form `.mp4` video clips.
* **Labelled Clips Directory (`labels_dir`):** The folder containing bounding box annotation `.zip` bundles exported from CVAT.
* **Source Videos Directory (`source_dir`):** Root storage hosting the original, continuous un-segmented baseline footage.

## Output

The script saves processed index states directly back to disk and returns `None`. The resulting outputs are organized into raw and processed subdirectories:

```text
ROOT_DIR/                                   # Data root directory configured in config.py
 └── data/
     ├── raw/
     │   └── all_clips_index_raw.csv        # Validated copy of raw ingestion data before asset filtering
     └── processed/
         └── processed_clips_index.csv      # Synchronized index with verified path mappings and paired label references

```

---

## How it works

1. The reader automatically detects the index file extension and applies the correct parsing engine (supporting `.csv`, `.xlsx`, `.parquet`, `.json`, and `.tsv`).
2. The input dataset is validated against the raw schema rules. If columns are missing or data types are incorrect, execution halts immediately.
3. The pipeline checks the physical presence of every clip file listed in the index. Rows pointing to missing assets are dropped from the final index, and a warning is logged.
4. A regex engine standardizes annotation filename variations down to unique numeric ID and part ID tokens to reliably link raw clips with hand-named CVAT exports (`.zip` annotation folders). Relative paths to these folders are then added to the data index under the column `labelled_clip_relative_path`
5. Once label paths are resolved and appended, the dataframe undergoes a final validation pass for type stability and chronological logic before being saved to disk.

---

## Core Pipeline Concepts

### Flexible Format Resolution

To maximize workflow flexibility, the ingestion framework identifies file suffixes at runtime and dynamically maps them to their respective reading lambdas. This allows the pipeline to accept human-readable spreadsheets (`.xlsx`), flat-files (`.csv`, `.tsv`), or column-compressed stores (`.parquet`) without requiring manual code changes.

### Strict Structural Alignment

Dataframe validation blocks default to `strict=True` and `coerce=False` layouts. Downstream modeling components depend on predictable datatypes; introducing unauthorized metadata columns or passing mismatched format types yields immediate terminal faults rather than letting bad data pass through silently.

### Chronological Validation

The pipeline evaluates temporal sequences to ensure absolute logical continuity across all recorded windows. Pandera mathematical checks verify that clip start-times consistently precede their respective end-times across all tracking boundaries:

interval_end_obs_sec > interval_start_obs_sec
part_end_obs_sec > part_start_obs_sec
source_segment_obs_end_sec > source_segment_obs_start_sec
clip_end_in_source_sec > clip_start_in_source_sec

### Unification & Regex Matching

Because filenames vary, the matching module automatically standardizes naming conventions to link CVAT annotation folders directly to their corresponding video clips.

* **Real-World Clip Name Parsing:** Currently checks for `CS_{numeric_ID}...` format near the filename beginning to capture the numeric ID, and checks for `..._part01.mp4` or `..._part02.mp4` to capture the part ID. If no part id is found it is labelled as None.

* **Real-World Variant Handling:** The module parses and resolves explicit part strings, whitespace discrepancies, and common human typos (e.g., `0003 - p2.zip`, `0004_part02zip.zip`, `0101_part01zip.zip`) matching annotation labels back to the original cross-sucking clips.

> **Note:** For more information see Data Requirements.

### Conflict Mitigation & Ambiguity Faults - Fixed Clips

When multiple annotation archives match a single clip index record, the engine checks file locations. If a naming conflict occurs between a file inside the `fixed_clips` directory and a base clip option, the script defaults to the base file and logs a `UserWarning`. However, if multiple conflicting files emerge without a clear fallback rule, a `ValueError` is triggered to protect training set integrity.

---

## Usage

### Command Line Execution

```bash
uv run scripts/read_data/read_all_clips_index.py \
  --index_path "data/cross_sucking_clips/all_clips_index.csv" \
  --clips_dir "data/unlabelled_clips" \
  --labels_dir "data/labelled_clips" \
  --force

```

---

## Function Reference

::: scripts.read_data.read_all_clips_index
options:
show_source: false
show_root_heading: true

---

## WIP

1. **Source Video Verification Integration:** Hook the `source_dir` path variable into an active file-checking sequence to purge row entries from the index if the underlying continuous source videos are missing.
2. **Part Index Boundary Constraints:** Reactivate the disabled internal validation check to programmatically enforce part count parameters:

part_index < part_count

3. **Regex Alignment Update:** Update `parse_unlabelled_name` to handle alternative separator variations. The pattern currently processes standard part strings (e.g., `_part01`), but it chokes on double underscores (e.g., `_part_01`). Updating this pattern will better capture common human-edited export names.