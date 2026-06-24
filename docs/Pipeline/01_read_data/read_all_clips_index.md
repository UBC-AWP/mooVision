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

## Filename Requirements

Human annotation environments often introduce variations in file naming styles. The matching module resolves these irregularities by stripping file extensions and isolating a standardized `(Numeric_ID, Part_Number)` tuple configuration. Naming conventions are not currently forced, however incorrectly formatted filenames will throw errors and break workflows. To avoid a situation in which unconsidered naming conventions silently pass intial checks and break downstream workflows, we suggest standardizing filenames using the following format:

* **Target Format: cross-sucking clip names** `CS_{clip_number}_{Weaning_period}_d{day_number}_p{pen_number}_cow{cow_identifier}_{ddmmyyyy}_{source_video_base_name}_{clip_start_time_s}_{clip_end_time_s}_{part or none}.mp4`. Where `part or none` is one of `part01`, `part02`, or blank. If blank, the file name should read as `..._{clip_end_time_s}.mp4`

* **Target Format: annotation folders** `{clip_number}_{part or none}.zip`. Where `part or none` is one of `part01`, `part02`, or blank. If blank, the file name should read as `{clip_number}.zip`

For examples, see rows one to three of `Accepted Formats Reference: Annotation Folders (CVAT Exports)`, and rows one to three of `Accepted Formats Reference: Cross-Sucking Clips` below.

---

## Filename Ingestion Reference

This project currently accepts cross-sucking clip names and annotaion folder names in the following formats. Attempting to pass other filename types may throw an error and terminate the workflows. To avoid silently passing incorrectly parsed files, follow the naming conventions above.

### Accepted Formats Reference: Annotation Folders (CVAT Exports)

| Example String | Extracted ID | Extracted Part | Matching Case / Pattern Type |
| --- | --- | --- | --- |
| `1234.zip` | `1234` | *None* | Standard clean ID format. |
| `2345_part02.zip` | `2345` | `2` | Zero-padded explicit part suffix (`_part0X`). |
| `2345_part01.zip` | `2345` | `1` | Zero-padded explicit part suffix (`_part0X`). |
| `3456_part1.zip` | `3456` | `1` | Single-digit explicit part suffix (`_partX`). |
| `3456_part2.zip` | `3456` | `2` | Single-digit explicit part suffix (`_partX`). |
| `4567_p01.zip` | `4567` | `1` | Zero-padded shorthand part suffix (`_p0X`). |
| `4567_p02.zip` | `4567` | `2` | Zero-padded shorthand part suffix (`_p0X`). |
| `CS_0317_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577.mp4.zip` | `317` | *None* | Double extension fallback (`.mp4.zip`) using structured base string. |
| `CS_0319_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part01.mp4.zip` | `319` | `1` | Double extension fallback with part notation embedded. |
| `CS_0319_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part02.mp4.zip` | `319` | `2` | Double extension fallback with part notation embedded. |
| `5678_p1.zip` | `5678` | `1` | Single-digit shorthand part suffix (`_pX`). |
| `5678_p2.zip` | `5678` | `2` | Single-digit shorthand part suffix (`_pX`). |
| `7891-p1.zip` | `7891` | `1` | Hyphenated shorthand part suffix (`-pX`). |
| `7891-p2.zip` | `7891` | `2` | Hyphenated shorthand part suffix (`-pX`). |
| `180.zip` | `180` | *None* | Clean fallback for shorter 3-digit numeric IDs. |
| `6789_part02zip.zip` | `6789` | `2` | Typo handling: captures trailing characters inside explicit name (`_part0Xzip`). |
| `9021 - p1.zip` | `9021` | `1` | Spaced hyphen shorthand variation (` - pX`). |
| `9021 - p2.zip` | `9021` | `2` | Spaced hyphen shorthand variation (` - pX`). |

### Accepted Formats Reference: Cross-Sucking Clips

| Example String | Extracted ID | Extracted Part | Matching Case / Pattern Type |
| --- | --- | --- | --- |
| `CS_0101_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577.mp4` | `101` | *None* | Standard structured baseline clip record. |
| `CS_0131_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part01.mp4` | `131` | `1` | Standard structured part 1 slice record. |
| `CS_0131_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part02.mp4` | `131` | `2` | Standard structured part 2 slice record. |

---

## Data Schema Reference

The pipeline enforces strict data types and structural configurations using `pandera` schemas. Both schemas are configured with `strict=True` (rejects unexpected columns) and `coerce=False` (rejects incorrect data types).

| Column Name | Data Type | Nullable | Schema Version | Description / Validation Rules |
| --- | --- | --- | --- | --- |
| `clip_name` | `str` | No | Both | Name of the unlabelled clip file. |
| `clip_relative_path` | `str` | No | Both | Relative file path to the unlabelled video snippet. |
| `clip_output_path` | `str` | No | Both | Target output path for the video clip. |
| `source_video_path` | `str` | No | Both | Full path to the original long-form source video. |
| `source_video_basename` | `str` | No | Both | File name (excluding directory path) of the source video. |
| `labelled_clip_relative_path` | `str` | No | **Processed Only** | Relative path to the matching CVAT `.zip` annotation archive. |
| `export_status` | `str` | No | Both | Status tracking string for the clip's export state. |
| `part_index` | `int` | No | Both | Component index for split clips. Must be $\ge 0$. |
| `part_count` | `int` | No | Both | Total number of parts split from the interval. Must be $\ge 1$. |
| `observation_id` | `str` | No | Both | Unique structural identifier for the behavioral observation. |
| `group_name` | `str` | No | Both | Experimental or cohort group identifier. |
| `phase` | `str` | No | Both | Trial phase description (e.g., `POSTWEAN`, `WEAN`). |
| `day` | `int` | No | Both | Timeline day number identifier. Must be $\ge 0$. |
| `pen` | `int` | No | Both | Experimental pen location number. Must be $\ge 0$. |
| `obs_date_raw` | `int` | No | Both | Raw unparsed date identifier integer (e.g., `02112025`). |
| `subject` | `str` | **Yes** | Both | Identified animal subject ID (allows up to 3 nulls from source). |
| `modifiers` | `str` | No | Both | Conditional behavior modification text tags. |
| `interval_start_obs_sec` | `float` | No | Both | Start time of continuous event in observation. Must be $\ge 0$. |
| `interval_end_obs_sec` | `float` | No | Both | End time of continuous event in observation. Must be $> \text{start}$. |
| `part_start_obs_sec` | `float` | No | Both | Segmented chunk start time within observation. Must be $\ge 0$. |
| `part_end_obs_sec` | `float` | No | Both | Segmented chunk end time within observation. Must be $> \text{start}$. |
| `source_segment_obs_start_sec` | `float` | No | Both | Source context segment match start marker. Must be $\ge 0$. |
| `source_segment_obs_end_sec` | `float` | No | Both | Source context segment match end marker. Must be $> \text{start}$. |
| `clip_start_in_source_sec` | `float` | No | Both | Precise crop baseline entry time in source video. Must be $\ge 0$. |
| `clip_end_in_source_sec` | `float` | No | Both | Precise crop baseline exit time in source video. Must be $> \text{start}$. |

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

* **Real-World Variant Handling:** The module parses and resolves explicit part strings, whitespace discrepancies, and common human typos (e.g., `0003 - p2.zip`, `0004_part02zip.zip`, `0101_part01zip.zip`) matching annotation labels back to the original cross-sucking clips. See the tables under `Accepted Formats Reference Above`.

### Conflict Mitigation & Ambiguity Faults - Fixed Clips

When multiple annotation archives match a single clip index record, the engine checks file locations. If a naming conflict occurs between a file inside the `fixed_clips` directory and a base clip option, the script defaults to the base file and logs a `UserWarning`. However, if multiple conflicting files emerge without a clear fallback rule, a `ValueError` is triggered to protect training set integrity.

---

## Usage

### Command Line Execution

```bash
uv run scripts/data_reading/read_data_from_index_file.py \
  --index_path "data/cross_sucking_clips/all_clips_index.csv" \
  --clips_dir "data/unlabelled_clips" \
  --labels_dir "data/labelled_clips" \
  --FORCE

```

---

## Function Reference

::: scripts.data_reading.read_all_clips_index
options:
show_source: false
show_root_heading: true

---

## WIP

1. **Source Video Verification Integration:** Hook the `source_dir` path variable into an active file-checking sequence to purge row entries from the index if the underlying continuous source videos are missing.
2. **Part Index Boundary Constraints:** Reactivate the disabled internal validation check to programmatically enforce part count parameters:

part_index < part_count

3. **Regex Alignment Update:** Update `parse_unlabelled_name` to handle alternative separator variations. The pattern currently processes standard part strings (e.g., `_part01`), but it chokes on double underscores (e.g., `_part_01`). Updating this pattern will better capture common human-edited export names.