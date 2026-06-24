# Data Requirements

Data requirements for cross-sucking video names, annotation folder names, data index columns, and video/annotaion folder file tpyes.

## Video and Annotation Folder File Types

Videos must be `.mp4` extensions. Annotation folders (CVAT Outputs) must be `.zip` folders.

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