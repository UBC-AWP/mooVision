"""
Data Schema for MooVision Data Files
"""

import pandera as pa

schema = pa.DataFrameSchema(
    columns={
        # --- Identity / path columns ---
        "clip_name": pa.Column(str, nullable=False),
        "clip_relative_path": pa.Column(str, nullable=False),
        "clip_output_path": pa.Column(str, nullable=False),
        "source_video_path": pa.Column(str, nullable=False),
        "source_video_basename": pa.Column(str, nullable=False),
        # --- Export status ---
        "export_status": pa.Column(str, nullable=False),
        # --- Part indexing ---
        "part_index": pa.Column(int, pa.Check.ge(0), nullable=False),
        "part_count": pa.Column(int, pa.Check.ge(1), nullable=False),
        # --- Observation metadata ---
        "observation_id": pa.Column(str, nullable=False),
        "group_name": pa.Column(str, nullable=False),
        "phase": pa.Column(str, nullable=False),
        "day": pa.Column(int, pa.Check.ge(0), nullable=False),
        "pen": pa.Column(int, pa.Check.ge(0), nullable=False),
        "obs_date_raw": pa.Column(int, nullable=False),
        "subject": pa.Column(str, nullable=True),  # 3 nulls in source
        "modifiers": pa.Column(str, nullable=False),
        # --- Interval timing (seconds) ---
        "interval_start_obs_sec": pa.Column(float, pa.Check.ge(0), nullable=False),
        "interval_end_obs_sec": pa.Column(float, pa.Check.ge(0), nullable=False),
        "part_start_obs_sec": pa.Column(float, pa.Check.ge(0), nullable=False),
        "part_end_obs_sec": pa.Column(float, pa.Check.ge(0), nullable=False),
        # --- Source segment timing (seconds) ---
        "source_segment_obs_start_sec": pa.Column(
            float, pa.Check.ge(0), nullable=False
        ),
        "source_segment_obs_end_sec": pa.Column(float, pa.Check.ge(0), nullable=False),
        "clip_start_in_source_sec": pa.Column(float, pa.Check.ge(0), nullable=False),
        "clip_end_in_source_sec": pa.Column(float, pa.Check.ge(0), nullable=False),
    },
    checks=[
        # End times must be after start times
        pa.Check(
            lambda df: (
                df["interval_end_obs_sec"] > df["interval_start_obs_sec"]
            ).all(),
            error="interval_end must be > interval_start",
        ),
        pa.Check(
            lambda df: (df["part_end_obs_sec"] > df["part_start_obs_sec"]).all(),
            error="part_end must be > part_start",
        ),
        pa.Check(
            lambda df: (
                df["source_segment_obs_end_sec"] > df["source_segment_obs_start_sec"]
            ).all(),
            error="source_segment_end must be > source_segment_start",
        ),
        pa.Check(
            lambda df: (
                df["clip_end_in_source_sec"] > df["clip_start_in_source_sec"]
            ).all(),
            error="clip_end must be > clip_start",
        ),
        # part_index must be < part_count
        pa.Check(
            lambda df: (df["part_index"] < df["part_count"]).all(),
            error="part_index must be < part_count",
        ),
    ],
    strict=True,  # fail on unexpected columns
    coerce=False,  # fail on unexpected types
)

processed_schema = pa.DataFrameSchema(
    columns={
        # --- Identity / path columns ---
        "clip_name": pa.Column(str, nullable=False),
        "clip_relative_path": pa.Column(str, nullable=False),
        "clip_output_path": pa.Column(str, nullable=False),
        "source_video_path": pa.Column(str, nullable=False),
        "source_video_basename": pa.Column(str, nullable=False),
        "labelled_clip_relative_path": pa.Column(str, nullable=False),
        # --- Export status ---
        "export_status": pa.Column(str, nullable=False),
        # --- Part indexing ---
        "part_index": pa.Column(int, pa.Check.ge(0), nullable=False),
        "part_count": pa.Column(int, pa.Check.ge(1), nullable=False),
        # --- Observation metadata ---
        "observation_id": pa.Column(str, nullable=False),
        "group_name": pa.Column(str, nullable=False),
        "phase": pa.Column(str, nullable=False),
        "day": pa.Column(int, pa.Check.ge(0), nullable=False),
        "pen": pa.Column(int, pa.Check.ge(0), nullable=False),
        "obs_date_raw": pa.Column(int, nullable=False),
        "subject": pa.Column(str, nullable=True),  # 3 nulls in source
        "modifiers": pa.Column(str, nullable=False),
        # --- Interval timing (seconds) ---
        "interval_start_obs_sec": pa.Column(float, pa.Check.ge(0), nullable=False),
        "interval_end_obs_sec": pa.Column(float, pa.Check.ge(0), nullable=False),
        "part_start_obs_sec": pa.Column(float, pa.Check.ge(0), nullable=False),
        "part_end_obs_sec": pa.Column(float, pa.Check.ge(0), nullable=False),
        # --- Source segment timing (seconds) ---
        "source_segment_obs_start_sec": pa.Column(
            float, pa.Check.ge(0), nullable=False
        ),
        "source_segment_obs_end_sec": pa.Column(float, pa.Check.ge(0), nullable=False),
        "clip_start_in_source_sec": pa.Column(float, pa.Check.ge(0), nullable=False),
        "clip_end_in_source_sec": pa.Column(float, pa.Check.ge(0), nullable=False),
    },
    checks=[
        # End times must be after start times
        pa.Check(
            lambda df: (
                df["interval_end_obs_sec"] > df["interval_start_obs_sec"]
            ).all(),
            error="interval_end must be > interval_start",
        ),
        pa.Check(
            lambda df: (df["part_end_obs_sec"] > df["part_start_obs_sec"]).all(),
            error="part_end must be > part_start",
        ),
        pa.Check(
            lambda df: (
                df["source_segment_obs_end_sec"] > df["source_segment_obs_start_sec"]
            ).all(),
            error="source_segment_end must be > source_segment_start",
        ),
        pa.Check(
            lambda df: (
                df["clip_end_in_source_sec"] > df["clip_start_in_source_sec"]
            ).all(),
            error="clip_end must be > clip_start",
        ),
        # part_index must be < part_count
        pa.Check(
            lambda df: (df["part_index"] < df["part_count"]).all(),
            error="part_index must be < part_count",
        ),
    ],
    strict=True,  # fail on unexpected columns
    coerce=False,  # fail on unexpected types
)
