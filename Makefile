# Use uv to run inside the project's managed environment
PY ?= uv run python

# If you change your scripts path, change below paths:
READ_INDEX := scripts/read_all_clips_index.py
SPLIT      := scripts/splitting.py
BASELINE   := scripts/baseline/baseline.py
EVAL       := scripts/evaluation.py
CLIP       := scripts/clipping.py

.PHONY: run pipeline read_index split baseline eval clip

run: pipeline
pipeline: read_index split baseline eval clip

read_index:
	$(PY) $(READ_INDEX)

split:
	$(PY) $(SPLIT)

baseline:
	$(PY) $(BASELINE)

eval:
	$(PY) $(EVAL)

clip:
	$(PY) $(CLIP)