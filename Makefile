# Use uv to run inside the project's managed environment
PY ?= uv run

# ── Script paths ────────────────────────────────────────────────────────────
READ_INDEX   := scripts/read_data/read_all_clips_index.py
SPLIT        := scripts/split_data/split_data.py
PREPROCESS   := scripts/preprocessing/preprocessing_yolo.py
TRAIN        := scripts/training/training_yolo.py
BASELINE     := scripts/run_models/baseline/baseline.py
RUN_YOLO     := scripts/run_models/run_testing.py
EVAL         := scripts/evaluation/evaluation.py
CLIP         := scripts/clipping/clipping.py

# ── Report paths ─────────────────────────────────────────────────────────────
ANALYSIS_SCRIPT ?= notebooks/distribution_analysis.py
REPORT_PDF_QMD  ?= reports/final/final_report_pdf.qmd
REPORT_HTML_QMD ?= reports/final/final_report_html.qmd

# ── Configurable parameters (override from CLI if needed) ───────────────────
DATASET          ?= data/training/pipeline_demo/dataset/dataset.yaml
PROJECT          ?= pipeline_demo
RUN_NAME         ?= demo_01
DEVICE           ?= cpu
EPOCHS           ?= 1
BATCH            ?= 8

TRAIN_PATH       ?= data/processed/pipeline_demo/train.csv
VAL_PATH         ?= data/processed/pipeline_demo/val.csv
PREPROCESS_OUT   ?= data/training/pipeline_demo/
FRAME_SKIP       ?= 100
TEST_CSV         ?= data/processed/pipeline_demo/test.csv

MODEL_PATH       ?= data/yolo_training_runs/pipeline_demo/demo_01/weights/best.pt

PRED_YOLO        ?= results/metadata/pipeline_demo/yolo/
PRED_SEQNMS      ?= results/metadata/pipeline_demo/seq-nms/
GT_CSV           ?= data/processed/processed_clips_index.csv
EVAL_OUT_YOLO    ?= results/evaluation_report_yolo.json
EVAL_OUT_SEQNMS  ?= results/evaluation_report_seq_nms.json
LABELLED_DIR     ?= cross_sucking_labelled

CLIP_INPUT       ?= results/metadata/pipeline_demo/yolo/ch02_20250913094601_results.json

# ── Targets ──────────────────────────────────────────────────────────────────
.PHONY: run pipeline \
        read_index split preprocess train \
        baseline run_yolo \
        eval_yolo eval_seqnms eval \
        clip

# Run the full demo pipeline end-to-end
run: pipeline
pipeline: read_index split preprocess train baseline run_yolo eval clip

# 1. Read raw clip index
read_index:
	$(PY) $(READ_INDEX) --force

# 2. Split data into train/val/test
split:
	$(PY) $(SPLIT) --force

# 3. Preprocess for YOLO fine-tuning
preprocess:
	$(PY) $(PREPROCESS) \
		--train_path=$(TRAIN_PATH) \
		--val_path=$(VAL_PATH) \
		--output_path=$(PREPROCESS_OUT) \
		--skip=10 \
		--force

# 4. Train YOLO object detection model
train:
	$(PY) $(TRAIN) \
		--dataset=$(DATASET) \
		--project=$(PROJECT) \
		--name=$(RUN_NAME) \
		--device=$(DEVICE) \
		--epochs=$(EPOCHS) \
		--batch=$(BATCH)

# 5. Run baseline detector on test videos
baseline:
	$(PY) $(BASELINE) \
		--csv $(TEST_CSV) \
		--frame_skip $(FRAME_SKIP)

# 6. Run fine-tuned YOLO on test videos
run_yolo:
	$(PY) $(RUN_YOLO) \
		--model_path $(MODEL_PATH) \
		--data_path $(TEST_CSV) \
		--chunk 0 \
		--chunk_pct 1.0

# 7a. Evaluate fine-tuned YOLO predictions
eval_yolo:
	$(PY) $(EVAL) \
		--predictions $(PRED_YOLO) \
		--ground_truth $(GT_CSV) \
		--output $(EVAL_OUT_YOLO) \
		--labelled_clips_dir $(LABELLED_DIR)

# 7b. Evaluate YOLO + Seq-NMS predictions
eval_seqnms:
	$(PY) $(EVAL) \
		--predictions $(PRED_SEQNMS) \
		--ground_truth $(GT_CSV) \
		--output $(EVAL_OUT_SEQNMS) \
		--labelled_clips_dir $(LABELLED_DIR)

# 7. Run both evaluations
eval: eval_yolo eval_seqnms

# 8. Clip events from prediction JSONs
clip:
	$(PY) $(CLIP) --input $(CLIP_INPUT)

# ── Report targets ────────────────────────────────────────────────────────────
.PHONY: figures report-pdf report-html report

# Execute analysis script to export static figures for PDF report.
# Must be run before report-pdf if results have changed.
figures:
	$(PY) python $(ANALYSIS_SCRIPT)

# Render PDF report (requires figures to be exported first)
report-pdf: figures
	$(PY) quarto render $(REPORT_PDF_QMD) --to pdf

# Render HTML report (interactive charts, no figures export needed)
report-html:
	$(PY) quarto render $(REPORT_HTML_QMD) --to html

# Render both
report: report-pdf report-html