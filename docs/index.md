# MooVision

Computer vision for cross-sucking detection of dairy calves.

For full documentation visit [mkdocs.org](https://www.mkdocs.org).

## Commands

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

To run the mkdocs:
`uv run mkdocs serve`

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.

## Pipeline Diagram

```mermaid
---
config:
  layout: elk
  elk: {}
  theme: base
---
flowchart TB
    indexCSV["index.csv"] --> readClips["read_all_clips_index.py"]
    readClips --> processedIndex["Processed Index Files"]
    processedIndex --> splitting["splitting.py"]
    splitting --> trainingIndex["Training Index"] & testingIndex["Testing Index"]
    trainingIndex --> preprocessing["preprocessing.py\n(can vary per model)"]
    preprocessing --> frameData["Frame-by-Frame Data"]
    frameData --> training["training.py"]
    training --> modelChoice{"Choose Training Script"}
    modelChoice --> yolo["training_yolo.py"] & model2["training_model2.py\n(or training_modelX.py)"]
    yolo --> trainedModel["Trained Model"]
    model2 --> trainedModel
    trainedModel --> testing["Model Testing"]
    testingIndex --> testing
    testing --> metadata["Metadata Output"]
    metadata --> evaluation["evaluation.py"] & clipping["clipping.py"]
    evaluation --> metrics["Evaluation Metrics\n(Accuracy, Precision, Recall, F1, F2, etc.)"]
    clipping --> clippedVideo["Result Clipped Video"]

     indexCSV:::dataStyle
     readClips:::scriptStyle
     processedIndex:::dataStyle
     splitting:::scriptStyle
     trainingIndex:::dataStyle
     testingIndex:::dataStyle
     preprocessing:::scriptStyle
     frameData:::dataStyle
     training:::scriptStyle
     modelChoice:::scriptStyle
     yolo:::scriptStyle
     model2:::scriptStyle
     trainedModel:::scriptStyle
     testing:::scriptStyle
     metadata:::dataStyle
     evaluation:::scriptStyle
     clipping:::scriptStyle
     metrics:::outputStyle
     clippedVideo:::outputStyle
    classDef scriptStyle stroke:#6366f1,fill:#eef2ff
    classDef dataStyle stroke:#2dd4bf,fill:#f0fdfa
    classDef outputStyle stroke:#f59e0b,fill:#fff7ed
    style indexCSV stroke:#FF6D00,fill:#FFE0B2
```
