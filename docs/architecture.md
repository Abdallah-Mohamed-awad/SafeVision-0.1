# SafeVision AI Architecture

## Training-time architecture

```text
Construction-PPE Dataset
        |
        v
Dataset Audit + Annotation Validation
        |
        v
Multi-label Train / Development Validation / Holdout Split
        |
        v
YOLO26n / YOLO26s Pretrained Weights
        |
        v
PyTorch Fine-tuning (AdamW + Weight Decay + Augmentation)
        |
        v
Early Stopping + Best/Last Checkpoints + results.csv
        |
        v
Validation Metrics + Per-class Error Analysis
        |
        v
Model Selection (critical missing-PPE recall, mAP, size, latency)
        |
        v
One-time Holdout Evaluation
        |
        v
models/best.pt + experiment artifacts
```

## Inference-time architecture

```text
User image / camera snapshot / video
        |
        v
Streamlit Input Validation
        |
        v
Shared Model Loader (cached checkpoint)
        |
        v
Ultralytics YOLO26 Inference
        |
        v
Bounding Boxes + Class Names + Confidence Scores
        |
        v
Class-level counts and visible missing-PPE region summary
        |
        v
Annotated image/video + CSV report
```

## Boundaries

- A missing-PPE object is a visible region and is not automatically associated with a specific worker.
- Video counts are detections over sampled frames, not unique people or incidents.
- The system is a decision-support tool and requires human review.
- The new holdout is from the same public dataset; it does not replace external testing on new workplaces/cameras.
