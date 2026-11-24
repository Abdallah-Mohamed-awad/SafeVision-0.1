# SafeVision AI — Presentation Speaker Notes (English)

## Slide 1 — Title
Introduce SafeVision AI as a computer-vision and deep-learning decision-support tool for workplace PPE monitoring. Explain that the project detects visible PPE and dataset-defined missing-PPE regions.

## Slide 2 — Problem
Manual PPE inspection is repetitive and difficult to scale. Image conditions such as lighting, distance, crowding and occlusion make the task challenging. The system highlights visual evidence for human review; it does not replace safety officers.

## Slide 3 — Scope
Explain the 12 dataset classes, including people, PPE items, forklift and visible missing-PPE regions. Be precise: a `no_helmet` detection is a region, not necessarily an identified worker-level violation. Person association and tracking are outside the validated first release.

## Slide 4 — Dataset
The published Construction-PPE dataset contains 1,000 images, 14,863 annotated boxes and 12 classes. Its published split has 800 training and 200 validation images, without an official independent test split. The source license requires attribution.

## Slide 5 — Pipeline
Walk through data audit, split creation, YOLO training, per-class evaluation and Streamlit inference. Explain that data preparation and evaluation are part of the AI system, not just the final dashboard.

## Slide 6 — Deep Learning
YOLO26 predicts object classes, confidence and bounding boxes. We use pretrained weights and fine-tune them on PPE data. The network learns visual features internally; we do not manually construct HOG/SIFT features for the main detector.

## Slide 7 — Previous Baseline Results
These are recorded validation results from the earlier experiments. YOLO26s had higher overall validation Precision, Recall, mAP50 and mAP50-95. The earlier comparison used batch size 8 for Nano and 4 for Small, so the experiment was not fully controlled. The old notebook exported Nano due to `USE_SMALL=False`. Do not call these independent test metrics.

## Slide 8 — Improved Training
The revised run is configured for up to 100 epochs, patience 20, AdamW, learning rate 0.001, weight decay 0.0005, warmup, cosine LR, augmentation, batch 4 for both model variants, and periodic checkpoints. The new controlled run must still be executed on GPU before claiming improved results.

## Slide 9 — Generalization
Explain the difference between overfitting and underfitting. We inspect training and validation losses together and monitor validation mAP. Regularization includes weight decay and data augmentation; Early Stopping controls training duration. No single setting guarantees generalization. Dropout is not added blindly to YOLO.

## Slide 10 — Evaluation
Precision measures prediction correctness; Recall measures ground-truth coverage; mAP measures localization/classification quality across IoU thresholds. For safety, we also examine per-class Recall for `no_helmet`, `no_vest`, `no_goggle`, `no_gloves` and `no_boots`. Selection also considers latency and checkpoint size.

## Slide 11 — Application
The Streamlit interface accepts images, camera snapshots and videos, runs the selected checkpoint, displays annotated media and exports class-level summaries. Video totals are frame-level detections, not unique workers.

## Slide 12 — Limitations
The model may miss small or occluded objects and may degrade under new lighting, cameras or worksites. A same-dataset holdout is not external domain testing. Workplace video requires authorization and privacy safeguards. Human review remains essential.

## Slide 13 — Conclusion
Summarize the practical AI pipeline and the next steps: execute the improved GPU run, inspect curves and per-class errors, choose a model on validation, evaluate once on the holdout, measure deployment latency, and conduct external field testing before broad generalization claims.

---

# Important presenter rule
Only present metrics that exist in the saved experiment artifacts. The earlier results are validation-only. Do not claim that the improved 100-epoch/early-stopping experiment or the optional weight-decay ablation has completed until you have executed it and retained its output files.
