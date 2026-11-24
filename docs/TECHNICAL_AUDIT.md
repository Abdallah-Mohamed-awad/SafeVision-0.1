# SafeVision AI — Technical Audit of the Previous Torch Version

**Audit basis:** project source archive, executed Colab notebook `SafeVision_Colab_Training (1).ipynb`, and uploaded `best.pt` file. This audit distinguishes observations from the previous run and changes prepared in the improved release.

## Verified from the executed notebook

- Environment reported Python 3.13.16, Ultralytics 8.4.216, PyTorch 2.14.1+cu130, NVIDIA Tesla T4.
- Dataset: 800 training images, 200 validation images, 12 classes, 2,961 instances in the 200-image validation subset.
- Both YOLO26n and YOLO26s used pretrained weights, 50 maximum epochs, image size 640, seed 42, `patience=12`, `cos_lr=True`, `close_mosaic=10`, AMP, and plots.
- YOLO26n used `batch=8`; YOLO26s used `batch=4`.
- `optimizer=auto` selected AdamW in the observed run and ignored the supplied `lr0`; log showed `AdamW(lr=0.000625, momentum=0.9)` and optimizer parameter groups with weight decay 0.0005 on eligible weights.
- Recorded overall validation results were:

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| YOLO26n | 0.858755 | 0.688773 | 0.748959 | 0.534114 |
| YOLO26s | 0.877609 | 0.767870 | 0.815548 | 0.600138 |

- `RUN_SMALL=True` trained YOLO26s, but `USE_SMALL=False` selected YOLO26n for export.
- The export cell saved a 5.19 MiB checkpoint to `models/best.pt`. This size is consistent with the Nano checkpoint downloaded/trained in the earlier notebook, not with the approximately 19.5 MB Small pretrained weight file. Checkpoint metadata should still be inspected when using the artifact.
- The original dataset YAML describes a train/validation split and no official independent test split.

## Issues and risks identified

1. **Comparison fairness:** the old comparison used different batch sizes (8 vs 4), so it is not fully controlled.
2. **Checkpoint mismatch:** the old notebook exported Nano despite better overall validation results for Small.
3. **Validation reuse:** the official 200-image validation split was already used for model comparison; it cannot be presented as an untouched independent test set.
4. **Legacy checkpoint leakage:** old `best.pt` was trained on all 800 source-training images. It cannot be fairly evaluated on a new holdout extracted from those same images.
5. **Incomplete data audit:** the previous pipeline did not provide a full invalid-label, duplicate, class-frequency, and box-geometry audit.
6. **Incomplete diagnostics:** plots were requested, but the user did not provide the complete run directory/results.csv artifacts for this audit. Overfitting cannot be conclusively judged from the notebook excerpt alone.
7. **Feature engineering wording:** YOLO automatically learns visual features; manual HOG/SIFT engineering is not a required stage for this detector. Dataset/annotation engineering remains important.
8. **Class-imbalance treatment:** the old run did not implement a dedicated class-balanced sampler or class-specific reweighting. Class counts and per-class metrics should be analyzed before adding such mechanisms.

## Improvements prepared in this release

- Centralized reproducible training configuration at `training/hyperparameters.yaml`.
- Dataset and annotation audit at `training/dataset_audit.py`.
- New same-dataset holdout split at `training/prepare_holdout.py` with multi-label stratification and exact duplicate exclusion.
- Controlled Nano/Small comparison with identical settings in `training/compare_models.py`, including per-class F1, critical-class recall/F1, and single-image latency measurements.
- Explicit AdamW, LR, weight decay, warmup, cosine schedule, Early Stopping, augmentation, and checkpointing.
- `training/plot_diagnostics.py` for training/validation curves and cautious divergence warnings.
- `training/evaluate.py` supports a final held-out `test` evaluation.
- `training/benchmark_inference.py` measures inference latency on the chosen hardware.
- Optional one-factor weight-decay experiment in `training/regularization_ablation.py`.
- Updated Colab notebook, report, presentation, file map, and student-facing guide.

## Not yet verified / must be executed

- The improved notebook has not been executed in this environment because no Ultralytics package, dataset download, or CUDA GPU is available here.
- No new model checkpoint or improved metrics are claimed.
- The newly prepared holdout is from the same dataset; external images from new sites/cameras remain necessary for real-world generalization claims.
- The optional weight-decay ablation must be run before claiming which weight-decay value performed best.
