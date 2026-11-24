# Model weights

The improved Colab notebook writes the newly selected checkpoint to `models/best.pt` after the controlled training and validation comparison.

The provided legacy checkpoint (if present as `legacy_baseline_best.pt`) is from the previous YOLO26n run trained on all 800 original training images. It is retained for reference only. It must not be used to evaluate a holdout split later extracted from those same 800 images.
