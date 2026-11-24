# SafeVision AI

**AI-Powered Workplace Safety & PPE Compliance Monitoring**  
Computer Vision · Deep Learning · PyTorch · Ultralytics YOLO26 · Streamlit

SafeVision AI detects people, visible PPE objects, and dataset-defined missing-PPE regions in images and videos. It is a decision-support tool for human safety officers, not a legal safety certification system.

## What is included

- Streamlit application for image, camera snapshot, and sampled video analysis.
- YOLO26n / YOLO26s fine-tuning pipeline.
- Reproducible experiment configuration with AdamW, weight decay, cosine LR, augmentation, Early Stopping, and checkpointing.
- Dataset audit and class distribution report.
- New same-dataset holdout split created from a portion of the original train split.
- Controlled model comparison with per-class missing-PPE metrics.
- Training/validation curve diagnostics for possible overfitting signals.
- Holdout evaluation and inference-latency benchmarking.
- Optional weight-decay ablation.
- Student-oriented Arabic learning and methodology guides.

## Important status

The previously supplied Colab run contains historical validation results:

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| YOLO26n | 85.88% | 68.88% | 74.90% | 53.41% |
| YOLO26s | 87.76% | 76.79% | 81.55% | 60.01% |

These are **validation results from the earlier experiment**, not independent test results. The older comparison used batch 8 for Nano and batch 4 for Small. The old `best.pt` was exported from Nano because `USE_SMALL=False` in that notebook.

The improved experiment pipeline is prepared, but it has **not been trained yet**. Do not claim improved metrics until the new Colab notebook has completed and artifacts are saved.

## Project structure

```text
SafeVision_AI_Torch/
├── app.py
├── app/
│   └── inference.py
├── data.yaml
├── training/
│   ├── hyperparameters.yaml
│   ├── train.py
│   ├── compare_models.py
│   ├── evaluate.py
│   ├── dataset_audit.py
│   ├── prepare_holdout.py
│   ├── plot_diagnostics.py
│   ├── benchmark_inference.py
│   └── regularization_ablation.py
├── notebooks/
│   ├── SafeVision_Colab_Training_Improved.ipynb
│   └── archive/ (historical baseline notebook)
├── models/
├── tests/
└── docs/
```

## Colab training (recommended)

1. Upload the complete `SafeVision_AI_Torch` folder to Google Drive as `SafeVision_AI`.
2. Open `notebooks/SafeVision_Colab_Training_Improved.ipynb` in Google Colab.
3. Select **Runtime → Change runtime type → GPU** (T4 or another available GPU).
4. Run the notebook from top to bottom.
5. Review dataset audit and split report before training.
6. Train YOLO26n and YOLO26s under identical settings.
7. Review training/validation curves, per-class missing-PPE recall, and whether all critical classes have valid validation metrics.
8. Allow the notebook to select a candidate using the documented safety-aware rule, then evaluate it once on the holdout.
9. The notebook writes the selected checkpoint to `models/best.pt`.

The first training run downloads the public pretrained YOLO26 weights and the Construction-PPE dataset when needed. Keep the Colab runtime connected and ensure Drive storage is available.

## Main training controls

See `training/hyperparameters.yaml`:

- Maximum epochs: 100.
- Early Stopping patience: 20 epochs without validation fitness improvement.
- Optimizer: AdamW.
- Initial learning rate: 0.001; final fraction: 0.01.
- Weight decay: 0.0005.
- Cosine learning-rate schedule.
- Batch size: 4 for both model variants.
- Image size: 640.
- Seed: 42 and deterministic mode.
- Data augmentation: HSV, translate, scale, horizontal flip, Mosaic; Mosaic is closed for the final 10 scheduled epochs.
- Best/last and periodic checkpoints.

Early Stopping is a training control, not proof that overfitting is impossible. Always inspect `results.csv` and diagnostic plots.

## Dataset split and evaluation integrity

The published dataset provides 800 train and 200 validation images, without an official test split. `training/prepare_holdout.py` creates a reproducible, multi-label-stratified holdout from the source training set and keeps the official 200 images as development validation. New models are retrained from public pretrained YOLO weights on the reduced training split; do not evaluate the old checkpoint on this new holdout because it saw all 800 source training images during its previous training.

The new holdout comes from the same dataset and is not external field testing. Collect and annotate genuinely new images from different sites/cameras before claiming real-world domain generalization.

## Local setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Place a trained checkpoint at `models/best.pt`, then run:

```bash
streamlit run app.py
```

On Windows, `setup_windows.bat` and `run_app.bat` are also provided.

## Useful commands

```bash
# Audit original dataset (will resolve/download via Ultralytics)
python training/dataset_audit.py --data-yaml data.yaml --output-dir runs/data_audit

# Build a new held-out split
python training/prepare_holdout.py --data-yaml data.yaml --output-dir datasets/safevision_holdout --test-size 0.10 --seed 42

# Train one model with the experiment config
python training/train.py --model yolo26n.pt --data datasets/safevision_holdout/data.safevision.yaml --name safevision_n

# Controlled comparison (same settings for both variants)
python training/compare_models.py --data datasets/safevision_holdout/data.safevision.yaml --epochs 100 --patience 20 --imgsz 640 --batch 4 --device 0

# Plot train/validation diagnostics
python training/plot_diagnostics.py runs/controlled_comparison/yolo26n/results.csv --output-dir runs/diagnostics/yolo26n

# Evaluate the selected checkpoint on the held-out test split (after model selection)
python training/evaluate.py --weights models/best.pt --data datasets/safevision_holdout/data.safevision.yaml --split test --device 0

# Benchmark one-image inference latency
python training/benchmark_inference.py --weights models/best.pt --source path/to/test-image.jpg --device 0
```

## Training and generalization guide

Read `docs/TRAINING_IMPROVEMENTS_AR.md` for the rationale behind regularization, Early Stopping, overfitting/underfitting diagnosis, feature learning, data leakage, and model selection. Read `docs/STUDY_GUIDE_AR.md` for the student-facing project explanation.

## Testing

```bash
pytest -q
python -m compileall -q app.py app training tests
```

The full inference tests require the project dependencies and a valid checkpoint for model-loading integration tests.

## Responsible use and limitations

- Missing-PPE boxes are visible missing-equipment regions, not guaranteed person-level violations.
- Counts in video are frame-level detections, not unique people/incidents.
- Small, occluded, low-light or unfamiliar scenes may cause misses.
- Human safety officers must review alerts.
- Obtain workplace-video permission and minimize personal-data retention.
- Review the dataset license and Ultralytics license terms before redistribution or commercial deployment.
