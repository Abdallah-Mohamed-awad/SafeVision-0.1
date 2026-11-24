"""Reproducible fine-tuning for SafeVision AI with explicit regularization.

Examples:
  python training/train.py --model yolo26n.pt --data data.safevision.yaml --name safevision_n
  python training/train.py --model yolo26s.pt --data data.safevision.yaml --batch 4

This script uses the official Ultralytics training pipeline. Early stopping is
controlled by `patience`; best.pt is the best checkpoint selected on validation.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "training" / "hyperparameters.yaml"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SafeVision PPE detector")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data", default=None, help="Dataset YAML. Defaults to the YAML in the config.")
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--name", default="safevision_training")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def build_train_kwargs(config: dict[str, Any], args: argparse.Namespace, data_yaml: Path, device: str) -> dict[str, Any]:
    hp = dict(config.get("training", {}))
    aug = dict(config.get("augmentation", {}))
    kwargs: dict[str, Any] = {
        **hp,
        **aug,
        "data": str(data_yaml),
        "project": str(ROOT / "runs" / "detect"),
        "name": args.name,
        "exist_ok": False,
        "seed": int(config.get("seed", 42)),
        "deterministic": bool(config.get("deterministic", True)),
        "device": device,
        "verbose": True,
    }
    overrides = {"epochs": args.epochs, "batch": args.batch, "imgsz": args.imgsz, "patience": args.patience}
    kwargs.update({key: val for key, val in overrides.items() if val is not None})
    if args.resume:
        kwargs["resume"] = True
    return kwargs


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    data_yaml = Path(args.data or config.get("data", "data.safevision.yaml"))
    if not data_yaml.is_absolute():
        data_yaml = (ROOT / data_yaml).resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}. Run prepare_holdout.py first.")

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("WARNING: CPU training will be slow; a Colab GPU is recommended.")

    model = YOLO(args.model)
    kwargs = build_train_kwargs(config, args, data_yaml, device)
    print("Training configuration:")
    print(json.dumps({k: v for k, v in kwargs.items() if k not in {"data", "project"}}, indent=2, default=str))
    model.train(**kwargs)

    save_dir = Path(model.trainer.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    last_weights = save_dir / "weights" / "last.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Training finished but best.pt was not found in {best_weights}")

    candidate_dir = ROOT / "models" / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    variant = Path(args.model).stem
    candidate = candidate_dir / f"{variant}_best.pt"
    shutil.copy2(best_weights, candidate)
    if last_weights.exists():
        shutil.copy2(last_weights, candidate_dir / f"{variant}_last.pt")
    manifest = {
        "model_variant": variant,
        "initial_weights": args.model,
        "best_checkpoint": str(candidate),
        "last_checkpoint": str(candidate_dir / f"{variant}_last.pt") if last_weights.exists() else None,
        "run_directory": str(save_dir),
        "dataset_yaml": str(data_yaml),
        "training_arguments": kwargs,
        "note": "Validation-selected checkpoint; do not call validation metrics independent test results.",
    }
    (candidate_dir / f"{variant}_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"\nBest checkpoint copied to: {candidate}")
    print(f"Full run artifacts: {save_dir}")
    print("Next: inspect results.csv, run diagnostics, and evaluate the selected model on the untouched holdout test split.")


if __name__ == "__main__":
    main()
