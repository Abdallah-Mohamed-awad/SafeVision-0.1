"""Controlled YOLO26n vs YOLO26s comparison with matching training settings."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
from typing import Any

import pandas as pd
import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
CRITICAL_CLASSES = {"no_helmet", "no_vest", "no_goggle", "no_gloves", "no_boots"}


def read_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def per_class_metrics(val_result: Any) -> list[dict[str, Any]]:
    box = val_result.box
    names = val_result.names
    rows = []
    class_ids = list(getattr(box, "ap_class_index", []))
    for idx, class_id in enumerate(class_ids):
        class_id = int(class_id)
        name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
        def safe_value(attr: str) -> float | None:
            values = getattr(box, attr, None)
            try:
                return float(values[idx])
            except (TypeError, IndexError, ValueError):
                return None
        precision = safe_value("p")
        recall = safe_value("r")
        f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and (precision + recall) > 0 else 0.0
        rows.append({
            "class_id": class_id,
            "class_name": str(name),
            "precision": precision,
            "recall": recall,
            "f1": float(f1),
            "mAP50": safe_value("ap50"),
            "mAP50_95": safe_value("ap"),
        })
    return rows


def best_epoch_from_csv(run_dir: Path) -> int | None:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return None
    frame = pd.read_csv(csv_path)
    frame.columns = [c.strip() for c in frame.columns]
    metric_col = next((c for c in frame.columns if "mAP50-95" in c), None)
    if metric_col is None or frame.empty:
        return None
    return int(frame[metric_col].astype(float).idxmax()) + 1



def measure_latency(model: YOLO, image_path: Path, device: str, imgsz: int, warmup: int = 5, runs: int = 20) -> dict[str, Any]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load latency sample image: {image_path}")
    for _ in range(warmup):
        model.predict(image, imgsz=imgsz, device=device, verbose=False)
    if str(device).isdigit() and torch.cuda.is_available():
        torch.cuda.synchronize()
    durations = []
    for _ in range(runs):
        start = time.perf_counter()
        model.predict(image, imgsz=imgsz, device=device, verbose=False)
        if str(device).isdigit() and torch.cuda.is_available():
            torch.cuda.synchronize()
        durations.append((time.perf_counter() - start) * 1000)
    mean_ms = sum(durations) / len(durations)
    ordered = sorted(durations)
    return {
        "latency_sample": str(image_path),
        "latency_warmup_runs": warmup,
        "latency_measured_runs": runs,
        "latency_ms_mean": round(mean_ms, 3),
        "latency_ms_median": round(ordered[len(ordered) // 2], 3),
        "approx_fps_from_latency": round(1000 / mean_ms, 2),
        "latency_note": "Single-image predict-call latency on the current device, including Ultralytics/Python overhead; measure again on deployment hardware."
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "training" / "hyperparameters.yaml"))
    parser.add_argument("--data", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--models", nargs="+", default=["yolo26n.pt", "yolo26s.pt"])
    args = parser.parse_args()

    cfg = read_config(Path(args.config))
    hp = dict(cfg.get("training", {}))
    aug = dict(cfg.get("augmentation", {}))
    data_yaml = Path(args.data or cfg.get("data", "data.safevision.yaml"))
    if not data_yaml.is_absolute():
        data_yaml = (ROOT / data_yaml).resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML missing: {data_yaml}")

    # Apply identical values for both variants. A common batch size makes the
    # comparison more controlled than the original 8-vs-4 experiment.
    if args.epochs is not None: hp["epochs"] = args.epochs
    if args.batch is not None: hp["batch"] = args.batch
    if args.imgsz is not None: hp["imgsz"] = args.imgsz
    if args.patience is not None: hp["patience"] = args.patience
    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    from ultralytics.data.utils import check_det_dataset
    dataset_info = check_det_dataset(str(data_yaml))
    raw_val = dataset_info["val"]
    val_dir = Path(raw_val[0] if isinstance(raw_val, (list, tuple)) else raw_val)
    sample_image = next((p for p in sorted(val_dir.rglob("*")) if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}), None)
    if sample_image is None:
        raise FileNotFoundError(f"No validation image found under {val_dir}")

    comparison_root = ROOT / "runs" / "controlled_comparison"
    candidates_dir = ROOT / "models" / "candidates"
    comparison_root.mkdir(parents=True, exist_ok=True)
    candidates_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for model_name in args.models:
        run_name = Path(model_name).stem
        print(f"\n===== Controlled training: {model_name} =====")
        model = YOLO(model_name)
        kwargs = {
            **hp,
            **aug,
            "data": str(data_yaml),
            "device": device,
            "workers": int(hp.get("workers", 2)),
            "project": str(comparison_root),
            "name": run_name,
            "exist_ok": False,
            "seed": int(cfg.get("seed", 42)),
            "deterministic": bool(cfg.get("deterministic", True)),
            "pretrained": True,
            "plots": True,
            "save": True,
            "verbose": True,
        }
        started = time.perf_counter()
        model.train(**kwargs)
        training_seconds = time.perf_counter() - started
        run_dir = Path(model.trainer.save_dir)
        best_path = run_dir / "weights" / "best.pt"
        if not best_path.exists():
            raise FileNotFoundError(f"Missing best checkpoint for {model_name}: {best_path}")
        candidate_path = candidates_dir / f"{run_name}_best.pt"
        shutil.copy2(best_path, candidate_path)

        val_model = YOLO(str(best_path))
        val_result = val_model.val(
            data=str(data_yaml), split="val", imgsz=int(hp.get("imgsz", 640)),
            device=device, plots=True,
            project=str(comparison_root / "validation"), name=run_name, exist_ok=True,
        )
        pc = per_class_metrics(val_result)
        latency = measure_latency(val_model, sample_image, device, int(hp.get("imgsz", 640)))
        critical_rows = [r for r in pc if r["class_name"] in CRITICAL_CLASSES and r["recall"] is not None]
        critical_recalls = [r["recall"] for r in critical_rows]
        critical_f1 = [r["f1"] for r in critical_rows]
        all_class_f1 = [r["f1"] for r in pc if r["f1"] is not None]
        observed_critical = {r["class_name"] for r in critical_rows}
        missing_critical = sorted(CRITICAL_CLASSES - observed_critical)
        size_mb = candidate_path.stat().st_size / (1024 * 1024)
        params = sum(p.numel() for p in val_model.model.parameters())
        record = {
            "model": run_name,
            "checkpoint": str(candidate_path),
            "run_directory": str(run_dir),
            "training_seconds": round(training_seconds, 2),
            "best_epoch_in_results_csv": best_epoch_from_csv(run_dir),
            "checkpoint_size_mb": round(size_mb, 3),
            "parameters": int(params),
            **latency,
            "precision": float(val_result.box.mp),
            "recall": float(val_result.box.mr),
            "mAP50": float(val_result.box.map50),
            "mAP50_95": float(val_result.box.map),
            "macro_f1_over_reported_classes": float(sum(all_class_f1) / len(all_class_f1)) if all_class_f1 else None,
            "critical_missing_ppe_macro_recall": float(sum(critical_recalls) / len(critical_recalls)) if critical_recalls else None,
            "critical_missing_ppe_macro_f1": float(sum(critical_f1) / len(critical_f1)) if critical_f1 else None,
            "critical_classes_missing_from_validation_metrics": missing_critical,
            "per_class": pc,
            "training_arguments": kwargs,
            "evaluation_type": "development_validation_not_independent_test",
        }
        records.append(record)
        print(json.dumps({k: record[k] for k in ("model", "precision", "recall", "macro_f1_over_reported_classes", "mAP50", "mAP50_95", "critical_missing_ppe_macro_recall", "critical_missing_ppe_macro_f1", "critical_classes_missing_from_validation_metrics", "latency_ms_mean", "approx_fps_from_latency", "checkpoint_size_mb", "parameters", "best_epoch_in_results_csv")}, indent=2))

    out_dir = comparison_root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    table = pd.DataFrame([{k: v for k, v in r.items() if k not in {"per_class", "training_arguments"}} for r in records])
    table.to_csv(out_dir / "comparison.csv", index=False)
    per_class = pd.DataFrame([{"model": r["model"], **m} for r in records for m in r["per_class"]])
    per_class.to_csv(out_dir / "per_class_metrics.csv", index=False)
    print(f"\nSaved controlled comparison to {out_dir}")
    print("Choose the final model using critical-class recall first, then mAP50-95, measured latency, and size. Do not select on overall mAP alone.")
    print("No independent test results are produced by this script.")


if __name__ == "__main__":
    main()
