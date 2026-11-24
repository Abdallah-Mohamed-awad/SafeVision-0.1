from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=str(ROOT / "models" / "best.pt"))
    parser.add_argument("--data", default=str(ROOT / "data.safevision.yaml"))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def per_class_rows(metrics: Any) -> list[dict[str, Any]]:
    box = metrics.box
    names = metrics.names
    rows = []
    for idx, class_id in enumerate(list(getattr(box, "ap_class_index", []))):
        class_id = int(class_id)
        name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
        def value(attr: str) -> float | None:
            try:
                return float(getattr(box, attr)[idx])
            except (TypeError, IndexError, ValueError):
                return None
        precision = value("p")
        recall = value("r")
        f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and (precision + recall) > 0 else 0.0
        rows.append({
            "class_id": class_id,
            "class_name": str(name),
            "precision": precision,
            "recall": recall,
            "f1": float(f1),
            "mAP50": value("ap50"),
            "mAP50_95": value("ap"),
        })
    return rows


def main() -> None:
    args = parse_args()
    weights = Path(args.weights).resolve()
    data_yaml = Path(args.data).resolve()
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml), split=args.split, imgsz=args.imgsz,
        device=args.device, plots=True,
        project=str(ROOT / "runs" / "final_evaluation"),
        name=f"{weights.stem}_{args.split}", exist_ok=True,
    )
    per_class = per_class_rows(metrics)
    result = {
        "evaluation_type": "held_out_same_dataset_test" if args.split == "test" else "development_validation",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "weights": str(weights),
        "data_yaml": str(data_yaml),
        "split": args.split,
        "imgsz": args.imgsz,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "macro_f1_over_reported_classes": float(sum(r["f1"] for r in per_class) / max(1, len(per_class))),
        "per_class": per_class,
        "limitations": [
            "The test split is held out from the source training split for the new controlled experiment.",
            "It is not an external site/camera/domain test set.",
            "The published 200-image validation split has been used for development and model selection."
        ] if args.split == "test" else ["Validation results are used for development/model selection, not as an independent test."],
    }
    output_dir = ROOT / "runs" / "final_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{weights.stem}_{args.split}_metrics.json"
    csv_path = output_dir / f"{weights.stem}_{args.split}_per_class.csv"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(result["per_class"]).to_csv(csv_path, index=False)
    print(json.dumps(result, indent=2))
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
