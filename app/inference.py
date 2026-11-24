"""Inference helpers for SafeVision AI."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
try:
    from ultralytics import YOLO
except ImportError:  # lets pure utility tests run without the optional inference dependency
    YOLO = None  # type: ignore[assignment]

VIOLATION_CLASSES = {"no_helmet", "no_vest", "no_gloves", "no_boots", "no_goggle"}
PPE_CLASSES = {"helmet", "vest", "gloves", "boots", "goggles"}


def load_model(weights_path: str | Path) -> YOLO:
    if YOLO is None:
        raise RuntimeError("Ultralytics is not installed. Install project dependencies with: pip install -r requirements.txt")
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained weights not found: {path}. Run the Colab training notebook, "
            "then place the resulting best.pt in the models/ folder."
        )
    return YOLO(str(path))



def run_image(
    model: YOLO,
    image: Image.Image,
    conf: float = 0.35,
    imgsz: int = 640,
) -> tuple[np.ndarray, Counter, list[dict[str, Any]]]:

    rgb = np.array(image.convert("RGB"))

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    result = model.predict(
        source=bgr,
        conf=conf,
        imgsz=imgsz,
        verbose=False
    )[0]

    annotated_rgb = np.asarray(result.plot(pil=True))

    counts = Counter()
    rows = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = str(model.names.get(class_id, class_id))
            score = float(box.conf.item())

            counts[class_name] += 1

            rows.append({
                "Class": class_name,
                "Confidence": round(score, 4)
            })

    return annotated_rgb, counts, rows



def run_video(
    model: YOLO,
    input_path: str | Path,
    output_path: str | Path,
    conf: float = 0.35,
    imgsz: int = 640,
    frame_stride: int = 2,
    max_frames: int = 600,
    progress_callback=None,
) -> tuple[Path, Counter, int]:
    """Annotate sampled frames and return aggregate detection counts.

    Counts are frame-level detections, not unique people or unique incidents.
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError("Could not open the uploaded video. Try MP4 (H.264) or AVI.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), max(fps / max(frame_stride, 1), 1), (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create the output video file.")

    counts: Counter = Counter()
    frame_idx = 0
    processed = 0
    try:
        while processed < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % max(frame_stride, 1) == 0:
                result = model.predict(source=frame, conf=conf, imgsz=imgsz, verbose=False)[0]
                annotated = result.plot()
                writer.write(annotated)
                if result.boxes is not None:
                    for cls_id in result.boxes.cls.tolist():
                        counts[str(model.names.get(int(cls_id), int(cls_id)))] += 1
                processed += 1
                if progress_callback and total > 0:
                    progress_callback(min(frame_idx / total, 1.0))
            frame_idx += 1
    finally:
        cap.release()
        writer.release()
    return out_path, counts, processed


def summarize_counts(counts: Counter) -> dict[str, int]:
    return {
        "people": int(counts.get("Person", 0)),
        "helmet_detected": int(counts.get("helmet", 0)),
        "vest_detected": int(counts.get("vest", 0)),
        "missing_helmet_regions": int(counts.get("no_helmet", 0)),
        "missing_vest_regions": int(counts.get("no_vest", 0)),
        "other_missing_ppe_regions": int(sum(counts.get(k, 0) for k in ("no_gloves", "no_boots", "no_goggle"))),
    }
