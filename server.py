"""SafeVision AI production server.

Run:  uvicorn server:app --host 0.0.0.0 --port 8000
Needs: pip install fastapi uvicorn python-multipart
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.inference import load_model, run_video

ROOT = Path(__file__).resolve().parent
WORK = Path(tempfile.gettempdir()) / "safevision"
WORK.mkdir(exist_ok=True)
CANDIDATES = [
    ROOT / "models" / "best.pt",
    ROOT / "models" / "candidates" / "yolo26s_best.pt",
    ROOT / "models" / "legacy_baseline_best.pt",
    ROOT / "models" / "candidates" / "yolo26n_best.pt",
]

app = FastAPI(title="SafeVision AI")
_lock = threading.Lock()  # one inference at a time; the model is not thread-safe
_model = None


def find_weights() -> Path | None:
    return next((p for p in CANDIDATES if p.is_file() and p.stat().st_size > 1_000_000), None)


def get_model():
    global _model
    if _model is None:
        w = find_weights()
        if w is None:
            raise HTTPException(503, "No trained weights found in models/.")
        _model = load_model(w)
    return _model


@app.get("/api/health")
def health():
    w = find_weights()
    return {"ok": w is not None, "weights": w.name if w else None}


@app.post("/api/detect")
def detect(file: UploadFile = File(...), conf: float = Form(0.35), imgsz: int = Form(640)):
    try:
        img = Image.open(file.file).convert("RGB")
    except Exception:
        raise HTTPException(400, "That file is not a readable image.")
    model = get_model()
    with _lock:
        r = model.predict(source=np.array(img)[:, :, ::-1], conf=conf, imgsz=imgsz, verbose=False)[0]
    boxes, counts = [], Counter()
    if r.boxes is not None:
        for xyxyn, c, s in zip(r.boxes.xyxyn.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            name = str(model.names.get(int(c), int(c)))
            counts[name] += 1
            boxes.append({"cls": name, "conf": round(s, 3), "box": [round(v, 4) for v in xyxyn]})
    return {"boxes": boxes, "counts": counts}


@app.post("/api/video")
def video(file: UploadFile = File(...), conf: float = Form(0.35), imgsz: int = Form(640),
          stride: int = Form(2), max_frames: int = Form(600)):
    for old in WORK.iterdir():  # drop jobs older than 1 hour
        if old.is_dir() and time.time() - old.stat().st_mtime > 3600:
            shutil.rmtree(old, ignore_errors=True)
    job = uuid.uuid4().hex
    d = WORK / job
    d.mkdir()
    src = d / ("input" + Path(file.filename or "v.mp4").suffix)
    with src.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    model = get_model()
    try:
        with _lock:
            out, counts, frames = run_video(model, src, d / "annotated.mp4", conf=conf, imgsz=imgsz,
                                            frame_stride=stride, max_frames=max_frames)
    except Exception as exc:
        raise HTTPException(400, str(exc))
    final = d / "result.mp4"
    if shutil.which("ffmpeg"):  # mp4v does not play in browsers; re-encode to H.264
        try:
            subprocess.run(["ffmpeg", "-y", "-i", str(out), "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                            str(final)], check=True, capture_output=True)
        except Exception:
            shutil.move(out, final)
    else:
        shutil.move(out, final)
    return {"url": f"/api/video/{job}", "counts": counts, "frames": frames}


@app.get("/api/video/{job}")
def get_video(job: str):
    p = WORK / job / "result.mp4"
    if not job.isalnum() or not p.is_file():
        raise HTTPException(404, "Result expired.")
    return FileResponse(p, media_type="video/mp4", filename="safevision_annotated.mp4")


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
