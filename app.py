from __future__ import annotations

import hashlib
import io
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from app.inference import (
    PPE_CLASSES,
    VIOLATION_CLASSES,
    load_model,
    run_image,
    run_video,
    summarize_counts,
)


# -----------------------------------------------------------------------------
# Application configuration
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
MAX_IMAGE_MB = 30
MAX_VIDEO_MB = 250
MODEL_CANDIDATES = [
    ROOT / "models" / "best.pt",
    ROOT / "models" / "candidates" / "yolo26s_best.pt",
    ROOT / "models" / "legacy_baseline_best.pt",
    ROOT / "models" / "candidates" / "yolo26n_best.pt",
]

st.set_page_config(
    page_title="SafeVision AI | PPE Inspection",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Visual system
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
      :root {
        --sv-bg: #0a111d;
        --sv-panel: #111d2c;
        --sv-panel-soft: #142337;
        --sv-line: rgba(150, 178, 205, .16);
        --sv-text: #edf4fb;
        --sv-muted: #92a6ba;
        --sv-teal: #42d6bd;
        --sv-blue: #72b4ff;
        --sv-amber: #f3bd66;
        --sv-red: #ff8585;
      }
      html, body, [class*="css"] {
        font-family: Inter, "Segoe UI", Arial, sans-serif;
      }
      .stApp {
        color: var(--sv-text);
        background:
          radial-gradient(ellipse at 82% 0%, rgba(39, 122, 133, .17), transparent 32%),
          radial-gradient(ellipse at 5% 25%, rgba(46, 85, 139, .13), transparent 30%),
          var(--sv-bg);
      }
      .block-container {
        max-width: 1500px;
        padding-top: 1.25rem;
        padding-bottom: 3rem;
      }
      [data-testid="stSidebar"] {
        background: #0c1725;
        border-right: 1px solid var(--sv-line);
      }
      [data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
      h1, h2, h3 { letter-spacing: -.035em; }
      h1 { font-weight: 820 !important; }
      h2, h3 { font-weight: 740 !important; }
      p, li { line-height: 1.55; }

      .sv-topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        padding: .35rem 0 1rem;
        border-bottom: 1px solid var(--sv-line);
        margin-bottom: 1.25rem;
      }
      .sv-brand { display:flex; align-items:center; gap:.8rem; }
      .sv-brand-mark {
        width: 42px; height: 42px; display:flex; align-items:center; justify-content:center;
        border-radius: 13px; color:#06211f; font-size:1.35rem; font-weight:900;
        background: linear-gradient(145deg, #56e0c7, #78baff);
        box-shadow: 0 8px 24px rgba(66,214,189,.16);
      }
      .sv-brand-name { font-size:1.12rem; font-weight:850; letter-spacing:-.04em; color:#f2f7fb; }
      .sv-brand-sub { color:var(--sv-muted); font-size:.72rem; margin-top:.05rem; }
      .sv-live {
        display:inline-flex; align-items:center; gap:.5rem; padding:.45rem .75rem;
        border:1px solid rgba(66,214,189,.28); border-radius:999px;
        background:rgba(66,214,189,.08); color:#baf6eb; font-size:.73rem; font-weight:750;
      }
      .sv-live-dot { width:7px; height:7px; border-radius:50%; background:var(--sv-teal); box-shadow:0 0 12px rgba(66,214,189,.65); }
      .sv-title { font-size:clamp(1.75rem, 3.1vw, 2.55rem); line-height:1.08; font-weight:850; letter-spacing:-.06em; color:#f3f8fc; margin:0; }
      .sv-title span { color:var(--sv-teal); }
      .sv-subtitle { color:#a0b3c6; font-size:.95rem; max-width:850px; margin-top:.55rem; }
      .sv-eyebrow { color:var(--sv-teal); font-size:.68rem; font-weight:850; letter-spacing:.18em; text-transform:uppercase; margin-bottom:.45rem; }
      .sv-section { display:flex; align-items:center; gap:.65rem; margin:1.5rem 0 .7rem; color:#dfeaf4; font-size:1.02rem; font-weight:780; }
      .sv-section-index { display:inline-flex; justify-content:center; align-items:center; width:25px; height:25px; border-radius:8px; color:#bff8ee; background:rgba(66,214,189,.1); border:1px solid rgba(66,214,189,.24); font-size:.7rem; }
      .sv-muted { color:var(--sv-muted); }
      .sv-card {
        border:1px solid var(--sv-line); border-radius:17px; padding:1rem 1.05rem;
        background:linear-gradient(145deg, rgba(20,36,55,.9), rgba(15,28,44,.9));
      }
      .sv-card-label { color:#a4b7c9; font-size:.72rem; font-weight:750; letter-spacing:.045em; text-transform:uppercase; }
      .sv-card-value { color:#f4f8fb; font-size:1.85rem; line-height:1.2; letter-spacing:-.055em; font-weight:850; margin-top:.25rem; }
      .sv-card-note { color:#8299ae; font-size:.72rem; margin-top:.24rem; }
      .sv-callout {
        border-left:3px solid var(--sv-teal); padding:.72rem .9rem;
        color:#b7c8d8; background:rgba(66,214,189,.06); border-radius:0 10px 10px 0; font-size:.83rem;
      }
      .sv-warning {
        border-left:3px solid var(--sv-amber); padding:.72rem .9rem;
        color:#e1cda8; background:rgba(243,189,102,.07); border-radius:0 10px 10px 0; font-size:.82rem;
      }
      .sv-result-heading { display:flex; align-items:center; justify-content:space-between; gap:1rem; margin:1.35rem 0 .7rem; }
      .sv-result-heading h3 { margin:0; }
      .sv-result-pill { color:#c2f6eb; background:rgba(66,214,189,.1); border:1px solid rgba(66,214,189,.28); border-radius:999px; padding:.35rem .65rem; font-size:.7rem; font-weight:800; white-space:nowrap; }
      .sv-detection-row { display:flex; justify-content:space-between; align-items:center; gap:.7rem; padding:.58rem 0; border-bottom:1px solid rgba(150,178,205,.11); }
      .sv-detection-row:last-child { border-bottom:0; }
      .sv-detection-name { color:#dce8f2; font-size:.84rem; font-weight:700; }
      .sv-detection-count { color:#f0f6fb; font-size:.84rem; font-weight:850; font-variant-numeric:tabular-nums; }
      .sv-footer { color:#71879b; border-top:1px solid var(--sv-line); margin-top:2.5rem; padding-top:1rem; font-size:.74rem; }
      [data-testid="stMetric"] { background:rgba(18,33,50,.76); border:1px solid var(--sv-line); border-radius:14px; padding:.7rem .9rem; }
      [data-testid="stMetricLabel"] { color:#a4b7c9; }
      [data-testid="stMetricValue"] { color:#f4f8fb; }
      div[role="radiogroup"] { gap:.45rem; flex-wrap:wrap; }
      div[role="radiogroup"] > label { border:1px solid var(--sv-line); border-radius:11px; padding:.48rem .75rem; background:rgba(17,30,46,.65); }
      div[role="radiogroup"] > label:has(input:checked) { border-color:rgba(66,214,189,.5); background:rgba(66,214,189,.12); }
      .stButton > button, .stDownloadButton > button { border-radius:10px; min-height:2.65rem; font-weight:760; }
      .stButton > button[kind="primary"] { background:linear-gradient(100deg,#42d6bd,#78baff); border:0; color:#071922; }
      [data-testid="stFileUploader"] { border:1px dashed rgba(132,166,197,.35); border-radius:14px; background:rgba(14,27,43,.45); }
      [data-testid="stTabs"] button { font-weight:720; }
      [data-testid="stExpander"] { border:1px solid var(--sv-line); border-radius:13px; background:rgba(15,28,43,.48); }
      .stProgress > div > div > div > div { background:var(--sv-teal); }
      @media (max-width: 760px) {
        .sv-topbar { align-items:flex-start; }
        .sv-live { display:none; }
        .sv-card-value { font-size:1.5rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Model and utility functions
# -----------------------------------------------------------------------------
def resolve_weights() -> Path | None:
    for candidate in MODEL_CANDIDATES:
        if candidate.is_file() and candidate.stat().st_size > 1_000_000:
            return candidate
    return None


@st.cache_resource(show_spinner="Loading the trained SafeVision model...")
def get_model(weights_path: str, modified_ns: int):
    # modified_ns invalidates the cache if the checkpoint is replaced in-place.
    _ = modified_ns
    return load_model(weights_path)


def fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_image_from_bytes(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as probe:
        probe.verify()
    with Image.open(io.BytesIO(data)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
    return image.copy()


def image_png_bytes(image_array: Any) -> bytes:
    image = Image.fromarray(image_array.astype("uint8"), mode="RGB")
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def csv_bytes(rows: list[dict], counts: Counter | dict) -> bytes:
    if rows:
        frame = pd.DataFrame(rows)
    else:
        frame = pd.DataFrame(
            [{"Class": name, "Count": int(value)} for name, value in counts.items()]
        )
    return frame.to_csv(index=False).encode("utf-8-sig")


def total_for(counts: Counter | dict, keys) -> int:
    return sum(int(counts.get(key, 0)) for key in keys)


def class_label(name: str) -> str:
    return str(name).replace("_", " ").title()


def technical_error(label: str, exc: Exception) -> None:
    st.error(label)
    with st.expander("Technical details", expanded=False):
        st.code(f"{type(exc).__name__}: {exc}")


def render_section(number: str, title: str, hint: str | None = None) -> None:
    hint_html = f'<span class="sv-muted" style="font-size:.78rem;font-weight:450">{hint}</span>' if hint else ""
    st.markdown(
        f'<div class="sv-section"><span class="sv-section-index">{number}</span>{title}{hint_html}</div>',
        unsafe_allow_html=True,
    )


def render_overview(counts: Counter | dict, *, is_video: bool = False) -> None:
    summary = summarize_counts(counts)
    total_detections = sum(int(value) for value in counts.values())
    ppe_detections = total_for(counts, PPE_CLASSES)
    missing_detections = total_for(counts, VIOLATION_CLASSES)

    labels = [
        ("People", summary.get("people", 0), "Person detections", "blue"),
        ("PPE items", ppe_detections, "Helmet · vest · gloves · boots · goggles", "teal"),
        ("Missing-PPE regions", missing_detections, "Dataset-defined labels", "amber"),
        ("Total detections", total_detections, "All detected objects and regions", "blue"),
    ]
    cols = st.columns(4, gap="medium")
    for col, (label, value, note, accent) in zip(cols, labels):
        with col:
            color = {"blue": "#72b4ff", "teal": "#42d6bd", "amber": "#f3bd66"}[accent]
            st.markdown(
                f'<div class="sv-card"><div class="sv-card-label">{label}</div>'
                f'<div class="sv-card-value" style="color:{color}">{int(value):,}</div>'
                f'<div class="sv-card-note">{note}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown(" ")
    if counts:
        left, right = st.columns([1.2, 1], gap="large")
        sorted_items = sorted(counts.items(), key=lambda item: int(item[1]), reverse=True)
        maximum = max(int(value) for _, value in sorted_items) or 1
        with left:
            st.markdown("**Detected class distribution**")
            for name, value in sorted_items:
                row = st.columns([1.5, 4, .45], vertical_alignment="center")
                row[0].markdown(f"<span style='color:#dce8f2;font-size:.83rem;font-weight:680'>{class_label(name)}</span>", unsafe_allow_html=True)
                row[1].progress(min(int(value) / maximum, 1.0))
                row[2].markdown(f"<span style='color:#f0f6fb;font-weight:800'>{int(value)}</span>", unsafe_allow_html=True)
        with right:
            st.markdown("**Detection summary**")
            for name, value in sorted_items:
                st.markdown(
                    f'<div class="sv-detection-row"><span class="sv-detection-name">{class_label(name)}</span>'
                    f'<span class="sv-detection-count">{int(value):,}</span></div>',
                    unsafe_allow_html=True,
                )
    else:
        st.info("No objects were detected above the selected confidence threshold. Try a slightly lower threshold or a clearer image.")

    if is_video:
        st.markdown(
            '<div class="sv-warning">Video totals are detections accumulated across sampled frames. They are not unique worker counts, unique violations, or incident counts.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="sv-callout">Each box and label is a model prediction. A missing-PPE region is not automatically linked to a specific person; review the visual evidence before acting.</div>',
            unsafe_allow_html=True,
        )


def render_detection_table(rows: list[dict], key: str) -> None:
    if not rows:
        st.info("No detection records are available for this result.")
        return
    frame = pd.DataFrame(rows).copy()
    if "Confidence" in frame.columns:
        frame["Confidence"] = frame["Confidence"].map(lambda value: f"{float(value):.1%}")
    if "Class" in frame.columns:
        frame["Class"] = frame["Class"].map(class_label)
    frame.insert(0, "Detection", [f"#{index:03d}" for index in range(1, len(frame) + 1)])
    st.dataframe(frame, width="stretch", hide_index=True, key=key)


def render_image_result(result: dict, result_key: str) -> None:
    # The annotated output is deliberately the first and default result view.
    st.markdown(
        '<div class="sv-result-heading"><h3>AI inspection result</h3><span class="sv-result-pill">ANNOTATED OUTPUT</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"{result['source_label']}  ·  {result['width']:,} × {result['height']:,} px  ·  "
        f"Confidence threshold {result['confidence']:.2f}  ·  Image size {result['image_size']}"
    )

    image_col, detail_col = st.columns([1.7, 1], gap="large")
    with image_col:
        view = st.radio(
            "Image view",
            ["AI annotated", "Original image"],
            horizontal=True,
            key=f"view_{result_key}",
            label_visibility="collapsed",
        )
        if view == "AI annotated":
            st.image(result["annotated"], width="stretch")
            st.caption("Bounding boxes, predicted classes, and confidence scores are drawn on the image by the YOLO model.")
        else:
            st.image(result["original"], width="stretch")
            st.caption("Original uploaded image before model annotation.")

        action_col1, action_col2 = st.columns(2)
        with action_col1:
            st.download_button(
                "Download annotated image",
                data=result["annotated_png"],
                file_name="safevision_annotated_image.png",
                mime="image/png",
                key=f"download_annotated_{result_key}",
                width="stretch",
                disabled=not bool(result["rows"]),
            )
        with action_col2:
            st.download_button(
                "Download detections CSV",
                data=csv_bytes(result["rows"], result["counts"]),
                file_name="safevision_detections.csv",
                mime="text/csv",
                key=f"download_csv_{result_key}",
                width="stretch",
            )

    with detail_col:
        st.markdown("**Detection overview**")
        total = sum(int(value) for value in result["counts"].values())
        people = int(result["counts"].get("Person", 0))
        missing = total_for(result["counts"], VIOLATION_CLASSES)
        k1, k2 = st.columns(2)
        k1.metric("Total objects", f"{total:,}")
        k2.metric("People", f"{people:,}")
        k3, k4 = st.columns(2)
        k3.metric("PPE detections", f"{total_for(result['counts'], PPE_CLASSES):,}")
        k4.metric("Missing-PPE regions", f"{missing:,}")

        st.markdown("**Detected objects**")
        if result["counts"]:
            for name, value in sorted(result["counts"].items(), key=lambda item: int(item[1]), reverse=True):
                st.markdown(
                    f'<div class="sv-detection-row"><span class="sv-detection-name">{class_label(name)}</span>'
                    f'<span class="sv-detection-count">{int(value):,}</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No objects detected at this confidence threshold.")
        st.markdown(" ")
        st.markdown(
            '<div class="sv-callout">The boxes are the primary visual evidence. Always review the highlighted regions and confidence values before making a safety decision.</div>',
            unsafe_allow_html=True,
        )

    with st.expander("Detailed detection records", expanded=True):
        render_detection_table(result["rows"], key=f"table_{result_key}")

    with st.expander("Image analysis summary", expanded=False):
        render_overview(result["counts"])


def render_video_result(result: dict) -> None:
    st.markdown(
        '<div class="sv-result-heading"><h3>AI video inspection</h3><span class="sv-result-pill">ANNOTATED VIDEO</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"{result['source_label']}  ·  {result['processed_frames']:,} sampled frames  ·  "
        f"Confidence {result['confidence']:.2f}  ·  Every {result['frame_stride']} frame(s)"
    )
    video_col, detail_col = st.columns([1.65, 1], gap="large")
    with video_col:
        st.video(result["video_bytes"])
        st.download_button(
            "Download annotated video",
            data=result["video_bytes"],
            file_name="safevision_annotated_video.mp4",
            mime="video/mp4",
            key="download_annotated_video",
            width="stretch",
        )
    with detail_col:
        st.markdown("**Frame-level detection summary**")
        total = sum(int(value) for value in result["counts"].values())
        st.metric("Total detections across sampled frames", f"{total:,}")
        for name, value in sorted(result["counts"].items(), key=lambda item: int(item[1]), reverse=True):
            st.markdown(
                f'<div class="sv-detection-row"><span class="sv-detection-name">{class_label(name)}</span>'
                f'<span class="sv-detection-count">{int(value):,}</span></div>',
                unsafe_allow_html=True,
            )
    with st.expander("Video analysis report", expanded=True):
        render_overview(result["counts"], is_video=True)
        rows = [
            {"Class": class_label(name), "Frame-level detections": int(value)}
            for name, value in sorted(result["counts"].items(), key=lambda item: int(item[1]), reverse=True)
        ]
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.download_button(
            "Download frame-level summary CSV",
            data=csv_bytes([], result["counts"]),
            file_name="safevision_video_summary.csv",
            mime="text/csv",
            key="download_video_summary",
        )


def run_image_workflow(uploaded_file, source_label: str, storage_key: str, model, confidence: float, image_size: int) -> None:
    raw = uploaded_file.getvalue()
    if len(raw) > MAX_IMAGE_MB * 1024 * 1024:
        st.error(f"Image exceeds the {MAX_IMAGE_MB} MB upload limit. Please resize it and try again.")
        return
    signature = fingerprint(raw)
    try:
        original = safe_image_from_bytes(raw)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        st.error("The selected file is not a supported or valid image.")
        with st.expander("Technical details"):
            st.code(f"{type(exc).__name__}: {exc}")
        return

    width, height = original.size
    render_section("01", "Image input", "Preview and validate before analysis")
    image_col, info_col = st.columns([1.4, 1], gap="large")
    with image_col:
        st.image(original, width="stretch")
    with info_col:
        with st.container(border=True):
            st.markdown("**Selected image**")
            st.write(f"**File:** {source_label}")
            st.write(f"**Resolution:** {width:,} × {height:,} px")
            st.write(f"**File size:** {len(raw) / (1024 * 1024):.2f} MB")
            st.write("**Color handling:** normalized to RGB for inference input")
            st.markdown('<div class="sv-muted" style="font-size:.78rem">Image remains in this local application session.</div>', unsafe_allow_html=True)
        run = st.button("Run AI inspection", type="primary", width="stretch", key=f"run_{storage_key}")
        if run:
            try:
                with st.spinner("Detecting people, PPE items, and missing-PPE regions..."):
                    annotated, counts, rows = run_image(
                        model, original, conf=float(confidence), imgsz=int(image_size)
                    )
                if annotated is None:
                    raise RuntimeError("The inference function returned no annotated image.")
                # The inference function must return an RGB annotated image.
                if getattr(annotated, "ndim", 0) != 3 or annotated.shape[2] != 3:
                    raise RuntimeError(f"Unexpected annotated-image shape: {getattr(annotated, 'shape', None)}")
                st.session_state[storage_key] = {
                    "signature": signature,
                    "confidence": float(confidence),
                    "image_size": int(image_size),
                    "source_label": source_label,
                    "original": original,
                    "annotated": annotated,
                    "annotated_png": image_png_bytes(annotated),
                    "counts": Counter(counts),
                    "rows": list(rows),
                    "width": width,
                    "height": height,
                }
                st.success("Inspection completed. The annotated output is shown below.")
            except Exception as exc:
                technical_error("Image inspection failed. The model output could not be prepared.", exc)

    result = st.session_state.get(storage_key)
    if result:
        parameters_match = (
            result["signature"] == signature
            and result["confidence"] == float(confidence)
            and result["image_size"] == int(image_size)
        )
        if parameters_match:
            render_section("02", "Detection results", "Bounding boxes and confidence labels")
            render_image_result(result, storage_key)
        elif result["signature"] != signature:
            st.info("A result from an earlier image is saved in this session. Run inspection to analyze the currently selected image.")
        else:
            st.info("The inspection settings changed. Run the analysis again to refresh the result.")


# -----------------------------------------------------------------------------
# Sidebar and model selection
# -----------------------------------------------------------------------------
weights = resolve_weights()

with st.sidebar:
    st.markdown("<div class='sv-eyebrow'>INSPECTION SETTINGS</div>", unsafe_allow_html=True)
    st.markdown("## SafeVision AI")
    st.caption("YOLO-based visible PPE inspection")
    st.divider()
    confidence = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.35,
        step=0.05,
        help="Lower values show more candidate detections but may increase false positives. Higher values are more selective.",
    )
    image_size = st.select_slider(
        "Inference image size",
        options=[320, 416, 512, 640, 768, 960],
        value=640,
        help="Larger values may improve small-object visibility but require more computation.",
    )
    st.divider()
    if weights:
        st.success("Trained checkpoint ready")
        st.caption(f"**Active model**\n`{weights.relative_to(ROOT).as_posix()}`")
        st.caption(f"Checkpoint: {weights.stat().st_size / (1024 * 1024):.1f} MB")
    else:
        st.error("Trained checkpoint not found")
        st.caption("Add the trained checkpoint under the `models` directory.")
    st.markdown(
        '<div class="sv-muted" style="font-size:.76rem">Inference runs locally. Uploaded media is processed by this application and is not sent to an external AI service.</div>',
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
model_badge = "MODEL READY" if weights else "MODEL NOT READY"
st.markdown(
    f"""
    <div class="sv-topbar">
      <div class="sv-brand">
        <div class="sv-brand-mark">SV</div>
        <div><div class="sv-brand-name">SafeVision AI</div><div class="sv-brand-sub">Workplace PPE inspection workspace</div></div>
      </div>
      <div class="sv-live"><span class="sv-live-dot"></span>{model_badge}</div>
    </div>
    <div class="sv-eyebrow">COMPUTER VISION · WORKPLACE SAFETY</div>
    <h1 class="sv-title">See the risk. <span>Review the evidence.</span></h1>
    <div class="sv-subtitle">Inspect worksite images and videos for people, personal protective equipment, and dataset-defined missing-PPE regions. Every result includes visual evidence for human review.</div>
    """,
    unsafe_allow_html=True,
)

if weights is None:
    st.warning("The interface is ready, but a trained checkpoint is missing. Add a trained .pt file and refresh the page.")
    st.code("models/\n  best.pt\n  candidates/\n    yolo26s_best.pt\n    yolo26n_best.pt", language="text")
    st.stop()

try:
    model = get_model(str(weights), weights.stat().st_mtime_ns)
except Exception as exc:
    technical_error("The trained checkpoint could not be loaded.", exc)
    st.stop()


# -----------------------------------------------------------------------------
# Workflow selection
# -----------------------------------------------------------------------------
render_section("00", "Choose an inspection mode")
mode = st.radio(
    "Inspection mode",
    ["Image", "Video", "Camera", "Model guide"],
    horizontal=True,
    label_visibility="collapsed",
    key="workflow_mode",
)


# -----------------------------------------------------------------------------
# Image upload workflow
# -----------------------------------------------------------------------------
if mode == "Image":
    st.caption("Upload a single image. The annotated image with object boxes and confidence labels will be displayed as the primary result.")
    uploaded_image = st.file_uploader(
        "Upload worksite image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="image_upload_only",
        help=f"JPG, PNG, BMP, and WebP. Maximum size: {MAX_IMAGE_MB} MB.",
    )
    if uploaded_image is not None:
        run_image_workflow(uploaded_image, uploaded_image.name, "image_result", model, confidence, image_size)
    else:
        st.markdown(
            '<div class="sv-card"><div class="sv-card-label">IMAGE INSPECTION</div>'
            '<div style="font-size:1.15rem;font-weight:800;margin:.4rem 0">Start with a site photo</div>'
            '<div class="sv-muted">After upload, SafeVision will detect visible people, PPE objects, and missing-PPE regions. The resulting annotated image will be shown directly on this page.</div></div>',
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# Video upload workflow
# -----------------------------------------------------------------------------
elif mode == "Video":
    st.caption("Upload a video clip to generate an annotated MP4 and a frame-level detection summary.")
    uploaded_video = st.file_uploader(
        "Upload worksite video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload_only",
        help=f"MP4, AVI, MOV, and MKV. Maximum size: {MAX_VIDEO_MB} MB.",
    )
    if uploaded_video is None:
        st.markdown(
            '<div class="sv-card"><div class="sv-card-label">VIDEO INSPECTION</div>'
            '<div style="font-size:1.15rem;font-weight:800;margin:.4rem 0">Review a recorded worksite clip</div>'
            '<div class="sv-muted">The application processes sampled frames, draws detections on the output video, and provides a downloadable annotated file.</div></div>',
            unsafe_allow_html=True,
        )
    else:
        video_data = uploaded_video.getvalue()
        if len(video_data) > MAX_VIDEO_MB * 1024 * 1024:
            st.error(f"Video exceeds the {MAX_VIDEO_MB} MB upload limit.")
            st.stop()
        video_sig = fingerprint(video_data)
        preview_col, settings_col = st.columns([1.45, 1], gap="large")
        with preview_col:
            st.markdown("**Video preview**")
            st.video(video_data)
        with settings_col:
            with st.container(border=True):
                st.markdown("**Clip details**")
                st.write(f"**File:** {uploaded_video.name}")
                st.write(f"**Size:** {len(video_data) / (1024 * 1024):.2f} MB")
            frame_stride = st.number_input(
                "Process every Nth frame",
                min_value=1,
                max_value=30,
                value=2,
                step=1,
                help="Larger values process fewer frames and reduce runtime.",
            )
            max_frames = st.number_input(
                "Maximum sampled frames",
                min_value=30,
                max_value=3000,
                value=600,
                step=30,
            )
            run_video = st.button("Run video inspection", type="primary", width="stretch", key="run_video_inspection")

        video_signature = f"{video_sig}:{confidence}:{image_size}:{frame_stride}:{max_frames}"
        if run_video:
            progress = st.progress(0.0, text="Preparing video...")
            try:
                with tempfile.TemporaryDirectory(prefix="safevision_") as temp_dir:
                    temp_root = Path(temp_dir)
                    input_path = temp_root / f"input_video{Path(uploaded_video.name).suffix.lower()}"
                    output_path = temp_root / "safevision_annotated.mp4"
                    input_path.write_bytes(video_data)

                    def update_progress(value: float) -> None:
                        progress.progress(
                            min(max(float(value), 0.0), 1.0),
                            text="Analyzing sampled frames and rendering detections...",
                        )

                    with st.spinner("Processing video frames. Runtime depends on clip length and hardware..."):
                        output_file, counts, processed_frames = run_video(
                            model=model,
                            input_path=input_path,
                            output_path=output_path,
                            conf=float(confidence),
                            imgsz=int(image_size),
                            frame_stride=int(frame_stride),
                            max_frames=int(max_frames),
                            progress_callback=update_progress,
                        )
                    output_bytes = Path(output_file).read_bytes()

                if not output_bytes:
                    raise RuntimeError("The video inference function returned an empty output file.")
                progress.progress(1.0, text="Video inspection completed.")
                st.session_state["video_result"] = {
                    "signature": video_signature,
                    "source_label": uploaded_video.name,
                    "confidence": float(confidence),
                    "image_size": int(image_size),
                    "frame_stride": int(frame_stride),
                    "processed_frames": int(processed_frames),
                    "counts": Counter(counts),
                    "video_bytes": output_bytes,
                }
                st.success(f"Inspection completed. {int(processed_frames):,} sampled frames were processed.")
            except Exception as exc:
                technical_error("Video inspection failed. The annotated output could not be created.", exc)

        video_result = st.session_state.get("video_result")
        if video_result:
            if video_result["signature"] == video_signature:
                render_section("02", "Video detection results", "Annotated output and frame-level summary")
                render_video_result(video_result)
            else:
                st.info("A previous video report is saved in this session. Run the inspection again to update it for the current clip and settings.")


# -----------------------------------------------------------------------------
# Camera workflow (separate from file upload)
# -----------------------------------------------------------------------------
elif mode == "Camera":
    st.caption("Capture a still image from your device camera, then run the same detection pipeline used for uploaded images.")
    camera_col, guidance_col = st.columns([1.4, 1], gap="large")
    with camera_col:
        camera_image = st.camera_input("Capture worksite image", key="safevision_camera")
    with guidance_col:
        st.markdown(
            '<div class="sv-card"><div class="sv-card-label">CAMERA WORKFLOW</div>'
            '<div style="font-size:1.06rem;font-weight:800;margin:.35rem 0 .65rem">Capture and inspect</div>'
            '<div class="sv-muted">1. Allow camera access in the browser.<br>2. Frame the worksite clearly.<br>3. Capture a photo.<br>4. Run the model and review the annotated result.</div>'
            '<div class="sv-warning" style="margin-top:.85rem">This mode captures a still image. It does not provide continuous live video monitoring.</div></div>',
            unsafe_allow_html=True,
        )
    if camera_image is not None:
        run_image_workflow(camera_image, "Camera capture", "camera_result", model, confidence, image_size)


# -----------------------------------------------------------------------------
# Model guide
# -----------------------------------------------------------------------------
else:
    st.caption("Reference guide for the model's output labels and responsible interpretation.")
    guide_left, guide_right = st.columns(2, gap="large")
    with guide_left:
        with st.container(border=True):
            st.markdown("**Person and PPE object classes**")
            for label in ["Person", "helmet", "vest", "gloves", "boots", "goggles", "forklift"]:
                st.markdown(f"<div class='sv-detection-row'><span class='sv-detection-name'>{class_label(label)}</span><span class='sv-muted'>Object class</span></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("**Missing-PPE region classes**")
            for label in sorted(VIOLATION_CLASSES):
                st.markdown(f"<div class='sv-detection-row'><span class='sv-detection-name'>{class_label(label)}</span><span class='sv-muted'>Region class</span></div>", unsafe_allow_html=True)
    with guide_right:
        with st.container(border=True):
            st.markdown("**How to read the annotated image**")
            st.markdown(
                "- **Bounding box:** predicted object or region location.\n"
                "- **Class label:** category predicted by the model.\n"
                "- **Confidence:** model score for that prediction; it is not a guarantee of correctness.\n"
                "- **Threshold:** filters low-confidence detections before display."
            )
        st.markdown(
            '<div class="sv-warning">SafeVision AI is a decision-support prototype. A safety professional must review the evidence. The current detector does not reliably associate every missing-PPE region with a specific worker.</div>',
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown(
    '<div class="sv-footer">SAFEVISION AI · YOLO26 · PYTORCH · STREAMLIT &nbsp;|&nbsp; Local PPE inspection workspace · Human review required</div>',
    unsafe_allow_html=True,
)
