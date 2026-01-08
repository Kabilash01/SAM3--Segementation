# ===============================
# Imports
# ===============================
import os
import uuid
import shutil
import subprocess
from pathlib import Path
from collections import defaultdict
from typing import Optional, Tuple, List

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import COLORS


# ===============================
# Paths & Environment
# ===============================
BASE_DIR = Path(__file__).resolve().parent
TEMP_IMAGE_PATH = BASE_DIR / "temp.jpg"
TEMP_VIDEO_ROOT = BASE_DIR / "temp_video"
HF_CACHE_DIR = BASE_DIR / "models"

os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("SAM3_DISABLE_BF16", "1")

TEMP_VIDEO_ROOT.mkdir(exist_ok=True)
HF_CACHE_DIR.mkdir(exist_ok=True)


# ===============================
# Globals (lazy)
# ===============================
IMAGE_MODEL = None
IMAGE_PROCESSOR: Optional[Sam3Processor] = None


# ===============================
# Helpers
# ===============================
def _np_image(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def _color_for_idx(idx: int) -> Tuple[int, int, int]:
    c = (COLORS[idx % len(COLORS)] * 255).astype(np.uint8)
    return int(c[0]), int(c[1]), int(c[2])


def _save_temp_image(img: Image.Image):
    img.save(TEMP_IMAGE_PATH)


def _load_temp_image() -> Optional[Image.Image]:
    return Image.open(TEMP_IMAGE_PATH).convert("RGB") if TEMP_IMAGE_PATH.exists() else None


def _mask_to_crop(img: Image.Image, mask: np.ndarray) -> Optional[Image.Image]:
    mask = mask.astype(bool)
    if not mask.any():
        return None
    y, x = np.where(mask)
    x0, x1 = x.min(), x.max() + 1
    y0, y1 = y.min(), y.max() + 1
    arr = _np_image(img)
    crop = arr[y0:y1, x0:x1]
    crop[~mask[y0:y1, x0:x1]] = 0
    return Image.fromarray(crop)


def _overlay_masks(img: Image.Image, masks, points=None):
    base = _np_image(img).copy()

    for i, m in enumerate(masks):
        m = m.squeeze().astype(bool)
        color = _color_for_idx(i)
        for c in range(3):
            base[..., c][m] = (
                0.6 * color[c] + 0.4 * base[..., c][m]
            ).astype(np.uint8)

    overlay = Image.fromarray(base)
    draw = ImageDraw.Draw(overlay)

    if points:
        for p in points:
            x, y = p["x"], p["y"]
            col = (0, 200, 0) if p["label"] == 1 else (200, 0, 0)
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=col, outline="white")

    return overlay


# ===============================
# Model loading
# ===============================
def ensure_models_loaded():
    global IMAGE_MODEL, IMAGE_PROCESSOR
    if IMAGE_PROCESSOR is None:
        IMAGE_MODEL = build_sam3_image_model(enable_inst_interactivity=True)
        IMAGE_PROCESSOR = Sam3Processor(IMAGE_MODEL, confidence_threshold=0.5)


def load_models():
    ensure_models_loaded()
    device = next(IMAGE_MODEL.parameters()).device
    return f"SAM3 loaded on {device}"


# ===============================
# Image inference
# ===============================
def run_image_inference(text_prompt, image_state):
    ensure_models_loaded()

    img = _load_temp_image()
    if img is None:
        raise gr.Error("Upload an image first")

    state = image_state["state"]
    IMAGE_PROCESSOR.reset_all_prompts(state)

    masks = []

    if text_prompt.strip():
        state = IMAGE_PROCESSOR.set_text_prompt(text_prompt.strip(), state)
        raw_masks = state.get("masks", None)
        if raw_masks is not None and len(raw_masks) > 0:
            masks.extend([m.squeeze().cpu().numpy() for m in raw_masks])

    if not masks:
        return img, [], "⚠️ No masks produced"

    overlay = _overlay_masks(img, masks)
    crops = [c for m in masks if (c := _mask_to_crop(img, m)) is not None]

    return overlay, crops, "Image segmentation complete"


# ===============================
# Video helpers
# ===============================
def extract_frames(video_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg not found in PATH")

    subprocess.run(
        [ffmpeg, "-i", str(video_path), "-q:v", "2", "-y", str(out_dir / "%06d.jpg")],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    frames = sorted(out_dir.glob("*.jpg"))
    if not frames:
        raise RuntimeError("No frames extracted")

    return frames


def stitch_video(frames_dir: Path, fps: float, out_path: Path):
    frames = sorted(frames_dir.glob("*.jpg"))
    sample = cv2.imread(str(frames[0]))
    h, w, _ = sample.shape

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    for f in frames:
        writer.write(cv2.imread(str(f)))

    writer.release()
    return out_path


# ===============================
# Video inference (FIXED)
# ===============================
def run_video(video_file, text_prompt):
    ensure_models_loaded()

    video_path = Path(video_file)
    job = TEMP_VIDEO_ROOT / uuid.uuid4().hex
    frames_dir = job / "frames"
    out_frames = job / "masked"

    frames = extract_frames(video_path, frames_dir)
    out_frames.mkdir(parents=True, exist_ok=True)

    for f in frames:
        img = Image.open(f)
        _save_temp_image(img)

        state = IMAGE_PROCESSOR.set_image(img, {})
        state = IMAGE_PROCESSOR.set_text_prompt(text_prompt, state)

        masks = state.get("masks", None)

        if masks is not None and len(masks) > 0:
            mask_np = masks[0].squeeze().cpu().numpy()
            overlay = _overlay_masks(img, [mask_np])
        else:
            overlay = img

        overlay.save(out_frames / f.name)

    out_video = stitch_video(out_frames, fps=10, out_path=job / "output.mp4")
    return str(out_video), "Video processing complete"


# ===============================
# Gradio UI
# ===============================
with gr.Blocks(title="SAM3 Image + Video App") as demo:
    status = gr.Textbox(label="Status", lines=4)

    with gr.Tabs():
        with gr.Tab("Image"):
            load_btn = gr.Button("Load models", variant="primary")
            prompt = gr.Textbox(label="Text prompt")
            run_btn = gr.Button("Run Image", variant="primary")

            img_in = gr.Image(type="pil")
            img_out = gr.Image(type="pil")
            crops = gr.Gallery(columns=3)

            image_state = gr.State()

            load_btn.click(load_models, outputs=status)

            img_in.upload(
                lambda img: (
                    {"state": IMAGE_PROCESSOR.set_image(img, {})},
                    img,
                    "Image loaded",
                ),
                img_in,
                [image_state, img_in, status],
            )

            run_btn.click(
                run_image_inference,
                [prompt, image_state],
                [img_out, crops, status],
            )

        with gr.Tab("Video"):
            video_in = gr.Video(label="Upload MP4")
            video_prompt = gr.Textbox(label="Text prompt", value="person")
            run_video_btn = gr.Button("Run Video", variant="primary")
            video_out = gr.Video()

            run_video_btn.click(
                run_video,
                [video_in, video_prompt],
                [video_out, status],
            )

if __name__ == "__main__":
    demo.launch()
