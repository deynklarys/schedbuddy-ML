import sys
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np
from jdeskew.estimator import get_angle
from PIL import Image
from scipy.ndimage import rotate

SUPPORTED_TYPES = {".jpg", ".pdf", ".jpeg", ".jfif", ".png"}
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def _normalize_input(input_path: Path) -> np.ndarray:
    if input_path.suffix.lower() == ".pdf":
        doc = fitz.open(str(input_path))
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(2, 2), colorspace=fitz.csRGB)
        doc.close()
        return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    return np.array(Image.open(input_path).convert("RGB"))


def _is_sharp_enough(img_bgr: np.ndarray, threshold: float = 25) -> bool:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() >= threshold


def _deskew(img_rgb: np.ndarray, threshold: float = 0.001) -> np.ndarray:
    angle = get_angle(img_rgb)

    if abs(angle) >= threshold:
        untilted = rotate(img_rgb, angle, reshape=True, cval=255)
        return np.clip(untilted, 0, 255).astype(np.uint8)

    return img_rgb  # no deskew applied, original file as np.ndarray


def _enhance_for_ocr(img_rgb: np.ndarray, darkness_threshold: int = 130) -> np.ndarray:
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if gray.mean() < darkness_threshold:
        print(f"Dark image (mean={gray.mean():.1f}).")
        print("Applying adaptive threshold.")
        result = cv2.adaptiveThreshold(
            gray.astype(np.uint8),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            41,  # blockSize: larger = adapts to broader lighting variation
            10,  # C: larger = more aggresive darkening cut
        )
        return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)  # new file as np.ndarray

    print(f"Image fine (mean={gray.mean():.1f}). No enchancement applied.")
    return img_rgb


def preprocess(input_path: str) -> str:
    """
    Full preprocessing pipeline: convert -> validate -> deskew -> enhance.
    Returns the path to the final, saved image.
    """
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    if p.suffix.lower() not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported filetype: {p.suffix}")

    img = _normalize_input(p)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if not _is_sharp_enough(img_bgr):
        raise ValueError("Photo quality too poor. Please retake.")
        # implement retake functionality here.

    steps_applied = []

    deskewed = _deskew(img)
    if deskewed is not img:
        steps_applied.append("deskewed")
        img = deskewed

    enhanced = _enhance_for_ocr(img)
    if enhanced is not img:
        steps_applied.append("enhanced")
        img = enhanced

    suffix = "_" + "_".join(steps_applied) if steps_applied else "_clean"
    out_dir = OUTPUT_DIR / p.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{p.stem}.jpg"

    Image.fromarray(img).save(str(out_path))
    print(f"Preprocessed image saved: {out_path}")

    return str(out_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python preprocess.py filename")

    else:
        print(preprocess(sys.argv[1]))
