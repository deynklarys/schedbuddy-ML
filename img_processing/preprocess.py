import shutil
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


def to_jpg(input_path: str) -> str:
    """
    Convert any supported file type to .jpg
    """
    p = Path(input_path)

    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")
    if p.suffix.lower() not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported filetype: {p.suffix}")

    if p.suffix.lower() == ".jpg":
        dest = OUTPUT_DIR / p.name
        shutil.copy(p, dest)
        return str(dest)

    doc = fitz.open(input_path)
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(2, 2), colorspace=fitz.csRGB)
    out_file = OUTPUT_DIR / f"{p.stem}.jpg"  # assumes first page contains sched
    pix.save(str(out_file))
    doc.close()

    return str(out_file)


def laplacian_variance(image_path: str) -> float:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"CV2 could not read: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def is_sharp_enough(image_path: str, threshold: float = 25) -> bool:
    return laplacian_variance(image_path) >= threshold


def deskew(image_path: str, threshold: float = 0.001) -> str:
    """
    Load image, detect skew, correct only if needed.
    thredhold: mininum degrees to apply correction (avoids micro-rotations)
    """
    path = Path(image_path)
    img = np.array(Image.open(path).convert("RGB"))
    angle = get_angle(img)

    if abs(angle) >= threshold:
        img = rotate(img, angle, reshape=True, cval=255)
        img = np.clip(img, 0, 255).astype(np.uint8)

        out_path = path.with_stem(path.stem + "_deskewed")
        Image.fromarray(img).save(out_path)
        return str(out_path)

    return image_path  # no deskew applied, original file


def enhance_for_ocr(image_path: str, darkness_threshold: int = 130) -> str:
    path = Path(image_path)
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise FileNotFoundError(f"CV2 could not read: {image_path}")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if gray.mean() < darkness_threshold:
        result = cv2.adaptiveThreshold(
            gray.astype(np.uint8),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            41,  # blockSize: larger = adapts to broader lighting variation
            10,  # C: larger = more aggresive darkening cut
        )
        out_path = path.with_stem(path.stem + "_enhanced")
        cv2.imwrite(str(out_path), result)
        print(f"Dark image (mean={gray.mean():.1f}).")
        print(f"Otsu applied:{out_path}")
        return str(out_path)

    print(f"Image fine (mean={gray.mean():.1f}). No enchancement applied.")
    return image_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python preprocess.py filename")

    else:
        input_file = sys.argv[1]
        result = to_jpg(input_file)
        print(f"Converted to {result}")

        if not is_sharp_enough(result):
            print("Photo quality too poor. Please retake.")
            sys.exit(1)

        after_deskew = deskew(result)
        print(f"Deskewed image saved as {after_deskew}")

        after_enhance = enhance_for_ocr(after_deskew)
        print(f"Final image: {after_enhance}")
