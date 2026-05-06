import shutil
import sys
from pathlib import Path

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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python preprocess.py filename")

    else:
        input_file = sys.argv[1]
        result = to_jpg(input_file)
        print(f"Converted to {result}")

        after_deskew = deskew(result)
        print(f"Deskewed image saved as {after_deskew}")
