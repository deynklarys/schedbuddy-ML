import shutil
import sys
from pathlib import Path

import fitz  # PyMuPDF

SUPPORTED_TYPES = {".jpg", ".pdf", ".jpeg", ".jfif", ".png"}
TARGET_FORMAT = ".jpg"


def to_jpg(input_path: str, output_dir: str = "output") -> str:
    """
    Convert any supported file type to .jpg
    """
    p = Path(input_path)
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")
    if p.suffix.lower() not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported filetype: {p.suffix}")

    if p.suffix.lower() == ".jpg":
        dest = out / p.name
        shutil.copy(p, dest)
        return str(dest)

    doc = fitz.open(input_path)
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(2, 2), colorspace=fitz.csRGB)
    out_file = out / f"{p.stem}.jpg"  # assumes first page contains sched
    pix.save(str(out_file))
    doc.close()

    return str(out_file)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        result = to_jpg(input_file)
        print(f"Converted to {result}")
    else:
        print("Usage: python preprocess.py filename")
