from pathlib import Path
import os
import shutil
import cv2

import logging 
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

from ultralytics import YOLO


# Modify as needed
sample_image = "1f4a47d1-242.png"

# Paths 
base_dir = Path(__file__).resolve().parent

img_path = base_dir / "sample" / sample_image
img_path_wo_ext = os.path.splitext(sample_image)[0]

OUTPUT_DIR  = base_dir / "output" / f"{img_path_wo_ext}"

# Remove old output if it exists
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TABLE_OUTPUT = OUTPUT_DIR / f"table_{img_path_wo_ext}.jpg"
LABEL_PATH = OUTPUT_DIR / "labels" / f"{img_path_wo_ext}.txt"
CROPPED_OUTPUT = OUTPUT_DIR / f"cropped_{img_path_wo_ext}.jpg"
STRUCT_OUTPUT = OUTPUT_DIR / f"struct_{img_path_wo_ext}.jpg"
DETECTIONS_JSON = OUTPUT_DIR / f"detections_{img_path_wo_ext}.json"
EXTRACTED_JSON = OUTPUT_DIR / f"extracted_{img_path_wo_ext}.json"

# -------------------------------------------------------------------
# Stage 1: Preprocessing (WIP)
# -------------------------------------------------------------------

# PDF to image conversion (if needed)

# Enhance image quality

# -----------------------------------------------------------------------------
# Stage 2: Table Detection
# -----------------------------------------------------------------------------

model = YOLO("table_detection/runs/detect/train/weights/best.pt")

# Run inference
results = model.predict(
    source=img_path,
    conf=0.8, 
    save_txt=True, 
    project=str(OUTPUT_DIR),
    name=".",
    exist_ok=True)

results[0].save(TABLE_OUTPUT)
