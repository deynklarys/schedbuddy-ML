from pathlib import Path
import os
import json
import shutil
import cv2

import logging 
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

from ultralytics import YOLO

from structure_detection.config import TESSERACT_CONFIG
from structure_detection.detector import BorderlessTableDetector
from structure_detection.extraction import extract_table


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
# Stage 2: Table Detection and Cropping
# -----------------------------------------------------------------------------
# TODO: improve error handling, might be better to wrap in a function and use try/except
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


# Crop table based on label
image_path = TABLE_OUTPUT
image = cv2.imread(str(image_path))
if image is None:
    print(f"Skipped unreadable image: {image_path}")
    exit(1)

if not LABEL_PATH.exists():
    print(f"Skipped (no label file): {LABEL_PATH}")
    exit(1)

height, width = image.shape[:2]
with LABEL_PATH.open("r", encoding="utf-8") as file:
    lines = [line.strip() for line in file if line.strip()]

table_class_id = 0
padding = 2

if lines[0]:
    parts = lines[0].split()
    if len(parts) < 5:
        logger.warning(f"Invalid label format in {LABEL_PATH}: {lines[0]}")
        exit(1)

    cls_id = int(float(parts[0]))
    if cls_id != table_class_id:
        logger.warning(f"Invalid class ID in {LABEL_PATH}: {cls_id}")
        exit(1)

    x_center_n, y_center_n, box_width_n, box_height_n = map(float, parts[1:5])

    box_width = box_width_n * width
    box_height = box_height_n * height
    x_center = x_center_n * width
    y_center = y_center_n * height

    x1 = int(round(x_center - box_width / 2))
    y1 = int(round(y_center - box_height / 2))
    x2 = int(round(x_center + box_width / 2))
    y2 = int(round(y_center + box_height / 2))

    x1 -= padding
    y1 -= padding
    x2 += padding
    y2 += padding

    x1 = max(0, min(x1, image.shape[1]))
    x2 = max(0, min(x2, image.shape[1]))
    y1 = max(0, min(y1, image.shape[0]))
    y2 = max(0, min(y2, image.shape[0]))
    if x2 <= x1 or y2 <= y1:
        logger.warning(f"Invalid bounding box for {image_path}: ({x1}, {y1}, {x2}, {y2})")
        exit(1)

    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        logger.warning(f"Invalid cropped image for {image_path}: ({x1}, {y1}, {x2}, {y2})")
        exit(1)

    output_path = OUTPUT_DIR / f"{img_path_wo_ext}_table{img_path.suffix}"
    cv2.imwrite(str(output_path), cropped)
    logger.info(f"Saved: {output_path}")

# -----------------------------------------------------------------------------
# Stage 3: Table Structure Detection
# -----------------------------------------------------------------------------
detector = BorderlessTableDetector(
image_path=output_path,
output_path=STRUCT_OUTPUT
)

detector.load_image()
detections, _ = detector.process(
    model_type="structure", threshold=0.9, show_plot=False, save_plot=True
)

# -----------------------------------------------------------------------------
# Stage 4: Data Extraction
# -----------------------------------------------------------------------------
table_data = extract_table(detector, detections)

output = extract_table(detector, detections)
Path(EXTRACTED_JSON).write_text(
    json.dumps(
        {
            "image file:": str(image_path),
            "ocr configuration:": TESSERACT_CONFIG,
            "headers": table_data.headers,
            "rows": table_data.rows,
        },
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)
logger.info("Table JSON saved: %s", EXTRACTED_JSON)