"""Entry point for table detection and extraction pipeline."""

from __future__ import annotations
import json
import logging
import argparse
import os
from pathlib import Path
from dataclasses import asdict

from .detector import BorderlessTableDetector
from .extraction import extract_table
from .config import TESSERACT_CONFIG
from .logger import log_time

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

@log_time
def main(input_path: str = "") -> None:
    base_dir = Path(__file__).resolve().parent
    image_path = base_dir / "test-images" / input_path
    OUTPUT_IMAGE = base_dir / "output.png"
    DETECTIONS_JSON = base_dir / "detections.json"
    input_path_wo_ext = os.path.splitext(input_path)[0]
    TABLE_JSON = base_dir / f"extracted-data/extracted_{input_path_wo_ext}.json"

    detector = BorderlessTableDetector(image_path, OUTPUT_IMAGE)

    # Run structure recognition
    detections, _ = detector.process(
        model_type="structure", threshold=0.9, show_plot=False, save_plot=True
    )

    # Save raw detections
    rows_det = [asdict(d) for d in detections if "row" in d.label.lower()]
    cols_det = [asdict(d) for d in detections if "column" in d.label.lower()]
    Path(DETECTIONS_JSON).write_text(
        json.dumps({"rows": rows_det, "columns": cols_det}, indent=2), encoding="utf-8"
    )
    logger.info("Detections saved as: %s (rows = %d, columns = %d)", DETECTIONS_JSON, len(rows_det), len(cols_det))

    # Extract table data via OCR
    table_data = extract_table(detector, detections)
    Path(TABLE_JSON).write_text(
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
    logger.info("Table JSON saved: %s", TABLE_JSON)

    # Preview first two rows
    print("\nFirst two rows:")
    for row in table_data.rows[:2]:
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run detection model")
    parser.add_argument("--image", default="", help="Path to test images (test-images/)")
    args = parser.parse_args()
    main(input_path=args.image)
