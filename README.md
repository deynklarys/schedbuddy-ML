# SchedBuddy-ML

Automated machine learning pipeline for extracting structured course schedule data from student document images.

## Project Overview

**Purpose:** Automatically generate timetables for Bicol University students by extracting and parsing schedule data from scanned course documents.

**Output:** Structured JSON containing course codes, subjects, units, class sections, meeting times, rooms, and faculty information.

**Core Technologies:**
- **Table Detection:** YOLOv11s (custom trained on schedule documents)
- **Structure Recognition:** Hugging Face Table Transformer (`microsoft/table-detection-structure-recognition`)
- **OCR:** Tesseract with custom PSM configurations
- **Image Processing:** OpenCV, Pillow
- **Data Validation:** Fuzzy matching, course database matching, semantic validation

---

## Pipeline Architecture

Complete inference pipeline with four stages:

```
Raw Document Image
    ↓
[1. Image Preprocessing] — Normalization, quality gates, enhancement (WIP)
    ↓
[2. Table Detection] — YOLO locates table regions
    ↓
[3. Structure Detection] — Table Transformer finds rows/columns
    ↓
[4. Data Extraction] — OCR + validation → structured JSON
```

**For detailed documentation, see [inference-pipeline.md](inference-pipeline.md)**

---

## Directory Structure

```
schedbuddy-ML/
├── databases/                  # Course databases (JSON)
├── detector.py                 # BorderlessTableDetector class
├── extraction.py               # OCR and data extraction
├── column_handlers.py          # Column-specific processing logic
├── models.py                   # Data model classes
├── config.py                   # Configuration constants
├── course_db.py                # Course database interface
├── utils.py                    # Utility functions
├── inference.py                # Main inference script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Key Components

### Stage 1: Image Preprocessing (WIP)

**Three-phase pipeline:**
- **Phase 0:** Document normalization (framing, perspective correction, orientation)
- **Phase 1:** Quality gates (resolution, blur, brightness, borders, skew)
- **Phase 2:** OCR enhancement (CLAHE lighting, minor deskewing)

**Features:**
- Configurable quality thresholds in `PreprocessingConfig`
- Rejects low-quality images early to save computation
- Caches analysis metadata for Phase 2 optimization

### Stage 2: Table Detection and Cropping

**YOLOv11s model for table localization:**
- Custom trained on Bicol University schedule documents
- Detects table bounding boxes in normalized images
- Confidence threshold: 0.80 (tunable)
- Output: Cropped table regions for structure detection

**Cropping**
- Extract table regions from detections

### Stage 3: Structure Detection and Data Extraction

**Table Transformer + Tesseract OCR:**

**BorderlessTableDetector** (`detector.py`):
- Loads Table Transformer model from Hugging Face
- Detects rows and columns within tables
- Coordinates OCR extraction for each cell
- Handles borderless table layouts

**Data Extraction** (`extraction.py`):
- OCR configuration: PSM 6 (block of text) — customizable
- Header detection and fuzzy matching against expected columns
- Cell content extraction and validation

**Validation & Matching** ( `column_handlers.py`):
- Fuzzy matching for headers: code, subject, units, class, days, time, room, faculty
- Course database matching (5 department + 1 GEC databases in `databases/`)
- Column-specific handlers for semantic validation:
  - Course codes against known database
  - Time parsing and validation
  - Day normalization (M, T, W, Th, F, S, Su)
  - Room code extraction
  - Faculty name processing

### Stage 4: Data Extraction

**Output Format (JSON):**
```json
{
  "image file:": "C:\\Computer-Science\\schedbuddy-ML\\output\\5ef068b5-113\\table_5ef068b5-113.jpg",
  "ocr configuration:": "--oem 3 --psm 6",
  "headers": [ "code", "subject", "units", "class", "days", "time", "room", "faculty" ],
  "rows": [
    {
      "code": "NSTP 12",
      "subject": "CWTS/LTS/ROTC",
      "units": {
        "credit": 0.0,
        "lec": 3.0,
        "lab": 0.0
      },
      "class": "BUCS-LTS-AM2",
      "schedules": [
        {
          "days": [
            "saturday"
          ],
          "time": {
            "start": "13:00 PM",
            "end": "16:00 PM"
          },
          "room": "CS-04-203",
          "faculty": "SERRANO, K."
        }
      ]
    }
  ]
}
```

---

## Configuration

### Structure Detection ([config.py](config.py))

- Detection threshold: 0.9
- Structure model: `microsoft/table-detection-structure-recognition`
- OCR config: `--oem 3 --psm 6`
- Course databases: 6 JSON files in `databases/`

### Header Matching ([extraction.py](extraction.py))

- Minimum fuzzy match score: 70%
- Expected headers: code, subject, units, class, days, time, room, faculty

---

## Dependencies

- `pillow >= 9.0` — Image I/O
- `opencv-python >= 4.5` — Image processing
- `torch` — Deep learning (for Table Transformer)
- `transformers` — Hugging Face models
- `pytesseract` — OCR interface
- `ultralytics` — YOLO training and inference
- `rapidfuzz` — Fuzzy text matching
- `numpy` — Numerical operations

See [requirements.txt](requirements.txt) for complete list with versions.

---

## References

- [Table Transformer (Hugging Face)](https://huggingface.co/microsoft/table-transformer-detection)
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
- [YOLOv11 Docs](https://docs.ultralytics.com/)
- [Borderless Tables Detection](https://github.com/ShakilMahmudShuvo/Borderless-Tables-Detection)

---

