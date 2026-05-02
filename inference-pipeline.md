# SchedBuddy Inference Pipeline Guide

Complete guide to the multi-stage inference pipeline that extracts structured course schedule data from student document images.

## Overview

The inference pipeline is a four-stage process that transforms raw schedule document images into structured JSON data:

```
Raw Image
    ↓
[1. Image Preprocessing]
    ↓
Normalized Image
    ↓
[2. Table Detection (YOLO)]
    ↓
Table Bounding Boxes
    ↓
[3. Structure Detection (Table Transformer)]
    ↓
Row/Column Detections
    ↓
[4. Data Extraction & Postprocessing]
    ↓
Structured JSON (Courses)
```

## Stage 1: Image Preprocessing

**Location:** [`img_processing/preprocess_img.py`](img_processing/preprocess_img.py)

**Purpose:** Normalize student schedule document images to ensure reliable OCR and model inference.

### Three-Phase Pipeline

#### Phase 0: Document Normalization (Unconditional)
Applied to **every** image automatically:
- **Step 0-A: Document Framing** — Crops excess background to isolate the document boundaries
- **Step 0-B: Flatness Correction** — Rectifies perspective distortion and page curl using contour detection
- **Step 0-C: Portrait Orientation Enforcement** — Ensures the document is upright and in portrait orientation

#### Phase 1: Quality Gate
Evaluates the Phase-0 normalized image against five quality checks:
1. **Resolution** — Minimum pixel dimensions for OCR reliability
2. **Blur Detection** — Ensures sharpness for accurate text recognition
3. **Brightness** — Validates exposure level
4. **Border Completeness** — Verifies document edges are visible
5. **Skew Detection** — Measures minor rotation angles

**Result:** Passes ✓ or Rejects ✗ (Phase 2 is never entered for rejected images)

#### Phase 2: OCR Enhancement (Conditional)
Only applied to images passing Phase 1:
- **Lighting Normalization** — CLAHE (Contrast Limited Adaptive Histogram Equalization) on the luminance channel
- **Minor Deskewing** — Corrects small rotation angles detected in Phase 1

### Configuration

All thresholds and parameters are defined in `PreprocessingConfig`:
- Resolution thresholds
- Blur detection sensitivity
- Brightness bounds
- Skew angle tolerance
- CLAHE parameters

### Usage Example

```python
from img_processing.preprocess_img import preprocess_image, PreprocessingConfig

# Load and preprocess image
config = PreprocessingConfig()
processed_img = preprocess_image("path/to/raw_image.jpg", config)

# Access processing metadata
if processed_img is not None:
    # Image passed all quality gates
    processed_img.save("output/normalized_image.jpg")
else:
    # Image failed quality check
    print("Image rejected - review quality thresholds")
```

---

## Stage 2: Table Detection

**Location:** [`table_detection/`](table_detection/)

**Purpose:** Locate and identify tables within the normalized document image using YOLO object detection.

### Model

- **Architecture:** YOLOv11s (custom trained)
- **Task:** Detect table objects in images
- **Input:** Preprocessed image (640×640 resolution)
- **Output:** Bounding boxes with confidence scores for detected tables

### Key Components

#### `yolo_model.py`
Low-level interface to YOLO model:
```python
from ultralytics import YOLO
model = YOLO("runs/detect/train/weights/best.pt")
results = model.predict(image_path, conf=0.25)
```

#### `run_pipeline.py`
Full five-step pipeline runner:
1. Split dataset into train/validation folders
2. Generate `data.yaml` configuration
3. Train YOLO model
4. Run inference on validation set
5. Crop detected tables from results

#### `crop_predict.py`
Extracts detected table regions as separate images using bounding box coordinates.

### Usage Example

```python
from table_detection.yolo_model import YOLO

# Load trained model
model = YOLO("table_detection/runs/detect/train/weights/best.pt")

# Run inference
results = model.predict("preprocessed_image.jpg", conf=0.25)

# Extract bounding boxes
for result in results:
    for box in result.boxes:
        xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
        confidence = box.conf[0].item()
        print(f"Table detected: ({xmin}, {ymin}) - ({xmax}, {ymax}), conf={confidence}")
```

---

## Stage 3: Structure Detection

**Location:** [`structure_detection/`](structure_detection/)

**Purpose:** Detect internal table structure (rows and columns) and extract cell content via OCR.

### Model Pipeline

#### 3.1 Table Structure Recognition

**Model:** Microsoft Table Transformer (`microsoft/table-detection-structure-recognition`)
- **Architecture:** ResNet backbone + Transformer encoder-decoder
- **Task:** Detect rows and columns within tables
- **Input:** Table region from Stage 2
- **Output:** Bounding boxes for rows and columns with confidence scores

#### 3.2 OCR Extraction

**Tool:** Tesseract OCR
- **Configuration:** PSM 6 (block of text) — optimal for multi-line cells
- **Process:** For each cell intersection (row × column), extract text via OCR
- **Output:** Cell content strings

### Key Components

#### `detector.py` — `BorderlessTableDetector`
Main orchestrator for table detection and structure recognition:

```python
from structure_detection.detector import BorderlessTableDetector

detector = BorderlessTableDetector(
    image_path="table.jpg",
    output_path="output.png",
    detection_model="microsoft/table-detection-model",
    structure_model="microsoft/table-detection-structure-recognition"
)

# Load image and run detection
detector.load_image()
detections, plot = detector.process(
    model_type="structure",
    threshold=0.9,
    show_plot=False,
    save_plot=True
)

# Detections list contains Detection objects:
# - label: "row" or "column"
# - bbox: [xmin, ymin, xmax, ymax]
# - score: confidence (0-1)
```

#### `extraction.py` — Data Extraction
Orchestrates OCR-based table data extraction:

```python
from structure_detection.extraction import extract_table

table_data = extract_table(detector, detections)
# Returns TableData object with:
# - headers: column names (matched against HEADER_NAMES)
# - rows: list of cell records per row
# - cells: raw OCR results
```

#### `match_text.py` — Text Matching
Fuzzy matching functions for intelligent data validation:
- `match_header(text)` — Match extracted text to expected column headers
- `match_course(text, db)` — Match course codes/names against known database

#### `column_handlers.py` — Column-Specific Logic
Custom processing for each column type:
- **Code Column:** Course code validation
- **Subject Column:** Course name matching and fuzzy search
- **Units Column:** Unit count extraction
- **Days Column:** Day normalization (M, T, W, Th, F, S, Su)
- **Time Column:** Time parsing and conflict detection
- **Room Column:** Room code extraction
- **Faculty Column:** Faculty name processing

### Configuration

**`config.py`:**
- `DETECTION_MODEL` — HuggingFace model ID for table detection
- `STRUCTURE_MODEL` — HuggingFace model ID for structure recognition
- `TESSERACT_CONFIG` — OCR parameters (PSM, OEM settings)
- `COLORS` — Visualization colors for bounding boxes

**`course_db.py`:**
- Course database loading from JSON files in `databases/`
- Fuzzy matching against known course codes and names

### Usage Example

```python
from structure_detection.main import main

# Full extraction pipeline
main(input_path="sample_image.jpg")
# Outputs:
# - detections.json (raw row/column detections)
# - extracted_sample_image.json (structured course data)
```

---

## Stage 4: Data Extraction & Postprocessing

**Location:** [`img_processing/postprocess_img.py`](img_processing/postprocess_img.py) and [`structure_detection/extraction.py`](structure_detection/extraction.py)

**Purpose:** Post-process OCR results, validate data, and produce final structured JSON output.

### Extraction Workflow

```
Raw OCR Text (per cell)
    ↓
[Text Cleaning]
    ↓
[Column-Specific Processing]
    ↓
[Validation & Fuzzy Matching]
    ↓
[Structured JSON Output]
```

### Data Models

#### Detection
```python
@dataclass
class Detection:
    label_id: int           # Model's class ID
    label: str              # "row" or "column"
    score: float            # Confidence 0-1
    bbox: list[float]       # [xmin, ymin, xmax, ymax]
    bbox_xywh: list[float]  # [x, y, width, height]
```

#### CellRecord
```python
@dataclass
class CellRecord:
    row_idx: int            # Row number
    col_idx: int            # Column number
    raw_text: str           # Raw OCR output
    processed_value: Any    # Processed value (typed)
    column_name: str        # "code", "subject", etc.
    confidence: float       # Processing confidence
```

#### TableData
```python
@dataclass
class TableData:
    headers: list[str]      # Column names
    rows: list[list[CellRecord]]  # 2D grid of cells
    cells: dict             # Raw OCR results
```

### Header Detection

**Process:**
1. Find header row using row detections with highest y-position
2. For each column, extract text from header cell
3. Fuzzy match against expected headers: `["code", "subject", "units", "class", "days", "time", "room", "faculty"]`
4. Initialize column handler for matched header type

**Threshold:** 70% minimum fuzzy match score

### Data Validation

Each column has custom validation logic:

| Column | Validation | Handler |
|--------|-----------|---------|
| Code | Match against known course database | `CourseCodeHandler` |
| Subject | Fuzzy match against course names | `SubjectHandler` |
| Units | Numeric extraction and range check | `UnitsHandler` |
| Days | Normalize to standard format (M, T, W, etc.) | `DaysHandler` |
| Time | Parse start/end times, check format | `TimeHandler` |
| Room | Extract room code/number | `RoomHandler` |
| Faculty | Clean faculty names | `FacultyHandler` |

### Output Format

**JSON Structure:**
```json
{
  "image_file": "path/to/image.jpg",
  "ocr_configuration": "--oem 3 --psm 6",
  "headers": ["code", "subject", "units", "class", "days", "time", "room", "faculty"],
  "rows": [
    {
      "code": {"raw": "CS101", "value": "CS101", "confidence": 0.95},
      "subject": {"raw": "Intro to CS", "value": "Introduction to Computer Science", "confidence": 0.88},
      "units": {"raw": "3", "value": 3, "confidence": 0.99},
      "days": {"raw": "MWF", "value": ["M", "W", "F"], "confidence": 0.92},
      "time": {"raw": "10:00-11:30", "value": {"start": "10:00", "end": "11:30"}, "confidence": 0.91},
      ...
    }
  ]
}
```

---

## Complete End-to-End Example

```python
from pathlib import Path
from img_processing.preprocess_img import preprocess_image
from structure_detection.detector import BorderlessTableDetector
from structure_detection.extraction import extract_table

# Stage 1: Preprocessing
raw_image_path = "raw_schedule.jpg"
preprocessed = preprocess_image(raw_image_path)

if preprocessed is None:
    print("Image failed quality check")
    exit(1)

preprocessed.save("preprocessed_schedule.jpg")

# Stage 2 & 3: Detection and Structure Recognition
detector = BorderlessTableDetector(
    image_path="preprocessed_schedule.jpg",
    output_path="output.png"
)

detector.load_image()
detections, _ = detector.process(
    model_type="structure",
    threshold=0.9
)

# Stage 4: Data Extraction
table_data = extract_table(detector, detections)

# Output structured data
import json
output = {
    "image_file": str(raw_image_path),
    "headers": table_data.headers,
    "rows": [
        {header: cell.processed_value for header, cell in zip(table_data.headers, row)}
        for row in table_data.rows
    ]
}

with open("extracted_schedule.json", "w") as f:
    json.dump(output, f, indent=2)
```

---

## Configuration and Tuning

### Performance Optimization

#### Batch Processing
```python
from pathlib import Path
from structure_detection.main import main

# Process multiple images
image_dir = Path("schedules/")
for image_path in image_dir.glob("*.jpg"):
    print(f"Processing {image_path.name}...")
    main(input_path=image_path.name)
```

#### Model Thresholds
- **Table Detection (YOLO):** `conf=0.25` (default) — lower for more detections
- **Structure Recognition:** `threshold=0.9` — increase for stricter matching
- **Header Matching:** `min_score=70` — fuzzy match confidence threshold

### Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| High preprocessing rejection rate | Low image quality | Review Phase 1 thresholds in `PreprocessingConfig` |
| Missing tables | YOLO confidence too high | Lower `conf` parameter in table detection |
| Incorrect column headers | Fuzzy matching too strict | Lower `min_score` in header matching |
| Wrong OCR text | Image quality or PSM mismatch | Try different PSM values (6, 7, 8, 13) |
| Slow inference | Batch size too large | Process images one at a time or reduce batch size |

### Required Dependencies

```
pillow >= 9.0
opencv-python >= 4.5
torch
transformers
pytesseract
ultralytics (YOLO)
rapidfuzz (fuzzy matching)
```

---

## Data Flow Summary

```
Input: Raw student schedule document image
    ↓
Phase 1: Preprocess → Normalized image (or rejected if quality gates fail)
    ↓
Phase 2: Detect → Table bounding boxes
    ↓
Phase 3: Structure → Row/column bounding boxes
    ↓
Phase 4: Extract → Raw OCR text per cell
    ↓
Phase 5: Validate & Match → Typed, validated values
    ↓
Output: Structured JSON with courses, times, rooms, faculty
```

---

## Key Insights

1. **Multi-model Approach:** The pipeline combines three separate models (preprocessing heuristics, YOLO, Table Transformer) for robust detection
2. **Quality-First:** Early rejection in preprocessing prevents wasted computation on low-quality images
3. **Semantic Validation:** Column handlers understand domain logic (course codes, time formats) beyond raw OCR
4. **Fuzzy Matching:** Tolerates OCR errors through intelligent text matching against known databases
5. **Modular Design:** Each stage can be tested and tuned independently

