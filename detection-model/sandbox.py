"""Table data extraction and OCR workflow."""

from __future__ import annotations
import json
import logging
from dataclasses import asdict
import re

from models import Detection, CellRecord, TableData
from utils import bbox_intersection, ocr_crop
from match_text import match_header, match_course
from parse_time import parse_time

logger = logging.getLogger(__name__)


def _normalize_header_key(text: str) -> str:
    """Normalize a header label for robust alias matching."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def normalize_days_cell(text: str) -> str:
    """Normalize common OCR spacing/casing issues in days strings."""
    return re.sub(r"\s+", "", text.upper())


def normalize_time_cell(text: str) -> str:
    """Normalize common OCR variants into `HH:MM AM - HH:MM PM` format."""
    cleaned = " ".join(
        text.replace("—", "-")
        .replace("–", "-")
        .replace(" to ", " - ")
        .split()
    )
    pattern = (
        r"(?i)\b(\d{1,2})(?::(\d{2}))?\s*([AP]M)\b\s*-\s*"
        r"(\d{1,2})(?::(\d{2}))?\s*([AP]M)\b"
    )
    match = re.search(pattern, cleaned)
    if not match:
        return cleaned

    h1, m1, ap1, h2, m2, ap2 = match.groups()
    start = f"{int(h1):02d}:{int(m1 or '00'):02d} {ap1.upper()}"
    end = f"{int(h2):02d}:{int(m2 or '00'):02d} {ap2.upper()}"
    return f"{start} - {end}"


def format_time_cell(text: str) -> list[dict[str, str]]:
    """Return time value as a list of {start, end} objects."""
    normalized = normalize_time_cell(text)
    if not normalized:
        return []

    try:
        parsed = parse_time(normalized)
        return [
            {
                "start": parsed.start.strftime("%I:%M %p"),
                "end": parsed.end.strftime("%I:%M %p"),
            }
        ]
    except ValueError:
        # Keep schema stable even when OCR text is noisy.
        return [{"start": "", "end": normalized}]


def _find_key_with_aliases(row: dict[str, object], aliases: set[str]) -> str | None:
    """Return first row key matching any normalized alias."""
    for key in row.keys():
        if _normalize_header_key(key) in aliases:
            return key
    return None


def _hashable_value(value: object) -> str:
    """Serialize values (including dicts) for deterministic grouping keys."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)


def consolidate_schedule_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """
    Consolidate expanded row fragments into a single row with `Schedules` entries.

    The function is resilient to either named headers (Days/Time/Room/Faculty)
    or positional fallback headers (col5-col8).
    """
    if not rows:
        return rows

    first = rows[0]
    days_key = _find_key_with_aliases(first, {"day", "days", "col5"})
    time_key = _find_key_with_aliases(first, {"time", "col6"})
    room_key = _find_key_with_aliases(first, {"room", "col7"})
    faculty_key = _find_key_with_aliases(first, {"faculty", "col8"})

    schedule_keys = {k for k in (days_key, time_key, room_key, faculty_key) if k}
    if not schedule_keys:
        return rows

    base_keys = [k for k in first.keys() if k not in schedule_keys]
    grouped: dict[tuple[str, ...], dict[str, object]] = {}
    order: list[tuple[str, ...]] = []

    for row in rows:
        grouping_key = tuple(_hashable_value(row.get(k, "")) for k in base_keys)
        if grouping_key not in grouped:
            grouped[grouping_key] = {k: row.get(k, "") for k in base_keys}
            grouped[grouping_key]["Schedules"] = []
            order.append(grouping_key)

        schedule = {
            "Days": normalize_days_cell(str(row.get(days_key, ""))) if days_key else "",
            "Time": format_time_cell(str(row.get(time_key, ""))) if time_key else [],
            "Room": str(row.get(room_key, "")).strip() if room_key else "",
            "Faculty": str(row.get(faculty_key, "")).strip() if faculty_key else "",
        }
        if any(schedule.values()):
            grouped[grouping_key]["Schedules"].append(schedule)

    consolidated_rows: list[dict[str, object]] = []
    for key in order:
        item = grouped[key]
        seen = set()
        unique_schedules = []
        for schedule in item["Schedules"]:
            schedule_key = json.dumps(schedule, sort_keys=True, ensure_ascii=False)
            if schedule_key in seen:
                continue
            seen.add(schedule_key)
            unique_schedules.append(schedule)

        item["Schedules"] = unique_schedules
        consolidated_rows.append(item)

    return consolidated_rows

def parse_units_cell(text: str) -> dict[str, float]:
    """
    Parse OCR text for units into numeric Credit/Lec/Lab values.

    Expected format is similar to "3.0 2.0 1.0". Comma decimals from noisy OCR
    (for example "3,0 2,0 1,0") are normalized to periods.

    Returns:
        Dictionary with keys "Credit", "Lec", and "Lab" as floats.
        Missing or invalid values default to 0.0.

    TODO: Parse extracted text and use it as subheader names.
    FIXME: Current implementation uses hardcoded subcolumns. 
    """
    sub_columns = ("Credit", "Lec", "Lab")
    units = re.findall(r"\d+(?:\.\d+)?", text.replace(",", "."))
    default = dict.fromkeys(sub_columns, 0.0)
    default.update(zip(sub_columns, map(float, units)))
    return default

def expand_multiline_rows(row: dict[str, str]) -> list[dict[str, str]]:
    """
    Expand rows with with multiline values separated by newline into 
    per-schedule-entry dicts. If a column has fewer lines than
    the max, the last known value is carried forward.
    
    FIXME: Current implementation repeats the entire row data changing only 
    the values on multiline columns.

    Current output:
        {
            "Code": "code",
            "Subject": "subject",
            "Units\nCredit Lee Lab": {
                "Credit": 3.0,
                "Lec": 2.0,
                "Lab": 1.0
            },
            "Class": "class",
            "Days": "TTh",
            "Time": "04:00 PM - 07:00 PM",
            "Room": "CS-02-104",
            "Faculty": "faculty"
            },
            {
            "Code": "code",
            "Subject": "subject",
            "Units\nCredit Lee Lab": {
                "Credit": 3.0,
                "Lec": 2.0,
                "Lab": 1.0
            },
            "Class": "class",
            "Days": "T",
            "Time": "10:00 AM - 12:00 PM",
            "Room": "CS-02-104",
            "Faculty": "faculty"
        }
    Goal output:
        {
            "Code": "code",
            "Subject": "subject",
            "Units": {
                "Credit": 3.0,
                "Lec": 2.0,
                "Lab": 1.0
            },
            "Class": "BSCS-3A",
            "Schedules": [
                {
                    "Days": "day",
                    "Time": "01:00 PM - 04:00 PM",
                    "Room": "CS-02-201 CS-02-105",
                    "Faculty": "class"
                },
                {
                    "Days": "day",
                    "Time": "01:00 PM - 04:00 PM",
                    "Room": "CS-02-201 CS-02-105",
                    "Faculty": "class"
                }
            ]
        }
    """

    split_rows = {
        col: [line.strip() for line in val.split("\n") if line.strip()]
        for col, val in row.items()
    }

    max_lines = max(len(lines) for lines in split_rows.values())

    entries = []
    last_entry = {} # carry forward the last non-empty value per column

    for i in range(max_lines):
        entry = {}
        for col, lines in split_rows.items():
            if i < len(lines):
                entry[col] = lines[i]
                last_entry[col] = lines[i]
            else: 
                entry[col] = last_entry.get(col, "")
        
        entries.append(entry)

    return entries

def extract_table(detector, detections: list[Detection]) -> TableData:
    """Extract structured table data from structure-model detections via OCR.
    
    Args:
        detector: BorderlessTableDetector instance with loaded image
        detections: output of process() with model_type="structure"
    
    Returns:
        TableData with headers, rows, and individual cell records

    FIXME: "Units" is hardcoded. Improve column checking for passing units cell text
    """

    if detector.image is None:
        raise RuntimeError("Call process() before extract_table().")

    rows = sorted(
        [d for d in detections if "row" in d.label.lower()],
        key=lambda d: d.bbox[1]  # sort by ymin
    )

    columns = sorted(
        [d for d in detections if d.label.lower() == "table column"],
        key=lambda d: d.bbox[0]  # sort by xmin
    )

    header_dets = [d for d in detections if "header" in d.label.lower()]

    n_cols = len(columns)
    header_names = [f"col{i + 1}" for i in range(n_cols)]

    # Build cell grid
    cell_records: list[CellRecord] = []
    rows_as_dicts: list[dict] = []

    data_rows = rows[1:] if len(rows) > 1 else []

    for r_idx, row in enumerate(data_rows, 1):
        col_dict: dict[str, str] = {}
        units_dict: dict[str, float] = {}
        
        for c_idx, col in enumerate(columns, 1):
            box = bbox_intersection(row.bbox, col.bbox)
            text = ocr_crop(detector.image, box) if box else ""
            col_name = header_names[c_idx - 1]
    
            cell_records.append(CellRecord(row=r_idx, column=c_idx, bbox=box, text=text))
    
            # Separate units column handling
            if "col3" in col_name:
                units_dict = parse_units_cell(text)
            else:
                col_dict[col_name] = text
        
        expanded = expand_multiline_rows(col_dict)
        
        for entry in expanded:
            entry["col3"] = units_dict
        
        rows_as_dicts.extend(expanded)

    # After the entire extraction loop, before header renaming
    for row in rows_as_dicts:
        if header_names[0] == "col1": 
            matched_code, score, subject = match_course(row["col1"], min_score=50)
            logger.info(f"Col1 fuzzy match: {row['col1']} → {matched_code} (score: {score})")
            row["col1"] = matched_code
            row["col2"] = subject
            
    # Temporarily mode  header naming after the data extraction  as too many hardcoding is expected. 
    # TODO: Find a way to parse Unit/Credit/Lec/Lab for sub-columning
    extracted = []
    if header_dets:
        header_box = header_dets[0].bbox
        for col in columns:
            header_cell = bbox_intersection(header_box, col.bbox)
            extracted_text = ocr_crop(detector.image, header_cell).strip()
            extracted_text, score = match_header(extracted_text, 50)
            logger.info(f"Match: {extracted_text} : {score}")
            extracted.append(extracted_text)

    if any(extracted):
        clean = [t or f"col_{i + 1}" for i, t in enumerate(extracted)]
        rows_as_dicts = [
            {clean[i]: row[header_names[i]] for i in range(n_cols)}
            for row in rows_as_dicts
        ]
        header_names = clean

    rows_as_dicts = consolidate_schedule_rows(rows_as_dicts)
    if rows_as_dicts and "Schedules" in rows_as_dicts[0]:
        header_names = [h for h in header_names if h not in {"Days", "Time", "Room", "Faculty", "col5", "col6", "col7", "col8"}]
        if "Schedules" not in header_names:
            header_names.append("Schedules")

    logger.info(
        "Extracted %d ouput rows (from %d detected rows) × %d columns", 
        len(rows_as_dicts), 
        len(data_rows),
        n_cols
    )
    return TableData(
        headers=header_names,
        rows=rows_as_dicts,
        cells=[asdict(c) for c in cell_records]
    )
