"""Table data extraction and OCR workflow."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from rapidfuzz import fuzz

from .column_handlers import ColumnHandler, get_handler
from .course_db import CourseDatabase
from .models import CellRecord, Detection, TableData
from .utils import bbox_intersection, ocr_crop

logger = logging.getLogger(__name__)

HEADER_NAMES = ["code", "subject", "units", "class", "days", "time", "room", "faculty"]

# Default course database
default_course_db: CourseDatabase = CourseDatabase.from_dir("structure_detection/databases")

# Fuzzy matching
def _best_fuzzy_match(
    query: str,
    candidates: list[str],
    min_score: int = 0,
) -> tuple[str, int]:
    best_name, best_score = query, -1
    for candidate in candidates:
        score = max(
            fuzz.partial_ratio(query, candidate),
            fuzz.ratio(query, candidate),
        )
        if score > min_score and score > best_score:
            best_name, best_score = candidate, score
    return best_name, best_score

def match_header(extracted_text: str, min_score: int = 0) -> tuple[str, int]:
    return _best_fuzzy_match(extracted_text, HEADER_NAMES, min_score)

def match_course(
    extracted_text: str,
    min_score: int = 0,
    db: CourseDatabase | None = None,
) -> tuple[str, int, str]:
    return (db or default_course_db).match(extracted_text, min_score)

# Header OCR  (runs first, before any data extraction)
def _resolve_column_handlers(
    detector,
    columns: list,
    header_dets: list,
) -> tuple[list[str], list[ColumnHandler]]:

    n_cols = len(columns)

    if not header_dets:
        names    = [f"col_{i + 1}" for i in range(n_cols)]
        handlers = [get_handler(name) for name in names]
        return names, handlers

    header_box = header_dets[0].bbox
    names:    list[str]           = []
    handlers: list[ColumnHandler] = []

    for col in columns:
        cell = bbox_intersection(header_box, col.bbox)
        raw  = ocr_crop(detector.image, cell).strip() if cell else ""

        canonical, score = match_header(raw, min_score=70)
        logger.info("Header match: %r → %r (score: %d)", raw, canonical, score)

        handler = get_handler(canonical)
        handler.configure(raw)

        names.append(canonical)
        handlers.append(handler)

    return names, handlers


def _expand_multiline_rows(
    row: dict[str, Any],
    schedule_fields: set[str],
) -> list[dict[str, Any]]:

    schedule_lines: dict[str, list[str]] = {}
    shared: dict[str, Any] = {}

    for col, val in row.items():
        if col in schedule_fields and isinstance(val, str):
            schedule_lines[col] = [ln.strip() for ln in val.split("\n") if ln.strip()]
        else:
            shared[col] = val

    if not schedule_lines:
        return [{**shared, "schedules": []}]

    max_slots = max(len(lines) for lines in schedule_lines.values())
    last: dict[str, str] = {}
    schedules: list[dict[str, Any]] = []

    for i in range(max_slots):
        slot: dict[str, Any] = {}
        for col, lines in schedule_lines.items():
            value     = lines[i] if i < len(lines) else last.get(col, "")
            last[col] = value
            slot[col] = value
        schedules.append(slot)

    return [{**shared, "schedules": schedules}]


def extract_table(
    detector,
    detections: list[Detection],
    db: CourseDatabase | None = None,
) -> TableData:

    if detector.image is None:
        raise RuntimeError("Call process() before extract_table().")

    rows = sorted(
        [d for d in detections if "row" in d.label.lower()],
        key=lambda d: d.bbox[1],
    )
    columns = sorted(
        [d for d in detections if d.label.lower() == "table column"],
        key=lambda d: d.bbox[0],
    )
    header_dets = [d for d in detections if "header" in d.label.lower()]

    # 1. Header OCR first so handlers are configured before data rows
    header_names, handlers = _resolve_column_handlers(detector, columns, header_dets)
    schedule_fields = {
        name for name, h in zip(header_names, handlers) if h.is_schedule_field
    }

    # 2. Cell extraction
    cell_records:  list[CellRecord] = []
    rows_as_dicts: list[dict]       = []
    data_rows = rows[1:] if len(rows) > 1 else []

    for r_idx, row in enumerate(data_rows, 1):
        raw_cells:    dict[str, str] = {}
        parsed_cells: dict[str, Any] = {}

        for c_idx, (col, name, handler) in enumerate(
            zip(columns, header_names, handlers), 1
        ):
            box  = bbox_intersection(row.bbox, col.bbox)
            text = ocr_crop(detector.image, box) if box else ""

            cell_records.append(CellRecord(row=r_idx, column=c_idx, bbox=box, text=text))

            if handler.is_schedule_field:
                raw_cells[name] = text
            else:
                try:
                    parsed_cells[name] = handler.parse_cell(text)
                except (ValueError, KeyError):
                    logger.warning(
                        "Handler %s failed on %r (row %d, col %s); raw text = %r. Sending empty string.",
                        type(handler).__name__, text, r_idx, name, text,
                    )
                    parsed_cells[name] = ""

        expanded = _expand_multiline_rows(
            {**parsed_cells, **raw_cells},
            schedule_fields,
        )

        for entry in expanded:
            _parse_schedule_slots(entry, header_names, handlers)

        rows_as_dicts.extend(expanded)

    # 3. Post-process: fuzzy-match course codes
    _apply_course_matching(rows_as_dicts, header_names, db=db)

    logger.info(
        "Extracted %d output rows (from %d detected rows) × %d columns",
        len(rows_as_dicts), len(data_rows), len(columns),
    )
    return TableData(
        headers=header_names,
        rows=rows_as_dicts,
        cells=[asdict(c) for c in cell_records],
    )

def _parse_schedule_slots(
    entry: dict[str, Any],
    header_names: list[str],
    handlers: list[ColumnHandler],
) -> None:
    handler_map = dict(zip(header_names, handlers))
    for slot in entry.get("schedules", []):
        for field, raw in list(slot.items()):
            if not isinstance(raw, str) or not raw.strip():
                continue
            handler = handler_map.get(field)
            if handler is None:
                continue
            try:
                slot[field] = handler.parse_cell(raw)
            except (ValueError, KeyError):
                logger.warning(
                    "Handler %s failed on slot field %r=%r; sending empty string.",
                    type(handler).__name__, field, raw,
                )
                slot[field] = ""


def _apply_course_matching(
    rows: list[dict[str, Any]],
    header_names: list[str],
    db: CourseDatabase | None = None,
) -> None:
    if not header_names:
        return
    code_col = header_names[0]
    subj_col = header_names[1] if len(header_names) > 1 else None

    for row in rows:
        raw = row.get(code_col, "")
        if not isinstance(raw, str) or not raw.strip():
            continue
        matched_code, score, subject = match_course(raw, min_score=70, db=db)
        logger.info("Course match: %r → %r (score: %d)", raw, matched_code, score)
        if score < 70:
            row[code_col] = ""
            row[subj_col] = ""
        else:            
            row[code_col] = matched_code
            if subj_col:
                row[subj_col] = subject