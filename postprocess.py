import logging
logger = logging.getLogger(__name__)

from typing import Any

from pydantic import BaseModel
from models import TimeRange, CourseRow, TableData

# ---------------------------------------------------------------------------
# Free-slot search configuration
# ---------------------------------------------------------------------------

# Search window: 7:00 AM – 9:00 PM (in minutes from midnight)
_SEARCH_START: int = 7 * 60
_SEARCH_END:   int = 21 * 60

# When no duration can be inferred, assume a one-hour class
_DEFAULT_DURATION: int = 60

# Granularity of the slot search (minutes)
_STEP: int = 30

# Day groups tried in order when assigning a fallback day pattern.
# MWF / TTh come first as they are the most common university patterns.
_FALLBACK_DAY_GROUPS: list[list[str]] = [
    ["monday"],
    ["tuesday"],
    ["wednesday"],
    ["thursday"],
    ["friday"],
    ["saturday"],
    ["sunday"],
]

# ---------------------------------------------------------------------------
# Free-slot helpers
# ---------------------------------------------------------------------------

def _is_valid_days(val: Any) -> bool:
    """Return True only when *val* is a non-empty list of non-empty strings."""
    return (
        isinstance(val, list)
        and len(val) > 0
        and all(isinstance(d, str) and d for d in val)
    )


def _is_valid_time(val: Any) -> bool:
    """Return True only when *val* is a parsed TimeRange object."""
    return isinstance(val, TimeRange)


def _collect_occupied_slots(
    rows: list[dict[str, Any]],
) -> list[tuple[list[str], int, int]]:
    """
    Walk every schedule slot across all rows and collect the ones that
    already have both valid days and a valid TimeRange.

    Returns a list of ``(days, start_mins, end_mins)`` tuples.
    """
    occupied: list[tuple[list[str], int, int]] = []
    for row in rows:
        for slot in row.get("schedules", []):
            days = slot.get("days")
            time_val = slot.get("time")
            if _is_valid_days(days) and _is_valid_time(time_val):
                occupied.append((list(days), time_val.start, time_val.end))
    return occupied

def _slots_overlap(
    days_a: list[str], start_a: int, end_a: int,
    days_b: list[str], start_b: int, end_b: int,
) -> bool:
    """Return True when two schedule slots share at least one day *and* their
    time windows overlap (touching edges do not count as overlap)."""
    if not any(d in days_b for d in days_a):
        return False
    return not (end_a <= start_b or start_a >= end_b)

def _find_free_slot(
    occupied: list[tuple[list[str], int, int]],
    duration_mins: int = _DEFAULT_DURATION,
    preferred_days: list[str] | None = None,
) -> tuple[list[str], int, int]:
    """
    Search for the earliest ``duration_mins``-long window that does not
    overlap with any entry in *occupied*.

    Tries *preferred_days* first (if given), then each group in
    ``_FALLBACK_DAY_GROUPS``.  Returns ``(days, start_mins, end_mins)``.
    """
    day_groups_to_try: list[list[str]] = (
        [preferred_days] if preferred_days else []
    ) + _FALLBACK_DAY_GROUPS

    for days in day_groups_to_try:
        for start in range(_SEARCH_START, _SEARCH_END - duration_mins + 1, _STEP):
            end = start + duration_mins
            if not any(
                _slots_overlap(days, start, end, occ_days, occ_start, occ_end)
                for occ_days, occ_start, occ_end in occupied
            ):
                return days, start, end

    # Absolute fallback — place after the search window so it never
    # collides with anything inside the window.
    logger.warning(
        "No free slot found within %02d:00–%02d:00; placing after search window.",
        _SEARCH_START // 60, _SEARCH_END // 60,
    )
    return ["monday"], _SEARCH_END, _SEARCH_END + duration_mins


def fill_missing_slots(rows: list[dict[str, Any]]) -> None:
    """
    Post-process every schedule slot in *rows* **in-place**.

    For each slot where OCR produced noisy or unparseable days / time:

    * Re-collect all currently valid occupied slots on every call so that
      newly assigned slots are already reserved for subsequent iterations.
    * Assign the earliest non-overlapping free slot that fits.

    Three cases are handled independently:

    +---------------+---------------+-----------------------------------------+
    | days valid?   | time valid?   | Action                                  |
    +===============+===============+=========================================+
    | ✓             | ✓             | Nothing to do.                          |
    +---------------+---------------+-----------------------------------------+
    | ✗             | ✓             | Keep time; find days with no conflict.  |
    +---------------+---------------+-----------------------------------------+
    | ✓             | ✗             | Keep days (preferred); find time.       |
    +---------------+---------------+-----------------------------------------+
    | ✗             | ✗             | Find both days and time from scratch.   |
    +---------------+---------------+-----------------------------------------+
    """
    for row in rows:
        for slot in row.get("schedules", []):
            days     = slot.get("days")
            time_val = slot.get("time")

            valid_days = _is_valid_days(days)
            valid_time = _is_valid_time(time_val)

            if valid_days and valid_time:
                continue  # Already fully parsed — nothing to do.

            # Re-collect each iteration so already-fixed slots are respected.
            occupied = _collect_occupied_slots(rows)

            if valid_time and not valid_days:
                # We know *when* but not *which days* — find a day pattern that
                # does not conflict with the known time window.
                duration = time_val.end - time_val.start
                new_days, new_start, new_end = _find_free_slot(occupied, duration)
                slot["days"] = new_days
                logger.info(
                    "Assigned fallback days=%s for noisy OCR "
                    "(time window %d–%d preserved).",
                    new_days, time_val.start, time_val.end,
                )

            elif valid_days and not valid_time:
                # We know *which days* but not *when* — try to book a slot on
                # those days, falling back to other day groups if all taken.
                new_days, new_start, new_end = _find_free_slot(
                    occupied, _DEFAULT_DURATION, preferred_days=list(days)
                )
                slot["days"] = new_days
                slot["time"] = TimeRange(start=new_start, end=new_end)
                if new_days != list(days):
                    logger.info(
                        "Original days=%s fully booked; reassigned to days=%s.",
                        days, new_days,
                    )
                logger.info(
                    "Assigned fallback time %d–%d on days=%s for noisy OCR.",
                    new_start, new_end, new_days,
                )

            else:
                # Both days and time are missing — pick everything from scratch.
                new_days, new_start, new_end = _find_free_slot(
                    occupied, _DEFAULT_DURATION
                )
                slot["days"] = new_days
                slot["time"] = TimeRange(start=new_start, end=new_end)
                logger.info(
                    "Assigned fallback slot days=%s, time=%d–%d for noisy OCR.",
                    new_days, new_start, new_end,
                )

def serialize_row(row: list[dict[str, str]] | dict[str, Any] | BaseModel) -> Any:
    """Convert Pydantic models in row to dicts recursively."""
    if isinstance(row, dict):
        return {k: serialize_row(v) for k, v in row.items()}
    elif isinstance(row, list):
        return [serialize_row(item) for item in row]
    elif isinstance(row, BaseModel):
        return row.model_dump()
    else:
        return row

def validate_course_rows(table: TableData) -> TableData:
    """
    Validate parsed course rows against the CourseRow model, logging any
    validation errors without raising exceptions.

    Returns the list of rows that passed validation.
    """
    valid_rows: list[dict] = []
    for idx, row in enumerate(table.rows, 1):
        try:
            CourseRow.model_validate(row)
            valid_rows.append(row)
        except Exception as e:
            logger.warning(f"Row {idx} failed validation: {e}")
            logger.debug(f"Invalid row data: {row}")
            
    return TableData(headers=table.headers, rows=valid_rows, cells=table.cells)