"""Column handler registry for table extraction.

Each column type owns its own cell-parsing logic and schedule-field flag.
The units handler additionally parses its sub-column names dynamically from
the raw header OCR text, so they no longer need to be hardcoded.

Adding support for a new column type:
    1. Subclass ``ColumnHandler``.
    2. Override ``parse_cell`` (and optionally ``configure``).
    3. Add an instance to ``COLUMN_REGISTRY`` under the canonical header name.
"""

from __future__ import annotations

import logging
import re
from rapidfuzz import fuzz
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)

class ColumnHandler(ABC):

    is_schedule_field: bool = False

    def configure(self, raw_header: str) -> None:  # noqa: B027
        """Called once with the raw OCR text of this column's header cell.

        Override to do dynamic configuration (e.g. reading sub-column names
        from the header text).  The default implementation is a no-op.
        """

    @abstractmethod
    def parse_cell(self, text: str) -> Any:
        """Convert raw OCR *text* for this column into a structured value."""

# Concrete handlers
class DefaultHandler(ColumnHandler):
    def parse_cell(self, text: str) -> str:
        return text.strip()

class UnitsHandler(ColumnHandler):

    _FALLBACK_SUB_COLS: tuple[str, ...] = ("credit", "lec", "lab")
    _HEADER_STOP_WORDS: frozenset[str] = frozenset({"units", "unit"})
    _KNOWN_SUB_COLS: tuple[str, ...] = ("credit", "lec", "lab")

    def __init__(self) -> None:
        self._sub_cols: tuple[str, ...] = self._FALLBACK_SUB_COLS

    def configure(self, raw_header: str) -> None:
        tokens = [
            t.lower()
            for t in re.split(r"[\s/\n]+", raw_header.strip())
            if t and t.lower() not in self._HEADER_STOP_WORDS
        ]
        if len(tokens) < 2:
            logger.debug(
                "UnitsHandler: could not parse sub-columns from %r, "
                "using fallback %s", raw_header, self._FALLBACK_SUB_COLS,
            )
            return

        matched = []
        for token in tokens:
            best, score = max(
                (
                    (col, max(fuzz.ratio(token, col), fuzz.partial_ratio(token, col)))
                    for col in self._KNOWN_SUB_COLS
                ),
                key=lambda x: x[1],
            )
            if token != best:
                logger.debug(
                    "UnitsHandler: corrected sub-column %r → %r (score %d)",
                    token, best, score,
                )
            matched.append(best)

        self._sub_cols = tuple(matched)
        logger.debug("UnitsHandler sub-columns configured: %s", self._sub_cols)

    @property
    def sub_columns(self) -> tuple[str, ...]:
        """The active sub-column names (read-only)."""
        return self._sub_cols

    def parse_cell(self, text: str) -> dict[str, float]:
        """Return ``{sub_col: value, …}`` for *text*."""
        numbers = re.findall(r"\d+(?:\.\d+)?", text.replace(",", "."))
        result: dict[str, float] = dict.fromkeys(self._sub_cols, 0.0)
        result.update(zip(self._sub_cols, map(float, numbers)))
        return result


class DaysHandler(ColumnHandler):
    is_schedule_field = True

    # Order matters: longer/overlapping tokens must come before shorter ones.
    _TOKENS: list[tuple[str, str]] = [
        ("Th", "thursday"),
        ("Sa", "saturday"),
        ("Su", "sunday"),
        ("M",  "monday"),
        ("T",  "tuesday"),
        ("W",  "wednesday"),
        ("F",  "friday"),
    ]

    def parse_cell(self, text: str) -> list[str]:
        """Return a list of full day names, e.g. ``["tuesday", "thursday"]``."""
        result: list[str] = []
        i = 0
        while i < len(text):
            for abbr, day in self._TOKENS:
                if text[i: i + len(abbr)] == abbr:
                    if day not in result:
                        result.append(day)
                    i += len(abbr)
                    break
            else:
                raise ValueError(
                    f"Unrecognised day token at position {i}: {text[i:]!r}"
                )
        return result


class TimeHandler(ColumnHandler):
    is_schedule_field = True
    _TIME_FORMAT = "%I:%M %p"

    def parse_cell(self, text: str) -> dict[str, int]:
        from datetime import datetime

        parts = [p.strip() for p in text.split("-")]
        if len(parts) != 2:
            raise ValueError(
                f"Expected 'HH:MM AM/PM - HH:MM AM/PM', got: {text!r}"
            )
        start = datetime.strptime(parts[0], self._TIME_FORMAT).time()
        end   = datetime.strptime(parts[1], self._TIME_FORMAT).time()
        return {
            # "start": start.hour * 60 + start.minute,
            # "end":   end.hour   * 60 + end.minute,
            "start": start.strftime("%H:%M %p"),
            "end":   end.strftime("%H:%M %p"),
        }


class RoomHandler(ColumnHandler):
    is_schedule_field = True

    def parse_cell(self, text: str) -> str:
        """Return the room location as a string."""
        return text.strip()


class FacultyHandler(ColumnHandler):
    is_schedule_field = True

    def parse_cell(self, text: str) -> str:
        """Return the faculty name as a string."""
        return text.strip()


# Registry
COLUMN_REGISTRY: dict[str, ColumnHandler] = {
    "code":    DefaultHandler(),
    "subject": DefaultHandler(),
    "units":   UnitsHandler(),
    "class":   DefaultHandler(),
    "days":    DaysHandler(),
    "time":    TimeHandler(),
    "room":    RoomHandler(),
    "faculty": FacultyHandler(),
}

_FALLBACK_HANDLER = DefaultHandler()


def get_handler(column_name: str) -> ColumnHandler:
    return COLUMN_REGISTRY.get(column_name, _FALLBACK_HANDLER)