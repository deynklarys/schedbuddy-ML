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

from models import TimeRange, UnitBreakdown

logger = logging.getLogger(__name__)

class ColumnHandler(ABC):

    is_schedule_field: bool = False

    def configure(self, raw_header: str) -> None:  # noqa: B027
        """Called once with the raw OCR text of this column's header cell.

        Override to do dynamic configuration (e.g. reading sub-column names
        from the header text).  The default implementation is a no-op.
        """

    @property
    def sub_columns(self) -> tuple[str, ...]:
        """The active sub-column names (read-only). Override in subclasses."""
        return ()

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

    def parse_cell(self, text: str) -> UnitBreakdown:
        """Return a ``UnitBreakdown`` for *text*."""
        numbers = re.findall(r"\d+(?:\.\d+)?", text.replace(",", "."))
        parsed = dict.fromkeys(self._sub_cols, 0.0)
        parsed.update(zip(self._sub_cols, map(float, numbers)))
        return UnitBreakdown(**parsed)


class DaysHandler(ColumnHandler):
    is_schedule_field = True

    _DAY_ALIASES: dict[str, list[str]] = {
        "monday":    ["Mon", "M"],
        "tuesday":   ["Tue", "Tu", "T"],
        "wednesday": ["Wed", "W"],
        "thursday":  ["Thu", "Th"],
        "friday":    ["Fri", "F"],
        "saturday":  ["Sat", "Sa", "S"],
        "sunday":    ["Sun", "Su"],
    }

    _TOKENS: list[tuple[str, str]] = sorted(
        [
            (abbr, day)
            for day, abbrs in _DAY_ALIASES.items()
            for abbr in abbrs
        ],
        key=lambda pair: len(pair[0]),
        reverse=True,
    )

    def parse_cell(self, text: str) -> list[str]:
        cleaned = re.sub(r"[^a-zA-Z]", "", text).lower()
        result: list[str] = []
        i = 0
        while i < len(cleaned):
            for abbr, day in self._TOKENS:
                if cleaned[i: i + len(abbr)] == abbr.lower():
                    if day not in result:
                        result.append(day)
                    i += len(abbr)
                    break
            else:
                logger.warning(f"Unrecognised day token at position {i}: {text[i:]!r}; sending empty string")
                result.append("")
                i += 1
        
        if all(day == "" for day in result):
            logger.warning(f"Failed to parse any valid day tokens from: {text!r}; returning empty list")
            return []
            
        return result


class TimeHandler(ColumnHandler):
    is_schedule_field = True

    _TIME_RE = re.compile(
        r'(\d{1,2}):(\d{2})\s*(AM|PM)',
        re.IGNORECASE
    )

    def parse_cell(self, text: str) -> TimeRange:
        matches = self._TIME_RE.findall(text)
        if len(matches) < 2:
            raise ValueError(
                f"Expected two time values (HH:MM AM/PM) in: {text!r}"
            )

        start = self._to_minutes(*matches[0])
        end   = self._to_minutes(*matches[1])
        return TimeRange(start=start, end=end)

    @staticmethod
    def _to_minutes(hour: str, minute: str, meridian: str) -> int:
        h, m = int(hour), int(minute)
        mer = meridian.upper()
        if mer == "PM" and h != 12:
            h += 12
        elif mer == "AM" and h == 12:
            h = 0
        return h * 60 + m


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

if __name__ == "__main__":
    # --- UnitsHandler smoke tests ---
    handler = get_handler("units")
    handler.configure("Units (Credit/Lec/Lab)")
    print(handler.sub_columns)
    print(handler.parse_cell("a 2,5 1.0"))

    # --- DaysHandler smoke tests ---
    dh = get_handler("days")
    cases = [
        ("MTh",       ["monday", "thursday"]),
        ("T Th",      ["tuesday", "thursday"]),
        ("MWF",       ["monday", "wednesday", "friday"]),
        ("Sa",        ["saturday"]),
        ("N/A",       []),
        ("M1Th",      ["monday", "thursday"]),   # digit noise stripped
        ("",          []),
        ("a ee ",          []),
    ]
    print("=== DaysHandler ===")
    for raw, expected in cases:
        result = dh.parse_cell(raw)
        status = "✓" if result == expected else f"✗ (expected {expected})"
        print(f"  {raw!r:12} → {result}  {status}")
 