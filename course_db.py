"""Course database registry.

Usages:
    # Load every JSON file in a folder (most common)
    db = CourseDatabase.from_dir("databases")

    # Load specific files
    db = CourseDatabase("databases/comsci.json", "databases/engineering.json")

    # Add files after construction
    db.load("databases/nursing.json")
    db.load_dir("extra_databases")

    # Look up the best fuzzy match
    code, score, subject = db.match("COSC 10", min_score=50)

    # Find out which file a code came from
    db.source_of("COSC 101")   # → "databases/comsci.json"
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rapidfuzz import fuzz

logger = logging.getLogger(__name__)


class CourseDatabase:

    def __init__(self, *paths: str | Path) -> None:
        self._data:    dict[str, str] = {}   # code → subject
        self._sources: dict[str, str] = {}   # code → source path

        for path in paths:
            self.load(path)

    @classmethod
    def from_dir(cls, folder: str | Path, pattern: str = "*.json") -> "CourseDatabase":

        instance = cls()
        instance.load_dir(folder, pattern)
        return instance
    
    def load(self, path: str | Path) -> None:

        path = Path(path)
        with path.open(encoding="utf-8") as fh:
            incoming: dict[str, str] = json.load(fh)

        duplicates = self._data.keys() & incoming.keys()
        if duplicates:
            logger.warning(
                "%s: %d code(s) already present and will be overwritten: %s",
                path.name,
                len(duplicates),
                ", ".join(sorted(duplicates)),
            )

        self._data.update(incoming)
        self._sources.update({code: str(path) for code in incoming})
        logger.debug("Loaded %d courses from %s (total: %d)", len(incoming), path.name, len(self._data))

    def load_dir(self, folder: str | Path, pattern: str = "*.json") -> None:

        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(f"Database folder not found: {folder}")
        if not folder.is_dir():
            raise NotADirectoryError(f"Expected a directory, got: {folder}")

        files = sorted(folder.glob(pattern))
        if not files:
            logger.warning("No files matching %r found in %s", pattern, folder)
            return

        for path in files:
            self.load(path)
        logger.info("Loaded %d file(s) from %s", len(files), folder)

    # Querying
    @property
    def codes(self) -> list[str]:
        """All course codes across every loaded database."""
        return list(self._data.keys())

    def subject_of(self, code: str) -> str:
        """Return the subject name for *code*, or ``""`` if not found."""
        return self._data.get(code, "")

    def source_of(self, code: str) -> str:
        """Return the file path that *code* was loaded from, or ``""``."""
        return self._sources.get(code, "")

    def match(
        self,
        query: str,
        min_score: int = 0,
    ) -> tuple[str, int, str]:

        best_code, best_score = query, -1

        for code in self._data:
            score = max(
                fuzz.partial_ratio(query, code),
                fuzz.ratio(query, code),
            )
            if score > min_score and score > best_score:
                best_code, best_score = code, score

        subject = self._data.get(best_code, "")
        return best_code, best_score, subject

    # Dunder helpers
    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        sources = sorted({Path(p).name for p in self._sources.values()})
        return f"CourseDatabase({len(self._data)} courses from {sources})"