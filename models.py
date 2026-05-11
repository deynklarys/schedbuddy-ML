"""Data classes for table detection and extraction."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from typing import List, Union
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Schedule domain models
# ---------------------------------------------------------------------------

class UnitBreakdown(BaseModel):
    credit: float = 0.0
    lec: float = 0.0
    lab: float = 0.0


class TimeRange(BaseModel):
    start: int = Field(..., ge=0, le=1439 )
    end: int = Field(..., ge=0, le=1439)
    
    @field_validator('end')
    @classmethod
    def end_after_start(cls, v, info):
        if info.data.get('start') and v <= info.data['start']:
            raise ValueError('end time must be after start time')
        return v


class Schedule(BaseModel):
    days: List[str] = Field(default_factory=list, examples=[["monday", "wednesday"]])
    time: Optional[TimeRange] = None
    room: Optional[str] = None
    faculty: Optional[str] = None


class CourseRow(BaseModel):
    code: Optional[str] = None
    subject: Optional[str] = None
    units: Optional[Union[UnitBreakdown, float]] = None
    class_section: Optional[str] = Field(None, alias="class")
    schedules: List[Schedule] = Field(default_factory=list)

@dataclass
class Detection:
    label_id: int
    label: str
    score: float
    bbox: list[float]  # [xmin, ymin, xmax, ymax]
    bbox_xywh: list[float]  # [x, y, w, h]


@dataclass
class CellRecord:
    row: int
    column: int
    bbox: Optional[list[int]]
    text: str


@dataclass
class TableData:
    headers: list[str]
    rows: list[dict[str, str]]
    cells: list[dict]
