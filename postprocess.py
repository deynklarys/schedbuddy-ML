import logging
logger = logging.getLogger(__name__)

from typing import Any

from pydantic import BaseModel
from models import TimeRange, CourseRow, TableData

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