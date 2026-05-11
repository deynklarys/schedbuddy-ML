import logging
logger = logging.getLogger(__name__)

from models import CourseRow, TableData

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