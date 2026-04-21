def normalize_days(days_string: str) -> list[str]:
    """
    Normalizes a concatenated day string (e.g. 'MWF', 'TTh', 'MTTh')
    into a list of full day names.

    Parsing is greedy left-to-right, longest match first.

    Args:
        days_string: Concatenated day abbreviations, e.g. 'TTh', 'MWF', 'MTWThF'

    Returns:
        A list of normalized full day names in order of appearance.

    Raises:
        ValueError: If a portion of the string cannot be matched to any day.
    """
    # Order matters: longer/overlapping tokens must come before shorter ones
    # e.g. 'Th' before 'T', 'Sa'/'Su' before 'S'
    TOKENS = [
        ("Th", "thursday"),
        ("Sa", "saturday"),
        ("Su", "sunday"),
        ("M",  "monday"),
        ("T",  "tuesday"),
        ("W",  "wednesday"),
        ("F",  "friday"),
    ]

    result = []
    i = 0
    while i < len(days_string):
        matched = False
        for abbr, day in TOKENS:
            if days_string[i:i + len(abbr)] == abbr:
                if day not in result:  # Avoid duplicates
                    result.append(day)
                i += len(abbr)
                matched = True
                break
        if not matched:
            raise ValueError(f"Unrecognized day token at position {i}: '{days_string[i:]}'")

    return result

