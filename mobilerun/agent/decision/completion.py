"""Validation shared by caller assertions and model-derived completion checks."""


def validate_completion_criteria(criteria: dict) -> None:
    if not isinstance(criteria, dict):
        raise ValueError("Jev completion must be a mapping")
    allowed = {
        "package",
        "activity_contains",
        "text_all",
        "text_any",
        "input_text",
        "checked",
    }
    if set(criteria) - allowed:
        raise ValueError("Unknown completion criterion")
    for name, value in criteria.items():
        if name in {"package", "activity_contains", "input_text"}:
            valid = isinstance(value, str) and bool(value)
        elif name in {"text_all", "text_any"}:
            valid = (
                isinstance(value, list)
                and bool(value)
                and all(isinstance(v, str) and v for v in value)
            )
        else:
            valid = (
                isinstance(value, dict)
                and bool(value)
                and all(
                    isinstance(k, str) and k and isinstance(v, bool)
                    for k, v in value.items()
                )
            )
        if not valid:
            raise ValueError(f"Invalid completion criterion: {name}")
