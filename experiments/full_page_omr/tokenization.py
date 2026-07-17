TOKENIZATION_MODES = frozenset({"kern", "ekern", "bekern"})


def validate_tokenization_mode(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("tokenization_mode must be a string")
    if value not in TOKENIZATION_MODES:
        supported = ", ".join(sorted(TOKENIZATION_MODES))
        raise ValueError(
            f"tokenization_mode must be one of {supported}; got {value!r}"
        )
    return value
