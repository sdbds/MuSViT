import re


TOKENIZATION_MODES = frozenset({"kern", "ekern", "bekern"})
KERN_AVOID_TOKENS = frozenset(
    {
        "*tremolo",
        "*staff2",
        "*staff1",
        "*Xped",
        "*ped",
        "*Xtuplet",
        "*tuplet",
        "*Xtremolo",
        "*cue",
        "*Xcue",
        "*rscale:1/2",
        "*rscale:1",
        "*kcancel",
        "*below",
    }
)


def validate_tokenization_mode(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("tokenization_mode must be a string")
    if value not in TOKENIZATION_MODES:
        supported = ", ".join(sorted(TOKENIZATION_MODES))
        raise ValueError(
            f"tokenization_mode must be one of {supported}; got {value!r}"
        )
    return value


def clean_kern(
    kern: str,
    avoid_tokens: frozenset[str] = KERN_AVOID_TOKENS,
) -> str:
    lines = []
    for line in kern.split("\n"):
        fields = line.split("\t")
        if any(token in fields for token in avoid_tokens):
            continue
        if all(token == "*" for token in fields):
            continue
        lines.append(line.replace("\n", ""))
    return "\n".join(lines)


def parse_kern_file(kern: str, tokenization_mode: str = "bekern") -> list[str]:
    tokenization_mode = validate_tokenization_mode(tokenization_mode)
    kern = clean_kern(kern)
    kern = kern.replace(" ", " <s> ")
    kern = kern.replace("\t", " <t> ")
    kern = kern.replace("\n", " <b> ")
    kern = kern.replace(" /", "")
    kern = kern.replace(" \\", "")
    kern = kern.replace("·/", "")
    kern = kern.replace("·\\", "")

    if tokenization_mode == "kern":
        kern = kern.replace("·", "").replace("@", "")
    elif tokenization_mode == "ekern":
        kern = kern.replace("·", " ").replace("@", "")
    else:
        kern = kern.replace("·", " ").replace("@", " ")

    tokens = kern.split(" ")[4:-1]
    return [re.sub(r"(?<=\=)\d+", "", token) for token in tokens]
