from dataclasses import dataclass
import operator

import editdistance


@dataclass(frozen=True)
class CanonicalTokenStream:
    tokens: tuple[str, ...]
    terminated_by_eos: bool
    truncated: bool


@dataclass(frozen=True)
class MetricViews:
    cer: tuple[str, ...]
    ser: tuple[str, ...]
    ler: tuple[str, ...]


def _token_strings(token_ids, i2w) -> tuple[str, ...]:
    tokens = []
    for raw_id in token_ids:
        if isinstance(raw_id, bool):
            raise TypeError("token ids must be integers, got bool")
        try:
            token_id = operator.index(raw_id)
        except TypeError as exc:
            raise TypeError(f"token ids must be integers, got {raw_id!r}") from exc
        try:
            token = i2w[token_id]
        except KeyError:
            raise KeyError(f"Unknown token id {token_id}") from None
        tokens.append(token)
    return tuple(tokens)


def _canonicalize_ids(token_ids, i2w, *, target: bool,
                      maxlen: int | None) -> CanonicalTokenStream:
    tokens = _token_strings(token_ids, i2w)
    if tokens and tokens[0] == "<bos>":
        tokens = tokens[1:]

    try:
        eos_index = tokens.index("<eos>")
    except ValueError:
        if target:
            raise ValueError("ground truth sequence is missing <eos>") from None
        if isinstance(maxlen, bool) or not isinstance(maxlen, int) or maxlen <= 0:
            raise ValueError("maxlen must be a positive integer")
        raw_length = len(tokens) + 1
        if raw_length < maxlen:
            raise ValueError("prediction stopped before maxlen without <eos>") from None
        if raw_length > maxlen:
            raise ValueError(
                f"prediction length {raw_length} exceeds maxlen {maxlen}"
            ) from None
        return CanonicalTokenStream(tokens, terminated_by_eos=False, truncated=True)

    content = tokens[:eos_index]
    if target:
        invalid_special = next(
            (token for token in content if token in {"<bos>", "<pad>"}),
            None,
        )
        if invalid_special is not None:
            raise ValueError(
                f"ground truth contains special token {invalid_special} before <eos>"
            )

    return CanonicalTokenStream(
        content,
        terminated_by_eos=True,
        truncated=False,
    )


def canonicalize_target_ids(token_ids, i2w) -> CanonicalTokenStream:
    return _canonicalize_ids(token_ids, i2w, target=True, maxlen=None)


def canonicalize_prediction_ids(token_ids, i2w, *, maxlen: int) -> CanonicalTokenStream:
    return _canonicalize_ids(token_ids, i2w, target=False, maxlen=maxlen)


def canonical_text(stream: CanonicalTokenStream) -> str:
    replacements = {
        "<s>": " ",
        "<t>": "\t",
        "<b>": "\n",
    }
    return "".join(replacements.get(token, token) for token in stream.tokens)


def _symbol_view(tokens: tuple[str, ...]) -> tuple[str, ...]:
    symbols = []
    current = []

    def flush_current():
        if current:
            symbols.append("".join(current))
            current.clear()

    for token in tokens:
        if token == "<s>":
            flush_current()
        elif token in {"<t>", "<b>"}:
            flush_current()
            symbols.append(token)
        else:
            current.append(token)
    flush_current()
    return tuple(symbols)


def metric_views(stream: CanonicalTokenStream) -> MetricViews:
    text = canonical_text(stream)
    return MetricViews(
        cer=tuple(text),
        ser=_symbol_view(stream.tokens),
        ler=tuple(text.splitlines()),
    )


def _micro_error_rate(predictions: tuple[MetricViews, ...],
                      targets: tuple[MetricViews, ...], view_name: str) -> float:
    accumulated_distance = 0
    accumulated_length = 0
    for prediction, target in zip(predictions, targets, strict=True):
        prediction_units = getattr(prediction, view_name)
        target_units = getattr(target, view_name)
        accumulated_distance += editdistance.eval(prediction_units, target_units)
        accumulated_length += len(target_units)

    if accumulated_length == 0:
        raise ValueError(f"ground-truth {view_name.upper()} unit count is zero")
    return 100.0 * accumulated_distance / accumulated_length


def compute_canonical_metrics(predictions, targets) -> tuple[float, float, float]:
    predictions = tuple(predictions)
    targets = tuple(targets)
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must contain the same number of samples")

    prediction_views = tuple(metric_views(stream) for stream in predictions)
    target_views = tuple(metric_views(stream) for stream in targets)
    return tuple(
        _micro_error_rate(prediction_views, target_views, view_name)
        for view_name in ("cer", "ser", "ler")
    )


def levenshtein_legacy(a, b):
    """Pure-Python reference retained only for migration-equivalence tests."""
    n, m = len(a), len(b)
    if n > m:
        a, b = b, a
        n, m = m, n

    current = range(n + 1)
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change += 1
            current[j] = min(add, delete, change)
    return current[n]


def parse_krn_content(krn, ler_parsing=False, cer_parsing=False):
    """Legacy parser retained for the one-time metric migration report."""
    if cer_parsing:
        krn = krn.replace("\n", " <b> ")
        krn = krn.replace("\t", " <t> ")
        tokens = krn.split(" ")
        characters = []
        for token in tokens:
            if token not in ["<b>", "<t>"]:
                characters.append(token)
            else:
                characters.extend(token)
        return characters
    if ler_parsing:
        krn_lines = krn.split("\n")
        for index, line in enumerate(krn_lines):
            line = line.replace("\n", " <b> ")
            line = line.replace("\t", " <t> ")
            krn_lines[index] = line
        return krn_lines

    krn = krn.replace("\n", " <b> ")
    krn = krn.replace("\t", " <t> ")
    return krn.split(" ")


def _compute_legacy_metric(predictions, targets):
    accumulated_distance = 0
    accumulated_length = 0
    for prediction, target in zip(predictions, targets):
        accumulated_distance += levenshtein_legacy(prediction, target)
        accumulated_length += len(target)
    return 100.0 * accumulated_distance / accumulated_length


def compute_poliphony_metrics_legacy(hyp_array, gt_array):
    hyp_cer = []
    gt_cer = []
    hyp_ser = []
    gt_ser = []
    hyp_ler = []
    gt_ler = []

    for hypothesis, ground_truth in zip(hyp_array, gt_array):
        hyp_ler.append(parse_krn_content(hypothesis, ler_parsing=True))
        gt_ler.append(parse_krn_content(ground_truth, ler_parsing=True))
        hyp_ser.append(parse_krn_content(hypothesis))
        gt_ser.append(parse_krn_content(ground_truth))
        hyp_cer.append(parse_krn_content(hypothesis, cer_parsing=True))
        gt_cer.append(parse_krn_content(ground_truth, cer_parsing=True))

    return (
        _compute_legacy_metric(hyp_cer, gt_cer),
        _compute_legacy_metric(hyp_ser, gt_ser),
        _compute_legacy_metric(hyp_ler, gt_ler),
    )


# Compatibility for the running legacy call site until the trainer migration lands.
compute_poliphony_metrics = compute_poliphony_metrics_legacy


def extract_music_text(array):
    lines = array.split("\n")
    lyrics = []
    symbols = []
    for index, line in enumerate(lines):
        if ".\t.\n" in line:
            continue
        if index > 0 and len(line.rstrip().split("\t")) > 1:
            symbols.append(line.rstrip().split("\t")[0])
            lyrics.append(line.rstrip().split("\t")[1])
    return lyrics, symbols, " ".join(lyrics)


def extract_music_textllevel(array):
    lines = []
    line_content = []
    complete_content = []
    for line in array.split("\n"):
        line = line.replace("\n", "<b>")
        line = line.split("\t")
        if len(line) > 1:
            line_content.append(line[0])
            complete_content.append(line[0])
            line_content.append("<t>")
            complete_content.append("<t>")
            for token in line[1]:
                if token != "<":
                    line_content.append(token)
                    complete_content.append(token)
                else:
                    line_content.append("<b>")
                    break
        lines.append(line_content)
        line_content = []
    return lines, complete_content
