"""Versioned closed-corpus vocabulary for staff-level OMR targets."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .errors import ProtocolError


VOCAB_SCHEMA = "staff_omr_vocab_v1"
TARGET_PARSER = "utf8_unicode_whitespace_v1"
TOKEN_SORT = "utf8_bytes_ascending_v1"


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], context: str
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ProtocolError(
            f"{context} fields mismatch; missing={missing}, extra={extra}"
        )


@dataclass(frozen=True, slots=True)
class Vocabulary:
    """Validated token/id mapping with CTC blank permanently reserved at zero."""

    dataset_id: str
    source_manifest_sha256: str
    tokens: tuple[str, ...]
    _token_to_id: Mapping[str, int] = field(repr=False, compare=False)

    @classmethod
    def from_document(
        cls,
        document: Any,
        *,
        manifest_sha256: str,
        dataset_id: str,
    ) -> "Vocabulary":
        if not isinstance(document, dict):
            raise ProtocolError("vocabulary must be a JSON object")
        _require_exact_keys(
            document,
            {
                "schema_version",
                "dataset_id",
                "vocabulary_scope",
                "source_manifest_sha256",
                "target_parser",
                "token_sort",
                "unicode_normalization",
                "blank_id",
                "tokens",
            },
            "vocabulary",
        )
        if document["schema_version"] != VOCAB_SCHEMA:
            raise ProtocolError(
                f"unsupported vocabulary schema {document['schema_version']!r}"
            )
        if document["dataset_id"] != dataset_id:
            raise ProtocolError(
                "vocabulary dataset_id does not match bundle/manifest dataset_id"
            )
        if document["source_manifest_sha256"] != manifest_sha256:
            raise ProtocolError(
                "vocabulary source_manifest_sha256 does not match manifest"
            )
        if document["vocabulary_scope"] != "closed_corpus":
            raise ProtocolError(
                "vocabulary_scope must be 'closed_corpus' for this protocol"
            )
        if document["target_parser"] != TARGET_PARSER:
            raise ProtocolError(
                f"target_parser must be {TARGET_PARSER!r}"
            )
        if document["token_sort"] != TOKEN_SORT:
            raise ProtocolError(f"token_sort must be {TOKEN_SORT!r}")
        if document["unicode_normalization"] != "none":
            raise ProtocolError("unicode_normalization must be 'none'")
        if (
            isinstance(document["blank_id"], bool)
            or document["blank_id"] != 0
        ):
            raise ProtocolError("vocabulary blank_id must be integer 0")

        raw_tokens = document["tokens"]
        if not isinstance(raw_tokens, list) or not raw_tokens:
            raise ProtocolError("vocabulary tokens must be a non-empty array")
        if any(not isinstance(token, str) or not token for token in raw_tokens):
            raise ProtocolError("vocabulary token values must be non-empty strings")
        if len(set(raw_tokens)) != len(raw_tokens):
            duplicates = sorted(
                {
                    token
                    for token in raw_tokens
                    if raw_tokens.count(token) > 1
                },
                key=lambda token: token.encode("utf-8"),
            )
            raise ProtocolError(
                f"vocabulary contains duplicate token(s): {duplicates}"
            )
        expected_order = sorted(
            raw_tokens, key=lambda token: token.encode("utf-8")
        )
        if raw_tokens != expected_order:
            raise ProtocolError(
                "vocabulary tokens are not sorted by UTF-8 bytes"
            )

        tokens = tuple(raw_tokens)
        mapping = MappingProxyType(
            {token: index + 1 for index, token in enumerate(tokens)}
        )
        return cls(
            dataset_id=dataset_id,
            source_manifest_sha256=manifest_sha256,
            tokens=tokens,
            _token_to_id=mapping,
        )

    @property
    def blank_id(self) -> int:
        return 0

    @property
    def num_classes(self) -> int:
        return len(self.tokens) + 1

    def encode(self, tokens: Sequence[str]) -> tuple[int, ...]:
        """Map tokens to ids, reporting every out-of-vocabulary token."""
        missing: dict[str, int] = {}
        encoded: list[int] = []
        for token in tokens:
            token_id = self._token_to_id.get(token)
            if token_id is None:
                missing[token] = missing.get(token, 0) + 1
            else:
                encoded.append(token_id)
        if missing:
            ordered = sorted(
                missing.items(), key=lambda item: item[0].encode("utf-8")
            )
            raise ProtocolError(f"OOV token counts: {ordered}")
        return tuple(encoded)

    def decode(self, token_ids: Sequence[int]) -> tuple[str, ...]:
        """Map non-blank ids back to tokens."""
        decoded: list[str] = []
        for token_id in token_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise ProtocolError(f"token id must be an integer: {token_id!r}")
            if token_id == self.blank_id:
                raise ProtocolError("blank id 0 does not map to a vocabulary token")
            if token_id < 1 or token_id > len(self.tokens):
                raise ProtocolError(
                    f"token id {token_id} is outside [1, {len(self.tokens)}]"
                )
            decoded.append(self.tokens[token_id - 1])
        return tuple(decoded)

    def to_document(self) -> dict[str, object]:
        return {
            "schema_version": VOCAB_SCHEMA,
            "dataset_id": self.dataset_id,
            "vocabulary_scope": "closed_corpus",
            "source_manifest_sha256": self.source_manifest_sha256,
            "target_parser": TARGET_PARSER,
            "token_sort": TOKEN_SORT,
            "unicode_normalization": "none",
            "blank_id": self.blank_id,
            "tokens": list(self.tokens),
        }
