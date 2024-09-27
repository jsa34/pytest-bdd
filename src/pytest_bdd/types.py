"""Common type definitions."""

from __future__ import annotations

import typing
from enum import Enum

if typing.TYPE_CHECKING:
    from typing_extensions import Self


class KeywordType(Enum):
    CONTEXT = "Context"
    CONJUNCTION = "Conjunction"
    ACTION = "Action"
    OUTCOME = "Outcome"

    @classmethod
    def all(cls):
        return [item.value for item in cls]

    @classmethod
    def all_except_conjunction(cls) -> list[Self]:
        return [item for item in cls if item != cls.CONJUNCTION]

    @classmethod
    def from_string(cls, value: str):
        try:
            # Try to find the enum member that matches the string value
            return cls(value)
        except ValueError:
            # Return None or raise an error if the string is not valid
            return None


class StepType(Enum):
    GIVEN = "given"
    WHEN = "when"
    THEN = "then"

    @classmethod
    def from_keyword_type(cls, keyword_type: KeywordType) -> StepType | None:
        mapping = {
            KeywordType.CONTEXT: cls.GIVEN,
            KeywordType.CONJUNCTION: None,
            KeywordType.ACTION: cls.WHEN,
            KeywordType.OUTCOME: cls.THEN,
        }
        return mapping.get(keyword_type, None)
