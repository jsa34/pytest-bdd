"""Common type definitions."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from typing_extensions import Literal

from enum import Enum


class KeywordType(Enum):
    CONTEXT = "Context"
    CONJUNCTION = "Conjunction"
    ACTION = "Action"
    OUTCOME = "Outcome"

    @classmethod
    def all(cls):
        return [item.value for item in cls]

    @classmethod
    def all_except_conjunction(cls):
        return [item.value for item in cls if item != cls.CONJUNCTION]

    @classmethod
    def from_string(cls, value: str):
        try:
            # Try to find the enum member that matches the string value
            return cls(value)
        except ValueError:
            # Return None or raise an error if the string is not valid
            return None


GIVEN: Literal["given"] = "given"
WHEN: Literal["when"] = "when"
THEN: Literal["then"] = "then"

STEP_TYPES = (GIVEN, WHEN, THEN)
