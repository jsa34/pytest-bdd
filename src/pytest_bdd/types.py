"""Common type definitions."""

from __future__ import annotations

from enum import Enum


class StepType(str, Enum):
    GIVEN = "given"
    WHEN = "when"
    THEN = "then"

    def from_value(cls, value: str) -> StepType:
        value = value.strip()
        # Case-insensitive lookup
        for step in cls:
            if step.value.lower() == value.lower():
                return step
        raise ValueError(f"{value} is not a valid StepType")

    def contains(cls, value: str) -> bool:
        """Check if the given string matches any StepType (case-insensitive)."""
        value = value.strip().lower()
        return any(step.value.lower() == value for step in cls)

    def order(self) -> int:
        """Return the order value for sorting."""
        order_map = {
            StepType.GIVEN: 1,
            StepType.WHEN: 2,
            StepType.THEN: 3,
        }
        return order_map.get(self, 0)
