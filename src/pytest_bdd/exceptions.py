"""pytest-bdd Exceptions."""

from __future__ import annotations


class ScenarioIsDecoratorOnly(Exception):
    """Scenario can be only used as decorator."""


class ScenarioValidationError(Exception):
    """Base class for scenario validation."""


class ScenarioNotFound(ScenarioValidationError):
    """Scenario Not Found."""


class StepDefinitionNotFoundError(Exception):
    """Step definition not found."""


class NoScenariosFound(Exception):
    """No scenarios found."""


class GherkinParseError(Exception):
    """Base class for all Gherkin parsing errors."""

    def __init__(self, message: str, line: int, line_content: str, filename: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.line = line
        self.line_content = line_content
        self.filename = filename

    def __str__(self) -> str:
        error_str = f"{self.message}\nLine number: {self.line}\nLine: {self.line_content}"
        if self.filename:
            error_str = error_str + f"\nFile: {self.filename}"
        return error_str


class FeatureError(GherkinParseError):
    pass


class BackgroundError(GherkinParseError):
    pass


class ScenarioError(GherkinParseError):
    pass


class StepError(GherkinParseError):
    pass


class RuleError(GherkinParseError):
    pass


class TokenError(GherkinParseError):
    pass
