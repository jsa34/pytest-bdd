from __future__ import annotations

import linecache
import re
import textwrap
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Union

from gherkin.errors import CompositeParserException
from gherkin.parser import Parser
from typing_extensions import Self

from . import exceptions
from .exceptions import StepError
from .types import KeywordType, StepType

STEP_PARAM_RE = re.compile(r"<(.+?)>")
COMMENT_RE = re.compile(r"(^|(?<=\s))#")

ERROR_PATTERNS = [
    (
        re.compile(r"expected:.*got 'Feature.*'"),
        exceptions.FeatureError,
        "Multiple features are not allowed in a single feature file.",
    ),
    (
        re.compile(r"expected:.*got '(?:Given|When|Then|And|But).*'"),
        exceptions.FeatureError,
        "Step definition outside of a Scenario or a Background.",
    ),
    (
        re.compile(r"expected:.*got 'Background.*'"),
        exceptions.BackgroundError,
        "Multiple 'Background' sections detected. Only one 'Background' is allowed per feature.",
    ),
    (
        re.compile(r"expected:.*got 'Scenario.*'"),
        exceptions.ScenarioError,
        "Misplaced or incorrect 'Scenario' keyword. Ensure it's correctly placed. There might be a missing Feature section.",
    ),
    (
        re.compile(r"expected:.*got 'Given.*'"),
        exceptions.StepError,
        "Improper step keyword detected. Ensure correct order and indentation for steps (Given, When, Then, etc.).",
    ),
    (
        re.compile(r"expected:.*got 'Rule.*'"),
        exceptions.RuleError,
        "Misplaced or incorrectly formatted 'Rule'. Ensure it follows the feature structure.",
    ),
    (
        re.compile(r"expected:.*got '.*'"),
        exceptions.TokenError,
        "Unexpected token found. Check Gherkin syntax near the reported error.",
    ),
]


@dataclass(frozen=True)
class Location:
    column: int
    line: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(column=data["column"], line=data["line"])


@dataclass(frozen=True)
class Comment:
    location: Location
    text: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(location=Location.from_dict(data["location"]), text=data["text"])


@dataclass(frozen=True)
class Cell:
    location: Location
    value: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(location=Location.from_dict(data["location"]), value=_convert_to_raw_string(data["value"]))


@dataclass(frozen=True)
class Row:
    id: str
    location: Location
    cells: list[Cell]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            id=data["id"],
            location=Location.from_dict(data["location"]),
            cells=[Cell.from_dict(cell) for cell in data["cells"]],
        )


@dataclass(frozen=True)
class DataTable:
    location: Location
    name: str | None = None
    table_header: Row | None = None
    table_body: list[Row] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            location=Location.from_dict(data["location"]),
            name=data.get("name"),
            table_header=Row.from_dict(data["tableHeader"]) if data.get("tableHeader") else None,
            table_body=[Row.from_dict(row) for row in data.get("tableBody", [])],
        )

    def as_contexts(self) -> Iterable[dict[str, Any]]:
        """Generate contexts for the examples."""
        if not self.table_header or not self.table_body:
            return  # If header or body is missing, nothing to yield

        example_params = [cell.value for cell in self.table_header.cells]
        for row in self.table_body:
            assert len(example_params) == len(row.cells), "Row length does not match header length"
            yield dict(zip(example_params, [cell.value for cell in row.cells]))


@dataclass(frozen=True)
class DocString:
    content: str
    delimiter: str
    location: Location

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            content=textwrap.dedent(data["content"]),
            delimiter=data["delimiter"],
            location=Location.from_dict(data["location"]),
        )


@dataclass
class Step:
    id: str
    keyword: str
    keyword_type: KeywordType
    location: Location
    text: str
    name: str | None = None
    raw_name: str | None = None
    data_table: DataTable | None = None
    doc_string: DocString | None = None
    step_type: StepType | None = None
    parent: Background | Scenario | None = None
    failed: bool = False
    duration: float | None = None

    def __post_init__(self):
        self._generate_initial_name()
        self.params = tuple(frozenset(STEP_PARAM_RE.findall(self.raw_name)))

    def __str__(self) -> str:
        """Return a string representation of the step."""
        return f'{self.keyword.capitalize()} "{self.name}"'

    def _generate_initial_name(self):
        """Generate an initial name based on the step's text and optional doc_string."""
        self.name = _strip_comments(self.text)
        if self.doc_string:
            self.name = f"{self.name}\n{self.doc_string.content}"
        # Populate a frozen copy of the name untouched by params later
        self.raw_name = self.name

    def get_parent_of_type(self, parent_type) -> Any | None:
        """Return the parent if it's of the specified type."""
        return self.parent if isinstance(self.parent, parent_type) else None

    @property
    def scenario(self) -> Scenario | None:
        return self.get_parent_of_type(Scenario)

    @property
    def background(self) -> Background | None:
        return self.get_parent_of_type(Background)

    def render(self, context: Mapping[str, Any]) -> None:
        """Render the step name with the given context and update the instance.

        Args:
            context (Mapping[str, Any]): The context for rendering the step name.
        """
        _render_steps([self], context)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            id=data["id"],
            keyword=str(data["keyword"]).capitalize().strip(),
            keyword_type=KeywordType.from_string(data["keywordType"]),
            location=Location.from_dict(data["location"]),
            text=data["text"],
            data_table=DataTable.from_dict(data["dataTable"]) if data.get("dataTable") else None,
            doc_string=DocString.from_dict(data["docString"]) if data.get("docString") else None,
        )


@dataclass(frozen=True)
class Tag:
    id: str
    location: Location
    name: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(id=data["id"], location=Location.from_dict(data["location"]), name=data["name"])


@dataclass
class Scenario:
    id: str
    keyword: str
    location: Location
    name: str
    description: str
    steps: list[Step]
    tags: set[Tag]
    examples: list[DataTable] = field(default_factory=list)
    parent: Feature | Rule = None

    def __post_init__(self):
        self.steps = _compute_step_type(self.steps)
        for step in self.steps:
            step.parent = self

    @cached_property
    def tag_names(self) -> set[str]:
        return _get_tag_names(self.tags)

    def render(self, context: Mapping[str, Any]) -> None:
        """Render the scenario's steps with the given context.

        Args:
            context (Mapping[str, Any]): The context for rendering steps.
        """
        _render_steps(self.steps, context)

    @cached_property
    def feature(self) -> Feature:
        return self.parent if _check_instance_by_name(self.parent, "Feature") else None

    @cached_property
    def rule(self) -> Rule:
        return self.parent if _check_instance_by_name(self.parent, "Rule") else None

    @property
    def all_steps(self) -> list[Step]:
        """Get all steps including background steps if present."""
        background_steps = self.feature.background_steps if self.feature else []
        return background_steps + self.steps

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            id=data["id"],
            keyword=data["keyword"],
            location=Location.from_dict(data["location"]),
            name=data["name"],
            description=textwrap.dedent(data["description"]),
            steps=[Step.from_dict(step) for step in data["steps"]],
            tags={Tag.from_dict(tag) for tag in data["tags"]},
            examples=[DataTable.from_dict(example) for example in data.get("examples", [])],
        )


@dataclass
class Rule:
    id: str
    keyword: str
    location: Location
    name: str
    description: str
    tags: set[Tag]
    children: list[Child]
    parent: Feature | None = None

    def __post_init__(self):
        for scenario in self.children:
            scenario.parent = self

    @cached_property
    def tag_names(self) -> set[str]:
        return _get_tag_names(self.tags)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            id=data["id"],
            keyword=data["keyword"],
            location=Location.from_dict(data["location"]),
            name=data["name"],
            description=textwrap.dedent(data["description"]),
            tags={Tag.from_dict(tag) for tag in data["tags"]},
            children=[Child.from_dict(child) for child in data["children"]],
        )


@dataclass
class Background:
    id: str
    keyword: str
    location: Location
    name: str
    description: str
    steps: list[Step]
    parent: Feature | None = None

    def __post_init__(self):
        self.steps = _compute_step_type(self.steps)
        for step in self.steps:
            step.parent = self

    def render(self, context: Mapping[str, Any]) -> None:
        """Render the scenario's steps with the given context.

        Args:
            context (Mapping[str, Any]): The context for rendering steps.
        """
        _render_steps(self.steps, context)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            id=data["id"],
            keyword=data["keyword"],
            location=Location.from_dict(data["location"]),
            name=data["name"],
            description=textwrap.dedent(data["description"]),
            steps=[Step.from_dict(step) for step in data["steps"]],
        )


@dataclass
class Child:
    background: Background | None = None
    rule: Rule | None = None
    scenario: Scenario | None = None
    parent: Feature | Rule | None = None

    def __post_init__(self):
        if self.scenario:
            self.scenario.parent = self.parent
        if self.background:
            self.background.parent = self.parent

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            background=Background.from_dict(data["background"]) if data.get("background") else None,
            rule=Rule.from_dict(data["rule"]) if data.get("rule") else None,
            scenario=Scenario.from_dict(data["scenario"]) if data.get("scenario") else None,
        )


@dataclass
class Feature:
    keyword: str
    location: Location
    name: str
    description: str
    children: list[Child]
    tags: set[Tag]
    language: str
    background: Background | None = None
    abs_filename: str | None = None
    rel_filename: str | None = None

    def __post_init__(self):
        for child in self.children:
            child.parent = self
            if child.scenario:
                child.scenario.parent = self
            if child.background:
                child.background.parent = self

    @property
    def scenarios(self) -> list[Scenario]:
        """Retrieve all scenarios, whether they are directly part of the feature or within rules."""
        return [child.scenario for child in self.children if child.scenario is not None]

    @cached_property
    def rules(self) -> list[Rule]:
        """Retrieve all rules within the feature."""
        return [child.rule for child in self.children if child.rule is not None]

    @property
    def background_steps(self) -> list[Step]:
        return self.background.steps if self.background else []

    def get_child_by_name(self, name: str) -> Scenario | Background | Rule | None:
        """
        Returns the child (Scenario or Background) that has the given name.
        """
        if self.background and self.background.name == name:
            return self.background
        for scenario in self.scenarios:
            if scenario.name == name:
                return scenario
        for rule in self.rules:
            if rule.name == name:
                return rule
        return None

    @cached_property
    def tag_names(self) -> set[str]:
        """Get all tag names of the feature."""
        return _get_tag_names(self.tags)

    def render(self, context: Mapping[str, Any]) -> None:
        """Render the feature’s background and its scenarios."""
        if self.background:
            self.background.render(context)
        for scenario in self.scenarios:
            scenario.render(context)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Factory method to create a Feature from a dictionary."""
        return cls(
            keyword=data["keyword"],
            location=Location.from_dict(data["location"]),
            name=data["name"],
            description=textwrap.dedent(data["description"]),
            children=[Child.from_dict(child) for child in data["children"]],
            tags={Tag.from_dict(tag) for tag in data["tags"]},
            language=data["language"],
            background=Background.from_dict(data["background"]) if data.get("background") else None,
        )


@dataclass
class GherkinDocument:
    feature: Feature
    comments: list[Comment]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            feature=Feature.from_dict(data["feature"]),
            comments=[Comment.from_dict(comment) for comment in data["comments"]],
        )


def _compute_step_type(steps: list[Step]) -> list[Step]:
    # Early exit if the steps list is empty
    if not steps:
        return []

    # Extract the first step and its keyword type
    first_step = steps[0]
    first_keyword_type = first_step.keyword_type

    # Validate that the first step does not start with a conjunction (e.g., "And")
    if first_keyword_type == KeywordType.CONJUNCTION:
        raise StepError(
            message=(
                f"Invalid first step: Expected 'Given', 'When', or 'Then' but got '{first_step.keyword}' "
                f"at line {first_step.location.line}."
            ),
            line=first_step.location.line,
            line_content=first_step.name,
        )

    # Initialize the current step type from the first step's keyword type
    current_type: StepType = StepType.from_keyword_type(first_keyword_type)

    # Loop over all steps, updating their step_type based on the keyword_type
    for step in steps:
        if step.keyword_type in KeywordType.all_except_conjunction():
            # Update current type if it's not a conjunction keyword
            current_type = StepType.from_keyword_type(step.keyword_type)
        # Assign the determined step type to the step
        step.step_type = current_type

    return steps


def get_gherkin_document(abs_filename: str, rel_filename: str, encoding: str = "utf-8") -> GherkinDocument:
    with open(abs_filename, encoding=encoding) as f:
        feature_file_text = f.read()

    try:
        gherkin_data = Parser().parse(feature_file_text)
    except CompositeParserException as e:
        message = e.args[0]
        line = e.errors[0].location["line"]
        line_content = linecache.getline(abs_filename, e.errors[0].location["line"]).rstrip("\n")
        filename = abs_filename
        handle_gherkin_parser_error(message, line, line_content, filename, e)
        # If no patterns matched, raise a generic GherkinParserError
        raise exceptions.GherkinParseError(f"Unknown parsing error: {message}", line, line_content, filename) from e

    # At this point, the `gherkin_data` should be valid if no exception was raised
    gherkin_doc = GherkinDocument.from_dict(gherkin_data)
    gherkin_doc.feature.abs_filename = abs_filename
    gherkin_doc.feature.rel_filename = rel_filename
    return gherkin_doc


def _check_instance_by_name(obj: Any, class_name: str) -> bool:
    return obj.__class__.__name__ == class_name


def _strip_comments(line: str) -> str:
    """Remove comments from a line of text.

    Args:
        line (str): The line of text from which to remove comments.

    Returns:
        str: The line of text without comments, with leading and trailing whitespace removed.
    """
    if "#" not in line:
        return line
    if res := COMMENT_RE.search(line):
        line = line[: res.start()]
    return line.strip()


def _get_tag_names(tags: set[Tag]):
    return {tag.name.lstrip("@") for tag in tags}


def _convert_to_raw_string(normal_string: str) -> str:
    return normal_string.replace("\\", "\\\\")


def _render_steps(steps: list[Step], context: Mapping[str, Any]) -> None:
    """
    Render multiple steps in batch by applying the context to each step's text.

    Args:
        steps (List[Step]): The list of steps to render.
        context (Mapping[str, Any]): The context to apply to the step names.
    """
    # Create a map of parameter replacements for all steps at once
    # This will store {param: replacement} for each variable found in steps
    replacements = {param: context.get(param, f"<{param}>") for step in steps for param in step.params}

    # Precompute replacement function
    def replacer(text: str) -> str:
        return STEP_PARAM_RE.sub(lambda m: replacements.get(m.group(1), m.group(0)), text)

    # Apply the replacement in batch
    for step in steps:
        step.name = replacer(step.raw_name)


def handle_gherkin_parser_error(
    raw_error: str, line: int, line_content: str, filename: str, original_exception: Exception | None = None
) -> None:
    """Map the error message to a specific exception type and raise it."""
    # Split the raw_error into individual lines
    error_lines = raw_error.splitlines()

    # Check each line against all error patterns
    for error_line in error_lines:
        for pattern, exception_class, message in ERROR_PATTERNS:
            if pattern.search(error_line):
                # If a match is found, raise the corresponding exception with the formatted message
                if original_exception:
                    raise exception_class(message, line, line_content, filename) from original_exception
                else:
                    raise exception_class(message, line, line_content, filename)
