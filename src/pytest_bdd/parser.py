from __future__ import annotations

import linecache
import os.path
import re
import textwrap
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Self, Sequence

from gherkin.errors import CompositeParserException
from gherkin.parser import Parser
from gherkin.token_scanner import TokenScanner

from .exceptions import FeatureError
from .types import StepType

STEP_PARAM_RE = re.compile(r"<(.+?)>")
COMMENT_RE = re.compile(r"(^|(?<=\s))#")


class GherkinModel:
    @staticmethod
    def strip_comments(line: str) -> str:
        """Remove comments from a line of text.

        Args:
            line (str): The line of text from which to remove comments.

        Returns:
            str: The line of text without comments, with leading and trailing whitespace removed.
        """
        if res := COMMENT_RE.search(line):
            line = line[: res.start()]
        return line.strip()

    @staticmethod
    def detent_text(text: str, strip_text: bool = True):
        """Remove tabbing from text.

        Args:
            text (str): The line of text from which to remove tabs.
            strip_text (Optional[bool]): Strip whitespace from text before returning.

        Returns:
            str: The line of text without tabs, and optionally returns with leading and trailing whitespace removed.
        """

        text = textwrap.dedent(text)
        return text.strip() if strip_text else text

    @staticmethod
    def get_tag_names(tag_data: list[dict]) -> set[str]:
        """Extract tag names from tag data.

        Args:
            tag_data (List[dict]): The tag data to extract names from.

        Returns:
            set[str]: A set of tag names.
        """
        return {tag["name"].lstrip("@") for tag in tag_data}


@dataclass(eq=False)
class Feature:
    """Represents a feature parsed from a feature file.

    Attributes:
        scenarios (OrderedDict[str, ScenarioTemplate]): A dictionary of scenarios in the feature.
        filename (str): The absolute path of the feature file.
        rel_filename (str): The relative path of the feature file.
        name (Optional[str]): The name of the feature.
        tags (set[str]): A set of tags associated with the feature.
        background (Optional[Background]): The background steps for the feature, if any.
        line_number (int): The line number where the feature starts in the file.
        description (str): The description of the feature.
    """

    scenarios: OrderedDict[str, ScenarioTemplate]
    filename: str
    rel_filename: str
    name: str | None
    tags: set[str]
    background: Background | None
    line_number: int
    description: str


@dataclass(eq=False)
class Examples:
    """Represents examples used in scenarios for parameterization.

    Attributes:
        line_number (Optional[int]): The line number where the examples start.
        name (Optional[str]): The name of the examples.
        example_params (Optional[List[str]]): The names of the parameters for the examples.
        examples (Optional[List[Sequence[str]]]): The list of example rows.
    """

    line_number: int | None = None
    name: str | None = None
    example_params: list[str] = field(default_factory=list)
    examples: list[Sequence[str]] = field(default_factory=list)

    def as_contexts(self) -> Iterable[dict[str, Any]]:
        """Generate contexts for the examples.

        Yields:
            Dict[str, Any]: A dictionary mapping parameter names to their values for each example row.
        """
        for row in self.examples:
            assert len(self.example_params) == len(row)
            yield dict(zip(self.example_params, row))

    @classmethod
    def from_gherkin_document_dict(cls, gherkin_dict: dict):
        return cls(
            line_number=gherkin_dict["location"]["line"],
            name=gherkin_dict["name"],
            example_params=[cell["value"] for cell in gherkin_dict["tableHeader"]["cells"]],
            examples=[
                [str(cell["value"]) if cell["value"] is not None else "" for cell in row["cells"]]
                for row in gherkin_dict["tableBody"]
            ],
        )

    def __bool__(self) -> bool:
        """Check if there are any examples.

        Returns:
            bool: True if there are examples, False otherwise.
        """
        return bool(self.examples)


@dataclass(eq=False)
class ScenarioTemplate(GherkinModel):
    """Represents a scenario template within a feature.

    Attributes:
        feature (Feature): The feature to which this scenario belongs.
        name (str): The name of the scenario.
        line_number (int): The line number where the scenario starts in the file.
        templated (bool): Whether the scenario is templated.
        description (Optional[str]): The description of the scenario.
        tags (set[str]): A set of tags associated with the scenario.
        _steps (List[Step]): The list of steps in the scenario (internal use only).
        examples (Optional[Examples]): The examples used for parameterization in the scenario.
    """

    feature: Feature
    name: str
    line_number: int
    templated: bool
    description: str | None = None
    tags: set[str] = field(default_factory=set)
    _steps: list[Step] = field(init=False, default_factory=list)
    examples: Examples | None = field(default_factory=Examples)

    def add_step(self, step: Step) -> None:
        """Add a step to the scenario.

        Args:
            step (Step): The step to add.
        """
        step.scenario = self
        self._steps.append(step)

    @property
    def steps(self) -> list[Step]:
        """Get all steps for the scenario, including background steps.

        Returns:
            List[Step]: A list of steps, including any background steps from the feature.
        """
        return (self.feature.background.steps if self.feature.background else []) + self._steps

    def render(self, context: Mapping[str, Any]) -> Scenario:
        """Render the scenario with the given context.

        Args:
            context (Mapping[str, Any]): The context for rendering steps.

        Returns:
            Scenario: A Scenario object with steps rendered based on the context.
        """
        background_steps = self.feature.background.steps if self.feature.background else []
        scenario_steps = [
            Step(
                name=step.render(context),
                type=step.type,
                indent=step.indent,
                line_number=step.line_number,
                keyword=step.keyword,
            )
            for step in self._steps
        ]
        steps = background_steps + scenario_steps
        return Scenario(
            feature=self.feature,
            name=self.name,
            line_number=self.line_number,
            steps=steps,
            tags=self.tags,
            description=self.description,
        )

    @classmethod
    def from_gherkin_document_dict(cls, feature, gherkin_dict: dict) -> Self:
        return cls(
            feature=feature,
            name=cls.strip_comments(gherkin_dict["name"]),
            line_number=gherkin_dict["location"]["line"],
            templated="examples" in gherkin_dict,
            tags=cls.get_tag_names(gherkin_dict["tags"]),
            description=textwrap.dedent(gherkin_dict.get("description", "")),
        )


@dataclass(eq=False)
class Scenario:
    """Represents a scenario with steps.

    Attributes:
        feature (Feature): The feature to which this scenario belongs.
        name (str): The name of the scenario.
        line_number (int): The line number where the scenario starts in the file.
        steps (List[Step]): The list of steps in the scenario.
        description (Optional[str]): The description of the scenario.
        tags (set[str]): A set of tags associated with the scenario.
    """

    feature: Feature
    name: str
    line_number: int
    steps: list[Step]
    description: str | None = None
    tags: set[str] = field(default_factory=set)


@dataclass(eq=False)
class Step(GherkinModel):
    """Represents a step within a scenario or background.

    Attributes:
        type (str): The type of step (e.g., 'given', 'when', 'then').
        _name (str): The name of the step.
        line_number (int): The line number where the step starts in the file.
        indent (int): The indentation level of the step.
        keyword (str): The keyword used for the step (e.g., 'Given', 'When', 'Then').
        failed (bool): Whether the step has failed (internal use only).
        scenario (Optional[ScenarioTemplate]): The scenario to which this step belongs (internal use only).
        background (Optional[Background]): The background to which this step belongs (internal use only).
        lines (List[str]): Additional lines for the step (internal use only).
    """

    type: str
    _name: str
    line_number: int
    indent: int
    keyword: str
    failed: bool = field(init=False, default=False)
    scenario: ScenarioTemplate | None = field(init=False, default=None)
    background: Background | None = field(init=False, default=None)
    lines: list[str] = field(init=False, default_factory=list)

    def __init__(
        self, name: str, type: str, indent: int, line_number: int, keyword: str, background: Background | None = None
    ) -> None:
        """Initialize a step.

        Args:
            name (str): The name of the step.
            type (str): The type of the step (e.g., 'given', 'when', 'then').
            indent (int): The indentation level of the step.
            line_number (int): The line number where the step starts in the file.
            keyword (str): The keyword used for the step (e.g., 'Given', 'When', 'Then').
            background (Optional[Background]): Background context of step (only if a background step)
        """
        self.name = name
        self.type = type
        self.indent = indent
        self.line_number = line_number
        self.keyword = keyword

    def __str__(self) -> str:
        """Return a string representation of the step.

        Returns:
            str: A string representation of the step.
        """
        return f'{self.type.capitalize()} "{self.name}"'

    @property
    def params(self) -> tuple[str, ...]:
        """Get the parameters in the step name.

        Returns:
            Tuple[str, ...]: A tuple of parameter names found in the step name.
        """
        return tuple(frozenset(STEP_PARAM_RE.findall(self.name)))

    def render(self, context: Mapping[str, Any]) -> str:
        """Render the step name with the given context, but avoid replacing text inside angle brackets if context is missing.

        Args:
            context (Mapping[str, Any]): The context for rendering the step name.

        Returns:
            str: The rendered step name with parameters replaced only if they exist in the context.
        """

        def replacer(m: re.Match) -> str:
            varname = m.group(1)
            # If the context contains the variable, replace it. Otherwise, leave it unchanged.
            return str(context.get(varname, f"<{varname}>"))

        return STEP_PARAM_RE.sub(replacer, self.name)

    @classmethod
    def from_gherkin_document_dict(
        cls, gherkin_dict: dict, step_type: StepType | None = None, background: Background | None = None
    ) -> Self:
        """Parse Gherkin step data into Step objects.

        Args:
            gherkin_dict (dict): The Gherkin step data.
            step_type (Optional[StepType]): provide the step_type to be associated if the keyword is And or But

        Returns:
            Step: A Step object.
        """
        name = cls.strip_comments(gherkin_dict["text"])
        if "docString" in gherkin_dict:
            doc_string = textwrap.dedent(gherkin_dict["docString"]["content"])
            name = f"{name}\n{doc_string}"
        keyword = cls.get_keyword(gherkin_dict)
        return cls(
            name=name,
            type=step_type or StepType.from_value(keyword),
            indent=gherkin_dict["location"]["column"] - 1,
            line_number=gherkin_dict["location"]["line"],
            keyword=keyword.title(),
        )

    @staticmethod
    def get_keyword(gherkin_dict: dict):
        return gherkin_dict["keyword"].strip()

    @classmethod
    def from_gherkin_document_list_of_dicts(
        cls, gherkin_dicts: list[dict], background: Background | None = None
    ) -> list[Self]:
        steps = []
        current_step_type = None
        for gherkin_dict in gherkin_dicts:
            raw_keyword = cls.get_keyword(gherkin_dict)
            if StepType.contains(raw_keyword):
                current_step_type = StepType.from_value(raw_keyword)
            step = cls.from_gherkin_document_dict(gherkin_dict, current_step_type)
            steps.append(step)
        return steps


@dataclass(eq=False)
class Background:
    """Represents the background steps for a feature.

    Attributes:
        feature (Feature): The feature to which this background belongs.
        line_number (int): The line number where the background starts in the file.
        steps (List[Step]): The list of steps in the background.
    """

    feature: Feature
    line_number: int
    steps: list[Step] = field(default_factory=list)

    @classmethod
    def from_gherkin_document_dict(cls, feature, gherkin_dict: dict) -> Self:
        cls(
            feature=feature,
            line_number=gherkin_dict["location"]["line"],
            steps=Step.from_gherkin_document_list_of_dicts(gherkin_dict["steps"], background=cls),
        )


class FeatureParser(GherkinModel):
    """Converts a feature file into a Feature object.

    Args:
        basedir (str): The basedir for locating feature files.
        filename (str): The filename of the feature file.
        encoding (str): File encoding of the feature file to parse.
    """

    def __init__(self, basedir: str, filename: str, encoding: str = "utf-8"):
        self.abs_filename = os.path.abspath(os.path.join(basedir, filename))
        self.rel_filename = os.path.join(os.path.basename(basedir), filename)
        self.encoding = encoding

    @staticmethod
    def parse_scenario(scenario_data: dict, feature: Feature) -> ScenarioTemplate:
        """Parse a scenario data dictionary into a ScenarioTemplate object.

        Args:
            scenario_data (dict): The dictionary containing scenario data.
            feature (Feature): The feature to which this scenario belongs.

        Returns:
            ScenarioTemplate: A ScenarioTemplate object representing the parsed scenario.
        """
        scenario = ScenarioTemplate.from_gherkin_document_dict(feature, scenario_data)
        for step in Step.from_gherkin_document_list_of_dicts(scenario_data["steps"]):
            scenario.add_step(step)

        if scenario.templated:
            for example_data in scenario_data["examples"]:
                scenario.examples = Examples.from_gherkin_document_dict(example_data)

        return scenario

    @staticmethod
    def parse_background(background_data: dict, feature: Feature) -> Background:
        return Background.from_gherkin_document_dict(feature, background_data)

    def _parse_feature_file(self) -> dict:
        """Parse a feature file into a Feature object.

        Returns:
            Dict: A Gherkin document representation of the feature file.

        Raises:
            FeatureError: If there is an error parsing the feature file.
        """
        with open(self.abs_filename, encoding=self.encoding) as f:
            file_contents = f.read()
        try:
            return Parser().parse(TokenScanner(file_contents))
        except CompositeParserException as e:
            raise FeatureError(
                e.args[0],
                e.errors[0].location["line"],
                linecache.getline(self.abs_filename, e.errors[0].location["line"]).rstrip("\n"),
                self.abs_filename,
            ) from e

    def parse(self):
        data = self._parse_feature_file()
        feature_data = data["feature"]
        feature = Feature(
            scenarios=OrderedDict(),
            filename=self.abs_filename,
            rel_filename=self.rel_filename,
            name=GherkinModel().strip_comments(feature_data["name"]),
            tags=self.get_tag_names(feature_data["tags"]),
            background=None,
            line_number=feature_data["location"]["line"],
            description=textwrap.dedent(feature_data.get("description", "")),
        )

        for child in feature_data["children"]:
            if "background" in child:
                feature.background = self.parse_background(child["background"], feature)
            elif "scenario" in child:
                scenario = self.parse_scenario(child["scenario"], feature)
                feature.scenarios[scenario.name] = scenario

        return feature
