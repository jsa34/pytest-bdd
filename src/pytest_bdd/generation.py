"""pytest-bdd missing test code generation."""

from __future__ import annotations

import itertools
import os.path
from typing import TYPE_CHECKING, cast

from _pytest._io import TerminalWriter
from attr import dataclass
from mako.lookup import TemplateLookup  # type: ignore

from .compat import getfixturedefs
from .feature import get_features
from .parser import Background, Feature, Scenario, Step
from .scenario import inject_fixturedefs_for_step, make_python_docstring, make_python_name, make_string_literal
from .steps import get_step_fixture_name
from .types import StepType

if TYPE_CHECKING:
    from typing import Any, Sequence

    from _pytest.config import Config
    from _pytest.config.argparsing import Parser
    from _pytest.fixtures import FixtureDef, FixtureManager
    from _pytest.main import Session
    from _pytest.nodes import Item


template_lookup = TemplateLookup(directories=[os.path.join(os.path.dirname(__file__), "templates")])


def add_options(parser: Parser) -> None:
    """Add pytest-bdd options."""
    group = parser.getgroup("bdd", "Generation")

    group._addoption(
        "--generate-missing",
        action="store_true",
        dest="generate_missing",
        default=False,
        help="Generate missing bdd test code for given feature files and exit.",
    )

    group._addoption(
        "--feature",
        metavar="FILE_OR_DIR",
        action="append",
        dest="features",
        help="Feature file or directory to generate missing code for. Multiple allowed.",
    )


def cmdline_main(config: Config) -> int | None:
    """Check config option to show missing code."""
    if config.option.generate_missing:
        return show_missing_code(config)
    return None  # Make mypy happy


def generate_code(feature: Feature, scenarios: list[Scenario], steps_missing_definition: list[Step]) -> str:
    """Generate test code for the given filenames."""
    grouped_steps = group_steps(steps_missing_definition)
    template = template_lookup.get_template("test.py.mak")
    code = template.render(
        feature=feature,
        scenarios=scenarios,
        steps=grouped_steps,
        make_python_name=make_python_name,
        make_python_docstring=make_python_docstring,
        make_string_literal=make_string_literal,
    )
    return cast(str, code)


def show_missing_code(config: Config) -> int:
    """Wrap pytest session to show missing code."""
    from _pytest.main import wrap_session

    return wrap_session(config, _show_missing_code_main)


def _find_step_fixturedef(fixturemanager: FixtureManager, item: Item, step: Step) -> Sequence[FixtureDef[Any]] | None:
    """Find step fixturedef."""
    with inject_fixturedefs_for_step(step=step, fixturemanager=fixturemanager, node=item):
        bdd_name = get_step_fixture_name(step=step)
        return getfixturedefs(fixturemanager, bdd_name, item)


def parse_feature_files(paths: list[str], **kwargs: Any) -> list[Feature]:
    """Parse feature files of given paths.

    :param paths: `list` of paths (file or dirs)

    :return: `list` of `Feature` objects.
    """
    return get_features(paths, **kwargs)


def group_steps(steps: list[Step]) -> list[Step]:
    """Group steps by type (GIVEN, WHEN, THEN) and remove duplicates."""
    order_mapping = {
        StepType.GIVEN: 0,
        StepType.WHEN: 1,
        StepType.THEN: 2,
    }

    # Sort steps by step_type (order: GIVEN -> WHEN -> THEN)
    steps = sorted(steps, key=lambda s: order_mapping.get(s.step_type, float("inf")))

    seen_steps = set()
    grouped_steps: list[Step] = []

    # Group steps by step_type and sort within the group by step name
    for step_missing_definition in itertools.chain.from_iterable(
        sorted(group, key=lambda s: s.name) for _, group in itertools.groupby(steps, lambda s: s.step_type)
    ):
        # Only add step if its name hasn't been seen before
        if step_missing_definition.name not in seen_steps:
            grouped_steps.append(step_missing_definition)
            seen_steps.add(step_missing_definition.name)

    return grouped_steps


def _show_missing_code_main(config: Config, session: Session) -> None:
    """Preparing fixture duplicates for output."""
    tw = TerminalWriter()
    session.perform_collect()

    fm = session._fixturemanager

    if config.option.features is None:
        tw.line("The --feature parameter is required.", red=True)
        session.exitstatus = 100
        return

    features = parse_feature_files(config.option.features)
    missing_info_found = False

    for feature in features:
        unbound_scenarios = find_unbound_scenarios(feature, session)
        if unbound_scenarios:
            missing_info_found = True

        steps_missing_definition = find_steps_missing_definition(feature, fm, session)
        if steps_missing_definition:
            missing_info_found = True

        print_missing_code(feature, unbound_scenarios, steps_missing_definition)

    if missing_info_found:
        session.exitstatus = 100


def find_unbound_scenarios(feature: Feature, session: Session) -> list[Scenario]:
    # Use a set for faster lookup, using a unique identifier like scenario.name
    bound_scenarios = set()

    for item in session.items:
        # Safeguard if item doesn't have 'obj' attribute (likely a test function)
        test_function = getattr(item, "obj", None)
        if test_function and hasattr(test_function, "__scenario__"):
            scenario = test_function.__scenario__
            # Assuming `Scenario` has a unique attribute like `name` to use for set lookup
            bound_scenarios.add(scenario.name)

    # Return scenarios from the feature that are not in bound_scenarios, using their name for comparison
    return [scenario for scenario in feature.scenarios if scenario.name not in bound_scenarios]


@dataclass
class StepsMissingDefinition:
    step: Step
    step_parent: Scenario | Background


def find_steps_missing_definition(feature: Feature, fixture_manager: FixtureManager, session: Session) -> list[Step]:
    steps_missing_definition = []

    def check_steps(steps):
        for item in session.items:
            for step in steps:
                if not _find_step_fixturedef(fixture_manager, item, step=step):
                    if not any(known.raw_name == step.raw_name for known in steps_missing_definition):
                        steps_missing_definition.append(step)

    for scenario in feature.scenarios:
        check_steps(scenario.all_steps)

    return steps_missing_definition


def print_missing_code(
    feature: Feature, unbound_scenarios: list[Scenario], steps_missing_definition: list[Step]
) -> None:
    """Print missing code with TerminalWriter."""
    tw = TerminalWriter()

    for scenario in unbound_scenarios:
        tw.line()
        tw.line(
            f'Scenario "{scenario.name}" is not bound to any test in the feature "{feature.name}"'
            f" in the file {feature.rel_filename}:{scenario.location.line}",
            red=True,
        )

    if unbound_scenarios:
        tw.sep("-", red=True)

    for step in steps_missing_definition:
        tw.line()
        if step.background:
            tw.line(
                f"Step {step} is not defined in the background of the feature"
                f' "{feature.name}" in the file'
                f" {feature.abs_filename}:{step.location.line}",
                red=True,
            )
        else:
            tw.line(
                f'Step {step} is not defined in the scenario "{step.scenario.name}" in the feature'
                f' "{feature.name}" in the file'
                f" {feature.abs_filename}:{step.location.line}",
                red=True,
            )

    if steps_missing_definition:
        tw.sep("-", red=True)

    tw.line("Please place the code above to the test file(s):")
    tw.line()

    code = generate_code(feature, unbound_scenarios, steps_missing_definition)
    tw.write(code)
