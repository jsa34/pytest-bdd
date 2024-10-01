"""${ feature.name or feature.rel_filename } feature tests."""

from pytest_bdd import (
    given,
    scenario,
    then,
    when,
)

# Define all scenarios in the feature
% for scenario in sorted(feature.scenarios, key=lambda scenario: scenario.name):
@scenario('${feature.rel_filename}', ${ make_string_literal(scenario.name)})
def test_${ make_python_name(scenario.name)}():
    ${make_python_docstring(scenario.name)}

% endfor

# Background steps (if they exist)
% if feature.background and feature.background.steps:
% for step in feature.background.steps:
@${step.step_type.value}(${ make_string_literal(step.name)})
def _():
    ${make_python_docstring(step.name)}
    raise NotImplementedError
% endfor
% endif

# Scenario steps
% for step in steps:
@${step.step_type.value}(${ make_string_literal(step.name)})
def _():
    ${make_python_docstring(step.name)}
    raise NotImplementedError
% endfor
