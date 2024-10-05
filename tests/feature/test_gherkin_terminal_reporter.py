from __future__ import annotations

import textwrap

import pytest

FEATURE = """\
Feature: Gherkin terminal output feature
    Scenario: Scenario example 1
        Given there is a bar
        When the bar is accessed
        Then world explodes
"""

TEST = """\
from pytest_bdd import given, when, then, scenario


@given('there is a bar')
def _():
    return 'bar'

@when('the bar is accessed')
def _():
    pass


@then('world explodes')
def _():
    pass


@scenario('test.feature', 'Scenario example 1')
def test_scenario_1():
    pass

"""


def test_default_output_should_be_the_same_as_regular_terminal_reporter(pytester):
    pytester.makefile(".feature", test=FEATURE)
    pytester.makepyfile(TEST)
    regular = pytester.runpytest()
    gherkin = pytester.runpytest("--gherkin-terminal-reporter")
    regular.assert_outcomes(passed=1, failed=0)
    gherkin.assert_outcomes(passed=1, failed=0)

    def parse_lines(lines: list[str]) -> list[str]:
        return [line for line in lines if not line.startswith("===")]

    assert all(l1 == l2 for l1, l2 in zip(parse_lines(regular.stdout.lines), parse_lines(gherkin.stdout.lines)))


def test_verbose_mode_should_display_feature_and_scenario_names_instead_of_test_names_in_a_single_line(pytester):
    pytester.makefile(".feature", test=FEATURE)
    pytester.makepyfile(TEST)
    result = pytester.runpytest("--gherkin-terminal-reporter", "-v")
    result.assert_outcomes(passed=1, failed=0)
    result.stdout.fnmatch_lines("Feature: Gherkin terminal output feature")
    result.stdout.fnmatch_lines("*Scenario: Scenario example 1 PASSED")


def test_verbose_mode_should_preserve_displaying_regular_tests_as_usual(pytester):
    pytester.makepyfile(
        textwrap.dedent(
            """\
        def test_1():
            pass
        """
        )
    )
    regular = pytester.runpytest()
    gherkin = pytester.runpytest("--gherkin-terminal-reporter", "-v")
    regular.assert_outcomes(passed=1, failed=0)
    gherkin.assert_outcomes(passed=1, failed=0)

    regular.stdout.re_match_lines(
        r"test_verbose_mode_should_preserve_displaying_regular_tests_as_usual\.py \.\s+\[100%\]"
    )
    gherkin.stdout.re_match_lines(
        r"test_verbose_mode_should_preserve_displaying_regular_tests_as_usual\.py::test_1 PASSED\s+\[100%\]"
    )


def test_double_verbose_mode_should_display_full_scenario_description(pytester):
    pytester.makefile(".feature", test=FEATURE)
    pytester.makepyfile(TEST)
    result = pytester.runpytest("--gherkin-terminal-reporter", "-vv")
    result.assert_outcomes(passed=1, failed=0)

    result.stdout.fnmatch_lines("*Scenario: Scenario example 1")
    result.stdout.fnmatch_lines("*Given there is a bar")
    result.stdout.fnmatch_lines("*When the bar is accessed")
    result.stdout.fnmatch_lines("*Then world explodes")
    result.stdout.fnmatch_lines("*PASSED")


@pytest.mark.parametrize("verbosity", ["", "-v", "-vv"])
def test_error_message_for_missing_steps(pytester, verbosity):
    pytester.makefile(".feature", test=FEATURE)
    pytester.makepyfile(
        textwrap.dedent(
            """\
        from pytest_bdd import scenarios

        scenarios('.')
        """
        )
    )
    result = pytester.runpytest("--gherkin-terminal-reporter", verbosity)
    result.assert_outcomes(passed=0, failed=1)
    result.stdout.fnmatch_lines(
        """*StepDefinitionNotFoundError: Step definition is not found: Given "there is a bar". """
        """Line 3 in scenario "Scenario example 1"*"""
    )


@pytest.mark.parametrize("verbosity", ["", "-v", "-vv"])
def test_error_message_should_be_displayed(pytester, verbosity):
    pytester.makefile(".feature", test=FEATURE)
    pytester.makepyfile(
        textwrap.dedent(
            """\
        from pytest_bdd import given, when, then, scenario


        @given('there is a bar')
        def _():
            return 'bar'

        @when('the bar is accessed')
        def _():
            pass


        @then('world explodes')
        def _():
            raise Exception("BIGBADABOOM")


        @scenario('test.feature', 'Scenario example 1')
        def test_scenario_1():
            pass
        """
        )
    )
    result = pytester.runpytest("--gherkin-terminal-reporter", verbosity)
    result.assert_outcomes(passed=0, failed=1)
    result.stdout.fnmatch_lines("E       Exception: BIGBADABOOM")
    result.stdout.fnmatch_lines("test_error_message_should_be_displayed.py:15: Exception")


def test_local_variables_should_be_displayed_when_showlocals_option_is_used(pytester):
    pytester.makefile(".feature", test=FEATURE)
    pytester.makepyfile(
        textwrap.dedent(
            """\
        from pytest_bdd import given, when, then, scenario


        @given('there is a bar')
        def _():
            return 'bar'

        @when('the bar is accessed')
        def _():
            pass


        @then('world explodes')
        def _():
            local_var = "MULTIPASS"
            raise Exception("BIGBADABOOM")


        @scenario('test.feature', 'Scenario example 1')
        def test_scenario_1():
            pass
        """
        )
    )
    result = pytester.runpytest("--gherkin-terminal-reporter", "--showlocals")
    result.assert_outcomes(passed=0, failed=1)
    result.stdout.fnmatch_lines("""request*=*<FixtureRequest for *""")
    result.stdout.fnmatch_lines("""local_var*=*MULTIPASS*""")


def test_step_parameters_should_be_replaced_by_their_values(pytester):
    example = {"start": 10, "eat": 3, "left": 7}
    pytester.makefile(
        ".feature",
        test=textwrap.dedent(
            """\
        Feature: Gherkin terminal output feature
            Scenario Outline: Scenario example 2
                Given there are <start> cucumbers
                When I eat <eat> cucumbers
                Then I should have <left> cucumbers

            Examples:
            | start | eat | left |
            |{start}|{eat}|{left}|
        """.format(
                **example
            )
        ),
    )
    pytester.makepyfile(
        test_gherkin=textwrap.dedent(
            """\
            from pytest_bdd import given, when, scenario, then, parsers

            @given(parsers.parse('there are {start} cucumbers'), target_fixture="start_cucumbers")
            def _(start):
                return start

            @when(parsers.parse('I eat {eat} cucumbers'))
            def _(start_cucumbers, eat):
                pass

            @then(parsers.parse('I should have {left} cucumbers'))
            def _(start_cucumbers, left):
                pass

            @scenario('test.feature', 'Scenario example 2')
            def test_scenario_2():
                pass
        """
        )
    )

    result = pytester.runpytest("--gherkin-terminal-reporter", "-vv")
    result.assert_outcomes(passed=1, failed=0)
    result.stdout.fnmatch_lines("*Scenario: Scenario example 2")
    result.stdout.fnmatch_lines("*Given there are {start} cucumbers".format(**example))
    result.stdout.fnmatch_lines("*When I eat {eat} cucumbers".format(**example))
    result.stdout.fnmatch_lines("*Then I should have {left} cucumbers".format(**example))
    result.stdout.fnmatch_lines("*PASSED")


import textwrap


def test_non_english_feature_reported_correctly_in_terminal(pytester):
    exemple = {"nombre_1": 2, "nombre_2": 3, "résultat": 5}

    # Create the feature file with the specified lines
    feature_content = textwrap.dedent(
        """\
        # language: fr

        Fonctionnalité: Additionner deux nombres
          Plan du scénario: Additionner deux nombres positifs
            Étant donné que j'introduis le premier nombre à ajouter: <Nombre 1>
            Et j'introduis le deuxième nombre à ajouter: <Nombre 2>
            Quand j'additionne les nombres
            Alors le résultat doit être <Résultat>

            Exemples:
              | Nombre 1   | Nombre 2   | Résultat   |
              | {nombre_1} | {nombre_2} | {résultat} |
        """.format(
            **exemple
        )
    )

    pytester.makefile(".feature", test=feature_content)
    pytester.makepyfile(
        test_gherkin=textwrap.dedent(
            """\
            from pytest_bdd import given, when, scenarios, then, parsers

            scenarios('test.feature')

            @given(parsers.parse("j'introduis le premier nombre à ajouter: {nombre_1:d}"), target_fixture="nombre_1")
            def _(nombre_1):
                return nombre_1

            @given(parsers.parse("j'introduis le deuxième nombre à ajouter: {nombre_2:d}"), target_fixture="nombre_2")
            def _(nombre_2):
                return nombre_2

            @when("j'additionne les nombres", target_fixture="total")
            def _(nombre_1, nombre_2):
                return nombre_1 + nombre_2

            @then(parsers.parse("le résultat doit être {résultat:d}"))
            def _(résultat, total):
                assert total == résultat
        """
        )
    )

    result = pytester.runpytest("--gherkin-terminal-reporter", "-vv")

    # Check that the test passed
    result.assert_outcomes(passed=1, failed=0)

    # Get the output from the pytest run
    output_lines = result.stdout

    # Assert that each line from the feature file is in the output
    for line in feature_content.splitlines():
        if "# language:" in line:
            continue
        if line:
            assert output_lines.fnmatch_lines(line.strip())
