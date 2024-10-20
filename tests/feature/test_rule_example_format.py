import textwrap


def test_rule_example_format(pytester):
    pytester.makefile(
        ".feature",
        rule_example=textwrap.dedent(
            """\
            Feature: Calculator

              In order to perform basic arithmetic operations
              As a user
              I want to use a calculator

              Background:
                Given I have got my calculator ready

              Scenario: I check the calculator powers on
                Given I press the power button
                Then the screen turns on

              Rule: Addition
                In order to add two numbers
                As a user, I want the calculator to give me the sum.

                Background:
                  Given I check the add button is working

                Example: Adding two positive numbers
                  Given the first number is 3
                  And the second number is 5
                  When I press add
                  Then the result should be 8

                Example: Adding a positive number and a negative number
                  Given the first number is 7
                  And the second number is -2
                  When I press add
                  Then the result should be 5

              Rule: Subtraction
                In order to subtract one number from another
                As a user, I want the calculator to give me the difference.

                Example: Subtracting a smaller number from a larger number
                  Given the first number is 10
                  And the second number is 4
                  When I press subtract
                  Then the result should be 6

                Example: Subtracting a larger number from a smaller number
                  Given the first number is 3
                  And the second number is 7
                  When I press subtract
                  Then the result should be -4
            """
        ),
    )

    pytester.makepyfile(
        textwrap.dedent(
            """\
        import pytest
        from pytest_bdd import given, when, then, parsers, scenarios


        scenarios("rule_example.feature")


        @given("I have got my calculator ready")
        def _():
            print("Calculator ready!")

        @given("I check the add button is working")
        def _():
            print("Add button check.")

        @given("I press the power button")
        def _():
            pass

        @then("the screen turns on")
        def _():
            pass

        @given(parsers.parse("the first number is {first_number}"), target_fixture="first_number")
        def _(first_number):
            return int(first_number)

        @given(parsers.parse("the second number is {second_number}"), target_fixture="second_number")
        def _(second_number):
            return int(second_number)

        @when("I press add", target_fixture="result")
        def _(first_number, second_number):
            return first_number + second_number

        @when("I press subtract", target_fixture="result")
        def _(first_number, second_number):
            return first_number - second_number

        @then(parsers.parse("the result should be {expected_result}"))
        def _(result, expected_result):
            assert result == int(expected_result)
        """
        )
    )
    result = pytester.runpytest("-s")
    result.assert_outcomes(passed=4)
    # Get all the lines from stdout
    assert result.stdout.lines.count("Calculator ready!")
