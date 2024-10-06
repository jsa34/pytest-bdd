from pathlib import Path

from src.pytest_bdd.parser import get_gherkin_document


def test_parser():
    test_dir = Path(__file__).parent
    feature_file = test_dir / "test.feature"
    feature_file_path = str(feature_file.resolve())

    gherkin_doc = get_gherkin_document(feature_file_path, str(feature_file))

    # Check all comments found
    assert len(gherkin_doc.comments) == 10

    # Check feature name
    assert gherkin_doc.feature.name == "User login"

    # Check count of scenarios
    assert len(gherkin_doc.feature.scenarios) == 9

    # Check count of rules
    assert len(gherkin_doc.feature.rules) == 2

    # Check for background presence and steps
    assert gherkin_doc.feature.background
    assert len(gherkin_doc.feature.background_steps) == 1

    # Check count of children in the feature (scenarios + rules)
    assert len(gherkin_doc.feature.children) == 12

    # Check first scenario's steps count
    assert len(gherkin_doc.feature.scenarios[0].steps) == 4

    # Check first scenario's steps including background
    assert len(gherkin_doc.feature.scenarios[0].all_steps) == 5

    # Check the name of the first scenario
    assert gherkin_doc.feature.scenarios[0].name == "Successful login with valid credentials"

    # Check tags on specific scenarios
    assert gherkin_doc.feature.scenarios[5].tag_names == {"login", "critical"}
    assert gherkin_doc.feature.scenarios[6].tag_names == {"smoke"}

    # Check scenario outline
    scenario_outline = gherkin_doc.feature.scenarios[1]
    assert scenario_outline.name == "Unsuccessful login with invalid credentials"
    assert len(scenario_outline.examples) == 1
    assert len(scenario_outline.examples[0].table_body) == 2  # Two rows in the example table
    assert [cell.value for cell in scenario_outline.examples[0].table_body[0].cells] == [
        "invalidUser",
        "wrongPass",
        "Invalid username or password",
    ]

    # Check steps of the scenario outline
    assert len(scenario_outline.steps) == 4

    # # Check rule names and examples
    # rule1 = gherkin_doc.feature.rules[0]
    # assert rule1.name == "a sale cannot happen if there is no stock"
    # assert len(rule1.examples) == 1
    # assert rule1.examples[0].name == "No chocolates left"
    # assert len(rule1.examples[0].steps) == 4

    # rule2 = gherkin_doc.feature.rules[1]
    # assert rule2.name == "A sale cannot happen if the customer does not have enough money"
    # assert len(rule2.children) == 2  # Two examples (happy and unhappy paths)
    # assert rule2.children[0].scenario.name == "Not enough money"
    # assert rule2.children[1].scenario.name == "Enough money"

    # Check if the doc string in the error message scenario is parsed correctly
    doc_string_scenario = gherkin_doc.feature.scenarios[8]
    assert doc_string_scenario.name == "Check login error message with detailed explanation"
    assert doc_string_scenario.steps[-1].doc_string.content == (
        "Your login attempt was unsuccessful.\n"
        "Please check your username and password and try again.\n"
        "If the problem persists, contact support."
    )

    # Check step contents and text in various scenarios
    assert gherkin_doc.feature.scenarios[2].steps[0].text == "the user enters an empty username"
    assert gherkin_doc.feature.scenarios[3].steps[1].text == "the user enters an empty password"
    assert gherkin_doc.feature.scenarios[4].steps[0].text == "the user enters \"admin' OR '1'='1\" as username"

    # # Check scenario using data tables
    # data_table_scenario = gherkin_doc.feature.scenarios[7]
    # assert data_table_scenario.name == "Login with multiple sets of credentials"
    # assert len(data_table_scenario.steps[0].data_table.table_body) == 3  # Three rows of users registered
    # assert len(data_table_scenario.steps[1].data_table.table_body) == 2  # Two sets of login attempts
    # assert len(data_table_scenario.steps[2].data_table.table_body) == 2  # Two expected outcomes

    # Check login button disabled scenario
    login_button_disabled_scenario = gherkin_doc.feature.scenarios[5]
    assert login_button_disabled_scenario.name == "Login button disabled for empty fields"
    assert len(login_button_disabled_scenario.steps) == 2
