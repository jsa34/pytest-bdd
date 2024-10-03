from pathlib import Path

from src.pytest_bdd.parser import get_gherkin_document


def test_parser():
    test_dir = Path(__file__).parent
    feature_file = test_dir / "test.feature"
    feature_file_path = str(feature_file.resolve())

    gherkin_doc = get_gherkin_document(feature_file_path, str(feature_file))

    # Check all comments found
    assert len(gherkin_doc.comments) == 10

    # Check elements of the feature parsed
    assert gherkin_doc.feature.name == "User login"
    assert len(gherkin_doc.feature.scenarios) == 9
    assert len(gherkin_doc.feature.rules) == 2
    assert gherkin_doc.feature.background
    assert len(gherkin_doc.feature.children) == 12

    # Check count background steps for feature
    assert len(gherkin_doc.feature.background_steps) == 1

    # Check count of first scenario steps
    assert len(gherkin_doc.feature.scenarios[0].steps) == 4
    # Check count of first scenario steps including background step
    assert len(gherkin_doc.feature.scenarios[0].all_steps) == 5
