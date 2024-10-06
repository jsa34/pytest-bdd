from pathlib import Path

from src.pytest_bdd.parser import get_gherkin_document


def test_portuguese_language():
    test_dir = Path(__file__).parent
    feature_file = test_dir / "portuguese.feature"
    feature_file_path = str(feature_file.resolve())

    # Call the function to parse the Gherkin document
    gherkin_doc = get_gherkin_document(feature_file_path)

    assert gherkin_doc == ""


def test_french_language():
    test_dir = Path(__file__).parent
    feature_file = test_dir / "french.feature"
    feature_file_path = str(feature_file.resolve())

    # Call the function to parse the Gherkin document
    gherkin_doc = get_gherkin_document(feature_file_path)

    assert gherkin_doc == ""
