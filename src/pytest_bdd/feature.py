"""Feature.

The way of describing the behavior is based on Gherkin language, but a very
limited version. It doesn't support any parameter tables.
If the parametrization is needed to generate more test cases it can be done
on the fixture level of the pytest.
The <variable> syntax can be used here to make a connection between steps and
it will also validate the parameters mentioned in the steps with ones
provided in the pytest parametrization table.

Syntax example:

    Feature: Articles
        Scenario: Publishing the article
            Given I'm an author user
            And I have an article
            When I go to the article page
            And I press the publish button
            Then I should not see the error message
            And the article should be published  # Note: will query the database

:note: The "#" symbol is used for comments.
:note: There are no multiline steps, the description of the step must fit in
one line.
"""

from __future__ import annotations

import glob
import os.path
from typing import Iterator

from .parser import Feature, get_gherkin_document

# Global features dictionary
features: dict[str, Feature] = {}


def get_feature(base_path: str, filename: str, encoding: str = "utf-8") -> Feature:
    """Get a feature by the filename.

    :param str base_path: Base feature directory.
    :param str filename: Filename of the feature file.
    :param str encoding: Feature file encoding.

    :return: `Feature` instance from the parsed feature cache.

    :note: The features are parsed on the execution of the test and
           stored in the global variable cache to improve the performance
           when multiple scenarios are referencing the same file.
    """
    __tracebackhide__ = True
    full_filename = os.path.abspath(os.path.join(base_path, filename))
    rel_filename = os.path.join(os.path.basename(base_path), filename)
    feature = features.get(full_filename)
    if not feature:
        gherkin_document = get_gherkin_document(full_filename, rel_filename, encoding)
        feature = gherkin_document.feature
        features[full_filename] = feature
    return feature


def get_features(paths: list[str], encoding: str = "utf-8") -> list[Feature]:
    """Get features for given paths.

    :param list paths: `list` of paths (file or dirs)
    :param str encoding: encoding to use when reading feature files (default utf-8)

    :return: `list` of `Feature` objects.
    """
    seen_names = set()
    _features = []
    for path in paths:
        if path not in seen_names:
            seen_names.add(path)
            if os.path.isdir(path):
                for feature_file in _find_feature_files(path):
                    base, name = os.path.split(feature_file)
                    feature = get_feature(base, name, encoding)
                    _features.append(feature)
            else:
                base, name = os.path.split(path)
                feature = get_feature(base, name, encoding)
                _features.append(feature)
    _features.sort(key=lambda _feature: _feature.name or _feature.abs_filename)
    return _features


def _find_feature_files(path: str) -> Iterator[str]:
    """Recursively find all `.feature` files in a given directory."""
    return glob.iglob(os.path.join(path, "**", "*.feature"), recursive=True)
