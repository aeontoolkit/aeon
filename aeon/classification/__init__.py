"""Classifier Base."""

__all__ = [
    "BaseClassifier",
    "DummyClassifier",
]

from aeon.classification._dummy import DummyClassifier
from aeon.classification.base import BaseClassifier
