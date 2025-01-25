"""Utility to determine type identifier of estimator, based on base class."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["get_identifier"]

from inspect import isclass

from aeon.base import BaseAeonEstimator
from aeon.utils.base._register import BASE_CLASS_REGISTER


def get_identifier(estimator):
    """Determine identifier string of an estimator.

    Parameters
    ----------
    estimator : class or object BaseAeonEstimator
        The estimator to determine the identifier for.

    Returns
    -------
    identifier : str
         aeon identifier string from BASE_CLASS_REGISTER for the estimator.

    Raises
    ------
    TypeError
        If no identifier can be determined for estimator
    """
    if isinstance(estimator, BaseAeonEstimator):
        identifiers = [
            key
            for key, value in BASE_CLASS_REGISTER.items()
            if isinstance(estimator, value)
        ]
    elif isclass(estimator) and issubclass(estimator, BaseAeonEstimator):
        identifiers = [
            key
            for key, value in BASE_CLASS_REGISTER.items()
            if isinstance(estimator, value) or issubclass(estimator, value)
        ]
    else:
        raise TypeError(
            "Estimator must be an instance or subclass of BaseAeonEstimator."
        )

    if len(identifiers) == 0:
        raise TypeError("Error, no identifiers could be determined for estimator")

    if len(identifiers) > 1 and "estimator" in identifiers:
        identifiers.remove("estimator")
    if len(identifiers) > 1 and "series-estimator" in identifiers:
        identifiers.remove("series-estimator")
    if len(identifiers) > 1 and "collection-estimator" in identifiers:
        identifiers.remove("collection-estimator")
    if len(identifiers) > 1 and "transformer" in identifiers:
        identifiers.remove("transformer")
    if len(identifiers) > 1 and "similarity-search" in identifiers:
        identifiers.remove("similarity-search")

    if len(identifiers) > 1:
        TypeError(
            f"Error, multiple identifiers could be determined for obj: {identifiers}"
        )

    return identifiers[0]
