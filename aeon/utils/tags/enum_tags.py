"""Apply Enumeration in module."""

__all__ = ["AlgorithmType"]

from enum import Enum


class AlgorithmType(Enum):
    """
    An enumeration of algorithm types and data structures.

    Attributes
    ----------
        Algorithm Types:
        - DISTANCE: Clustering based on distance metrics
        - DEEPLEARNING: Clustering using deep learning techniques
        - FEATURE: Clustering driven by feature extraction

        Data Structure Types:
        - NP_LIST: Numpy list-based data structure
        - NUMPY3D: Three-dimensional Numpy array
    """

    # Algorithm types for clustering strategies
    DISTANCE = "distance"
    DEEPLEARNING = "deeplearning"
    FEATURE = "feature"

    # Data structure types for clustering input
    NP_LIST = "np-list"
    NUMPY3D = "numpy3D"
