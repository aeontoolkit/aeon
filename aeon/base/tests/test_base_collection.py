"""Tests for BaseCollectionEstimator."""

__maintainer__ = ["TonyBagnall", "MatthewMiddlehurst"]

import numpy as np
import pytest

from aeon.base import BaseCollectionEstimator
from aeon.testing.mock_estimators import MockClassifier
from aeon.testing.testing_data import (
    EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION,
    EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION,
    MISSING_VALUES_CLASSIFICATION,
    UNEQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION,
    UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION,
)
from aeon.utils.data_types import COLLECTIONS_DATA_TYPES
from aeon.utils.validation import get_type


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_get_metadata(data):
    """Test get meta data."""
    # equal length univariate
    if data in EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION:
        meta = BaseCollectionEstimator._get_X_metadata(
            EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[data]["train"][0]
        )
        assert not meta["multivariate"]
        assert not meta["missing_values"]
        assert not meta["unequal_length"]
        assert meta["n_cases"] == 10
        assert meta["n_channels"] == 1
        assert meta["n_timepoints"] == 20

    # unequal length univariate
    if data in UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION:
        meta = BaseCollectionEstimator._get_X_metadata(
            UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[data]["train"][0]
        )
        assert not meta["multivariate"]
        assert not meta["missing_values"]
        assert meta["unequal_length"]
        assert meta["n_cases"] == 10
        assert meta["n_channels"] == 1
        assert meta["n_timepoints"] is None

    # equal length multivariate
    if data in EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION:
        meta = BaseCollectionEstimator._get_X_metadata(
            EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[data]["train"][0]
        )
        assert meta["multivariate"]
        assert not meta["missing_values"]
        assert not meta["unequal_length"]
        assert meta["n_cases"] == 10
        assert meta["n_channels"] == 2
        assert meta["n_timepoints"] == 20

    # unequal length multivariate
    if data in UNEQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION:
        meta = BaseCollectionEstimator._get_X_metadata(
            UNEQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[data]["train"][0]
        )
        assert meta["multivariate"]
        assert not meta["missing_values"]
        assert meta["unequal_length"]
        assert meta["n_cases"] == 10
        assert meta["n_channels"] == 2
        assert meta["n_timepoints"] is None

    # missing data
    if data in MISSING_VALUES_CLASSIFICATION:
        meta = BaseCollectionEstimator._get_X_metadata(
            MISSING_VALUES_CLASSIFICATION[data]["train"][0]
        )
        assert not meta["multivariate"]
        assert meta["missing_values"]
        assert not meta["unequal_length"]
        assert meta["n_cases"] == 10
        assert meta["n_channels"] == 1
        assert meta["n_timepoints"] == 20


def test_check_X():
    """Test if capabilities correctly tested."""
    dummy1 = MockClassifier()
    dummy2 = MockClassifier()
    all_tags = {
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:missing_values": True,
    }
    dummy2.set_tags(**all_tags)

    # univariate equal length
    X = EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION["numpy3D"]["train"][0].copy()
    assert dummy1._check_X(X) and dummy2._check_X(X)

    # univariate missing values
    X[3][0][6] = np.nan
    assert dummy2._check_X(X)
    with pytest.raises(ValueError, match=r"has missing values, but"):
        dummy1._check_X(X)

    # multivariate equal length
    X = EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION["numpy3D"]["train"][0].copy()
    assert dummy2._check_X(X)
    with pytest.raises(ValueError, match=r"has multivariate series, but"):
        dummy1._check_X(X)

    # multivariate missing values
    X[2][1][5] = np.nan
    assert dummy2._check_X(X)
    with pytest.raises(
        ValueError, match=r"has missing values and multivariate series, but"
    ):
        dummy1._check_X(X)

    # univariate equal length
    X = UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION["np-list"]["train"][0]
    assert dummy2._check_X(X)
    with pytest.raises(ValueError, match=r"has unequal length series, but"):
        dummy1._check_X(X)

    # multivariate unequal length
    X = UNEQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION["np-list"]["train"][0]
    assert dummy2._check_X(X)
    with pytest.raises(
        ValueError, match=r"has multivariate series and unequal length series, but"
    ):
        dummy1._check_X(X)

    # different number of channels
    X = [np.random.random(size=(2, 10)), np.random.random(size=(3, 10))]
    with pytest.raises(ValueError):
        dummy2._check_X(X)

    # invalid list type
    X = ["Does", "Not", "Accept", "List", "of", "String"]
    with pytest.raises(TypeError, match=r"passed a list containing <class 'str'>"):
        dummy1._check_X(X)

    # invalid type
    X = MockClassifier()
    with pytest.raises(
        TypeError,
        match="must be of type np.ndarray, pd.DataFrame or list of"
        " np.ndarray/pd.DataFrame",
    ):
        dummy1._check_X(X)


@pytest.mark.parametrize("internal_type", COLLECTIONS_DATA_TYPES)
@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_convert_X(internal_type, data):
    """Test conversion function.

    The conversion functionality of convertCollection is tested in the utils module.
    This test runs a subset of these but also checks classifiers with multiple
    internal types.
    """
    cls = MockClassifier()

    # Equal length should default to numpy3D
    X = EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[data]["train"][0]
    cls.metadata_ = cls._check_X(X)
    X2 = cls._convert_X(X)
    assert get_type(X2) == cls.get_tag("X_inner_type")

    # should all convert to numpy3D
    cls.set_tags(**{"X_inner_type": "numpy3D"})
    X2 = cls._convert_X(X)
    assert get_type(X2) == "numpy3D"

    # Same as above but as list
    cls.set_tags(**{"X_inner_type": ["numpy3D"]})
    X2 = cls._convert_X(X)
    assert get_type(X2) == "numpy3D"

    # Set cls inner type to just internal_type, should convert to internal_type
    cls.set_tags(**{"X_inner_type": internal_type})
    X2 = cls._convert_X(X)
    assert get_type(X2) == internal_type

    # Should not convert, as datatype is already present
    cls.set_tags(**{"X_inner_type": ["numpy3D", data]})
    X2 = cls._convert_X(X)
    assert get_type(X2) == data

    # Should always convert to numpy3D unless data is already internal_type, as it is
    # the highest priority type
    cls.set_tags(**{"X_inner_type": ["numpy3D", internal_type]})
    X2 = cls._convert_X(X)
    assert get_type(X2) == "numpy3D" if data != internal_type else internal_type

    if data in UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION.keys():
        if internal_type in UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION.keys():
            X = UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[data]["train"][0]

            # Should stay as internal_type
            cls.set_tags(**{"capability:unequal_length": True})
            cls.set_tags(**{"X_inner_type": ["np-list", data]})
            X2 = cls._convert_X(X)
            assert get_type(X2) == data

            # np-list is the highest priority type for unequal length
            cls.set_tags(**{"capability:unequal_length": True})
            cls.set_tags(**{"X_inner_type": [internal_type, "np-list"]})
            X2 = cls._convert_X(X)
            assert get_type(X2) == "np-list" if data != internal_type else internal_type


@pytest.mark.parametrize("data", COLLECTIONS_DATA_TYPES)
def test_preprocess_collection(data):
    """Test the functionality for preprocessing fit."""
    data = EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[data]["train"][0]
    data2 = np.random.random(size=(11, 1, 30))
    cls = MockClassifier()

    X = cls._preprocess_collection(data)
    assert cls._n_jobs == 1
    assert len(cls.metadata_) == 6
    assert get_type(X) == "numpy3D"

    tags = {"capability:multithreading": True}
    cls = MockClassifier()
    cls.set_tags(**tags)
    assert cls._n_jobs == 1

    # Test two calls do not overwrite metadata (predict should not reset fit meta)
    cls = MockClassifier()
    cls._preprocess_collection(data)
    meta = cls.metadata_
    cls._preprocess_collection(data2)
    assert meta == cls.metadata_


def test_convert_np_list():
    """Test np-list of 1D numpy converted to 2D."""
    x1 = np.random.random(size=(1, 10))
    x2 = np.random.rand(20)
    x3 = np.random.rand(30)
    np_list = [x1, x2, x3]
    np2 = BaseCollectionEstimator._reshape_np_list(np_list)
    assert len(np2) == len(np_list)
    assert np2[0].shape == (1, 10)
    assert np2[1].shape == (1, 20)
    assert np2[2].shape == (1, 30)
    dummy1 = MockClassifier()
    x1 = np.random.random(size=(1, 10))
    x2 = np.random.rand(10)
    x3 = np.random.rand(10)
    np_list = [x1, x2, x3]
    np3 = dummy1._preprocess_collection(np_list)
    assert isinstance(np3, np.ndarray)
    assert np3.shape == (3, 1, 10)
