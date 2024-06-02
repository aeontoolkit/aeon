"""Tests for DummySimilaritySearch."""

__maintainer__ = []


import numpy as np
import pytest
from numpy.testing import assert_array_equal

from aeon.similarity_search.query_search.dummy import DummyQuerySearch

DATATYPES = ["int64", "float64"]


@pytest.mark.parametrize("dtype", DATATYPES)
def test_DummyQuerySearch(dtype):
    """Test the functionality of DummyQuerySearch."""
    X = np.asarray(
        [[[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 4, 4, 5, 6, 5, 4]]], dtype=dtype
    )
    q = np.asarray([[3, 4, 5]], dtype=dtype)

    search = DummyQuerySearch()
    search.fit(X)
    idx = search.predict(q)
    assert_array_equal(idx, (0, 2))

    search = DummyQuerySearch(normalize=True)
    search.fit(X)
    q = np.asarray([[8, 8, 10]], dtype=dtype)
    idx = search.predict(q)
    assert_array_equal(idx, (1, 2))

    search = DummyQuerySearch(normalize=True)
    search.fit(X)
    idx = search.predict(q, X_index=(1, 2))
    assert_array_equal(idx, (1, 0))


@pytest.mark.parametrize("dtype", DATATYPES)
def test_DummyQuerySearch_unequal_length(dtype):
    """Test the functionality of DummyQuerySearch on unequal length data."""
    X = [
        np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=dtype),
        np.array([[1, 2, 4, 4, 5, 6, 5]], dtype=dtype),
    ]

    q = np.asarray([[3, 4, 5]], dtype=dtype)

    search = DummyQuerySearch()
    search.fit(X)
    idx = search.predict(q)
    assert_array_equal(idx, (0, 2))

    search = DummyQuerySearch(normalize=True)
    search.fit(X)
    q = np.asarray([[8, 8, 10]], dtype=dtype)
    idx = search.predict(q)
    assert_array_equal(idx, (1, 2))

    search = DummyQuerySearch(normalize=True)
    search.fit(X)
    idx = search.predict(q, X_index=(1, 2))
    assert_array_equal(idx, (1, 0))
