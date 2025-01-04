"""Test functions for data writing."""

import os
import tempfile

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from aeon.datasets import load_from_arff_file, load_from_ts_file, write_to_ts_file
from aeon.datasets._data_writers import _write_header, write_to_arff_file
from aeon.testing.data_generation import (
    make_example_3d_numpy,
    make_example_3d_numpy_list,
)
from aeon.testing.testing_config import PR_TESTING


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
@pytest.mark.parametrize("regression", [True, False])
@pytest.mark.parametrize("problem_name", ["first", "second.ts"])
def test_write_to_ts_file_equal_length(regression, problem_name):
    """Test function to write an equal length classification and regegression dataset.

    creates an equal length problem, writes locally, reloads, then compares data. It
    then deletes the files.
    """
    X, y = make_example_3d_numpy(regression_target=regression)
    with tempfile.TemporaryDirectory() as tmp:
        write_to_ts_file(
            X=X, path=tmp, y=y, problem_name=problem_name, regression=regression
        )
        load_path = os.path.join(tmp, problem_name)
        newX, newy = load_from_ts_file(full_file_path_and_name=load_path)
        assert isinstance(newX, np.ndarray)
        assert X.shape == newX.shape
        assert X[0][0][0] == newX[0][0][0]
        if not regression:
            y = y.astype(str)
            np.testing.assert_array_equal(y, newy)
        else:
            np.testing.assert_array_almost_equal(y, newy)


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
@pytest.mark.parametrize("problem_name", ["first", "second.ts"])
def test_write_regression_to_ts_file_equal_length(problem_name):
    """Test function to write a regression dataset.

    Loads equal and unequal length problems into both data frames and numpy arrays,
    writes locally, reloads, then compares all class labels. It then delete the files.
    """
    X, y = make_example_3d_numpy(regression_target=True)
    with tempfile.TemporaryDirectory() as tmp:
        write_to_ts_file(X=X, path=tmp, y=y, problem_name=problem_name + ".ts")
        load_path = os.path.join(tmp, problem_name)
        newX, newy = load_from_ts_file(full_file_path_and_name=load_path)
        assert isinstance(newX, np.ndarray)
        assert X.shape == newX.shape
        assert X[0][0][0] == newX[0][0][0]
        y = y.astype(str)
        assert np.array_equal(y, newy)
        write_to_ts_file(X=X, path=tmp, y=None, problem_name=problem_name + ".ts")


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of intermittent fail for read/write",
)
@pytest.mark.parametrize("problem_name", ["Testy", "Testy2.ts"])
def test_write_to_ts_file_unequal_length(problem_name):
    """Test function to write a dataset.

    Loads equal and unequal length problems into both data frames and numpy arrays,
    writes locally, reloads, then compares all class labels. It then delete the files.
    """
    X, y = make_example_3d_numpy_list()
    with tempfile.TemporaryDirectory() as tmp:
        write_to_ts_file(X=X, path=tmp, y=y, problem_name=problem_name)
        load_path = os.path.join(tmp, problem_name)
        newX, newy = load_from_ts_file(full_file_path_and_name=load_path)
        assert isinstance(newX, list)
        assert len(X) == len(newX)
        assert X[0][0][0] == newX[0][0][0]
        y = y.astype(str)
        assert np.array_equal(y, newy)


def test_write_data_to_ts_file_invalid():
    """Test function to check the handling of invalid inputs by write_to_ts_file."""
    with pytest.raises(TypeError, match="Wrong input data type"):
        write_to_ts_file("A string", "path")
    X, _ = make_example_3d_numpy(n_cases=6, n_timepoints=10, n_channels=1)
    y = np.ndarray([0, 1, 1, 0, 1])
    with pytest.raises(
        IndexError,
        match="The number of cases in X does not match the number of values in y",
    ):
        write_to_ts_file(X, "temp", y=y)


def test_write_to_arff_wrong_inputs():
    """Tests whether error thrown if wrong input."""
    # load an example dataset
    with tempfile.TemporaryDirectory() as tmp:
        X = "A string"
        y = [1, 2, 3, 4]
        with pytest.raises(TypeError, match="Wrong input data type"):
            write_to_arff_file(X, y, tmp)
        X2, y2 = make_example_3d_numpy(n_cases=5, n_channels=2)
        with pytest.raises(ValueError, match="must be a 3D array with shape"):
            write_to_arff_file(X2, y2, tmp)


def test_write_header():
    """Test _write_header."""
    with tempfile.TemporaryDirectory() as tmp:
        problem_name = "header.csv"
        with pytest.raises(
            ValueError, match="Cannot have class_labels true for a regression problem"
        ):
            _write_header(tmp, problem_name, class_labels=True, regression=True)
    _write_header(
        tmp,
        problem_name,
        extension=".csv",
        comment="Hello",
        regression=True,
    )


def test_write_to_arff_file():
    """Test function to check writing into an ARFF file and loading from it."""
    X, y = make_example_3d_numpy()

    with tempfile.TemporaryDirectory() as tmp:
        write_to_arff_file(X, y, tmp, problem_name="Test_arff", header="Description")

        load_path = os.path.join(tmp, "Test_arff.arff")
        X_new, y_new = load_from_arff_file(full_file_path_and_name=load_path)

        assert isinstance(X_new, np.ndarray)
        assert X.shape == X_new.shape
        assert_array_equal(X, X_new)
        assert_array_equal(y.astype(str), y_new)
