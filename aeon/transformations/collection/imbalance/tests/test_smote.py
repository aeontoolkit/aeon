"""Test function for SMOTE."""

import numpy as np
import pytest

from aeon.testing.data_generation import make_example_3d_numpy
from aeon.transformations.collection.imbalance import SMOTE
from aeon.utils.validation._dependencies import _check_soft_dependencies


def test_smote():
    """Test the SMOTE class.

    This function creates a 3D numpy array, applies
    SMOTE using the SMOTE class, and asserts that the
    transformed data has a balanced number of samples.
    """
    n_samples = 100  # Total number of labels
    majority_num = 90  # number of majority class
    minority_num = n_samples - majority_num  # number of minority class

    X = np.random.rand(n_samples, 1, 10)
    y = np.array([0] * majority_num + [1] * minority_num)

    transformer = SMOTE()
    transformer.fit(X, y)
    res_X, res_y = transformer.transform(X, y)
    _, res_count = np.unique(res_y, return_counts=True)

    assert len(res_X) == 2 * majority_num
    assert len(res_y) == 2 * majority_num
    assert res_count[0] == majority_num
    assert res_count[1] == majority_num


@pytest.mark.skipif(
    not _check_soft_dependencies(
        "imbalanced-learn",
        package_import_alias={"imbalanced-learn": "imblearn"},
        severity="none",
    ),
    reason="skip test if required soft dependency imbalanced-learn not available",
)
def test_equivalence_imbalance():
    """Test ported SMOTE code produces the same as imblearn version."""
    from imblearn.over_sampling import SMOTE as imbSMOTE

    X, y = make_example_3d_numpy(n_cases=20, n_channels=1)
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    X = X.squeeze()
    s1 = imbSMOTE(random_state=49)
    X2, y2 = s1.fit_resample(X, y)
    s2 = SMOTE(random_state=49)
    X3, y3 = s2.fit_transform(X, y)
    X3 = X3.squeeze()
    assert np.array_equal(y2, y3)
    assert np.allclose(X2, X3, atol=1e-4)
