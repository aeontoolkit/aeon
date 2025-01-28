"""Test ADASYN oversampler ported from imblearn."""

import numpy as np
import pytest

from aeon.testing.data_generation import make_example_3d_numpy
from aeon.transformations.collection.imbalance import ADASYN
from aeon.utils.validation._dependencies import _check_soft_dependencies


def test_adasyn():
    """Test the ADASYN class.

    This function creates a 3D numpy array, applies
    ADASYN using the ADASYN class, and asserts that the
    transformed data has a balanced number of samples.
    ADASYN is a variant of SMOTE that generates synthetic samples,
    but it focuses on generating samples near the decision boundary.
    Therefore, sometimes, it may generate more or less samples than SMOTE,
    which is why we only check if the number of samples is nearly balanced.
    """
    n_samples = 100  # Total number of labels
    majority_num = 90  # number of majority class
    minority_num = n_samples - majority_num  # number of minority class

    X = np.random.rand(n_samples, 1, 10)
    y = np.array([0] * majority_num + [1] * minority_num)

    transformer = ADASYN()
    transformer.fit(X, y)
    res_X, res_y = transformer.transform(X, y)
    _, res_count = np.unique(res_y, return_counts=True)

    assert np.abs(len(res_X) - 2 * majority_num) < minority_num
    assert np.abs(len(res_y) - 2 * majority_num) < minority_num
    assert res_count[0] == majority_num
    assert np.abs(res_count[0] - res_count[1]) < minority_num


@pytest.mark.skipif(
    not _check_soft_dependencies(
        "imbalanced-learn",
        package_import_alias={"imbalanced-learn": "imblearn"},
        severity="none",
    ),
    reason="skip test if required soft dependency imbalanced-learn not available",
)
def test_equivalence_imbalance():
    """Test ported ADASYN code produces the same as imblearn version."""
    from imblearn.over_sampling import ADASYN as imbADASYN

    X, y = make_example_3d_numpy(n_cases=20, n_channels=1)
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    X = X.squeeze()
    s1 = imbADASYN(random_state=49)
    X2, y2 = s1.fit_resample(X, y)
    s2 = ADASYN(random_state=49)
    X3, y3 = s2.fit_transform(X, y)
    X3 = X3.squeeze()
    assert np.array_equal(y2, y3)
    assert np.allclose(X2, X3, atol=1e-4)
