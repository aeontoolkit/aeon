"""Test functions in yield estimator checks."""

import numpy as np

from aeon.testing.estimator_checking._yield_estimator_checks import _compare_outputs


def test_compare_outputs():
    """Test compare outputs for array, divct and tuple."""
    a1 = np.random.random((5, 1, 10))
    a2 = np.random.random((5, 10))
    assert _compare_outputs(a1, a1)
    assert not _compare_outputs(a1, a2)
    d1 = {"a": a1, "b": a2}
    d2 = {"a": a1, "b": a1}
    assert _compare_outputs(d1, d1)
    assert not _compare_outputs(d1, d2)
    t1 = (a1, a2)
    t2 = (a1, a1)
    assert _compare_outputs(t1, t1)
    assert not _compare_outputs(t1, t2)
