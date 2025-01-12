"""Test cases for the CBOSS functionality."""

import numpy as np
import pytest

from aeon.classification.dictionary_based._cboss import ContractableBOSS


def test_invalid_window_size():
    """
    Test that a ValueError is raised if min_window is greater than max_window + 1.

    This is to prevent invalid configurations of window sizes in ContractableBOSS.
    """
    cboss = ContractableBOSS(min_window=20, max_win_len_prop=0.1)
    X = np.random.randn(30, 10)
    y = np.random.randint(0, 2, size=30)

    # Check if the ValueError is raised when calling fit
    with pytest.raises(ValueError, match="min_window .* is bigger than max_window"):
        cboss.fit(X, y)
