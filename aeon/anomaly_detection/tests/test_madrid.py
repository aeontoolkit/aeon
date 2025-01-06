"""Tests for the MADRID class."""

__mentainer__ = ["acquayefrank"]

import pytest
import numpy as np

from aeon.anomaly_detection import MADRID

TEST_DATA = np.array(
    [
        20.2185297756533,
        -105.097578872683,
        -250.631170598857,
        -367.640267546770,
        -440.795694514124,
        -484.844607033234,
        -429.612414332529,
        -387.926902521381,
        -302.775578817544,
        -285.154718026416,
        -333.603843534684,
        -408.339060212994,
        -509.726903588524,
        -675.148130332975,
        -776.996414586199,
        -858.126218663558,
        -858.869234821577,
        -775.901606259732,
        -633.141058269163,
        -464.351649283751,
        -276.298272192354,
        -147.919321236800,
        -540.529156042038,
        -18.7507118180904,
        -98.9509797716845,
        -247.904747009230,
        -372.736021887472,
        -480.420694767257,
        -550.866264803162,
        -577.223297568274,
        -544.107408904777,
        -475.049370723107,
        -412.372315469016,
        -368.477985574912,
        -369.124494651837,
        -450.728585418889,
        -580.746017562320,
        -684.940897962263,
        -819.404710010413,
        -890.917095056103,
        -873.084035728888,
        -800.774105902329,
        -658.926357190422,
        -443.574951554361,
        -285.015994846434,
        -134.647712722159,
        -31.7542427746267,
        -15.5309673691411,
        -77.7214785008164,
        -201.111639169828,
    ]
)

toy_data = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 3])


def test_min_length_predict():
    with pytest.raises(
        ValueError,
        match=f"Series length of X {TEST_DATA.shape[0]} is less than min_length 500",
    ):
        mad = MADRID(500, 1000)
        mad.predict(TEST_DATA)


def test_contains_constant_regions():
    mad = MADRID(3, 5)
    assert mad._contains_constant_regions(toy_data, 3)
    assert not mad._contains_constant_regions(TEST_DATA, 3)


def test_contains_constant_regions_in_predict():
    with pytest.raises(ValueError):
        mad = MADRID(3, 5)
        mad.predict(toy_data)
    mad = MADRID(3, 5)
    mad.predict(TEST_DATA)


def test_madrid():
    """Test MADRID output."""
    mad = MADRID(5, 15)
    pred = mad.predict(TEST_DATA)