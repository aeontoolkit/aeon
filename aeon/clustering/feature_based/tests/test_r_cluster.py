"""Test For RCluster."""

import numpy as np
import pytest
from sklearn import metrics

from aeon.clustering.feature_based._r_cluster import RClusterer
from aeon.utils.validation._dependencies import _check_estimator_deps

X_ = [
    [
        1.5980065,
        1.5994389,
        1.5705293,
        1.5504735,
        1.507371,
        1.4343414,
        1.3689859,
        1.3052934,
        1.2103053,
        1.1166533,
    ],
    [
        1.7011456,
        1.670645,
        1.6188844,
        1.5468045,
        1.4754685,
        1.3912091,
        1.3058823,
        1.237313,
        1.1534138,
        1.0696899,
    ],
    [
        1.722342,
        1.6953288,
        1.656946,
        1.6063123,
        1.5118241,
        1.4141477,
        1.3136877,
        1.2132338,
        1.1129779,
        1.0150805,
    ],
    [
        1.7262632,
        1.659836,
        1.5731083,
        1.4962643,
        1.4090704,
        1.3324426,
        1.2457422,
        1.1588819,
        1.0733612,
        0.9871649,
    ],
    [
        1.7789757,
        1.7612025,
        1.7030841,
        1.610572,
        1.4920881,
        1.3686543,
        1.2447608,
        1.1209,
        1.0107619,
        0.9001682,
    ],
    [
        1.7996215,
        1.7427012,
        1.6864861,
        1.6326717,
        1.5324101,
        1.4225861,
        1.3113219,
        1.2012383,
        1.0899248,
        0.9785759,
    ],
    [
        1.7490938,
        1.7266423,
        1.6593817,
        1.5595723,
        1.4572895,
        1.355191,
        1.2521086,
        1.1618543,
        1.0623266,
        0.9609945,
    ],
    [
        1.3476895,
        1.2373582,
        1.1288056,
        1.0218658,
        0.9392247,
        0.84710395,
        0.75024295,
        0.65884495,
        0.56604975,
        0.4741342,
    ],
    [
        1.6956215,
        1.633777,
        1.5959885,
        1.5069915,
        1.4142802,
        1.3230939,
        1.2419277,
        1.1857506,
        1.1216865,
        1.0483568,
    ],
    [
        1.722719,
        1.7132868,
        1.6652519,
        1.586769,
        1.4954436,
        1.4038439,
        1.3122748,
        1.2204062,
        1.1295636,
        1.0408053,
    ],
]
Y = ["22", "28", "21", "15", "2", "18", "21", "36", "11", "21"]


@pytest.mark.skipif(
    not _check_estimator_deps(RClusterer, severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_r_cluster():
    """Test implementation of RCluster."""
    X_train = np.array(X_)
    X = np.expand_dims(X_train, axis=1)
    Rcluster = RClusterer(n_clusters=8, n_init=10, random_state=1)
    labels_pred1 = Rcluster.fit_predict(X)
    score = metrics.adjusted_rand_score(labels_true=Y, labels_pred=labels_pred1)
    assert score > 0.36
