.. _anomaly_detection_ref:

Anomaly Detection
=================

The :mod:`aeon.anomaly_detection` module contains algorithms and composition tools for time series classification.

All detectors in `aeon`  can be listed using the `aeon.utils.discovery.all_estimators` utility,
using ``estimator_types="anomaly-detector"``, optionally filtered by tags.
Valid tags can be listed by calling the function `aeon.utils.discovery.all_tags_for_estimator`.

Each detector in this module specifies its supported input data format, output data
format, and learning type as an overview table in its documentation. Some detectors
support multiple learning types.

Detectors
---------

.. currentmodule:: aeon.anomaly_detection

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CBLOF
    COPOD
    DWT_MLEAD
    IsolationForest
    KMeansAD
    LeftSTAMPi
    LOF
    MERLIN
    OneClassSVM
    PyODAdapter
    ROCKAD
    STOMP
    STRAY
    IDK

Base
----

.. currentmodule:: aeon.anomaly_detection.base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseAnomalyDetector
