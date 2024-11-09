.. _anomaly_detection_ref:

Anomaly Detection
=================

Time Series Anomaly Detection aims at discovering regions of a time series that in
some way are not representative of the underlying generative process.
The :mod:`aeon.anomaly_detection` module contains algorithms and tools
for time series anomaly detection. The detectors have different capabilities that can
be grouped into the following categories, where ``m`` is the number of time points and
``d`` is the number of channels for a time series:

Input data format (one of the following):
    Univariate series (default):
        Example: :class:`~aeon.anomaly_detection.MERLIN`.

        - np.ndarray, shape ``(m,)``, ``(m, 1)`` or ``(1, m)`` depending on axis.
        - pd.DataFrame, shape ``(m, 1)`` or ``(1, m)`` depending on axis.
        - pd.Series, shape ``(m,)``.
    Multivariate series:
        Example: :class:`~aeon.anomaly_detection.KMeansAD`.

        - np.ndarray array, shape ``(m, d)`` or ``(d, m)`` depending on axis.
        - pd.DataFrame ``(m, d)`` or ``(d, m)`` depending on axis.

Output data format (one of the following):
    Anomaly scores (default):
        np.ndarray, shape ``(m,)`` of type float. For each point of the input time
        series, the anomaly score is a float value indicating the degree of
        anomalousness. The higher the score, the more anomalous the point. The
        detectors return raw anomaly scores that are not normalized.
        Example: :class:`~aeon.anomaly_detection.PyODAdapter`.
    Binary classification:
        np.ndarray, shape ``(m,)`` of type bool or int. For each point of the input
        time series, the output is a boolean or integer value indicating whether the
        point is anomalous (``True``/``1``) or not (``False``/``0``).
        Example: :class:`~aeon.anomaly_detection.STRAY`.

Detector learning types:
    Unsupervised (default):
        Unsupervised detectors do not require any training data and can directly be
        used on the target time series. You would usually call the ``fit_predict``
        method on these detectors.
        Example: :class:`~aeon.anomaly_detection.DWT_MLEAD`.
    Semi-supervised:
        Semi-supervised detectors require a training step on a time series without
        anomalies (normal behaving time series). The target value ``y`` would
        consist of only zeros. You would usually first call the ``fit`` method on the
        training time series and then the ``predict`` method on your target time series.
        Example: :class:`~aeon.anomaly_detection.KMeansAD`.
    Supervised:
        Supervised detectors require a training step on a time series with known
        anomalies (anomalies should be present and must be annotated). The detector
        implements the ``fit`` method, and the target value ``y`` consists of zeros
        and ones; ones indicating points of an anomaly. You would usually first call
        the ``fit`` method on the training data and then the ``predict`` method on your
        target time series.

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
    MERLIN
    PyODAdapter
    STRAY
    STOMP
    LSTM_AD
