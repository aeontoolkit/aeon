.. _clustering_ref:

Clustering
==========

The :mod:`aeon.clustering` module contains algorithms for time series clustering.

All clusterers in `aeon` can be listed using the `aeon.registry.all_estimators`
utility, using `estimator_types="clusterer"`, optionally filtered by tags.
Valid tags can be listed using `aeon.registry.all_tags`.

Clustering Algorithms
---------------------

.. currentmodule:: aeon.clustering

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    KASBA
    TimeSeriesKMeans
    TimeSeriesKMedoids
    TimeSeriesKShape
    TimeSeriesKernelKMeans
    TimeSeriesCLARA
    TimeSeriesCLARANS
    ElasticSOM
    KSpectralCentroid
    RClusterer

Deep learning
-------------

.. currentmodule:: aeon.clustering.deep_learning

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    AEFCNClusterer
    AEResNetClusterer
    AEDCNNClusterer
    AEDRNNClusterer
    AEAttentionBiGRUClusterer
    AEBiGRUClusterer

Feature-based
-------------

.. currentmodule:: aeon.clustering.feature_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Catch22Clusterer
    SummaryClusterer
    TSFreshClusterer

Compose
-------

.. currentmodule:: aeon.clustering.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ClustererPipeline

Averaging
---------

.. currentmodule:: aeon.clustering.averaging

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    kasba_average
    elastic_barycenter_average
    mean_average
    petitjean_barycenter_average
    subgradient_barycenter_average
    shift_invariant_average

Dummy
-----

.. currentmodule:: aeon.clustering

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DummyClusterer

Base
----

.. currentmodule:: aeon.clustering.base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseClusterer

.. currentmodule:: aeon.clustering.deep_learning

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseDeepClusterer
