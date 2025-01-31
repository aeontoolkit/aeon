"""Catch22 Clusterer.

Pipeline clusterer using the Catch22 transformer and an estimator.
"""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["Catch22Clusterer"]

import numpy as np
from sklearn.cluster import KMeans

from aeon.base._base import _clone_estimator
from aeon.clustering import BaseClusterer
from aeon.transformations.collection.feature_based import Catch22
from aeon.utils.tags.enum_tags import AlgorithmType


class Catch22Clusterer(BaseClusterer):
    """
    Canonical Time-series Characteristics (catch22) clusterer.

    This clusterer simply transforms the input data using the Catch22 [1]_
    transformer and builds a provided estimator using the transformed data.

    Parameters
    ----------
    features : int/str or List of int/str, default="all"
        The Catch22 features to extract by feature index, feature name as a str or as a
        list of names or indices for multiple features. If "all", all features are
        extracted. Valid features are as follows:
            ["DN_HistogramMode_5", "DN_HistogramMode_10",
            "SB_BinaryStats_diff_longstretch0", "DN_OutlierInclude_p_001_mdrmd",
            "DN_OutlierInclude_n_001_mdrmd", "CO_f1ecac", "CO_FirstMin_ac",
            "SP_Summaries_welch_rect_area_5_1", "SP_Summaries_welch_rect_centroid",
            "FC_LocalSimple_mean3_stderr", "CO_trev_1_num", "CO_HistogramAMI_even_2_5",
            "IN_AutoMutualInfoStats_40_gaussian_fmmi", "MD_hrv_classic_pnn40",
            "SB_BinaryStats_mean_longstretch1", "SB_MotifThree_quantile_hh",
            "FC_LocalSimple_mean1_tauresrat", "CO_Embed2_Dist_tau_d_expfit_meandiff",
            "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",
            "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
            "SB_TransitionMatrix_3ac_sumdiagcov", "PD_PeriodicityWang_th0_01"]
    catch24 : bool, default=True
        Extract the mean and standard deviation as well as the 22 Catch22 features if
        true. If a List of specific features to extract is provided, "Mean" and/or
        "StandardDeviation" must be added to the List to extract these features.
    outlier_norm : bool, optional, default=False
        Normalise each series during the two outlier Catch22 features, which can take a
        while to process for large values.
    replace_nans : bool, default=True
        Replace NaN or inf values from the Catch22 transform with 0.
    use_pycatch22 : bool, default=False
        Wraps the C based pycatch22 implementation for aeon.
        (https://github.com/DynamicsAndNeuralSystems/pycatch22). This requires the
        ``pycatch22`` package to be installed if True.
    estimator : sklearn clusterer, default=None
        An sklearn estimator to be built using the transformed data.
        Defaults to sklearn KMeans().
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    parallel_backend : str, ParallelBackendBase instance or None, default=None
        Specify the parallelisation backend implementation in joblib for Catch22,
        if None a 'prefer' value of "threads" is used by default.
        Valid options are "loky", "multiprocessing", "threading" or a custom backend.
        See the joblib Parallel documentation for more details.

    See Also
    --------
    Catch22
        Catch22 transformer in aeon/transformations/collection.

    References
    ----------
    .. [1] Lubba, Carl H., et al. "catch22: Canonical time-series characteristics."
        Data Mining and Knowledge Discovery 33.6 (2019): 1821-1852.
        https://link.springer.com/article/10.1007/s10618-019-00647-x

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.cluster import KMeans
    >>> from aeon.clustering.feature_based import Catch22Clusterer
    >>> X = np.random.random(size=(10,2,20))
    >>> clst = Catch22Clusterer(estimator=KMeans(n_clusters=2))
    >>> clst.fit(X)
    Catch22Clusterer(...)
    >>> preds = clst.predict(X)
    """

    _tags = {
        "X_inner_type": [
            AlgorithmType.NP_LIST.value,
            AlgorithmType.NUMPY3D.value,
        ],
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:multithreading": True,
        "algorithm_type": AlgorithmType.FEATURE.value,
    }

    def __init__(
        self,
        features="all",
        catch24=True,
        outlier_norm=False,
        replace_nans=True,
        use_pycatch22=False,
        estimator=None,
        random_state=None,
        n_jobs=1,
        parallel_backend=None,
    ):
        self.features = features
        self.catch24 = catch24
        self.outlier_norm = outlier_norm
        self.replace_nans = replace_nans
        self.use_pycatch22 = use_pycatch22
        self.estimator = estimator
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend

        super().__init__()

    def _fit(self, X, y=None):
        """Fit a pipeline on cases X.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_cases, n_channels, n_timepoints]
            The training data.
        y : array-like, shape = [n_cases]
            Ignored. The class labels.

        Returns
        -------
        self :
            Reference to self.

        Returns
        -------
        self :
            Reference to self.
        """
        self._transformer = Catch22(
            features=self.features,
            catch24=self.catch24,
            outlier_norm=self.outlier_norm,
            replace_nans=self.replace_nans,
            use_pycatch22=self.use_pycatch22,
            n_jobs=self._n_jobs,
            parallel_backend=self.parallel_backend,
        )

        self._estimator = _clone_estimator(
            (KMeans() if self.estimator is None else self.estimator),
            self.random_state,
        )

        m = getattr(self._estimator, "n_jobs", None)
        if m is not None:
            self._estimator.n_jobs = self._n_jobs

        X_t = self._transformer.fit_transform(X, y)
        self._estimator.fit(X_t, y)
        self.labels_ = self._estimator.labels_

        return self

    def _predict(self, X) -> np.ndarray:
        """Predict class values of n instances in X.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_cases, n_channels, n_timepoints]
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_cases]
            Predicted class labels.
        """
        return self._estimator.predict(self._transformer.transform(X))

    def _predict_proba(self, X) -> np.ndarray:
        """Predict class values of n instances in X.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_cases, n_channels, n_timepoints]
            The data to make predictions for.

        Returns
        -------
        y : 2D array of shape [n_cases, n_classes] - predicted class probabilities
            1st dimension indices correspond to instance indices in X
            2nd dimension indices correspond to possible labels (integers)
            (i, j)-th entry is predictive probability that i-th instance is of class j
        """
        m = getattr(self._estimator, "predict_proba", None)
        if callable(m):
            return self._estimator.predict_proba(self._transformer.transform(X))
        else:
            return super()._predict_proba(X)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        return {
            "features": (
                "Mean",
                "DN_HistogramMode_5",
                "SB_BinaryStats_mean_longstretch1",
            ),
        }
