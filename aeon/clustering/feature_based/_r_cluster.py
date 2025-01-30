import multiprocessing

import numpy as np
from numba import get_num_threads, set_num_threads
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from aeon.clustering.base import BaseClusterer
from aeon.transformations.collection.convolution_based._minirocket import (
    _fit_biases,
    _fit_dilations,
    _quantiles,
    _static_transform_uni,
)


class RClusterer(BaseClusterer):
    """Implementation of Time Series R Cluster.

    Adapted from the implementation used in [1]_

    Parameters
    ----------
    num_kernels : int , default = 84
         The number of convolutional kernels used to transform the input time series
         These kernels are fixed and pre-defined (not random) and are
         optimized for computational speed and
         feature diversity
    max_dilations_per_kernel : int , default = 32
         The maximum number of dilation rates applied to each kernel
         Dilations control the spacing of the kernel's receptive field
         over the time series,capturing patterns at varying scales
    num_cluster : int , default = 8
         The number of clusters used
    num_features : int, default=500
        Number of features need for fit_dilations method.
    n_init : int, default=10
        Number of times the R-Cluster algorithm will be run with different
        centroid seeds. The final result will be the best output of n_init
        consecutive runs in terms of inertia.
    random_state : int, Random state or None, default=None
    n_jobs : int, default=1
         The number of jobs to run in parallel for `transform`. ``-1``
         means using all
         processors.

    Notes
    -----
    Adapted from the implementation from source code
    https://github.com/jorgemarcoes/R-Clustering/blob/main/R_Clustering_on_UCR_Archive.ipynb

    References
    ----------
    .. [1]  Time series clustering with random convolutional kernels
    https://link.springer.com/article/10.1007/s10618-024-01018-x
    """

    _tags = {
        "capability:multivariate": False,
        "capability:multithreading": True,
        "capability:unequal_length": False,
        "capability:missing_values": False,
    }

    def __init__(
        self,
        n_kernels=84,
        max_dilations_per_kernel=32,
        n_clusters=8,
        n_init=10,
        num_features=500,
        random_state=None,
        n_jobs=1,
    ):
        self.num_features = num_features
        self.n_init = n_init
        self.n_jobs = n_jobs
        self.n_kernels = n_kernels
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.indices = np.array(
            (
                1,
                3,
                6,
                1,
                2,
                7,
                1,
                2,
                3,
                0,
                2,
                3,
                1,
                4,
                5,
                0,
                1,
                3,
                3,
                5,
                6,
                0,
                1,
                2,
                2,
                5,
                8,
                1,
                3,
                7,
                0,
                1,
                8,
                4,
                6,
                7,
                0,
                1,
                4,
                3,
                4,
                6,
                0,
                4,
                5,
                2,
                6,
                7,
                5,
                6,
                7,
                0,
                1,
                6,
                4,
                5,
                7,
                4,
                7,
                8,
                1,
                6,
                8,
                0,
                2,
                6,
                5,
                6,
                8,
                2,
                5,
                7,
                0,
                1,
                7,
                0,
                7,
                8,
                0,
                3,
                5,
                0,
                3,
                7,
                2,
                3,
                8,
                2,
                3,
                4,
                1,
                4,
                6,
                3,
                4,
                5,
                0,
                3,
                8,
                4,
                5,
                8,
                0,
                4,
                6,
                1,
                4,
                8,
                6,
                7,
                8,
                4,
                6,
                8,
                0,
                3,
                4,
                1,
                3,
                4,
                1,
                5,
                7,
                1,
                4,
                7,
                1,
                2,
                8,
                0,
                6,
                7,
                1,
                6,
                7,
                1,
                3,
                5,
                0,
                1,
                5,
                0,
                4,
                8,
                4,
                5,
                6,
                0,
                2,
                5,
                3,
                5,
                7,
                0,
                2,
                4,
                2,
                6,
                8,
                2,
                3,
                7,
                2,
                5,
                6,
                2,
                4,
                8,
                0,
                2,
                7,
                3,
                6,
                8,
                2,
                3,
                6,
                3,
                7,
                8,
                0,
                5,
                8,
                1,
                2,
                6,
                2,
                3,
                5,
                1,
                5,
                8,
                3,
                6,
                7,
                3,
                4,
                7,
                0,
                4,
                7,
                3,
                5,
                8,
                2,
                4,
                5,
                1,
                2,
                5,
                2,
                7,
                8,
                2,
                4,
                6,
                0,
                5,
                6,
                3,
                4,
                8,
                0,
                6,
                8,
                2,
                4,
                7,
                0,
                2,
                8,
                0,
                3,
                6,
                5,
                7,
                8,
                1,
                5,
                6,
                1,
                2,
                4,
                0,
                5,
                7,
                1,
                3,
                8,
                1,
                7,
                8,
            ),
            dtype=np.int32,
        ).reshape(84, 3)
        super().__init__()

    def _get_parameterised_data(self, X):
        random_state = np.random.RandomState(self.random_state)
        X = X.astype(np.float32)

        _, n_channels, n_timepoints = X.shape

        dilations, num_features_per_dilation = _fit_dilations(
            n_timepoints, self.num_features, self.max_dilations_per_kernel
        )

        num_features_per_kernel = np.sum(num_features_per_dilation)

        quantiles = _quantiles(self.n_kernels * num_features_per_kernel)

        quantiles = random_state.permutation(quantiles)

        n_dilations = len(dilations)
        n_combinations = self.n_kernels * n_dilations
        max_n_channels = min(n_channels, 9)
        max_exponent = np.log2(max_n_channels + 1)
        n_channels_per_combination = (
            2 ** np.random.uniform(0, max_exponent, n_combinations)
        ).astype(np.int32)
        channel_indices = np.zeros(n_channels_per_combination.sum(), dtype=np.int32)
        n_channels_start = 0
        for combination_index in range(n_combinations):
            n_channels_this_combination = n_channels_per_combination[combination_index]
            n_channels_end = n_channels_start + n_channels_this_combination
            channel_indices[n_channels_start:n_channels_end] = np.random.choice(
                n_channels, n_channels_this_combination, replace=False
            )
            n_channels_start = n_channels_end
        biases = _fit_biases(
            X,
            n_channels_per_combination,
            channel_indices,
            dilations,
            num_features_per_dilation,
            quantiles,
            self.indices,
            self.random_state,
        )

        return (
            np.array([_], dtype=np.int32),
            np.array([_], dtype=np.int32),
            dilations,
            num_features_per_dilation,
            biases,
        )

    def check_params(self, X):
        X = X.astype(np.float32)
        if self.n_jobs < 1 or self.n_jobs > multiprocessing.cpu_count():
            n_jobs = multiprocessing.cpu_count()
        else:
            n_jobs = self.n_jobs
        set_num_threads(n_jobs)
        return X

    def _get_transformed_data(self, X, parameters):
        prev_threads = get_num_threads()
        X = self.check_params(X)
        X = X.squeeze(1)
        X_ = _static_transform_uni(X, parameters, self.indices)
        set_num_threads(prev_threads)
        return X_

    def _fit(self, X, y=None):
        parameters = self._get_parameterised_data(X)

        transformed_data = self._get_transformed_data(X=X, parameters=parameters)

        self.scaler = StandardScaler()
        X_std = self.scaler.fit_transform(transformed_data)

        pca = PCA().fit(X_std)
        optimal_dimensions = np.argmax(pca.explained_variance_ratio_ < 0.01)

        optimal_dimensions = max(
            1, min(optimal_dimensions, X_std.shape[0], X_std.shape[1])
        )

        self.pca = PCA(n_components=optimal_dimensions, random_state=self.random_state)
        transformed_data_pca = self.pca.fit_transform(X_std)
        self.estimator = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
        )
        self.estimator.fit(transformed_data_pca)
        self.labels_ = self.estimator.labels_

    def _predict(self, X, y=None) -> np.ndarray:
        parameters = self._get_parameterised_data(X)

        transformed_data = self._get_transformed_data(X=X, parameters=parameters)

        X_std = self.scaler.fit_transform(transformed_data)

        transformed_data_pca = self.pca.fit_transform(X_std)
        return self.estimator.predict(transformed_data_pca)

    def _fit_predict(self, X, y=None) -> np.ndarray:
        parameters = self._get_parameterised_data(X)
        transformed_data = self._get_transformed_data(X=X, parameters=parameters)
        self.scaler = StandardScaler()
        X_std = self.scaler.fit_transform(transformed_data)

        pca = PCA().fit(X_std)
        optimal_dimensions = np.argmax(pca.explained_variance_ratio_ < 0.01)

        optimal_dimensions = max(
            1, min(optimal_dimensions, X_std.shape[0], X_std.shape[1])
        )
        self.pca = PCA(n_components=optimal_dimensions, random_state=self.random_state)
        transformed_data_pca = self.pca.fit_transform(X_std)
        self.estimator = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
        )
        Y = self.estimator.fit_predict(transformed_data_pca)
        self.labels_ = self.estimator.labels_
        return Y

    @classmethod
    def _get_test_params(cls, parameter_set="default") -> dict:
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        return {
            "n_clusters": 2,
        }
