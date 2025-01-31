from typing import Optional

import numpy as np

from aeon.anomaly_detection.base import BaseAnomalyDetector
from aeon.utils.windowing import reverse_windowing


class IDK(BaseAnomalyDetector):
    """IDK² and s-IDK² anomaly detector.

    The Isolation Distributional Kernel (IDK) is a data-dependent kernel for efficient
    anomaly detection, improving accuracy without explicit learning. Its extension,
    IDK², simplifies group anomaly detection, outperforming traditional methods in
    speed and effectiveness.

    Parameters
    ----------
    psi1 : int
         The number of samples randomly selected in each iteration to construct the
         feature map matrix during the first stage. This parameter determines the
         granularity of the first-stage feature representation. Higher values allow
         the model to capture more detailed data characteristics but
         increase computational complexity.
    psi2 : int
         The number of samples randomly selected in each iteration to construct
         the feature map matrix during the second stage. This parameter
         determines the granularity of the second-stage feature representation.
         Higher values allow the model to capture more detailed
         data characteristics but increase computational complexity.
    width : int
         The size of the sliding or fixed-width window used for anomaly detection.
         For fixed-width processing, this defines the length of each segment analyzed.
         In sliding window mode, it specifies the length of the window moving
         across the data.
         Smaller values lead to more localized anomaly detection, while
         larger values capture
         broader trends.
    t : int, default=100
         The number of iterations (time steps) for random sampling to
         construct the feature
         maps. Each iteration generates a set of random samples, which contribute to the
         feature map matrix. Larger values improve the robustness of the feature maps
         but increase the runtime.
    sliding : bool, default=False
         Determines whether a sliding window approach is used for anomaly detection.
         If True, the model computes scores for overlapping windows across the
         time series,
         providing more detailed anomaly scores at each step. If False, the
         model processes
         the data in fixed-width segments, offering faster computation at the
         cost of granularity.
    random_state : int, Random state or None, default=None

    Notes
    -----
    This implementation is inspired by the Isolation Distributional Kernel (IDK)
    approach as detailed in [1]_.
    The code is adapted from the open-source repository [2]_.

    References
    ----------
    [1] Isolation Distributional Kernel: A New Tool for Kernel-Based Anomaly Detection.
         DOI: https://dl.acm.org/doi/10.1145/3394486.3403062

    [2] GitHub Repository:
         IsolationKernel/Codes: IDK Implementation for Time Series Data
         URL: https://github.com/IsolationKernel/Codes/tree/main/IDK/TS
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": False,
        "capability:missing_values": False,
    }

    def __init__(
        self,
        psi1: int = 8,
        psi2: int = 2,
        width: int = 1,
        t: int = 100,
        sliding: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        self.psi1 = psi1
        self.psi2 = psi2
        self.width = width
        self.t = t
        self.sliding = sliding
        self.random_state = random_state
        super().__init__(axis=0)

    def _compute_point_to_sample(self, X, sample_indices):
        sample = X[sample_indices, :]
        tem1 = np.dot(np.square(X), np.ones(sample.T.shape))
        tem2 = np.dot(np.ones(X.shape), np.square(sample.T))
        point2sample = tem1 + tem2 - 2 * np.dot(X, sample.T)

        sample2sample = point2sample[sample_indices, :]
        row, col = np.diag_indices_from(sample2sample)
        sample2sample[row, col] = np.nan

        radius_list = np.nanmin(sample2sample, axis=1)
        min_dist_point2sample = np.argmin(point2sample, axis=1)

        return point2sample, radius_list, min_dist_point2sample

    def _ik_inne_fm(self, X, psi, t, rng):
        onepoint_matrix = np.zeros((X.shape[0], t * psi), dtype=int)
        for time in range(t):
            sample_indices = rng.choice(len(X), size=psi, replace=False)
            point2sample, radius_list, min_dist_point2sample = (
                self._compute_point_to_sample(X, sample_indices)
            )

            min_point2sample_index = np.argmin(point2sample, axis=1)
            min_dist_point2sample = min_point2sample_index + time * psi
            point2sample_value = point2sample[
                range(len(onepoint_matrix)), min_point2sample_index
            ]
            ind = point2sample_value < radius_list[min_point2sample_index]
            onepoint_matrix[ind, min_dist_point2sample[ind]] = 1

        return onepoint_matrix

    def _idk(self, X, psi, t, rng):
        point_fm_list = self._ik_inne_fm(X=X, psi=psi, t=t, rng=rng)
        feature_mean_map = np.mean(point_fm_list, axis=0)
        return np.dot(point_fm_list, feature_mean_map) / t

    def _idk_t(self, X, rng):
        window_num = int(np.ceil(X.shape[0] / self.width))
        featuremap_count = np.zeros((window_num, self.t * self.psi1))
        onepoint_matrix = np.full((X.shape[0], self.t), -1)

        for time in range(self.t):
            sample_indices = rng.choice(X.shape[0], size=self.psi1, replace=False)
            point2sample, radius_list, min_dist_point2sample = (
                self._compute_point_to_sample(X, sample_indices)
            )

            for i in range(X.shape[0]):
                if (
                    point2sample[i][min_dist_point2sample[i]]
                    < radius_list[min_dist_point2sample[i]]
                ):
                    onepoint_matrix[i][time] = (
                        min_dist_point2sample[i] + time * self.psi1
                    )
                    featuremap_count[(int)(i / self.width)][
                        onepoint_matrix[i][time]
                    ] += 1

        for i in range(window_num):
            featuremap_count[i] /= self.width
        isextra = X.shape[0] - (int)(X.shape[0] / self.width) * self.width
        if isextra > 0:
            featuremap_count[-1] /= isextra
        if isextra > 0:
            featuremap_count = np.delete(
                featuremap_count, [featuremap_count.shape[0] - 1], axis=0
            )

        return self._idk(featuremap_count, psi=self.psi2, t=self.t, rng=rng)

    def _idk_square_sliding(self, X, rng):
        point_fm_list = self._ik_inne_fm(X=X, psi=self.psi1, t=self.t, rng=rng)
        point_fm_list = np.insert(point_fm_list, 0, 0, axis=0)
        cumsum = np.cumsum(point_fm_list, axis=0)

        subsequence_fm_list = (cumsum[self.width :] - cumsum[: -self.width]) / float(
            self.width
        )

        return self._idk(X=subsequence_fm_list, psi=self.psi2, t=self.t, rng=rng)

    def _predict(self, X):
        rng = np.random.default_rng(self.random_state)
        if self.sliding:
            sliding_output = self._idk_square_sliding(X, rng)
            reversed_output = reverse_windowing(
                y=sliding_output,
                window_size=self.width,
                stride=1,
                reduction=np.nanmean,
            )
            return reversed_output
        else:
            return self._idk_t(X, rng)

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
        params : dict
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        return {
            "psi1": 8,
            "psi2": 2,
            "width": 1,
        }
