"""ADASYN over sampling algorithm."""

import numpy as np
from sklearn.utils import check_random_state

from aeon.transformations.collection.imbalance._smote import SMOTE

__maintainer__ = ["TonyBagnall"]
__all__ = ["ADASYN"]


class ADASYN(SMOTE):
    """
    Over-sampling using Adaptive Synthetic Sampling (ADASYN).

    Adaptation of imblearn.over_sampling.ADASYN
    original authors:
    #          Guillaume Lemaitre <g.lemaitre58@gmail.com>
    #          Christos Aridas
    # License: MIT

    This transformer extends SMOTE, but it generates different number of
    samples depending on an estimate of the local distribution of the class
    to be oversampled.
    """

    def __init__(self, random_state=None, k_neighbors=5):
        super().__init__(random_state=random_state, k_neighbors=k_neighbors)

    def _transform(self, X, y=None):
        X = np.squeeze(X, axis=1)
        random_state = check_random_state(self.random_state)
        X_resampled = [X.copy()]
        y_resampled = [y.copy()]

        # got the minority class label and the number needs to be generated
        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue
            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = X[target_class_indices]

            self.nn_.fit(X)
            nns = self.nn_.kneighbors(X_class, return_distance=False)[:, 1:]
            # The ratio is computed using a one-vs-rest manner. Using majority
            # in multi-class would lead to slightly different results at the
            # cost of introducing a new parameter.
            n_neighbors = self.nn_.n_neighbors - 1
            ratio_nn = np.sum(y[nns] != class_sample, axis=1) / n_neighbors
            if not np.sum(ratio_nn):
                raise RuntimeError(
                    "Not any neigbours belong to the majority"
                    " class. This case will induce a NaN case"
                    " with a division by zero. ADASYN is not"
                    " suited for this specific dataset."
                    " Use SMOTE instead."
                )
            ratio_nn /= np.sum(ratio_nn)
            n_samples_generate = np.rint(ratio_nn * n_samples).astype(int)
            # rounding may cause new amount for n_samples
            n_samples = np.sum(n_samples_generate)
            if not n_samples:
                raise ValueError(
                    "No samples will be generated with the provided ratio settings."
                )

            # the nearest neighbors need to be fitted only on the current class
            # to find the class NN to generate new samples
            self.nn_.fit(X_class)
            nns = self.nn_.kneighbors(X_class, return_distance=False)[:, 1:]

            enumerated_class_indices = np.arange(len(target_class_indices))
            rows = np.repeat(enumerated_class_indices, n_samples_generate)
            cols = random_state.choice(n_neighbors, size=n_samples)
            diffs = X_class[nns[rows, cols]] - X_class[rows]
            steps = random_state.uniform(size=(n_samples, 1))
            X_new = X_class[rows] + steps * diffs

            X_new = X_new.astype(X.dtype)
            y_new = np.full(n_samples, fill_value=class_sample, dtype=y.dtype)
            X_resampled.append(X_new)
            y_resampled.append(y_new)
        X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)

        X_resampled = X_resampled[:, np.newaxis, :]
        return X_resampled, y_resampled
