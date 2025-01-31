"""Base class for deep clustering."""

__maintainer__ = []

from abc import abstractmethod
from copy import deepcopy

from aeon.base._base import _clone_estimator
from aeon.clustering._k_means import TimeSeriesKMeans
from aeon.clustering.base import BaseClusterer
from aeon.utils.tags.enum_tags import AlgorithmType


class BaseDeepClusterer(BaseClusterer):
    """Abstract base class for deep learning time series clusterers.

    Parameters
    ----------
    estimator : aeon clusterer, default=None
        An aeon estimator to be built using the transformed data.
        Defaults to aeon TimeSeriesKMeans() with euclidean distance
        and mean averaging method and n_clusters set to 2.
    batch_size : int, default = 40
        training batch size for the model
    last_file_name : str, default = "last_model"
        The name of the file of the last model, used
        only if save_last_model_to_file is used in
        child class.

    """

    _tags = {
        "X_inner_type": AlgorithmType.NUMPY3D.value,
        "capability:multivariate": True,
        "algorithm_type": AlgorithmType.DEEPLEARNING.value,
        "non_deterministic": True,
        "cant_pickle": True,
        "python_dependencies": "tensorflow",
    }

    @abstractmethod
    def __init__(
        self,
        estimator=None,
        batch_size=32,
        last_file_name="last_model",
    ):
        self.estimator = estimator
        self.batch_size = batch_size
        self.last_file_name = last_file_name

        self.model_ = None

        super().__init__()

    @abstractmethod
    def build_model(self, input_shape):
        """Construct a compiled, un-trained, keras model that is ready for training.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer

        Returns
        -------
        A compiled Keras Model
        """
        ...

    def summary(self):
        """
        Summary function to return the losses/metrics for model fit.

        Returns
        -------
        history : dict or None,
            Dictionary containing model's train/validation losses and metrics

        """
        return self.history.history if self.history is not None else None

    def save_last_model_to_file(self, file_path="./"):
        """Save the last epoch of the trained deep learning model.

        Parameters
        ----------
        file_path : str, default = "./"
            The directory where the model will be saved

        Returns
        -------
        None
        """
        self.model_.save(file_path + self.last_file_name + ".keras")

    def _fit_clustering(self, X):
        """Train the clustering algorithm in the latent space.

        Parameters
        ----------
        X : np.ndarray, shape=(n_cases, n_timepoints, n_channels)
            The input time series.
        """
        self._estimator = (
            TimeSeriesKMeans(
                n_clusters=2, distance="euclidean", averaging_method="mean"
            )
            if self.estimator is None
            else _clone_estimator(self.estimator)
        )

        latent_space = self.model_.layers[1].predict(X)
        self._estimator.fit(X=latent_space)
        if hasattr(self._estimator, "labels_"):
            self.labels_ = self._estimator.labels_
        else:
            self.labels_ = self._estimator.predict(X=latent_space)

        return self

    def _predict(self, X):
        # Transpose to conform to Keras input style.
        X = X.transpose(0, 2, 1)
        latent_space = self.model_.layers[1].predict(X)
        clusters = self._estimator.predict(latent_space)

        return clusters

    def _predict_proba(self, X):
        # Transpose to conform to Keras input style.
        X = X.transpose(0, 2, 1)
        latent_space = self.model_.layers[1].predict(X)
        clusters_proba = self._estimator.predict_proba(latent_space)

        return clusters_proba

    def load_model(self, model_path, estimator):
        """Load a pre-trained keras model instead of fitting.

        When calling this function, all functionalities can be used
        such as predict, predict_proba etc. with the loaded model.

        Parameters
        ----------
        model_path : str (path including model name and extension)
            The directory where the model will be saved including the model
            name with a ".keras" extension.
            Example: model_path="path/to/file/best_model.keras"
        estimator : estimator : aeon clusterer
            Pre-trained clusterer needed for loading model.

        Returns
        -------
        None
        """
        import tensorflow as tf

        self.model_ = tf.keras.models.load_model(model_path)
        self.is_fitted = True

        # use deep copy to preserve fit state
        self._estimator = deepcopy(estimator)

    def _get_model_checkpoint_callback(self, callbacks, file_path, file_name):
        import tensorflow as tf

        model_checkpoint_ = tf.keras.callbacks.ModelCheckpoint(
            filepath=file_path + file_name + ".keras",
            monitor="loss",
            save_best_only=True,
        )

        if isinstance(callbacks, list):
            return callbacks + [model_checkpoint_]
        else:
            return [callbacks] + [model_checkpoint_]
