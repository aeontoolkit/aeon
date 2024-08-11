"""Base class for deep clustering."""

__maintainer__ = []

from abc import ABC, abstractmethod

from aeon.clustering._k_means import TimeSeriesKMeans
from aeon.clustering._k_medoids import TimeSeriesKMedoids
from aeon.clustering._k_shapes import TimeSeriesKShapes
from aeon.clustering.base import BaseClusterer
from aeon.clustering.dummy import DummyClusterer


class BaseDeepClusterer(BaseClusterer, ABC):
    """Abstract base class for deep learning time series clusterers.

    Parameters
    ----------
    n_clusters : int, default=None
        Number of clusters for the deep learning model.
    clustering_algorithm : str, {'kmeans', 'kshape', 'kmedoids', 'dummy'},
        default="kmeans"
        The clustering algorithm used in the latent space.
        Options include:
        'kmeans' for Kmeans clustering,
        'kshape' for KShape clustering,
        'kmedoids' for KMedoids clustering.
    clustering_params : dict, default=None
        Dictionary containing the parameters of the clustering algorithm chosen.
    batch_size : int, default = 40
        training batch size for the model
    last_file_name : str, default = "last_model"
        The name of the file of the last model, used
        only if save_last_model_to_file is used

    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:multivariate": True,
        "algorithm_type": "deeplearning",
        "non-deterministic": True,
        "cant-pickle": True,
        "python_dependencies": "tensorflow",
    }

    def __init__(
        self,
        n_clusters,
        clustering_algorithm="kmeans",
        clustering_params=None,
        batch_size=32,
        last_file_name="last_file",
    ):
        self.clustering_algorithm = clustering_algorithm
        self.clustering_params = clustering_params
        self.batch_size = batch_size
        self.last_file_name = last_file_name

        self.model_ = None
        self.clusterer = None

        super().__init__(n_clusters)

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
        if self.clustering_params is None:
            clustering_params_ = dict()
        else:
            clustering_params_ = self.clustering_params
            # clustering_params_["n_clusters"] = self.n_clusters
        if self.clustering_algorithm == "dummy":
            self.clusterer = DummyClusterer(
                n_clusters=self.n_clusters, **clustering_params_
            )
        elif self.clustering_algorithm == "kmeans":
            self.clusterer = TimeSeriesKMeans(
                n_clusters=self.n_clusters, **clustering_params_
            )
        elif self.clustering_algorithm == "kshape":
            self.clusterer = TimeSeriesKShapes(
                n_clusters=self.n_clusters, **clustering_params_
            )
        elif self.clustering_algorithm == "kmedoids":
            self.clusterer = TimeSeriesKMedoids(
                n_clusters=self.n_clusters, **clustering_params_
            )
        else:
            raise ValueError(
                f"Invalid input for 'clustering_algorithm': {self.clustering_algorithm}"
            )
        latent_space = self.model_.layers[1].predict(X)
        self.clusterer.fit(X=latent_space)

        return self

    def _predict(self, X):
        # Transpose to conform to Keras input style.
        X = X.transpose(0, 2, 1)
        latent_space = self.model_.layers[1].predict(X)
        clusters = self.clusterer.predict(latent_space)

        return clusters

    def _predict_proba(self, X):
        # Transpose to conform to Keras input style.
        X = X.transpose(0, 2, 1)
        latent_space = self.model_.layers[1].predict(X)
        clusters_proba = self.clusterer.predict_proba(latent_space)

        return clusters_proba

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
