"""
Abstract base class for the Keras neural network anomaly detectors.

The reason for this class between BaseAnomalyDetector and deep_learning
anomaly detectors is because we can generalise tags, _predict and _predict_proba
"""

__all__ = ["BaseDeepAnomalyDetector"]

from abc import ABC, abstractmethod

from aeon.anomaly_detection.base import BaseAnomalyDetector


class BaseDeepAnomalyDetector(BaseAnomalyDetector, ABC):
    """Abstract base class for deep learning time series anomaly detection.

    The base anomaly detector provides a deep learning default method for
    _predict and _predict_proba, and provides a new abstract method for building a
    model.

    Parameters
    ----------
    batch_size : int, default = 40
        training batch size for the model
    last_file_name : str, default = "last_model"
        The name of the file of the last model, used
        only if save_last_model_to_file is used

    Arguments
    ---------
    self.model = None

    """

    _tags = {
        "capability:multivariate": True,
        "algorithm_type": "deeplearning",
        "non_deterministic": True,
        "cant_pickle": True,
        "python_dependencies": "tensorflow",
    }

    def __init__(
        self,
        batch_size=40,
        last_file_name="last_model",
    ):
        self.batch_size = batch_size
        self.last_file_name = last_file_name
        self.model_ = None

        super().__init__(axis=0)

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

    def load_model(self, model_path, classes):
        """Load a pre-trained keras model instead of fitting.

        When calling this function, all functionalities can be used
        such as predict, predict_proba etc. with the loaded model.

        Parameters
        ----------
        model_path : str (path including model name and extension)
            The directory where the model will be loaded from including the model
            name with a ".keras" extension.
            Example: model_path="path/to/file/best_model.keras"

        Returns
        -------
        None
        """
        import tensorflow as tf

        self.model_ = tf.keras.models.load_model(model_path)
        self.is_fitted = True

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
