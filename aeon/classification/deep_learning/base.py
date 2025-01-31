"""
Abstract base class for the Keras neural network classifiers.

    class name: BaseDeepClassifier

Defining methods:
    fitting         - fit(self, X, y)
    predicting      - predict(self, X)
                    - predict_proba(self, X)
    model building - build_model(self, input_shape, n_classes) (abstract method)

Inherited inspection methods:
    hyper-parameter inspection  - get_params()
    fitted parameter inspection - get_fitted_params()

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - is_fitted (property)
    fitted state inspection - check_is_fitted()

The reason for this class between BaseClassifier and deep_learning classifiers is
because we can generalise tags, _predict and _predict_proba
"""

__maintainer__ = ["hadifawaz1999"]
__all__ = ["BaseDeepClassifier"]

from abc import abstractmethod

import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import check_random_state

from aeon.classification.base import BaseClassifier


class BaseDeepClassifier(BaseClassifier):
    """Abstract base class for deep learning time series classifiers.

    The base classifier provides a deep learning default method for
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
        "X_inner_type": "numpy3D",
        "capability:multivariate": True,
        "algorithm_type": "deeplearning",
        "non_deterministic": True,
        "cant_pickle": True,
        "python_dependencies": "tensorflow",
    }

    @abstractmethod
    def __init__(
        self,
        batch_size=40,
        random_state=None,
        last_file_name="last_model",
    ):
        self.batch_size = batch_size
        self.random_state = random_state
        self.last_file_name = last_file_name

        self.model_ = None

        super().__init__()

    @abstractmethod
    def build_model(self, input_shape, n_classes):
        """Construct a compiled, un-trained, keras model that is ready for training.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer
        n_classes : int
            The number of classes, which shall become the size of the output
            layer

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

    def _predict(self, X):
        probs = self._predict_proba(X)
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in probs
            ]
        )

    def _predict_proba(self, X):
        """Find probability estimates for each class for all cases in X.

        Parameters
        ----------
        X : an np.ndarray of shape = (n_cases, n_channels, n_timepoints)
            The training input samples. input_checks : boolean
            Whether to check the X parameter

        Returns
        -------
        output : array of shape = [n_cases, n_classes] of probabilities
        """
        # Transpose to work correctly with keras
        X = X.transpose((0, 2, 1))
        probs = self.model_.predict(X, self.batch_size)

        # check if binary classification
        if probs.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            probs = np.hstack([1 - probs, probs])
        probs = probs / probs.sum(axis=1, keepdims=1)
        return probs

    def convert_y_to_keras(self, y):
        """Convert y to required Keras format."""
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        self.n_classes_ = len(self.classes_)
        y = y.reshape(len(y), 1)
        # Adjustment to allow deprecated attribute "sparse for older versions
        import sklearn
        from packaging import version

        # Get the installed version of scikit-learn
        installed_version = sklearn.__version__
        # Compare the installed version with the target version
        # categories='auto' to get rid of FutureWarning
        if version.parse(installed_version) < version.parse("1.2"):
            self.onehot_encoder = OneHotEncoder(sparse=False)
        else:
            self.onehot_encoder = OneHotEncoder(sparse_output=False)
        y = self.onehot_encoder.fit_transform(y)
        return y

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
            The directory where the model will be saved including the model
            name with a ".keras" extension.
            Example: model_path="path/to/file/best_model.keras"
        classes : np.ndarray
            The set of unique classes the pre-trained loaded model is trained
            to predict during the classification task.

        Returns
        -------
        None
        """
        import tensorflow as tf

        self.model_ = tf.keras.models.load_model(model_path)
        self.is_fitted = True

        self.classes_ = classes
        self.n_classes_ = len(self.classes_)

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
